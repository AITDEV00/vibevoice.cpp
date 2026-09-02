#!/usr/bin/env python3
"""vibevoice-server — minimal OpenAI-compatible HTTP server over libvibevoice.

Serves the flat vv_capi_* ABI (include/vibevoice_capi.h) through the same
dlopen binding that LocalAI's purego backend uses, exposing:

    POST /v1/audio/transcriptions   (multipart form: file, model, ...)
        -> {"text": "..."}                    or segments=true: full JSON
    POST /v1/audio/speech           (json body: model, input, voice, ...)
        -> audio/wav (24 kHz mono, 16-bit)
    GET  /v1/models
    GET  /health

Dependencies: Python 3 stdlib only. Audio decoding for non-WAV uploads
(mp3 etc.) shells out to ffmpeg when available; 24 kHz mono s16le WAVs
are passed straight to the engine.

Launch:
    VIBEVOICE_BACKEND=metal python3 examples/server/server.py \
        --asr-model models/vibevoice-asr-q4_k.gguf \
        --tokenizer models/tokenizer.gguf \
        --voice models/voice-en-Carter_man.gguf \
        --port 8080

The engine is a single process-global (see vv_capi_load), so requests are
serialized behind a lock — same contract as LocalAI's base.SingleThread.
"""
import argparse
import ctypes
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SAMPLE_RATE = 24000

# --------------------------------------------------------------------------
# engine binding (mirrors scripts/test_capi_dlopen.py / LocalAI main.go)
# --------------------------------------------------------------------------

engine_lock = threading.Lock()
_lib = None


def load_engine(lib_path, tts_model, asr_model, tokenizer, voice, threads=0):
    global _lib
    lib = ctypes.CDLL(lib_path)

    lib.vv_capi_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                 ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.vv_capi_load.restype = ctypes.c_int

    lib.vv_capi_tts.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
                                ctypes.c_char_p, ctypes.c_int, ctypes.c_float,
                                ctypes.c_int, ctypes.c_uint32]
    lib.vv_capi_tts.restype = ctypes.c_int

    lib.vv_capi_tts_stream.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                       ctypes.c_int, ctypes.c_float,
                                       ctypes.c_int, ctypes.c_uint32,
                                       ctypes.c_void_p, ctypes.c_void_p]
    lib.vv_capi_tts_stream.restype = ctypes.c_int

    lib.vv_capi_asr.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                ctypes.c_size_t, ctypes.c_int]
    lib.vv_capi_asr.restype = ctypes.c_int

    lib.vv_capi_unload.argtypes = []
    lib.vv_capi_unload.restype = None

    lib.vv_capi_version.argtypes = []
    lib.vv_capi_version.restype = ctypes.c_char_p

    rc = lib.vv_capi_load(
        tts_model.encode() if tts_model else None,
        asr_model.encode() if asr_model else None,
        tokenizer.encode() if tokenizer else None,
        voice.encode() if voice else None,
        threads)
    if rc != 0:
        raise RuntimeError(f"vv_capi_load failed (rc={rc})")
    _lib = lib
    print(f"[engine] loaded (vibevoice {lib.vv_capi_version().decode()}), "
          f"tts={tts_model or '-'} asr={asr_model or '-'} "
          f"tokenizer={tokenizer} voice={voice or '-'}")


def capi_asr(wav_path, max_new_tokens):
    """vv_capi_asr with the grows-on-demand protocol."""
    cap = 256 * 1024
    while True:
        buf = ctypes.create_string_buffer(cap)
        rc = _lib.vv_capi_asr(wav_path.encode(), buf, cap, max_new_tokens)
        if rc < 0 and -rc > cap and cap < 64 * 1024 * 1024:
            cap = -rc + 64
            continue
        if rc < 0:
            raise RuntimeError(f"vv_capi_asr failed (rc={rc})")
        if rc == 0:
            return []
        return json.loads(buf.value.decode())


def capi_tts(text, voice, ref_audios, dst, steps, cfg, max_frames, seed):
    arr = (ctypes.c_char_p * len(ref_audios))(
        *[a.encode() for a in ref_audios]) if ref_audios else None
    rc = _lib.vv_capi_tts(text.encode(),
                          voice.encode() if voice else None,
                          arr, len(ref_audios),
                          dst.encode(), steps, cfg, max_frames, seed)
    if rc != 0:
        raise RuntimeError(f"vv_capi_tts failed (rc={rc})")


def to_wav_24k_mono(src_path):
    """Return a 24 kHz mono s16le WAV path; converts via ffmpeg if needed."""
    with open(src_path, "rb") as f:
        head = f.read(44)
    is_wav = head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    if is_wav:
        import wave
        try:
            with wave.open(src_path, "rb") as w:
                if (w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
                        and w.getsampwidth() == 2):
                    return src_path  # already the model's native format
        except (wave.Error, EOFError):
            pass
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            f"{src_path} is not 24 kHz mono s16le WAV and ffmpeg is "
            "unavailable to convert it")
    dst = os.path.join(tempfile.gettempdir(),
                       f"vv-{uuid.uuid4().hex}.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_path,
                    "-ac", "1", "-ar", str(SAMPLE_RATE), dst], check=True)
    return dst


# --------------------------------------------------------------------------
# multipart/form-data parsing (stdlib-only, minimal)
# --------------------------------------------------------------------------

def parse_multipart(body, boundary):
    """Returns dict field -> (filename|None, bytes)."""
    fields = {}
    delim = b"--" + boundary
    for part in body.split(delim):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        raw_head, payload = part.split(b"\r\n\r\n", 1)
        headers = {}
        for line in raw_head.decode("utf-8", "replace").split("\r\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        cd = headers.get("content-disposition", "")
        name_m = re.search(r'name="([^"]*)"', cd)
        file_m = re.search(r'filename="([^"]*)"', cd)
        if name_m:
            fields[name_m.group(1)] = (
                file_m.group(1) if file_m else None, payload)
    return fields


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "vibevoice-server/0.1"

    # ---- helpers -------------------------------------------------------
    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_wav(self, path, code=200):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(code)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, code, message):
        self.send_json({"error": {"message": message,
                                  "type": "invalid_request_error",
                                  "code": code}}, code)

    def log_message(self, fmt, *args):  # compact logs
        sys.stderr.write("[http] " + (fmt % args) + "\n")

    # ---- routes --------------------------------------------------------
    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            return self.send_json({"status": "ok",
                                   "backend": os.environ.get(
                                       "VIBEVOICE_BACKEND", "auto")})
        if self.path == "/v1/models":
            models = []
            if self.server.tts_model:
                models.append({"id": self.server.tts_model,
                               "object": "model",
                               "owned_by": "vibevoice.cpp"})
            if self.server.asr_model:
                models.append({"id": self.server.asr_model,
                               "object": "model",
                               "owned_by": "vibevoice.cpp"})
            return self.send_json({"object": "list", "data": models})
        return self.send_error_json(404, f"unknown route {self.path}")

    def do_POST(self):
        try:
            if self.path == "/v1/audio/transcriptions":
                return self.handle_transcriptions()
            if self.path == "/v1/audio/speech":
                return self.handle_speech()
            return self.send_error_json(404, f"unknown route {self.path}")
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001 — convert any failure to 500
            return self.send_error_json(500, str(e))

    # ---- POST /v1/audio/transcriptions ---------------------------------
    def handle_transcriptions(self):
        ct = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ct:
            return self.send_error_json(
                400, "expected multipart/form-data (curl -F file=@…)")
        # boundary is a str from the header; the parser splits raw bytes,
        # so it must be encoded to bytes (b"--" + str raises TypeError)
        boundary = ct.split("boundary=")[1].strip().strip('"').encode()
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        fields = parse_multipart(body, boundary)
        if "file" not in fields:
            return self.send_error_json(400, "missing 'file' field")

        filename, payload = fields["file"]
        want_segments = (fields.get("segments") or (None, b""))[1] in (
            b"true", b"1")
        max_new = 8192
        if fields.get("max_new_tokens"):
            try:
                max_new = int((fields.get("max_new_tokens")[1]).decode())
            except ValueError:
                pass

        # engine contract: 24 kHz mono s16le wav
        tmp_in = os.path.join(
            tempfile.gettempdir(), f"vv-up-{uuid.uuid4().hex}-{filename or 'a'}")
        wav = None
        with open(tmp_in, "wb") as f:
            f.write(payload)
        try:
            wav = to_wav_24k_mono(tmp_in)
            with engine_lock:
                segments = capi_asr(wav, max_new)
        finally:
            if wav != tmp_in and os.path.exists(tmp_in):
                os.unlink(tmp_in)

        if want_segments:
            return self.send_json(segments)
        text = " ".join(seg["Content"] for seg in segments)
        return self.send_json({"text": text})

    # ---- POST /v1/audio/speech -----------------------------------------
    def handle_speech(self):
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length) or b"{}")
        text = (req.get("input") or "").strip()
        if not text:
            return self.send_error_json(400, "'input' is required")

        voice = req.get("voice") or self.server.default_voice
        ref_audios = req.get("ref_audio") or []
        if isinstance(ref_audios, str):
            ref_audios = [p.strip() for p in ref_audios.split(",") if p.strip()]
        steps = int(req.get("steps") or 20)
        cfg = float(req.get("cfg") or 1.3)
        max_frames = int(req.get("max_frames") or 200)
        seed = int(req.get("seed") or 0)

        if not self.server.tts_model:
            return self.send_error_json(
                503, "no TTS model loaded (start server with --tts-model)")
        if not voice and not ref_audios:
            return self.send_error_json(
                400, "'voice' or 'ref_audio' required (none configured)")

        out = os.path.join(tempfile.gettempdir(),
                           f"vv-out-{uuid.uuid4().hex}.wav")
        try:
            with engine_lock:
                capi_tts(text, voice, ref_audios, out,
                         steps, cfg, max_frames, seed)
            return self.send_wav(out)
        finally:
            if os.path.exists(out):
                os.unlink(out)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="OpenAI-compatible server over libvibevoice (dlopen)")
    ap.add_argument("--lib", default="build-shared/libvibevoice.dylib")
    ap.add_argument("--tts-model", default=None)
    ap.add_argument("--asr-model", default=None)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--voice", default=None,
                    help="default voice gguf for /v1/audio/speech")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if not args.tts_model and not args.asr_model:
        ap.error("at least one of --tts-model / --asr-model is required")

    load_engine(args.lib, args.tts_model, args.asr_model,
                args.tokenizer, args.voice, args.threads)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.tts_model = args.tts_model
    httpd.asr_model = args.asr_model
    httpd.default_voice = args.voice
    print(f"[server] listening on http://{args.host}:{args.port}")
    print(f"[server] routes: POST /v1/audio/speech, "
          f"POST /v1/audio/transcriptions, GET /v1/models, GET /health")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] shutting down")
        _lib.vv_capi_unload()


if __name__ == "__main__":
    sys.exit(main())
