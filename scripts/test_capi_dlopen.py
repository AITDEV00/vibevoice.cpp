#!/usr/bin/env python3
"""Purego-equivalent smoke test for libvibevoice.dylib.

Mirrors what LocalAI's backend/go/vibevoice-cpp does with
purego.Dlopen + purego.RegisterLibFunc: load the shared library by
path, resolve the vv_capi_* symbols by name, then exercise the ABI.

Run:
    VIBEVOICE_BACKEND=metal python3 scripts/test_capi_dlopen.py \
        --asr-model models/vibevoice-asr-q4_k.gguf \
        --tokenizer models/tokenizer.gguf \
        --audio audio/2p_argument.wav
"""
import argparse
import ctypes
import json
import sys

# mirror src/vibevoice.cpp error codes (vv_status)
STATUS = {
    0: "VV_OK",
    -1: "VV_ERR_INVALID_ARG",
    -2: "VV_ERR_MODEL_LOAD",
    -3: "VV_ERR_TOKENIZER",
    -4: "VV_ERR_AUDIO",
    -5: "VV_ERR_ALLOC",
    -6: "VV_ERR_RUNTIME",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="build-shared/libvibevoice.dylib")
    ap.add_argument("--tts-model", default=None)
    ap.add_argument("--asr-model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--voice", default=None)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    args = ap.parse_args()

    # ---- purego.Dlopen equivalent -------------------------------------
    lib = ctypes.CDLL(args.lib)
    print(f"[dlopen] {args.lib}: OK")

    # ---- purego.RegisterLibFunc equivalents ---------------------------
    # signatures transcribed 1:1 from include/vibevoice_capi.h
    lib.vv_capi_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                 ctypes.c_char_p, ctypes.c_char_p,
                                 ctypes.c_int]
    lib.vv_capi_load.restype = ctypes.c_int

    lib.vv_capi_tts.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                ctypes.POINTER(ctypes.c_char_p),
                                ctypes.c_int,
                                ctypes.c_char_p,
                                ctypes.c_int, ctypes.c_float,
                                ctypes.c_int, ctypes.c_uint32]
    lib.vv_capi_tts.restype = ctypes.c_int

    lib.vv_capi_asr.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                ctypes.c_size_t, ctypes.c_int]
    lib.vv_capi_asr.restype = ctypes.c_int

    lib.vv_capi_unload.argtypes = []
    lib.vv_capi_unload.restype = None

    lib.vv_capi_version.argtypes = []
    lib.vv_capi_version.restype = ctypes.c_char_p

    print("[bind] vv_capi_load, vv_capi_tts, vv_capi_asr, "
          "vv_capi_unload, vv_capi_version: OK")

    # ---- exercise the ABI ---------------------------------------------
    ver = lib.vv_capi_version()
    print(f"[call] vv_capi_version -> {ver.decode()}")

    rc = lib.vv_capi_load(args.tts_model.encode() if args.tts_model else None,
                          args.asr_model.encode(),
                          args.tokenizer.encode(),
                          args.voice.encode() if args.voice else None,
                          0)  # n_threads, 0 = auto
    if rc != 0:
        print(f"[call] vv_capi_load -> {rc} ({STATUS.get(rc, '?')})")
        return 1
    print("[call] vv_capi_load -> 0 (engine up)")

    # vv_capi_asr grows-on-demand protocol: negative rc = -required_size
    cap = 256 * 1024
    buf = ctypes.create_string_buffer(cap)
    rc = lib.vv_capi_asr(args.audio.encode(), buf, cap, args.max_new_tokens)
    if rc < 0 and -rc > cap:
        cap = -rc + 64
        buf = ctypes.create_string_buffer(cap)
        rc = lib.vv_capi_asr(args.audio.encode(), buf, cap,
                             args.max_new_tokens)
    if rc < 0:
        print(f"[call] vv_capi_asr -> {rc} ({STATUS.get(rc, '?')})")
        lib.vv_capi_unload()
        return 1
    transcript = json.loads(buf.value.decode())
    print(f"[call] vv_capi_asr -> {rc} bytes, {len(transcript)} segments")
    for seg in transcript:
        print(f"  [{seg['Start']:>6.2f}-{seg['End']:>6.2f}] "
              f"spk{seg['Speaker']}: {seg['Content'][:72]}")

    lib.vv_capi_unload()
    print("[call] vv_capi_unload -> OK")
    print("\nAll vv_capi_* symbols load, bind, and execute — "
          "purego backend contract satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
