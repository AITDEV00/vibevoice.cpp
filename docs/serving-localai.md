# Serving vibevoice.cpp via LocalAI — build + verify + install guide

This is the step-by-step record of setting up the **purego serving path** on
macOS (Apple Silicon) from this repo, exactly as performed on 2026-09-01. It
mirrors what LocalAI's `backend/go/vibevoice-cpp` expects: a shared library
exporting the flat `vv_capi_*` ABI from `include/vibevoice_capi.h`, loaded at
runtime via `dlopen`.

Worked example outputs are shown inline (Apple M1 Max, q4_k ASR model).

---

## TL;DR — run the built-in OpenAI-compatible server

A stdlib-only Python server lives at `examples/server/server.py` (enabled by
`VIBEVOICE_BUILD_SERVER=ON`, which now wires `examples/server/`):

```bash
# one-time: build the shared library (§2)
VIBEVOICE_BACKEND=metal python3 examples/server/server.py \
    --asr-model models/vibevoice-asr-q4_k.gguf \
    --tokenizer models/tokenizer.gguf \
    --port 8080

# OpenAI-compatible transcription (mp3 accepted; ffmpeg converts internally)
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@audio/2p_argument.mp3
# -> {"text": "I can't believe you did it again. ..."}

# diarized segments (vibevoice native JSON) instead of flat text
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@clip.wav -F segments=true

# TTS (start the server with --tts-model and optionally --voice)
curl http://127.0.0.1:8080/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"input":"Hello from the server.","voice":"models/voice-en-Carter_man.gguf"}' \
    --output out.wav

# introspection
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/health        # {"status":"ok","backend":"metal"}
```

The server binds the same `vv_capi_*` symbols the purego backend uses,
serializes requests behind a lock (single process-global engine), and honors
`VIBEVOICE_BACKEND` for device selection.

---

## 0. Prerequisites

| Tool | Purpose | Check |
|------|---------|-------|
| Xcode command line tools | AppleClang toolchain | `xcode-select -p` |
| CMake ≥ 3.16 | build system | `cmake --version` |
| ffmpeg | mp3→WAV conversion, `atempo` speed changes | `ffmpeg -version` |
| Python 3 | verification script | `python3 --version` |
| `hf` (huggingface-cli) | model download | `hf --version` |
| ~10 GB disk | q4_k ASR model + tokenizer | — |

One-time clone (the ggml submodule is mandatory — a bare clone leaves
`third_party/ggml` empty and CMake fails with *"does not contain a
CMakeLists.txt"*):

```bash
git clone --recursive https://github.com/localai-org/vibevoice.cpp
cd vibevoice.cpp

# if you already cloned without --recursive:
git submodule update --init --recursive
```

## 1. Get the models

`hf download <repo> <filename> --local-dir <dir>` — filename is a **separate
argument** (not part of the repo id):

```bash
mkdir -p models
hf download mudler/vibevoice.cpp-models tokenizer.gguf --local-dir models
hf download mudler/vibevoice.cpp-models vibevoice-asr-q4_k.gguf --local-dir models
# TTS (optional):
hf download mudler/vibevoice.cpp-models vibevoice-realtime-0.5B-q8_0.gguf --local-dir models
hf download mudler/vibevoice.cpp-models voice-en-Carter_man.gguf --local-dir models
# or everything at once (~15 GB):
# hf download mudler/vibevoice.cpp-models --local-dir models
```

## 2. Build the shared library (Metal)

```bash
cmake -B build-shared \
    -DVIBEVOICE_SHARED=ON \
    -DVIBEVOICE_BUILD_TESTS=OFF \
    -DVIBEVOICE_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-shared -j
```

Notes:
- A **separate build dir** (`build-shared/`) is used so the CLI build
  (`build/`) stays intact; both can coexist.
- `VIBEVOICE_SHARED=ON` produces `build-shared/libvibevoice.dylib`
  (macOS) / `libvibevoice.so` (Linux).
- Metal: ggml's CMake auto-enables `GGML_METAL=ON` on Apple Silicon, and the
  top-level `CMakeLists.txt` links `ggml-metal` + defines
  `VIBEVOICE_HAVE_METAL` whenever that target exists — so **no extra flag is
  needed**. To force it explicitly: `-DVIBEVOICE_GGML_METAL=ON`.
- Runtime backend is chosen by `VIBEVOICE_BACKEND` env var (`metal`, `cuda`,
  `vulkan`, `cpu`); unset = best-available (Metal on Apple Silicon).

## 3. Verify the exported symbols

`nm -gU` lists exported (undefined-suppressed, global) symbols:

```bash
nm -gU build-shared/libvibevoice.dylib | grep vv_capi
```

Expected (leading `_` is standard Mach-O mangling; `dlsym`/purego handle it):

```
000000000000547c T _vv_capi_asr
0000000000004150 T _vv_capi_load
0000000000004a54 T _vv_capi_tts
000000000000506c T _vv_capi_tts_stream
000000000000576c T _vv_capi_unload
000000000000583c T _vv_capi_version
0000000000005848 T _vv_capi_voice_clone
```

Confirm the Metal backend is linked:

```bash
otool -L build-shared/libvibevoice.dylib | grep ggml
# libggml-metal.0.dylib must appear
```

On Linux the equivalent checks are `nm -D --defined-only
libvibevoice.so | grep vv_capi` and `ldd libvibevoice.so`.

## 4. Smoke-test the ABI the purego way

`scripts/test_capi_dlopen.py` replicates LocalAI's binding flow —
`purego.Dlopen` → `purego.RegisterLibFunc` — using Python ctypes:

```bash
VIBEVOICE_BACKEND=metal python3 scripts/test_capi_dlopen.py \
    --asr-model models/vibevoice-asr-q4_k.gguf \
    --tokenizer models/tokenizer.gguf \
    --audio audio/2p_argument.wav
```

Expected output (abbreviated):

```
[dlopen] build-shared/libvibevoice.dylib: OK
[bind] vv_capi_load, vv_capi_tts, vv_capi_asr, vv_capi_unload, vv_capi_version: OK
[call] vv_capi_version -> vibevoice.cpp 0.1.0 (capi)
[call] vv_capi_load -> 0 (engine up)
[call] vv_capi_asr -> 1461 bytes, 6 segments
  [  0.00-  8.12] spk0: I can't believe you did it again. ...
...
[call] vv_capi_unload -> OK
All vv_capi_* symbols load, bind, and execute — purego backend contract satisfied.
```

The script exercises the full contract LocalAI relies on:
- **Symbol binding by name** — same signatures as `vibevoice_capi.h`
- **`vv_capi_load(tts, asr, tokenizer, voice, n_threads)`** — `NULL`s allowed
  for absent models, `n_threads=0` = auto
- **`vv_capi_asr` grows-on-demand protocol** — negative rc = `-required_size`,
  re-call once with a larger buffer (LocalAI's `callASR` does the same)

If this passes, the library is drop-in ready for `backend/go/vibevoice-cpp`.

## 5. (Optional) CLI round-trip — the batch equivalent

```bash
cmake -B build -DVIBEVOICE_BUILD_TESTS=ON && cmake --build build -j
ffmpeg -y -i audio/2p_argument.mp3 -ac 1 -ar 24000 audio/2p_argument.wav
VIBEVOICE_BACKEND=metal ./build/bin/vibevoice-cli asr \
    --model models/vibevoice-asr-q4_k.gguf \
    --tokenizer models/tokenizer.gguf \
    --audio audio/2p_argument.wav --max-new-tokens 8192
```

Timing reference (M1 Max, q4_k ASR):

| Clip | Audio | load | inference | RTF |
|------|-------|------|-----------|-----|
| 1.0× | 68.5 s | 14.4 s (first run, shader compile) | 34.8 s | 0.507 |
| 1.5× | 45.7 s | 2.6 s | 25.4 s | 0.557 |
| 2.0× | 34.3 s | 2.8 s | 22.3 s | 0.652 |

Speed-up clips: `ffmpeg -filter:a "atempo=1.5"` keeps pitch, halves nothing
else — see §7.

## 6. Wire into LocalAI

The Go backend lives at
[`mudler/LocalAI/backend/go/vibevoice-cpp`](https://github.com/mudler/LocalAI/tree/master/backend/go/vibevoice-cpp).
It binds the same symbols (`vv_capi_load`, `vv_capi_tts`, `vv_capi_tts_stream`,
`vv_capi_asr`, `vv_capi_unload`, `vv_capi_version`) via purego and serves them
over gRPC behind LocalAI's OpenAI-compatible endpoints.

Build the backend against this library:

```bash
git clone https://github.com/mudler/LocalAI
cd LocalAI/backend/go/vibevoice-cpp

# point purego at the dylib built in §2 (default is ./libgovibevoicecpp-fallback.dylib)
export VIBEVOICECPP_LIBRARY=/abs/path/to/vibevoice.cpp/build-shared/libvibevoice.dylib

go build -o vibevoice-cpp-backend .
```

Then register a model in LocalAI (model YAML), roughly:

```yaml
name: vibevoice-asr
backend: vibevoice-cpp
parameters:
  model: /abs/path/to/models/vibevoice-asr-q4_k.gguf
options:
  tokenizer: /abs/path/to/models/tokenizer.gguf
```

…or via the gallery / `local-ai run` flow. The backend requires
`options: [tokenizer=<path>]` and resolves `ModelFile` to either a TTS
(realtime-0.5B) or ASR gguf by name. ffmpeg is only needed at request time for
audio that isn't already 24 kHz mono s16le WAV.

## 7. Appendix — audio prep cheatsheet

```bash
# mp3 → 24 kHz mono s16le WAV (the format the model expects)
ffmpeg -y -i input.mp3 -ac 1 -ar 24000 output.wav

# same, sped up 1.5× without pitch change (atempo range 0.5–100; chain for >2)
ffmpeg -y -i input.mp3 -filter:a "atempo=1.5" -ac 1 -ar 24000 output_1_5x.wav
ffmpeg -y -i input.mp3 -filter:a "atempo=2.0" -ac 1 -ar 24000 output_2x.wav

# chipmunk variant (rate shift, pitch rises)
ffmpeg -y -i input.mp3 -filter:a "asetrate=24000*1.5,aresample=24000" -ac 1 output_fast.wav
```

Known accuracy behavior of the ASR on sped-up speech (this clip):
- 1.0× / 1.5× — verbatim transcript
- 2.0× — mostly good; can drop a closing phrase and garble fast final words

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| CMake: `third_party/ggml … does not contain a CMakeLists.txt` | `git submodule update --init --recursive` |
| `hf download` `Invalid value. Repo id must be…` | filename must be a separate arg: `hf download <repo> <file> --local-dir models` |
| First ASR run takes ~10 s extra | Metal shader pipeline compile; cached per machine, later runs ~2.5 s load |
| `backend: VIBEVOICE_BACKEND=… requested but no matching device` | backend not compiled in — rebuild with `-DVIBEVOICE_GGML_METAL=ON` (or use the default best-available) |
| `vv_capi_asr` returns negative rc | grows-on-demand: re-call with buffer of `-rc` bytes (script does this automatically) |
| TTS from CLI is non-deterministic | pass `--seed N` (server: `"seed"` in JSON body) |
| dylib not found by Go backend | `VIBEVOICECPP_LIBRARY` must be an absolute path to `libvibevoice.dylib` |
| server: `can't concat str to bytes` on multipart | fixed — multipart boundary must be `.encode()`d before byte-splitting (older copies of server.py) |
