# VibeVoice ASR: Curl Guide

> **Scope**: How to call Microsoft VibeVoice automatic speech recognition (ASR) through the LiteLLM gateway, with sample curl commands and expected response shapes
>
> **Related docs**: [HAMSA_STT_TTS_GUIDE.md](HAMSA_STT_TTS_GUIDE.md) (native `/v1/audio/transcriptions` STT provider), [INCEPTION_TTS_STT_GUIDE.md](INCEPTION_TTS_STT_GUIDE.md) (similar OpenAI-compatible audio provider)

---

## Gateway endpoint

| Capability                 | Gateway URL                                                      |
| -------------------------- | ---------------------------------------------------------------- |
| ASR (chat-style, audio-in) | `https://litellm.adeoaiengine.ecouncil.ae/v1/chat/completions` |

Authentication is the standard LiteLLM gateway key passed as a Bearer token. The VibeVoice pod itself does not require auth, but the gateway always requires it:

```
Authorization: Bearer <your-api-key>
```

The gateway maps the user-facing model name to the internal pod model. The discovery controller registers the model as `hosted_vllm/microsoft/VibeVoice-ASR`, so you send `microsoft/VibeVoice-ASR` as the model and the gateway resolves the provider, api_base, and routing.

> **Important**: VibeVoice-ASR is served as a **vLLM chat-completions** model, not through the native `/v1/audio/transcriptions` multipart endpoint. Audio is passed inline as a base64 `input_audio` content part inside a normal chat request. Calling `/v1/audio/transcriptions` returns `404 Not Found` from the upstream.

---

## Model overview

| Property         | Value                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------- |
| Model ID         | `microsoft/VibeVoice-ASR`                                                                                |
| Underlying model | `hosted_vllm/microsoft/VibeVoice-ASR` (upstream: `VibeVoice-ASR-awq-int4`, served via vLLM `0.27.1`) |
| Type             | Audio-to-text (ASR) with speaker diarization                                                               |
| Mode             | `audio_transcription`                                                                                    |
| Input            | Base64 WAV audio as an`input_audio` content part                                                         |
| Output           | JSON array of segments`{Start, End, Speaker, Content}` inside `message.content`                        |

Confirm it is registered on the gateway:

```bash
curl -sk "https://litellm.adeoaiengine.ecouncil.ae/v1/models" \
  -H "Authorization: Bearer <your-api-key>"
```

---

## 1. Transcribe audio (single request)

### Endpoint

```
POST /v1/chat/completions
```

### Request shape

A standard OpenAI chat-completions request. The system prompt tells the model to emit JSON; the user message carries a text instruction plus the audio itself:

| Field                                    | Required | Notes                                                                                                                                                                |
| ---------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`                                | yes      | User-facing name`microsoft/VibeVoice-ASR`                                                                                                                          |
| `messages[0]` (system)                 | yes      | Instructs the model to transcribe audio to JSON                                                                                                                      |
| `messages[1].content[0]` (text)        | yes      | Prompt that names the required keys                                                                                                                                  |
| `messages[1].content[1]` (input_audio) | yes      | The audio:`{"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}`                                                                                |
| `max_tokens`                           | no       | Leave generous headroom (e.g. 300+) so the JSON array is not truncated                                                                                               |
| `temperature`                          | no       | **`0.0` (recommended)** — greedy decoding, the API equivalent of HF `do_sample=False`. Deterministic: identical audio always yields byte-identical output |
| `repetition_penalty`                   | no       | **`1.05` (recommended)** — mild penalty that suppresses the model's known post-audio repetition loop; passes through the gateway to vLLM                    |
| `do_sample`                            | no       | HF-style`do_sample: false` is accepted (and ignored) by vLLM; use `temperature: 0.0` instead — that is the real greedy switch                                   |

The `input_audio.data` field is the base64-encoded WAV payload. Use the standard OpenAI content-type pattern for audio input.

### Sample curl

```bash
curl -sk https://litellm.adeoaiengine.ecouncil.ae/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/VibeVoice-ASR",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant that transcribes audio input into text output in JSON format."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "This is a 1.8 seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content"
          },
          {
            "type": "input_audio",
            "input_audio": {
              "data": "<base64-wav>",
              "format": "wav"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }'
```

### Building the base64 payload

Encode your audio file before sending:

```bash
B64=$(python3 -c "import base64;print(base64.b64encode(open('audio.wav','rb').read()).decode())")
```

### Response shape

The response is an OpenAI `chat.completion`. The transcription is a JSON array of segments embedded as a string inside `message.content`, one object per detected speaker/utterance:

| Segment key | Type   | Meaning                          |
| ----------- | ------ | -------------------------------- |
| `Start`   | float  | Start time in seconds            |
| `End`     | float  | End time in seconds              |
| `Speaker` | int    | Diarized speaker index (0-based) |
| `Content` | string | Transcribed text                 |

Example response:

```json
{
  "id": "chatcmpl-a883420fc9898ded",
  "created": 1788353170,
  "model": "microsoft/VibeVoice-ASR",
  "object": "chat.completion",
  "system_fingerprint": "vllm-0.27.1-f8eb07bf",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "[{\"Start\":0,\"End\":1.8,\"Speaker\":0,\"Content\":\"[Music]\"}]\n",
        "role": "assistant",
        "provider_specific_fields": {
          "refusal": null,
          "reasoning": null
        }
      }
    }
  ],
  "usage": {
    "completion_tokens": 26,
    "prompt_tokens": 76,
    "total_tokens": 102
  }
}
```

The example above is from a live test: a 1.8-second WAV returned a single segment `{"Start":0,"End":1.8,"Speaker":0,"Content":"[Music]"}`. The `system_fingerprint` shows the upstream vLLM version (`vllm-0.27.1-f8eb07bf`).

---

## Response formatting tips

- **Give `max_tokens` headroom.** If `max_tokens` is too small, `finish_reason` becomes `"length"` and the JSON array is cut off mid-string. A few hundred tokens is safe for a short clip.
- **Parse the JSON string.** `message.content` is a JSON-encoded string; decode it (e.g. `json.loads`) before consuming the segments. It may be prefixed by a leading `assistant\n` in some responses.
- **Empty/non-speech audio** still yields a valid segment (e.g. `Content: "[Music]"`), so always handle the array.

---

## Recommended sampling parameters

| Parameter              | Value                      | Why                                                                                                                                                     |
| ---------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `temperature`        | `0.0`                    | Greedy decoding (`do_sample=False` equivalent). **Verified deterministic**: 3 repeat calls of the same audio produce byte-identical transcripts |
| `repetition_penalty` | `1.05`                   | Suppresses the post-audio repetition loop the model exhibits when it fails to emit EOS after the last segment                                           |
| `max_tokens`         | ~35 × audio-seconds + 500 | Rule of thumb; short 46 s clip needs ~1500, a 60 min file needs ~24000                                                                                  |

```json
"temperature": 0.0,
"repetition_penalty": 1.05,
"max_tokens": 1500
```

---

## Measured performance (2026-09-02, production deployment)

Single request through the gateway (client-side wall time incl. TLS + gateway):

| Audio                               |  Processing |                               RTF |
| ----------------------------------- | ----------: | --------------------------------: |
| 45.7 s clip                         | 0.34–4.5 s |                       ~0.01–0.10 |
| 19.3 min (`low-res-May_1_5x.wav`) |      74.1 s | **0.064** (~16× real-time) |

Concurrency (burst of parallel requests through the gateway):

| Burst                          | Result                                  |      Aggregate throughput |
| ------------------------------ | --------------------------------------- | ------------------------: |
| 4 simultaneous                 | 4/4 OK, all`finish=stop`              |            35× real-time |
| 8 simultaneous                 | 8/8 OK, per-request 0.9–1.6 s          | **200× real-time** |
| **16 × 19.3-min files** | **16/16 OK, all `finish=stop`** | **697× real-time** |
| 8 + 8 mixed (May + Shamma)     | 16/16 OK, 127 s wall                    |           138× real-time |

Notes:

- All requests completed with `finish_reason: "stop"` — no truncation, no repetition loops under concurrency.
- The gateway + upstream batch well: 8 parallel requests complete in ~1.6 s wall.
- Every burst leaves the pod healthy (no crashes, no restarts).

---

## Determinism notes

- **Sequential requests are byte-identical** for the same audio (`temperature=0`).
- **Concurrent requests may vary slightly** in wording at Arabic/English
  code-switch points and short backchannel utterances (~0.07% of words measured,
  99.93% word-level similarity across 8 concurrent runs of the same file).
  Cause: fp16 batch-dependent kernel numerics — different batch shapes change
  accumulation order, flipping near-tie tokens. Timestamps are ≥99% identical
  (±0.1 s on affected segments). Not a bug; send a file solo if you need
  strict reproducibility.

---

## What the gateway transforms

The gateway maps the user-facing model name (`microsoft/VibeVoice-ASR`) to the internal pod deployment via the database config lookup, resolves the upstream `api_base` and the `hosted_vllm` provider, and forwards the OpenAI-compatible chat request. LiteLLM-internal params are stripped before the request reaches the upstream pod. No audio or transcription content is altered in transit.

---

## Quick reference

| Capability               | Method   | Endpoint                 | Model                       |
| ------------------------ | -------- | ------------------------ | --------------------------- |
| ASR (audio in chat JSON) | `POST` | `/v1/chat/completions` | `microsoft/VibeVoice-ASR` |

### Production-ready request (copy-paste script)

Verified working 2026-09-02. Auto-sizes `max_tokens` from the audio duration
(~35 tokens/audio-second + 500, capped at 24000) and writes the payload to a
temp file — large base64 payloads exceed the shell argv limit, so the body
must be passed via `-d @file`, not as an inline argument.

```bash
#!/usr/bin/env bash
# transcribe <audio-file> [max_tokens]
set -euo pipefail
FILE="${1:?usage: transcribe <audio-file> [max_tokens]}"
MAX_TOKENS="${2:-}"
LITELLM_URL="${LITELLM_URL:-https://litellm.adeoaiengine.ecouncil.ae/v1/chat/completions}"
LITELLM_KEY="${LITELLM_KEY:-<your-api-key>}"

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$FILE")
if [[ -z "${MAX_TOKENS}" ]]; then
  MAX_TOKENS=$(python3 -c "print(min(24000, int(${DUR:-60}) * 35 + 500))")
fi

PAYLOAD=$(mktemp /tmp/vibevoice-payload.XXXXXX.json)
trap 'rm -f "$PAYLOAD"' EXIT
python3 - "$FILE" "$MAX_TOKENS" "$DUR" "$PAYLOAD" <<'PY'
import base64, json, sys
path, max_tokens, dur, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
b64 = base64.b64encode(open(path, "rb").read()).decode()
payload = {
    "model": "microsoft/VibeVoice-ASR",
    "messages": [
        {"role": "system",
         "content": "You are a helpful assistant that transcribes audio input "
                    "into text output in JSON format."},
        {"role": "user",
         "content": [
             {"type": "text",
              "text": f"This is a {float(dur):.1f} seconds audio, please "
                      f"transcribe it with these keys: Start time, End time, "
                      f"Speaker ID, Content"},
             {"type": "input_audio",
              "input_audio": {"data": b64, "format": "wav"}},
         ]},
    ],
    "max_tokens": int(max_tokens),
    "temperature": 0.0,
    "repetition_penalty": 1.05,
}
with open(out, "w") as f:
    json.dump(payload, f)
PY

curl -sk "$LITELLM_URL" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "Content-Type: application/json" \
  -d @"$PAYLOAD"
```

Or inline (short clips only — argv limit):

```bash
B64=$(python3 -c "import base64;print(base64.b64encode(open('audio.wav','rb').read()).decode())")
curl -sk https://litellm.adeoaiengine.ecouncil.ae/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"microsoft/VibeVoice-ASR\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant that transcribes audio input into text output in JSON format.\"},
      {\"role\": \"user\", \"content\": [
        {\"type\": \"text\", \"text\": \"This is a 60 seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content\"},
        {\"type\": \"input_audio\", \"input_audio\": {\"data\": \"$B64\", \"format\": \"wav\"}}
      ]}
    ],
    \"max_tokens\": 2600,
    \"temperature\": 0.0,
    \"repetition_penalty\": 1.05
  }"
```
