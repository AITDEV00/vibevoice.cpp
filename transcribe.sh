#!/usr/bin/env bash
# =============================================================================
# VibeVoice ASR — production transcription request (LiteLLM gateway)
#
# VERIFIED WORKING 2026-09-02 against the AL_AIN production deployment.
# Usage:
#   ./transcribe.sh <audio-file> [max_tokens]
# Env:
#   LITELLM_KEY   gateway API key (default: the test key used in validation)
#   LITELLM_URL   gateway endpoint (default: production gateway)
# Output:
#   Raw chat.completion JSON on stdout; transcript segments are inside
#   choices[0].message.content as a JSON string array
#   [{Start, End, Speaker, Content}, ...]
# =============================================================================
set -euo pipefail

FILE="${1:?usage: transcribe.sh <audio-file> [max_tokens]}"
MAX_TOKENS="${2:-}"

LITELLM_URL="${LITELLM_URL:-https://litellm.adeoaiengine.ecouncil.ae/v1/chat/completions}"
LITELLM_KEY="${LITELLM_KEY:-sk-KdqW6uIv_J2kK3pDbVI3CA}"

# max_tokens rule of thumb: ~35 tokens per audio-second + 500 headroom
if [[ -z "${MAX_TOKENS}" ]]; then
  DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$FILE")
  MAX_TOKENS=$(python3 -c "print(min(24000, int(${DUR:-60}) * 35 + 500))")
fi

B64=$(python3 -c "import base64,sys;print(base64.b64encode(open(sys.argv[1],'rb').read()).decode())" "$FILE")

# Build the JSON payload in a temp file: large base64 payloads exceed the OS
# argv limit (~2 MB), so the request body must NOT go through a command-line
# argument. Passing -d @file keeps curl inside its own limits.
PAYLOAD=$(mktemp /tmp/vibevoice-payload.XXXXXX.json)
trap 'rm -f "$PAYLOAD"' EXIT
python3 - "$FILE" "$MAX_TOKENS" "${DUR:-60}" "$PAYLOAD" <<'PY'
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
