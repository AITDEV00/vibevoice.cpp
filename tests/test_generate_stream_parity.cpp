// Streaming-generate plumbing test (realtime-0.5b).
//
// Because vibevoice_tts_generate is now a thin wrapper around
// vibevoice_tts_generate_streaming, a naive "streaming == batch" assertion is
// tautological — the decode bit-exactness is already proven by Task 2's
// test_decoder_chunked_parity gate. This test instead exercises the STREAMING
// PLUMBING itself:
//   - the callback fires >= 2 times for a multi-word sentence (multiple windows)
//   - concatenated callback audio length == batch length, and first/last
//     samples match (no dropped/duplicated windows, correct accumulation)
//   - an on_chunk returning false aborts early (fewer samples) and cleanly
//
// Gated: needs a realtime-0.5b gguf (VIBEVOICE_TTS_MODEL), a tokenizer
// (VIBEVOICE_TOKENIZER) and a voice gguf (VIBEVOICE_VOICE). Skips (return 77)
// when any is unset. Gated by VIBEVOICE_TEST_LARGE in CMake.

#include "tokenizer.hpp"
#include "vibevoice_tts.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
bool file_ok(const char* p) {
    if (!p || !*p) return false;
    FILE* f = std::fopen(p, "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}
}  // namespace

int main() {
    const char* model_env = std::getenv("VIBEVOICE_TTS_MODEL");
    const char* tok_env   = std::getenv("VIBEVOICE_TOKENIZER");
    const char* voice_env = std::getenv("VIBEVOICE_VOICE");
    if (!file_ok(model_env) || !file_ok(tok_env) || !file_ok(voice_env)) {
        std::fprintf(stderr,
                     "skip: set VIBEVOICE_TTS_MODEL, VIBEVOICE_TOKENIZER and "
                     "VIBEVOICE_VOICE to valid gguf paths.\n");
        return 77;
    }

    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_env, &model)) {
        std::fprintf(stderr, "FAIL: load model\n");
        return 1;
    }
    if (model.variant != "realtime-0.5b") {
        std::fprintf(stderr, "FAIL: want realtime-0.5b got %s\n",
                     model.variant.c_str());
        return 2;
    }
    if (!model.tokenizer.load_from_file(tok_env)) {
        std::fprintf(stderr, "FAIL: load tokenizer\n");
        return 3;
    }
    vv::VibeVoiceVoice voice;
    if (!vv::vibevoice_voice_load(voice_env, model, &voice)) {
        std::fprintf(stderr, "FAIL: load voice\n");
        return 4;
    }

    // A multi-word sentence — long enough to generate more than one 6-frame
    // speech window so we can assert the callback fires multiple times.
    const std::string text =
        "Hello world, this is a test of the streaming synthesis system.";

    vv::VibeVoiceTTSParams p;
    p.voice             = &voice;
    p.max_speech_frames = 60;
    p.n_diffusion_steps = 10;
    p.seed              = 12345;   // fixed so batch and streaming are identical
    p.verbose           = false;

    // ---- 1. batch reference ----
    std::vector<float> ref;
    int rc = vv::vibevoice_tts_generate(&model, text, p, &ref);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: batch generate rc=%d\n", rc);
        return 5;
    }
    if (ref.empty()) {
        std::fprintf(stderr, "FAIL: batch produced empty audio\n");
        return 6;
    }

    // ---- 2. streaming, accumulate every chunk ----
    std::vector<float> got;
    got.reserve(ref.size());
    int chunk_count = 0;
    rc = vv::vibevoice_tts_generate_streaming(
        &model, text, p,
        [&](const float* s, int n) {
            ++chunk_count;
            got.insert(got.end(), s, s + n);
            return true;
        });
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: streaming generate rc=%d\n", rc);
        return 7;
    }

    std::printf("stream plumbing: %d chunks, batch=%zu streaming=%zu samples\n",
                chunk_count, ref.size(), got.size());

    // (a) callback fired for multiple windows.
    if (chunk_count < 2) {
        std::fprintf(stderr, "FAIL: expected >=2 chunks, got %d\n", chunk_count);
        return 8;
    }
    // (b) accumulation is complete and correctly ordered.
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "FAIL: length mismatch streaming=%zu batch=%zu\n",
                     got.size(), ref.size());
        return 9;
    }
    if (got.front() != ref.front() || got.back() != ref.back()) {
        std::fprintf(stderr,
                     "FAIL: boundary sample mismatch: front %.6f vs %.6f, "
                     "back %.6f vs %.6f\n",
                     got.front(), ref.front(), got.back(), ref.back());
        return 10;
    }

    // ---- 3. abort after the first chunk ----
    std::vector<float> aborted;
    int abort_calls = 0;
    rc = vv::vibevoice_tts_generate_streaming(
        &model, text, p,
        [&](const float* s, int n) {
            ++abort_calls;
            if (abort_calls >= 2) return false;  // stop after the 1st chunk
            aborted.insert(aborted.end(), s, s + n);
            return true;
        });
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: aborted streaming rc=%d (want clean 0)\n", rc);
        return 11;
    }
    if (aborted.empty()) {
        std::fprintf(stderr, "FAIL: abort kept no audio\n");
        return 12;
    }
    if (aborted.size() >= ref.size()) {
        std::fprintf(stderr,
                     "FAIL: abort did not stop early (aborted=%zu full=%zu)\n",
                     aborted.size(), ref.size());
        return 13;
    }
    std::printf("abort: stopped after %d call(s), kept %zu of %zu samples\n",
                abort_calls, aborted.size(), ref.size());

    return 0;
}
