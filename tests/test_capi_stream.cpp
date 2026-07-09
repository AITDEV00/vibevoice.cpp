// Streaming C ABI parity test for vv_capi_tts_stream (include/vibevoice_capi.h).
//
// Proves the callback path is bit-for-bit identical to the file path: the
// int16 PCM delivered to vv_pcm_cb, concatenated, must equal the PCM samples
// vv_capi_tts writes to a WAV for the same text/voice/seed. Same generate
// path, same float→int16 conversion (clamp + std::lround(x*32767)).
//
// Skips (return 77) unless VIBEVOICE_TTS_MODEL, VIBEVOICE_TOKENIZER,
// VIBEVOICE_VOICE are all set. Gated by VIBEVOICE_TEST_LARGE in CMake.

#include "vibevoice_capi.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

bool file_ok(const char* p) {
    if (!p || !*p) return false;
    FILE* f = std::fopen(p, "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

// Read a 16-bit PCM WAV, stripping the canonical 44-byte header, into int16.
// vv_capi_tts writes exactly this layout (drwav RIFF/PCM16, mono, 24 kHz).
bool read_wav_pcm16(const char* path, std::vector<int16_t>* out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size <= 44) { std::fclose(f); return false; }
    // Skip the 44-byte header (RIFF + fmt + data chunk headers).
    if (std::fseek(f, 44, SEEK_SET) != 0) { std::fclose(f); return false; }
    const size_t n_bytes  = static_cast<size_t>(size - 44);
    const size_t n_sample = n_bytes / sizeof(int16_t);
    out->resize(n_sample);
    size_t got = std::fread(out->data(), sizeof(int16_t), n_sample, f);
    std::fclose(f);
    return got == n_sample;
}

// Callback: append every streamed int16 sample into the sink vector.
int append_cb(const int16_t* samples, int n_samples, void* user) {
    auto* sink = static_cast<std::vector<int16_t>*>(user);
    sink->insert(sink->end(), samples, samples + n_samples);
    return 0;  // keep going
}

}  // namespace

int main() {
    const char* tts   = std::getenv("VIBEVOICE_TTS_MODEL");
    const char* tok   = std::getenv("VIBEVOICE_TOKENIZER");
    const char* voice = std::getenv("VIBEVOICE_VOICE");
    if (!file_ok(tts) || !file_ok(tok) || !file_ok(voice)) {
        std::fprintf(stderr,
            "skip: capi stream test needs VIBEVOICE_{TTS_MODEL,TOKENIZER,"
            "VOICE} all set.\n");
        return 77;
    }

    std::printf("[capi-stream] %s\n", vv_capi_version());

    int rc = vv_capi_load(tts, /*asr=*/nullptr, tok, voice, /*n_threads=*/0);
    if (rc != 0) { std::fprintf(stderr, "FAIL: vv_capi_load rc=%d\n", rc); return 1; }

    const char*    text  = "Hello world this is a streaming parity test.";
    const int      steps = 20;
    const float    cfg   = 1.3f;
    const int      maxfr = 200;
    const uint32_t seed  = 0xCAFE;

    // --- File path: vv_capi_tts → WAV → read PCM back. ---
    const char* wav = "/tmp/vibevoice_capi_stream.wav";
    rc = vv_capi_tts(text, /*voice_path=*/nullptr,
                     /*ref_audio_paths=*/nullptr, /*n_ref=*/0,
                     wav, steps, cfg, maxfr, seed);
    if (rc != 0) { std::fprintf(stderr, "FAIL: vv_capi_tts rc=%d\n", rc); return 2; }

    std::vector<int16_t> file_pcm;
    if (!read_wav_pcm16(wav, &file_pcm) || file_pcm.empty()) {
        std::fprintf(stderr, "FAIL: could not read PCM from %s\n", wav);
        return 3;
    }
    std::remove(wav);

    // --- Callback path: vv_capi_tts_stream → append int16. Same seed. ---
    std::vector<int16_t> stream_pcm;
    stream_pcm.reserve(file_pcm.size());
    rc = vv_capi_tts_stream(text, /*voice_path=*/nullptr,
                            steps, cfg, maxfr, seed, append_cb, &stream_pcm);
    if (rc != 0) { std::fprintf(stderr, "FAIL: vv_capi_tts_stream rc=%d\n", rc); return 4; }

    // --- Assert byte-identical. ---
    if (stream_pcm.size() != file_pcm.size()) {
        std::fprintf(stderr,
            "FAIL: sample count mismatch: stream=%zu file=%zu\n",
            stream_pcm.size(), file_pcm.size());
        return 5;
    }
    for (size_t i = 0; i < file_pcm.size(); ++i) {
        if (stream_pcm[i] != file_pcm[i]) {
            std::fprintf(stderr,
                "FAIL: sample %zu differs: stream=%d file=%d\n",
                i, stream_pcm[i], file_pcm[i]);
            return 6;
        }
    }

    std::printf("[capi-stream] OK: %zu int16 samples identical (callback == file)\n",
                file_pcm.size());
    vv_capi_unload();
    return 0;
}
