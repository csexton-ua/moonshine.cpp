#include "moonshine.h"
#include "moonshine-audio.h"
#include "moonshine-tokenizer.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_pass = 0;
static int n_fail = 0;

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            n_fail++; \
        } else { \
            n_pass++; \
        } \
    } while (0)

#define TEST_ASSERT_MSG(expr, msg) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "FAIL: %s:%d: %s (%s)\n", __FILE__, __LINE__, #expr, msg); \
            n_fail++; \
        } else { \
            n_pass++; \
        } \
    } while (0)

static std::string fixtures_path;
static std::string model_path;

static void test_wav_loading() {
    fprintf(stderr, "=== test_wav_loading ===\n");

    std::string wav_path = fixtures_path + "/beckett.wav";
    std::vector<float> audio;
    int32_t sample_rate = 0;

    bool ok = moonshine_load_wav(wav_path.c_str(), audio, &sample_rate);
    TEST_ASSERT(ok);
    TEST_ASSERT(sample_rate == 16000);

    // beckett.wav is ~10s at 16kHz = ~160000 samples, allow some range
    TEST_ASSERT_MSG(audio.size() > 140000, "too few samples");
    TEST_ASSERT_MSG(audio.size() < 180000, "too many samples");

    // All values should be in [-1, 1]
    bool in_range = true;
    for (size_t i = 0; i < audio.size(); i++) {
        if (audio[i] < -1.0f || audio[i] > 1.0f) {
            in_range = false;
            break;
        }
    }
    TEST_ASSERT(in_range);
}

static void test_tokenizer() {
    fprintf(stderr, "=== test_tokenizer ===\n");

    // Find tokenizer.bin alongside model, or in fixtures
    std::string tok_path;
    if (!model_path.empty()) {
        // derive from model directory
        std::string dir = model_path;
        size_t pos = dir.find_last_of("/\\");
        if (pos != std::string::npos) dir = dir.substr(0, pos);
        tok_path = dir + "/tokenizer.bin";
    } else {
        fprintf(stderr, "  SKIP: no --model provided, cannot locate tokenizer\n");
        return;
    }

    moonshine_tokenizer tokenizer;
    bool ok = tokenizer.load(tok_path.c_str());
    TEST_ASSERT(ok);

    // vocab_size should be reasonable (32768 for moonshine)
    TEST_ASSERT_MSG(tokenizer.vocab_size() > 1000, "vocab too small");
    TEST_ASSERT_MSG(tokenizer.vocab_size() < 100000, "vocab too large");

    // Token 0 is typically a special token — should produce empty string
    {
        std::vector<int32_t> ids = {0};
        std::string text = tokenizer.tokens_to_text(ids);
        TEST_ASSERT_MSG(text.empty(), "token 0 should be empty/special");
    }

    // A known token should decode to something non-empty
    {
        std::vector<int32_t> ids = {1000};
        std::string text = tokenizer.tokens_to_text(ids);
        TEST_ASSERT_MSG(!text.empty(), "token 1000 should decode to something");
    }
}

static void test_transcription() {
    fprintf(stderr, "=== test_transcription ===\n");

    if (model_path.empty()) {
        fprintf(stderr, "  SKIP: no --model provided\n");
        return;
    }

    struct moonshine_context * ctx = moonshine_init(model_path.c_str());
    TEST_ASSERT(ctx != nullptr);
    if (!ctx) return;

    std::string wav_path = fixtures_path + "/beckett.wav";
    std::vector<float> audio;
    int32_t sample_rate = 0;
    bool ok = moonshine_load_wav(wav_path.c_str(), audio, &sample_rate);
    TEST_ASSERT(ok);

    const char * text = moonshine_transcribe(ctx, audio.data(), (int)audio.size());
    TEST_ASSERT(text != nullptr);
    TEST_ASSERT(strlen(text) > 0);

    fprintf(stderr, "  transcription: \"%s\"\n", text);

    // Check timing was recorded
    struct moonshine_timing timing = {};
    moonshine_get_timing(ctx, &timing);
    TEST_ASSERT(timing.encode_ms > 0);
    TEST_ASSERT(timing.decode_ms > 0);
    TEST_ASSERT(timing.n_tokens > 0);
    TEST_ASSERT(timing.n_samples == (int)audio.size());

    moonshine_free(ctx);
}

int main(int argc, char ** argv) {
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--fixtures") == 0 && i + 1 < argc) {
            fixtures_path = argv[++i];
        }
    }

    if (fixtures_path.empty()) {
        // Default: tests/fixtures relative to executable or current dir
        fixtures_path = "tests/fixtures";
    }

    fprintf(stderr, "fixtures: %s\n", fixtures_path.c_str());
    if (!model_path.empty()) {
        fprintf(stderr, "model:    %s\n", model_path.c_str());
    }

    test_wav_loading();
    test_tokenizer();
    test_transcription();

    fprintf(stderr, "\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
