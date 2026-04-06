#include "moonshine-streaming.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Generate a sine wave test signal
static std::vector<float> make_sine_wave(float freq_hz, float duration_s, int sample_rate = 16000) {
    int n = (int)(duration_s * sample_rate);
    std::vector<float> audio(n);
    for (int i = 0; i < n; i++) {
        audio[i] = 0.5f * sinf(2.0f * M_PI * freq_hz * i / sample_rate);
    }
    return audio;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    bool verbose = false;
    const char * dump_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_path = argv[++i];
        }
    }

    if (!model_path) {
        fprintf(stderr, "usage: %s -m <model.gguf> [-v] [--dump output.bin]\n", argv[0]);
        return 1;
    }

    // Load model
    printf("Loading model: %s\n", model_path);
    struct moonshine_streaming_context * ctx = moonshine_streaming_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    if (verbose) {
        moonshine_streaming_print_model_info(ctx);
    }

    // Generate 1 second of 440Hz sine wave
    auto audio = make_sine_wave(440.0f, 1.0f);
    printf("Audio: %zu samples (%.2f sec)\n", audio.size(), audio.size() / 16000.0f);

    // Run frontend
    float * features = nullptr;
    int seq_len = 0;
    int hidden_dim = 0;

    int ret = moonshine_streaming_frontend(ctx, audio.data(), (int)audio.size(),
                                            &features, &seq_len, &hidden_dim);
    if (ret != 0) {
        fprintf(stderr, "Frontend failed\n");
        moonshine_streaming_free(ctx);
        return 1;
    }

    printf("Frontend output: [%d, %d] (hidden x seq_len)\n", hidden_dim, seq_len);

    // Print some statistics
    float min_val = features[0], max_val = features[0], sum = 0;
    for (int i = 0; i < hidden_dim * seq_len; i++) {
        if (features[i] < min_val) min_val = features[i];
        if (features[i] > max_val) max_val = features[i];
        sum += features[i];
    }
    float mean = sum / (hidden_dim * seq_len);
    printf("Stats: min=%.4f max=%.4f mean=%.4f\n", min_val, max_val, mean);

    // Print first few values for comparison with Python
    printf("First 10 values: ");
    for (int i = 0; i < 10 && i < hidden_dim * seq_len; i++) {
        printf("%.6f ", features[i]);
    }
    printf("\n");

    // Dump to binary file for comparison
    if (dump_path) {
        FILE * f = fopen(dump_path, "wb");
        if (f) {
            int32_t dims[2] = { hidden_dim, seq_len };
            fwrite(dims, sizeof(int32_t), 2, f);
            fwrite(features, sizeof(float), hidden_dim * seq_len, f);
            fclose(f);
            printf("Dumped features to %s\n", dump_path);
        }
    }

    free(features);
    moonshine_streaming_free(ctx);
    printf("OK\n");
    return 0;
}
