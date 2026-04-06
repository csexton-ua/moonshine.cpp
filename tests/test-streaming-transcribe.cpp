#include "moonshine-streaming.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Minimal WAV loader (16-bit PCM or 32-bit float, mono, any sample rate)
static bool load_wav(const char * path, std::vector<float> & audio, int & sample_rate) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open: %s\n", path);
        return false;
    }

    char riff[4]; fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0) { fclose(f); return false; }
    fseek(f, 4, SEEK_CUR); // skip file size
    char wave[4]; fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4) != 0) { fclose(f); return false; }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sr = 0;

    // Find fmt and data chunks
    while (true) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            long pos = ftell(f);
            fread(&audio_format, 2, 1, f);
            fread(&num_channels, 2, 1, f);
            fread(&sr, 4, 1, f);
            fseek(f, 6, SEEK_CUR); // skip byte rate + block align
            fread(&bits_per_sample, 2, 1, f);
            fseek(f, pos + chunk_size, SEEK_SET);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            sample_rate = (int)sr;
            if (audio_format == 1 && bits_per_sample == 16) {
                int n_samples = chunk_size / (2 * num_channels);
                audio.resize(n_samples);
                std::vector<int16_t> buf(n_samples * num_channels);
                fread(buf.data(), 2, n_samples * num_channels, f);
                for (int i = 0; i < n_samples; i++) {
                    float sum = 0;
                    for (int c = 0; c < num_channels; c++) {
                        sum += buf[i * num_channels + c] / 32768.0f;
                    }
                    audio[i] = sum / num_channels;
                }
            } else if (audio_format == 3 && bits_per_sample == 32) {
                int n_samples = chunk_size / (4 * num_channels);
                audio.resize(n_samples);
                std::vector<float> buf(n_samples * num_channels);
                fread(buf.data(), 4, n_samples * num_channels, f);
                for (int i = 0; i < n_samples; i++) {
                    float sum = 0;
                    for (int c = 0; c < num_channels; c++) {
                        sum += buf[i * num_channels + c];
                    }
                    audio[i] = sum / num_channels;
                }
            } else {
                fprintf(stderr, "Unsupported WAV format: fmt=%d bits=%d\n", audio_format, bits_per_sample);
                fclose(f);
                return false;
            }
            fclose(f);
            return true;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    fclose(f);
    return false;
}

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
    const char * audio_path = nullptr;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--audio") == 0) && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        }
    }

    if (!model_path) {
        fprintf(stderr, "usage: %s -m <model.gguf> [-a <audio.wav>] [-v]\n", argv[0]);
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

    // Load or generate audio
    std::vector<float> audio;
    if (audio_path) {
        int sr = 0;
        if (!load_wav(audio_path, audio, sr)) {
            fprintf(stderr, "Failed to load audio: %s\n", audio_path);
            moonshine_streaming_free(ctx);
            return 1;
        }
        if (sr != 16000) {
            fprintf(stderr, "WARNING: audio is %dHz, expected 16000Hz\n", sr);
        }
        printf("Audio: %s (%d samples, %.2fs)\n", audio_path, (int)audio.size(), audio.size() / 16000.0);
    } else {
        audio = make_sine_wave(440.0f, 1.0f);
        printf("Audio: 1s 440Hz sine wave (%d samples)\n", (int)audio.size());
    }

    // First, run frontend+encoder+adapter separately to check intermediate values
    if (verbose) {
        float * features = nullptr;
        int feat_seq = 0, feat_hidden = 0;
        int ret = moonshine_streaming_frontend(ctx, audio.data(), (int)audio.size(),
                                                &features, &feat_seq, &feat_hidden);
        if (ret == 0) {
            printf("\nFrontend output: [%d, %d]\n", feat_hidden, feat_seq);
            printf("  first 10 values: ");
            for (int i = 0; i < 10 && i < feat_hidden; i++) printf("%.6f ", features[i]);
            printf("\n");

            float * encoded = nullptr;
            int enc_seq = 0, enc_hidden = 0;
            ret = moonshine_streaming_encoder(ctx, features, feat_seq, feat_hidden,
                                               &encoded, &enc_seq, &enc_hidden);
            if (ret == 0) {
                printf("Encoder output: [%d, %d]\n", enc_hidden, enc_seq);
                printf("  first 10 values: ");
                for (int i = 0; i < 10 && i < enc_hidden; i++) printf("%.6f ", encoded[i]);
                printf("\n");
                free(encoded);
            }
            free(features);
        }
    }

    // Transcribe
    printf("\nTranscribing...\n");
    const char * text = moonshine_streaming_transcribe(ctx, audio.data(), (int)audio.size());

    if (text && text[0] != '\0') {
        printf("\nResult: \"%s\"\n", text);
    } else {
        printf("\nResult: (empty)\n");
    }

    moonshine_streaming_free(ctx);
    return 0;
}
