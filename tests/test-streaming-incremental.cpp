#include "moonshine-streaming.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
    fseek(f, 4, SEEK_CUR);
    char wave[4]; fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4) != 0) { fclose(f); return false; }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sr = 0;

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
            fseek(f, 6, SEEK_CUR);
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
    int chunk_ms = 1000;  // default 1s chunks

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--audio") == 0) && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (strcmp(argv[i], "--chunk-ms") == 0 && i + 1 < argc) {
            chunk_ms = atoi(argv[++i]);
        }
    }

    if (!model_path) {
        fprintf(stderr, "usage: %s -m <model.gguf> [-a <audio.wav>] [--chunk-ms N]\n", argv[0]);
        return 1;
    }

    // Load model
    printf("Loading model: %s\n", model_path);
    struct moonshine_streaming_context * ctx = moonshine_streaming_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
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
        audio = make_sine_wave(440.0f, 3.0f);
        printf("Audio: 3s 440Hz sine wave (%d samples)\n", (int)audio.size());
    }

    // ── Step 1: Batch transcribe (ground truth) ─────────────────────────────
    printf("\n=== Batch transcribe (ground truth) ===\n");
    const char * batch_text = moonshine_streaming_transcribe(ctx, audio.data(), (int)audio.size());
    std::string batch_result = batch_text ? batch_text : "";
    printf("Batch result: \"%s\"\n", batch_result.c_str());

    // ── Step 2: Incremental streaming ───────────────────────────────────────
    printf("\n=== Incremental streaming (chunk_ms=%d) ===\n", chunk_ms);

    const int chunk_samples = 16000 * chunk_ms / 1000;
    const int total_samples = (int)audio.size();

    struct moonshine_stream_state * state = moonshine_stream_create(ctx);
    if (!state) {
        fprintf(stderr, "Failed to create stream state\n");
        moonshine_streaming_free(ctx);
        return 1;
    }

    int offset = 0;
    int chunk_num = 0;
    std::string last_text;

    while (offset < total_samples) {
        int remaining = total_samples - offset;
        int this_chunk = (remaining < chunk_samples) ? remaining : chunk_samples;
        bool is_last = (offset + this_chunk >= total_samples);

        chunk_num++;
        printf("\n--- Chunk %d: samples [%d, %d) (%.2fs - %.2fs)%s ---\n",
               chunk_num, offset, offset + this_chunk,
               offset / 16000.0, (offset + this_chunk) / 16000.0,
               is_last ? " [FINAL]" : "");

        // Feed audio
        int new_features = moonshine_stream_process_audio(ctx, state,
                                                           audio.data() + offset, this_chunk);
        if (new_features < 0) {
            fprintf(stderr, "process_audio failed\n");
            break;
        }
        printf("  new features: %d\n", new_features);

        // Encode
        int new_memory = moonshine_stream_encode(ctx, state, is_last);
        if (new_memory < 0) {
            fprintf(stderr, "encode failed\n");
            break;
        }
        printf("  new memory frames: %d\n", new_memory);

        // Decode (only if we have new memory to decode)
        if (new_memory > 0 || is_last) {
            const char * text = moonshine_stream_decode(ctx, state);
            last_text = text ? text : "";
            printf("  partial transcript: \"%s\"\n", last_text.c_str());
        } else {
            printf("  (no new memory, skipping decode)\n");
        }

        offset += this_chunk;
    }

    printf("\n=== Comparison ===\n");
    printf("Batch:     \"%s\"\n", batch_result.c_str());
    printf("Streaming: \"%s\"\n", last_text.c_str());

    bool match = (batch_result == last_text);
    printf("Match: %s\n", match ? "YES" : "NO");

    if (!match && !batch_result.empty()) {
        fprintf(stderr, "\nWARNING: Streaming result differs from batch result.\n");
        fprintf(stderr, "This may indicate a bug in the incremental encode/decode logic.\n");
    }

    moonshine_stream_free(state);
    moonshine_streaming_free(ctx);

    return match ? 0 : (batch_result.empty() ? 0 : 1);
}
