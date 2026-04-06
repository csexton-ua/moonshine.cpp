#include "moonshine.h"
#include "moonshine-streaming.h"
#include "moonshine-detect.h"
#include "moonshine-audio.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct cli_params {
    const char * model_path    = nullptr;
    const char * tokenizer_path = nullptr;
    const char * audio_path    = nullptr;
    int  n_threads  = 4;
    bool verbose    = false;
    bool benchmark  = false;
    int  bench_runs = 10;
    bool stream     = false;
    float chunk_sec = 1.0f;     // seconds per chunk in --stream mode
};

static void print_usage(const char * prog) {
    fprintf(stderr,
        "usage: %s [options] <audio.wav>\n"
        "\n"
        "options:\n"
        "  -m, --model PATH       Path to GGUF model file (required)\n"
        "  -t, --tokenizer PATH   Path to tokenizer.bin (default: auto-detect)\n"
        "      --threads N        Number of threads (default: 4)\n"
        "  -v, --verbose          Print timing info to stderr\n"
        "      --benchmark [N]    Run N times and print stats (default: 10, implies -v)\n"
        "      --stream           Incremental streaming mode (v2 models only)\n"
        "      --chunk-sec F      Seconds per chunk in stream mode (default: 1.0)\n"
        "  -h, --help             Show this help\n"
        "\n"
        "The model architecture (v1 or v2/streaming) is auto-detected from the GGUF file.\n"
        "Both model types produce transcription output. The --stream flag enables\n"
        "incremental processing for v2 models, printing partial results as they appear.\n",
        prog);
}

static double print_bench_line(const char * label, const std::vector<double> & v) {
    double s = 0; for (double x : v) s += x;
    double a = s / v.size();
    fprintf(stderr, "  %-7s avg=%7.1f ms  min=%7.1f ms  max=%7.1f ms\n",
            label, a,
            *std::min_element(v.begin(), v.end()),
            *std::max_element(v.begin(), v.end()));
    return a;
}

static void print_perf_stats(int n_tokens, double total_ms, double audio_duration_s) {
    if (n_tokens > 0 && total_ms > 0) {
        fprintf(stderr, "  tokens/sec: %.1f\n", n_tokens / (total_ms / 1000.0));
    }
    if (audio_duration_s > 0 && total_ms > 0) {
        fprintf(stderr, "  real-time factor: %.3f\n", (total_ms / 1000.0) / audio_duration_s);
    }
}

static bool parse_args(int argc, char ** argv, cli_params & p) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];

        if ((strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) && i + 1 < argc) {
            p.model_path = argv[++i];
        } else if ((strcmp(arg, "-t") == 0 || strcmp(arg, "--tokenizer") == 0) && i + 1 < argc) {
            p.tokenizer_path = argv[++i];
        } else if (strcmp(arg, "--threads") == 0 && i + 1 < argc) {
            p.n_threads = atoi(argv[++i]);
            if (p.n_threads < 1) p.n_threads = 1;
        } else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            p.verbose = true;
        } else if (strcmp(arg, "--benchmark") == 0) {
            p.benchmark = true;
            p.verbose = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                p.bench_runs = atoi(argv[++i]);
                if (p.bench_runs < 1) p.bench_runs = 1;
            }
        } else if (strcmp(arg, "--stream") == 0) {
            p.stream = true;
        } else if (strcmp(arg, "--chunk-sec") == 0 && i + 1 < argc) {
            p.chunk_sec = (float)atof(argv[++i]);
            if (p.chunk_sec < 0.1f) p.chunk_sec = 0.1f;
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (arg[0] == '-') {
            fprintf(stderr, "unknown option: %s\n", arg);
            return false;
        } else {
            p.audio_path = arg;
        }
    }
    return true;
}

// ── v1 batch transcription ──────────────────────────────────────────────────

static int run_v1(const cli_params & p, const std::vector<float> & audio, double audio_duration_s) {
    struct moonshine_init_params init_params = {};
    init_params.model_path     = p.model_path;
    init_params.tokenizer_path = p.tokenizer_path;
    init_params.n_threads      = p.n_threads;

    struct moonshine_context * ctx = moonshine_init_with_params(init_params);
    if (!ctx) {
        fprintf(stderr, "failed to load v1 model from '%s'\n", p.model_path);
        return 1;
    }

    if (p.verbose) {
        fprintf(stderr, "model:   %s (v1)\n", p.model_path);
        fprintf(stderr, "threads: %d\n", moonshine_get_n_threads(ctx));
        fprintf(stderr, "audio:   %s (%zu samples, %.2f s)\n",
                p.audio_path, audio.size(), audio_duration_s);
    }

    if (p.benchmark) {
        std::vector<double> total_ms(p.bench_runs);
        std::vector<double> encode_ms(p.bench_runs);
        std::vector<double> decode_ms(p.bench_runs);
        int n_tokens = 0;

        for (int run = 0; run < p.bench_runs; run++) {
            const char * text = moonshine_transcribe(ctx, audio.data(), (int)audio.size());

            struct moonshine_timing timing = {};
            moonshine_get_timing(ctx, &timing);

            total_ms[run]  = timing.encode_ms + timing.decode_ms;
            encode_ms[run] = timing.encode_ms;
            decode_ms[run] = timing.decode_ms;
            n_tokens       = timing.n_tokens;

            if (run == 0 && text && text[0]) {
                printf("%s\n", text);
            }

            fprintf(stderr, "  run %2d: total=%7.1f ms  encode=%7.1f ms  decode=%7.1f ms\n",
                    run + 1, total_ms[run], encode_ms[run], decode_ms[run]);
        }

        fprintf(stderr, "\nbenchmark results (%d runs):\n", p.bench_runs);
        double avg_total = print_bench_line("total:", total_ms);
        print_bench_line("encode:", encode_ms);
        print_bench_line("decode:", decode_ms);
        print_perf_stats(n_tokens, avg_total, audio_duration_s);
    } else {
        const char * text = moonshine_transcribe(ctx, audio.data(), (int)audio.size());

        if (text && text[0]) {
            printf("%s\n", text);
        } else {
            fprintf(stderr, "transcription failed or produced empty output\n");
            moonshine_free(ctx);
            return 1;
        }

        if (p.verbose) {
            struct moonshine_timing timing = {};
            moonshine_get_timing(ctx, &timing);
            double total = timing.encode_ms + timing.decode_ms;

            fprintf(stderr, "\ntiming:\n");
            fprintf(stderr, "  encode:    %7.1f ms\n", timing.encode_ms);
            fprintf(stderr, "  decode:    %7.1f ms  (%d tokens, %.1f ms/token)\n",
                    timing.decode_ms, timing.n_tokens,
                    timing.n_tokens > 0 ? timing.decode_ms / timing.n_tokens : 0.0);
            fprintf(stderr, "  total:     %7.1f ms\n", total);
            print_perf_stats(timing.n_tokens, total, audio_duration_s);
        }
    }

    moonshine_free(ctx);
    return 0;
}

// ── v2 batch transcription ──────────────────────────────────────────────────

static int run_v2_batch(const cli_params & p, const std::vector<float> & audio, double audio_duration_s) {
    struct moonshine_streaming_init_params init_params = {};
    init_params.model_path     = p.model_path;
    init_params.tokenizer_path = p.tokenizer_path;
    init_params.n_threads      = p.n_threads;

    struct moonshine_streaming_context * ctx = moonshine_streaming_init_with_params(init_params);
    if (!ctx) {
        fprintf(stderr, "failed to load v2 model from '%s'\n", p.model_path);
        return 1;
    }

    if (p.verbose) {
        fprintf(stderr, "model:   %s (v2/streaming)\n", p.model_path);
        fprintf(stderr, "audio:   %s (%zu samples, %.2f s)\n",
                p.audio_path, audio.size(), audio_duration_s);
    }

    const char * text = moonshine_streaming_transcribe(ctx, audio.data(), (int)audio.size());

    if (text && text[0]) {
        printf("%s\n", text);
    } else {
        fprintf(stderr, "transcription failed or produced empty output\n");
        moonshine_streaming_free(ctx);
        return 1;
    }

    moonshine_streaming_free(ctx);
    return 0;
}

// ── v2 incremental streaming ────────────────────────────────────────────────

static int run_v2_stream(const cli_params & p, const std::vector<float> & audio, double audio_duration_s) {
    struct moonshine_streaming_init_params init_params = {};
    init_params.model_path     = p.model_path;
    init_params.tokenizer_path = p.tokenizer_path;
    init_params.n_threads      = p.n_threads;

    struct moonshine_streaming_context * ctx = moonshine_streaming_init_with_params(init_params);
    if (!ctx) {
        fprintf(stderr, "failed to load v2 model from '%s'\n", p.model_path);
        return 1;
    }

    if (p.verbose) {
        fprintf(stderr, "model:   %s (v2/streaming, incremental)\n", p.model_path);
        fprintf(stderr, "audio:   %s (%zu samples, %.2f s)\n",
                p.audio_path, audio.size(), audio_duration_s);
        fprintf(stderr, "chunk:   %.1f s (%d samples)\n",
                p.chunk_sec, (int)(p.chunk_sec * 16000));
    }

    struct moonshine_stream_state * state = moonshine_stream_create(ctx);
    if (!state) {
        fprintf(stderr, "failed to create streaming state\n");
        moonshine_streaming_free(ctx);
        return 1;
    }

    const int chunk_samples = (int)(p.chunk_sec * 16000);
    const int total_samples = (int)audio.size();
    int offset = 0;
    int prev_len = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    while (offset < total_samples) {
        int remaining = total_samples - offset;
        int n = (remaining < chunk_samples) ? remaining : chunk_samples;
        bool is_final = (offset + n >= total_samples);

        int new_features = moonshine_stream_process_audio(ctx, state, audio.data() + offset, n);
        if (new_features < 0) {
            fprintf(stderr, "stream_process_audio failed at offset %d\n", offset);
            break;
        }

        int new_frames = moonshine_stream_encode(ctx, state, is_final);
        if (new_frames < 0) {
            fprintf(stderr, "stream_encode failed at offset %d\n", offset);
            break;
        }

        if (new_frames > 0) {
            const char * text = moonshine_stream_decode(ctx, state);
            if (text) {
                int cur_len = (int)strlen(text);
                if (cur_len > prev_len) {
                    // Print only the new portion
                    printf("%s", text + prev_len);
                    fflush(stdout);
                    prev_len = cur_len;
                }
            }
        }

        offset += n;

        if (p.verbose && !is_final) {
            double elapsed_s = (double)offset / 16000.0;
            fprintf(stderr, "  [%.1f / %.1f s] features=%d frames=%d\n",
                    elapsed_s, audio_duration_s, new_features, new_frames);
        }
    }

    // Ensure final newline
    if (prev_len > 0) {
        printf("\n");
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (p.verbose) {
        fprintf(stderr, "\nstreaming total: %.1f ms (%.3f real-time factor)\n",
                total_ms, (total_ms / 1000.0) / audio_duration_s);
    }

    moonshine_stream_free(state);
    moonshine_streaming_free(ctx);
    return 0;
}

// ── main ────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    cli_params p;
    if (!parse_args(argc, argv, p)) {
        print_usage(argv[0]);
        return 1;
    }

    if (!p.model_path || !p.audio_path) {
        print_usage(argv[0]);
        return 1;
    }

    // Auto-detect model architecture
    enum moonshine_arch arch = moonshine_detect_arch(p.model_path);

    if (arch == MOONSHINE_ARCH_UNKNOWN) {
        fprintf(stderr, "error: could not detect model architecture from '%s'\n", p.model_path);
        return 1;
    }

    if (p.stream && arch != MOONSHINE_ARCH_V2) {
        fprintf(stderr, "error: --stream requires a v2 (streaming) model\n");
        return 1;
    }

    // Load audio
    std::vector<float> audio;
    int32_t sample_rate = 0;
    if (!moonshine_load_wav(p.audio_path, audio, &sample_rate)) {
        fprintf(stderr, "failed to load WAV file '%s'\n", p.audio_path);
        return 1;
    }

    double audio_duration_s = (double)audio.size() / sample_rate;

    // Dispatch to the appropriate code path
    if (arch == MOONSHINE_ARCH_V1) {
        return run_v1(p, audio, audio_duration_s);
    } else if (p.stream) {
        return run_v2_stream(p, audio, audio_duration_s);
    } else {
        return run_v2_batch(p, audio, audio_duration_s);
    }
}
