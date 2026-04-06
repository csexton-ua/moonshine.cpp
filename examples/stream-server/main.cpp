/*
 * moonshine-stream-server: long-lived process for WebSocket-based streaming.
 *
 * Reads a binary protocol from stdin (audio chunks), runs the incremental
 * streaming API, and writes partial transcripts to stdout.
 *
 * Protocol (stdin, little-endian):
 *   [uint32 msg_type] [uint32 payload_bytes] [payload...]
 *     msg_type 1 = AUDIO   (payload: float32 PCM, 16kHz mono)
 *     msg_type 2 = FINALIZE (payload: empty)
 *     msg_type 3 = RESET    (payload: empty)
 *
 * Protocol (stdout, little-endian):
 *   [uint32 payload_bytes] [uint8 flags] [UTF-8 text]
 *     flags & 0x01 = segment_final (silence/time-limit auto-reset boundary)
 */

#include "moonshine-streaming.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

enum msg_type : uint32_t {
    MSG_AUDIO    = 1,
    MSG_FINALIZE = 2,
    MSG_RESET    = 3,
};

static bool read_exact(FILE * f, void * buf, size_t n) {
    size_t got = fread(buf, 1, n, f);
    return got == n;
}

static float compute_rms(const float * samples, int n) {
    if (n <= 0) return 0.0f;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)samples[i] * samples[i];
    }
    return (float)sqrt(sum / n);
}

static bool write_exact(FILE * f, const void * buf, size_t n) {
    size_t wrote = fwrite(buf, 1, n, f);
    return wrote == n;
}

static const uint8_t RESP_SEGMENT_FINAL = 1;

static bool send_response(const char * text, uint8_t flags = 0) {
    uint32_t text_len = text ? (uint32_t)strlen(text) : 0;
    uint32_t payload_len = 1 + text_len;
    if (!write_exact(stdout, &payload_len, 4)) return false;
    if (!write_exact(stdout, &flags, 1)) return false;
    if (text_len > 0 && !write_exact(stdout, text, text_len)) return false;
    fflush(stdout);
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    if (!model_path) {
        fprintf(stderr, "usage: %s -m <model.gguf>\n", argv[0]);
        return 1;
    }

#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    // Load model
    fprintf(stderr, "[stream-server] loading model: %s\n", model_path);
    struct moonshine_streaming_context * ctx = moonshine_streaming_init(model_path);
    if (!ctx) {
        fprintf(stderr, "[stream-server] failed to load model\n");
        return 1;
    }

    struct moonshine_stream_state * state = moonshine_stream_create(ctx);
    if (!state) {
        fprintf(stderr, "[stream-server] failed to create stream state\n");
        moonshine_streaming_free(ctx);
        return 1;
    }

    const int hard_reset_threshold = 750;        // ~15 seconds at 50 fps
    const int min_segment_frames = 100;          // ~2 seconds minimum before silence reset
    const float silence_rms_threshold = 0.008f;
    const int silence_chunks_required = 2;       // ~400ms of silence
    int silence_chunk_count = 0;
    fprintf(stderr, "[stream-server] segmentation: hard limit %d frames (~%ds), max %d\n",
            hard_reset_threshold, hard_reset_threshold / 50,
            moonshine_stream_get_max_position(ctx));

    // Signal ready
    fprintf(stdout, "READY\n");
    fflush(stdout);
    fprintf(stderr, "[stream-server] ready\n");

    // Main loop: read messages from stdin
    std::vector<float> audio;
    while (true) {
        uint32_t header[2]; // [msg_type, payload_bytes]
        if (!read_exact(stdin, header, 8)) {
            break;
        }

        uint32_t type = header[0];
        uint32_t payload_bytes = header[1];

        if (type == MSG_AUDIO) {
            if (payload_bytes == 0 || payload_bytes % sizeof(float) != 0) {
                fprintf(stderr, "[stream-server] invalid audio payload: %u bytes\n", payload_bytes);
                break;
            }

            int n_samples = payload_bytes / sizeof(float);
            audio.resize(n_samples);
            if (!read_exact(stdin, audio.data(), payload_bytes)) {
                fprintf(stderr, "[stream-server] failed to read audio payload\n");
                break;
            }

            int new_features = moonshine_stream_process_audio(ctx, state, audio.data(), n_samples);
            if (new_features < 0) {
                fprintf(stderr, "[stream-server] process_audio failed\n");
                send_response("");
                continue;
            }

            // Check segmentation before encoding: silence or hard time limit
            const int pos = moonshine_stream_get_position(state);
            const float rms = compute_rms(audio.data(), n_samples);

            if (rms < silence_rms_threshold) {
                silence_chunk_count++;
            } else {
                silence_chunk_count = 0;
            }

            const bool at_hard_limit = (pos >= hard_reset_threshold);
            const bool at_silence = (pos >= min_segment_frames &&
                                     silence_chunk_count >= silence_chunks_required);

            if (at_hard_limit || at_silence) {
                fprintf(stderr, "[stream-server] %s reset at position %d\n",
                        at_hard_limit ? "hard" : "silence", pos);
                moonshine_stream_encode(ctx, state, true);
                const char * final_text = moonshine_stream_decode(ctx, state);
                if (!send_response(final_text ? final_text : "", RESP_SEGMENT_FINAL)) break;
                moonshine_stream_reset(state);
                silence_chunk_count = 0;
                continue;
            }

            int new_memory = moonshine_stream_encode(ctx, state, false);
            if (new_memory < 0) {
                fprintf(stderr, "[stream-server] encode failed\n");
                send_response("");
                continue;
            }

            const char * text = "";
            if (new_memory > 0) {
                const char * decoded = moonshine_stream_decode(ctx, state);
                if (decoded) text = decoded;
            }

            if (!send_response(text)) break;

        } else if (type == MSG_FINALIZE) {
            // Skip any payload (should be 0)
            if (payload_bytes > 0) {
                std::vector<char> discard(payload_bytes);
                read_exact(stdin, discard.data(), payload_bytes);
            }

            // Flush final frames
            moonshine_stream_encode(ctx, state, true);
            const char * text = moonshine_stream_decode(ctx, state);
            if (!send_response(text ? text : "")) break;

        } else if (type == MSG_RESET) {
            if (payload_bytes > 0) {
                std::vector<char> discard(payload_bytes);
                read_exact(stdin, discard.data(), payload_bytes);
            }

            moonshine_stream_reset(state);
            if (!send_response("")) break;

        } else {
            fprintf(stderr, "[stream-server] unknown message type: %u\n", type);
            break;
        }
    }

    moonshine_stream_free(state);
    moonshine_streaming_free(ctx);
    fprintf(stderr, "[stream-server] shutdown\n");
    return 0;
}
