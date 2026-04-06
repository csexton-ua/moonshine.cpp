#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct moonshine_streaming_context;

struct moonshine_streaming_init_params {
    const char * model_path;
    const char * tokenizer_path;  // NULL = auto-detect from model directory
    int          n_threads;       // 0 = default (4)
};

struct moonshine_streaming_context * moonshine_streaming_init(const char * model_path);
struct moonshine_streaming_context * moonshine_streaming_init_with_params(
    struct moonshine_streaming_init_params params);
void moonshine_streaming_free(struct moonshine_streaming_context * ctx);

void moonshine_streaming_print_model_info(struct moonshine_streaming_context * ctx);

// Run the frontend (audio -> encoder features). Caller must free(*out_features) when done.
// Returns 0 on success. Output is [hidden, seq_len] in row-major order.
int moonshine_streaming_frontend(
    struct moonshine_streaming_context * ctx,
    const float * audio, int n_samples,
    float ** out_features, int * out_seq_len, int * out_hidden_dim);

// Run the encoder (features -> encoded). Caller must free(*out_encoded) when done.
// Input/output are [hidden, seq_len] in row-major order (same layout as frontend output).
int moonshine_streaming_encoder(
    struct moonshine_streaming_context * ctx,
    const float * features, int seq_len, int hidden_dim,
    float ** out_encoded, int * out_seq_len, int * out_hidden_dim);

// Full transcription: audio -> text. Returns pointer valid until next call or free.
const char * moonshine_streaming_transcribe(
    struct moonshine_streaming_context * ctx,
    const float * audio, int n_samples);

// ── Incremental streaming API (Step 6) ──────────────────────────────────────
//
// Feed audio chunks incrementally, get partial transcripts as audio arrives.
// Usage:
//   state = moonshine_stream_create(ctx);
//   while (have_audio) {
//       moonshine_stream_process_audio(ctx, state, chunk, chunk_len);
//       moonshine_stream_encode(ctx, state, false);
//       text = moonshine_stream_decode(ctx, state);
//   }
//   moonshine_stream_encode(ctx, state, true);  // flush final frames
//   text = moonshine_stream_decode(ctx, state);
//   moonshine_stream_free(state);

struct moonshine_stream_state;

// Create/destroy streaming session state
struct moonshine_stream_state * moonshine_stream_create(
    struct moonshine_streaming_context * ctx);
void moonshine_stream_free(struct moonshine_stream_state * state);

// Feed audio chunk. Re-runs frontend on all accumulated audio.
// Returns number of new frontend features, or -1 on error.
int moonshine_stream_process_audio(
    struct moonshine_streaming_context * ctx,
    struct moonshine_stream_state * state,
    const float * audio, int n_samples);

// Run encoder on accumulated features. Only "stable" frames (past lookahead)
// are emitted unless is_final=true. Returns number of new memory frames, or -1 on error.
int moonshine_stream_encode(
    struct moonshine_streaming_context * ctx,
    struct moonshine_stream_state * state,
    bool is_final);

// Run full decode on current memory. Returns text valid until next decode/reset/free.
const char * moonshine_stream_decode(
    struct moonshine_streaming_context * ctx,
    struct moonshine_stream_state * state);

// Reset state for new utterance (reuse same allocation)
void moonshine_stream_reset(struct moonshine_stream_state * state);

#ifdef __cplusplus
}
#endif
