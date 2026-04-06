#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "moonshine-impl.h"  // reuse moonshine_kv_cache

#include <vector>

struct moonshine_streaming_hparams {
    // Encoder
    uint32_t enc_hidden_size;
    uint32_t enc_n_layers;
    uint32_t enc_n_heads;
    uint32_t enc_n_kv_heads;
    uint32_t enc_head_dim;
    uint32_t enc_intermediate;

    // Decoder (can differ from encoder)
    uint32_t dec_hidden_size;
    uint32_t dec_n_layers;
    uint32_t dec_n_heads;
    uint32_t dec_n_kv_heads;
    uint32_t dec_head_dim;
    uint32_t dec_intermediate;

    // Shared
    uint32_t vocab_size;
    uint32_t bos_token_id;
    uint32_t eos_token_id;

    // RoPE (decoder only — encoder has no positional encoding)
    float rope_theta;
    float partial_rotary_factor;

    // Frontend
    uint32_t frame_len;           // 80 (5ms @ 16kHz)
    uint32_t conv1_kernel_size;   // 5
    uint32_t conv1_stride;        // 2
    uint32_t conv2_kernel_size;   // 5
    uint32_t conv2_stride;        // 2
    float    asinh_k;             // exp(log_k) — precomputed at load time

    // Adapter
    uint32_t max_position_embeddings;  // 4096

    // Encoder sliding windows: [left, right] per layer, flattened
    std::vector<int32_t> sliding_windows;

    // Derived
    uint32_t enc_rotary_dim;  // dec_head_dim * partial_rotary_factor
};

// Encoder layer (streaming): no attention bias, unit-offset layernorm, GELU FFN with bias
struct moonshine_streaming_enc_layer {
    struct ggml_tensor * attn_norm;     // unit-offset: applied as (gamma + 1) * LN(x)
    struct ggml_tensor * attn_q;
    struct ggml_tensor * attn_k;
    struct ggml_tensor * attn_v;
    struct ggml_tensor * attn_o;
    struct ggml_tensor * ffn_norm;      // unit-offset
    struct ggml_tensor * ffn_fc1_w;
    struct ggml_tensor * ffn_fc1_b;
    struct ggml_tensor * ffn_fc2_w;
    struct ggml_tensor * ffn_fc2_b;
};

// Decoder layer (streaming): standard layernorm, gated SiLU FFN with fc1/fc2, cross-attn
struct moonshine_streaming_dec_layer {
    // Self-attention
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * attn_q;
    struct ggml_tensor * attn_k;
    struct ggml_tensor * attn_v;
    struct ggml_tensor * attn_o;
    // Cross-attention
    struct ggml_tensor * cross_attn_norm;
    struct ggml_tensor * cross_attn_q;
    struct ggml_tensor * cross_attn_k;
    struct ggml_tensor * cross_attn_v;
    struct ggml_tensor * cross_attn_o;
    // FFN (gated SiLU: fc1 produces 2*intermediate, split into gate+value)
    struct ggml_tensor * ffn_norm;
    struct ggml_tensor * ffn_fc1_w;
    struct ggml_tensor * ffn_fc1_b;
    struct ggml_tensor * ffn_fc2_w;
    struct ggml_tensor * ffn_fc2_b;
};

struct moonshine_streaming_model {
    moonshine_streaming_hparams hparams;

    // Frontend
    struct ggml_tensor * frontend_linear_w;   // [frame_len, enc_hidden]
    struct ggml_tensor * frontend_conv1_w;    // [K, enc_hidden, enc_hidden*2]
    struct ggml_tensor * frontend_conv1_b;    // [enc_hidden*2]
    struct ggml_tensor * frontend_conv2_w;    // [K, enc_hidden*2, enc_hidden]
    struct ggml_tensor * frontend_conv2_b;    // [enc_hidden]

    // Encoder layers
    std::vector<moonshine_streaming_enc_layer> enc_layers;

    // Encoder output norm (unit-offset)
    struct ggml_tensor * enc_output_norm;

    // Adapter
    struct ggml_tensor * adapter_pos_emb;     // [max_pos, enc_hidden]
    struct ggml_tensor * adapter_proj;        // [enc_hidden, dec_hidden] (nullptr if dims match)

    // Decoder
    struct ggml_tensor * dec_embed;           // [dec_hidden, vocab_size]
    struct ggml_tensor * dec_output_norm;
    struct ggml_tensor * dec_output;          // [dec_hidden, vocab_size] — separate (no weight tying)

    // Decoder layers
    std::vector<moonshine_streaming_dec_layer> dec_layers;

    // ggml state
    ggml_backend_buffer_t buf_w = nullptr;
    struct ggml_context * ctx_w = nullptr;

    moonshine_streaming_model() = default;
    ~moonshine_streaming_model() {
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_w);
    }

    moonshine_streaming_model(const moonshine_streaming_model &) = delete;
    moonshine_streaming_model & operator=(const moonshine_streaming_model &) = delete;
};

// ── Incremental streaming state ─────────────────────────────────────────────

struct moonshine_stream_state {
    // Incremental frontend buffers (Step 8)
    std::vector<float> sample_buffer;     // leftover audio samples (< frame_len)
    std::vector<float> conv1_buffer;      // causal conv1 carry state, ggml layout [buf_len, hidden, 1]
    std::vector<float> conv2_buffer;      // causal conv2 carry state, ggml layout [buf_len, hidden*2, 1]

    // Accumulated frontend features: hidden * feature_count floats
    // Layout matches ggml [hidden, feature_count]: feature_count groups of hidden floats
    std::vector<float> accumulated_features;
    int accumulated_feature_count = 0;

    // Encoder tracking
    int encoder_frames_emitted = 0;

    // Adapter position offset (cumulative frames passed to adapter)
    int adapter_pos_offset = 0;

    // Memory accumulator: dec_hidden * memory_len floats
    std::vector<float> memory;
    int memory_len = 0;

    // Cross-KV cache validity (invalidated when memory grows)
    bool cross_kv_valid = false;

    // Result text buffer (valid until next decode or reset)
    std::string result_text;

    // Initial buffer sizes for reset (set by create)
    int conv1_init_size = 0;   // (K1-1) * hidden
    int conv2_init_size = 0;   // (K2-1) * hidden * 2
};
