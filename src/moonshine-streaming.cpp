#include "moonshine-streaming.h"
#include "moonshine-streaming-impl.h"
#include "moonshine-tokenizer.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct moonshine_streaming_context {
    moonshine_streaming_model model;
    moonshine_tokenizer tokenizer;
    ggml_backend_t backend = nullptr;
    std::string result_text;

    // Decoder state (initialized per-transcription, cleaned up after)
    moonshine_kv_cache kv_self;
    moonshine_kv_cache kv_cross;
    std::vector<float> encoder_out;
    int enc_len = 0;

    int n_threads = 4;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

static struct ggml_tensor * checked_get_tensor(struct ggml_context * ctx, const char * name, bool & ok) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "%s: tensor '%s' not found\n", __func__, name);
        ok = false;
    }
    return t;
}

static uint32_t gguf_get_u32(struct gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        fprintf(stderr, "warning: GGUF key '%s' not found, using 0\n", key);
        return 0;
    }
    return gguf_get_val_u32(ctx, id);
}

static float gguf_get_f32(struct gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        fprintf(stderr, "warning: GGUF key '%s' not found, using 0\n", key);
        return 0.0f;
    }
    return gguf_get_val_f32(ctx, id);
}

static std::string dir_of(const std::string & path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    return path.substr(0, pos);
}

// ── Model loading ────────────────────────────────────────────────────────────

struct moonshine_streaming_context * moonshine_streaming_init(const char * model_path) {
    struct moonshine_streaming_init_params params = {};
    params.model_path = model_path;
    params.tokenizer_path = nullptr;
    params.n_threads = 0;
    return moonshine_streaming_init_with_params(params);
}

struct moonshine_streaming_context * moonshine_streaming_init_with_params(
        struct moonshine_streaming_init_params params) {
    const char * model_path = params.model_path;
    auto * ctx = new moonshine_streaming_context();
    auto & model = ctx->model;

    ctx->n_threads = (params.n_threads > 0) ? params.n_threads : 4;

    // 1. Open GGUF file (no_alloc: create tensor metadata only)
    struct gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &model.ctx_w,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path, gguf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: failed to open GGUF file '%s'\n", __func__, model_path);
        delete ctx;
        return nullptr;
    }

    // Verify architecture
    {
        int64_t arch_key = gguf_find_key(ctx_gguf, "general.architecture");
        if (arch_key >= 0) {
            const char * arch = gguf_get_val_str(ctx_gguf, arch_key);
            if (strcmp(arch, "moonshine_streaming") != 0) {
                fprintf(stderr, "%s: expected architecture 'moonshine_streaming', got '%s'\n",
                        __func__, arch);
                gguf_free(ctx_gguf);
                delete ctx;
                return nullptr;
            }
        }
    }

    // 2. Read hyperparameters
    auto & hp = model.hparams;

    hp.enc_hidden_size   = gguf_get_u32(ctx_gguf, "moonshine_streaming.encoder.embedding_length");
    hp.enc_n_layers      = gguf_get_u32(ctx_gguf, "moonshine_streaming.encoder.block_count");
    hp.enc_n_heads       = gguf_get_u32(ctx_gguf, "moonshine_streaming.encoder.attention.head_count");
    hp.enc_n_kv_heads    = gguf_get_u32(ctx_gguf, "moonshine_streaming.encoder.attention.head_count_kv");
    hp.enc_intermediate  = gguf_get_u32(ctx_gguf, "moonshine_streaming.encoder.feed_forward_length");

    // Try explicit head_dim, else derive from hidden/heads
    {
        int64_t hd_key = gguf_find_key(ctx_gguf, "moonshine_streaming.encoder.attention.head_dim");
        if (hd_key >= 0) {
            hp.enc_head_dim = gguf_get_val_u32(ctx_gguf, hd_key);
        } else {
            hp.enc_head_dim = hp.enc_hidden_size / hp.enc_n_heads;
        }
    }

    hp.dec_hidden_size   = gguf_get_u32(ctx_gguf, "moonshine_streaming.decoder.embedding_length");
    hp.dec_n_layers      = gguf_get_u32(ctx_gguf, "moonshine_streaming.decoder.block_count");
    hp.dec_n_heads       = gguf_get_u32(ctx_gguf, "moonshine_streaming.decoder.attention.head_count");
    hp.dec_n_kv_heads    = gguf_get_u32(ctx_gguf, "moonshine_streaming.decoder.attention.head_count_kv");
    hp.dec_intermediate  = gguf_get_u32(ctx_gguf, "moonshine_streaming.decoder.feed_forward_length");
    {
        int64_t hd_key = gguf_find_key(ctx_gguf, "moonshine_streaming.decoder.attention.head_dim");
        if (hd_key >= 0) {
            hp.dec_head_dim = gguf_get_val_u32(ctx_gguf, hd_key);
        } else {
            hp.dec_head_dim = hp.dec_hidden_size / hp.dec_n_heads;
        }
    }

    hp.vocab_size  = gguf_get_u32(ctx_gguf, "moonshine_streaming.vocab_size");
    hp.bos_token_id = gguf_get_u32(ctx_gguf, "moonshine_streaming.bos_token_id");
    hp.eos_token_id = gguf_get_u32(ctx_gguf, "moonshine_streaming.eos_token_id");

    hp.rope_theta            = gguf_get_f32(ctx_gguf, "moonshine_streaming.rope.freq_base");
    hp.partial_rotary_factor = gguf_get_f32(ctx_gguf, "moonshine_streaming.decoder.partial_rotary_factor");

    hp.frame_len          = gguf_get_u32(ctx_gguf, "moonshine_streaming.frontend.frame_len");
    hp.conv1_kernel_size   = gguf_get_u32(ctx_gguf, "moonshine_streaming.frontend.conv1_kernel_size");
    hp.conv1_stride        = gguf_get_u32(ctx_gguf, "moonshine_streaming.frontend.conv1_stride");
    hp.conv2_kernel_size   = gguf_get_u32(ctx_gguf, "moonshine_streaming.frontend.conv2_kernel_size");
    hp.conv2_stride        = gguf_get_u32(ctx_gguf, "moonshine_streaming.frontend.conv2_stride");

    hp.max_position_embeddings = gguf_get_u32(ctx_gguf, "moonshine_streaming.adapter.max_position_embeddings");

    // Sliding windows array
    {
        int64_t sw_key = gguf_find_key(ctx_gguf, "moonshine_streaming.encoder.sliding_windows");
        if (sw_key >= 0) {
            // Array of int32 pairs, flattened: [left0, right0, left1, right1, ...]
            const size_t n = gguf_get_arr_n(ctx_gguf, sw_key);
            hp.sliding_windows.resize(n);
            const void * arr_data = gguf_get_arr_data(ctx_gguf, sw_key);
            memcpy(hp.sliding_windows.data(), arr_data, n * sizeof(int32_t));
        }
    }

    // Derived
    hp.enc_rotary_dim = (uint32_t)(hp.dec_head_dim * hp.partial_rotary_factor);

    // Validate
    if (hp.enc_hidden_size == 0 || hp.enc_n_heads == 0 || hp.enc_n_layers == 0 ||
        hp.dec_hidden_size == 0 || hp.dec_n_heads == 0 || hp.dec_n_layers == 0) {
        fprintf(stderr, "%s: invalid hparams (enc: hidden=%u heads=%u layers=%u, dec: hidden=%u heads=%u layers=%u)\n",
                __func__, hp.enc_hidden_size, hp.enc_n_heads, hp.enc_n_layers,
                hp.dec_hidden_size, hp.dec_n_heads, hp.dec_n_layers);
        gguf_free(ctx_gguf);
        delete ctx;
        return nullptr;
    }

    // 3. Allocate tensor buffer
    model.buf_w = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx_w, ggml_backend_cpu_buffer_type());
    if (!model.buf_w) {
        fprintf(stderr, "%s: failed to allocate tensor buffer\n", __func__);
        gguf_free(ctx_gguf);
        delete ctx;
        return nullptr;
    }

    // 4. Load tensor data from GGUF
    FILE * f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, model_path);
        gguf_free(ctx_gguf);
        delete ctx;
        return nullptr;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    const size_t data_offset = gguf_get_data_offset(ctx_gguf);
    std::vector<uint8_t> read_buf;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model.ctx_w, name);
        if (!tensor) {
            fprintf(stderr, "%s: tensor '%s' not found in context\n", __func__, name);
            fclose(f);
            gguf_free(ctx_gguf);
            delete ctx;
            return nullptr;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        const size_t offs = data_offset + gguf_get_tensor_offset(ctx_gguf, i);
        read_buf.resize(nbytes);

        if (fseek(f, offs, SEEK_SET) != 0 || fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "%s: failed to read tensor '%s'\n", __func__, name);
            fclose(f);
            gguf_free(ctx_gguf);
            delete ctx;
            return nullptr;
        }

        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }

    fclose(f);

    // Read asinh log_k scalar value before freeing GGUF context
    {
        struct ggml_tensor * log_k_tensor = ggml_get_tensor(model.ctx_w, "frontend.asinh.log_k");
        if (log_k_tensor) {
            float log_k_val = 0.0f;
            ggml_backend_tensor_get(log_k_tensor, &log_k_val, 0, sizeof(float));
            hp.asinh_k = expf(log_k_val);
        } else {
            fprintf(stderr, "%s: warning: frontend.asinh.log_k not found, using k=0.75\n", __func__);
            hp.asinh_k = 0.75f;
        }
    }

    gguf_free(ctx_gguf);

    // 5. Map tensors to model struct
    bool ok = true;

    // Frontend
    model.frontend_linear_w = checked_get_tensor(model.ctx_w, "frontend.linear.weight", ok);
    model.frontend_conv1_w  = checked_get_tensor(model.ctx_w, "frontend.conv1.weight", ok);
    model.frontend_conv1_b  = checked_get_tensor(model.ctx_w, "frontend.conv1.bias", ok);
    model.frontend_conv2_w  = checked_get_tensor(model.ctx_w, "frontend.conv2.weight", ok);
    model.frontend_conv2_b  = checked_get_tensor(model.ctx_w, "frontend.conv2.bias", ok);

    // Encoder layers
    model.enc_layers.resize(hp.enc_n_layers);
    for (uint32_t i = 0; i < hp.enc_n_layers; i++) {
        auto & layer = model.enc_layers[i];
        char name[128];

        snprintf(name, sizeof(name), "encoder.layers.%u.attn_norm.weight", i);
        layer.attn_norm = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "encoder.layers.%u.attn.q.weight", i);
        layer.attn_q = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "encoder.layers.%u.attn.k.weight", i);
        layer.attn_k = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "encoder.layers.%u.attn.v.weight", i);
        layer.attn_v = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "encoder.layers.%u.attn.o.weight", i);
        layer.attn_o = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "encoder.layers.%u.ffn_norm.weight", i);
        layer.ffn_norm = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "encoder.layers.%u.ffn.fc1.weight", i);
        layer.ffn_fc1_w = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "encoder.layers.%u.ffn.fc1.bias", i);
        layer.ffn_fc1_b = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "encoder.layers.%u.ffn.fc2.weight", i);
        layer.ffn_fc2_w = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "encoder.layers.%u.ffn.fc2.bias", i);
        layer.ffn_fc2_b = checked_get_tensor(model.ctx_w, name, ok);
    }

    // Encoder output norm
    model.enc_output_norm = checked_get_tensor(model.ctx_w, "encoder.output_norm.weight", ok);

    // Adapter
    model.adapter_pos_emb = checked_get_tensor(model.ctx_w, "adapter.pos_emb.weight", ok);
    // Projection only exists when enc_hidden != dec_hidden
    model.adapter_proj = ggml_get_tensor(model.ctx_w, "adapter.proj.weight");

    // Decoder
    model.dec_embed       = checked_get_tensor(model.ctx_w, "decoder.embed_tokens.weight", ok);
    model.dec_output_norm = checked_get_tensor(model.ctx_w, "decoder.output_norm.weight", ok);
    model.dec_output      = checked_get_tensor(model.ctx_w, "decoder.output.weight", ok);

    // Decoder layers
    model.dec_layers.resize(hp.dec_n_layers);
    for (uint32_t i = 0; i < hp.dec_n_layers; i++) {
        auto & layer = model.dec_layers[i];
        char name[128];

        snprintf(name, sizeof(name), "decoder.layers.%u.attn_norm.weight", i);
        layer.attn_norm = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%u.attn.q.weight", i);
        layer.attn_q = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "decoder.layers.%u.attn.k.weight", i);
        layer.attn_k = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "decoder.layers.%u.attn.v.weight", i);
        layer.attn_v = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "decoder.layers.%u.attn.o.weight", i);
        layer.attn_o = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%u.cross_attn_norm.weight", i);
        layer.cross_attn_norm = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "decoder.layers.%u.cross_attn.q.weight", i);
        layer.cross_attn_q = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "decoder.layers.%u.cross_attn.k.weight", i);
        layer.cross_attn_k = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "decoder.layers.%u.cross_attn.v.weight", i);
        layer.cross_attn_v = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "decoder.layers.%u.cross_attn.o.weight", i);
        layer.cross_attn_o = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%u.ffn_norm.weight", i);
        layer.ffn_norm = checked_get_tensor(model.ctx_w, name, ok);

        snprintf(name, sizeof(name), "decoder.layers.%u.ffn.fc1.weight", i);
        layer.ffn_fc1_w = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "decoder.layers.%u.ffn.fc1.bias", i);
        layer.ffn_fc1_b = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "decoder.layers.%u.ffn.fc2.weight", i);
        layer.ffn_fc2_w = checked_get_tensor(model.ctx_w, name, ok);
        snprintf(name, sizeof(name), "decoder.layers.%u.ffn.fc2.bias", i);
        layer.ffn_fc2_b = checked_get_tensor(model.ctx_w, name, ok);
    }

    if (!ok) {
        fprintf(stderr, "%s: one or more tensors missing from model\n", __func__);
        delete ctx;
        return nullptr;
    }

    // 6. Load tokenizer
    std::string tokenizer_path;
    if (params.tokenizer_path) {
        tokenizer_path = params.tokenizer_path;
    } else {
        tokenizer_path = dir_of(model_path) + "/tokenizer.bin";
    }
    if (!ctx->tokenizer.load(tokenizer_path.c_str())) {
        fprintf(stderr, "%s: failed to load tokenizer from '%s'\n", __func__, tokenizer_path.c_str());
        delete ctx;
        return nullptr;
    }

    // 7. Initialize CPU backend
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        fprintf(stderr, "%s: failed to init CPU backend\n", __func__);
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    return ctx;
}

// ── Frontend ─────────────────────────────────────────────────────────────────

// F32 conv1d via im2col (same helper as v1)
static struct ggml_tensor * conv_1d_f32(
        struct ggml_context * ctx0,
        struct ggml_tensor  * kernel,  // [K, IC, OC]
        struct ggml_tensor  * input,   // [IL, IC, N]
        int stride, int pad, int dil) {

    struct ggml_tensor * im2col = ggml_im2col(ctx0, kernel, input,
                                              stride, 0, pad, 0, dil, 0, false, GGML_TYPE_F32);

    struct ggml_tensor * result =
        ggml_mul_mat(ctx0,
                ggml_reshape_2d(ctx0, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
                ggml_reshape_2d(ctx0, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));

    result = ggml_reshape_3d(ctx0, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);

    return result;
}

// Optional struct to capture pre-conv concatenated tensors (for incremental buffer extraction)
struct frontend_intermediates {
    struct ggml_tensor * pre_conv1 = nullptr;  // [buf1+n_frames, hidden, 1]
    struct ggml_tensor * pre_conv2 = nullptr;  // [buf2+out1, hidden*2, 1]
};

// Build the streaming frontend graph:
//   raw audio → frames → CMVN → asinh → Linear+SiLU → CausalConv1+SiLU → CausalConv2
//
// Returns tensor of shape [enc_hidden, seq_len] (same layout as v1 encoder input)
// If intermediates is non-null, captures pre-conv concatenated tensors as graph outputs.
static struct ggml_tensor * build_streaming_frontend(
        struct ggml_context * ctx0,
        const moonshine_streaming_model & model,
        struct ggml_tensor * audio,       // [n_samples] F32
        struct ggml_tensor * conv1_pad,   // [left_pad1, hidden, 1] F32
        struct ggml_tensor * conv2_pad,   // [left_pad2, hidden*2, 1] F32
        struct ggml_tensor * one_const,   // [1] F32 — set to 1.0
        int n_samples,
        struct frontend_intermediates * intermediates = nullptr) {

    const auto & hp = model.hparams;
    const int frame_len  = hp.frame_len;      // 80
    const int hidden     = hp.enc_hidden_size; // 320
    const int num_frames = n_samples / frame_len;

    // 1. Reshape raw audio to frames: [n_samples] → [frame_len=80, num_frames]
    //    ggml ne[0]=80 (frame elements), ne[1]=num_frames
    struct ggml_tensor * frames = ggml_reshape_2d(ctx0, audio, frame_len, num_frames);

    // 2. CMVN: per-frame normalization along ne[0]
    //    ggml_norm computes (x - mean) / sqrt(var + eps) per row
    struct ggml_tensor * normed = ggml_norm(ctx0, frames, 1e-6f);

    // 3. Asinh compression: asinh(k * x) = log(k*x + sqrt((k*x)^2 + 1))
    //    k = exp(log_k) is precomputed in hparams
    struct ggml_tensor * kx = ggml_scale(ctx0, normed, hp.asinh_k);
    struct ggml_tensor * kx_sq = ggml_sqr(ctx0, kx);
    struct ggml_tensor * kx_sq_p1 = ggml_add(ctx0, kx_sq, one_const);  // broadcast scalar 1.0
    struct ggml_tensor * sqrt_val = ggml_sqrt(ctx0, kx_sq_p1);
    struct ggml_tensor * sum_val = ggml_add(ctx0, kx, sqrt_val);
    struct ggml_tensor * compressed = ggml_log(ctx0, sum_val);

    // 4. Linear(frame_len → hidden) + SiLU
    //    weight: ggml [frame_len=80, hidden=320]
    //    input:  [80, num_frames]
    //    output: [hidden=320, num_frames]
    struct ggml_tensor * projected = ggml_mul_mat(ctx0, model.frontend_linear_w, compressed);
    struct ggml_tensor * activated = ggml_silu(ctx0, projected);

    // 5. Transpose to [num_frames, hidden] for conv1d input format [time, channels]
    //    Then reshape to 3D: [num_frames, hidden, 1]
    struct ggml_tensor * cur = ggml_cont(ctx0, ggml_transpose(ctx0, activated));
    cur = ggml_reshape_3d(ctx0, cur, num_frames, hidden, 1);

    // 6. CausalConv1d #1: hidden → hidden*2, k=5, s=2
    //    Left-pad by (kernel-1) = 4 on time axis
    {
        // Concatenate pad/buffer + data along dim 0 (time)
        struct ggml_tensor * padded = ggml_concat(ctx0, conv1_pad, cur, 0);
        if (intermediates) {
            ggml_set_name(padded, "pre_conv1");
            ggml_set_output(padded);
            intermediates->pre_conv1 = padded;
        }
        // Conv1d with no padding
        cur = conv_1d_f32(ctx0, model.frontend_conv1_w, padded, hp.conv1_stride, 0, 1);
        // Add bias: bias [hidden*2] → reshape to [1, hidden*2, 1] for broadcast
        cur = ggml_add(ctx0, cur,
                       ggml_reshape_3d(ctx0, model.frontend_conv1_b, 1, model.frontend_conv1_b->ne[0], 1));
        // SiLU activation
        cur = ggml_silu(ctx0, cur);
    }

    // 7. CausalConv1d #2: hidden*2 → hidden, k=5, s=2
    {
        struct ggml_tensor * padded = ggml_concat(ctx0, conv2_pad, cur, 0);
        if (intermediates) {
            ggml_set_name(padded, "pre_conv2");
            ggml_set_output(padded);
            intermediates->pre_conv2 = padded;
        }
        cur = conv_1d_f32(ctx0, model.frontend_conv2_w, padded, hp.conv2_stride, 0, 1);
        cur = ggml_add(ctx0, cur,
                       ggml_reshape_3d(ctx0, model.frontend_conv2_b, 1, model.frontend_conv2_b->ne[0], 1));
        // No activation after conv2 (per Python reference)
    }

    // 8. Reshape to [seq_len, hidden] → transpose to [hidden, seq_len]
    const int64_t out_seq = cur->ne[0];
    cur = ggml_reshape_2d(ctx0, cur, out_seq, hidden);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    return cur;
}

static int moonshine_streaming_run_frontend(
        struct moonshine_streaming_context * ctx,
        const float * audio, int n_samples,
        std::vector<float> & features_out, int & seq_len_out) {

    const auto & hp = ctx->model.hparams;
    const int hidden = hp.enc_hidden_size;
    const int frame_len = hp.frame_len;

    // Pad audio to multiple of frame_len
    int padded_samples = n_samples;
    if (n_samples % frame_len != 0) {
        padded_samples = ((n_samples / frame_len) + 1) * frame_len;
    }

    // Compute padding sizes for causal convolutions
    const int left_pad1 = hp.conv1_kernel_size - 1;  // 4
    const int left_pad2 = hp.conv2_kernel_size - 1;  // 4

    // Estimate number of tensors needed for the graph
    const size_t n_tensors = 80;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    // Create input tensors
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, padded_samples);
    ggml_set_name(input, "audio_input");
    ggml_set_input(input);

    // Zero-padding tensors for causal convolutions
    struct ggml_tensor * conv1_pad = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, left_pad1, hidden, 1);
    ggml_set_name(conv1_pad, "conv1_pad");
    ggml_set_input(conv1_pad);

    struct ggml_tensor * conv2_pad = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, left_pad2, hidden * 2, 1);
    ggml_set_name(conv2_pad, "conv2_pad");
    ggml_set_input(conv2_pad);

    // Constant tensor: 1.0 for asinh computation
    struct ggml_tensor * one_const = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(one_const, "one_const");
    ggml_set_input(one_const);

    // Build frontend graph
    struct ggml_tensor * output = build_streaming_frontend(
        ctx0, ctx->model, input, conv1_pad, conv2_pad, one_const, padded_samples);
    ggml_set_name(output, "frontend_output");
    ggml_set_output(output);

    struct ggml_cgraph * graph = ggml_new_graph(ctx0);
    ggml_build_forward_expand(graph, output);

    // Allocate and compute
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(gallocr, graph)) {
        fprintf(stderr, "%s: failed to alloc graph\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    // Set input data
    // Audio: copy actual samples, zero-pad remainder
    {
        std::vector<float> padded_audio(padded_samples, 0.0f);
        memcpy(padded_audio.data(), audio, n_samples * sizeof(float));
        ggml_backend_tensor_set(input, padded_audio.data(), 0, padded_samples * sizeof(float));
    }

    // Zero padding for causal convs
    {
        std::vector<float> zeros1(left_pad1 * hidden, 0.0f);
        ggml_backend_tensor_set(conv1_pad, zeros1.data(), 0, zeros1.size() * sizeof(float));

        std::vector<float> zeros2(left_pad2 * hidden * 2, 0.0f);
        ggml_backend_tensor_set(conv2_pad, zeros2.data(), 0, zeros2.size() * sizeof(float));
    }

    // Constant 1.0
    {
        float one = 1.0f;
        ggml_backend_tensor_set(one_const, &one, 0, sizeof(float));
    }

    if (ggml_backend_graph_compute(ctx->backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    // Read output: [hidden, seq_len]
    const int out_hidden = (int)output->ne[0];
    const int out_seq    = (int)output->ne[1];
    const size_t out_bytes = out_hidden * out_seq * sizeof(float);

    features_out.resize(out_hidden * out_seq);
    seq_len_out = out_seq;
    ggml_backend_tensor_get(output, features_out.data(), 0, out_bytes);

    ggml_gallocr_free(gallocr);
    ggml_free(ctx0);
    return 0;
}

// Extract tail from a channel-major ggml buffer.
// ggml layout for [time, channels, 1]: data[c * time + t]
// Extracts the last tail_len time steps into out with layout [tail_len, channels, 1].
static void extract_conv_tail(
        const float * data, int total_time, int channels, int consumed,
        float * out, int tail_len) {
    for (int c = 0; c < channels; c++) {
        memcpy(out + c * tail_len,
               data + c * total_time + consumed,
               tail_len * sizeof(float));
    }
}

// Incremental frontend: processes only new audio frames, maintaining causal conv
// state across calls. Updates accumulated_features in-place.
// Returns number of new feature frames, or -1 on error.
static int moonshine_streaming_run_frontend_incremental(
        struct moonshine_streaming_context * ctx,
        const float * new_audio, int n_new_samples,
        struct moonshine_stream_state * state) {

    const auto & hp = ctx->model.hparams;
    const int hidden = hp.enc_hidden_size;
    const int frame_len = hp.frame_len;
    const int K1 = hp.conv1_kernel_size, S1 = hp.conv1_stride;
    const int K2 = hp.conv2_kernel_size, S2 = hp.conv2_stride;

    // ── 1. Sample buffering ──
    state->sample_buffer.insert(state->sample_buffer.end(),
                                 new_audio, new_audio + n_new_samples);

    int total_samples = (int)state->sample_buffer.size();
    int n_frames = total_samples / frame_len;

    if (n_frames == 0) return 0;

    // ── 2. Check conv pipeline would produce output ──
    int buf1_len = (int)(state->conv1_buffer.size() / hidden);
    int total1 = buf1_len + n_frames;
    int out1_len = (total1 >= K1) ? ((total1 - K1) / S1 + 1) : 0;

    if (out1_len == 0) return 0;  // not enough — buffer more audio

    int buf2_len = (int)(state->conv2_buffer.size() / (hidden * 2));
    int total2 = buf2_len + out1_len;
    int out2_len = (total2 >= K2) ? ((total2 - K2) / S2 + 1) : 0;

    if (out2_len == 0) return 0;  // not enough — buffer more audio

    // ── 3. Build graph ──
    int frame_samples = n_frames * frame_len;

    const size_t n_tensors = 80;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, frame_samples);
    ggml_set_name(input, "audio_input");
    ggml_set_input(input);

    struct ggml_tensor * conv1_buf_t = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32,
                                                           buf1_len, hidden, 1);
    ggml_set_name(conv1_buf_t, "conv1_buf");
    ggml_set_input(conv1_buf_t);

    struct ggml_tensor * conv2_buf_t = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32,
                                                           buf2_len, hidden * 2, 1);
    ggml_set_name(conv2_buf_t, "conv2_buf");
    ggml_set_input(conv2_buf_t);

    struct ggml_tensor * one_const = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(one_const, "one_const");
    ggml_set_input(one_const);

    frontend_intermediates intermediates;
    struct ggml_tensor * output = build_streaming_frontend(
        ctx0, ctx->model, input, conv1_buf_t, conv2_buf_t, one_const, frame_samples,
        &intermediates);
    ggml_set_name(output, "frontend_output");
    ggml_set_output(output);

    struct ggml_cgraph * graph = ggml_new_graph(ctx0);
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t gallocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(gallocr, graph)) {
        fprintf(stderr, "%s: failed to alloc graph\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    // ── 4. Set input data ──
    ggml_backend_tensor_set(input, state->sample_buffer.data(), 0,
                             frame_samples * sizeof(float));
    ggml_backend_tensor_set(conv1_buf_t, state->conv1_buffer.data(), 0,
                             state->conv1_buffer.size() * sizeof(float));
    ggml_backend_tensor_set(conv2_buf_t, state->conv2_buffer.data(), 0,
                             state->conv2_buffer.size() * sizeof(float));
    {
        float one = 1.0f;
        ggml_backend_tensor_set(one_const, &one, 0, sizeof(float));
    }

    // ── 5. Compute ──
    if (ggml_backend_graph_compute(ctx->backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    // ── 6. Read output features and append to accumulated_features ──
    const int out_hidden = (int)output->ne[0];
    const int out_seq    = (int)output->ne[1];

    int old_count = state->accumulated_feature_count;
    state->accumulated_features.resize((old_count + out_seq) * out_hidden);
    ggml_backend_tensor_get(output,
                             state->accumulated_features.data() + old_count * out_hidden,
                             0, out_hidden * out_seq * sizeof(float));
    state->accumulated_feature_count = old_count + out_seq;

    // ── 7. Update conv buffers from pre-conv intermediates ──

    // Conv1: extract tail of pre_conv1 [total1, hidden, 1]
    {
        int consumed1 = S1 * out1_len;
        int new_buf1_len = total1 - consumed1;
        std::vector<float> pre_data(total1 * hidden);
        ggml_backend_tensor_get(intermediates.pre_conv1, pre_data.data(), 0,
                                 pre_data.size() * sizeof(float));
        state->conv1_buffer.resize(new_buf1_len * hidden);
        extract_conv_tail(pre_data.data(), total1, hidden, consumed1,
                          state->conv1_buffer.data(), new_buf1_len);
    }

    // Conv2: extract tail of pre_conv2 [total2, hidden*2, 1]
    {
        int consumed2 = S2 * out2_len;
        int new_buf2_len = total2 - consumed2;
        std::vector<float> pre_data(total2 * hidden * 2);
        ggml_backend_tensor_get(intermediates.pre_conv2, pre_data.data(), 0,
                                 pre_data.size() * sizeof(float));
        state->conv2_buffer.resize(new_buf2_len * hidden * 2);
        extract_conv_tail(pre_data.data(), total2, hidden * 2, consumed2,
                          state->conv2_buffer.data(), new_buf2_len);
    }

    // ── 8. Consume processed samples from sample buffer ──
    state->sample_buffer.erase(state->sample_buffer.begin(),
                                state->sample_buffer.begin() + frame_samples);

    ggml_gallocr_free(gallocr);
    ggml_free(ctx0);
    return out_seq;
}

// ── Encoder ──────────────────────────────────────────────────────────────────

// Unit-offset LayerNorm: LN(x) * (gamma + 1.0)
// Standard LN normalizes, then scales by gamma. Unit-offset adds 1 to gamma,
// which biases the scaling toward identity (helpful for training stability).
static struct ggml_tensor * unit_offset_norm(
        struct ggml_context * ctx0,
        struct ggml_tensor * x,
        struct ggml_tensor * gamma,
        struct ggml_tensor * ones,   // [hidden] tensor of 1.0s
        float eps) {
    struct ggml_tensor * normed = ggml_norm(ctx0, x, eps);
    struct ggml_tensor * scale = ggml_add(ctx0, gamma, ones);
    return ggml_mul(ctx0, normed, scale);
}

// Build sliding-window attention mask for one encoder layer.
// Returns a 2D mask [seq_len, seq_len] where allowed positions are 0.0
// and blocked positions are -INFINITY.
// For position q attending to position k:
//   allow if (q - k) >= 0 && (q - k) <= left_window   (k is to the left of q)
//   allow if (k - q) > 0  && (k - q) <= right_window   (k is to the right of q)
static struct ggml_tensor * build_sliding_window_mask(
        struct ggml_context * ctx0,
        int seq_len) {
    // flash_attn_ext expects mask in FP16 format
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, seq_len, seq_len);
    ggml_set_name(mask, "attn_mask");
    ggml_set_input(mask);  // will be filled before compute
    return mask;
}

// Fill a sliding-window mask buffer in FP16 for flash_attn_ext.
// Mask layout: [n_kv, n_batch] = [seq_len, seq_len] in ggml column-major.
// For query position q and key position k: mask_data[q * seq_len + k]
// is added to the attention score. 0 = allowed, -inf = blocked.
static void fill_sliding_window_mask(
        ggml_fp16_t * mask_data, int seq_len, int left_window, int right_window) {
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < seq_len; q++) {
        for (int k = 0; k < seq_len; k++) {
            int diff = q - k;  // positive means k is to the left
            bool allowed = false;
            if (diff >= 0 && diff <= left_window) allowed = true;   // k at or left of q
            if (diff < 0 && (-diff) <= right_window) allowed = true; // k to the right of q
            mask_data[q * seq_len + k] = allowed ? zero : neg_inf;
        }
    }
}

// Build the streaming encoder graph:
//   features [hidden, seq_len] → N transformer layers → output [hidden, seq_len]
//
// Each layer: unit-offset-LN → sliding-window attention → residual →
//             unit-offset-LN → FFN (fc1+GELU+fc2) → residual
// Final: unit-offset-LN
//
// Key differences from v1:
//   - No RoPE (ergodic — no positional encoding)
//   - Sliding-window bidirectional attention with per-layer (left, right) windows
//   - Unit-offset LayerNorm: LN(x) * (gamma + 1.0)
//   - No bias on attention projections
static struct ggml_tensor * build_streaming_encoder(
        struct ggml_context * ctx0,
        const moonshine_streaming_model & model,
        struct ggml_tensor * features,     // [hidden, seq_len]
        struct ggml_tensor * ones_hidden,  // [hidden] filled with 1.0 for unit-offset
        struct ggml_tensor ** layer_masks, // [n_layers] mask tensors, each [seq_len, seq_len]
        int seq_len) {

    const auto & hp = model.hparams;
    const int n_heads    = hp.enc_n_heads;
    const int n_kv_heads = hp.enc_n_kv_heads;
    const int head_dim   = hp.enc_head_dim;
    const float eps = 1e-5f;

    struct ggml_tensor * cur = features;

    for (uint32_t il = 0; il < hp.enc_n_layers; il++) {
        const auto & layer = model.enc_layers[il];

        struct ggml_tensor * residual = cur;

        // Pre-norm for attention (unit-offset LayerNorm)
        cur = unit_offset_norm(ctx0, cur, layer.attn_norm, ones_hidden, eps);

        // QKV projections: [hidden, seq_len] → [head_dim*n_heads, seq_len]
        // No bias on attention projections
        struct ggml_tensor * Q = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * K = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * V = ggml_mul_mat(ctx0, layer.attn_v, cur);

        // Reshape to multi-head: [head_dim, n_heads, seq_len]
        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, seq_len);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_kv_heads, seq_len);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_kv_heads, seq_len);

        // No RoPE — encoder is ergodic (no positional encoding)

        // Permute for flash attention:
        //   [head_dim, n_heads, seq_len] → [head_dim, seq_len, n_heads]
        Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);

        // Sliding-window attention with mask
        float scale = 1.0f / sqrtf((float)head_dim);
        struct ggml_tensor * attn = ggml_flash_attn_ext(ctx0, Q, K, V,
                                                         layer_masks[il], scale, 0.0f, 0.0f);

        // Result is [head_dim, n_heads, seq_len] — reshape to [hidden, seq_len]
        attn = ggml_reshape_2d(ctx0, attn, head_dim * n_heads, seq_len);

        // Output projection (no bias)
        cur = ggml_mul_mat(ctx0, layer.attn_o, attn);

        // Residual connection
        cur = ggml_add(ctx0, cur, residual);

        residual = cur;

        // Pre-norm for FFN (unit-offset LayerNorm)
        cur = unit_offset_norm(ctx0, cur, layer.ffn_norm, ones_hidden, eps);

        // FFN: fc1 + bias + GELU + fc2 + bias
        cur = ggml_mul_mat(ctx0, layer.ffn_fc1_w, cur);
        cur = ggml_add(ctx0, cur, layer.ffn_fc1_b);
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_mul_mat(ctx0, layer.ffn_fc2_w, cur);
        cur = ggml_add(ctx0, cur, layer.ffn_fc2_b);

        // Residual connection
        cur = ggml_add(ctx0, cur, residual);
    }

    // Final encoder output norm (unit-offset LayerNorm)
    cur = unit_offset_norm(ctx0, cur, model.enc_output_norm, ones_hidden, eps);

    return cur;
}

static int moonshine_streaming_run_encoder(
        struct moonshine_streaming_context * ctx,
        const float * features_in, int seq_len, int hidden_dim,
        std::vector<float> & encoded_out,
        bool use_sliding_window = false) {

    const auto & hp = ctx->model.hparams;
    const int hidden = hp.enc_hidden_size;
    const int n_layers = hp.enc_n_layers;

    if (hidden_dim != (int)hidden) {
        fprintf(stderr, "%s: hidden_dim mismatch: got %d, expected %u\n",
                __func__, hidden_dim, hidden);
        return -1;
    }

    // Estimate tensor count: encoder layers need ~15 tensors each + overhead
    const size_t n_tensors = 20 * n_layers + 30;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    // Input features tensor: [hidden, seq_len]
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, seq_len);
    ggml_set_name(input, "encoder_input");
    ggml_set_input(input);

    // Ones tensor for unit-offset LayerNorm: [hidden]
    struct ggml_tensor * ones_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden);
    ggml_set_name(ones_hidden, "ones_hidden");
    ggml_set_input(ones_hidden);

    // Per-layer sliding-window masks
    // In batch mode (no padding), the Python reference uses full attention (no masks).
    // Sliding-window masks are only applied when there's a padding attention_mask,
    // which happens during streaming inference (Step 6). For batch mode, nullptr = full attention.
    std::vector<struct ggml_tensor *> layer_masks(n_layers, nullptr);

    // When use_sliding_window is true (streaming mode), build per-layer masks
    if (use_sliding_window) {
        for (int il = 0; il < n_layers; il++) {
            int left_window = 16, right_window = 4;  // defaults
            if ((size_t)(il * 2 + 1) < hp.sliding_windows.size()) {
                left_window  = hp.sliding_windows[il * 2];
                right_window = hp.sliding_windows[il * 2 + 1];
            }

            // If window covers entire sequence, skip mask (nullptr = full attention)
            if (left_window >= seq_len && right_window >= seq_len) {
                continue;
            }

            layer_masks[il] = build_sliding_window_mask(ctx0, seq_len);
            char name[64];
            snprintf(name, sizeof(name), "mask_layer_%d", il);
            ggml_set_name(layer_masks[il], name);
        }
    }

    // Build encoder graph
    struct ggml_tensor * output = build_streaming_encoder(
        ctx0, ctx->model, input, ones_hidden, layer_masks.data(), seq_len);
    ggml_set_name(output, "encoder_output");
    ggml_set_output(output);

    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx0, n_tensors * 2, false);
    ggml_build_forward_expand(graph, output);

    // Allocate and compute
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(gallocr, graph)) {
        fprintf(stderr, "%s: failed to alloc graph\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    // Set input data
    ggml_backend_tensor_set(input, features_in, 0, hidden * seq_len * sizeof(float));

    // Fill ones tensor
    {
        std::vector<float> ones(hidden, 1.0f);
        ggml_backend_tensor_set(ones_hidden, ones.data(), 0, hidden * sizeof(float));
    }

    // Fill sliding-window masks
    for (int il = 0; il < n_layers; il++) {
        if (layer_masks[il] == nullptr) continue;

        int left_window = 16, right_window = 4;
        if ((size_t)(il * 2 + 1) < hp.sliding_windows.size()) {
            left_window  = hp.sliding_windows[il * 2];
            right_window = hp.sliding_windows[il * 2 + 1];
        }

        std::vector<ggml_fp16_t> mask_data(seq_len * seq_len);
        fill_sliding_window_mask(mask_data.data(), seq_len, left_window, right_window);
        ggml_backend_tensor_set(layer_masks[il], mask_data.data(), 0,
                                seq_len * seq_len * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_graph_compute(ctx->backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    // Read output: [hidden, seq_len]
    const size_t out_bytes = hidden * seq_len * sizeof(float);
    encoded_out.resize(hidden * seq_len);
    ggml_backend_tensor_get(output, encoded_out.data(), 0, out_bytes);

    ggml_gallocr_free(gallocr);
    ggml_free(ctx0);
    return 0;
}

// ── Public API ───────────────────────────────────────────────────────────────

int moonshine_streaming_frontend(
        struct moonshine_streaming_context * ctx,
        const float * audio, int n_samples,
        float ** out_features, int * out_seq_len, int * out_hidden_dim) {
    if (!ctx || !audio || n_samples <= 0) return -1;

    std::vector<float> features;
    int seq_len = 0;

    int ret = moonshine_streaming_run_frontend(ctx, audio, n_samples, features, seq_len);
    if (ret != 0) return ret;

    const int hidden_dim = (int)ctx->model.hparams.enc_hidden_size;
    const size_t out_bytes = hidden_dim * seq_len * sizeof(float);

    float * buf = (float *)malloc(out_bytes);
    if (!buf) return -1;

    memcpy(buf, features.data(), out_bytes);

    *out_features   = buf;
    *out_seq_len    = seq_len;
    *out_hidden_dim = hidden_dim;
    return 0;
}

int moonshine_streaming_encoder(
        struct moonshine_streaming_context * ctx,
        const float * features, int seq_len, int hidden_dim,
        float ** out_encoded, int * out_seq_len, int * out_hidden_dim) {
    if (!ctx || !features || seq_len <= 0) return -1;

    std::vector<float> encoded;

    int ret = moonshine_streaming_run_encoder(ctx, features, seq_len, hidden_dim, encoded);
    if (ret != 0) return ret;

    const int hidden = (int)ctx->model.hparams.enc_hidden_size;
    const size_t out_bytes = hidden * seq_len * sizeof(float);

    float * buf = (float *)malloc(out_bytes);
    if (!buf) return -1;

    memcpy(buf, encoded.data(), out_bytes);

    *out_encoded    = buf;
    *out_seq_len    = seq_len;
    *out_hidden_dim = hidden;
    return 0;
}

// ── KV cache (same as v1, duplicated since it's static there) ────────────────

static bool moonshine_kv_cache_init(moonshine_kv_cache & cache, int n_layers, int max_len,
                                     int n_kv_heads, int head_dim) {
    cache.max_len = max_len;
    cache.n = 0;
    cache.k.resize(n_layers);
    cache.v.resize(n_layers);

    const size_t n_tensors = 2 * n_layers;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + 256;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    cache.ctx = ggml_init(params);
    if (!cache.ctx) return false;

    for (int i = 0; i < n_layers; i++) {
        cache.k[i] = ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, head_dim, max_len, n_kv_heads);
        cache.v[i] = ggml_new_tensor_3d(cache.ctx, GGML_TYPE_F32, head_dim, max_len, n_kv_heads);
    }

    cache.buf = ggml_backend_alloc_ctx_tensors_from_buft(cache.ctx, ggml_backend_cpu_buffer_type());
    if (!cache.buf) return false;

    ggml_backend_buffer_clear(cache.buf, 0);
    return true;
}

// ── Adapter ───────────────────────────────────────────────────────────────────

// Run adapter: encoder output → add positional embeddings → optional projection
// Input: [enc_hidden, seq_len]. Output: [dec_hidden, seq_len] (same if dims match).
static int moonshine_streaming_run_adapter(
        struct moonshine_streaming_context * ctx,
        const float * encoded_in, int seq_len,
        std::vector<float> & memory_out,
        int pos_offset = 0) {

    const auto & model = ctx->model;
    const auto & hp = model.hparams;
    const int enc_hidden = hp.enc_hidden_size;
    const int dec_hidden = hp.dec_hidden_size;
    const bool needs_proj = (model.adapter_proj != nullptr);

    const size_t n_tensors = 20;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    // Input: encoder output [enc_hidden, seq_len]
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, enc_hidden, seq_len);
    ggml_set_name(input, "adapter_input");
    ggml_set_input(input);

    // Position indices [0, 1, ..., seq_len-1] for ggml_get_rows
    struct ggml_tensor * pos_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // Extract positional embeddings: adapter_pos_emb[0:seq_len]
    // adapter_pos_emb is [enc_hidden, max_pos] in ggml (ne[0]=enc_hidden, ne[1]=max_pos)
    // ggml_get_rows selects rows by index → [enc_hidden, seq_len]
    struct ggml_tensor * pos_emb = ggml_get_rows(ctx0, model.adapter_pos_emb, pos_ids);

    // Add positional embeddings to encoder output
    struct ggml_tensor * cur = ggml_add(ctx0, input, pos_emb);

    // Optional projection: enc_hidden → dec_hidden
    if (needs_proj) {
        cur = ggml_mul_mat(ctx0, model.adapter_proj, cur);
    }

    ggml_set_name(cur, "adapter_output");
    ggml_set_output(cur);

    struct ggml_cgraph * graph = ggml_new_graph(ctx0);
    ggml_build_forward_expand(graph, cur);

    // Allocate and compute
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(gallocr, graph)) {
        fprintf(stderr, "%s: failed to alloc graph\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    // Set input data
    ggml_backend_tensor_set(input, encoded_in, 0, enc_hidden * seq_len * sizeof(float));

    // Fill position indices (with offset for incremental streaming)
    {
        std::vector<int32_t> ids(seq_len);
        for (int i = 0; i < seq_len; i++) ids[i] = i + pos_offset;
        ggml_backend_tensor_set(pos_ids, ids.data(), 0, seq_len * sizeof(int32_t));
    }

    if (ggml_backend_graph_compute(ctx->backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    // Read output
    const int out_hidden = needs_proj ? dec_hidden : enc_hidden;
    const size_t out_bytes = out_hidden * seq_len * sizeof(float);
    memory_out.resize(out_hidden * seq_len);
    ggml_backend_tensor_get(cur, memory_out.data(), 0, out_bytes);

    ggml_gallocr_free(gallocr);
    ggml_free(ctx0);
    return 0;
}

// ── Cross-attention KV precomputation ────────────────────────────────────────

static int moonshine_streaming_precompute_cross_kv(struct moonshine_streaming_context * ctx,
                                                     const float * memory, int memory_len) {
    const auto & model = ctx->model;
    const auto & hp = model.hparams;
    const int n_layers   = hp.dec_n_layers;
    const int dec_hidden = hp.dec_hidden_size;
    const int n_kv_heads = hp.dec_n_kv_heads;
    const int head_dim   = hp.dec_head_dim;

    const size_t n_tensors = n_layers * 10 + 10;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) return -1;

    // Input: adapter output (memory) [dec_hidden, memory_len]
    struct ggml_tensor * mem_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dec_hidden, memory_len);
    ggml_set_name(mem_in, "cross_mem_in");
    ggml_set_input(mem_in);

    struct ggml_cgraph * graph = ggml_new_graph(ctx0);

    std::vector<struct ggml_tensor *> k_outputs(n_layers);
    std::vector<struct ggml_tensor *> v_outputs(n_layers);

    for (int i = 0; i < n_layers; i++) {
        const auto & layer = model.dec_layers[i];

        // K = cross_attn_k * mem_in → [n_kv_heads*head_dim, memory_len]
        struct ggml_tensor * K = ggml_mul_mat(ctx0, layer.cross_attn_k, mem_in);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_kv_heads, memory_len);
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        char name_k[64];
        snprintf(name_k, sizeof(name_k), "cross_k_%d", i);
        ggml_set_name(K, name_k);
        ggml_set_output(K);
        k_outputs[i] = K;
        ggml_build_forward_expand(graph, K);

        // V = cross_attn_v * mem_in → [n_kv_heads*head_dim, memory_len]
        struct ggml_tensor * V = ggml_mul_mat(ctx0, layer.cross_attn_v, mem_in);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_kv_heads, memory_len);
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
        char name_v[64];
        snprintf(name_v, sizeof(name_v), "cross_v_%d", i);
        ggml_set_name(V, name_v);
        ggml_set_output(V);
        v_outputs[i] = V;
        ggml_build_forward_expand(graph, V);
    }

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(gallocr, graph)) {
        fprintf(stderr, "%s: failed to alloc graph\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_set(mem_in, memory, 0, dec_hidden * memory_len * sizeof(float));

    if (ggml_backend_graph_compute(ctx->backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed\n", __func__);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    // Copy results into cross KV cache
    const size_t kv_bytes = head_dim * memory_len * n_kv_heads * sizeof(float);
    for (int i = 0; i < n_layers; i++) {
        ggml_backend_tensor_get(k_outputs[i], ctx->kv_cross.k[i]->data, 0, kv_bytes);
        ggml_backend_tensor_get(v_outputs[i], ctx->kv_cross.v[i]->data, 0, kv_bytes);
    }
    ctx->kv_cross.n = memory_len;

    ggml_gallocr_free(gallocr);
    ggml_free(ctx0);
    return 0;
}

// ── Decoder ──────────────────────────────────────────────────────────────────

// Build a single decoder step graph (one token)
static struct ggml_tensor * streaming_build_decoder_step(
        struct ggml_context * ctx0,
        const moonshine_streaming_model & model,
        moonshine_kv_cache & kv_self,
        moonshine_kv_cache & kv_cross,
        struct ggml_tensor * token_id,   // [1] I32
        struct ggml_tensor * dec_pos,    // [1] I32
        int memory_len,
        int cur_pos,
        struct ggml_cgraph * graph) {

    const auto & hp = model.hparams;
    const int n_heads      = hp.dec_n_heads;
    const int n_kv_heads   = hp.dec_n_kv_heads;
    const int head_dim     = hp.dec_head_dim;
    const int rotary_dim   = hp.enc_rotary_dim;  // dec_head_dim * partial_rotary_factor
    const int intermediate = hp.dec_intermediate;
    const float eps        = 1e-5f;
    const float rope_theta = hp.rope_theta;
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Token embedding: [dec_hidden, 1]
    struct ggml_tensor * cur = ggml_get_rows(ctx0, model.dec_embed, token_id);

    for (uint32_t il = 0; il < hp.dec_n_layers; il++) {
        const auto & layer = model.dec_layers[il];

        // === Self-attention ===
        struct ggml_tensor * residual = cur;

        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        struct ggml_tensor * Q = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * K_new = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * V_new = ggml_mul_mat(ctx0, layer.attn_v, cur);

        Q     = ggml_reshape_3d(ctx0, Q,     head_dim, n_heads,    1);
        K_new = ggml_reshape_3d(ctx0, K_new, head_dim, n_kv_heads, 1);
        V_new = ggml_reshape_3d(ctx0, V_new, head_dim, n_kv_heads, 1);

        // Partial RoPE on Q and K
        Q     = ggml_rope_ext(ctx0, Q,     dec_pos, nullptr, rotary_dim, 0, 0,
                              rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K_new = ggml_rope_ext(ctx0, K_new, dec_pos, nullptr, rotary_dim, 0, 0,
                              rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute to [head_dim, 1, n_heads] for cache layout
        K_new = ggml_permute(ctx0, K_new, 0, 2, 1, 3);
        V_new = ggml_permute(ctx0, V_new, 0, 2, 1, 3);

        // Write K_new, V_new into self-attn cache at cur_pos
        struct ggml_tensor * k_cache_slice = ggml_view_3d(ctx0, kv_self.k[il],
            head_dim, 1, n_kv_heads,
            kv_self.k[il]->nb[1], kv_self.k[il]->nb[2],
            cur_pos * kv_self.k[il]->nb[1]);
        struct ggml_tensor * v_cache_slice = ggml_view_3d(ctx0, kv_self.v[il],
            head_dim, 1, n_kv_heads,
            kv_self.v[il]->nb[1], kv_self.v[il]->nb[2],
            cur_pos * kv_self.v[il]->nb[1]);

        ggml_build_forward_expand(graph, ggml_cpy(ctx0, K_new, k_cache_slice));
        ggml_build_forward_expand(graph, ggml_cpy(ctx0, V_new, v_cache_slice));

        // Read filled portion of cache [0..cur_pos+1]
        int kv_len = cur_pos + 1;
        struct ggml_tensor * K_cached = ggml_view_3d(ctx0, kv_self.k[il],
            head_dim, kv_len, n_kv_heads,
            kv_self.k[il]->nb[1], kv_self.k[il]->nb[2], 0);
        struct ggml_tensor * V_cached = ggml_view_3d(ctx0, kv_self.v[il],
            head_dim, kv_len, n_kv_heads,
            kv_self.v[il]->nb[1], kv_self.v[il]->nb[2], 0);

        // Permute Q for flash_attn: [head_dim, n_heads, 1] → [head_dim, 1, n_heads]
        Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);

        struct ggml_tensor * attn = ggml_flash_attn_ext(ctx0, Q, K_cached, V_cached,
                                                        nullptr, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, n_heads * head_dim, 1);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_o, attn), residual);

        // === Cross-attention ===
        residual = cur;

        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.cross_attn_norm);

        struct ggml_tensor * Q_cross = ggml_mul_mat(ctx0, layer.cross_attn_q, cur);
        Q_cross = ggml_reshape_3d(ctx0, Q_cross, head_dim, n_heads, 1);
        // No RoPE for cross-attention
        Q_cross = ggml_permute(ctx0, Q_cross, 0, 2, 1, 3);

        // Read cross KV from precomputed cache: [head_dim, memory_len, n_kv_heads]
        struct ggml_tensor * K_cross = ggml_view_3d(ctx0, kv_cross.k[il],
            head_dim, memory_len, n_kv_heads,
            kv_cross.k[il]->nb[1], kv_cross.k[il]->nb[2], 0);
        struct ggml_tensor * V_cross = ggml_view_3d(ctx0, kv_cross.v[il],
            head_dim, memory_len, n_kv_heads,
            kv_cross.v[il]->nb[1], kv_cross.v[il]->nb[2], 0);

        struct ggml_tensor * cross_attn = ggml_flash_attn_ext(ctx0, Q_cross, K_cross, V_cross,
                                                               nullptr, scale, 0.0f, 0.0f);
        cross_attn = ggml_reshape_2d(ctx0, cross_attn, n_heads * head_dim, 1);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.cross_attn_o, cross_attn), residual);

        // === Gated SiLU FFN ===
        residual = cur;

        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor * fc1_out = ggml_mul_mat(ctx0, layer.ffn_fc1_w, cur);
        fc1_out = ggml_add(ctx0, fc1_out, layer.ffn_fc1_b);

        // Split: first half = value, second half = gate
        struct ggml_tensor * value_half = ggml_view_2d(ctx0, fc1_out,
            intermediate, 1, fc1_out->nb[1], 0);
        struct ggml_tensor * gate_half = ggml_view_2d(ctx0, fc1_out,
            intermediate, 1, fc1_out->nb[1], intermediate * sizeof(float));

        gate_half = ggml_silu(ctx0, gate_half);
        cur = ggml_mul(ctx0, gate_half, value_half);

        cur = ggml_mul_mat(ctx0, layer.ffn_fc2_w, cur);
        cur = ggml_add(ctx0, cur, layer.ffn_fc2_b);
        cur = ggml_add(ctx0, cur, residual);
    }

    // Final output norm
    cur = ggml_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model.dec_output_norm);

    // Project to vocab: [vocab_size, 1]
    struct ggml_tensor * logits = ggml_mul_mat(ctx0, model.dec_output, cur);

    return logits;
}

// Execute a single decoder step
static int streaming_decode_step(struct moonshine_streaming_context * ctx, int32_t token_id,
                                  std::vector<float> & logits_out,
                                  ggml_gallocr_t gallocr) {
    const auto & hp = ctx->model.hparams;
    const int cur_pos = ctx->kv_self.n;

    const size_t n_tensors = hp.dec_n_layers * 60 + 50;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    struct ggml_tensor * inp_token = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_token, "token_id");
    ggml_set_input(inp_token);

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_pos, "dec_pos");
    ggml_set_input(inp_pos);

    struct ggml_cgraph * graph = ggml_new_graph(ctx0);

    struct ggml_tensor * logits = streaming_build_decoder_step(
        ctx0, ctx->model, ctx->kv_self, ctx->kv_cross,
        inp_token, inp_pos, ctx->kv_cross.n, cur_pos, graph);

    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(graph, logits);

    if (!ggml_gallocr_alloc_graph(gallocr, graph)) {
        fprintf(stderr, "%s: failed to alloc graph\n", __func__);
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_set(inp_token, &token_id, 0, sizeof(int32_t));
    int32_t pos_val = cur_pos;
    ggml_backend_tensor_set(inp_pos, &pos_val, 0, sizeof(int32_t));

    if (ggml_backend_graph_compute(ctx->backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed\n", __func__);
        ggml_free(ctx0);
        return -1;
    }

    logits_out.resize(hp.vocab_size);
    ggml_backend_tensor_get(logits, logits_out.data(), 0, hp.vocab_size * sizeof(float));

    ctx->kv_self.n++;

    ggml_free(ctx0);
    return 0;
}

// ── Transcribe ───────────────────────────────────────────────────────────────

const char * moonshine_streaming_transcribe(struct moonshine_streaming_context * ctx,
                                             const float * audio, int n_samples) {
    if (!ctx || !audio || n_samples <= 0) {
        return "";
    }

    const auto & hp = ctx->model.hparams;

    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Run frontend
    std::vector<float> features;
    int feat_seq_len = 0;
    int ret = moonshine_streaming_run_frontend(ctx, audio, n_samples, features, feat_seq_len);
    if (ret != 0) {
        fprintf(stderr, "%s: frontend failed\n", __func__);
        return "";
    }

    // 2. Run encoder (use sliding-window masks — model was trained with them)
    std::vector<float> encoded;
    ret = moonshine_streaming_run_encoder(ctx, features.data(), feat_seq_len, hp.enc_hidden_size,
                                          encoded, /*use_sliding_window=*/true);
    if (ret != 0) {
        fprintf(stderr, "%s: encoder failed\n", __func__);
        return "";
    }

    // 3. Run adapter
    std::vector<float> memory;
    ret = moonshine_streaming_run_adapter(ctx, encoded.data(), feat_seq_len, memory);
    if (ret != 0) {
        fprintf(stderr, "%s: adapter failed\n", __func__);
        return "";
    }

    const int memory_len = feat_seq_len;

    auto t_encode_done = std::chrono::high_resolution_clock::now();

    // 4. Init KV caches
    int max_gen = (int)(ceil((double)n_samples / 16000.0 * 6.5));
    if (max_gen > 194) max_gen = 194;
    int max_len = max_gen + 1;  // +1 for BOS

    if (!moonshine_kv_cache_init(ctx->kv_self, hp.dec_n_layers, max_len,
                                  hp.dec_n_kv_heads, hp.dec_head_dim)) {
        fprintf(stderr, "%s: failed to init self KV cache\n", __func__);
        return "";
    }

    if (!moonshine_kv_cache_init(ctx->kv_cross, hp.dec_n_layers, memory_len,
                                  hp.dec_n_kv_heads, hp.dec_head_dim)) {
        fprintf(stderr, "%s: failed to init cross KV cache\n", __func__);
        ctx->kv_self.reset();
        return "";
    }

    // 5. Precompute cross-attention KV
    ret = moonshine_streaming_precompute_cross_kv(ctx, memory.data(), memory_len);
    if (ret != 0) {
        fprintf(stderr, "%s: cross-KV precompute failed\n", __func__);
        ctx->kv_self.reset();
        ctx->kv_cross.reset();
        return "";
    }

    // Free memory buffer — no longer needed after cross-KV precompute
    { std::vector<float>().swap(memory); }

    // 6. Pre-allocate decoder compute buffer with max-size graph
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    {
        const size_t n_tensors = hp.dec_n_layers * 60 + 50;
        const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
        struct ggml_init_params gp = { mem_size, nullptr, true };
        struct ggml_context * plan_ctx = ggml_init(gp);
        struct ggml_tensor * t = ggml_new_tensor_1d(plan_ctx, GGML_TYPE_I32, 1);
        ggml_set_name(t, "token_id"); ggml_set_input(t);
        struct ggml_tensor * p = ggml_new_tensor_1d(plan_ctx, GGML_TYPE_I32, 1);
        ggml_set_name(p, "dec_pos"); ggml_set_input(p);
        struct ggml_cgraph * plan_graph = ggml_new_graph(plan_ctx);
        struct ggml_tensor * plan_logits = streaming_build_decoder_step(
            plan_ctx, ctx->model, ctx->kv_self, ctx->kv_cross,
            t, p, memory_len, max_len - 1, plan_graph);
        ggml_set_output(plan_logits);
        ggml_build_forward_expand(plan_graph, plan_logits);
        ggml_gallocr_reserve(gallocr, plan_graph);
        ggml_free(plan_ctx);
    }

    // 7. Greedy decode loop
    std::vector<int32_t> tokens;
    int32_t token = (int32_t)hp.bos_token_id;
    std::vector<float> logits(hp.vocab_size);

    for (int step = 0; step < max_len; step++) {
        ret = streaming_decode_step(ctx, token, logits, gallocr);
        if (ret != 0) {
            break;
        }

        // Argmax
        int32_t best = 0;
        float best_val = logits[0];
        for (int i = 1; i < (int)hp.vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }

        if (best == (int32_t)hp.eos_token_id) break;

        tokens.push_back(best);
        token = best;
    }

    auto t_decode_done = std::chrono::high_resolution_clock::now();

    double encode_ms = std::chrono::duration<double, std::milli>(t_encode_done - t_start).count();
    double decode_ms = std::chrono::duration<double, std::milli>(t_decode_done - t_encode_done).count();
    double total_ms  = std::chrono::duration<double, std::milli>(t_decode_done - t_start).count();

    fprintf(stderr, "streaming: encode=%.1fms decode=%.1fms total=%.1fms tokens=%d\n",
            encode_ms, decode_ms, total_ms, (int)tokens.size());

    // 8. Convert to text
    ctx->result_text = ctx->tokenizer.tokens_to_text(tokens);

    // 9. Cleanup
    ggml_gallocr_free(gallocr);
    ctx->kv_self.reset();
    ctx->kv_cross.reset();

    return ctx->result_text.c_str();
}

void moonshine_streaming_free(struct moonshine_streaming_context * ctx) {
    if (!ctx) return;
    ggml_backend_free(ctx->backend);
    delete ctx;
}

void moonshine_streaming_print_model_info(struct moonshine_streaming_context * ctx) {
    if (!ctx) return;

    const auto & hp = ctx->model.hparams;

    printf("=== Moonshine Streaming Model Info ===\n");
    printf("Encoder:\n");
    printf("  hidden_size:      %u\n", hp.enc_hidden_size);
    printf("  n_layers:         %u\n", hp.enc_n_layers);
    printf("  n_heads:          %u\n", hp.enc_n_heads);
    printf("  n_kv_heads:       %u\n", hp.enc_n_kv_heads);
    printf("  head_dim:         %u\n", hp.enc_head_dim);
    printf("  intermediate:     %u\n", hp.enc_intermediate);
    printf("Decoder:\n");
    printf("  hidden_size:      %u\n", hp.dec_hidden_size);
    printf("  n_layers:         %u\n", hp.dec_n_layers);
    printf("  n_heads:          %u\n", hp.dec_n_heads);
    printf("  n_kv_heads:       %u\n", hp.dec_n_kv_heads);
    printf("  head_dim:         %u\n", hp.dec_head_dim);
    printf("  intermediate:     %u\n", hp.dec_intermediate);
    printf("Frontend:\n");
    printf("  frame_len:        %u\n", hp.frame_len);
    printf("  conv1:            k=%u s=%u\n", hp.conv1_kernel_size, hp.conv1_stride);
    printf("  conv2:            k=%u s=%u\n", hp.conv2_kernel_size, hp.conv2_stride);
    printf("  asinh_k:          %g\n", hp.asinh_k);
    printf("Shared:\n");
    printf("  vocab_size:       %u\n", hp.vocab_size);
    printf("  rope_theta:       %g\n", hp.rope_theta);
    printf("  partial_rotary:   %g\n", hp.partial_rotary_factor);
    printf("  max_pos_emb:      %u\n", hp.max_position_embeddings);
    printf("Sliding windows:    ");
    for (size_t i = 0; i + 1 < hp.sliding_windows.size(); i += 2) {
        printf("(%d,%d) ", hp.sliding_windows[i], hp.sliding_windows[i+1]);
    }
    printf("\n");

    // Tensor table
    printf("\nTensors:\n");
    printf("  %-50s %6s %20s %10s\n", "Name", "Type", "Shape", "Bytes");
    printf("  %-50s %6s %20s %10s\n", "----", "----", "-----", "-----");

    int n_tensors = 0;
    size_t total_bytes = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx->model.ctx_w);
         t != nullptr;
         t = ggml_get_next_tensor(ctx->model.ctx_w, t)) {

        char shape[64];
        if (t->ne[2] > 1) {
            snprintf(shape, sizeof(shape), "[%lld,%lld,%lld]",
                     (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2]);
        } else if (t->ne[1] > 1) {
            snprintf(shape, sizeof(shape), "[%lld,%lld]",
                     (long long)t->ne[0], (long long)t->ne[1]);
        } else {
            snprintf(shape, sizeof(shape), "[%lld]", (long long)t->ne[0]);
        }

        size_t nbytes = ggml_nbytes(t);
        printf("  %-50s %6s %20s %10zu\n",
               ggml_get_name(t), ggml_type_name(t->type), shape, nbytes);
        n_tensors++;
        total_bytes += nbytes;
    }

    printf("  %-50s %6s %20s %10zu\n", "TOTAL", "", "", total_bytes);
    printf("  %d tensors, %.1f MB\n", n_tensors, total_bytes / (1024.0 * 1024.0));
}

// ── Incremental Streaming API ────────────────────────────────────────────────

moonshine_stream_state * moonshine_stream_create(moonshine_streaming_context * ctx) {
    if (!ctx) return nullptr;
    auto * state = new moonshine_stream_state();

    const auto & hp = ctx->model.hparams;
    int buf1_len = hp.conv1_kernel_size - 1;  // initial causal left-pad (zeros)
    int buf2_len = hp.conv2_kernel_size - 1;
    state->conv1_buffer.assign(buf1_len * hp.enc_hidden_size, 0.0f);
    state->conv2_buffer.assign(buf2_len * hp.enc_hidden_size * 2, 0.0f);
    state->conv1_init_size = (int)state->conv1_buffer.size();
    state->conv2_init_size = (int)state->conv2_buffer.size();

    return state;
}

void moonshine_stream_free(moonshine_stream_state * state) {
    delete state;
}

void moonshine_stream_reset(moonshine_stream_state * state) {
    if (!state) return;
    state->sample_buffer.clear();
    state->conv1_buffer.assign(state->conv1_init_size, 0.0f);
    state->conv2_buffer.assign(state->conv2_init_size, 0.0f);
    state->accumulated_features.clear();
    state->accumulated_feature_count = 0;
    state->encoder_frames_emitted = 0;
    state->adapter_pos_offset = 0;
    state->memory.clear();
    state->memory_len = 0;
    state->cross_kv_valid = false;
    state->result_text.clear();
}

int moonshine_stream_process_audio(
        moonshine_streaming_context * ctx,
        moonshine_stream_state * state,
        const float * audio, int n_samples) {
    if (!ctx || !state || !audio || n_samples <= 0) return -1;

    return moonshine_streaming_run_frontend_incremental(ctx, audio, n_samples, state);
}

int moonshine_stream_encode(
        moonshine_streaming_context * ctx,
        moonshine_stream_state * state,
        bool is_final) {
    if (!ctx || !state) return -1;

    const auto & hp = ctx->model.hparams;

    // Flush leftover samples on final call by zero-padding to a complete frame
    if (is_final && !state->sample_buffer.empty()) {
        int remaining = (int)state->sample_buffer.size();
        int pad_needed = hp.frame_len - remaining;
        if (pad_needed > 0) {
            std::vector<float> zeros(pad_needed, 0.0f);
            moonshine_stream_process_audio(ctx, state, zeros.data(), pad_needed);
        }
    }
    const int hidden = hp.enc_hidden_size;
    const int total = state->accumulated_feature_count;

    if (total == 0) return 0;

    // Compute total lookahead = sum of right windows across all layers
    int total_lookahead = 0;
    for (size_t i = 1; i < hp.sliding_windows.size(); i += 2) {
        total_lookahead += hp.sliding_windows[i];
    }

    const int stable_count = is_final ? total
                                      : std::max(0, total - total_lookahead);
    const int new_frames = stable_count - state->encoder_frames_emitted;

    if (new_frames <= 0) return 0;

    // Compute encoder window with left context
    const int left_context = 16 * (int)hp.enc_n_layers;
    const int window_start = std::max(0, state->encoder_frames_emitted - left_context);
    const int window_len = total - window_start;

    // Run encoder on window (with sliding-window attention masks)
    const float * window_data = state->accumulated_features.data() + window_start * hidden;
    std::vector<float> encoded;
    int ret = moonshine_streaming_run_encoder(ctx, window_data, window_len, hidden,
                                              encoded, /*use_sliding_window=*/true);
    if (ret != 0) return -1;

    // Extract new stable frames from encoder output
    const int start_idx = state->encoder_frames_emitted - window_start;
    std::vector<float> new_encoded(new_frames * hidden);
    memcpy(new_encoded.data(), encoded.data() + start_idx * hidden,
           new_frames * hidden * sizeof(float));

    // Run adapter on new frames with position offset
    std::vector<float> adapter_out;
    ret = moonshine_streaming_run_adapter(ctx, new_encoded.data(), new_frames,
                                          adapter_out, state->adapter_pos_offset);
    if (ret != 0) return -1;

    // Append adapter output to memory
    state->memory.insert(state->memory.end(), adapter_out.begin(), adapter_out.end());
    state->memory_len += new_frames;

    // Update state
    state->encoder_frames_emitted = stable_count;
    state->adapter_pos_offset += new_frames;
    state->cross_kv_valid = false;

    return new_frames;
}

const char * moonshine_stream_decode(
        moonshine_streaming_context * ctx,
        moonshine_stream_state * state) {
    if (!ctx || !state || state->memory_len == 0) {
        if (state) state->result_text.clear();
        return "";
    }

    const auto & hp = ctx->model.hparams;
    const int memory_len = state->memory_len;

    // Estimate max tokens from audio duration
    // memory_len frames * 320 samples/frame / 16000 Hz = seconds
    const double duration_s = (double)memory_len * 320.0 / 16000.0;
    int max_gen = (int)(ceil(duration_s * 6.5));
    if (max_gen < 1) max_gen = 1;
    if (max_gen > 448) max_gen = 448;
    const int max_len = max_gen + 1;  // +1 for BOS

    // Init KV caches
    if (!moonshine_kv_cache_init(ctx->kv_self, hp.dec_n_layers, max_len,
                                  hp.dec_n_kv_heads, hp.dec_head_dim)) {
        fprintf(stderr, "%s: failed to init self KV cache\n", __func__);
        return "";
    }

    if (!moonshine_kv_cache_init(ctx->kv_cross, hp.dec_n_layers, memory_len,
                                  hp.dec_n_kv_heads, hp.dec_head_dim)) {
        fprintf(stderr, "%s: failed to init cross KV cache\n", __func__);
        ctx->kv_self.reset();
        return "";
    }

    // Precompute cross-attention KV from full memory
    int ret = moonshine_streaming_precompute_cross_kv(ctx, state->memory.data(), memory_len);
    if (ret != 0) {
        fprintf(stderr, "%s: cross-KV precompute failed\n", __func__);
        ctx->kv_self.reset();
        ctx->kv_cross.reset();
        return "";
    }

    // Pre-allocate decoder compute buffer with max-size graph (reserve pattern)
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    {
        const size_t n_tensors = hp.dec_n_layers * 60 + 50;
        const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead();
        struct ggml_init_params gp = { mem_size, nullptr, true };
        struct ggml_context * plan_ctx = ggml_init(gp);
        struct ggml_tensor * t = ggml_new_tensor_1d(plan_ctx, GGML_TYPE_I32, 1);
        ggml_set_name(t, "token_id"); ggml_set_input(t);
        struct ggml_tensor * p = ggml_new_tensor_1d(plan_ctx, GGML_TYPE_I32, 1);
        ggml_set_name(p, "dec_pos"); ggml_set_input(p);
        struct ggml_cgraph * plan_graph = ggml_new_graph(plan_ctx);
        struct ggml_tensor * plan_logits = streaming_build_decoder_step(
            plan_ctx, ctx->model, ctx->kv_self, ctx->kv_cross,
            t, p, memory_len, max_len - 1, plan_graph);
        ggml_set_output(plan_logits);
        ggml_build_forward_expand(plan_graph, plan_logits);
        ggml_gallocr_reserve(gallocr, plan_graph);
        ggml_free(plan_ctx);
    }

    // Greedy decode from BOS
    std::vector<int32_t> tokens;
    int32_t token = (int32_t)hp.bos_token_id;
    std::vector<float> logits(hp.vocab_size);

    for (int step = 0; step < max_len; step++) {
        ret = streaming_decode_step(ctx, token, logits, gallocr);
        if (ret != 0) break;

        int32_t best = 0;
        float best_val = logits[0];
        for (int i = 1; i < (int)hp.vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }

        if (best == (int32_t)hp.eos_token_id) break;

        tokens.push_back(best);
        token = best;
    }

    state->result_text = ctx->tokenizer.tokens_to_text(tokens);

    ggml_gallocr_free(gallocr);
    ctx->kv_self.reset();
    ctx->kv_cross.reset();

    return state->result_text.c_str();
}
