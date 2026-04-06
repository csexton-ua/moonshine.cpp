#include "moonshine-detect.h"

#include "gguf.h"

#include <cstdio>
#include <cstring>

enum moonshine_arch moonshine_detect_arch(const char * model_path) {
    if (!model_path) {
        return MOONSHINE_ARCH_UNKNOWN;
    }

    // Open GGUF with no_alloc — we only need metadata, not tensor data
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };

    struct gguf_context * ctx = gguf_init_from_file(model_path, params);
    if (!ctx) {
        return MOONSHINE_ARCH_UNKNOWN;
    }

    enum moonshine_arch arch = MOONSHINE_ARCH_UNKNOWN;

    int64_t key_id = gguf_find_key(ctx, "general.architecture");
    if (key_id >= 0) {
        const char * val = gguf_get_val_str(ctx, key_id);
        if (val) {
            if (strcmp(val, "moonshine") == 0) {
                arch = MOONSHINE_ARCH_V1;
            } else if (strcmp(val, "moonshine_streaming") == 0) {
                arch = MOONSHINE_ARCH_V2;
            } else {
                fprintf(stderr, "%s: unrecognized architecture '%s'\n", __func__, val);
            }
        }
    } else {
        // Fallback: check for architecture-specific keys
        if (gguf_find_key(ctx, "moonshine_streaming.encoder.embedding_length") >= 0) {
            arch = MOONSHINE_ARCH_V2;
        } else if (gguf_find_key(ctx, "moonshine.encoder.embedding_length") >= 0) {
            arch = MOONSHINE_ARCH_V1;
        }
    }

    gguf_free(ctx);
    return arch;
}
