#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Model architecture types
enum moonshine_arch {
    MOONSHINE_ARCH_UNKNOWN  = 0,
    MOONSHINE_ARCH_V1       = 1,  // "moonshine" — full-attention, batch-only
    MOONSHINE_ARCH_V2       = 2,  // "moonshine_streaming" — sliding-window, supports streaming
};

// Read general.architecture from a GGUF file without loading the full model.
// Returns MOONSHINE_ARCH_UNKNOWN on error or unrecognized architecture.
enum moonshine_arch moonshine_detect_arch(const char * model_path);

#ifdef __cplusplus
}
#endif
