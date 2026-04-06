"""
Test script to generate ground-truth conv stem outputs from the HuggingFace
Moonshine model for comparison with the C++ implementation.
"""

import torch
import numpy as np
from transformers import AutoModel


def run_conv_stem(encoder, audio, label=""):
    """Run just the conv stem portion of the encoder and print results."""
    x = audio.unsqueeze(0).unsqueeze(0)  # [n_samples] -> [1, 1, n_samples]

    with torch.no_grad():
        # Conv1 (no bias) + tanh
        x = encoder.conv1(x)
        x = torch.tanh(x)
        # GroupNorm + affine
        x = encoder.groupnorm(x)
        # Conv2 + bias + GELU
        x = encoder.conv2(x)
        x = torch.nn.functional.gelu(x)
        # Conv3 + bias + GELU
        x = encoder.conv3(x)
        x = torch.nn.functional.gelu(x)

    # x shape: [1, hidden, seq_len]
    # Transpose to [1, seq_len, hidden] then squeeze batch
    out = x.squeeze(0)  # [hidden, seq_len]

    out_np = out.numpy()

    print(f"\n=== {label} ===")
    print(f"Input: {audio.shape[0]} samples")
    print(f"Output shape: {out_np.shape}  (hidden={out_np.shape[0]}, seq_len={out_np.shape[1]})")
    # Print first 10 values in column-major order (ggml layout: hidden dim varies fastest)
    col_major = out_np.T.flatten()[:10]  # transpose to [seq, hidden], then flatten
    print(f"First 10 values (col-major/ggml): {col_major}")
    print(f"Min: {out_np.min():.6f}, Max: {out_np.max():.6f}, Mean: {out_np.mean():.6f}")
    return out_np


def main():
    print("Loading moonshine-tiny model...")
    model = AutoModel.from_pretrained("usefulsensors/moonshine-tiny", trust_remote_code=True)
    model.eval()

    encoder = model.encoder

    # Print conv stem layer info
    print(f"\nConv1: {encoder.conv1}")
    print(f"GroupNorm: {encoder.groupnorm}")
    print(f"Conv2: {encoder.conv2}")
    print(f"Conv3: {encoder.conv3}")

    # Test 1: 440 Hz sine wave (1 second)
    t = torch.linspace(0, 1, 16000, dtype=torch.float32)
    audio_sine = 0.5 * torch.sin(2 * np.pi * 440 * t)
    run_conv_stem(encoder, audio_sine, "440 Hz sine (16000 samples)")

    # Test 2: Zeros (1 second)
    audio_zeros = torch.zeros(16000)
    run_conv_stem(encoder, audio_zeros, "Zeros (16000 samples)")


if __name__ == "__main__":
    main()
