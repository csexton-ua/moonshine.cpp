"""
Test script to generate ground-truth full encoder outputs from the HuggingFace
Moonshine model for comparison with the C++ implementation.
"""

import torch
import numpy as np
from transformers import AutoModel


def run_full_encoder(model, audio, label=""):
    """Run the full encoder (conv stem + transformer layers + output norm)."""
    x = audio.unsqueeze(0)  # [n_samples] -> [1, n_samples]

    with torch.no_grad():
        # The encoder forward pass does conv stem + transformer + output norm
        encoder_output = model.encoder(x)

    # encoder_output is a model output object; get the tensor
    if hasattr(encoder_output, 'last_hidden_state'):
        hidden = encoder_output.last_hidden_state
    elif isinstance(encoder_output, tuple):
        hidden = encoder_output[0]
    else:
        hidden = encoder_output

    # hidden shape: [1, seq_len, hidden]
    out = hidden.squeeze(0)  # [seq_len, hidden]

    # ggml layout: ne[0]=hidden fastest → numpy [seq_len, hidden] row-major
    out_np = out.numpy()  # [seq_len, hidden]

    print(f"\n=== {label} ===")
    print(f"Input: {audio.shape[0]} samples")
    print(f"Output shape: seq_len={out_np.shape[0]}, hidden={out_np.shape[1]}")
    flat = out_np.flatten()[:10]  # first 10 hidden dims at seq pos 0
    print(f"First 10 values (ggml layout): {' '.join(f'{v:.6f}' for v in flat)}")
    print(f"Min: {out_np.min():.6f}, Max: {out_np.max():.6f}, Mean: {out_np.mean():.6f}")
    return out_np


def main():
    print("Loading moonshine-tiny model...")
    model = AutoModel.from_pretrained("usefulsensors/moonshine-tiny", trust_remote_code=True)
    model.eval()

    # Test 1: 440 Hz sine wave (1 second) — match C++ WAV generation exactly
    sr = 16000
    t = torch.arange(sr, dtype=torch.float32) / sr
    audio_sine = 0.5 * torch.sin(2 * np.pi * 440 * t)
    # Quantize to int16 then back to float, matching WAV round-trip
    audio_sine_i16 = (audio_sine * 32767).to(torch.int16)
    audio_sine = audio_sine_i16.to(torch.float32) / 32768.0
    run_full_encoder(model, audio_sine, "440 Hz sine (16000 samples, WAV-matched)")

    # Test 2: Zeros (1 second)
    audio_zeros = torch.zeros(16000)
    run_full_encoder(model, audio_zeros, "Zeros (16000 samples)")


if __name__ == "__main__":
    main()
