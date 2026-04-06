#!/usr/bin/env python3
"""
Compare C++ streaming encoder output against Python reference.

Usage:
    # Step 1: Run C++ and dump encoder output
    ./build/bin/test-streaming-encoder -m models/moonshine-streaming-tiny-f32.gguf --dump /tmp/cpp_encoder.bin

    # Step 2: Run this script to compare
    python models/test_streaming_encoder.py --cpp-dump /tmp/cpp_encoder.bin

    # Or just run Python reference without comparison:
    python models/test_streaming_encoder.py
"""

import argparse
import struct
import sys

import numpy as np
import torch
from transformers import AutoModel


def make_sine_wave(freq_hz=440.0, duration_s=1.0, sample_rate=16000):
    """Generate same test signal as C++ test."""
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    return 0.5 * np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)


def run_python_encoder(audio: np.ndarray) -> tuple:
    """Run the Python reference frontend + encoder."""
    model = AutoModel.from_pretrained(
        "usefulsensors/moonshine-streaming-tiny",
        trust_remote_code=True,
    )
    model.eval()

    with torch.no_grad():
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, N]

        # Run frontend (embedder)
        features, _ = model.encoder.embedder(audio_tensor)
        print(f"  Frontend output: {features.shape}")  # [1, seq_len, hidden]

        # Run full encoder (includes layers + final_norm)
        encoder_output = model.encoder(audio_tensor)
        hidden_states = encoder_output.last_hidden_state  # [1, seq_len, hidden]

        # Return [seq_len, hidden]
        return hidden_states.squeeze(0).numpy()


def load_cpp_dump(path: str):
    """Load binary dump from C++ test: [hidden_dim(i32), seq_len(i32), float32 data...]"""
    with open(path, "rb") as f:
        hidden_dim, seq_len = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32)
        # C++ stores [hidden, seq_len] in ggml column-major = [seq_len, hidden] in numpy row-major
        return data.reshape(seq_len, hidden_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-dump", help="Path to C++ binary dump for comparison")
    args = parser.parse_args()

    audio = make_sine_wave(440.0, 1.0)
    print(f"Audio: {len(audio)} samples ({len(audio)/16000:.2f} sec)")

    # Run Python reference
    print("Running Python frontend + encoder...")
    py_encoded = run_python_encoder(audio)
    print(f"Python encoder output: {py_encoded.shape} (seq_len x hidden)")
    print(f"Stats: min={py_encoded.min():.4f} max={py_encoded.max():.4f} mean={py_encoded.mean():.4f}")
    print(f"First 10 values (row-major): {py_encoded.ravel()[:10]}")

    if args.cpp_dump:
        print(f"\nLoading C++ dump: {args.cpp_dump}")
        cpp_encoded = load_cpp_dump(args.cpp_dump)
        print(f"C++ output: {cpp_encoded.shape}")

        if py_encoded.shape != cpp_encoded.shape:
            print(f"SHAPE MISMATCH: Python {py_encoded.shape} vs C++ {cpp_encoded.shape}")
            sys.exit(1)

        diff = np.abs(py_encoded - cpp_encoded)
        print(f"\nComparison:")
        print(f"  Max abs diff:  {diff.max():.6e}")
        print(f"  Mean abs diff: {diff.mean():.6e}")
        print(f"  RMS diff:      {np.sqrt((diff**2).mean()):.6e}")

        # Print first few values side by side
        print(f"\nFirst 5 values (Python vs C++):")
        for i in range(min(5, py_encoded.shape[0])):
            for j in range(min(5, py_encoded.shape[1])):
                print(f"  [{i},{j}] py={py_encoded[i,j]:+.6f}  cpp={cpp_encoded[i,j]:+.6f}  diff={diff[i,j]:.2e}")

        if diff.max() < 1e-4:
            print("\nPASS: outputs match within tolerance (1e-4)")
        elif diff.max() < 1e-3:
            print("\nWARN: outputs close but max diff > 1e-4")
        else:
            print("\nFAIL: outputs differ significantly")
            sys.exit(1)
    else:
        # Just dump Python output for manual comparison
        print(f"\nPython first 10 (col-major, matching C++ layout):")
        py_colmajor = py_encoded.T  # [hidden, seq_len]
        print(f"  {py_colmajor.ravel()[:10]}")


if __name__ == "__main__":
    main()
