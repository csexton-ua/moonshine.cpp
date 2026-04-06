#!/usr/bin/env python3
"""
Compare C++ streaming frontend output against Python reference.

Usage:
    # Step 1: Run C++ and dump features
    ./build/bin/test-streaming-frontend -m models/moonshine-streaming-tiny-f32.gguf --dump /tmp/cpp_frontend.bin

    # Step 2: Run this script to compare
    python models/test_streaming_frontend.py --cpp-dump /tmp/cpp_frontend.bin

    # Or just run Python reference without comparison:
    python models/test_streaming_frontend.py
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


def run_python_frontend(audio: np.ndarray) -> np.ndarray:
    """Run the Python reference frontend (encoder embedder only)."""
    model = AutoModel.from_pretrained(
        "usefulsensors/moonshine-streaming-tiny",
        trust_remote_code=True,
    )
    model.eval()

    embedder = model.encoder.embedder

    with torch.no_grad():
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, N]
        features, _ = embedder(audio_tensor)
        # features: [1, seq_len, hidden] in PyTorch
        return features.squeeze(0).numpy()  # [seq_len, hidden]


def load_cpp_dump(path: str):
    """Load binary dump from C++ test: [hidden_dim(i32), seq_len(i32), float32 data...]"""
    with open(path, "rb") as f:
        hidden_dim, seq_len = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32)
        # ggml stores ne[0]=hidden, ne[1]=seq in column-major order
        # which is equivalent to [seq_len, hidden] in numpy row-major
        return data.reshape(seq_len, hidden_dim)  # → [seq_len, hidden]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-dump", help="Path to C++ binary dump for comparison")
    args = parser.parse_args()

    audio = make_sine_wave(440.0, 1.0)
    print(f"Audio: {len(audio)} samples ({len(audio)/16000:.2f} sec)")

    # Run Python reference
    print("Running Python frontend...")
    py_features = run_python_frontend(audio)
    print(f"Python output: {py_features.shape} (seq_len x hidden)")
    print(f"Stats: min={py_features.min():.4f} max={py_features.max():.4f} mean={py_features.mean():.4f}")
    print(f"First 10 values (row-major): {py_features.ravel()[:10]}")

    if args.cpp_dump:
        print(f"\nLoading C++ dump: {args.cpp_dump}")
        cpp_features = load_cpp_dump(args.cpp_dump)
        print(f"C++ output: {cpp_features.shape}")

        if py_features.shape != cpp_features.shape:
            print(f"SHAPE MISMATCH: Python {py_features.shape} vs C++ {cpp_features.shape}")
            sys.exit(1)

        diff = np.abs(py_features - cpp_features)
        print(f"\nComparison:")
        print(f"  Max abs diff:  {diff.max():.6e}")
        print(f"  Mean abs diff: {diff.mean():.6e}")
        print(f"  RMS diff:      {np.sqrt((diff**2).mean()):.6e}")

        # Print first few values side by side
        print(f"\nFirst 5 values (Python vs C++):")
        for i in range(min(5, py_features.shape[0])):
            for j in range(min(5, py_features.shape[1])):
                print(f"  [{i},{j}] py={py_features[i,j]:+.6f}  cpp={cpp_features[i,j]:+.6f}  diff={diff[i,j]:.2e}")

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
        # C++ stores [hidden, seq_len], so column-major traversal = iterate hidden first
        py_colmajor = py_features.T  # [hidden, seq_len]
        print(f"  {py_colmajor.ravel()[:10]}")


if __name__ == "__main__":
    main()
