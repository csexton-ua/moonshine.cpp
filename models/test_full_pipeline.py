"""
Test script to generate ground-truth full pipeline outputs from the HuggingFace
Moonshine model for comparison with the C++ implementation.

Tests encoder output, greedy decode token IDs, first-step logits, and final text.
"""

import torch
import numpy as np


def main():
    from transformers import AutoModelForSpeechSeq2Seq

    print("Loading moonshine-tiny model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "usefulsensors/moonshine-tiny", trust_remote_code=True
    )
    model.eval()

    # --- Test 1: 440 Hz sine wave (WAV-quantized) ---
    sr = 16000
    t = torch.arange(sr, dtype=torch.float32) / sr
    audio_sine = 0.5 * torch.sin(2 * np.pi * 440 * t)
    # Quantize to int16 then back to float, matching WAV round-trip
    audio_sine_i16 = (audio_sine * 32767).to(torch.int16)
    audio_sine = audio_sine_i16.to(torch.float32) / 32768.0

    print("\n=== 440 Hz sine wave (16000 samples, WAV-matched) ===")

    with torch.no_grad():
        # Encoder output
        enc_out = model.encoder(audio_sine.unsqueeze(0))
        if hasattr(enc_out, "last_hidden_state"):
            enc_hidden = enc_out.last_hidden_state
        elif isinstance(enc_out, tuple):
            enc_hidden = enc_out[0]
        else:
            enc_hidden = enc_out

        print(f"Encoder output shape: {enc_hidden.shape}")
        flat = enc_hidden.squeeze(0).numpy().flatten()[:10]
        print(f"Encoder first 10: {' '.join(f'{v:.6f}' for v in flat)}")

        # Generate tokens (greedy)
        tokens = model.generate(audio_sine.unsqueeze(0))
        token_ids = tokens[0].tolist()
        print(f"Generated token IDs: {token_ids}")
        print(f"Number of tokens: {len(token_ids)}")

        # Manual first decoder step with BOS token for logit comparison
        bos_id = 1
        decoder_input = torch.tensor([[bos_id]], dtype=torch.long)

        # Try to run decoder manually
        try:
            dec_out = model.decoder(
                input_ids=decoder_input, encoder_hidden_states=enc_hidden
            )
            if hasattr(dec_out, "last_hidden_state"):
                dec_hidden = dec_out.last_hidden_state
            elif isinstance(dec_out, tuple):
                dec_hidden = dec_out[0]
            else:
                dec_hidden = dec_out

            # Project to vocab (check for proj_out or lm_head)
            if hasattr(model, "proj_out"):
                logits = model.proj_out(dec_hidden)
            elif hasattr(model, "lm_head"):
                logits = model.lm_head(dec_hidden)
            else:
                # Weight tying: use embedding weight
                logits = torch.nn.functional.linear(
                    dec_hidden, model.decoder.embed_tokens.weight
                )

            print(f"\nFirst step logits shape: {logits.shape}")
            top10_vals, top10_ids = logits[0, 0].topk(10)
            print("Top 10 logits at step 0 (BOS input):")
            for val, idx in zip(top10_vals.numpy(), top10_ids.numpy()):
                print(f"  token {idx:5d}: {val:.6f}")
        except Exception as e:
            print(f"\nManual decoder step failed: {e}")

    # --- Test 2: two_cities_16k.wav ---
    try:
        import soundfile as sf

        audio_tc, sr_tc = sf.read("test-assets/two_cities_16k.wav")
        audio_tc = torch.tensor(audio_tc, dtype=torch.float32)
        if len(audio_tc.shape) > 1:
            audio_tc = audio_tc.mean(dim=-1)

        print(f"\n=== two_cities_16k.wav ({audio_tc.shape[0]} samples) ===")

        with torch.no_grad():
            tokens_tc = model.generate(audio_tc.unsqueeze(0))
            tc_ids = tokens_tc[0].tolist()
            print(f"Generated token IDs: {tc_ids}")
            print(f"Number of tokens: {len(tc_ids)}")
    except FileNotFoundError:
        print("\nSkipping two_cities test: test-assets/two_cities_16k.wav not found")
    except ImportError:
        print("\nSkipping two_cities test: soundfile not installed")


if __name__ == "__main__":
    main()
