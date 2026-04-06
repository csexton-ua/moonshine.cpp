#!/usr/bin/env python3
"""Reference test for streaming moonshine adapter + decoder.

Runs the full streaming pipeline through the HuggingFace model and prints
intermediate values for comparison against the C++ implementation.
"""

import argparse
import sys

import numpy as np
import torch


def load_wav(path):
    """Load a 16kHz mono WAV file as float32 numpy array."""
    import wave
    with wave.open(path, 'rb') as w:
        assert w.getnchannels() == 1, f"Expected mono, got {w.getnchannels()} channels"
        assert w.getframerate() == 16000, f"Expected 16kHz, got {w.getframerate()}Hz"
        sw = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sw == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    return samples


def make_sine_wave(freq_hz=440.0, duration_s=1.0, sample_rate=16000):
    """Generate a sine wave test signal."""
    t = np.arange(int(duration_s * sample_rate), dtype=np.float32) / sample_rate
    return (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Test streaming adapter+decoder against HF reference")
    parser.add_argument("--audio", type=str, default=None, help="Path to WAV file (default: 1s sine wave)")
    parser.add_argument("--model", type=str, default="UsefulSensors/moonshine-streaming-tiny",
                        help="HuggingFace model name")
    parser.add_argument("--dump-adapter", action="store_true", help="Dump adapter output values")
    args = parser.parse_args()

    # Load or generate audio
    if args.audio:
        audio = load_wav(args.audio)
        print(f"Loaded audio: {args.audio} ({len(audio)} samples, {len(audio)/16000:.2f}s)")
    else:
        audio = make_sine_wave(440.0, 1.0)
        print(f"Generated 1s 440Hz sine wave ({len(audio)} samples)")

    # Load model
    print(f"Loading model: {args.model}")
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoConfig
    except ImportError:
        print("Error: transformers not found. Install with: pip install transformers")
        sys.exit(1)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    print(f"\nConfig:")
    enc_config = config.encoder if hasattr(config, 'encoder') else config
    print(f"  encoder: hidden={enc_config.hidden_size}, layers={enc_config.num_hidden_layers}, "
          f"heads={enc_config.num_attention_heads}")
    print(f"  decoder: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}")
    print(f"  vocab_size={config.vocab_size}")

    # Prepare input — pad to multiple of frame_len (80)
    frame_len = 80
    if len(audio) % frame_len != 0:
        pad_len = frame_len - (len(audio) % frame_len)
        audio = np.concatenate([audio, np.zeros(pad_len, dtype=np.float32)])
        print(f"  padded to {len(audio)} samples")
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, n_samples]

    with torch.no_grad():
        # Step 1: Run encoder (frontend + transformer)
        encoder = model.model.encoder
        encoder_output = encoder(audio_tensor)
        enc_hidden = encoder_output.last_hidden_state  # [1, T, enc_hidden]
        print(f"\nEncoder output: shape={list(enc_hidden.shape)}")
        print(f"  first 10 values: {enc_hidden[0, 0, :10].numpy()}")

        # Step 2: Adapter — pos_emb + optional proj
        decoder = model.model.decoder
        T = enc_hidden.shape[1]

        # Position embeddings
        pos_emb = decoder.pos_emb.weight[:T]  # [T, enc_hidden]
        memory = enc_hidden + pos_emb.unsqueeze(0)  # [1, T, enc_hidden]

        # Projection (if dims differ)
        if hasattr(decoder, 'proj') and decoder.proj is not None:
            memory = decoder.proj(memory)
            print(f"  adapter projection applied: enc_hidden → dec_hidden")

        print(f"\nAdapter output (memory): shape={list(memory.shape)}")
        print(f"  first 10 values: {memory[0, 0, :10].numpy()}")

        if args.dump_adapter:
            print(f"\n  Adapter output full first frame ({memory.shape[-1]} values):")
            vals = memory[0, 0, :].numpy()
            for i in range(0, len(vals), 10):
                chunk = vals[i:i+10]
                print(f"    [{i}:{i+len(chunk)}] = {chunk}")

        # Step 3: Full transcription via generate()
        print(f"\nRunning greedy decoding...")
        input_values = audio_tensor

        # Use model.generate() for reference output
        generated_ids = model.generate(
            input_values,
            max_new_tokens=194,
            do_sample=False,  # greedy
        )

        tokens = generated_ids[0].tolist()
        # Remove BOS if present
        if tokens and tokens[0] == 1:
            tokens = tokens[1:]
        # Remove EOS if present
        if tokens and tokens[-1] == 2:
            tokens = tokens[:-1]

        print(f"  tokens ({len(tokens)}): {tokens}")

        # Decode tokens using tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"  text: \"{text}\"")
        except Exception as e:
            print(f"  (tokenizer decode failed: {e})")

        # Step 4: Manual decode step for first token (for intermediate comparison)
        print(f"\nManual decode — first token logits:")
        # Get decoder input: BOS token
        bos_id = config.bos_token_id if hasattr(config, 'bos_token_id') else 1
        decoder_input_ids = torch.tensor([[bos_id]])

        # Run decoder manually
        dec_output = model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=memory,
        )
        hidden_state = dec_output.last_hidden_state  # [1, 1, dec_hidden]
        print(f"  decoder hidden (after all layers): first 10 = {hidden_state[0, 0, :10].numpy()}")

        # Project to logits
        if hasattr(model, 'proj_out'):
            logits = model.proj_out(hidden_state)
        else:
            logits = hidden_state @ model.model.decoder.embed_tokens.weight.T
        print(f"  logits shape: {list(logits.shape)}")

        # Top-5 tokens
        top5 = torch.topk(logits[0, 0], 5)
        print(f"  top-5 token IDs: {top5.indices.tolist()}")
        print(f"  top-5 logit values: {top5.values.tolist()}")
        print(f"  argmax token: {logits[0, 0].argmax().item()}")


if __name__ == "__main__":
    main()
