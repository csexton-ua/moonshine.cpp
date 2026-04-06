# moonshine.cpp

Dependency-free C++ implementation of the [Moonshine](https://github.com/usefulsensors/moonshine) automatic speech recognition model, using [ggml](https://github.com/ggml-org/ggml) as the tensor computation backend.

Supports both Moonshine v1 (batch) and v2 (streaming) model architectures.

## Quick Start

```bash
# Clone with ggml submodule
git clone --recurse-submodules https://github.com/csexton/moonshine.cpp
cd moonshine.cpp

# Build
cmake -B build
cmake --build build

# Convert a model (see Models section below)
pip install gguf safetensors huggingface_hub numpy
python models/convert-moonshine-to-gguf.py \
    --model usefulsensors/moonshine-tiny \
    --outfile models/moonshine-tiny-f32.gguf

# Transcribe
./build/bin/moonshine-cli -m models/moonshine-tiny-f32.gguf audio.wav
```

## Features

- No external dependencies beyond ggml (vendored)
- Moonshine v1 (batch) and v2 (streaming) model support
- Real-time incremental streaming transcription
- 16-bit PCM and 32-bit float WAV input
- Configurable thread count
- Verbose timing output and benchmark mode
- C API for library integration
- CTest test suite
- Web demo with microphone capture

## Models

Model files are not included in this repository. Use the included conversion scripts to convert models from HuggingFace to GGUF format.

### Available models

| Model | HuggingFace ID | Architecture |
|-------|----------------|--------------|
| Moonshine Tiny | `usefulsensors/moonshine-tiny` | v1 (batch) |
| Moonshine Base | `usefulsensors/moonshine-base` | v1 (batch) |
| Moonshine Streaming Tiny | `usefulsensors/moonshine-streaming-tiny` | v2 (streaming) |

### Converting models

Install Python dependencies:

```bash
pip install gguf safetensors huggingface_hub numpy
```

Convert a v1 model:

```bash
python models/convert-moonshine-to-gguf.py \
    --model usefulsensors/moonshine-tiny \
    --outfile models/moonshine-tiny-f32.gguf
```

Convert a v2 (streaming) model:

```bash
python models/convert-moonshine-streaming-to-gguf.py \
    --model usefulsensors/moonshine-streaming-tiny \
    --outfile models/moonshine-streaming-tiny-f32.gguf
```

Both scripts accept `--outtype f16` for half-precision models (smaller, slightly faster on some hardware).

## Build

```bash
cmake -B build
cmake --build build
```

The build produces:
- `build/libmoonshine.a` — v1 static library
- `build/libmoonshine-streaming.a` — v2 streaming static library
- `build/libmoonshine-detect.a` — architecture detection library
- `build/bin/moonshine-cli` — command-line tool
- `build/bin/moonshine-stream-server` — streaming server (binary protocol over stdin/stdout)

### Build options

| Option | Default | Description |
|--------|---------|-------------|
| `MOONSHINE_BUILD_EXAMPLES` | `ON` | Build the CLI tool and stream server |
| `MOONSHINE_BUILD_TESTS` | `ON` | Build test programs |

## Usage

### Batch transcription

```bash
# Basic transcription (text to stdout)
./build/bin/moonshine-cli -m models/moonshine-tiny-f32.gguf audio.wav

# With timing info
./build/bin/moonshine-cli -m models/moonshine-tiny-f32.gguf -v audio.wav

# Benchmark (3 runs)
./build/bin/moonshine-cli -m models/moonshine-tiny-f32.gguf --benchmark 3 audio.wav
```

### Streaming transcription (v2 models)

```bash
# Stream a file in 1-second chunks
./build/bin/moonshine-cli -m models/moonshine-streaming-tiny-f32.gguf --stream audio.wav

# Custom chunk size
./build/bin/moonshine-cli -m models/moonshine-streaming-tiny-f32.gguf --stream --chunk-sec 0.5 audio.wav
```

The model architecture is auto-detected from the GGUF file. Using `--stream` with a v1 model will print an error.

### CLI options

```
usage: moonshine-cli [options] <audio.wav>

options:
  -m, --model PATH       Path to GGUF model file (required)
  -t, --tokenizer PATH   Path to tokenizer.bin (default: auto-detect)
      --threads N        Number of threads (default: 4)
  -v, --verbose          Print timing info to stderr
      --benchmark [N]    Run N times and print stats (default: 10, implies -v)
      --stream           Incremental streaming mode (v2 models only)
      --chunk-sec F      Seconds per chunk in stream mode (default: 1.0)
  -h, --help             Show this help
```

Transcription text goes to stdout; all other output goes to stderr. This makes it safe to pipe:

```bash
./build/bin/moonshine-cli -m model.gguf audio.wav 2>/dev/null > transcript.txt
```

## Web Demo

A browser-based demo with live microphone capture is included in `examples/web/`.

```bash
# Requires ffmpeg for audio conversion
python3 examples/web/server.py
# Open http://localhost:8765
```

The web server auto-detects available models in `models/` and supports both batch transcription (v1 and v2) and real-time WebSocket streaming (v2 only). Set `PORT=9000` to use a different port.

## C API

### v1 (batch)

```c
#include "moonshine.h"

struct moonshine_context * ctx = moonshine_init("model.gguf");

const char * text = moonshine_transcribe(ctx, audio_samples, n_samples);

struct moonshine_timing timing;
moonshine_get_timing(ctx, &timing);
printf("encode: %.1f ms, decode: %.1f ms\n", timing.encode_ms, timing.decode_ms);

moonshine_free(ctx);
```

### v2 (streaming)

```c
#include "moonshine-streaming.h"

moonshine_streaming_context * ctx = moonshine_streaming_init("streaming-model.gguf");

// Create a streaming session
moonshine_stream_state * state = moonshine_stream_create(ctx);

// Feed audio incrementally
while (have_audio) {
    moonshine_stream_process_audio(ctx, state, chunk, chunk_len);
    moonshine_stream_encode(ctx, state, false);
    const char * text = moonshine_stream_decode(ctx, state);
    printf("%s\n", text);
}

// Flush remaining audio
moonshine_stream_encode(ctx, state, true);
const char * final = moonshine_stream_decode(ctx, state);

moonshine_stream_reset(state);  // reset for next utterance
moonshine_stream_free(state);
moonshine_streaming_free(ctx);
```

### Architecture detection

```c
#include "moonshine-detect.h"

enum moonshine_arch arch = moonshine_detect_arch("model.gguf");
// MOONSHINE_ARCH_V1 (batch) or MOONSHINE_ARCH_V2 (streaming)
```

## Tests

```bash
cd build && ctest --output-on-failure
```

Tests require a model file at `models/moonshine-tiny-f32.gguf` and the WAV fixture at `tests/fixtures/beckett.wav`.

## License

MIT
