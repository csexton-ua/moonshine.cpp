#include "moonshine-audio.h"

#include <cstdio>
#include <cstring>

bool moonshine_load_wav(const char * path, std::vector<float> & audio, int32_t * sample_rate) {
    audio.clear();

    FILE * file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, path);
        return false;
    }

    // Read RIFF header
    char riff_header[4];
    if (fread(riff_header, 1, 4, file) != 4 || strncmp(riff_header, "RIFF", 4) != 0) {
        fclose(file);
        fprintf(stderr, "%s: not a RIFF file\n", __func__);
        return false;
    }

    // Skip chunk size, check WAVE
    fseek(file, 4, SEEK_CUR);
    char wave_header[4];
    if (fread(wave_header, 1, 4, file) != 4 || strncmp(wave_header, "WAVE", 4) != 0) {
        fclose(file);
        fprintf(stderr, "%s: not a WAVE file\n", __func__);
        return false;
    }

    // Find "fmt " chunk
    char chunk_id[4];
    uint32_t chunk_size = 0;
    bool found_fmt = false;
    while (fread(chunk_id, 1, 4, file) == 4) {
        if (fread(&chunk_size, 4, 1, file) != 1) break;
        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            found_fmt = true;
            break;
        }
        fseek(file, chunk_size, SEEK_CUR);
    }
    if (!found_fmt) {
        fclose(file);
        fprintf(stderr, "%s: no fmt chunk found\n", __func__);
        return false;
    }

    // Parse fmt chunk
    if (chunk_size < 16) {
        fclose(file);
        fprintf(stderr, "%s: fmt chunk too small\n", __func__);
        return false;
    }
    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sr = 0, byte_rate = 0;
    uint16_t block_align = 0;
    if (fread(&audio_format,    2, 1, file) != 1 ||
        fread(&num_channels,    2, 1, file) != 1 ||
        fread(&sr,              4, 1, file) != 1 ||
        fread(&byte_rate,       4, 1, file) != 1 ||
        fread(&block_align,     2, 1, file) != 1 ||
        fread(&bits_per_sample, 2, 1, file) != 1) {
        fclose(file);
        fprintf(stderr, "%s: failed to read fmt chunk\n", __func__);
        return false;
    }
    if (chunk_size > 16) fseek(file, chunk_size - 16, SEEK_CUR);

    (void)byte_rate;
    (void)block_align;

    const bool is_pcm16   = (audio_format == 1 && bits_per_sample == 16);
    const bool is_float32 = (audio_format == 3 && bits_per_sample == 32);

    if (!is_pcm16 && !is_float32) {
        fclose(file);
        fprintf(stderr, "%s: unsupported WAV format (audio_format=%u, bits_per_sample=%u). "
                "Only 16-bit PCM and 32-bit float are supported.\n",
                __func__, audio_format, bits_per_sample);
        return false;
    }

    if (num_channels != 1) {
        fprintf(stderr, "%s: warning: WAV has %u channels, expected mono\n", __func__, num_channels);
    }

    if (sr != 16000) {
        fprintf(stderr, "%s: warning: sample rate is %u, expected 16000\n", __func__, sr);
    }

    // Find "data" chunk
    bool found_data = false;
    while (fread(chunk_id, 1, 4, file) == 4) {
        if (fread(&chunk_size, 4, 1, file) != 1) break;
        if (strncmp(chunk_id, "data", 4) == 0) {
            found_data = true;
            break;
        }
        fseek(file, chunk_size, SEEK_CUR);
    }
    if (!found_data) {
        fclose(file);
        fprintf(stderr, "%s: no data chunk found\n", __func__);
        return false;
    }

    const size_t bytes_per_sample = is_pcm16 ? sizeof(int16_t) : sizeof(float);
    const size_t total_samples = chunk_size / bytes_per_sample;
    const size_t samples_per_channel = total_samples / num_channels;
    if (samples_per_channel == 0) {
        fclose(file);
        fprintf(stderr, "%s: no samples found\n", __func__);
        return false;
    }

    if (is_pcm16) {
        // Read 16-bit PCM samples in bulk, then convert to float
        std::vector<int16_t> raw(total_samples);
        if (fread(raw.data(), sizeof(int16_t), total_samples, file) != total_samples) {
            fclose(file);
            fprintf(stderr, "%s: failed to read audio data\n", __func__);
            return false;
        }

        audio.resize(samples_per_channel);
        if (num_channels == 1) {
            for (size_t i = 0; i < samples_per_channel; i++) {
                audio[i] = (float)raw[i] / 32768.0f;
            }
        } else {
            for (size_t i = 0; i < samples_per_channel; i++) {
                float sum = 0.0f;
                for (int ch = 0; ch < num_channels; ch++) {
                    sum += (float)raw[i * num_channels + ch] / 32768.0f;
                }
                audio[i] = sum / num_channels;
            }
        }
    } else {
        // Read 32-bit float samples directly (no normalization needed)
        if (num_channels == 1) {
            // Mono: read directly into output, no temp buffer needed
            audio.resize(samples_per_channel);
            if (fread(audio.data(), sizeof(float), samples_per_channel, file) != samples_per_channel) {
                fclose(file);
                fprintf(stderr, "%s: failed to read audio data\n", __func__);
                return false;
            }
        } else {
            std::vector<float> raw(total_samples);
            if (fread(raw.data(), sizeof(float), total_samples, file) != total_samples) {
                fclose(file);
                fprintf(stderr, "%s: failed to read audio data\n", __func__);
                return false;
            }
            audio.resize(samples_per_channel);
            for (size_t i = 0; i < samples_per_channel; i++) {
                float sum = 0.0f;
                for (int ch = 0; ch < num_channels; ch++) {
                    sum += raw[i * num_channels + ch];
                }
                audio[i] = sum / num_channels;
            }
        }
    }

    fclose(file);
    *sample_rate = (int32_t)sr;
    return true;
}
