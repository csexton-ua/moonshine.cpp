#!/usr/bin/env python3
"""Minimal web server for testing moonshine.cpp transcription."""

import base64
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Project root = two levels up from this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLI = PROJECT_ROOT / "build" / "bin" / "moonshine-cli"
STREAM_SERVER = PROJECT_ROOT / "build" / "bin" / "moonshine-stream-server"
MODELS_DIR = PROJECT_ROOT / "models"
HTML = Path(__file__).resolve().parent / "index.html"

PORT = int(os.environ.get("PORT", 8765))


MAX_AUDIO_SEC = 10  # Moonshine supports up to ~10s of audio


def prepare_wav(input_path: str, output_path: str) -> bool:
    """Convert/resample/trim audio to 16kHz mono WAV (max MAX_AUDIO_SEC seconds) using ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-t", str(MAX_AUDIO_SEC),
             "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
             output_path],
            capture_output=True, check=True, timeout=15,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def detect_model_arch(model_path: str) -> str:
    """Detect model architecture by reading the GGUF general.architecture key."""
    try:
        with open(model_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return "unknown"
            version = int.from_bytes(f.read(4), "little")
            if version < 2:
                return "unknown"
            _tensor_count = int.from_bytes(f.read(8), "little")
            kv_count = int.from_bytes(f.read(8), "little")

            for _ in range(kv_count):
                key_len = int.from_bytes(f.read(8), "little")
                key = f.read(key_len).decode("utf-8", errors="replace")
                val_type = int.from_bytes(f.read(4), "little")

                if key == "general.architecture" and val_type == 8:  # 8 = STRING
                    str_len = int.from_bytes(f.read(8), "little")
                    val = f.read(str_len).decode("utf-8", errors="replace")
                    if val == "moonshine_streaming":
                        return "v2"
                    elif val == "moonshine":
                        return "v1"
                    else:
                        return "unknown"
                else:
                    _skip_gguf_value(f, val_type)
    except Exception:
        pass
    return "unknown"


def _skip_gguf_value(f, val_type: int):
    """Skip a GGUF value based on its type tag."""
    fixed_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if val_type in fixed_sizes:
        f.read(fixed_sizes[val_type])
    elif val_type == 8:  # string
        str_len = int.from_bytes(f.read(8), "little")
        f.read(str_len)
    elif val_type == 9:  # array
        elem_type = int.from_bytes(f.read(4), "little")
        count = int.from_bytes(f.read(8), "little")
        for _ in range(count):
            _skip_gguf_value(f, elem_type)


def find_models():
    """Find all .gguf files in the models directory with architecture info."""
    models = {}
    for f in sorted(MODELS_DIR.glob("*.gguf")):
        name = f.stem
        size_mb = f.stat().st_size / (1024 * 1024)
        arch = detect_model_arch(str(f))
        models[name] = {"path": str(f), "size_mb": round(size_mb, 1), "arch": arch}
    return models


MODELS = find_models()
DEFAULT_MODEL = next(iter(MODELS)) if MODELS else None


def _parse_multipart(content_type: str, body: bytes) -> dict:
    """Parse multipart/form-data and return dict of field name -> value."""
    match = re.search(r'boundary=([^\s;]+)', content_type)
    if not match:
        return {}
    boundary = match.group(1).encode()
    parts = body.split(b"--" + boundary)
    fields = {}
    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end < 0:
            continue
        header_block = part[:header_end].decode(errors="replace")
        part_data = part[header_end + 4:]
        if part_data.endswith(b"\r\n"):
            part_data = part_data[:-2]

        name_match = re.search(r'name="([^"]*)"', header_block)
        if name_match:
            name = name_match.group(1)
            fields[name] = part_data
            fn_match = re.search(r'filename="([^"]*)"', header_block)
            if fn_match and fn_match.group(1):
                fields[name + "_filename"] = fn_match.group(1).encode()
    return fields


# ── WebSocket helpers (RFC 6455, minimal implementation) ─────────────────────

WS_MAGIC = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

WS_OP_TEXT   = 0x01
WS_OP_BINARY = 0x02
WS_OP_CLOSE  = 0x08
WS_OP_PING   = 0x09
WS_OP_PONG   = 0x0A

# Subprocess message types (must match moonshine-stream-server protocol)
MSG_AUDIO    = 1
MSG_FINALIZE = 2
MSG_RESET    = 3

# Response flags (must match RESP_* in moonshine-stream-server)
RESP_FLAG_SEGMENT_FINAL = 0x01

MAX_RESPONSE_LEN = 1_000_000


def ws_accept_key(key: str) -> str:
    """Compute Sec-WebSocket-Accept from client key."""
    digest = hashlib.sha1(key.encode() + WS_MAGIC).digest()
    return base64.b64encode(digest).decode()


def _sock_recv_exact(sock, n):
    """Read exactly n bytes from a socket."""
    buf = bytearray(n)
    pos = 0
    while pos < n:
        nbytes = sock.recv_into(memoryview(buf)[pos:])
        if not nbytes:
            return bytes(buf[:pos])
        pos += nbytes
    return bytes(buf)


def ws_read_frame(sock):
    """Read one WebSocket frame. Returns (opcode, payload) or (None, None) on close/error."""
    header = _sock_recv_exact(sock, 2)
    if len(header) < 2:
        return None, None

    opcode = header[0] & 0x0F
    masked = header[1] & 0x80
    length = header[1] & 0x7F

    if length == 126:
        data = _sock_recv_exact(sock, 2)
        if len(data) < 2:
            return None, None
        length = struct.unpack(">H", data)[0]
    elif length == 127:
        data = _sock_recv_exact(sock, 8)
        if len(data) < 8:
            return None, None
        length = struct.unpack(">Q", data)[0]

    mask_key = None
    if masked:
        mask_key = _sock_recv_exact(sock, 4)
        if len(mask_key) < 4:
            return None, None

    payload = _sock_recv_exact(sock, length) if length > 0 else b""
    if len(payload) < length:
        return None, None

    if mask_key:
        # XOR unmask via big-integer operation (fast in CPython)
        n = len(payload)
        mask_extended = (mask_key * (n // 4 + 1))[:n]
        p_int = int.from_bytes(payload, "big")
        m_int = int.from_bytes(mask_extended, "big")
        payload = (p_int ^ m_int).to_bytes(n, "big")

    return opcode, payload


def ws_send_frame(sock, opcode, payload):
    """Send a WebSocket frame (server-to-client, unmasked)."""
    frame = bytearray()
    frame.append(0x80 | opcode)

    length = len(payload)
    if length < 126:
        frame.append(length)
    elif length < 65536:
        frame.append(126)
        frame.extend(struct.pack(">H", length))
    else:
        frame.append(127)
        frame.extend(struct.pack(">Q", length))

    frame.extend(payload)
    sock.sendall(bytes(frame))


def ws_send_text(sock, text):
    ws_send_frame(sock, WS_OP_TEXT, text.encode("utf-8"))


def ws_send_close(sock, code=1000):
    ws_send_frame(sock, WS_OP_CLOSE, struct.pack(">H", code))


def _read_subprocess_response(stdout):
    """Read a length-prefixed response from the stream server.

    Protocol: [uint32 payload_len] [uint8 flags] [UTF-8 text]
    flags & 0x01 = segment_final (auto-reset boundary)

    Returns (text, segment_final) or (None, False) on error.
    """
    len_data = stdout.read(4)
    if len(len_data) < 4:
        return None, False
    payload_len = struct.unpack("<I", len_data)[0]
    if payload_len > MAX_RESPONSE_LEN:
        return None, False
    if payload_len == 0:
        return "", False
    # Read flags byte
    flags_data = stdout.read(1)
    if len(flags_data) < 1:
        return None, False
    flags = flags_data[0]
    segment_final = bool(flags & RESP_FLAG_SEGMENT_FINAL)
    # Read text
    text_len = payload_len - 1
    if text_len == 0:
        return "", segment_final
    text_data = stdout.read(text_len)
    if len(text_data) < text_len:
        return None, False
    return text_data.decode("utf-8", errors="replace"), segment_final


# ── HTTP Handler ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            content = HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif path == "/models":
            self._json_response(200, {
                "models": {
                    k: {"size_mb": v["size_mb"], "arch": v["arch"]}
                    for k, v in MODELS.items()
                },
                "default": DEFAULT_MODEL,
                "ws_available": STREAM_SERVER.exists(),
            })
        elif path == "/ws":
            self._handle_ws(parsed)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/transcribe":
            self._handle_transcribe()
        elif self.path == "/stream":
            self._handle_stream()
        else:
            self.send_error(404)

    def _read_multipart(self):
        """Read and parse multipart body. Returns (fields, error_msg)."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            return None, "Expected multipart/form-data"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        fields = _parse_multipart(content_type, body)
        if not fields:
            return None, "Failed to parse multipart body"
        return fields, None

    def _prepare_audio(self, fields):
        """Extract audio from fields, convert to WAV. Returns (wav_path, tmpdir, model_name, error_msg)."""
        audio_data = fields.get("audio")
        if audio_data is None:
            return None, None, None, "No audio file uploaded"

        model_name = fields.get("model", b"").decode().strip() or DEFAULT_MODEL
        if model_name not in MODELS:
            return None, None, None, f"Unknown model: {model_name}"

        filename = fields.get("audio_filename", b"audio.bin").decode()
        tmpdir = tempfile.mkdtemp()
        raw_path = os.path.join(tmpdir, filename)
        wav_path = os.path.join(tmpdir, "input.wav")

        with open(raw_path, "wb") as f:
            f.write(audio_data)

        if not prepare_wav(raw_path, wav_path):
            return None, tmpdir, None, "ffmpeg conversion failed. Install ffmpeg or upload a 16kHz mono WAV."

        return wav_path, tmpdir, model_name, None

    def _handle_transcribe(self):
        """Standard batch transcription (works with both v1 and v2 models)."""
        fields, err = self._read_multipart()
        if err:
            self._json_response(400, {"error": err})
            return

        wav_path, tmpdir, model_name, err = self._prepare_audio(fields)
        if err:
            self._json_response(400 if "Unknown model" in err or "No audio" in err else 500,
                                {"error": err})
            if tmpdir:
                _cleanup_tmpdir(tmpdir)
            return

        model_path = MODELS[model_name]["path"]

        try:
            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    [str(CLI), "-m", model_path, wav_path],
                    capture_output=True, text=True, timeout=30,
                )
            except subprocess.TimeoutExpired:
                self._json_response(500, {"error": "Transcription timed out"})
                return

            elapsed_ms = round((time.monotonic() - t0) * 1000)

            if proc.returncode != 0:
                self._json_response(500, {
                    "error": f"moonshine-cli failed: {proc.stderr.strip()}"
                })
                return

            text = proc.stdout.strip()
            arch = MODELS[model_name]["arch"]
            self._json_response(200, {"text": text, "time_ms": elapsed_ms, "arch": arch})
        finally:
            _cleanup_tmpdir(tmpdir)

    def _handle_stream(self):
        """Streaming transcription with chunked transfer encoding (v2 models only)."""
        fields, err = self._read_multipart()
        if err:
            self._json_response(400, {"error": err})
            return

        wav_path, tmpdir, model_name, err = self._prepare_audio(fields)
        if err:
            self._json_response(400 if "Unknown model" in err or "No audio" in err else 500,
                                {"error": err})
            if tmpdir:
                _cleanup_tmpdir(tmpdir)
            return

        model_info = MODELS[model_name]
        if model_info["arch"] != "v2":
            self._json_response(400, {
                "error": f"Streaming requires a v2 model. '{model_name}' is {model_info['arch']}."
            })
            _cleanup_tmpdir(tmpdir)
            return

        model_path = model_info["path"]

        try:
            proc = subprocess.Popen(
                [str(CLI), "-m", model_path, "--stream", "--chunk-sec", "0.5", wav_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            buf = ""
            while True:
                ch = proc.stdout.read(1)
                if not ch:
                    break
                buf += ch
                if ch == "\n" or len(buf) >= 80:
                    event = f"data: {json.dumps({'text': buf})}\n\n"
                    self._send_chunk(event)
                    buf = ""

            if buf:
                event = f"data: {json.dumps({'text': buf})}\n\n"
                self._send_chunk(event)

            self._send_chunk("data: [DONE]\n\n")
            self._send_chunk("")

            proc.wait(timeout=5)
        except Exception as e:
            try:
                event = f"data: {json.dumps({'error': str(e)})}\n\n"
                self._send_chunk(event)
                self._send_chunk("")
            except Exception:
                pass
        finally:
            _cleanup_tmpdir(tmpdir)

    def _handle_ws(self, parsed):
        """WebSocket streaming: relay audio between browser and moonshine-stream-server."""
        upgrade = self.headers.get("Upgrade", "").lower()
        ws_key = self.headers.get("Sec-WebSocket-Key", "")
        if upgrade != "websocket" or not ws_key:
            self.send_error(400, "Expected WebSocket upgrade")
            return

        if not STREAM_SERVER.exists():
            self.send_error(503, "moonshine-stream-server not built")
            return

        params = parse_qs(parsed.query)
        model_name = params.get("model", [DEFAULT_MODEL])[0] if params.get("model") else DEFAULT_MODEL
        if model_name not in MODELS:
            self.send_error(400, f"Unknown model: {model_name}")
            return

        model_info = MODELS[model_name]
        if model_info["arch"] != "v2":
            self.send_error(400, "WebSocket streaming requires a v2 model")
            return

        # Write handshake directly to socket (BaseHTTPRequestHandler's response
        # machinery can interfere with the subsequent WebSocket binary protocol)
        accept = ws_accept_key(ws_key)
        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        ).encode("ascii")
        sock = self.connection
        sock.sendall(handshake)
        self.close_connection = True

        proc = subprocess.Popen(
            [str(STREAM_SERVER), "-m", model_info["path"]],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )

        try:
            ready_line = proc.stdout.readline()
            if not ready_line.startswith(b"READY"):
                ws_send_text(sock, json.dumps({"error": "Stream server failed to start"}))
                ws_send_close(sock)
                proc.kill()
                return

            ws_send_text(sock, json.dumps({"status": "ready"}))

            while True:
                opcode, payload = ws_read_frame(sock)

                if opcode is None or opcode == WS_OP_CLOSE:
                    break

                if opcode == WS_OP_PING:
                    ws_send_frame(sock, WS_OP_PONG, payload)
                    continue

                if opcode == WS_OP_BINARY:
                    try:
                        header = struct.pack("<II", MSG_AUDIO, len(payload))
                        proc.stdin.write(header + payload)
                        proc.stdin.flush()
                    except BrokenPipeError:
                        ws_send_text(sock, json.dumps({"error": "Stream server crashed"}))
                        break

                    text, segment_final = _read_subprocess_response(proc.stdout)
                    if text is None:
                        break
                    msg_out = {"text": text}
                    if segment_final:
                        msg_out["segment_final"] = True
                    ws_send_text(sock, json.dumps(msg_out))

                elif opcode == WS_OP_TEXT:
                    try:
                        msg = json.loads(payload.decode("utf-8"))
                        if msg.get("action") == "finalize":
                            header = struct.pack("<II", MSG_FINALIZE, 0)
                            proc.stdin.write(header)
                            proc.stdin.flush()
                            text, _ = _read_subprocess_response(proc.stdout)
                            if text is not None:
                                ws_send_text(sock, json.dumps({"text": text, "final": True}))

                        elif msg.get("action") == "reset":
                            header = struct.pack("<II", MSG_RESET, 0)
                            proc.stdin.write(header)
                            proc.stdin.flush()
                            _read_subprocess_response(proc.stdout)  # discard ack

                    except (json.JSONDecodeError, UnicodeDecodeError, BrokenPipeError):
                        pass

        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception:
            traceback.print_exc()
        finally:
            try:
                header = struct.pack("<II", MSG_FINALIZE, 0)
                proc.stdin.write(header)
                proc.stdin.flush()
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            try:
                ws_send_close(sock)
            except Exception:
                pass

    def _send_chunk(self, data: str):
        """Send a chunk in chunked transfer encoding."""
        encoded = data.encode("utf-8")
        self.wfile.write(f"{len(encoded):x}\r\n".encode())
        self.wfile.write(encoded)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        sys.stderr.write(f"[web] {format % args}\n")


def _cleanup_tmpdir(tmpdir: str):
    """Remove temporary directory and contents."""
    import shutil
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass


def main():
    if not CLI.exists():
        sys.exit(f"Error: CLI not found at {CLI}\nRun: cmake -B build && cmake --build build")
    if not MODELS:
        sys.exit(f"Error: No .gguf models found in {MODELS_DIR}")

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedServer(("", PORT), Handler)
    print(f"moonshine.cpp web demo")
    print(f"  Models:")
    for name, info in MODELS.items():
        print(f"    {name} ({info['size_mb']} MB, {info['arch']})")
    ws_status = "available" if STREAM_SERVER.exists() else "not available (build moonshine-stream-server)"
    print(f"  WebSocket streaming: {ws_status}")
    print(f"  Server: http://localhost:{PORT}")
    print(f"  Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
