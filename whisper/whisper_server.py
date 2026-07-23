#!/usr/bin/env python
#
# Minimal HTTP wrapper around whisper_demo.py so Lemonade can drive the AMD GPU
# Whisper pipeline as a wrapped-server subprocess.
#
# Endpoints:
#   GET  /health      -> {"status":"ok", ...}
#   POST /inference   -> raw audio bytes in the request body (wav/flac/etc.),
#                        returns {"text": "..."}. The OpenAI-shaped response
#                        {"text": ...} is assembled by Lemonade's backend.
#
# The encoder/decoder ORT sessions are built once at startup and kept warm.
#
# DEPENDENCY: this imports `whisper_demo` (whisper_demo.py) and runs inside the
# Python environment from the AMD GPU package (onnxruntime + AMDGPU/hipep EP,
# transformers, soundfile). whisper_demo.py is shipped by AMD's gpu-test-package
# whisper demo and is NOT included here — place it (and set up the Python env)
# next to this file. See the repository README ("Whisper transcription server").

import argparse
import io
import json
import os
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf
from transformers import WhisperFeatureExtractor, WhisperTokenizer

# Make lib/ (hipdnn_backend.dll) discoverable before importing the demo, which
# triggers the EP/runtime path setup.
_HERE = pathlib.Path(__file__).resolve().parent
_LIB = _HERE / "lib"
if _LIB.is_dir():
    os.environ["PATH"] = os.pathsep.join([str(_LIB), os.environ.get("PATH", "")])
    try:
        os.add_dll_directory(str(_LIB))
    except OSError:
        pass

from whisper_demo import (  # noqa: E402
    greedy_decode_morphizen,
    load_variant,
    make_session_factory,
    setup_runtime_env,
)

_LOCK = threading.Lock()
# Tokenizer + feature extractor are loaded once at startup (not per request).
_STATE = {"factory": None, "variant": None, "model_dir": None,
          "tokenizer": None, "fe": None, "ready": False}


def features_from_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded audio bytes -> Whisper log-mel features (16 kHz mono)."""
    audio, sr = sf.read(io.BytesIO(data))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        # Prototype-quality linear resample to 16 kHz.
        n_out = int(round(len(audio) * 16000 / sr))
        if n_out > 0:
            x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
    out = _STATE["fe"](audio, sampling_rate=16000, return_tensors="np")
    return out["input_features"].astype(np.float32)


def transcribe(data: bytes, max_length: int = 200) -> dict:
    with _LOCK:
        factory = _STATE["factory"]
        variant = _STATE["variant"]
        tokenizer = _STATE["tokenizer"]
        audio_fp = features_from_bytes(data)
        timings = {}
        t0 = time.perf_counter()
        tokens = greedy_decode_morphizen(
            factory, audio_fp, variant, max_length=max_length, timings=timings
        )
        wall_s = time.perf_counter() - t0
        body = []
        for tok in tokens:
            if tok == variant.eot:
                break
            body.append(tok)
        text = tokenizer.decode(body, skip_special_tokens=True).strip()
    return {"text": text, "wall_s": wall_s, "timings": timings}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # quiet default logging

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._json(200, {
                "status": "ok" if _STATE["ready"] else "loading",
                "model": pathlib.Path(_STATE["model_dir"]).name if _STATE["model_dir"] else None,
                "device": "gpu",
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/inference":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length) if length > 0 else b""
            if not data:
                self._json(400, {"error": "empty audio body"})
                return
            result = transcribe(data)
            self._json(200, {"text": result["text"]})
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": {"message": str(e), "type": "transcription_error"}})


def main() -> int:
    ap = argparse.ArgumentParser(description="Whisper AMD GPU transcription server")
    ap.add_argument("-m", "--model-path", type=pathlib.Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    model_dir = args.model_path
    if not model_dir.exists():
        raise FileNotFoundError(f"model dir not found: {model_dir}")

    print(f"[whisper-server] loading {model_dir} ...", flush=True)
    ep_dir = setup_runtime_env()
    variant = load_variant(model_dir)

    # Wrap the session factory in a cache so each ONNX session (encoder /
    # prefill / decode) is compiled once and reused across requests, instead of
    # rebuilding + recompiling GPU kernels on every transcription.
    base_factory = make_session_factory(model_dir, ep_dir)
    _sessions = {}

    def factory(name):
        if name not in _sessions:
            s = base_factory(name)
            try:
                print(f"[whisper-server] session {name} providers={s.get_providers()}", flush=True)
            except Exception:
                pass
            _sessions[name] = s
        return _sessions[name]

    tokenizer = WhisperTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    fe = WhisperFeatureExtractor(feature_size=variant.n_mels, sampling_rate=16000)
    _STATE.update(factory=factory, variant=variant, model_dir=str(model_dir),
                  tokenizer=tokenizer, fe=fe)

    # Warm up (compile GPU kernels) with a short silent clip so the first real
    # request is fast and /health flips to ok only once the model truly works.
    warm = np.zeros(16000, dtype=np.float32)
    warm_fp = fe(warm, sampling_rate=16000, return_tensors="np")["input_features"].astype(np.float32)
    greedy_decode_morphizen(factory, warm_fp, variant, max_length=8)

    _STATE["ready"] = True
    print(f"[whisper-server] ready on {args.host}:{args.port}", flush=True)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
