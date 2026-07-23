# AMD GPU LLM Server (AMDGPU EP)

A lightweight, OpenAI API-compatible server for running LLMs and vision-language
models (VLMs) on **AMD GPUs** using ONNX Runtime GenAI (OGA) with the **hipep /
AMDGPU execution provider**. Validated on **gfx1151 (Strix Halo)**.

It is the GPU counterpart to [ryzenai-server](https://github.com/lemonade-sdk/ryzenai-server)
(NPU) and is consumed by [Lemonade](https://github.com/lemonade-sdk/lemonade) as
the **`amdgpu-llm`** wrapped-server backend. A companion Python
[whisper transcription server](#whisper-transcription-server) provides the
**`amdgpu-whisper`** speech-to-text backend.

## Features

- **OpenAI API compatible:** `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/health`
- **Vision (VLM):** OpenAI `image_url` content (base64 data URLs) → OGA multimodal processor
- **Streaming** (SSE) and non-streaming
- **Tool/function calling**
- **GPU execution** via the AMDGPU EP, selected by the model's `genai_config.json`
- **Context window** driven by `--ctx-size` and the model's native `context_length`

## How it works

The AMDGPU EP is selected entirely by the model. A model runs on the GPU when its
`genai_config.json` sets:

```json
"provider_options": [ { "AMDGPU": { "profile": "llm" } } ]
```

The server is a thin OGA wrapper: it loads the model with `OgaModel::Create`,
applies the chat template, and generates. For VLMs it uses `OgaMultiModalProcessor`
to fold decoded images into the prompt.

## Prerequisites

- **Windows 11 (x64)**
- **AMD GPU, gfx1151 (Strix Halo)**, with the AMD graphics driver installed
  (provides `amdhip64_7.dll`)
- **Visual Studio 2022 or 2026**, **CMake 3.20+**
- The **AMD GPU package** ("gpu-test-package") — ships `onnxruntime-genai.dll`,
  the AMDGPU/hipep EP, and the ROCm runtime DLLs. Point `AMDGPU_PKG_ROOT` at the
  extracted directory (it must contain `bin/onnxruntime-genai.dll`).

> The package ships the OGA DLL but **no import library and no OGA headers**, so
> the build (a) synthesizes `onnxruntime-genai.lib` from the DLL's exports via
> `tools/gen_oga_import_lib.ps1` (dumpbin → .def → lib.exe) and (b) vendors the
> matching upstream OGA C/C++ headers (v0.14.0) into `external/oga`.

## Building

```powershell
# Point at the extracted AMD GPU package (contains bin\onnxruntime-genai.dll)
$env:AMDGPU_PKG_ROOT = "C:\path\to\gpu-test-package"
#   ... or pass -DAMDGPU_PKG_ROOT=C:\path\to\gpu-test-package to cmake

cmake -S . -B build -G "Visual Studio 17 2022" -A x64      # or "Visual Studio 18 2026"
cmake --build build --config Release
```

Output: `build\bin\Release\amdgpu-server.exe`, with the package's EP/ROCm/OGA DLLs
copied next to it (so the loader resolves everything from the exe's own directory).

Header-only deps (cpp-httplib, nlohmann/json) and the OGA headers are downloaded
into `external/` at configure time.

## Running

```powershell
build\bin\Release\amdgpu-server.exe -m C:\path\to\onnx-model --port 8080 --verbose
```

| Flag | Meaning |
|------|---------|
| `-m, --model PATH` | ONNX model directory (with `genai_config.json`) |
| `-p, --port PORT` | Port (default 8080) |
| `--host HOST` | Bind address (default 127.0.0.1) |
| `-c, --ctx-size N` | Context window; `<=0` uses the model's native `context_length` |
| `-v, --verbose` | Verbose logging |

The model's `genai_config.json` must select the AMDGPU provider (see above);
models exported for DirectML ship `"provider_options": [{"dml": {}}]` — replace
that block.

### Vision (VLM) requests

Send OpenAI-style content arrays; base64 `image_url` data URLs are decoded and fed
through the multimodal processor:

```json
{
  "model": "ignored",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  ]}]
}
```

## Whisper transcription server

`whisper/whisper_server.py` is a small HTTP wrapper that runs the AMD GPU Whisper
ONNX pipeline (encoder + decoder-with-KV-cache) on the AMDGPU EP and exposes it to
Lemonade as the **`amdgpu-whisper`** backend.

- `GET /health` → `{"status": "ok", ...}`
- `POST /inference` → raw audio bytes in the body → `{"text": "..."}`

Unlike the C++ LLM server, Whisper stays in Python because it relies on Hugging
Face `transformers` (feature extractor + tokenizer) and a hand-rolled ORT
encode/prefill/decode loop. It runs inside a Python environment built from the AMD
package wheels (custom `onnxruntime` + the AMDGPU/hipep EP + `transformers` +
`soundfile`), and it **imports `whisper_demo.py`**, which is shipped by AMD's
gpu-test-package whisper demo and must be placed next to `whisper_server.py`
(not included here).

```powershell
# in a Python 3.14 env with the AMD wheels installed, and whisper_demo.py alongside:
python whisper/whisper_server.py -m C:\path\to\whisper-onnx-model --port 8081
```

## Integration with Lemonade

When run under [Lemonade](https://github.com/lemonade-sdk/lemonade), these servers
are managed as wrapped-server subprocesses:

- **`amdgpu-llm`** → `amdgpu-server.exe` (this repo, C++)
- **`amdgpu-whisper`** → `whisper/whisper_server.py` (Python)

Lemonade forwards OpenAI-compatible requests to the subprocess and routes by model
recipe. Both are `DEVICE_GPU` and can be loaded concurrently.

## License

Source code is **MIT** (see `LICENSE`). The AMD GPU runtime DLLs copied next to the
binary at build time are licensed under the **AMD Software End User License
Agreement** (see `AMD_LICENSE`); they are not part of this repository and are not
redistributed here.
