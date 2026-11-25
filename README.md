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
- **KV-cache reuse** across chat turns; `POST /v1/sessions/reset` starts a fresh
  conversation.

## How it works

The AMDGPU EP is selected entirely by the model. A model runs on the GPU when its
`genai_config.json` sets:

```json
"provider_options": [ { "AMDGPU": { "profile": "hip" } } ]
```

AMD publishes ready-to-run exports on Hugging Face as `amd/<model>_gpuep_*`; the
[hip-ep model list](https://github.com/ROCm/hip-ep/blob/main/docs/models.md)
describes them.

The server is a thin OGA wrapper: it loads the model with `OgaModel::Create`,
applies the chat template, and generates. For VLMs it uses `OgaMultiModalProcessor`
to fold decoded images into the prompt.

## Prerequisites

- **Windows 11 (x64)**
- **AMD GPU, gfx1151 (Strix Halo)**, with the AMD graphics driver installed
- **Visual Studio 2022 or 2026**, **CMake 3.20+**
- The **HIP EP package** — from the [ROCm/hip-ep releases](https://github.com/ROCm/hip-ep/releases):
  download `hipep_<version>_windows.zip` (currently `hipep_0.5.3_windows.zip`),
  extract it, and point `AMDGPU_PKG_ROOT` at the extracted directory (it must
  contain `bin/onnxruntime-genai.dll`). This package ships `onnxruntime-genai.dll`,
  the AMDGPU/hipep EP, and the ROCm runtime DLLs.

> The package ships the OGA DLL but **no import library and no OGA headers**, so
> the build (a) synthesizes `onnxruntime-genai.lib` from the DLL's exports via
> `tools/gen_oga_import_lib.ps1` (dumpbin → .def → lib.exe) and (b) uses the
> matching OGA C/C++ headers (v0.19.0) vendored in `third_party/onnxruntime-genai`.

Linux is not supported yet: the hip-ep Linux package has not been validated with
this server.

## Building

```powershell
# Point at the extracted HIP EP package (contains bin\onnxruntime-genai.dll)
$env:AMDGPU_PKG_ROOT = "C:\path\to\hipep_0.5.3_windows"
#   ... or pass -DAMDGPU_PKG_ROOT=C:\path\to\hipep_0.5.3_windows to cmake

cmake -S . -B build -G "Visual Studio 17 2022" -A x64      # or "Visual Studio 18 2026"
cmake --build build --config Release
```

Output: `build\package\Release\`, the HIP EP package tree exactly as shipped with
`amdgpu-server.exe` built into its `bin\` (so the loader resolves every DLL from
the exe's own directory). This directory is what a release archive contains.

Header-only deps (cpp-httplib, nlohmann/json) are downloaded into `external/` at
configure time.

## Running

```powershell
build\package\Release\bin\amdgpu-server.exe -m C:\path\to\onnx-model --port 8080 --verbose
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

Download a model into a plain directory, for example:

```powershell
hf download amd/Qwen2.5-14B-Instruct_gpuep_1.0_gpu_128 --local-dir C:\models\Qwen2.5-14B-Instruct
```

ONNX Runtime refuses weight files that resolve outside the model directory, so a
Hugging Face cache snapshot made of symlinks does not load.

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

## Testing

`ctest --test-dir build -C Release` runs the tool-call unit tests; they need no GPU.

`test/test_server.py` starts the server on a model and exercises every endpoint,
KV-cache reuse and, for vision models, image input. It needs a supported AMD GPU:

```powershell
pip install -r test\requirements.txt
python test\test_server.py --model C:\models\Qwen2.5-14B-Instruct
```

## Releases

The project version in `CMakeLists.txt` is the ROCm/hip-ep release the server is
built against, and every release is tagged `v<version>` (hip-ep v0.5.3 → `v0.5.3`).

Releases are built by CI (`.github/workflows/build_and_release.yml`): a Windows
runner downloads `hipep_<version>_windows.zip` from
[ROCm/hip-ep](https://github.com/ROCm/hip-ep/releases) (pinned by `HIP_EP_VERSION`
and its SHA-256), builds `amdgpu-server.exe`, and archives the staged tree as
`amdgpu-server-windows-x64.zip`. Pushing the `v<version>` tag publishes the GitHub
release "AMD GPU Server v\<version\>" with that archive attached. Lemonade resolves
it via `lemonade-sdk/amdgpu-server` → `amdgpu-server-windows-x64.zip` at the
version pinned in Lemonade's `backend_versions.json`.

To move to a newer hip-ep release:

1. Set `HIP_EP_VERSION` and `HIP_EP_WINDOWS_SHA256` in the workflow and the
   project version in `CMakeLists.txt`.
2. If the package bundles a different OGA version, update the vendored headers
   (see `third_party/onnxruntime-genai/README.md`).
3. Build, run `test/test_server.py` on gfx1151 hardware, then push the tag.

Note: CI validates compilation and packaging only; runtime is validated on
gfx1151 hardware.

## License

Source code is **MIT** (see `LICENSE`). `third_party/onnxruntime-genai` holds the
ONNX Runtime GenAI headers, which are MIT-licensed (see the `LICENSE` in that
directory).

**Release archive (`amdgpu-server-windows-x64.zip`):**

- `amdgpu-server.exe` and the header-only dependencies it is built from
  (**cpp-httplib**, **nlohmann/json**) are MIT licensed.
- The AMD GPU runtime files around it are licensed under the **AMD Software End
  User License Agreement**; `AMD_LICENSE` is at the root of the archive.

The runtime files are not part of this repository: the build copies them from the
HIP EP package.
