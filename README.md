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
"provider_options": [ { "AMDGPU": { "profile": "hip" } } ]
```

The server is a thin OGA wrapper: it loads the model with `OgaModel::Create`,
applies the chat template, and generates. For VLMs it uses `OgaMultiModalProcessor`
to fold decoded images into the prompt.

## Prerequisites

- **Windows 11 or Linux, x64**
- **AMD GPU, gfx1151 (Strix Halo)**, with the AMD graphics driver installed
  (provides `amdhip64_7.dll`)
- **Visual Studio 2022** on Windows, or a C++17 compiler on Linux, and **CMake 3.20+**
- The [HIP EP v0.5.3 runtime](https://github.com/ROCm/hip-ep/releases/tag/v0.5.3),
  with ONNX Runtime 1.31.0 and OGA 0.19.0 plus upstream PR 2875
- On Linux, **TheRock 10.1.0 for gfx1151** and a compatible AMD GPU driver.
  The upstream HIP EP Linux archive excludes TheRock libraries. Configure the
  host's library search path for that runtime before loading the server.

Download the archive for your platform, verify its SHA256, and extract it.
Point `AMDGPU_PKG_ROOT` at the directory containing `bin/` and `lib/`.

| Platform | HIP EP archive | SHA256 |
|----------|----------------|--------|
| Windows x64 | `hipep_0.5.3_windows.zip` | `ee1aadbf46142b1a4804fae531a6bc476966eca07e301f75310084f8e32f3c07` |
| Linux x64 | `hipep_0.5.3_linux.zip` | `d0f72eb9b477c5fa990f4142f65a7c959d54a32f38bd936b9f8aa7581849ec76` |

The Windows package contains the OGA DLL without its import library. The build
generates `onnxruntime-genai.lib` with `tools/gen_oga_import_lib.ps1` and the
MSVC tools. Both platform packages omit the OGA C and C++ headers.

## Building

The HIP EP 0.5.3 baseline is experimental. Windows builds and basic inference
work, but sequential chat requests with Qwen3.5 fail when the server partially
rewinds recurrent state. See [baseline results](docs/hip053-baseline-results.md)
and the [release test plan](docs/test-plan.md). Tag builds create draft
prereleases until the compatibility and application validation gates pass.

```powershell
# Point at the extracted AMD GPU package (contains bin\onnxruntime-genai.dll)
$env:AMDGPU_PKG_ROOT = "C:\path\to\hipep_0.5.3_windows"

cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

On Linux:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DAMDGPU_PKG_ROOT=/path/to/hipep_0.5.3_linux
cmake --build build -j"$(nproc)"
```

The server is at `build/package/Release/bin/amdgpu-server.exe` on Windows or
`build/package/Release/bin/amdgpu-server` on Linux. The complete upstream package
tree is at `build/package/Release/`. The build preserves runtime directories,
symlinks, compiler resources, and configuration files. Linux resolves bundled
libraries relative to `bin/` through `$ORIGIN/../lib`.

Header-only dependencies and the OGA headers download into versioned directories
under `build/_deps/` at configure time. Existing `external/oga` files are ignored.
For an offline build, provide matching OGA headers with
`-DOGA_INCLUDE_DIR=/path/to/headers`. The runtime uses OGA 0.19.0 with PR 2875.
Check that the header snapshot includes any public API changes from that patch.

As of November 19, 2026, the official OGA 0.19.0 header and PR 2875 URLs return
HTTP 404. A clean build requires a recovered matching header snapshot until
those sources are available. Older headers are not a verified substitute.
The manifest records both header SHA256 values, the identified snapshot version,
and its provenance. Header compatibility remains marked `unverified` until the
matching runtime patch can be inspected.

To stage an install, run `cmake --install build --config Release --prefix dist`.
Verify the package before distributing it:

```bash
python tools/verify_package.py --platform linux \
    --runtime-root /path/to/hipep_0.5.3_linux --package-root dist
```

For Windows, use `--platform windows`. The verifier compares every upstream file
and symlink with the staged package and checks executable permissions on Linux.

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

## Releases

The release workflow uses self-hosted Windows and Linux runners in `dev_lab`.
It verifies the pinned HIP EP archive hash, builds the server, checks the
complete runtime tree, then extracts and checks the distributable archive.
The smoke check runs `--help` and loads OGA. Hardware inference is a separate
acceptance check on gfx1151.

Pushing a `v*` tag publishes these assets with SHA256 sidecars:

| Platform | Server archive | Executable inside the archive |
|----------|----------------|-------------------------------|
| Windows x64 | `amdgpu-server-windows-x64.zip` | `bin/amdgpu-server.exe` |
| Linux x64 | `amdgpu-server-linux-x64.tar.gz` | `bin/amdgpu-server` |

Each archive preserves the upstream HIP EP package tree and includes
`runtime-versions.json`, `LICENSE`, and `AMD_LICENSE`. Lemonade's installer
discovers the executable recursively. Its backend release pin must use the
matching platform filename and archive hash.

## License

Source code is **MIT**, as stated in `LICENSE`. The packaged AMD GPU runtime is
subject to the **AMD Software End User License Agreement** in `AMD_LICENSE` and
the applicable licenses of its upstream components. The staging step retains
any license files present in the HIP EP archive.
