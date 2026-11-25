# ONNX Runtime GenAI headers

`include/ort_genai.h` and `include/ort_genai_c.h` are the unmodified C and C++
API headers of ONNX Runtime GenAI (OGA) **0.19.0**, the version bundled with
[ROCm/hip-ep v0.5.3](https://github.com/ROCm/hip-ep/releases/tag/v0.5.3). The
HIP EP package ships `onnxruntime-genai.dll` without headers, so they live here.

- Source: NuGet package
  [`Microsoft.ML.OnnxRuntimeGenAI` 0.19.0](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntimeGenAI/0.19.0),
  directory `build/native/include/`
- Upstream commit: `d01b2441a72d4e3223415b9b08474c6e65fbc028` (`rel-0.19.0`)
- License: MIT, see [`LICENSE`](LICENSE)

## Updating

When the pinned HIP EP release moves to another OGA version, replace both
headers with the ones from the matching NuGet package and update `OGA_VERSION`
in the top-level `CMakeLists.txt`. The C header and the DLL agree when every
`OGA_API_CALL` function the header declares is exported by the package's
`bin/onnxruntime-genai.dll` (`dumpbin /exports`). For 0.19.0 both list the same
241 functions.
