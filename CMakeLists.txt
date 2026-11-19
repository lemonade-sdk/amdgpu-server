cmake_minimum_required(VERSION 3.20)
project(amdgpu-server VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directories
set(AMDGPU_STAGE_DIR "${CMAKE_BINARY_DIR}/package/$<CONFIG>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${AMDGPU_STAGE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

message(STATUS "Building AMD GPU LLM Server (hipep/AMDGPU OGA backend)")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(NOT WIN32 AND NOT (UNIX AND NOT APPLE))
    message(FATAL_ERROR "amdgpu-server targets Windows and Linux (gfx1151) only")
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# AMD GPU package (from ROCm/hip-ep releases): the extracted gpu-test-package.
#   Windows: bin/onnxruntime-genai.dll + EP/ROCm DLLs (no import lib, no headers)
#   Linux:   lib/libonnxruntime-genai.so + EP/ROCm .so ; bin/ JIT toolchain
# Provide the directory via the AMDGPU_PKG_ROOT env var (preferred) or
# -DAMDGPU_PKG_ROOT=...
# ---------------------------------------------------------------------------
if(NOT DEFINED AMDGPU_PKG_ROOT AND DEFINED ENV{AMDGPU_PKG_ROOT})
    set(AMDGPU_PKG_ROOT "$ENV{AMDGPU_PKG_ROOT}" CACHE PATH "Path to extracted HIP EP package")
endif()
if(NOT DEFINED AMDGPU_PKG_ROOT)
    message(FATAL_ERROR "AMDGPU_PKG_ROOT is not set. Point it at the extracted "
        "gpu-test-package directory, e.g. -DAMDGPU_PKG_ROOT=/path/to/gpu-test-package "
        "or set the AMDGPU_PKG_ROOT environment variable.")
endif()

get_filename_component(AMDGPU_PKG_ROOT "${AMDGPU_PKG_ROOT}" ABSOLUTE)
set(HIP_EP_VERSION "0.5.3")
set(ORT_VERSION "1.31.0")
set(OGA_VERSION "0.19.0")
set(OGA_RUNTIME_PATCH "2875")

# ---------------------------------------------------------------------------
# Locate / prepare the ONNX Runtime GenAI import target (OGA_LIB).
#   Windows: the package ships only the DLL, so synthesize an import lib.
#   Linux:   link libonnxruntime-genai.so directly.
# ---------------------------------------------------------------------------
if(WIN32)
    set(AMDGPU_PKG_BIN "${AMDGPU_PKG_ROOT}/bin")
    set(OGA_DLL "${AMDGPU_PKG_BIN}/onnxruntime-genai.dll")
    if(NOT EXISTS "${OGA_DLL}")
        message(FATAL_ERROR "onnxruntime-genai.dll not found at ${OGA_DLL}. Check AMDGPU_PKG_ROOT.")
    endif()
    message(STATUS "AMD GPU package bin: ${AMDGPU_PKG_BIN}")

    # Locate MSVC dumpbin.exe / lib.exe next to the active compiler.
    get_filename_component(MSVC_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(DUMPBIN_EXE "${MSVC_BIN_DIR}/dumpbin.exe")
    set(LIB_EXE "${MSVC_BIN_DIR}/lib.exe")
    if(NOT EXISTS "${DUMPBIN_EXE}" OR NOT EXISTS "${LIB_EXE}")
        message(FATAL_ERROR "dumpbin.exe / lib.exe not found next to ${CMAKE_CXX_COMPILER}")
    endif()

    # Synthesize onnxruntime-genai.lib from the DLL exports (configure time).
    set(OGA_IMPORT_DIR "${CMAKE_BINARY_DIR}/oga_import")
    set(OGA_LIB "${OGA_IMPORT_DIR}/onnxruntime-genai.lib")
    execute_process(
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass
                -File "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_oga_import_lib.ps1"
                -Dll "${OGA_DLL}"
                -OutDir "${OGA_IMPORT_DIR}"
                -Dumpbin "${DUMPBIN_EXE}"
                -Lib "${LIB_EXE}"
        RESULT_VARIABLE OGA_IMPORT_RC
        OUTPUT_VARIABLE OGA_IMPORT_OUT
        ERROR_VARIABLE OGA_IMPORT_ERR
    )
    if(NOT OGA_IMPORT_RC EQUAL 0 OR NOT EXISTS "${OGA_LIB}")
        message(FATAL_ERROR "Failed to generate OGA import lib:\n${OGA_IMPORT_OUT}\n${OGA_IMPORT_ERR}")
    endif()
    message(STATUS "Generated OGA import lib: ${OGA_LIB}")
else()
    set(AMDGPU_PKG_LIB "${AMDGPU_PKG_ROOT}/lib")
    unset(OGA_LIB CACHE)
    find_library(OGA_LIB
        NAMES onnxruntime-genai
        PATHS "${AMDGPU_PKG_LIB}"
        REQUIRED
        NO_DEFAULT_PATH
    )
    message(STATUS "AMD GPU package lib: ${AMDGPU_PKG_LIB}")
    message(STATUS "ONNX Runtime GenAI library: ${OGA_LIB}")
endif()

# Vendor OGA C/C++ headers matching the package version (not shipped in it).
set(OGA_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_deps/oga-${OGA_VERSION}/include"
    CACHE PATH "OGA headers matching HIP EP ${HIP_EP_VERSION}, including any public API changes from PR ${OGA_RUNTIME_PATCH}")
foreach(HDR ort_genai.h ort_genai_c.h)
    if(NOT EXISTS "${OGA_INCLUDE_DIR}/${HDR}")
        message(STATUS "Downloading ${HDR} (v${OGA_VERSION})...")
        file(DOWNLOAD
            "https://raw.githubusercontent.com/microsoft/onnxruntime-genai/v${OGA_VERSION}/src/${HDR}"
            "${OGA_INCLUDE_DIR}/${HDR}"
            SHOW_PROGRESS
            TLS_VERIFY ON
            STATUS HDR_DOWNLOAD_STATUS
        )
        list(GET HDR_DOWNLOAD_STATUS 0 HDR_DOWNLOAD_CODE)
        if(NOT HDR_DOWNLOAD_CODE EQUAL 0)
            file(REMOVE "${OGA_INCLUDE_DIR}/${HDR}")
            message(FATAL_ERROR "Failed to download OGA ${OGA_VERSION} ${HDR}: ${HDR_DOWNLOAD_STATUS}. "
                "Use -DOGA_INCLUDE_DIR=/path/to/matching/headers for an offline build.")
        endif()
    endif()
endforeach()

file(SHA256 "${OGA_INCLUDE_DIR}/ort_genai.h" OGA_CXX_HEADER_SHA256)
file(SHA256 "${OGA_INCLUDE_DIR}/ort_genai_c.h" OGA_C_HEADER_SHA256)
set(OGA_HEADER_VERSION "unknown")
set(OGA_HEADER_SOURCE "user-provided snapshot")
if(OGA_CXX_HEADER_SHA256 STREQUAL "34062055a4def20bd0ce1316c3bea1e207bc5a959abe425a30e4494a7d7b72fb"
        AND OGA_C_HEADER_SHA256 STREQUAL "75e2681365e87f0f40f7098d28d9b3e65a8a62f306b270cecc338e8caa8b9a70")
    set(OGA_HEADER_VERSION "0.14.0")
    set(OGA_HEADER_SOURCE "recovered v0.14.0 snapshot, amd_runtime_server commit 580ba9f651fdf4d5f432aee9f76984d86faee6de")
elseif(OGA_INCLUDE_DIR STREQUAL "${CMAKE_BINARY_DIR}/_deps/oga-${OGA_VERSION}/include")
    set(OGA_HEADER_VERSION "${OGA_VERSION}")
    set(OGA_HEADER_SOURCE "microsoft/onnxruntime-genai v${OGA_VERSION}")
endif()
message(STATUS "OGA header snapshot: ${OGA_HEADER_VERSION}; compatibility with runtime PR ${OGA_RUNTIME_PATCH} is unverified")

# Header-only deps
set(HTTPLIB_DIR "${CMAKE_BINARY_DIR}/_deps/cpp-httplib-0.26.0")
if(NOT EXISTS "${HTTPLIB_DIR}/httplib.h")
    message(STATUS "Downloading cpp-httplib...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.26.0/httplib.h"
        "${HTTPLIB_DIR}/httplib.h"
        EXPECTED_HASH SHA256=4a42da28ce477d06f53942c172197d3b83b4fe42b7e37b8af78c70e063c06da7
        SHOW_PROGRESS
        TLS_VERIFY ON
        STATUS HTTPLIB_DOWNLOAD_STATUS
    )
    list(GET HTTPLIB_DOWNLOAD_STATUS 0 HTTPLIB_DOWNLOAD_CODE)
    if(NOT HTTPLIB_DOWNLOAD_CODE EQUAL 0)
        file(REMOVE "${HTTPLIB_DIR}/httplib.h")
        message(FATAL_ERROR "Failed to download cpp-httplib: ${HTTPLIB_DOWNLOAD_STATUS}")
    endif()
endif()

file(SHA256 "${HTTPLIB_DIR}/httplib.h" HTTPLIB_SHA256)
if(NOT HTTPLIB_SHA256 STREQUAL "4a42da28ce477d06f53942c172197d3b83b4fe42b7e37b8af78c70e063c06da7")
    message(FATAL_ERROR "cpp-httplib v0.26.0 header SHA256 mismatch")
endif()

set(JSON_DIR "${CMAKE_BINARY_DIR}/_deps/json-3.11.3")
if(NOT EXISTS "${JSON_DIR}/json.hpp")
    message(STATUS "Downloading nlohmann/json...")
    file(DOWNLOAD
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
        "${JSON_DIR}/json.hpp"
        EXPECTED_HASH SHA256=9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6
        SHOW_PROGRESS
        TLS_VERIFY ON
        STATUS JSON_DOWNLOAD_STATUS
    )
    list(GET JSON_DOWNLOAD_STATUS 0 JSON_DOWNLOAD_CODE)
    if(NOT JSON_DOWNLOAD_CODE EQUAL 0)
        file(REMOVE "${JSON_DIR}/json.hpp")
        message(FATAL_ERROR "Failed to download nlohmann/json: ${JSON_DOWNLOAD_STATUS}")
    endif()
endif()

file(SHA256 "${JSON_DIR}/json.hpp" JSON_SHA256)
if(NOT JSON_SHA256 STREQUAL "9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6")
    message(FATAL_ERROR "nlohmann/json v3.11.3 header SHA256 mismatch")
endif()

set(SOURCES
    src/main.cpp
    src/server.cpp
    src/inference_engine.cpp
    src/command_line.cpp
    src/types.cpp
    src/tool_calls.cpp
    src/reasoning.cpp
)

add_executable(amdgpu-server ${SOURCES})

target_include_directories(amdgpu-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OGA_INCLUDE_DIR}
    ${HTTPLIB_DIR}
    ${JSON_DIR}
)

target_link_libraries(amdgpu-server PRIVATE
    ${OGA_LIB}
    Threads::Threads
)

target_compile_definitions(amdgpu-server PRIVATE
    CPPHTTPLIB_THREAD_POOL_COUNT=8
    RYZENAI_SERVER_VERSION="${PROJECT_VERSION}"
)

if(WIN32)
    target_compile_definitions(amdgpu-server PRIVATE _WIN32_WINNT=0x0A00)
    target_link_libraries(amdgpu-server PRIVATE ws2_32 wbemuuid ole32 oleaut32 shell32)
    set_target_properties(amdgpu-server PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")

else()
    # Linux: resolve the package .so via $ORIGIN so the binary is relocatable.
    # Use DT_RPATH (not DT_RUNPATH) so the path also applies to transitive deps.
    target_link_libraries(amdgpu-server PRIVATE ${CMAKE_DL_LIBS})
    target_link_options(amdgpu-server PRIVATE "-Wl,--disable-new-dtags")
    set_target_properties(amdgpu-server PROPERTIES
        BUILD_RPATH "$ORIGIN/../lib"
        INSTALL_RPATH "$ORIGIN/../lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

if(MSVC)
    target_compile_options(amdgpu-server PRIVATE /W4)
endif()

configure_file(tools/runtime-versions.json.in runtime-versions.json @ONLY)
add_custom_target(stage_runtime ALL
    COMMAND "${CMAKE_COMMAND}"
        "-DPACKAGE_ROOT=${AMDGPU_PKG_ROOT}"
        "-DDESTINATION=${AMDGPU_STAGE_DIR}"
        "-DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DVERSION_FILE=${CMAKE_BINARY_DIR}/runtime-versions.json"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/stage_runtime.cmake"
    DEPENDS amdgpu-server
    COMMENT "Preserving HIP EP package directories, symlinks, permissions, and licenses"
    VERBATIM
)

install(DIRECTORY "${AMDGPU_PKG_ROOT}/" DESTINATION . USE_SOURCE_PERMISSIONS)
install(TARGETS amdgpu-server RUNTIME DESTINATION bin)
install(FILES LICENSE AMD_LICENSE "${CMAKE_BINARY_DIR}/runtime-versions.json" DESTINATION .)

message(STATUS "==========================================================")
message(STATUS "AMD GPU LLM Server Configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Package root:      ${AMDGPU_PKG_ROOT}")
message(STATUS "  OGA version:       ${OGA_VERSION}")
message(STATUS "  Output directory:  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "==========================================================")
