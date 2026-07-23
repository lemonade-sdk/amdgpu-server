cmake_minimum_required(VERSION 3.20)
project(amdgpu-server VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

message(STATUS "Building AMD GPU LLM Server (hipep/AMDGPU OGA backend)")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(NOT WIN32)
    message(FATAL_ERROR "amdgpu-server currently targets Windows (gfx1151) only")
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# AMD GPU (hipep) package: ships onnxruntime-genai.dll + EP/ROCm DLLs in bin/,
# but NO import lib and NO OGA headers. We synthesize the import lib from the
# DLL and vendor the matching upstream OGA headers below.
# Provide the extracted gpu-test-package directory via the AMDGPU_PKG_ROOT
# environment variable (preferred) or -DAMDGPU_PKG_ROOT=...
# ---------------------------------------------------------------------------
if(DEFINED ENV{AMDGPU_PKG_ROOT})
    set(AMDGPU_PKG_ROOT "$ENV{AMDGPU_PKG_ROOT}" CACHE PATH "Path to extracted gpu-test-package" FORCE)
endif()
if(NOT DEFINED AMDGPU_PKG_ROOT)
    message(FATAL_ERROR "AMDGPU_PKG_ROOT is not set. Point it at the extracted "
        "gpu-test-package directory, e.g. -DAMDGPU_PKG_ROOT=C:/path/to/gpu-test-package "
        "or set the AMDGPU_PKG_ROOT environment variable.")
endif()

set(AMDGPU_PKG_BIN "${AMDGPU_PKG_ROOT}/bin")
set(OGA_DLL "${AMDGPU_PKG_BIN}/onnxruntime-genai.dll")
if(NOT EXISTS "${OGA_DLL}")
    message(FATAL_ERROR "onnxruntime-genai.dll not found at ${OGA_DLL}. Set AMDGPU_PKG_ROOT to the extracted gpu-test-package directory.")
endif()
message(STATUS "AMD GPU package bin: ${AMDGPU_PKG_BIN}")

# Pinned OGA version shipped in this package (wheel: onnxruntime_genai_directml-0.14.0)
set(OGA_VERSION "0.14.0")

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

# Vendor OGA C/C++ headers matching the DLL version (not shipped in the package).
set(OGA_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/external/oga")
foreach(HDR ort_genai.h ort_genai_c.h)
    if(NOT EXISTS "${OGA_INCLUDE}/${HDR}")
        message(STATUS "Downloading ${HDR} (v${OGA_VERSION})...")
        file(DOWNLOAD
            "https://raw.githubusercontent.com/microsoft/onnxruntime-genai/v${OGA_VERSION}/src/${HDR}"
            "${OGA_INCLUDE}/${HDR}"
            SHOW_PROGRESS
            STATUS HDR_DOWNLOAD_STATUS
        )
        list(GET HDR_DOWNLOAD_STATUS 0 HDR_DOWNLOAD_CODE)
        if(NOT HDR_DOWNLOAD_CODE EQUAL 0)
            message(FATAL_ERROR "Failed to download ${HDR}: ${HDR_DOWNLOAD_STATUS}")
        endif()
    endif()
endforeach()

# Header-only deps (same as ryzenai-server)
include(FetchContent)

set(HTTPLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/cpp-httplib")
if(NOT EXISTS "${HTTPLIB_DIR}/httplib.h")
    message(STATUS "Downloading cpp-httplib...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.26.0/httplib.h"
        "${HTTPLIB_DIR}/httplib.h"
        SHOW_PROGRESS
    )
endif()

set(JSON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/json")
if(NOT EXISTS "${JSON_DIR}/json.hpp")
    message(STATUS "Downloading nlohmann/json...")
    file(DOWNLOAD
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
        "${JSON_DIR}/json.hpp"
        SHOW_PROGRESS
    )
endif()

# Source files (driver_check.cpp dropped: NPU-only, irrelevant for GPU)
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
    ${OGA_INCLUDE}
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

# Windows settings
target_compile_definitions(amdgpu-server PRIVATE _WIN32_WINNT=0x0A00)
target_link_libraries(amdgpu-server PRIVATE ws2_32 wbemuuid ole32 oleaut32 shell32)
set_target_properties(amdgpu-server PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")

# Copy the entire package bin/ (EP DLLs, ROCm runtime, rocblas/, hipblaslt/,
# hipdnn_plugins/) next to the exe so the loader resolves everything from its
# own directory. copy_directory_if_different avoids recopying large DLLs.
add_custom_command(TARGET amdgpu-server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        "${AMDGPU_PKG_BIN}" "$<TARGET_FILE_DIR:amdgpu-server>"
    COMMENT "Copying AMD GPU package runtime (DLLs + ROCm + EP) next to amdgpu-server.exe"
)

add_custom_command(TARGET amdgpu-server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/AMD_LICENSE
        $<TARGET_FILE_DIR:amdgpu-server>/AMD_LICENSE
    COMMENT "Copying AMD_LICENSE"
)

if(MSVC)
    target_compile_options(amdgpu-server PRIVATE /W4)
endif()

install(TARGETS amdgpu-server RUNTIME DESTINATION bin)

message(STATUS "==========================================================")
message(STATUS "AMD GPU LLM Server Configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Package root:      ${AMDGPU_PKG_ROOT}")
message(STATUS "  OGA version:       ${OGA_VERSION}")
message(STATUS "  Output directory:  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "==========================================================")
