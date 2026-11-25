cmake_minimum_required(VERSION 3.20)
# The version tracks the ROCm/hip-ep release the server is built against.
project(amdgpu-server VERSION 0.5.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

message(STATUS "Building AMD GPU LLM Server (hipep/AMDGPU OGA backend)")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(NOT WIN32 AND NOT (UNIX AND NOT APPLE))
    message(FATAL_ERROR "amdgpu-server targets Windows and Linux (gfx1151) only")
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# AMD GPU package (from ROCm/hip-ep releases): the extracted hipep_<version>_<os>.zip.
#   Windows: bin/onnxruntime-genai.dll + EP/ROCm DLLs (no import lib, no headers)
#   Linux:   lib/libonnxruntime-genai.so + EP/ROCm .so ; bin/ JIT toolchain
# Provide the directory via the AMDGPU_PKG_ROOT env var (preferred) or
# -DAMDGPU_PKG_ROOT=...
# ---------------------------------------------------------------------------
if(DEFINED ENV{AMDGPU_PKG_ROOT})
    set(AMDGPU_PKG_ROOT "$ENV{AMDGPU_PKG_ROOT}" CACHE PATH "Path to extracted HIP EP package" FORCE)
endif()
if(NOT DEFINED AMDGPU_PKG_ROOT)
    message(FATAL_ERROR "AMDGPU_PKG_ROOT is not set. Point it at the extracted "
        "HIP EP package directory, e.g. -DAMDGPU_PKG_ROOT=/path/to/hipep_0.5.3_windows "
        "or set the AMDGPU_PKG_ROOT environment variable.")
endif()

# POST_BUILD commands run from the build directory, so a relative root would
# resolve against the wrong base. Normalise against the source dir up front.
get_filename_component(AMDGPU_PKG_ROOT "${AMDGPU_PKG_ROOT}" ABSOLUTE
                       BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# OGA version bundled with the HIP EP package. The package ships no headers, so
# the matching ones are vendored in third_party/onnxruntime-genai.
set(OGA_VERSION "0.19.0")
set(OGA_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-genai/include")

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
    find_library(OGA_LIB
        NAMES onnxruntime-genai
        PATHS "${AMDGPU_PKG_LIB}"
        REQUIRED
        NO_DEFAULT_PATH
    )
    message(STATUS "AMD GPU package lib: ${AMDGPU_PKG_LIB}")
    message(STATUS "ONNX Runtime GenAI library: ${OGA_LIB}")
endif()

# Header-only deps
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

if(WIN32)
    target_compile_definitions(amdgpu-server PRIVATE _WIN32_WINNT=0x0A00)
    target_link_libraries(amdgpu-server PRIVATE ws2_32 wbemuuid ole32 oleaut32 shell32)
    set_target_properties(amdgpu-server PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")

    # Stage the release layout: the package tree exactly as shipped, with the
    # server built into its bin/ so the loader resolves every DLL from the exe's
    # own directory. The tree is kept whole rather than flattened because that
    # is the layout the EP is built and tested in, so every relative lookup
    # inside it stays valid.
    set(AMDGPU_STAGE_DIR "${CMAKE_BINARY_DIR}/package/$<CONFIG>")
    set_target_properties(amdgpu-server PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${AMDGPU_STAGE_DIR}/bin"
    )
    add_custom_command(TARGET amdgpu-server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${AMDGPU_PKG_ROOT}" "${AMDGPU_STAGE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
            "${CMAKE_CURRENT_SOURCE_DIR}/AMD_LICENSE"
            "${AMDGPU_STAGE_DIR}"
        COMMENT "Staging the HIP EP package tree around amdgpu-server.exe"
    )
else()
    # Linux: resolve the package .so via $ORIGIN so the binary is relocatable.
    # Use DT_RPATH (not DT_RUNPATH) so the path also applies to transitive deps.
    target_link_libraries(amdgpu-server PRIVATE ${CMAKE_DL_LIBS})
    target_link_options(amdgpu-server PRIVATE "-Wl,--disable-new-dtags")
    set_target_properties(amdgpu-server PROPERTIES
        BUILD_RPATH "$ORIGIN/lib:$ORIGIN"
        INSTALL_RPATH "$ORIGIN/lib:$ORIGIN"
    )

    # Mirror the package's own bin/ + lib/ layout next to the binary instead of
    # flattening both into one directory. Flattening collides (bin/clang is an
    # executable, lib/clang/ is a directory) and would break the JIT toolchain
    # regardless: clang resolves its resource headers as <exedir>/../lib/clang/<ver>.
    # Preserving the package's internal structure keeps every relative lookup
    # inside it valid, whatever those lookups turn out to be.
    #
    # cp -a preserves the SONAME symlink chain that individual copies lose, and is
    # invoked directly rather than through a shell -- the minimal Linux runner
    # image has no bash. Stale trees are removed first so re-runs do not nest.
    add_custom_command(TARGET amdgpu-server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf
            "$<TARGET_FILE_DIR:amdgpu-server>/lib" "$<TARGET_FILE_DIR:amdgpu-server>/bin"
        COMMAND cp -a "${AMDGPU_PKG_ROOT}/lib" "${AMDGPU_PKG_ROOT}/bin"
            "$<TARGET_FILE_DIR:amdgpu-server>/"
        COMMENT "Copying AMD GPU package runtime (lib/ + bin/) next to amdgpu-server"
        VERBATIM
    )

    add_custom_command(TARGET amdgpu-server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/AMD_LICENSE
            $<TARGET_FILE_DIR:amdgpu-server>/AMD_LICENSE
        COMMENT "Copying AMD_LICENSE"
    )
endif()

if(MSVC)
    target_compile_options(amdgpu-server PRIVATE /W4)
endif()

install(TARGETS amdgpu-server RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# Tests. tool_calls.cpp has no OGA dependency, so the tool-call suite builds and
# runs on any machine -- no GPU package and no AMD hardware required.
# ---------------------------------------------------------------------------
option(BUILD_TESTING "Build the test suite" ON)
if(BUILD_TESTING)
    enable_testing()

    add_executable(test_tool_calls
        test/test_tool_calls.cpp
        src/tool_calls.cpp
    )
    target_include_directories(test_tool_calls PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${JSON_DIR}
    )
    if(MSVC)
        target_compile_options(test_tool_calls PRIVATE /W4)
    endif()

    add_test(NAME ToolCalls COMMAND test_tool_calls)
endif()

message(STATUS "==========================================================")
message(STATUS "AMD GPU LLM Server Configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Package root:      ${AMDGPU_PKG_ROOT}")
message(STATUS "  OGA version:       ${OGA_VERSION}")
if(WIN32)
    message(STATUS "  Output directory:  ${CMAKE_BINARY_DIR}/package/<config>/bin")
else()
    message(STATUS "  Output directory:  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()
message(STATUS "==========================================================")
