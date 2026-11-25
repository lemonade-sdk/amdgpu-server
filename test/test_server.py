#!/usr/bin/env python3
"""
AMD GPU Server Verification Tests

This script tests the amdgpu-server endpoints for basic functionality.
It starts the server, runs tests against all OpenAI-compatible endpoints,
and then stops the server.

Usage:
    python test_server.py --model <model directory or Hugging Face repo id>
"""

import argparse
import base64
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import unittest
import signal
import zlib
import requests
from pathlib import Path
import httpx

import openai


# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
MAX_NEW_TOKENS = 32
TEST_PROMPT = "What is the population of Paris?"

# Timeout for all API requests (in seconds)
REQUEST_TIMEOUT = 120

# Global reference to server log file for printing on failure
SERVER_LOG_FILE: Path = None


def get_model_path(model: str) -> str:
    """Resolve a model directory, or a Hugging Face repo id cached under HF_HOME."""
    if Path(model).is_dir():
        return str(Path(model).resolve())

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    # HuggingFace hub stores models with -- separator instead of /
    model_dir_name = f"models--{model.replace('/', '--')}"
    model_path = Path(hf_home) / "hub" / model_dir_name

    # Find the actual model files in the snapshots directory
    snapshots_dir = model_path / "snapshots"
    if snapshots_dir.exists():
        # Get the latest snapshot (there should be one)
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            return str(snapshots[0])

    # If no snapshots, return the model path directly
    return str(model_path)


def print_server_log():
    """Print the server log file contents for debugging."""
    global SERVER_LOG_FILE
    if SERVER_LOG_FILE and SERVER_LOG_FILE.exists():
        print(f"\n{'='*60}")
        print(f"  SERVER LOG ({SERVER_LOG_FILE})")
        print(f"{'='*60}")
        try:
            content = SERVER_LOG_FILE.read_text(encoding="utf-8", errors="replace")
            # Print last 200 lines to avoid overwhelming output
            lines = content.splitlines()
            if len(lines) > 200:
                print(f"... (showing last 200 of {len(lines)} lines) ...")
                lines = lines[-200:]
            print("\n".join(lines))
        except Exception as e:
            print(f"[Error] Could not read server log: {e}")
        print(f"{'='*60}\n")
    else:
        print("[Warning] No server log file available")


def solid_png(width: int = 64, height: int = 64, rgb=(220, 30, 30)) -> bytes:
    """Build a single-colour PNG without an imaging dependency."""
    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def iter_sse_events(response):
    """Yield the JSON events of an SSE response, reading it through to the end."""
    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        if line_str.startswith("data: ") and line_str[6:] != "[DONE]":
            yield json.loads(line_str[6:])


def chat(messages, max_tokens: int, stream: bool = False):
    """Run a deterministic chat completion and return (content, reasoning)."""
    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": "amdgpu",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": stream,
        },
        stream=stream,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        raise AssertionError(f"HTTP {response.status_code}: {response.text}")

    if not stream:
        message = response.json()["choices"][0]["message"]
        return message.get("content") or "", message.get("reasoning_content") or ""

    content = ""
    reasoning = ""
    for event in iter_sse_events(response):
        if "error" in event:
            raise AssertionError(f"Stream error: {event['error']}")
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
    return content, reasoning


def reset_session():
    """Drop the server's chat KV cache so the next request starts fresh."""
    response = requests.post(
        f"{SERVER_URL}/v1/sessions/reset", json={}, timeout=REQUEST_TIMEOUT
    )
    if response.status_code != 200:
        raise AssertionError(f"HTTP {response.status_code}: {response.text}")


class ServerProcess:
    """Manages the amdgpu-server process lifecycle."""

    def __init__(self, server_exe: str, model_path: str, log_dir: Path):
        self.server_exe = server_exe
        self.model_path = model_path
        self.process = None
        self.log_dir = log_dir
        self.log_file = log_dir / "server_gpu.log"
        self.log_handle = None

    def start(self, timeout: int = 300):
        """Start the server and wait for it to be ready."""
        global SERVER_LOG_FILE
        SERVER_LOG_FILE = self.log_file

        cmd = [
            self.server_exe,
            "-m",
            self.model_path,
            "--host",
            SERVER_HOST,
            "--port",
            str(SERVER_PORT),
        ]

        print(f"\n[Test] Starting server with command: {' '.join(cmd)}")
        print(f"[Test] Server log file: {self.log_file}")

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Open log file for writing
        self.log_handle = open(self.log_file, "w", encoding="utf-8")

        # Start the server process with output redirected to log file
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            ),
        )

        # Wait for server to be ready by polling health endpoint
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{SERVER_URL}/health", timeout=2)
                if response.status_code == 200:
                    print(
                        f"[Test] Server is ready (took {time.time() - start_time:.1f}s)"
                    )
                    return True
            except requests.exceptions.ConnectionError:
                pass
            except requests.exceptions.Timeout:
                pass

            # Check if process died
            if self.process.poll() is not None:
                self.log_handle.flush()
                print(f"[Test] Server process died!")
                print_server_log()
                raise RuntimeError("Server process terminated unexpectedly")

            time.sleep(1)

        # Timeout - print the log for debugging
        self.log_handle.flush()
        print_server_log()
        raise TimeoutError(f"Server did not become ready within {timeout} seconds")

    def stop(self):
        """Stop the server process."""
        if self.process is None:
            return

        print("[Test] Stopping server...")

        if sys.platform == "win32":
            # On Windows, send CTRL+BREAK to the process group
            self.process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            # On Unix, send SIGTERM
            self.process.terminate()

        try:
            self.process.wait(timeout=10)
            print("[Test] Server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("[Test] Server did not stop gracefully, killing...")
            self.process.kill()
            self.process.wait()

        self.process = None

        # Close log file handle
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None


class AMDGPUServerTests(unittest.TestCase):
    """Test cases for amdgpu-server endpoints."""

    client: openai.OpenAI = None
    multimodal: bool = False

    @classmethod
    def setUpClass(cls):
        """Set up the OpenAI client with timeout."""
        cls.client = openai.OpenAI(
            base_url=f"{SERVER_URL}/v1",
            api_key="not-needed",  # amdgpu-server doesn't require API key
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        )

    def test_01_health_endpoint(self):
        """Test that the /health endpoint returns OK status."""
        response = requests.get(f"{SERVER_URL}/health", timeout=REQUEST_TIMEOUT)

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")
        self.assertIn("model", data)
        self.assertEqual(data.get("execution_mode"), "gpu")

        print(
            f"[Test] Health check passed - model: {data['model']}, mode: {data['execution_mode']}"
        )

    def test_02_completions_non_streaming(self):
        """Test /v1/completions endpoint without streaming."""
        response = self.client.completions.create(
            model="amdgpu",
            prompt=TEST_PROMPT,
            max_tokens=MAX_NEW_TOKENS,
            stream=False,
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices), 0)

        text = response.choices[0].text
        self.assertIsNotNone(text)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text.strip()), 0, "Response text should not be empty")

        print(
            f"[Test] Completions (non-streaming) passed - response length: {len(text)} chars"
        )

    def test_03_completions_streaming(self):
        """Test /v1/completions endpoint with streaming."""
        stream = self.client.completions.create(
            model="amdgpu",
            prompt=TEST_PROMPT,
            max_tokens=MAX_NEW_TOKENS,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                text = chunk.choices[0].text
                if text:
                    chunks.append(text)

        full_response = "".join(chunks)
        self.assertGreater(len(chunks), 0, "Should receive at least one chunk")
        self.assertGreater(
            len(full_response.strip()), 0, "Combined response should not be empty"
        )

        print(
            f"[Test] Completions (streaming) passed - {len(chunks)} chunks, {len(full_response)} chars total"
        )

    def test_04_chat_completions_non_streaming(self):
        """Test /v1/chat/completions endpoint without streaming."""
        response = self.client.chat.completions.create(
            model="amdgpu",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_NEW_TOKENS,
            stream=False,
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices), 0)

        message = response.choices[0].message
        self.assertIsNotNone(message)
        self.assertIsNotNone(message.content)
        self.assertGreater(
            len(message.content.strip()), 0, "Response content should not be empty"
        )

        print(
            f"[Test] Chat completions (non-streaming) passed - response length: {len(message.content)} chars"
        )

    def test_05_chat_completions_streaming(self):
        """Test /v1/chat/completions endpoint with streaming."""
        stream = self.client.chat.completions.create(
            model="amdgpu",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_NEW_TOKENS,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    chunks.append(delta.content)

        full_response = "".join(chunks)
        self.assertGreater(len(chunks), 0, "Should receive at least one chunk")
        self.assertGreater(
            len(full_response.strip()), 0, "Combined response should not be empty"
        )

        print(
            f"[Test] Chat completions (streaming) passed - {len(chunks)} chunks, {len(full_response)} chars total"
        )

    def test_06_responses_non_streaming(self):
        """Test /v1/responses endpoint without streaming."""
        # The responses API is not directly supported by the openai library,
        # so we make a direct HTTP request
        response = requests.post(
            f"{SERVER_URL}/v1/responses",
            json={
                "model": "amdgpu",
                "input": TEST_PROMPT,
                "max_output_tokens": MAX_NEW_TOKENS,
                "stream": False,
            },
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("output", data)
        self.assertIsInstance(data["output"], list)
        self.assertGreater(len(data["output"]), 0)

        # Extract text from output
        output_item = data["output"][0]
        self.assertIn("content", output_item)
        self.assertGreater(len(output_item["content"]), 0)

        text_content = output_item["content"][0]
        self.assertIn("text", text_content)
        self.assertGreater(
            len(text_content["text"].strip()), 0, "Response text should not be empty"
        )

        print(
            f"[Test] Responses (non-streaming) passed - response length: {len(text_content['text'])} chars"
        )

    def test_07_responses_streaming(self):
        """Test /v1/responses endpoint with streaming."""
        response = requests.post(
            f"{SERVER_URL}/v1/responses",
            json={
                "model": "amdgpu",
                "input": TEST_PROMPT,
                "max_output_tokens": MAX_NEW_TOKENS,
                "stream": True,
            },
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=REQUEST_TIMEOUT,
        )

        self.assertEqual(response.status_code, 200)

        chunks = []
        completed_event_received = False

        # Reading to the end also checks that the server terminates the stream.
        for event in iter_sse_events(response):
            # Check for delta events
            if event.get("type") == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    chunks.append(delta)

            # Check for completed event
            if event.get("type") == "response.completed":
                completed_event_received = True

        full_response = "".join(chunks)
        self.assertGreater(len(chunks), 0, "Should receive at least one delta chunk")
        self.assertTrue(
            completed_event_received, "Should receive response.completed event"
        )
        self.assertGreater(
            len(full_response.strip()), 0, "Combined response should not be empty"
        )

        print(
            f"[Test] Responses (streaming) passed - {len(chunks)} chunks, {len(full_response)} chars total"
        )

    def test_08_cached_prompt_matches_fresh_session(self):
        """A prompt the KV cache already holds must answer like a fresh session."""
        messages = [{"role": "user", "content": "Reply with exactly one word: hello."}]

        reset_session()
        fresh = chat(messages, max_tokens=64)
        self.assertGreater(len("".join(fresh).strip()), 0, "Response should not be empty")

        self.assertEqual(chat(messages, max_tokens=64), fresh, "Repeated prompt")
        self.assertEqual(
            chat(messages, max_tokens=64, stream=True),
            fresh,
            "Repeated prompt, streamed",
        )

        reset_session()
        self.assertEqual(
            chat(messages, max_tokens=64, stream=True), fresh, "Fresh session, streamed"
        )

        print(f"[Test] Cached prompt passed - response: {fresh[0].strip()!r}")

    def test_09_unrelated_prompt_matches_fresh_session(self):
        """A prompt that shares no history with the KV cache must answer like a fresh session."""
        first = [{"role": "user", "content": "Reply with exactly this code: RED361."}]
        second = [{"role": "user", "content": "Name the capital of France in one word."}]

        reset_session()
        fresh = chat(second, max_tokens=64)
        self.assertGreater(len("".join(fresh).strip()), 0, "Response should not be empty")

        reset_session()
        chat(first, max_tokens=64)
        self.assertEqual(chat(second, max_tokens=64), fresh, "Prompt after another one")

        print(f"[Test] Unrelated prompt passed - response: {fresh[0].strip()!r}")

    def test_10_multi_turn_conversation(self):
        """A follow-up turn must see the earlier turns of the conversation."""
        reset_session()
        messages = [
            {"role": "user", "content": "Remember the code ORANGE731. Say OK."}
        ]
        content, _ = chat(messages, max_tokens=256)

        messages += [
            {"role": "assistant", "content": content},
            {"role": "user", "content": "What code did I give you?"},
        ]
        content, reasoning = chat(messages, max_tokens=256)
        self.assertIn("ORANGE731", content + reasoning)

        print("[Test] Multi-turn conversation passed")

    def test_11_image_input(self):
        """Test /v1/chat/completions with an image (vision models only)."""
        if not self.multimodal:
            self.skipTest("Model has no vision pipeline")

        image_url = "data:image/png;base64," + base64.b64encode(solid_png()).decode(
            "ascii"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour is this image? One word."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        answer = chat(messages, max_tokens=256)
        self.assertIn("red", "".join(answer).lower())
        self.assertEqual(
            chat(messages, max_tokens=256, stream=True), answer, "Image prompt, streamed"
        )

        print("[Test] Image input passed")


def find_server_executable() -> str:
    """Find the amdgpu-server executable."""
    # Check common locations
    possible_paths = [
        # Relative to test directory (CI build output)
        Path(__file__).parent.parent
        / "build"
        / "package"
        / "Release"
        / "bin"
        / "amdgpu-server.exe",
        # Extracted release archive in the current directory
        Path("bin") / "amdgpu-server.exe",
        # Current directory
        Path("amdgpu-server.exe"),
        # Build directory variations
        Path("build") / "package" / "Release" / "bin" / "amdgpu-server.exe",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path.resolve())

    # Check PATH
    exe_in_path = shutil.which("amdgpu-server") or shutil.which("amdgpu-server.exe")
    if exe_in_path:
        return exe_in_path

    raise FileNotFoundError(
        f"Could not find amdgpu-server executable. Searched:\n"
        + "\n".join(f"  - {p}" for p in possible_paths)
    )


def main():
    global SERVER_PORT, SERVER_URL

    parser = argparse.ArgumentParser(
        description="Run AMD GPU Server verification tests"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model directory, or a Hugging Face repo id already downloaded to $HF_HOME/hub",
    )
    parser.add_argument(
        "--server-exe",
        type=str,
        default=None,
        help="Path to amdgpu-server executable (auto-detected if not specified)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=SERVER_PORT,
        help=f"Port for the server under test (default: {SERVER_PORT})",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to store server logs (defaults to ./test_logs)",
    )

    args = parser.parse_args()

    SERVER_PORT = args.port
    SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

    print(f"\n{'='*60}")
    print(f"  AMD GPU Server Verification Tests")
    print(f"  Model: {args.model}")
    print(f"{'='*60}\n")

    # Find server executable
    if args.server_exe:
        server_exe = args.server_exe
    else:
        server_exe = find_server_executable()

    print(f"[Test] Server executable: {server_exe}")

    # Get model path
    model_path = get_model_path(args.model)
    print(f"[Test] Model path: {model_path}")

    # Verify model exists
    if not Path(model_path).exists():
        print(f"[Error] Model path does not exist: {model_path}")
        print(f"[Error] Please ensure the model is downloaded to $HF_HOME/hub")
        sys.exit(1)

    # Set up log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path(__file__).parent / "test_logs"

    print(f"[Test] Log directory: {log_dir}")

    # Start server
    server = ServerProcess(server_exe, model_path, log_dir)

    try:
        server.start()

        # Vision tests only apply to models that ship an image processor
        AMDGPUServerTests.multimodal = (
            Path(model_path) / "processor_config.json"
        ).exists()

        # Run tests
        print(f"\n{'='*60}")
        print(f"  Running Tests")
        print(f"{'='*60}\n")

        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(AMDGPUServerTests)

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # Return exit code based on test results
        if result.wasSuccessful():
            print(f"\n{'='*60}")
            print(f"  All tests PASSED for model: {args.model}")
            print(f"{'='*60}\n")
            return 0
        else:
            print(f"\n{'='*60}")
            print(f"  Some tests FAILED for model: {args.model}")
            print(f"{'='*60}\n")
            # Print server log on failure
            print_server_log()
            return 1

    except Exception as e:
        print(f"[Error] Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        # Print server log on exception
        print_server_log()
        return 1

    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
