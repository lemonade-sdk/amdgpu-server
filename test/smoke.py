#!/usr/bin/env python3
"""Run an isolated server and save API smoke evidence using only Python stdlib."""

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class Smoke:
    def __init__(self, args):
        self.args = args
        self.results = []
        self.exchanges = []

    def request(self, path, body=None, *, stream=False, cancel=False, expected=200):
        record = {"path": path, "request": body, "cancel_after_output": cancel}
        self.exchanges.append(record)
        payload = body if isinstance(body, str) else json.dumps(body)
        connection = http.client.HTTPConnection("127.0.0.1", self.args.port,
                                                timeout=self.args.request_timeout)
        deadline = time.monotonic() + self.args.request_timeout
        try:
            connection.request("GET" if body is None else "POST", path,
                               body=None if body is None else payload,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            record.update(status=response.status, headers=dict(response.getheaders()))
            if stream:
                require(response.status == expected, f"HTTP {response.status}")
                require("text/event-stream" in response.getheader("Content-Type", ""),
                        "Missing SSE content type")
                events = []
                record["events"] = events
                raw_lines = []
                record["raw_lines"] = raw_lines
                while True:
                    require(time.monotonic() < deadline, "SSE exceeded request deadline")
                    line = response.readline().decode("utf-8").strip()
                    raw_lines.append(line)
                    if not line and response.isclosed():
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        record["done"] = True
                        break
                    event = json.loads(data)
                    require("error" not in event, f"SSE error: {event}")
                    events.append(event)
                    if cancel and any(c.get("delta", {}).get("content") or
                                      c.get("delta", {}).get("reasoning_content")
                                      for c in event.get("choices", [])):
                        record["cancelled"] = True
                        break
                require(record.get("cancelled") if cancel else record.get("done"),
                        "No generated token before cancellation" if cancel else "Missing [DONE]")
                return events
            record["raw_body"] = response.read().decode("utf-8")
            require(response.status == expected, f"HTTP {response.status}: {record['raw_body']}")
            return json.loads(record["raw_body"])
        except Exception as error:
            record["error"] = str(error)
            raise
        finally:
            connection.close()
            self.save()

    def save(self):
        (self.args.evidence / "results.json").write_text(json.dumps({
            "cases": self.results, "exchanges": self.exchanges,
            "hip_execution_verified": False,
            "hip_note": "API checks do not prove provider dispatch. Review runtime logs separately."
        }, indent=2), encoding="utf-8")

    def case(self, name, check):
        result = {"name": name, "status": "running"}
        self.results.append(result)
        started = time.monotonic()
        try:
            check()
            result["status"] = "pass"
        except Exception as error:
            result.update(status="fail", error=str(error))
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        self.save()
        print(f"{result['status']}: {name} {result.get('error', '')}", flush=True)

    def chat_body(self, messages=None, stream=False):
        return {"model": "amdgpu", "messages": messages or [
            {"role": "user", "content": "Reply with the word hello. /no_think"}],
            "max_tokens": self.args.max_tokens, "temperature": 0, "stream": stream}

    def chat(self, messages=None):
        data = self.request("/v1/chat/completions", self.chat_body(messages))
        choice = data["choices"][0]
        require(choice.get("finish_reason") in ("stop", "length"), "Missing finish reason")
        message = choice["message"]
        require(message.get("role") == "assistant", "Wrong response role")
        require(bool((message.get("content") or "").strip()),
                "Empty final answer; reasoning-only output is not a usable answer")
        require(data["usage"]["completion_tokens"] > 0, "No completion tokens")
        return message

    def chat_stream(self):
        events = self.request("/v1/chat/completions", self.chat_body(stream=True), stream=True)
        choices = [c for event in events for c in event.get("choices", [])]
        require(any(c.get("finish_reason") in ("stop", "length") for c in choices),
                "Missing final finish reason")
        require(any((c.get("delta", {}).get("content") or "").strip() for c in choices),
                "No final-answer text in SSE")

    def multiturn(self):
        messages = [{"role": "user", "content": "Remember the code ORANGE731. Say OK. /no_think"}]
        answer = self.chat(messages)
        messages += [{"role": "assistant", "content": answer["content"]},
                     {"role": "user", "content": "What code did I give you? /no_think"}]
        require("ORANGE731" in self.chat(messages)["content"], "Multi-turn recall lost the code")

    def reset(self):
        result = self.request("/v1/sessions/reset", {})
        require(result.get("status") == "success", "Session reset did not succeed")
        answer = self.chat([{"role": "user", "content": "Reply only BLUE942. /no_think"}])
        require("BLUE942" in answer["content"] and "ORANGE731" not in answer["content"],
                "New conversation answer is wrong or contains prior conversation code")

    def repeated_chat(self):
        self.request("/v1/sessions/reset", {})
        first = self.chat()
        second = self.chat()
        require("hello" in first["content"].lower() and "hello" in second["content"].lower(),
                "Repeated prompt did not produce two usable answers")

    def conversation_isolation(self):
        for code, other in (("RED361", "GREEN482"), ("GREEN482", "RED361")):
            messages = [{"role": "user", "content": f"Reply only {code}. /no_think"}]
            answer = self.chat(messages)["content"]
            require(code in answer and other not in answer,
                    "Independent conversation returned the wrong code or leaked prior context")

    def invalid_then_recovery(self):
        result = self.request("/v1/chat/completions", "{invalid", expected=400)
        require("error" in result, "Malformed JSON did not return an error object")
        self.chat()

    def cancel_then_recovery(self):
        body = self.chat_body([{"role": "user", "content": "Count from 1 to 1000."}], stream=True)
        body["max_tokens"] = max(512, self.args.max_tokens)
        self.request("/v1/chat/completions", body, stream=True, cancel=True)
        self.chat()

    def completions(self, stream):
        data = self.request("/v1/completions", {
            "model": "amdgpu", "prompt": "The capital of France is",
            "max_tokens": self.args.max_tokens, "temperature": 0, "stream": stream
        }, stream=stream)
        choices = [c for e in data for c in e.get("choices", [])] if stream else data["choices"]
        require(any((c.get("text") or "").strip() for c in choices), "No completion text")
        require(any(c.get("finish_reason") in ("stop", "length") for c in choices),
                "Missing completion finish reason")

    def responses(self, stream):
        data = self.request("/v1/responses", {
            "model": "amdgpu", "input": "Say hello.", "max_output_tokens": self.args.max_tokens,
            "temperature": 0, "stream": stream
        }, stream=stream)
        if stream:
            completed = [e["response"] for e in data if e.get("type") == "response.completed"]
            require(bool(completed), "Missing response.completed")
            data = completed[-1]
        require(any(c.get("text", "").strip() for item in data["output"]
                    for c in item.get("content", []) if c.get("type") == "output_text"),
                "No Responses text")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-exe", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--cache-checks", action="store_true",
                        help="Also check repeated prompts and independent conversations")
    args = parser.parse_args()
    args.server_exe = args.server_exe.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    args.evidence = args.evidence.resolve()
    args.evidence.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    command = [str(args.server_exe), "-m", str(args.model), "--host", "127.0.0.1",
               "--port", str(args.port), "--ctx-size", str(args.ctx_size), "--verbose"]
    metadata = {"command": command, "platform": platform.platform(), "python": sys.version,
                "server_sha256": sha256(args.server_exe),
                "model_config_sha256": sha256(args.model / "genai_config.json"),
                "environment": {k: os.environ.get(k) for k in
                                ("PATH", "HIP_PATH", "ROCM_PATH", "AMDGPU_PKG_ROOT")}}
    (args.evidence / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    smoke = Smoke(args)
    process = None
    try:
        with (args.evidence / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=args.server_exe.parent, stdout=log,
                                       stderr=subprocess.STDOUT)
            deadline = time.monotonic() + args.startup_timeout
            while True:
                require(process.poll() is None, f"Server exited with {process.returncode}")
                require(time.monotonic() < deadline, "Server startup deadline exceeded")
                connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=1)
                try:
                    connection.request("GET", "/health")
                    if connection.getresponse().status == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    connection.close()
                time.sleep(0.5)
            smoke.case("health_gpu_config", lambda: require(
                smoke.request("/health").get("execution_mode") == "gpu", "Mode is not gpu"))
            smoke.case("chat_json", smoke.chat)
            smoke.case("chat_sse", smoke.chat_stream)
            smoke.case("multiturn", smoke.multiturn)
            smoke.case("session_reset", smoke.reset)
            smoke.case("malformed_json_recovery", smoke.invalid_then_recovery)
            smoke.case("cancel_recovery", smoke.cancel_then_recovery)
            for stream in (False, True):
                smoke.case(f"completions_stream_{stream}", lambda: smoke.completions(stream))
                smoke.case(f"responses_stream_{stream}", lambda: smoke.responses(stream))
            if args.cache_checks:
                smoke.case("repeated_chat", smoke.repeated_chat)
                smoke.case("conversation_isolation", smoke.conversation_isolation)
    except Exception as error:
        smoke.results.append({"name": "server_lifecycle", "status": "fail", "error": str(error)})
        print(f"fail: server_lifecycle {error}", flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        smoke.save()
    return int(any(r["status"] != "pass" for r in smoke.results))


if __name__ == "__main__":
    sys.exit(main())
