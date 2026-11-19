# HIP EP baseline test plan

The first baseline uses server main commit `2bb943241852201af0318d776cd4f118fd4a2fa5` and HIP EP v0.5.3. It excludes Shi Li's PRs. Keep the package, dependency hashes, model configuration, build logs, API evidence and limitations together. An API pass alone does not establish HIP execution or release readiness.

## Build and package gate

1. Verify the official Windows archive SHA256 is `ee1aadbf46142b1a4804fae531a6bc476966eca07e301f75310084f8e32f3c07`. Verify the Linux archive SHA256 is `d0f72eb9b477c5fa990f4142f65a7c959d54a32f38bd936b9f8aa7581849ec76`.
2. Record compiler, source commit, OGA header source and hashes, and runtime version. HIP v0.5.3 declares OGA v0.19.0 plus PR 2875. Older headers used for an experiment must remain labeled unverified against this runtime.
3. Build and install into an empty directory. Preserve the upstream `bin`, `lib` and root configuration paths. Check inventory and dependency loading after moving the package to another directory.
4. Run `bin/amdgpu-server --help` with a minimal PATH. On Windows use `amdgpu-server.exe`. A successful help command proves loading only.
5. Verify redistribution notices and operating-system prerequisites before publication. Linux requires separate verification of its external GPU runtime dependencies.

## Real GPU and API gate

Use an idle supported GPU. Record its model and driver. Keep existing processes and shared models intact. If a smaller context is necessary, create a model overlay with copied configuration and hardlinked weights, then record the original and modified configuration hashes. `--ctx-size` alone may not bound generator allocation in this server revision.

Run with Python 3.11 or later. The runner uses only the standard library and owns the process it starts.

```sh
python test/smoke.py --server-exe /path/to/package/bin/amdgpu-server \
  --model /path/to/model --evidence /path/to/new-results-directory \
  --ctx-size 4096 --max-tokens 256 --port 18081 --cache-checks
```

| Check | Required observation |
| --- | --- |
| Startup and health | The owned process remains alive; health reports GPU configuration. |
| Actual HIP execution | Model selects AMDGPU profile `hip`; runtime logs or profiling establish provider dispatch; CPU fallback is disabled where supported. |
| Chat JSON and SSE | Nonempty final-answer text, finish reason, positive token count for JSON, valid SSE JSON and `[DONE]`. Reasoning-only output fails the final-answer check. |
| Conversation | A second turn recalls a supplied code; reset succeeds; a new conversation answers its own request without the previous code. With `--cache-checks`, repeated prompts both return usable answers and unrelated prompts return their own distinct codes. |
| Recovery | Malformed JSON returns HTTP 400; disconnect after a generated SSE token; subsequent requests return usable output. |
| Completions and Responses | Both JSON and SSE produce text and terminal records. |
| Tool calls | A tool-capable model returns a parseable named function with schema-valid arguments, then consumes the tool result. Run separately before advertising tool support. |
| Limits and concurrency | Oversized prompt and concurrent requests produce bounded behavior without a crash or conversation leakage. Run separately before release. |
| Repeated lifecycle | Repeat model load, request, unload, reload; confirm resource reclamation and stable behavior. |

The runner snapshots each serialized request before sending it. Run `python -m unittest discover -s test -p test_smoke.py -v` to verify that later changes to a conversation do not alter saved request evidence. Historical results captured before this correction can show later messages in the first-turn request record. Preserve those records and annotate this limitation.

A reasoning model can exhaust its output budget before it produces final-answer text. Keep that failure, then run a separately labeled diagnostic with a larger budget. The diagnostic does not replace the original result or establish general conversation correctness.

The automated smoke runner covers basic API behavior and recovery. The last three rows and provider-dispatch proof require additional release validation. A model-dependent failure is recorded, then investigated with the same package and another supported model if available. Do not change baseline inference code to conceal it.

## Lemonade gate

After the server package passes its gates, pin the released artifact in Lemonade. Verify a fresh download and extraction, recursive executable discovery, local port selection, load and health polling, JSON/SSE forwarding, reset, stop and relaunch. Check the final application using the same model and requests. Preserve the server release URL and hash in evidence. Maintainers publish the official release if the authenticated contributor lacks repository write permission.

## Shi Li PR comparison

Review each PR's target, complete diff and commit ancestry. Rebase or test-merge only in an isolated candidate branch after saving the baseline. Reuse the exact runtime, model overlay, driver and request corpus. Add cases for KV reuse, divergent conversation prefixes, reset during/after generation, tool JSON repair and any modified packaging behavior. Merge only after review and applicable checks pass. Produce a separately named package with the merged commit and its own evidence.
