# Recurrent-state cache validation

Repeated chat requests on Qwen3.5-9B with HIP EP 0.5.3 fail on the original server because OGA rejects partial recurrent-state rewind. A fresh request works, and an explicit session reset restores generation. The fix reuses the generator only when the next token sequence strictly extends the full cached sequence. Equal, shorter, divergent, or changed-sampling requests create a new generator.

The change applies to the shared text generator used by JSON and streaming requests. It removes partial-prefix reuse for all text models. True extensions retain reuse. No performance comparison or model-specific capability detection was performed.

## Environment

- Original inference source is main commit `2bb943241852201af0318d776cd4f118fd4a2fa5`.
- Windows x64, Radeon 8060S, driver 32.0.22050.4002, MSVC 19.44.35230.0.
- Qwen3.5-9B uses an isolated 4096-context configuration with the AMDGPU `hip` profile. Model weights are unchanged.
- Runtime is HIP EP 0.5.3, ORT 1.31.0, and OGA 0.19.0 with PR 2875.
- The build uses identified OGA 0.14.0 headers because matching public 0.19.0 sources were unavailable. Header provenance confirmation is deferred. Exercised API calls do not establish full compatibility.
- The tested candidate executable SHA256 is `7c6afeb0505566651bd105877082da48635d2d1372f09e5784f35af7cce506d5`. The archived build-input snapshot records the exact inputs. A later header-comment correction does not change declarations or implementation.

## Observed results

The runner and packaging changes are in [PR 3](https://github.com/lemonade-sdk/amdgpu-server/pull/3). Requests use temperature 0. Runs below use the same runtime and model overlay.

| Run | Result |
| --- | --- |
| Original executable, 256-token budget, 13 cases | 7 passed, 6 failed. Repeated prompts and unrelated conversations reproduce the partial-rewind exception. |
| Candidate, corrected request recorder, 256-token budget, 13 cases | 12 passed, 1 failed. JSON, SSE, reset, malformed-input recovery, cancellation recovery, Completions, Responses, repeated prompts, and unrelated conversations passed. |
| Candidate, separate multi-turn diagnostic, 1024-token budget | Passed. The second response contains `ORANGE731` after 602 completion tokens. |
| Candidate, strict-prefix append diagnostic | Passed. Runtime logs show nonzero prefix reuse. Warm and reset/cold outputs both equal ` 9, 10, `. |
| Request-recorder regression with a local HTTP server | Failed before the correction because saved evidence contained a later message absent from the wire request. Passed after the correction. |

The remaining 256-token failure returns HTTP 200 and reasoning text but no final answer. It remains a failed case. The larger-budget diagnostic does not convert the 13-case result into a full pass. The server also hardcodes `finish_reason` to `stop`, including at token-budget exhaustion. Correct termination reporting needs a separate API change.

The original runner retained a mutable reference to each request. Later multi-turn messages therefore altered the saved first-turn request record in historical runs. Actual requests and response fields were unaffected. The corrected candidate run snapshots the serialized request. Historical evidence remains unchanged and carries this limitation.

## Reproduction

Use the runner from PR 3 with the candidate package.

```sh
python test/smoke.py --server-exe /path/to/package/bin/amdgpu-server.exe \
  --model /path/to/model --evidence /path/to/new-results-directory \
  --ctx-size 4096 --max-tokens 256 --cache-checks
```

The experimental package retains raw requests, responses, server logs, the corrected runner, and diagnostic scripts under `validation`. The original baseline package remains separate.

The append diagnostic sends `The first twenty integers are 1, 2, 3, 4, 5,` to `/v1/responses` with an eight-token output budget. It appends the generated text and `<|im_end|>\nContinue the list:` to that prompt. It checks nonzero prefix reuse, then resets and submits the same extended prompt to compare output. This tokenizer-specific diagnostic is not a general model acceptance test.

## Remaining release checks

Matching header provenance, Linux runtime compatibility, tool calls, limits and concurrency, repeated lifecycle, cross-party verification, and Lemonade application tests remain open. Shi Li's PRs are excluded from this candidate. The package is experimental and no production release is qualified by these results.
