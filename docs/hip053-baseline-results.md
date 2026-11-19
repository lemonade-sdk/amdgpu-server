# HIP EP 0.5.3 baseline results

The Windows baseline builds and performs HIP inference, but fails conversation
reuse checks. It is an experimental diagnostic package and is not ready for
Lemonade release integration.

## Inputs

- Server inference source is unchanged from main commit
  `2bb943241852201af0318d776cd4f118fd4a2fa5`. No Shi Li PR is included.
- Runtime is the official HIP EP v0.5.3 Windows archive. Its SHA256 is
  `ee1aadbf46142b1a4804fae531a6bc476966eca07e301f75310084f8e32f3c07`.
- Runtime OGA is v0.19.0 plus upstream PR 2875. Its matching public header URLs
  returned HTTP 404. The experiment used recovered v0.14.0 headers with hashes
  recorded in `runtime-versions.json`. Runtime API compatibility remains
  unverified beyond the exercised calls.
- Build used MSVC 19.44.35230.0. GPU tests used Radeon 8060S with driver
  32.0.22050.4002 and a local Qwen3.5-9B HIP model.
- The model overlay copied configuration and hardlinked weights. It changed
  `model.context_length` and `search.max_length` from 262144 to 4096. The original
  model stayed unchanged. AMDGPU profile `hip` and disabled CPU fallback stayed
  in all configured model stages.

## Observed results on November 20, 2026

All seven server source files compiled. No inference source changes were needed
for compilation or linking. The installed package retained all 536 upstream
runtime entries. The moved package passed `--help` with only Windows system
directories on PATH.

The standard-library runner in `test/smoke.py` completed eleven cases.

| Case | Result |
| --- | --- |
| Health reports GPU configuration | Pass |
| Initial chat JSON with final answer | Pass |
| Chat SSE after a prior chat | Fail |
| Multi-turn recall | Fail |
| Explicit reset and a new conversation | Pass |
| Malformed JSON and subsequent valid chat | Fail during valid chat; malformed JSON returned 400 |
| Cancellation and subsequent valid chat | Fail before cancellation because previous cache state could not rewind |
| Completions JSON and SSE | Both pass |
| Responses JSON and SSE | Both pass |

A separate diagnostic run established that fresh chat SSE passes. Repeating its
prompt fails. Reset then chat passes. After reset, disconnecting after the first
generated SSE token succeeds, but the next valid chat fails. Another reset
restores usable output. This distinguishes streaming transport from the cache
reuse failure.

The diagnostic run enabled `HIPDNN_EP_PERF=1`. The runtime emitted per-inference
GPU event measurements and operation tables. Together with the selected HIP
profile, this establishes actual HIP execution for the successful requests.
These runs are not performance comparisons.

## BASELINE-1: recurrent-state rewind blocks conversations

The server computes a longest shared token prefix in `prepareTextGenerator`
and calls `RewindTo(common)` when cached tokens extend beyond that prefix.
For the tested recurrent model, OGA returns:

```text
RecurrentState::RewindTo(18) is not supported without a snapshot.
Recurrent states cannot be partially rewound; call Snapshot() at the target length first.
```

The same failure occurs at other nonzero prefix lengths. Explicit reset clears
the generator and restores successful requests. A follow-up must define a valid
cache policy for recurrent models, such as recreating the generator when a
supported snapshot is unavailable, and verify divergent prompts, cancellation,
reset and multi-turn behavior against this saved baseline. The baseline keeps
the failure visible.

## Remaining release gates

Resolve matching OGA headers and review the patched API. Fix BASELINE-1 and rerun
the failing request sequence. Validate tool calls, context limits, concurrent
clients and lifecycle behavior from [the test plan](test-plan.md). Complete the
Linux runtime and host ABI checks. Then publish a tested server release and
validate its download, model load, forwarding and lifecycle in Lemonade.

Shi Li's PRs remain separate candidates. PR #1 changes inference and cache
behavior from an older base. PR #2 changes tool-call parsing, emission, tests
and packaging; its current diff does not change `inference_engine.cpp` and does
not address BASELINE-1. Neither PR was merged as part of the baseline.
