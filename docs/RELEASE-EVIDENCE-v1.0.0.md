# Release evidence — v1.0.0

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE-EVIDENCE-v1.0.0.vi.md)

## Scope

This document is the immutable final release evidence record for the public `v1.0.0` release of the TranslateGemma 27B IT text + vision REST server for Kaggle TPU v5e-8. It supersedes all candidate/pre-publication evidence documents.

## Validated release identity

- Validated release commit: `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de`
- Validated release tree: `56320fdd6cede00589c363912513190ed3d4be08`
- Annotated v1.0.0 tag object: `6319d40dd7f6ca61d59e2604f4a3a029b08f14db`
- Tag peeled target: `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de`
- Release workflow: `33180641938 — completed / success`
- Fresh evidence: `translategemma-27b-v100-evidence-20260828T125147Z.zip`
- Fresh evidence SHA256: `9b6e1d717e4f5daab84c4ce112bbfff29fdec13fd9fcf8c2aa9138794d308ce9`

## Runtime facts (final acceptance run)

- TPU devices: 8.
- Logical model workers: 1.
- Mesh: `[1,8]`.
- Dtype: BF16 / bfloat16.
- Generation: split prefill/decode compile.
- Strict weights: 1247/1247.
- Jobs: 6 completed / 0 failed.
- Worker restarts: 0.
- Peak cgroup memory: 206.012 GiB.
- Memory guard: 300 GiB.
- Text semantic acceptance: PASS.
- Vision semantic acceptance: PASS.
- Vision tensor path: `[1,2,896,896,3]`.
- HOT cache reuse: true.

## Text acceptance (final run)

- PRIME client time: 910.902414 s.
- HOT-1 client time: 0.208106 s; TTFT: 36.2 ms; throughput ≈ 68.41 tok/s.
- HOT-2 client time: 0.206174 s; TTFT: 34.9 ms; throughput ≈ 69.22 tok/s.
- Semantic result: PASS.

## Vision acceptance (final run)

- PRIME client time: 702.710737 s.
- HOT-1 client time: 0.579547 s; TTFT: 140.4 ms; throughput ≈ 72.50 tok/s.
- HOT-2 client time: 0.579338 s; TTFT: 139.7 ms; throughput ≈ 72.34 tok/s.
- Semantic result: PASS.

## Stability

- Completed acceptance jobs: 6.
- Failed jobs: 0.
- Automatic worker restarts: 0.
- No OOM or progressive HOT-request memory growth was observed.

## Artifact integrity

The sanitized fresh evidence ZIP external SHA256 matched, all internal SHA256 entries matched, and runtime credential files were excluded. The public source ZIP, the public Kaggle notebook, and the fresh evidence ZIP are checksummed in the GitHub Release assets.

## Notes on PRIME latency

The slow PRIME client times are expected first-shape JAX/XLA compilation and are not a failure. The HOT-1 and HOT-2 calls demonstrate executable-cache reuse and steady-state latency after the initial compilation.

## Provenance boundary

The public `v1.0.0` annotated tag has been published and independently verified. It remains pinned to the exact Kaggle-validated runtime commit `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de`. Later documentation-only maintenance on `main` does not alter the validated `v1.0.0` runtime implementation or release assets.

`v1.0.0` runtime validation applies exactly to commit `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de`. Any later `main` commit created by the post-release closure is documentation/metadata only and must not be interpreted as a new runtime validation target.