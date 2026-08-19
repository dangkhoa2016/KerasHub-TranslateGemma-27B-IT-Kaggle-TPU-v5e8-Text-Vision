# Release evidence — v1.0.0

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE-EVIDENCE-v1.0.0.vi.md)

## Scope

This document records the real Kaggle TPU v5e-8 hardening evidence for the final `v1.0.0` publication candidate before the last GitHub-only publication gate.

## Source and tests

- Base commit before final integration: `1be6d5867ce3686103ef234cf99244b34a6be55d`.
- Hardening TDD RED and GREEN phases passed.
- Full validated Kaggle suite: 149/149 tests passed before release/tag overwrite hardening was added.
- Runtime: TranslateGemma 27B IT, JAX, BF16, TPU v5e-8, 8 devices, mesh `[1,8]`.

## Model and memory

- Strict model weights: 1247/1247.
- Logical parameter size: 51.884850 GiB.
- Sharded parameter bytes: 95.05234%; unknown sharding bytes: 0%.
- Peak cgroup memory: 206.049 GiB.
- Memory guard: 300 GiB; measured headroom: 93.951 GiB.

## Text acceptance

- PRIME client time: 496.534 s; inference: 494.610 s; TTFT: 493.744 s.
- PRIME compile: prefill 316.407 s + decode 177.338 s; cache reuse false.
- HOT-1 client time: 0.207547 s; TTFT 0.035955 s; 67.81 tokens/s; cache reuse true.
- HOT-2 client time: 0.207038 s; TTFT 0.034631 s; 69.02 tokens/s; cache reuse true.
- Semantic result: PASS.

## Vision acceptance

- PRIME client time: 364.415 s; inference: 362.979 s; TTFT: 361.673 s.
- PRIME compile: prefill 183.918 s + decode 177.755 s; cache reuse false.
- HOT-1 client time: 0.581767 s; TTFT 0.140516 s; 70.95 tokens/s; cache reuse true.
- HOT-2 client time: 0.579858 s; TTFT 0.140904 s; 71.19 tokens/s; cache reuse true.
- Semantic result: PASS.

## Stability

- Completed acceptance jobs: 6.
- Failed jobs: 0.
- Automatic worker restarts: 0.
- No OOM or progressive HOT-request memory growth was observed.

## Artifact integrity

The sanitized evidence ZIP external SHA256 matched, all internal SHA256 entries matched, and runtime credential files were excluded.

## Single-release overwrite gate

The final publication keeps only `v1.0.0`. The existing annotated remote tag must remain untouched until a new Kaggle notebook imported directly from the amended GitHub `main` passes **Restart Session → Run All** and produces fresh sanitized evidence.

After that PASS, the tag may be moved to final `main` only with an exact `--force-with-lease` guard against the previously recorded remote tag object SHA, then the existing GitHub Release `v1.0.0` and its assets are refreshed in place.
