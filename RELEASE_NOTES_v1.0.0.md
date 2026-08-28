# TranslateGemma 27B IT v1.0.0 — Release Notes

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE_NOTES_v1.0.0.vi.md)

## Overview

`v1.0.0` is the first and only public release in this publication cycle of the TranslateGemma 27B IT text + vision REST server for Kaggle TPU v5e-8.

The final public snapshot combines the validated 8-device TPU runtime, authenticated REST API, async job workflow, PRIME/HOT semantic acceptance, sanitized evidence collection, bilingual documentation, and GitHub-import-first Kaggle notebook without changing the frozen TPU inference core.

## Validated release identity

- Validated release commit: `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de`
- Annotated v1.0.0 tag object: `6319d40dd7f6ca61d59e2604f4a3a029b08f14db`
- Release workflow: `33180641938 — completed / success`
- Fresh evidence: `translategemma-27b-v100-evidence-20260828T125147Z.zip`
- See [docs/RELEASE-EVIDENCE-v1.0.0.md](docs/RELEASE-EVIDENCE-v1.0.0.md) for the final release evidence record.

## Validated runtime contract

- TranslateGemma 27B IT on exactly 8 TPU devices with one logical worker.
- Keras ModelParallel mesh `[1,8]`, BF16 inference, split-compile generation, and strict 1247/1247 weight loading.
- Logical parameter size 51.884850 GiB; 95.05234% of parameter bytes sharded; unknown sharding bytes 0%.
- Peak cgroup memory 206.012 GiB, leaving 93.988 GiB below the 300 GiB memory guard.
- Six acceptance jobs completed, zero failed jobs, and zero automatic worker restarts.
- Vision tensor path `[1,2,896,896,3]` validated.

## PRIME and HOT acceptance

Text PRIME took 910.902414 seconds client-side while the first-shape prefill/decode JAX/XLA compilation populated executable caches. Text HOT-1 and HOT-2 then completed in about 0.208 and 0.206 seconds client-side with cache reuse, TTFT around 34.9–36.2 ms, and about 68.4–69.2 generated tokens/second.

Vision PRIME took 702.710737 seconds client-side. Vision HOT-1 and HOT-2 completed in about 0.5796 and 0.5793 seconds with cache reuse, TTFT around 139.7–140.4 ms, and about 72.3–72.5 generated tokens/second.

Semantic validation passed for every PRIME/HOT text and vision result. The slow PRIME values are expected first-shape JAX/XLA compilation and are not a failure; the HOT calls demonstrate executable-cache reuse and steady-state latency.

## Evidence and notebook hardening

- Text and vision smoke scripts run `PRIME → HOT-1 → HOT-2` semantic acceptance and retain compile/cache/timing metrics.
- Fresh Kaggle validation recreates `.env` from `.env.example` deterministically.
- Sanitized evidence is collected while the server is still available and includes memory telemetry, acceptance JSON, source snapshots, and checksums.
- Runtime endpoint evidence writes an explicit unavailable marker instead of silently omitting `/info` or `/health` evidence.
- Normal worker shutdown interruption is handled without a cosmetic `KeyboardInterrupt` traceback.

## Single-release publication policy

This publication cycle intentionally exposes only `v1.0.0`. The public annotated tag is now published and independently verified; it remains pinned to the exact Kaggle-validated runtime commit `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de`. No successor release is part of this publication cycle, and later documentation-only changes on `main` do not alter the validated runtime or release assets.

## Frozen TPU engine integrity

The accepted TPU inference core remains unchanged: `engine.py`, `distribution.py`, and `generation.py` retain the proven v1.0.0 engine behavior and checksums.

## Release artifacts

GitHub Actions verifies unit/contract tests, documentation parity, syntax, secret scanning, source ZIP integrity, and SHA256/MD5 manifests, then refreshes the existing `v1.0.0` release assets.

## Scope

Model weights are not bundled. Temporary tunnel URLs are not production infrastructure, and real TPU validation remains separate from CPU-friendly CI to preserve accelerator quota.