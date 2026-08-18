# TranslateGemma 27B IT v1.0.0 — Release Notes

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE_NOTES_v1.0.0.vi.md)

## Overview

`v1.0.0` is the first and only public release in this publication cycle of the TranslateGemma 27B IT text + vision REST server for Kaggle TPU v5e-8.

The final public snapshot combines the validated 8-device TPU runtime, authenticated REST API, async job workflow, PRIME/HOT semantic acceptance, sanitized evidence collection, bilingual documentation, and GitHub-import-first Kaggle notebook without changing the frozen TPU inference core.

## Validated runtime contract

- TranslateGemma 27B IT on exactly 8 TPU devices with one logical worker.
- Keras ModelParallel mesh `[1,8]`, BF16 inference, split-compile generation, and strict 1247/1247 weight loading.
- Logical parameter size 51.884850 GiB; 95.05234% of parameter bytes sharded; unknown sharding bytes 0%.
- Peak cgroup memory 206.049 GiB, leaving 93.951 GiB below the 300 GiB memory guard.
- Six acceptance jobs completed, zero failed jobs, and zero automatic worker restarts.

## PRIME and HOT acceptance

Text PRIME took 496.534 seconds client-side while prefill/decode compilation populated executable caches. Text HOT-1 and HOT-2 completed in about 0.207 seconds client-side with cache reuse, TTFT around 35 ms, and about 68–69 generated tokens/second.

Vision PRIME took 364.415 seconds client-side. Vision HOT-1 and HOT-2 completed in about 0.58 seconds with cache reuse, TTFT around 141 ms, and about 71 generated tokens/second.

Semantic validation passed for every PRIME/HOT text and vision result. The evidence confirms that first-shape JAX/XLA compilation, rather than warm decode runtime, dominates initial latency.

## Evidence and notebook hardening

- Text and vision smoke scripts run `PRIME → HOT-1 → HOT-2` semantic acceptance and retain compile/cache/timing metrics.
- Fresh Kaggle validation recreates `.env` from `.env.example` deterministically.
- Sanitized evidence is collected while the server is still available and includes memory telemetry, acceptance JSON, source snapshots, and checksums.
- Runtime endpoint evidence writes an explicit unavailable marker instead of silently omitting `/info` or `/health` evidence.
- Normal worker shutdown interruption is handled without a cosmetic `KeyboardInterrupt` traceback.

## Single-release publication policy

This publication cycle intentionally exposes only `v1.0.0`. Earlier `v1.0.0` tag/release material was pre-publication validation material and is refreshed in place rather than superseded by a new version.

After the final `main` is amended and a fresh Kaggle notebook imported from GitHub passes **Restart Session → Run All**, the existing annotated `v1.0.0` tag is moved to that final commit with a lease-protected force update. The operation fails closed if the remote tag changed after the expected old tag object SHA was recorded.

The existing GitHub Release `v1.0.0` is then edited in place and all source/notebook assets are uploaded with `--clobber`. No successor release is part of this publication cycle.

## Frozen TPU engine integrity

The accepted TPU inference core remains unchanged: `engine.py`, `distribution.py`, and `generation.py` retain the proven v1.0.0 engine behavior and checksums.

## Release artifacts

GitHub Actions verifies unit/contract tests, documentation parity, syntax, secret scanning, source ZIP integrity, and SHA256/MD5 manifests, then refreshes the existing `v1.0.0` release assets.

## Scope

Model weights are not bundled. Temporary tunnel URLs are not production infrastructure, and real TPU validation remains separate from CPU-friendly CI to preserve accelerator quota.
