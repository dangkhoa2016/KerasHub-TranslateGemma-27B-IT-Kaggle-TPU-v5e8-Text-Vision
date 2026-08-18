# TranslateGemma 27B IT v1.0.0 — Release Notes

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE_NOTES_v1.0.0.vi.md)

## Overview

`v1.0.0` is the first stable release of the TranslateGemma 27B IT text + vision REST server for Kaggle TPU v5e-8.

It packages the validated 8-device TPU runtime, authenticated REST API, async job workflow, Python and Node.js clients, bilingual documentation, and a GitHub-import-first Kaggle notebook into one reproducible release.

## Release highlights

- TranslateGemma 27B IT sharded across exactly 8 TPU devices with one logical worker.
- ModelParallel mesh `[1,8]` with axes `[batch, model]`.
- BF16 inference, strict checkpoint loading, and `split_compile` generation.
- Authenticated text and vision translation with synchronous and asynchronous endpoints.
- Bounded job queue, readiness/liveness endpoints, runtime metadata, and controlled TPU-worker restart.
- Kaggle notebook designed for a fresh **Restart Session → Run All** acceptance workflow.
- Long-running cold compilation handled through async submit + bounded polling instead of one long-lived client socket.

## Validated runtime contract

```text
Model                       TranslateGemma-27B-IT
Backend                     JAX
Framework                   Keras 3 + KerasHub
Accelerator                 Kaggle TPU v5e-8 / v5litepod-8
TPU devices                 8
Logical workers             1
Mesh                        [1,8]
Mesh axes                   [batch, model]
Dtype                       bfloat16
Generation                  split_compile
Strict weight loading       true
Model weights               1247 / 1247
Vision                      enabled
Coordinator                 Waitress
Server request timeout      900 seconds
```

The accepted environment used Python 3.12.x, Keras 3.15.1, KerasHub 0.31.0, JAX 0.10.2, jaxlib 0.10.2, and `libtpu` 0.0.17.

## End-to-end acceptance

The release was validated on real Kaggle TPU hardware with:

- fresh dependency/bootstrap checks;
- exactly 8 TPU devices and mesh `[1,8]`;
- strict 1247/1247 weight loading;
- authenticated `/info` runtime inspection;
- text translation completion;
- vision translation completion;
- async `queued → processing → completed` job progression;
- controlled worker restart with coordinator continuity;
- clean final shutdown without an orphan managed TPU worker.

A fresh public-style Kaggle retest also confirmed that a cold vision compile can legitimately take about 833 seconds before the first token. The server and TPU worker remained healthy and the vision job completed successfully.

## Cold-compile timeout hardening

The final `v1.0.0` snapshot removes the public notebook/client mismatch discovered during fresh acceptance testing.

High-level Python text and image translation now use async submission plus `/result/<job_id>` polling by default. The public Kaggle notebook uses the already-validated `scripts/test_vision.sh` path with a 30-second per-request timeout and an 1800-second overall polling window. Explicit synchronous client calls remain available through `--sync` for intentionally bounded workloads.

This prevents a valid long-running TPU job from being mistaken for a model failure simply because a client socket expired first.

## Frozen TPU engine integrity

The accepted TPU inference core remains unchanged:

```text
1a2658c55df2a204d59dc18960bd490e0231ef2c6d7582c406dc2b5a23fe1048  src/translategemma_server/tpu/engine.py
e07b7ac54b600a5cbfdaede8c2daa534797bcb7bcea70dfcb8f19ab1b9ac8d13  src/translategemma_server/tpu/distribution.py
4c5a17835d2f1d4601c28bd5bbd8781426f8ab63fa45c0893133a5285d1df5f8  src/translategemma_server/tpu/generation.py
```

## API surface

The service exposes authenticated text and image translation, sync/async job endpoints, `/result/<job_id>` polling, health/readiness endpoints, `/info`, controlled restart, Python and Node.js clients, and an optional Cloudflare Quick Tunnel for temporary remote access.

## Release artifacts

GitHub Actions builds and verifies:

- the source archive;
- the Kaggle notebook artifact;
- SHA256 and MD5 manifests for both;
- Python and Bash syntax;
- Node.js client syntax;
- unit/contract tests;
- bilingual documentation parity;
- secret scanning and ZIP integrity.

## Scope

This repository is a Kaggle-oriented serving implementation. Model weights are not bundled. Temporary tunnel URLs are not production infrastructure, and real TPU validation is intentionally separated from CPU-friendly CI to preserve accelerator quota.
