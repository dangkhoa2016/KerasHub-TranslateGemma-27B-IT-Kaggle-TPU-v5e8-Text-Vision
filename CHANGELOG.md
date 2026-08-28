# Changelog

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

All notable public releases are documented here.

## v1.0.0 — 2026-08-28

First public release of the TranslateGemma 27B IT text + vision REST server for Kaggle TPU v5e-8.

### Runtime

- One logical TranslateGemma `translategemma_27b_it` model sharded across exactly 8 TPU devices with Keras ModelParallel mesh `[1,8]`.
- BF16 inference, strict 1247/1247 checkpoint loading, split prefill/decode compilation, and unchanged frozen TPU inference core.
- Text and vision semantic acceptance runs `PRIME → HOT-1 → HOT-2` and records compile/cache/timing telemetry.
- Final validated evidence: 6/6 acceptance jobs completed, 0 failed jobs, 0 automatic worker restarts, and peak cgroup memory 206.012 GiB below the 300 GiB guard.

### Release and evidence hardening

- Deterministic fresh-Kaggle `.env` recreation and sanitized evidence collection while the server is still available.
- Evidence includes memory telemetry, acceptance JSON, source snapshot, checksums, and explicit endpoint-unavailable markers when final runtime endpoints cannot be queried.
- Normal worker shutdown interruption is handled without a cosmetic `KeyboardInterrupt` traceback.
- Public release identity remains exclusively `v1.0.0`, pinned to the Kaggle-validated commit `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de` through annotated tag object `6319d40dd7f6ca61d59e2604f4a3a029b08f14db`.
- Checksummed source/notebook/evidence assets are published on GitHub Release `v1.0.0`.

### Developer experience

- GitHub-import-first Kaggle notebook workflow with fresh **Restart Session → Run All** as the validated publication path.
- Python and Node.js clients, CPU-friendly CI, bilingual documentation parity, secret scanning, and clean source packaging.

## Post-release maintenance

Changes made after `v1.0.0` on `main` are documentation/metadata only unless explicitly stated otherwise. They do not constitute a new runtime validation target.