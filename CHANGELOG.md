# Changelog

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

All notable public releases are documented here.

## v1.0.0 — 2026-08-28

First public release of the TranslateGemma 27B IT text + vision REST server for Kaggle TPU v5e-8.

### Runtime

- One logical TranslateGemma `translategemma_27b_it` model sharded across exactly 8 TPU devices with Keras ModelParallel mesh `[1,8]`.
- BF16 inference, strict 1247/1247 checkpoint loading, split prefill/decode compilation, and unchanged frozen TPU inference core.
- Text and vision semantic acceptance now runs `PRIME → HOT-1 → HOT-2` and records compile/cache/timing telemetry.
- Validated candidate evidence: 6/6 acceptance jobs completed, 0 failed jobs, 0 automatic worker restarts, and peak cgroup memory 206.049 GiB below the 300 GiB guard.

### Release and evidence hardening

- Deterministic fresh-Kaggle `.env` recreation and sanitized evidence collection while the server is still available.
- Evidence includes memory telemetry, acceptance JSON, source snapshot, checksums, and explicit endpoint-unavailable markers when final runtime endpoints cannot be queried.
- Normal worker shutdown interruption is handled without a cosmetic `KeyboardInterrupt` traceback.
- Public release identity remains exclusively `v1.0.0`; the pre-publication tag refresh is guarded by an exact remote-tag lease instead of an unconditional force push.
- GitHub Release `v1.0.0` is refreshed in place and its checksummed source/notebook assets are replaced with `--clobber`.

### Developer experience

- GitHub-import-first Kaggle notebook workflow with fresh **Restart Session → Run All** as the final publication gate.
- Python and Node.js clients, CPU-friendly CI, bilingual documentation parity, secret scanning, and clean source packaging.
