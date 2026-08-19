# Changelog

> 🌐 Language / Ngôn ngữ: [English](CHANGELOG.md) | **Tiếng Việt**

Tất cả public release đáng chú ý được ghi tại đây.

## v1.0.0 — 2026-08-28

Public release đầu tiên của REST server TranslateGemma 27B IT text + vision cho Kaggle TPU v5e-8.

### Runtime

- Một logical model TranslateGemma `translategemma_27b_it` được shard trên đúng 8 TPU devices với Keras ModelParallel mesh `[1,8]`.
- BF16 inference, strict checkpoint loading 1247/1247, split prefill/decode compilation và TPU inference core đã khóa không thay đổi.
- Semantic acceptance cho text và vision giờ chạy `PRIME → HOT-1 → HOT-2` và ghi compile/cache/timing telemetry.
- Candidate evidence đã validation: 6/6 acceptance jobs hoàn tất, 0 failed jobs, 0 automatic worker restarts và peak cgroup memory 206.049 GiB dưới memory guard 300 GiB.

### Hardening release và evidence

- Tái tạo `.env` deterministic cho fresh Kaggle và thu sanitized evidence khi server vẫn còn khả dụng.
- Evidence gồm memory telemetry, acceptance JSON, source snapshot, checksums và explicit endpoint-unavailable marker khi không query được final runtime endpoint.
- Normal worker shutdown interruption được xử lý mà không tạo cosmetic `KeyboardInterrupt` traceback.
- Public release identity chỉ duy nhất `v1.0.0`; pre-publication tag refresh được bảo vệ bằng exact remote-tag lease thay vì unconditional force push.
- GitHub Release `v1.0.0` được refresh tại chỗ và checksummed source/notebook assets được thay bằng `--clobber`.

### Trải nghiệm developer

- Workflow Kaggle ưu tiên import notebook trực tiếp từ GitHub với fresh **Restart Session → Run All** làm final publication gate.
- Python và Node.js clients, CPU-friendly CI, bilingual documentation parity, secret scanning và clean source packaging.
