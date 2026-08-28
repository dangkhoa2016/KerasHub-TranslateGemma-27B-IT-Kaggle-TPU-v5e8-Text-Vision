# Changelog

> 🌐 Language / Ngôn ngữ: [English](CHANGELOG.md) | **Tiếng Việt**

Tất cả public release đáng chú ý được ghi tại đây.

## v1.0.0 — 2026-08-28

Public release đầu tiên của REST server TranslateGemma 27B IT text + vision cho Kaggle TPU v5e-8.

### Runtime

- Một logical model TranslateGemma `translategemma_27b_it` được shard trên đúng 8 TPU devices với Keras ModelParallel mesh `[1,8]`.
- BF16 inference, strict checkpoint loading 1247/1247, split prefill/decode compilation và TPU inference core đã khóa không thay đổi.
- Semantic acceptance cho text và vision chạy `PRIME → HOT-1 → HOT-2` và ghi compile/cache/timing telemetry.
- Final validated evidence: 6/6 acceptance jobs hoàn tất, 0 failed jobs, 0 automatic worker restarts và peak cgroup memory 206.012 GiB dưới memory guard 300 GiB.

### Hardening release và evidence

- Tái tạo `.env` deterministic cho fresh Kaggle và thu sanitized evidence khi server vẫn còn khả dụng.
- Evidence gồm memory telemetry, acceptance JSON, source snapshot, checksums và explicit endpoint-unavailable marker khi không query được final runtime endpoint.
- Normal worker shutdown interruption được xử lý mà không tạo cosmetic `KeyboardInterrupt` traceback.
- Public release identity chỉ duy nhất `v1.0.0`, ghim chặt vào commit đã Kaggle-validated `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de` qua annotated tag object `6319d40dd7f6ca61d59e2604f4a3a029b08f14db`.
- Các asset source/notebook/evidence được checksum được publish trên GitHub Release `v1.0.0`.

### Trải nghiệm developer

- Workflow Kaggle ưu tiên import notebook trực tiếp từ GitHub với fresh **Restart Session → Run All** làm publication path đã validation.
- Python và Node.js clients, CPU-friendly CI, bilingual documentation parity, secret scanning và clean source packaging.

## Bảo trì sau release

Các thay đổi sau `v1.0.0` trên `main` chỉ là documentation/metadata trừ khi được nói rõ khác đi. Chúng không phải là mục tiêu runtime validation mới.