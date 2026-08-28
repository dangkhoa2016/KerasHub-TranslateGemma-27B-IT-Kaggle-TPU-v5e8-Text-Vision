# TranslateGemma 27B IT v1.0.0 — Ghi chú phát hành

> 🌐 Language / Ngôn ngữ: [English](RELEASE_NOTES_v1.0.0.md) | **Tiếng Việt**

## Tổng quan

`v1.0.0` là public release đầu tiên và duy nhất trong publication cycle này của REST server TranslateGemma 27B IT text + vision cho Kaggle TPU v5e-8.

Final public snapshot kết hợp runtime TPU 8 thiết bị đã validation, REST API có authentication, async job workflow, PRIME/HOT semantic acceptance, sanitized evidence collection, tài liệu song ngữ và Kaggle notebook ưu tiên import từ GitHub mà không thay đổi TPU inference core đã khóa.

## Danh tính release đã validation

- Validated release commit: `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de`
- Annotated v1.0.0 tag object: `6319d40dd7f6ca61d59e2604f4a3a029b08f14db`
- Release workflow: `33180641938 — completed / success`
- Fresh evidence: `translategemma-27b-v100-evidence-20260828T125147Z.zip`
- Xem [docs/RELEASE-EVIDENCE-v1.0.0.vi.md](docs/RELEASE-EVIDENCE-v1.0.0.vi.md) cho final release evidence record.

## Runtime contract đã validation

- TranslateGemma 27B IT trên đúng 8 TPU devices với một logical worker.
- Keras ModelParallel mesh `[1,8]`, BF16 inference, split-compile generation và strict weight loading 1247/1247.
- Logical parameter size 51.884850 GiB; 95.05234% parameter bytes được shard; unknown sharding bytes 0%.
- Peak cgroup memory 206.012 GiB, còn 93.988 GiB dưới memory guard 300 GiB.
- Sáu acceptance jobs hoàn tất, không có failed job và không có automatic worker restart.
- Vision tensor path `[1,2,896,896,3]` đã validation.

## PRIME và HOT acceptance

Text PRIME mất 910.902414 giây client-side trong khi first-shape prefill/decode JAX/XLA compilation tạo executable caches. Text HOT-1 và HOT-2 sau đó hoàn tất khoảng 0.208 và 0.206 giây client-side với cache reuse, TTFT khoảng 34.9–36.2 ms và khoảng 68.4–69.2 generated tokens/second.

Vision PRIME mất 702.710737 giây client-side. Vision HOT-1 và HOT-2 hoàn tất khoảng 0.5796 và 0.5793 giây với cache reuse, TTFT khoảng 139.7–140.4 ms và khoảng 72.3–72.5 generated tokens/second.

Semantic validation PASS cho mọi kết quả PRIME/HOT text và vision. PRIME chậm là do first-shape JAX/XLA compilation như dự kiến, không phải lỗi; các lần gọi HOT chứng minh executable-cache reuse và steady-state latency.

## Hardening evidence và notebook

- Text và vision smoke scripts chạy `PRIME → HOT-1 → HOT-2` semantic acceptance và giữ compile/cache/timing metrics.
- Fresh Kaggle validation tái tạo `.env` từ `.env.example` theo cách deterministic.
- Sanitized evidence được thu khi server vẫn còn khả dụng và gồm memory telemetry, acceptance JSON, source snapshots và checksums.
- Runtime endpoint evidence ghi explicit unavailable marker thay vì âm thầm bỏ `/info` hoặc `/health` evidence.
- Normal worker shutdown interruption được xử lý mà không tạo cosmetic `KeyboardInterrupt` traceback.

## Chính sách single-release publication

Publication cycle này chủ ý chỉ public `v1.0.0`. Annotated tag công khai đã được public và xác minh độc lập; tag được ghim chặt vào đúng runtime commit đã Kaggle-validated `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de`. Publication cycle này không có successor release và các thay đổi chỉ tài liệu sau này trên `main` không làm thay đổi runtime hoặc release assets đã validation.

## Integrity của TPU engine đã khóa

TPU inference core đã chấp nhận không thay đổi: `engine.py`, `distribution.py` và `generation.py` giữ nguyên proven v1.0.0 engine behavior và checksums.

## Release artifacts

GitHub Actions verify unit/contract tests, documentation parity, syntax, secret scanning, source ZIP integrity và SHA256/MD5 manifests, sau đó refresh assets của release `v1.0.0` hiện có.

## Phạm vi

Model weights không được bundle. Temporary tunnel URLs không phải production infrastructure và real TPU validation vẫn tách khỏi CPU-friendly CI để bảo toàn accelerator quota.