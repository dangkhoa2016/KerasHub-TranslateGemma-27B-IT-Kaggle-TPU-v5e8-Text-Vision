# TranslateGemma 27B IT v1.0.0 — Ghi chú phát hành

> 🌐 Language / Ngôn ngữ: [English](RELEASE_NOTES_v1.0.0.md) | **Tiếng Việt**

## Tổng quan

`v1.0.0` là public release đầu tiên và duy nhất trong publication cycle này của REST server TranslateGemma 27B IT text + vision cho Kaggle TPU v5e-8.

Final public snapshot kết hợp runtime TPU 8 thiết bị đã validation, REST API có authentication, async job workflow, PRIME/HOT semantic acceptance, sanitized evidence collection, tài liệu song ngữ và Kaggle notebook ưu tiên import từ GitHub mà không thay đổi TPU inference core đã khóa.

## Runtime contract đã validation

- TranslateGemma 27B IT trên đúng 8 TPU devices với một logical worker.
- Keras ModelParallel mesh `[1,8]`, BF16 inference, split-compile generation và strict weight loading 1247/1247.
- Logical parameter size 51.884850 GiB; 95.05234% parameter bytes được shard; unknown sharding bytes 0%.
- Peak cgroup memory 206.049 GiB, còn 93.951 GiB dưới memory guard 300 GiB.
- Sáu acceptance jobs hoàn tất, không có failed job và không có automatic worker restart.

## PRIME và HOT acceptance

Text PRIME mất 496.534 giây client-side trong khi prefill/decode compilation tạo executable caches. Text HOT-1 và HOT-2 hoàn tất khoảng 0.207 giây client-side với cache reuse, TTFT khoảng 35 ms và khoảng 68–69 generated tokens/second.

Vision PRIME mất 364.415 giây client-side. Vision HOT-1 và HOT-2 hoàn tất khoảng 0.58 giây với cache reuse, TTFT khoảng 141 ms và khoảng 71 generated tokens/second.

Semantic validation PASS cho mọi kết quả PRIME/HOT text và vision. Evidence xác nhận first-shape JAX/XLA compilation, không phải warm decode runtime, là thành phần chi phối initial latency.

## Hardening evidence và notebook

- Text và vision smoke scripts chạy `PRIME → HOT-1 → HOT-2` semantic acceptance và giữ compile/cache/timing metrics.
- Fresh Kaggle validation tái tạo `.env` từ `.env.example` theo cách deterministic.
- Sanitized evidence được thu khi server vẫn còn khả dụng và gồm memory telemetry, acceptance JSON, source snapshots và checksums.
- Runtime endpoint evidence ghi explicit unavailable marker thay vì âm thầm bỏ `/info` hoặc `/health` evidence.
- Normal worker shutdown interruption được xử lý mà không tạo cosmetic `KeyboardInterrupt` traceback.

## Chính sách single-release publication

Publication cycle này chủ ý chỉ public `v1.0.0`. Tag/release material `v1.0.0` trước đó là material validation trước publication và được refresh tại chỗ thay vì tạo version mới.

Sau khi final `main` được amend và một Kaggle notebook mới import từ GitHub PASS **Restart Session → Run All**, annotated tag `v1.0.0` hiện có được chuyển sang final commit bằng lease-protected force update. Thao tác fail-closed nếu remote tag đã thay đổi sau khi expected old tag object SHA được ghi nhận.

GitHub Release `v1.0.0` hiện có sau đó được edit tại chỗ và toàn bộ source/notebook assets được upload bằng `--clobber`. Publication cycle này không có successor release.

## Integrity của TPU engine đã khóa

TPU inference core đã chấp nhận không thay đổi: `engine.py`, `distribution.py` và `generation.py` giữ nguyên proven v1.0.0 engine behavior và checksums.

## Release artifacts

GitHub Actions verify unit/contract tests, documentation parity, syntax, secret scanning, source ZIP integrity và SHA256/MD5 manifests, sau đó refresh assets của release `v1.0.0` hiện có.

## Phạm vi

Model weights không được bundle. Temporary tunnel URLs không phải production infrastructure và real TPU validation vẫn tách khỏi CPU-friendly CI để bảo toàn accelerator quota.
