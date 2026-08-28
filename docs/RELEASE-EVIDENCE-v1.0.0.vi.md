# Evidence phát hành — v1.0.0

> 🌐 Language / Ngôn ngữ: [English](RELEASE-EVIDENCE-v1.0.0.md) | **Tiếng Việt**

## Phạm vi

Tài liệu này là final release evidence record bất biến cho public release `v1.0.0` của REST server TranslateGemma 27B IT text + vision cho Kaggle TPU v5e-8. Nó thay thế toàn bộ tài liệu evidence candidate/trước-publication.

## Danh tính release đã validation

- Validated release commit: `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de`
- Validated release tree: `56320fdd6cede00589c363912513190ed3d4be08`
- Annotated v1.0.0 tag object: `6319d40dd7f6ca61d59e2604f4a3a029b08f14db`
- Tag peeled target: `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de`
- Release workflow: `33180641938 — completed / success`
- Fresh evidence: `translategemma-27b-v100-evidence-20260828T125147Z.zip`
- Fresh evidence SHA256: `9b6e1d717e4f5daab84c4ce112bbfff29fdec13fd9fcf8c2aa9138794d308ce9`

## Runtime facts (lần chạy acceptance cuối cùng)

- TPU devices: 8.
- Logical model workers: 1.
- Mesh: `[1,8]`.
- Dtype: BF16 / bfloat16.
- Generation: split prefill/decode compile.
- Strict weights: 1247/1247.
- Jobs: 6 completed / 0 failed.
- Worker restarts: 0.
- Peak cgroup memory: 206.012 GiB.
- Memory guard: 300 GiB.
- Text semantic acceptance: PASS.
- Vision semantic acceptance: PASS.
- Vision tensor path: `[1,2,896,896,3]`.
- HOT cache reuse: true.

## Text acceptance (lần chạy cuối cùng)

- PRIME client time: 910.902414 s.
- HOT-1 client time: 0.208106 s; TTFT: 36.2 ms; throughput ≈ 68.41 tok/s.
- HOT-2 client time: 0.206174 s; TTFT: 34.9 ms; throughput ≈ 69.22 tok/s.
- Semantic result: PASS.

## Vision acceptance (lần chạy cuối cùng)

- PRIME client time: 702.710737 s.
- HOT-1 client time: 0.579547 s; TTFT: 140.4 ms; throughput ≈ 72.50 tok/s.
- HOT-2 client time: 0.579338 s; TTFT: 139.7 ms; throughput ≈ 72.34 tok/s.
- Semantic result: PASS.

## Stability

- Completed acceptance jobs: 6.
- Failed jobs: 0.
- Automatic worker restarts: 0.
- Không quan sát OOM hoặc progressive HOT-request memory growth.

## Artifact integrity

External SHA256 của fresh evidence ZIP sanitized khớp, toàn bộ internal SHA256 entries khớp và runtime credential files đã được loại trừ. Source ZIP công khai, Kaggle notebook công khai và fresh evidence ZIP được kiểm checksum trong GitHub Release assets.

## Ghi chú về PRIME latency

Giá trị PRIME client time chậm là do first-shape JAX/XLA compilation như dự kiến, không phải lỗi. Các lần gọi HOT-1 và HOT-2 chứng minh executable-cache reuse và steady-state latency sau lần compilation đầu tiên.

## Provenance boundary

Annotated tag `v1.0.0` công khai đã được public và xác minh độc lập. Tag được ghim chặt vào đúng runtime commit đã Kaggle-validated `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de`. Việc bảo trì chỉ tài liệu sau này trên `main` không làm thay đổi runtime implementation hoặc release assets của `v1.0.0` đã validation.

Runtime validation `v1.0.0` áp dụng chính xác cho commit `df13a7f6b304c8cdeafa5c15e2d1f75fc73d36de`. Bất kỳ commit `main` nào sau này do post-release closure tạo ra chỉ là documentation/metadata và không được diễn giải như một mục tiêu runtime validation mới.