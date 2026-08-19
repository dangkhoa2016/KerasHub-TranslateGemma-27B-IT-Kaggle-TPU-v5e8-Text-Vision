# Evidence phát hành — v1.0.0

> 🌐 Language / Ngôn ngữ: [English](RELEASE-EVIDENCE-v1.0.0.md) | **Tiếng Việt**

## Phạm vi

Tài liệu này ghi real Kaggle TPU v5e-8 hardening evidence cho final publication candidate `v1.0.0` trước GitHub-only publication gate cuối cùng.

## Source và tests

- Base commit trước final integration: `1be6d5867ce3686103ef234cf99244b34a6be55d`.
- Hardening TDD RED và GREEN phases PASS.
- Full validated Kaggle suite: 149/149 tests PASS trước khi bổ sung release/tag overwrite hardening.
- Runtime: TranslateGemma 27B IT, JAX, BF16, TPU v5e-8, 8 devices, mesh `[1,8]`.

## Model và memory

- Strict model weights: 1247/1247.
- Logical parameter size: 51.884850 GiB.
- Sharded parameter bytes: 95.05234%; unknown sharding bytes: 0%.
- Peak cgroup memory: 206.049 GiB.
- Memory guard: 300 GiB; measured headroom: 93.951 GiB.

## Text acceptance

- PRIME client time: 496.534 s; inference: 494.610 s; TTFT: 493.744 s.
- PRIME compile: prefill 316.407 s + decode 177.338 s; cache reuse false.
- HOT-1 client time: 0.207547 s; TTFT 0.035955 s; 67.81 tokens/s; cache reuse true.
- HOT-2 client time: 0.207038 s; TTFT 0.034631 s; 69.02 tokens/s; cache reuse true.
- Semantic result: PASS.

## Vision acceptance

- PRIME client time: 364.415 s; inference: 362.979 s; TTFT: 361.673 s.
- PRIME compile: prefill 183.918 s + decode 177.755 s; cache reuse false.
- HOT-1 client time: 0.581767 s; TTFT 0.140516 s; 70.95 tokens/s; cache reuse true.
- HOT-2 client time: 0.579858 s; TTFT 0.140904 s; 71.19 tokens/s; cache reuse true.
- Semantic result: PASS.

## Stability

- Completed acceptance jobs: 6.
- Failed jobs: 0.
- Automatic worker restarts: 0.
- Không quan sát OOM hoặc progressive HOT-request memory growth.

## Artifact integrity

Sanitized evidence ZIP external SHA256 khớp, toàn bộ internal SHA256 entries khớp và runtime credential files đã được loại trừ.

## Single-release overwrite gate

Final publication chỉ giữ `v1.0.0`. Annotated remote tag hiện có phải giữ nguyên cho tới khi một Kaggle notebook mới import trực tiếp từ amended GitHub `main` PASS **Restart Session → Run All** và tạo fresh sanitized evidence.

Sau PASS đó, tag chỉ được chuyển sang final `main` với exact `--force-with-lease` guard dựa trên remote tag object SHA đã ghi nhận trước đó, rồi GitHub Release `v1.0.0` hiện có và assets được refresh tại chỗ.
