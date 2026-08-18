# TranslateGemma 27B IT v1.0.0 — Ghi chú phát hành

> 🌐 Language / Ngôn ngữ: [English](RELEASE_NOTES_v1.0.0.md) | **Tiếng Việt**

## Tổng quan

`v1.0.0` là stable release đầu tiên của REST server TranslateGemma 27B IT text + vision cho Kaggle TPU v5e-8.

Release đóng gói runtime TPU 8 thiết bị đã được validation, REST API có authentication, async job workflow, Python và Node.js clients, tài liệu song ngữ và Kaggle notebook ưu tiên import từ GitHub thành một release có thể tái lập.

## Điểm nổi bật của release

- TranslateGemma 27B IT được shard trên đúng 8 TPU devices với một logical worker.
- ModelParallel mesh `[1,8]` với axes `[batch, model]`.
- BF16 inference, strict checkpoint loading và generation `split_compile`.
- Text và vision translation có authentication với cả synchronous và asynchronous endpoints.
- Bounded job queue, readiness/liveness endpoints, runtime metadata và controlled TPU-worker restart.
- Kaggle notebook được thiết kế cho fresh acceptance workflow **Restart Session → Run All**.
- Cold compilation chạy lâu được xử lý bằng async submit + bounded polling thay vì giữ một client socket mở quá lâu.

## Runtime contract đã validation

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

Môi trường được chấp nhận dùng Python 3.12.x, Keras 3.15.1, KerasHub 0.31.0, JAX 0.10.2, jaxlib 0.10.2 và `libtpu` 0.0.17.

## End-to-end acceptance

Release đã được validation trên Kaggle TPU thật với:

- fresh dependency/bootstrap checks;
- đúng 8 TPU devices và mesh `[1,8]`;
- strict 1247/1247 weight loading;
- authenticated `/info` runtime inspection;
- text translation hoàn tất;
- vision translation hoàn tất;
- async job progression `queued → processing → completed`;
- controlled worker restart trong khi coordinator được giữ ổn định;
- final shutdown sạch, không còn orphan managed TPU worker.

Một fresh public-style Kaggle retest cũng xác nhận cold vision compile hợp lệ có thể mất khoảng 833 giây trước first token. Server và TPU worker vẫn khỏe mạnh và vision job hoàn tất thành công.

## Hardening timeout khi cold compile

Final snapshot `v1.0.0` loại bỏ mismatch giữa public notebook/client được phát hiện khi fresh acceptance testing.

High-level Python text và image translation giờ mặc định dùng async submission cùng `/result/<job_id>` polling. Public Kaggle notebook dùng path `scripts/test_vision.sh` đã được validation với per-request timeout 30 giây và overall polling window 1800 giây. Explicit synchronous client calls vẫn có sẵn qua `--sync` cho workload được chủ động giới hạn thời gian.

Cách này ngăn một TPU job hợp lệ chạy lâu bị hiểu nhầm là model failure chỉ vì client socket hết hạn trước.

## Integrity của TPU engine đã khóa

TPU inference core đã được chấp nhận vẫn không thay đổi:

```text
1a2658c55df2a204d59dc18960bd490e0231ef2c6d7582c406dc2b5a23fe1048  src/translategemma_server/tpu/engine.py
e07b7ac54b600a5cbfdaede8c2daa534797bcb7bcea70dfcb8f19ab1b9ac8d13  src/translategemma_server/tpu/distribution.py
4c5a17835d2f1d4601c28bd5bbd8781426f8ab63fa45c0893133a5285d1df5f8  src/translategemma_server/tpu/generation.py
```

## Bề mặt API

Service cung cấp text/image translation có authentication, sync/async job endpoints, `/result/<job_id>` polling, health/readiness endpoints, `/info`, controlled restart, Python và Node.js clients cùng Cloudflare Quick Tunnel tùy chọn cho temporary remote access.

## Release artifacts

GitHub Actions build và verify:

- source archive;
- Kaggle notebook artifact;
- SHA256 và MD5 manifests cho cả hai;
- Python và Bash syntax;
- Node.js client syntax;
- unit/contract tests;
- bilingual documentation parity;
- secret scanning và ZIP integrity.

## Phạm vi

Repository này là serving implementation hướng Kaggle. Model weights không được bundle. Temporary tunnel URLs không phải production infrastructure, và real TPU validation được tách khỏi CPU-friendly CI để bảo toàn accelerator quota.
