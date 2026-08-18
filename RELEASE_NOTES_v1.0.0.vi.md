# Ghi chú phát hành — v1.0.0

> 🌐 Language / Ngôn ngữ: [English](RELEASE_NOTES_v1.0.0.md) | **Tiếng Việt**

## Tổng quan

`v1.0.0` là public release đầu tiên của REST server TranslateGemma 27B IT text + vision cho Kaggle TPU v5e-8.

Release tập trung vào reproducible serving, operational contracts rõ ràng, safe packaging và workflow Kaggle ưu tiên import trực tiếp từ GitHub.

## Runtime đã validation

End-to-end Kaggle validation của release này đã chứng minh:

```text
public CPU contract tests   PASS
TPU devices                8
logical workers            1
mesh                       [1,8]
mesh axes                  [batch, model]
dtype                      bfloat16
generation                 split_compile
strict weight loading      true
model weights              1247
trainable weights          1247
vision                     enabled
text smoke test            PASS
multipart vision           PASS
authenticated restart      PASS
final health/memory gate    PASS
```

Môi trường validation dùng Python 3.12.x, Keras 3.15.1, KerasHub 0.31.0, JAX 0.10.2, jaxlib 0.10.2 và `libtpu` 0.0.17.

## Integrity của TPU engine đã khóa

TPU inference core được theo dõi bằng các SHA256 sau:

```text
1a2658c55df2a204d59dc18960bd490e0231ef2c6d7582c406dc2b5a23fe1048  src/translategemma_server/tpu/engine.py
e07b7ac54b600a5cbfdaede8c2daa534797bcb7bcea70dfcb8f19ab1b9ac8d13  src/translategemma_server/tpu/distribution.py
4c5a17835d2f1d4601c28bd5bbd8781426f8ab63fa45c0893133a5285d1df5f8  src/translategemma_server/tpu/generation.py
```

## Kiến trúc serving

- một logical model TranslateGemma 27B IT;
- một TPU worker trải trên đủ 8 TPU devices;
- ModelParallel mesh `[1,8]` với axes `[batch, model]`;
- BF16 inference và strict checkpoint loading;
- split prefill/decode JIT với Python autoregressive loop;
- Flask application được phục vụ bởi một Waitress coordinator process với bounded jobs và lifecycle supervision.

## Public API

Release cung cấp text/image translation có authentication, sync/async job endpoints, health/readiness/runtime metadata, restart supervision, Python và Node.js clients cùng Cloudflare Quick Tunnel tùy chọn.

## Kaggle startup hardening

Setup giữ nguyên JAX/JAXLIB do Kaggle cung cấp. Nếu `libtpu` chưa tồn tại, helper cài `libtpu==0.0.17` bằng `--no-deps`; nếu đã có thì giữ runtime hiện tại. TPU run thật dùng `TPU_PREFLIGHT_MODE=required` để đúng 8 TPU devices luôn là hard gate.

## Documentation và repository hygiene

Public repository có tài liệu English/Vietnamese theo cặp, community templates, CI CPU-friendly, notebook JSON validation, kiểm tra documentation parity, source packaging, SHA256 manifests và secret scanning.

## Phạm vi

Repository này là implementation serving hướng Kaggle. Nó không bundle TranslateGemma model weights và không xem temporary tunnel endpoints là production infrastructure.
