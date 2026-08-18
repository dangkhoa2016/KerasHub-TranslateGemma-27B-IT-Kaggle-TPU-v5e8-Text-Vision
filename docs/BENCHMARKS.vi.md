# Evidence validation và runtime

> 🌐 Language / Ngôn ngữ: [English](BENCHMARKS.md) | **Tiếng Việt**

Các observation này đến từ một run Kaggle TPU v5e-8 thật đã validation và là evidence values, không phải performance guarantees hay service-level objectives.

## Functional acceptance

Runtime 27B đã validation đạt:

```text
TPU devices                8
mesh                       [1,8]
dtype                      bfloat16
strict weights             1247/1247
generation                 split_compile
text translation           PASS
direct image translation   PASS
authenticated restart      PASS
```

## Startup và readiness

Initial model load quan sát được là `686.861 s`. Replacement worker tạo qua authenticated restart load trong `724.519 s` và trở thành ready trong khi Waitress coordinator process vẫn sống.

Trong lúc worker loading, `/health/live` có thể thành công còn `/health/ready` đúng thiết kế vẫn báo not-ready. Sự tách biệt này là có chủ ý và cần được giữ nguyên.

## Evidence request cold và warm

Text và vision request đầu tiên bao gồm JAX compilation nên mất vài phút. Shape warm tương thích reuse compiled prefill/decode graphs.

Evidence quan sát từ run đã validation:

```text
cold text client total     708.874 s
cold text model total      708.749947 s
cold image client total    519.628 s
cold image model total     518.80455 s
warm async client total    0.256 s
warm async model total     0.158778 s
prefill cache reused       true
decode cache reused        true
```

Các số này chỉ được giữ để giải thích cold-versus-warm behavior. Kaggle images, model mounts, cache state và runtime versions có thể thay đổi.

## Evidence memory và sharding

Runtime giữ cgroup hard guard `300 GiB`. Sampling validation trước đó quan sát run peak khoảng `172.351 GiB`; final safety gate quan sát current cgroup memory `63.318 GiB` và không có breach artifact.

Byte-weighted model telemetry quan sát xấp xỉ `95.05234%` sharded parameter bytes, `4.94766%` replicated bytes và `0%` unknown sharding bytes.

## Chính sách validation

Ordinary documentation hoặc API changes không nên tiêu TPU quota cho repeated performance measurements. Chạy lại accelerator validation khi thay đổi có thể ảnh hưởng đáng kể tới inference, compilation, memory use, checkpoint loading hoặc device topology.
