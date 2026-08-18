# Evidence phát hành — v1.0.0

> 🌐 Language / Ngôn ngữ: [English](RELEASE-EVIDENCE-v1.0.0.md) | **Tiếng Việt**

## Mục đích

Tài liệu này ghi lại evidence Kaggle TPU v5e-8 thật dùng để publish public release đầu tiên `v1.0.0`. Public repository không expose lịch sử version phát triển private.

## Runtime contract được khóa

```text
model                    translategemma_27b_it
TPU devices              8
logical model count      1
TPU worker count         1
mesh                     [1,8]
axes                     [batch,model]
dtype                    bfloat16
checkpoint loading       strict
generation               split_compile
memory hard guard        300 GiB
HTTP server              Waitress
inference concurrency    1
```

## End-to-end acceptance

```text
TPU ready                    PASS
safe runtime info            PASS
synchronous text             PASS
long HTTP request             PASS
direct image translation     PASS
ALL-CAPS semantic correction PASS
async submit/poll             PASS
JIT cache reuse               PASS
authenticated worker restart PASS
replacement worker ready     PASS
final health/memory gate      PASS
```

## Evidence checkpoint và sharding

Checkpoint đã validation expose `1247` model weights và `1247` trainable weights. Byte-weighted telemetry quan sát xấp xỉ `95.05234%` sharded parameter bytes, `4.94766%` replicated parameter bytes và `0%` unknown sharding bytes.

## Runtime observations

```text
initial model load          686.861 s
cold text client total      708.874 s
cold text model total       708.749947 s
cold image client total     519.628 s
cold image model total      518.80455 s
warm async client total     0.256 s
warm async model total      0.158778 s
replacement worker load     724.519 s
final cgroup memory          63.318 GiB
```

Các số này là evidence từ một môi trường đã validation, không phải performance targets.

## Safety evidence

Final safety gate không thấy `worker_failed`, `MemoryError`, tín hiệu out-of-memory, tín hiệu `Killed`, memory-guard breach artifact hay HTTP 500 bất thường. Warning startup TPU/JAX đã biết không tự động được coi là failure khi readiness và safety gates đều pass.

## Quy tắc public release

`v1.0.0` là project version public duy nhất được thể hiện trong repository này. Dependency versions, TPU hardware labels, protocol versions và file-format versions là technical identifiers độc lập, không phải project release versions.
