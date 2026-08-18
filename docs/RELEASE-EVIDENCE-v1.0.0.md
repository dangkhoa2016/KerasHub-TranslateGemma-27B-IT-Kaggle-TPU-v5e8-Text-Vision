# Release evidence — v1.0.0

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE-EVIDENCE-v1.0.0.vi.md)

## Purpose

This document records the real Kaggle TPU v5e-8 evidence used to publish the first public `v1.0.0` release. The public repository does not expose private development version history.

## Frozen runtime contract

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

## Checkpoint and sharding evidence

The validated checkpoint exposed `1247` model weights and `1247` trainable weights. Byte-weighted telemetry observed approximately `95.05234%` sharded parameter bytes, `4.94766%` replicated parameter bytes, and `0%` unknown sharding bytes.

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

These numbers are evidence from one validated environment, not performance targets.

## Safety evidence

The final safety gate found no `worker_failed`, `MemoryError`, out-of-memory signal, `Killed` signal, memory-guard breach artifact, or unexpected HTTP 500. Known TPU/JAX startup warnings alone are not treated as failures when readiness and safety gates pass.

## Public release rule

`v1.0.0` is the only public project version represented by this repository. Dependency versions, TPU hardware labels, protocol versions, and file-format versions are independent technical identifiers and are not project release versions.
