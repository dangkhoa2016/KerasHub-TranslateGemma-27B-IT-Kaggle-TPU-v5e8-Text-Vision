# Validation and runtime evidence

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](BENCHMARKS.vi.md)

These observations come from a validated real Kaggle TPU v5e-8 run and are evidence values, not performance guarantees or service-level objectives.

## Functional acceptance

The validated 27B runtime reached:

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

## Startup and readiness

The observed initial model load was `686.861 s`. A replacement worker created through the authenticated restart path loaded in `724.519 s` and became ready while the Waitress coordinator process stayed alive.

During worker loading, `/health/live` can succeed while `/health/ready` correctly reports not-ready. This distinction is intentional and should be preserved.

## Cold and warm request evidence

The first text and vision requests included JAX compilation and therefore took several minutes. Compatible warm shapes reused the compiled prefill/decode graphs.

Observed evidence from the validated run:

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

These numbers are retained only to explain cold-versus-warm behavior. Kaggle images, model mounts, cache state, and runtime versions can change.

## Memory and sharding evidence

The runtime keeps a `300 GiB` cgroup hard guard. Earlier validated sampling observed a run peak around `172.351 GiB`; the final safety gate observed current cgroup memory at `63.318 GiB` with no breach artifact.

Byte-weighted model telemetry observed approximately `95.05234%` sharded parameter bytes, `4.94766%` replicated bytes, and `0%` unknown sharding bytes.

## Validation policy

Ordinary documentation or API changes should not consume TPU quota for repeated performance measurements. Re-run accelerator validation when a change can materially affect inference, compilation, memory use, checkpoint loading, or device topology.
