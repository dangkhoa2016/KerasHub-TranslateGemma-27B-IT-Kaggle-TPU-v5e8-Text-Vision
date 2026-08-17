#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import api_smoke  # noqa: E402
from semantic_acceptance import SemanticValidationError, validate_translation  # noqa: E402

GIB = 1024 ** 3


def phase_names(hot_runs: int) -> list[str]:
    if hot_runs < 0:
        raise ValueError("hot_runs must be >= 0")
    return ["prime"] + [f"hot-{index}" for index in range(1, hot_runs + 1)]


def _read_cgroup_bytes(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw or raw == "max":
            return None
        return int(raw)
    except (OSError, ValueError):
        return None


def memory_snapshot() -> dict[str, float | None]:
    current = _read_cgroup_bytes(Path("/sys/fs/cgroup/memory.current"))
    peak = _read_cgroup_bytes(Path("/sys/fs/cgroup/memory.peak"))
    return {
        "current_gib": None if current is None else round(current / GIB, 3),
        "peak_gib": None if peak is None else round(peak / GIB, 3),
    }


def summarize_result(
    response: dict[str, Any],
    *,
    client_elapsed_seconds: float,
) -> dict[str, Any]:
    metrics = response.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    metric_keys = (
        "compile_prefill_seconds",
        "compile_decode_seconds",
        "prefill_runtime_seconds",
        "decode_runtime_seconds",
        "time_to_first_token_seconds",
        "total_seconds",
        "generated_tokens",
        "tokens_per_second",
        "prefill_cache_reused",
        "decode_cache_reused",
        "generation_mode",
        "preprocess_diagnostics",
    )
    selected_metrics = {key: metrics.get(key) for key in metric_keys if key in metrics}
    selected_metrics["prefill_cache_reused"] = cache_reused(
        metrics, "prefill_cache_reused", "compile_prefill_seconds"
    )
    selected_metrics["decode_cache_reused"] = cache_reused(
        metrics, "decode_cache_reused", "compile_decode_seconds"
    )
    return {
        "status": response.get("status"),
        "translation": response.get("translation"),
        "client_elapsed_seconds": round(float(client_elapsed_seconds), 6),
        "inference_seconds": response.get("inference_seconds"),
        "metrics": selected_metrics,
        "runtime": response.get("runtime") or {},
        "worker_id": response.get("worker_id"),
        "job_id": response.get("job_id"),
    }


def cache_reused(metrics: dict[str, Any], flag_key: str, compile_key: str) -> bool:
    if flag_key in metrics:
        return metrics.get(flag_key) is True
    compile_seconds = metrics.get(compile_key)
    return (
        isinstance(compile_seconds, (int, float))
        and not isinstance(compile_seconds, bool)
        and compile_seconds == 0
    )


def hot_cache_reuse_errors(response: dict[str, Any]) -> list[str]:
    metrics = response.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    pairs = (
        ("prefill_cache_reused", "compile_prefill_seconds"),
        ("decode_cache_reused", "compile_decode_seconds"),
    )
    return [flag_key for flag_key, compile_key in pairs if not cache_reused(metrics, flag_key, compile_key)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run semantic PRIME + HOT acceptance against one async endpoint"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--payload-file", type=Path, required=True)
    parser.add_argument("--expectation-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--prime-poll-interval", type=float, default=2.0)
    parser.add_argument("--hot-poll-interval", type=float, default=0.05)
    parser.add_argument("--hot-runs", type=int, default=2)
    args = parser.parse_args()

    payload = json.loads(args.payload_file.read_text(encoding="utf-8"))
    expectation = json.loads(args.expectation_file.read_text(encoding="utf-8"))
    requester = api_smoke.make_http_requester(
        args.base_url,
        args.api_key,
        args.request_timeout,
    )

    report: dict[str, Any] = {
        "schema": "translategemma-acceptance-v1",
        "name": args.name,
        "endpoint": args.path,
        "started_at": time.time(),
        "memory_before": memory_snapshot(),
        "phases": [],
    }

    for phase in phase_names(args.hot_runs):
        poll_interval = (
            args.prime_poll_interval if phase == "prime" else args.hot_poll_interval
        )
        started = time.monotonic()

        def progress(event: dict[str, Any], *, current_phase: str = phase) -> None:
            output = dict(event)
            output["phase"] = current_phase
            output["elapsed_seconds"] = round(time.monotonic() - started, 3)
            print(
                json.dumps({"acceptance_progress": output}, ensure_ascii=False),
                file=sys.stderr,
                flush=True,
            )

        try:
            status, response = api_smoke.submit_and_wait(
                requester,
                args.path,
                payload,
                timeout=args.timeout,
                poll_interval=poll_interval,
                progress_fn=progress,
            )
        except TimeoutError as exc:
            report["phases"].append({
                "phase": phase,
                "passed": False,
                "error": str(exc),
                "memory_after": memory_snapshot(),
            })
            break

        elapsed = time.monotonic() - started
        phase_report: dict[str, Any] = {
            "phase": phase,
            "http_status": status,
            "result": summarize_result(response, client_elapsed_seconds=elapsed),
            "response": response,
            "memory_after": memory_snapshot(),
        }
        try:
            semantic = validate_translation(response, expectation)
            phase_report["semantic"] = semantic
            cache_errors = hot_cache_reuse_errors(response) if phase.startswith("hot-") else []
            phase_report["cache_reuse_errors"] = cache_errors
            phase_report["passed"] = status == 200 and not cache_errors
        except SemanticValidationError as exc:
            phase_report["semantic"] = json.loads(str(exc))
            phase_report["passed"] = False
        report["phases"].append(phase_report)
        if not phase_report["passed"]:
            break

    report["finished_at"] = time.time()
    report["memory_after"] = memory_snapshot()
    report["passed"] = (
        len(report["phases"]) == len(phase_names(args.hot_runs))
        and all(bool(item.get("passed")) for item in report["phases"])
    )

    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "name": args.name,
        "passed": report["passed"],
        "report_file": str(args.report_file),
        "phases": [
            {
                "phase": item.get("phase"),
                "passed": item.get("passed"),
                "client_elapsed_seconds": (item.get("result") or {}).get("client_elapsed_seconds"),
                "inference_seconds": (item.get("result") or {}).get("inference_seconds"),
                "metrics": (item.get("result") or {}).get("metrics"),
                "translation": (item.get("result") or {}).get("translation"),
            }
            for item in report["phases"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
