#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

RequestFn = Callable[[str, str, dict[str, Any] | None], tuple[int, dict[str, Any]]]


def submit_and_wait(
    request_fn: RequestFn,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    poll_interval: float,
    progress_fn: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int, dict[str, Any]]:
    status, data = request_fn("POST", path, payload)
    if status != 202:
        return status, data

    job_id = data.get("job_id")
    if not job_id:
        return status, data

    last_job_status = None

    def report(current_status: int, current_data: dict[str, Any]) -> None:
        nonlocal last_job_status
        if progress_fn is None:
            return
        job_status = str(current_data.get("status") or f"http-{current_status}")
        if job_status == last_job_status:
            return
        last_job_status = job_status
        progress_fn({
            "job_id": job_id,
            "status": job_status,
            "http_status": current_status,
        })

    report(status, data)
    deadline = time.monotonic() + timeout
    result_path = f"/result/{job_id}"
    while time.monotonic() < deadline:
        if poll_interval:
            time.sleep(poll_interval)
        status, data = request_fn("GET", result_path, None)
        report(status, data)
        if status != 202:
            return status, data
    raise TimeoutError(f"Job {job_id} did not finish within {timeout:g}s")


def make_http_requester(base_url: str, api_key: str, request_timeout: float) -> RequestFn:
    base_url = base_url.rstrip("/")

    def request_json(
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        body = None
        headers = {"Authorization": f"Bearer {api_key}"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                raw = response.read().decode("utf-8")
                return response.status, json.loads(raw)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"error": raw or exc.reason}
            return exc.code, data

    return request_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--payload-file", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=620)
    parser.add_argument("--poll-interval", type=float, default=2)
    args = parser.parse_args()

    payload = json.loads(Path(args.payload_file).read_text(encoding="utf-8"))
    requester = make_http_requester(
        args.base_url,
        args.api_key,
        args.request_timeout,
    )
    started = time.monotonic()

    def report_progress(event: dict[str, Any]) -> None:
        output = dict(event)
        output["elapsed_seconds"] = round(time.monotonic() - started, 3)
        print(
            json.dumps({"smoke_progress": output}, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )

    try:
        status, data = submit_and_wait(
            requester,
            args.path,
            payload,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            progress_fn=report_progress,
        )
    except TimeoutError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1

    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
