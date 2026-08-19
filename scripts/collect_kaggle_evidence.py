#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

GIB = 1024 ** 3
FORBIDDEN_PARTS = {".git", "__pycache__"}
FORBIDDEN_EXACT = {
    ".env",
    "data/api_key.txt",
    "data/restart_secret.txt",
    "data/tunnel_url.txt",
}
SAFE_ENV_KEYS = (
    "EXPECTED_TPU_DEVICES",
    "EXPECTED_LIBTPU_VERSION",
    "TPU_PREFLIGHT_MODE",
    "REQUIRE_V5E8",
    "MESH_SHAPE",
    "MESH_AXIS_NAMES",
    "DATA_PARALLEL_AXIS",
    "MODEL_PARALLEL_AXIS",
    "MODEL_DTYPE",
    "VISION_ENABLED",
    "GENERATION_SPLIT_COMPILE",
    "GENERATION_BUCKETING",
    "GENERATION_LENGTH_BUCKETS",
    "VISION_MIN_GENERATION_LENGTH",
    "DEFAULT_OUTPUT_TOKENS",
    "MAX_OUTPUT_TOKENS",
    "MEMORY_GUARD_GIB",
    "MEMORY_POLL_SECONDS",
    "JAX_PLATFORMS",
    "JAX_COMPILATION_CACHE_DIR",
)
SENSITIVE_JSON_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|bearer(?:[_-]?token)?|"
    r"credential(?:s)?|password|private[_-]?key|(?:restart[_-]?)?secret|"
    r"(?:github|gh)[_-]?token|github_pat|token)",
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\bgh[pousr]_[a-z0-9]{20,}\b"),
    re.compile(r"(?i)\bgithub_pat_[a-z0-9_]{20,}\b"),
    re.compile(
        r"(?im)(?:^|[,{\s])[\"']?(?:api[_-]?key|access[_-]?token|"
        r"authorization|credential|password|private[_-]?key|secret|token)"
        r"[\"']?\s*[:=]\s*[\"']?\S+"
    ),
)


def is_forbidden_archive_name(name: str) -> bool:
    normalized = Path(name).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    path = Path(normalized)
    if normalized in FORBIDDEN_EXACT:
        return True
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    if path.suffix == ".pid" or (path.parts and path.parts[0] == "state"):
        return True
    return False


def changed_source_paths(status_text: str) -> list[str]:
    paths: list[str] = []
    for raw in status_text.splitlines():
        if len(raw) < 3:
            continue
        if len(raw) >= 3 and raw[2] == " ":
            candidate = raw[3:].strip()
        elif len(raw) >= 2 and raw[1] == " ":
            candidate = raw[2:].strip()
        else:
            continue
        if " -> " in candidate:
            candidate = candidate.rsplit(" -> ", 1)[-1]
        if not candidate or is_forbidden_archive_name(candidate):
            continue
        paths.append(candidate)
    return list(dict.fromkeys(paths))


def require_clean_git_tree(status_text: str) -> None:
    if status_text.strip():
        raise RuntimeError(
            "GitHub-only Kaggle evidence collection requires a clean git status --porcelain=v1 tree"
        )


def require_regular_input(path: Path, allowed_root: Path, *, label: str) -> Path:
    """Reject symlinked and escaping inputs before evidence code reads or copies them."""
    allowed_root = allowed_root.resolve()
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlinked {label}: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Required {label} is not a readable regular file: {path}") from exc
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing out-of-root {label}: {path}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"Required {label} is not a regular file: {path}")
    return resolved


def _credential_json_value(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if SENSITIVE_JSON_KEY.fullmatch(str(key)) and nested not in (None, "", False, [], {}):
                return True
            if _credential_json_value(nested):
                return True
    elif isinstance(value, list):
        return any(_credential_json_value(item) for item in value)
    return False


def scan_evidence_member(path: Path, *, runtime_secrets: list[str]) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Evidence member is not a regular file: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    for secret in runtime_secrets:
        if secret and secret in text:
            raise RuntimeError(f"Runtime secret leaked into evidence member: {path.name}")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        raise RuntimeError(f"Credential material leaked into evidence member: {path.name}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return
    if _credential_json_value(payload):
        raise RuntimeError(f"Credential-like JSON value leaked into evidence member: {path.name}")


def read_simple_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def memory_snapshot() -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for label, path in (
        ("current_gib", Path("/sys/fs/cgroup/memory.current")),
        ("peak_gib", Path("/sys/fs/cgroup/memory.peak")),
    ):
        try:
            raw = path.read_text(encoding="utf-8").strip()
            result[label] = None if raw == "max" else round(int(raw) / GIB, 3)
        except (OSError, ValueError):
            result[label] = None
    return result


def request_json(base_url: str, api_key: str, path: str) -> dict[str, Any] | None:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return None


def runtime_endpoint_evidence(
    data: dict[str, Any] | None,
    endpoint: str,
    *,
    reason: str = "unavailable",
) -> dict[str, Any]:
    """Return endpoint data or an explicit machine-readable unavailable marker."""
    if data is not None:
        return data
    return {"available": False, "endpoint": endpoint, "reason": reason}


def redact_text(text: str, secrets: list[str]) -> str:
    output = text
    for secret in secrets:
        if secret:
            output = output.replace(secret, "<redacted-runtime-secret>")
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a sanitized Kaggle review evidence ZIP")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--acceptance-dir",
        type=Path,
        default=Path("/kaggle/working/translategemma-27b-v100-acceptance"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = args.output or Path(
        f"/kaggle/working/translategemma-27b-v100-evidence-{timestamp}.zip"
    )
    output = output.resolve()

    def git(*parts: str) -> str:
        return subprocess.check_output(
            ["git", *parts], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()

    try:
        status = git("status", "--porcelain=v1")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Cannot verify GitHub-only evidence source tree") from exc
    require_clean_git_tree(status)

    api_key_path = root / "data/api_key.txt"
    restart_path = root / "data/restart_secret.txt"
    runtime_secrets = []
    for path in (api_key_path, restart_path):
        if path.exists() or path.is_symlink():
            path = require_regular_input(path, root, label="runtime secret input")
            value = path.read_text(encoding="utf-8", errors="ignore").strip()
            if value:
                runtime_secrets.append(value)

    with tempfile.TemporaryDirectory(prefix="tg27b-evidence-") as temp_name:
        evidence = Path(temp_name) / "evidence"
        evidence.mkdir(parents=True)

        try:
            head = git("rev-parse", "HEAD")
        except subprocess.CalledProcessError:
            head = "unknown"
        try:
            diff = subprocess.check_output(
                ["git", "diff", "--binary", "--", "."],
                cwd=root,
                text=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            diff = exc.output or ""

        env_path = root / ".env"
        env = read_simple_env(
            require_regular_input(env_path, root, label="environment input")
            if env_path.exists() or env_path.is_symlink()
            else env_path
        )
        safe_env = {key: env.get(key) for key in SAFE_ENV_KEYS if key in env}
        summary = {
            "schema": "translategemma-kaggle-evidence-v1",
            "created_at_utc": timestamp,
            "git_head": head,
            "git_status_porcelain": status.splitlines() if status else [],
            "memory": memory_snapshot(),
            "safe_env": safe_env,
        }
        write_json(evidence / "run-summary.json", summary)
        (evidence / "git-diff.patch").write_text(
            redact_text(diff, runtime_secrets), encoding="utf-8"
        )

        source_notebook = root / "notebooks/kaggle-tpu-v5e8-text-vision.ipynb"
        if source_notebook.exists() or source_notebook.is_symlink():
            source_notebook = require_regular_input(
                source_notebook, root, label="source notebook"
            )
            shutil.copy2(source_notebook, evidence / "source-notebook.ipynb")

        if args.acceptance_dir.is_symlink():
            raise RuntimeError(f"Refusing symlinked acceptance directory: {args.acceptance_dir}")
        if args.acceptance_dir.is_dir():
            acceptance_root = args.acceptance_dir.resolve()
            for source in sorted(args.acceptance_dir.glob("*.json")):
                source = require_regular_input(
                    source, acceptance_root, label="acceptance JSON"
                )
                target = evidence / source.name
                target.write_text(
                    redact_text(source.read_text(encoding="utf-8"), runtime_secrets),
                    encoding="utf-8",
                )

        base_url = f"http://127.0.0.1:{env.get('PORT', '7860')}"
        api_key = runtime_secrets[0] if api_key_path.is_file() and runtime_secrets else ""
        if api_key:
            info = request_json(base_url, api_key, "/info")
            health = request_json(base_url, api_key, "/health/ready?all=1&details=1")
            info_payload = runtime_endpoint_evidence(info, "/info")
            health_payload = runtime_endpoint_evidence(
                health, "/health/ready?all=1&details=1"
            )
        else:
            info_payload = runtime_endpoint_evidence(
                None, "/info", reason="api_key_unavailable"
            )
            health_payload = runtime_endpoint_evidence(
                None,
                "/health/ready?all=1&details=1",
                reason="api_key_unavailable",
            )
        write_json(evidence / "info-final.json", info_payload)
        write_json(evidence / "health-final.json", health_payload)

        sys.path.insert(0, str(root / "scripts"))
        from sanitize_runtime_log import sanitize_runtime_log_text  # type: ignore

        sanitizer_summary: dict[str, Any] = {}
        for source_name in ("server.log", "server.stdout.log"):
            source = root / "log" / source_name
            if not source.exists() and not source.is_symlink():
                continue
            source = require_regular_input(source, root, label="runtime log")
            raw = redact_text(
                source.read_text(encoding="utf-8", errors="replace"),
                runtime_secrets,
            )
            cleaned, counts = sanitize_runtime_log_text(raw)
            (evidence / f"{source_name}.sanitized.txt").write_text(
                cleaned, encoding="utf-8"
            )
            sanitizer_summary[source_name] = counts
        write_json(evidence / "runtime-log-sanitizer-summary.json", sanitizer_summary)

        manifest: dict[str, str] = {}
        for path in sorted(p for p in evidence.rglob("*") if p.is_file()):
            rel = path.relative_to(evidence).as_posix()
            if is_forbidden_archive_name(rel):
                raise RuntimeError(f"Forbidden evidence member: {rel}")
            scan_evidence_member(path, runtime_secrets=runtime_secrets)
            manifest[rel] = sha256_file(path)
        write_json(evidence / "SHA256SUMS.json", manifest)

        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(p for p in evidence.rglob("*") if p.is_file()):
                rel = path.relative_to(evidence).as_posix()
                if is_forbidden_archive_name(rel):
                    raise RuntimeError(f"Forbidden evidence member: {rel}")
                scan_evidence_member(path, runtime_secrets=runtime_secrets)
                archive.write(path, rel)

    digest = sha256_file(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"EVIDENCE_ZIP={output}")
    print(f"EVIDENCE_SHA256={digest}")
    print(f"EVIDENCE_SHA256_FILE={sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
