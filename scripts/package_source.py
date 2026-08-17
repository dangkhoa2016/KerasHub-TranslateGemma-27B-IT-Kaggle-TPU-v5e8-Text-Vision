#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import stat
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_TOP = {".git", "log", "state", ".private", "__pycache__"}
EXCLUDED_FILES = {
    Path(".env"),
    Path("data/api_key.txt"),
    Path("data/restart_secret.txt"),
    Path("data/tunnel_url.txt"),
    Path("PRIVATE-BACKUP-WARNING.txt"),
    Path("PRIVATE-BACKUP-MANIFEST.json"),
    Path("SOURCE-MANIFEST.sha256"),
}


def skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if rel in EXCLUDED_FILES:
        return True
    if rel.parts and rel.parts[0] in EXCLUDED_TOP:
        return True
    if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache"} for part in rel.parts):
        return True
    if path.is_file() and path.suffix in {".pyc", ".pyo", ".zip"}:
        return True
    return False


def _git_tracked_files(root: Path) -> list[Path] | None:
    try:
        top_level = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    if Path(top_level).resolve() != root.resolve():
        return None

    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "-z"],
        check=True,
        capture_output=True,
    )
    paths = [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        names = ", ".join(str(path.relative_to(root)) for path in missing)
        raise RuntimeError(f"Tracked source files are missing from the checkout: {names}")
    return paths


def _source_files(root: Path, output: Path) -> list[Path]:
    candidates = _git_tracked_files(root)
    if candidates is None:
        candidates = list(root.rglob("*"))
    return [
        path
        for path in sorted(candidates)
        if path.is_file() and path.resolve() != output and not skip(path, root)
    ]


def _manifest(files: list[Path], root: Path) -> str:
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    return "\n".join(lines) + ("\n" if lines else "")


def _load_secret_scanner():
    path = ROOT / "scripts/secret_scan.py"
    spec = importlib.util.spec_from_file_location("translategemma_secret_scan", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _zip_datetime() -> tuple[int, int, int, int, int, int]:
    raw_epoch = os.environ.get("SOURCE_DATE_EPOCH", "315532800")
    try:
        epoch = int(raw_epoch)
    except ValueError as exc:
        raise RuntimeError(f"Invalid SOURCE_DATE_EPOCH: {raw_epoch!r}") from exc
    epoch = min(max(epoch, 315532800), 4354819198)
    return time.gmtime(epoch)[:6]


def _zip_info(name: str, *, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_zip_datetime())
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = stat.S_IFREG | (0o755 if executable else 0o644)
    info.external_attr = mode << 16
    return info


def create_source_zip(
    output: Path,
    root: Path = ROOT,
    archive_prefix: str | None = None,
) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = archive_prefix or root.name
    if not prefix or Path(prefix).name != prefix:
        raise ValueError(f"Invalid archive prefix: {prefix!r}")
    files = _source_files(root, output)
    manifest = _manifest(files, root)

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            rel = path.relative_to(root)
            info = _zip_info(
                f"{prefix}/{rel.as_posix()}",
                executable=bool(path.stat().st_mode & stat.S_IXUSR),
            )
            zf.writestr(info, path.read_bytes())
        zf.writestr(_zip_info(f"{prefix}/SOURCE-MANIFEST.sha256"), manifest)

    issues = _load_secret_scanner().scan_zip(output)
    if issues:
        output.unlink(missing_ok=True)
        raise RuntimeError("Clean source secret scan failed:\n- " + "\n- ".join(issues))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default=str(ROOT.parent / (ROOT.name + ".zip")))
    parser.add_argument("--prefix")
    args = parser.parse_args()
    print(create_source_zip(Path(args.output), archive_prefix=args.prefix))


if __name__ == "__main__":
    main()
