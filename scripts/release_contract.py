#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RELEASE_TAG = "v1.0.0"
PUBLIC_RELEASE_VERSION = "1.0.0"
PUBLIC_TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".json", ".sh", ".txt", ".example", ".ipynb"}
FORBIDDEN_PUBLIC_TAG = re.compile(
    r"\bv(?:0|1|2|3|4|5|6|7|8|9)\.(?!0\.0\b)\d+(?:\.\d+)*(?:[-.]\w+)?",
    re.IGNORECASE,
)


def project_version(root: Path = ROOT) -> str:
    text = (root / "src/translategemma_server/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise RuntimeError("cannot determine project __version__")
    return match.group(1)


def forbidden_public_release_tags(root: Path = ROOT) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        if path.suffix.lower() not in PUBLIC_TEXT_SUFFIXES and path.name not in {"NOTICE", ".env.example"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits.extend((path.relative_to(root).as_posix(), match.group(0)) for match in FORBIDDEN_PUBLIC_TAG.finditer(text))
    return hits


def validate_release_contract(tag: str, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    version = project_version(root)
    if version != PUBLIC_RELEASE_VERSION:
        errors.append(f"only public package version {PUBLIC_RELEASE_VERSION} is allowed, got {version}")
    if tag != PUBLIC_RELEASE_TAG:
        errors.append(f"only public release tag {PUBLIC_RELEASE_TAG} is allowed, got {tag}")
    forbidden_tags = forbidden_public_release_tags(root)
    if forbidden_tags:
        errors.append(f"forbidden public release tags: {forbidden_tags}")
    for path in (
        root / "RELEASE_NOTES_v1.0.0.md",
        root / "RELEASE_NOTES_v1.0.0.vi.md",
        root / "docs/RELEASE-EVIDENCE-v1.0.0.md",
        root / "docs/RELEASE-EVIDENCE-v1.0.0.vi.md",
    ):
        if not path.is_file():
            errors.append(f"missing release document: {path.relative_to(root)}")
    allowed_release_docs = {"RELEASE_NOTES_v1.0.0.md", "RELEASE_NOTES_v1.0.0.vi.md"}
    extra_release_docs = [p for p in root.glob("RELEASE_NOTES_v*.md") if p.name not in allowed_release_docs]
    allowed_evidence_docs = {"RELEASE-EVIDENCE-v1.0.0.md", "RELEASE-EVIDENCE-v1.0.0.vi.md"}
    extra_evidence_docs = [p for p in (root / "docs").glob("RELEASE-EVIDENCE-v*.md") if p.name not in allowed_evidence_docs]
    if extra_release_docs or extra_evidence_docs:
        errors.append("release documents for versions other than v1.0.0 are forbidden")
    for name in ("README.md", "README.vi.md"):
        text = (root / name).read_text(encoding="utf-8")
        if "**Release:** `v1.0.0`" not in text:
            errors.append(f"{name} does not identify current release v1.0.0")
    for name in ("CHANGELOG.md", "CHANGELOG.vi.md"):
        text = (root / name).read_text(encoding="utf-8")
        if text.count("## v1.0.0 —") != 1:
            errors.append(f"{name} must contain exactly one v1.0.0 release heading")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the single public v1.0.0 release contract")
    parser.add_argument("tag")
    args = parser.parse_args()
    errors = validate_release_contract(args.tag)
    if errors:
        print("[release-contract] FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"[release-contract] PASS: only {PUBLIC_RELEASE_TAG} / {PUBLIC_RELEASE_VERSION} is allowed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
