#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


class SemanticValidationError(ValueError):
    pass


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _normalized_terms(values: list[Any] | tuple[Any, ...]) -> list[str]:
    return [term for term in (normalize_text(str(v)) for v in values) if term]


def validate_translation(
    result: dict[str, Any],
    expectation: dict[str, Any],
) -> dict[str, Any]:
    translation = str(result.get("translation") or "").strip()
    normalized = normalize_text(translation)
    errors: list[dict[str, Any]] = []
    matched_concepts: list[str] = []

    if result.get("status") != "completed":
        errors.append({"type": "job_status", "value": result.get("status")})
    if not normalized:
        errors.append({"type": "empty_translation"})

    for index, concept in enumerate(expectation.get("required_concepts") or []):
        name = str(concept.get("name") or f"concept_{index}")
        alternatives = _normalized_terms(concept.get("any_of") or [])
        if alternatives and any(term in normalized for term in alternatives):
            matched_concepts.append(name)
        else:
            errors.append({
                "type": "missing_concept",
                "name": name,
                "any_of": alternatives,
            })

    source_echo = [
        term
        for term in _normalized_terms(expectation.get("forbidden_source_echo") or [])
        if term in normalized
    ]
    if source_echo:
        errors.append({"type": "source_echo", "matches": source_echo})

    confusions = [
        term
        for term in _normalized_terms(expectation.get("known_confusions") or [])
        if term in normalized
    ]
    if confusions:
        errors.append({"type": "known_confusion", "matches": confusions})

    report = {
        "passed": not errors,
        "matched_concepts": matched_concepts,
        "translation": translation,
        "errors": errors,
    }
    if errors:
        raise SemanticValidationError(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a completed translation semantically")
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--expectation-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path)
    args = parser.parse_args()

    result = json.loads(args.result_file.read_text(encoding="utf-8"))
    expectation = json.loads(args.expectation_file.read_text(encoding="utf-8"))
    try:
        report = validate_translation(result, expectation)
    except SemanticValidationError as exc:
        report = json.loads(str(exc))
        if args.report_file:
            args.report_file.parent.mkdir(parents=True, exist_ok=True)
            args.report_file.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
