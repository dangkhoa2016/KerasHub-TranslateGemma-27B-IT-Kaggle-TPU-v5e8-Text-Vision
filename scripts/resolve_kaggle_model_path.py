#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REQUIRED_MODEL_FILES = (
    "config.json",
    "preprocessor.json",
    "assets/tokenizer/vocabulary.spm",
)
WEIGHTS_MONOLITHIC = "model.weights.h5"
WEIGHTS_SHARDED_INDEX = "model.weights.json"
WEIGHTS_SHARDED_GLOB = "model_*.weights.h5"
DEFAULT_PRESET = "translategemma_27b_it"
DEFAULT_INPUT_ROOT = Path("/kaggle/input")


def model_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not all((path / relative).is_file() for relative in REQUIRED_MODEL_FILES):
        return False
    if (path / WEIGHTS_MONOLITHIC).is_file():
        return True
    return (path / WEIGHTS_SHARDED_INDEX).is_file() and any(
        path.glob(WEIGHTS_SHARDED_GLOB)
    )


def version_key(path: Path) -> tuple[int, int | str, str]:
    try:
        return (1, int(path.name), str(path))
    except ValueError:
        return (0, path.name, str(path))


def discover_from_base(base: Path) -> Path | None:
    base = base.expanduser()
    if model_complete(base):
        return base.resolve()
    if not base.is_dir():
        return None
    candidates = [child for child in base.iterdir() if model_complete(child)]
    if not candidates:
        return None
    return sorted(candidates, key=version_key, reverse=True)[0].resolve()


def _preset_roots(input_root: Path, preset_name: str):
    if not input_root.is_dir():
        return

    search_roots = []
    models_root = input_root / "models"
    if models_root.is_dir():
        search_roots.append(models_root)
    search_roots.append(input_root)

    seen: set[Path] = set()
    for root in search_roots:
        for dirpath, dirnames, _filenames in os.walk(root):
            current = Path(dirpath)
            if current.name == preset_name:
                resolved = current.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield current
                dirnames[:] = []
                continue
            dirnames[:] = [
                name
                for name in dirnames
                if name not in {".git", "__pycache__", ".cache"}
            ]


def resolve_model_path(
    *,
    preferred: Path | None,
    input_root: Path = DEFAULT_INPUT_ROOT,
    preset_name: str = DEFAULT_PRESET,
    strict_preferred: bool = False,
) -> Path:
    if preferred is not None:
        resolved = discover_from_base(preferred)
        if resolved is not None:
            return resolved
        if strict_preferred:
            raise FileNotFoundError(
                f"Explicit MODEL_PATH/MODEL_BASE is not a complete Keras preset: {preferred}"
            )

    candidates: list[Path] = []
    for preset_root in _preset_roots(input_root, preset_name):
        resolved = discover_from_base(preset_root)
        if resolved is not None and resolved not in candidates:
            candidates.append(resolved)

    if candidates:
        return sorted(candidates, key=version_key, reverse=True)[0]

    preferred_text = str(preferred) if preferred is not None else "<unset>"
    raise FileNotFoundError(
        "Could not find an attached complete Keras TranslateGemma preset. "
        f"preset={preset_name!r}, preferred={preferred_text!r}, "
        f"searched_input_root={str(input_root)!r}. "
        "Attach the Kaggle Keras TranslateGemma model containing this preset "
        "to the notebook, then restart the session and run all cells again."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve an attached Kaggle Keras model without downloading it."
    )
    parser.add_argument("--preferred", default="")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--preset-name", default=DEFAULT_PRESET)
    parser.add_argument("--strict-preferred", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preferred = Path(args.preferred) if args.preferred.strip() else None
    try:
        resolved = resolve_model_path(
            preferred=preferred,
            input_root=Path(args.input_root),
            preset_name=args.preset_name,
            strict_preferred=args.strict_preferred,
        )
    except (FileNotFoundError, OSError) as exc:
        print(f"[model-resolver] ERROR: {exc}", file=sys.stderr)
        return 2
    print(resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
