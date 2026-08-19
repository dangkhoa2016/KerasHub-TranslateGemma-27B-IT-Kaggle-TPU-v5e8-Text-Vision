#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

tag="${1:?usage: build_release_artifacts.sh TAG [OUTPUT_DIR]}"
output_dir="${2:-$ROOT_DIR/dist}"

if [[ "$tag" != "v1.0.0" ]]; then
  echo "Unsupported public release tag: $tag (only v1.0.0 is allowed)" >&2
  exit 2
fi
python3 "$ROOT_DIR/scripts/release_contract.py" "$tag"

mkdir -p "$output_dir"
output_dir="$(realpath "$output_dir")"
base="KerasHub-TranslateGemma-27B-IT-Kaggle-TPU-v5e8-Text-Vision-${tag}"
archive="$output_dir/${base}.zip"
notebook="$output_dir/${base}-kaggle.ipynb"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$ROOT_DIR" show -s --format=%ct HEAD)}"

rm -f "$archive" "$notebook" "$archive.sha256" "$archive.md5" "$notebook.sha256" "$notebook.md5"
python3 "$ROOT_DIR/scripts/package_source.py" "$archive" --prefix "$base"
cp "$ROOT_DIR/notebooks/kaggle-tpu-v5e8-text-vision.ipynb" "$notebook"
python3 "$ROOT_DIR/scripts/secret_scan.py" "$archive"
unzip -t "$archive"
(
  cd "$output_dir"
  sha256sum "${base}.zip" > "${base}.zip.sha256"
  sha256sum "${base}-kaggle.ipynb" > "${base}-kaggle.ipynb.sha256"
  md5sum "${base}.zip" > "${base}.zip.md5"
  md5sum "${base}-kaggle.ipynb" > "${base}-kaggle.ipynb.md5"
)
