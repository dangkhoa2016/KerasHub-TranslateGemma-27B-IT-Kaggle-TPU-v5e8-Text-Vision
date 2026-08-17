#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env
key="$(api_key)"
[[ -n "$key" ]] || { echo "API key unavailable" >&2; exit 1; }

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
cat > "$tmp" <<'JSON'
{
  "text": "Good morning! How are you?",
  "source_lang": "English",
  "target_lang": "Vietnamese",
  "max_new_tokens": 128
}
JSON

acceptance_dir="${ACCEPTANCE_OUTPUT_DIR:-/kaggle/working/translategemma-27b-v100-acceptance}"
mkdir -p "$acceptance_dir"

python3 scripts/run_acceptance.py \
  --name text \
  --base-url "$(server_base_url)" \
  --api-key "$key" \
  --path /translate/async \
  --payload-file "$tmp" \
  --expectation-file assets/text-smoke.expectation.json \
  --report-file "$acceptance_dir/text-acceptance.json" \
  --timeout "${SMOKE_TIMEOUT:-1800}" \
  --request-timeout "${SMOKE_REQUEST_TIMEOUT:-30}" \
  --prime-poll-interval "${SMOKE_PRIME_POLL_INTERVAL:-2}" \
  --hot-poll-interval "${SMOKE_HOT_POLL_INTERVAL:-0.05}" \
  --hot-runs "${SMOKE_HOT_RUNS:-2}"
