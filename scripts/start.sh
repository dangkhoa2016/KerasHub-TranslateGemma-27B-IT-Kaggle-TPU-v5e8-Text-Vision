#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

model_preferred="${MODEL_PATH:-${MODEL_BASE:-/kaggle/input/models/keras/translategemma/keras/translategemma_27b_it}}"
resolver_args=(
  --preferred "$model_preferred"
  --input-root "${KAGGLE_INPUT_ROOT:-/kaggle/input}"
  --preset-name translategemma_27b_it
)
if [[ -n "${MODEL_PATH:-}" ]]; then
  resolver_args+=(--strict-preferred)
fi
MODEL_PATH="$(python3 scripts/resolve_kaggle_model_path.py "${resolver_args[@]}")"
export MODEL_PATH="$MODEL_PATH"
echo "[model-resolver] resolved attached preset: $MODEL_PATH"

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/configure_kaggle_tpu.sh"
pid_file="$ROOT_DIR/state/server.pid"
worker_pid_file="$ROOT_DIR/state/worker.pid"
if pid="$(read_managed_pid "$pid_file" "src/server.py" 2>/dev/null)"; then
  echo "Server already running (PID $pid)."
  exit 0
fi
if worker_pid="$(read_managed_pid "$worker_pid_file" "multiprocessing.spawn" 2>/dev/null)"; then
  echo "Refusing to start: managed TPU worker PID $worker_pid is still alive. Run scripts/stop.sh first." >&2
  exit 1
fi
archive_dir="$ROOT_DIR/log/archive"
mkdir -p "$archive_dir"
stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
for name in server.log server.stdout.log; do
  if [[ -s "$ROOT_DIR/log/$name" ]]; then
    mv "$ROOT_DIR/log/$name" "$archive_dir/${name%.log}-$stamp.log"
  fi
done
: > "$ROOT_DIR/log/server.stdout.log"
nohup python3 -u src/server.py > log/server.stdout.log 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$pid_file"

base="$(server_base_url)"
timeout="${START_LIVENESS_TIMEOUT:-30}"
deadline=$((SECONDS + timeout))
while (( SECONDS < deadline )); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Waitress coordinator exited before liveness became ready." >&2
    tail -n 120 "$ROOT_DIR/log/server.stdout.log" >&2 || true
    rm -f "$pid_file"
    exit 1
  fi
  if curl -fsS --max-time 2 "$base/health/live" >/dev/null 2>&1; then
    echo "Started Waitress coordinator PID $pid; liveness verified at $base/health/live"
    echo "Readiness stays 503 while the single TPU worker loads all 8 devices."
    exit 0
  fi
  sleep 1
done

echo "Waitress coordinator did not become live within ${timeout}s." >&2
tail -n 120 "$ROOT_DIR/log/server.stdout.log" >&2 || true
kill -TERM "$pid" 2>/dev/null || true
rm -f "$pid_file"
exit 1
