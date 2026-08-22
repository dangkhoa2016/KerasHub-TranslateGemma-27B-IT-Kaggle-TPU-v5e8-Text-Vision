#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

pid_file="$ROOT_DIR/state/server.pid"
worker_pid_file="$ROOT_DIR/state/worker.pid"
coordinator_pid=""
worker_pid=""

coordinator_pid="$(read_managed_pid "$pid_file" "src/server.py" 2>/dev/null || true)"
worker_pid="$(read_managed_pid "$worker_pid_file" "multiprocessing.spawn" 2>/dev/null || true)"

if [[ -z "$coordinator_pid" && -z "$worker_pid" ]]; then
  echo "Server is not running."
  exit 0
fi

if [[ -n "$coordinator_pid" ]]; then
  echo "Stopping coordinator PID $coordinator_pid; waiting for managed TPU worker shutdown."
  kill -TERM "$coordinator_pid" 2>/dev/null || true
elif [[ -n "$worker_pid" ]]; then
  echo "Coordinator is absent; cleaning orphan managed TPU worker PID $worker_pid." >&2
  kill -TERM "$worker_pid" 2>/dev/null || true
fi

deadline=$((SECONDS + ${STOP_WAIT_SECONDS:-330}))
while (( SECONDS < deadline )); do
  coordinator_alive=false
  worker_alive=false
  [[ -n "$coordinator_pid" ]] && kill -0 "$coordinator_pid" 2>/dev/null && coordinator_alive=true
  [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null && worker_alive=true
  if [[ "$coordinator_alive" == false && "$worker_alive" == false ]]; then
    rm -f "$pid_file" "$worker_pid_file"
    echo "Coordinator and TPU worker stopped."
    exit 0
  fi
  sleep 1
done

if [[ -n "$coordinator_pid" ]] && pid_matches_cmdline "$coordinator_pid" "src/server.py"; then
  echo "Coordinator graceful stop timed out; sending SIGKILL to PID $coordinator_pid." >&2
  kill -KILL "$coordinator_pid" 2>/dev/null || true
fi
if [[ -n "$worker_pid" ]] && pid_matches_cmdline "$worker_pid" "multiprocessing.spawn"; then
  echo "TPU worker graceful stop timed out; sending SIGKILL to PID $worker_pid." >&2
  kill -KILL "$worker_pid" 2>/dev/null || true
fi

for _ in $(seq 1 10); do
  coordinator_alive=false
  worker_alive=false
  [[ -n "$coordinator_pid" ]] && kill -0 "$coordinator_pid" 2>/dev/null && coordinator_alive=true
  [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null && worker_alive=true
  if [[ "$coordinator_alive" == false && "$worker_alive" == false ]]; then
    rm -f "$pid_file" "$worker_pid_file"
    echo "Coordinator and TPU worker stopped after forced cleanup."
    exit 0
  fi
  sleep 1
done

echo "ERROR: managed coordinator or TPU worker is still alive after forced cleanup." >&2
exit 1
