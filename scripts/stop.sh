#!/usr/bin/env bash
# Tear down: close the tunnel and stop marimo. --release also cancels the
# Slurm job (freeing the GPU).
set -euo pipefail

SSH_HOST="${SSH_HOST:-coreweave-login}"
DATA_DIR=/mnt/data/jbutt+cwb607
REMOTE_DIR="$DATA_DIR/qwen2.1"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$LOCAL_DIR/.launch"

RELEASE=0
[[ "${1:-}" == "--release" ]] && RELEASE=1

if [[ -f "$STATE_DIR/tunnel.pid" ]]; then
  kill "$(cat "$STATE_DIR/tunnel.pid")" 2>/dev/null && echo "Tunnel closed." || true
  rm -f "$STATE_DIR/tunnel.pid"
fi

JOBID=""
[[ -f "$STATE_DIR/state" ]] && JOBID=$(sed -n 's/^JOBID=//p' "$STATE_DIR/state")
[[ -z "$JOBID" ]] && JOBID=$(ssh "$SSH_HOST" "squeue --me --name=jbutt-interactive -h -o %A -t RUNNING | head -1")

if [[ -n "$JOBID" ]]; then
  ssh "$SSH_HOST" "srun --jobid=$JOBID --overlap pkill -f '$REMOTE_DIR/.venv/bin/marimo' || true" \
    && echo "marimo stopped."
  if [[ $RELEASE == 1 ]]; then
    ssh "$SSH_HOST" "scancel $JOBID" && echo "Job $JOBID cancelled."
  else
    echo "Job $JOBID left running (use --release to free the GPU)."
  fi
else
  echo "No running job found."
fi
