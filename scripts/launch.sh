#!/usr/bin/env bash
# One command to use the app: sync code, allocate (or reuse) the GPU job,
# start marimo on the compute node, open the SSH tunnel, print the URL.
#
#   scripts/launch.sh                # app mode (marimo run)
#   scripts/launch.sh --mode edit    # notebook mode (marimo edit)
#   scripts/launch.sh --no-sync      # skip the rsync push
set -euo pipefail

SSH_HOST="${SSH_HOST:-coreweave-login}"
DATA_DIR=/mnt/data/jbutt+cwb607
REMOTE_DIR="$DATA_DIR/qwen2.1"
JOB_NAME=jbutt-interactive
PORT=2718
MODE=run
SYNC=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --no-sync) SYNC=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
[[ "$MODE" == run || "$MODE" == edit ]] || { echo "--mode must be run or edit" >&2; exit 1; }

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$LOCAL_DIR/.launch"
mkdir -p "$STATE_DIR"

[[ $SYNC == 1 ]] && "$LOCAL_DIR/scripts/sync.sh"

# --- find or create the Slurm job -------------------------------------------
JOBID=$(ssh "$SSH_HOST" "squeue --me --name=$JOB_NAME -h -o %A -t RUNNING,PENDING | head -1")
if [[ -z "$JOBID" ]]; then
  echo "No existing job — submitting…"
  JOBID=$(ssh "$SSH_HOST" "sbatch --parsable --job-name=$JOB_NAME --gres=gpu:h100:1 --mem=0 --cpus-per-task=16 --time=24:00:00 --wrap='sleep infinity'")
fi
echo "Job: $JOBID"

STATE=""
for _ in $(seq 1 60); do
  STATE=$(ssh "$SSH_HOST" "squeue -j $JOBID -h -o %T" || true)
  [[ "$STATE" == RUNNING ]] && break
  echo "  waiting for allocation (state: ${STATE:-unknown})…"
  sleep 5
done
[[ "$STATE" == RUNNING ]] || { echo "Job never reached RUNNING" >&2; exit 1; }

NODE=$(ssh "$SSH_HOST" "squeue -j $JOBID -h -o %N")
NODE_IP=$(ssh "$SSH_HOST" "scontrol show node $NODE" | sed -n 's/.*NodeAddr=\([^ ]*\).*/\1/p')
echo "Node: $NODE ($NODE_IP)"

# --- (re)start marimo inside the job -----------------------------------------
ssh "$SSH_HOST" "srun --jobid=$JOBID --overlap pkill -f '$REMOTE_DIR/.venv/bin/marimo' || true"
AUTH=""
[[ "$MODE" == edit ]] && AUTH="--no-token"   # tunnel is the auth boundary
echo "Starting marimo $MODE…"
ssh "$SSH_HOST" "nohup srun --jobid=$JOBID --overlap bash -c 'source $REMOTE_DIR/.venv/bin/activate && export HF_HOME=$DATA_DIR/hf && cd $REMOTE_DIR && exec marimo $MODE app.py --host $NODE_IP --port $PORT --headless --watch $AUTH' > $DATA_DIR/marimo.log 2>&1 &"

echo -n "Waiting for marimo"
for _ in $(seq 1 48); do
  CODE=$(ssh "$SSH_HOST" "curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://$NODE_IP:$PORT/" || true)
  [[ "$CODE" == 200 ]] && break
  echo -n "."
  sleep 5
done
echo
[[ "${CODE:-}" == 200 ]] || { echo "marimo did not come up — check $DATA_DIR/marimo.log" >&2; exit 1; }

# --- tunnel -------------------------------------------------------------------
if [[ -f "$STATE_DIR/tunnel.pid" ]]; then
  kill "$(cat "$STATE_DIR/tunnel.pid")" 2>/dev/null || true
  rm -f "$STATE_DIR/tunnel.pid"
fi
ssh -N -L "$PORT:$NODE_IP:$PORT" "$SSH_HOST" &
TUNNEL_PID=$!
disown "$TUNNEL_PID"
echo "$TUNNEL_PID" > "$STATE_DIR/tunnel.pid"
printf 'JOBID=%s\nNODE=%s\nNODE_IP=%s\nPORT=%s\nMODE=%s\n' "$JOBID" "$NODE" "$NODE_IP" "$PORT" "$MODE" > "$STATE_DIR/state"

echo
echo "✅ ready: http://localhost:$PORT  (mode: $MODE, job $JOBID on $NODE)"
echo "   logs:  ssh $SSH_HOST tail -f $DATA_DIR/marimo.log"
echo "   stop:  scripts/stop.sh [--release]"
