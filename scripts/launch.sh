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
VLLM_ENV="$DATA_DIR/vllm-env"
JOB_NAME=jbutt-interactive
GPUS=2                # GPU 0: image pipeline · GPU 1: prompt enhancers
PORT=2718
PE_T2I_PORT=8001
PE_I2I_PORT=8002
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

# Weave/Forge tracing project travels from local .env into the remote env.
WEAVE_PROJECT=""
if [[ -f "$LOCAL_DIR/.env" ]] && grep -q '^WEAVE_PROJECT=' "$LOCAL_DIR/.env"; then
  WEAVE_PROJECT=$(grep '^WEAVE_PROJECT=' "$LOCAL_DIR/.env" | cut -d= -f2-)
fi
if [[ -z "${WEAVE_PROJECT}" ]]; then
  echo "warning: WEAVE_PROJECT not set in .env - app will run untraced" >&2
fi

# --- find or create the Slurm job -------------------------------------------
JOBID=$(ssh "$SSH_HOST" "squeue --me --name=$JOB_NAME -h -o %A -t RUNNING,PENDING | head -1")
if [[ -n "$JOBID" ]]; then
  # an existing job with the wrong GPU count can't be grown; replace it
  GRES=$(ssh "$SSH_HOST" "squeue -j $JOBID -h -o %b")
  if ! echo "$GRES" | grep -q "gpu:h100:$GPUS"; then
    echo "Existing job $JOBID has gres '$GRES', need gpu:h100:$GPUS — cancelling it."
    ssh "$SSH_HOST" "scancel $JOBID"
    JOBID=""
  fi
fi
if [[ -z "$JOBID" ]]; then
  echo "Submitting ${GPUS}-GPU job..."
  JOBID=$(ssh "$SSH_HOST" "sbatch --parsable --job-name=$JOB_NAME --gres=gpu:h100:$GPUS --mem=0 --cpus-per-task=16 --time=24:00:00 --wrap='sleep infinity'")
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

# --- (re)start the prompt-enhancement servers on GPU 1 ------------------------
# NB: pkill -f patterns are regexes — keep them metachar-free (the project
# path contains '+', which silently never matches)
start_pe() { # <model_id> <port> <logname> <mem_util>
  ssh "$SSH_HOST" "nohup srun --jobid=$JOBID --overlap bash $REMOTE_DIR/scripts/pe_server.sh $1 $NODE_IP $2 $4 > $DATA_DIR/$3.log 2>&1 &"
}
pe_health() { # <port> -> http code
  ssh "$SSH_HOST" "curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://$NODE_IP:$1/health" || true
}
wait_pe() { # <port> <label> -> 0 if healthy
  echo -n "Waiting for $2"
  for _ in $(seq 1 60); do
    [[ "$(pe_health "$1")" == 200 ]] && echo && return 0
    echo -n "."
    sleep 10
  done
  echo
  return 1
}
PE_FAILED=0
if [[ "$(pe_health "$PE_T2I_PORT")" == 200 && "$(pe_health "$PE_I2I_PORT")" == 200 ]]; then
  echo "Prompt enhancers already up."
else
  ssh "$SSH_HOST" "srun --jobid=$JOBID --overlap pkill -f 'bin/vllm' || true"
  sleep 3
  # Sequential start: vLLM sizes its KV cache against TOTAL GPU memory minus
  # what's already allocated, so the instances must not profile concurrently.
  echo "Starting prompt enhancers (vLLM on GPU 1, sequential)..."
  start_pe "Qwen/Qwen-Image-2.1-PE-T2I" "$PE_T2I_PORT" pe_t2i 0.45
  wait_pe "$PE_T2I_PORT" "PE-T2I (first model load takes a few minutes)" || PE_FAILED=1
  if [[ $PE_FAILED == 0 ]]; then
    # vLLM requires util*total <= free at startup; T2I already holds ~0.45
    start_pe "Qwen/Qwen-Image-2.1-PE-I2I" "$PE_I2I_PORT" pe_i2i 0.53
    wait_pe "$PE_I2I_PORT" "PE-I2I" || PE_FAILED=1
  fi
fi
if [[ $PE_FAILED == 1 ]]; then
  echo "⚠️  enhancers not healthy — continuing; the app falls back to raw prompts." >&2
  echo "    check: ssh $SSH_HOST tail $DATA_DIR/pe_t2i.log $DATA_DIR/pe_i2i.log" >&2
fi

# --- (re)start marimo inside the job -----------------------------------------
ssh "$SSH_HOST" "srun --jobid=$JOBID --overlap pkill -f 'bin/marimo' || true"
sleep 3  # let the port free up
AUTH=""
[[ "$MODE" == edit ]] && AUTH="--no-token"   # tunnel is the auth boundary
echo "Starting marimo ${MODE}..."
ssh "$SSH_HOST" "nohup srun --jobid=$JOBID --overlap bash -c 'source $REMOTE_DIR/.venv/bin/activate && export CUDA_VISIBLE_DEVICES=0 HF_HOME=$DATA_DIR/hf PE_T2I_URL=http://$NODE_IP:$PE_T2I_PORT PE_I2I_URL=http://$NODE_IP:$PE_I2I_PORT WEAVE_PROJECT=${WEAVE_PROJECT} && [ -f $DATA_DIR/.wandb_key ] && export WANDB_API_KEY=\$(cat $DATA_DIR/.wandb_key); export WANDB_DIR=$DATA_DIR/wandb WANDB_CACHE_DIR=$DATA_DIR/wandb/cache WANDB_CONFIG_DIR=$DATA_DIR/wandb/config WEAVE_USE_SERVER_CACHE=false; mkdir -p $DATA_DIR/wandb/cache $DATA_DIR/wandb/config; cd $REMOTE_DIR && exec marimo $MODE app.py --host $NODE_IP --port $PORT --headless --watch $AUTH' > $DATA_DIR/marimo.log 2>&1 &"

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
ssh -N -L "$PORT:$NODE_IP:$PORT" "$SSH_HOST" >/dev/null 2>&1 &
TUNNEL_PID=$!
disown "$TUNNEL_PID"
echo "$TUNNEL_PID" > "$STATE_DIR/tunnel.pid"
printf 'JOBID=%s\nNODE=%s\nNODE_IP=%s\nPORT=%s\nMODE=%s\n' "$JOBID" "$NODE" "$NODE_IP" "$PORT" "$MODE" > "$STATE_DIR/state"

echo
echo "✅ ready: http://localhost:$PORT  (mode: $MODE, job $JOBID on $NODE)"
echo "   logs:  ssh $SSH_HOST tail -f $DATA_DIR/marimo.log"
echo "   stop:  scripts/stop.sh [--release]"
