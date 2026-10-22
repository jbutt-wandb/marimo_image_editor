#!/usr/bin/env bash
# Runs ON the compute node (via srun from launch.sh): one vLLM server for a
# prompt-enhancement model on GPU 1. Flags mirror the official
# prompt_rewrite/serve.sh (reasoning parser, mm limit, bf16).
#   pe_server.sh <model_id> <host> <port> <gpu_mem_util>
# gpu_mem_util is this server's own budget as a fraction of TOTAL GPU memory,
# and vLLM requires that much to be FREE at startup — the two PE instances
# share GPU 1, so the first uses 0.45 and the second 0.53, started
# sequentially (concurrent startup makes the free-memory math race).
set -euo pipefail

MODEL=$1; HOST=$2; PORT=$3; MEM_UTIL=$4
# launch.sh passes DATA_DIR through srun env; fall back to this node account
DATA_DIR="${DATA_DIR:-/mnt/data/$(whoami)}"

export CUDA_VISIBLE_DEVICES=1
export HF_HOME="$DATA_DIR/hf"
# vllm shells out to ninja (torch.compile); the venv's bin must be on PATH
export PATH="$DATA_DIR/vllm-env/bin:$PATH"

exec "$DATA_DIR/vllm-env/bin/vllm" serve "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 24576 \
  --gpu-memory-utilization "$MEM_UTIL" \
  --max-num-seqs 16 \
  --limit-mm-per-prompt '{"image": 10}' \
  --reasoning-parser qwen3
