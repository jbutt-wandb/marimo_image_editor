#!/usr/bin/env bash
# Push the local project (source of truth) to the cluster. If dependency files
# changed, re-sync the remote venv with uv and pull the updated uv.lock back.
set -euo pipefail

SSH_HOST="${SSH_HOST:-coreweave-login}"
DATA_DIR=/mnt/data/jbutt+cwb607
REMOTE_DIR="$DATA_DIR/qwen2.1"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UV="UV_CACHE_DIR=$DATA_DIR/cache/uv UV_PYTHON_INSTALL_DIR=$DATA_DIR/uv-python UV_PYTHON_PREFERENCE=only-managed $DATA_DIR/bin/uv"

FORCE_DEPS=0
[[ "${1:-}" == "--deps" ]] && FORCE_DEPS=1

EXCLUDES=(
  --exclude=.venv --exclude=.git --exclude=outputs/ --exclude=.launch/
  --exclude=.ipynb_checkpoints/ --exclude=__pycache__/ --exclude=.marimo/
  --exclude=__marimo__/ --exclude=.DS_Store
)

echo "Syncing $LOCAL_DIR -> $SSH_HOST:$REMOTE_DIR"
CHANGES=$(rsync -ai --delete "${EXCLUDES[@]}" "$LOCAL_DIR/" "$SSH_HOST:$REMOTE_DIR/")
echo "$CHANGES" | grep -v '^\.' || true

if [[ $FORCE_DEPS == 1 ]] || echo "$CHANGES" | grep -qE 'pyproject\.toml|uv\.lock'; then
  echo "Dependency files changed — running remote uv sync…"
  ssh "$SSH_HOST" "cd $REMOTE_DIR && $UV sync"
  # the remote re-lock is authoritative; bring it home
  rsync -a "$SSH_HOST:$REMOTE_DIR/uv.lock" "$LOCAL_DIR/uv.lock"
fi

echo "Done."
