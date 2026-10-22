#!/usr/bin/env bash
# Push the local project (source of truth) to the cluster. If dependency files
# changed, re-sync the remote venv with uv and pull the updated uv.lock back.
set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# SSH_HOST / CLUSTER_USER / DATA_DIR / REMOTE_DIR — per-user, see config.sh
source "$LOCAL_DIR/scripts/config.sh"
UV="UV_CACHE_DIR=$DATA_DIR/cache/uv UV_PYTHON_INSTALL_DIR=$DATA_DIR/uv-python UV_PYTHON_PREFERENCE=only-managed $DATA_DIR/bin/uv"

FORCE_DEPS=0
[[ "${1:-}" == "--deps" ]] && FORCE_DEPS=1

EXCLUDES=(
  --exclude=.venv --exclude=.git --exclude=.env --exclude=outputs/ --exclude=.launch/
  --exclude=.ipynb_checkpoints/ --exclude=__pycache__/ --exclude=.marimo/
  --exclude=__marimo__/ --exclude=.DS_Store
)

echo "Syncing $LOCAL_DIR -> $SSH_HOST:$REMOTE_DIR"
CHANGES=$(rsync -ai --delete "${EXCLUDES[@]}" "$LOCAL_DIR/" "$SSH_HOST:$REMOTE_DIR/")
echo "$CHANGES" | grep -v '^\.' || true

# W&B/Forge key travels outside the rsync tree: local .env -> chmod-600 file
if [[ -f "$LOCAL_DIR/.env" ]] && grep -q '^WANDB_API_KEY=' "$LOCAL_DIR/.env"; then
  grep '^WANDB_API_KEY=' "$LOCAL_DIR/.env" | cut -d= -f2- \
    | ssh "$SSH_HOST" "umask 077 && cat > $DATA_DIR/.wandb_key"
fi

if [[ $FORCE_DEPS == 1 ]] || echo "$CHANGES" | grep -qE 'pyproject\.toml|uv\.lock'; then
  echo "Dependency files changed — running remote uv sync…"
  ssh "$SSH_HOST" "cd $REMOTE_DIR && $UV sync"
  # the remote re-lock is authoritative; bring it home
  rsync -a "$SSH_HOST:$REMOTE_DIR/uv.lock" "$LOCAL_DIR/uv.lock"
fi

echo "Done."
