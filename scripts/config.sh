# Sourced by launch.sh / sync.sh / stop.sh — resolves per-user cluster settings.
# Nothing here is specific to one person: every value can be overridden, and
# the defaults are derived from YOUR cluster account.
#
# Precedence (first match wins):
#   1. environment variable          e.g. DATA_DIR=/mnt/data/me scripts/launch.sh
#   2. entry in the repo's .env      e.g. CLUSTER_USER=me+org01
#   3. derived default               CLUSTER_USER via `ssh $SSH_HOST whoami`,
#                                    DATA_DIR=/mnt/data/$CLUSTER_USER,
#                                    JOB_NAME=<user-without-+suffix>-interactive
#
# Tip: set CLUSTER_USER in .env to skip the ssh whoami round-trip per command.
# Expects LOCAL_DIR to be set by the sourcing script.

SSH_HOST="${SSH_HOST:-coreweave-login}"

_env_get() { # <key> -> value from .env, empty if absent
  if [[ -f "$LOCAL_DIR/.env" ]]; then
    grep "^$1=" "$LOCAL_DIR/.env" | head -1 | cut -d= -f2- || true
  fi
}

CLUSTER_USER="${CLUSTER_USER:-$(_env_get CLUSTER_USER)}"
if [[ -z "$CLUSTER_USER" ]]; then
  CLUSTER_USER=$(ssh "$SSH_HOST" whoami)
fi
[[ -n "$CLUSTER_USER" ]] || { echo "config.sh: could not determine CLUSTER_USER" >&2; exit 1; }

DATA_DIR="${DATA_DIR:-$(_env_get DATA_DIR)}"
DATA_DIR="${DATA_DIR:-/mnt/data/$CLUSTER_USER}"

# Slurm job names disallow nothing here, but keep them short: strip any
# +org suffix from the account name (me+org01 -> me-interactive).
JOB_NAME="${JOB_NAME:-$(_env_get JOB_NAME)}"
JOB_NAME="${JOB_NAME:-${CLUSTER_USER%%+*}-interactive}"

REMOTE_DIR="$DATA_DIR/qwen2.1"
