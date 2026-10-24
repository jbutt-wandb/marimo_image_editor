#!/usr/bin/env bash
# One command to use the app — all logic lives in scripts/sandbox.py.
#   scripts/launch.sh          # create/reuse the sandboxes, print+open the URL
#   scripts/launch.sh --seed   # one-time: build the FSS snapshot (venvs + weights)
set -euo pipefail
cd "$(dirname "$0")/.."
# --python 3.12: cwsandbox needs >=3.11; the repo's .python-version (3.10)
# would otherwise win over the script's own requires-python.
[[ "${1:-}" == "--seed" ]] && { shift; exec uv run --python 3.12 scripts/sandbox.py seed "$@"; }
exec uv run --python 3.12 scripts/sandbox.py up "$@"
