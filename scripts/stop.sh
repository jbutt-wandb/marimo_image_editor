#!/usr/bin/env bash
# Tear down the app's sandboxes (billing stops). All logic in scripts/sandbox.py.
#   scripts/stop.sh                     # keep the snapshot for fast relaunch
#   scripts/stop.sh --delete-snapshot   # also drop the snapshot (re-seed needed)
set -euo pipefail
cd "$(dirname "$0")/.."
# --python 3.12: cwsandbox needs >=3.11 (repo .python-version is 3.10)
exec uv run --python 3.12 scripts/sandbox.py down "$@"
