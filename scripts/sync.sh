#!/usr/bin/env bash
# Push local code (source of truth) to the running studio sandbox and restart
# marimo. All logic lives in scripts/sandbox.py.
set -euo pipefail
cd "$(dirname "$0")/.."
# --python 3.12: cwsandbox needs >=3.11 (repo .python-version is 3.10)
exec uv run --python 3.12 scripts/sandbox.py sync "$@"
