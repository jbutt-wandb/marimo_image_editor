# Image Studio

A [marimo](https://marimo.io) app for working with instruction-driven image
models — text-to-image, instruction editing, multi-reference composition, and
draw-on-image region editing — running on a CoreWeave H100 and used locally
through an SSH tunnel. The currently hosted model is **Qwen-Image-2.1**; the
UI and library are structured so other models can slot in behind the same
interface (`src/qwen2_1/model.py` is the only model-specific layer).

## Layout

| Path | What |
|---|---|
| `app.py` | The marimo app (4 tabs: Create · Edit · Region · Multi-region + gallery) |
| `src/qwen2_1/` | Library: pipeline wrapper, mask/annotation helpers, PencilCanvas widget, recipe storage, PE client |
| `scripts/` | sync / launch / stop / pe_server (runs on the node) |
| `scratchpad.ipynb` | Original Jupyter experiments (reference, unused by the app) |

This local repo is the **source of truth**. The cluster copy at
`/mnt/data/jbutt+cwb607/qwen2.1/` (SSH host `coreweave-login`) holds the venv,
the HF model cache, and executes everything.

## Usage

```bash
scripts/launch.sh              # sync → allocate GPU (or reuse job) → marimo run → tunnel
# open http://localhost:2718
scripts/launch.sh --mode edit  # same, but editable notebook UI
scripts/stop.sh                # stop marimo + tunnel, keep the GPU job
scripts/stop.sh --release      # …and cancel the Slurm job
scripts/sync.sh                # push code only (marimo --watch hot-reloads app.py)
```

## Prompt enhancement

The job allocates **2 GPUs**: GPU 0 runs the image pipeline, GPU 1 hosts the two
official prompt-enhancement models (`…-PE-T2I`, `…-PE-I2I`, Qwen3.5-VL 9B) as
vLLM servers (`scripts/pe_server.sh`, dedicated venv `/mnt/data/…/vllm-env`).
Every prompt is implicitly rewritten into a detailed, generation-optimised one
before hitting the image model (`src/qwen2_1/enhance.py`, mirroring the official
`prompt_rewrite/` protocol — system prompts vendored in `src/qwen2_1/pe_prompts/`).
Recipes record both `original_prompt` and the rewritten `prompt`. The "Auto"
aspect ratio uses the enhancer's suggested `wh_ratio`. Toggle off under
Advanced; if the servers are down the app silently falls back to raw prompts.

Notes:
- First generation loads the model (~38 GB bf16, a couple of minutes). Only one
  process may hold it on the 80 GB H100.
- `sync.sh` runs a remote `uv sync` when `pyproject.toml`/`uv.lock` change and
  copies the re-locked `uv.lock` back.
- `app.py` edits hot-reload via `--watch`; changes under `src/` need a restart
  (`scripts/launch.sh --no-sync` is enough — it restarts marimo).
- The venv must stay on uv-managed Python (3.10): compute nodes lack the dev
  headers system Python needs for Triton's JIT.

## Current model: Qwen-Image-2.1

One `QwenImage21Pipeline` serves every mode; `image=` selects it
(none = T2I, one = edit, list ≤ 10 = multi-reference). There is **no mask
input**: region edits bake colored annotations into the image and name them in
the prompt ("in the red marked region, …"). Guidance (`true_cfg_scale` > 1)
only takes effect together with a negative prompt. Explicit width/height are
required for 2K output in T2I; edits inherit the input size.
