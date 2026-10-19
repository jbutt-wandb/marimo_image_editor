# Image Studio

A [marimo](https://marimo.io) web app for working with instruction-driven image
models — text-to-image, instruction editing, multi-reference composition, and
draw-on-image region editing — running on a CoreWeave Slurm cluster (2× H100)
and used from your Mac through an SSH tunnel. The hosted model is
**Qwen-Image-2.1**, fronted by the official Qwen prompt-enhancement models and
orchestrated as agents (OpenAI Agents SDK), with every generation traced to
[Weave](https://wandb.ai/wandb-smle/jb_image_editing_tool/weave).

| Doc | What |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Components, request lifecycle, compute table, scripts, failure modes, gotchas |
| [`docs/model.md`](docs/model.md) | Qwen-Image-2.1 capabilities, resolutions, prompting tips, PE models |

## Prerequisites

- **Mac with SSH access to the cluster** via a host alias `coreweave-login` in
  `~/.ssh/config` (the scripts use `ssh coreweave-login …` for everything;
  override with `SSH_HOST=<alias>`).
- **Local `.env`** in the repo root containing `WANDB_API_KEY=…` (gitignored).
  `sync.sh` copies the key — outside the rsync tree — into a `chmod 600` file
  on the cluster (`/mnt/data/jbutt+cwb607/.wandb_key`); without it the app
  simply runs untraced.
- **[uv](https://docs.astral.sh/uv/)** installed locally (and available on the
  cluster at `/mnt/data/jbutt+cwb607/bin/uv` — used for remote dependency syncs).
- **Cluster storage** under `/mnt/data/jbutt+cwb607/`: the project copy
  (`qwen2.1/` with its uv venv), a separate vLLM venv (`vllm-env/`), and the
  HF model cache (`hf/`). This local repo is the **source of truth**; the
  cluster copy executes.

## Running it

```bash
# One command, from the repo root on your Mac:
scripts/launch.sh
# 1. rsync the repo to the cluster (scripts/sync.sh)
# 2. find or submit the 2-GPU Slurm job (name jbutt-interactive, 24 h)
# 3. health-check / start the two prompt-enhancement vLLM servers on GPU 1
# 4. (re)start marimo on GPU 0 with all env wired up
# 5. open the SSH tunnel localhost:2718 -> compute node

# then open:
open http://localhost:2718
```

Variants and teardown:

```bash
scripts/launch.sh --mode edit   # editable marimo notebook UI instead of the app
scripts/launch.sh --no-sync     # skip the rsync (also the way to restart marimo
                                # after changing code under src/)
scripts/sync.sh                 # push code only; app.py hot-reloads via --watch
scripts/stop.sh                 # stop marimo + PE servers + tunnel, KEEP the GPU job
scripts/stop.sh --release       # ... and cancel the Slurm job (frees both GPUs)
```

Timing expectations on a cold start:

- **PE servers**: ~5–10 min the first time (two 9B models load sequentially on
  GPU 1). `launch.sh` waits for their `/health` endpoints and continues even if
  they fail (the app then uses raw prompts).
- **First generation**: the image pipeline loads lazily on the first click —
  allow **~2–4 min** (~38 GB bf16 onto GPU 0). Later generations skip this.

Logs live on the cluster: `ssh coreweave-login tail -f
/mnt/data/jbutt+cwb607/marimo.log` (also `pe_t2i.log`, `pe_i2i.log`).

## Tracing

When `WANDB_API_KEY` is present, every user action becomes one Weave trace in
**[wandb-smle/jb_image_editing_tool](https://wandb.ai/wandb-smle/jb_image_editing_tool/weave)**
(W&B / CoreWeave Forge): the prompt-enhancement agent run and the pipeline call
nest under a per-click root, agents appear in the Agents tab, and each output
image is appended — with its full parameter recipe and base64 input/output
images — to the versioned `image-generation-log` dataset, which doubles as
distillation/fine-tuning data. Weave is the system of record for generated
images; nothing is durably stored on cluster disk. Without the key, the app
runs identically but untraced.
