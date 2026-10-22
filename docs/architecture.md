# Architecture

Image Studio is a marimo app served headless from a CoreWeave Slurm compute
node and used from a Mac over an SSH tunnel. One Slurm job holds two H100s:
GPU 0 runs the diffusers image pipeline inside the marimo process, GPU 1 hosts
two vLLM servers for the official Qwen prompt-enhancement (PE) models. All
observability goes to Weave (W&B / CoreWeave Forge).

## Components

```
┌─ Mac ───────────────────────────────┐
│  browser → http://localhost:2718    │
│  scripts/launch.sh · sync.sh ·      │
│  stop.sh   (.env holds WANDB key)   │
└───────┬──────────────┬──────────────┘
        │ ssh tunnel   │ rsync + sbatch/srun via ssh
        │ 2718         ▼
        │      ┌─ coreweave-login ─┐
        │      │  Slurm control    │
        │      └─────────┬─────────┘
        ▼                ▼
┌─ compute node — Slurm job "$JOB_NAME" (2× H100 80GB, 24 h) ─────────────────┐
│                                                                             │
│  GPU 0                                   GPU 1                              │
│  ┌─────────────────────────────┐         ┌─────────────────────────────┐    │
│  │ marimo run app.py           │  HTTP   │ vLLM: …-PE-T2I  :8001       │    │
│  │  --headless --watch :2718   │ ──────▶ │ vLLM: …-PE-I2I  :8002       │    │
│  │ ├ OpenAI-Agents PE agents   │         │ (Qwen3.5-VL 9B rewriters,   │    │
│  │ ├ QwenImage21Pipeline bf16  │         │  bf16, reasoning-parser     │    │
│  │ │  (~38 GB resident,        │         │  qwen3, venv: vllm-env)     │    │
│  │ │   ~66 GB peak @ 2K)       │         └─────────────────────────────┘    │
│  │ └ weave tracing + dataset ──┼──▶ $WEAVE_PROJECT (from .env)              │
│  └─────────────────────────────┘                                            │
│                                                                             │
│  $DATA_DIR/: qwen2.1/ (code + .venv, py3.10) · vllm-env/                    │
│  (py3.12) · hf/ (model cache) · .wandb_key · *.log                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

The local repo is the source of truth; `scripts/sync.sh` pushes it to
`$DATA_DIR/qwen2.1/`, which is what actually executes.

## Request lifecycle — what happens when a user clicks Generate

The same flow serves all four tabs (Create / Edit / Region / Multi-region);
only the input assembly differs. Everything below runs inside the marimo
process; `studio.generate(...)` is the per-click entry point and the Weave
trace root.

1. **Input assembly (app.py)**
   - *Create*: just the prompt (+ aspect/resolution/transparent settings).
   - *Edit*: 1–10 uploaded images; the first is the edit target, the rest are
     references.
   - *Region / Multi-region*: the user's canvas strokes are rasterized to a
     mask and **baked into the image as colored boundary annotations**
     (`masks.py` — the model has no mask input, see
     [docs/model.md](model.md)). Multi-region composes one prompt from the
     per-color instructions ("In the red marked region, X. In the green
     marked region, Y.").

2. **Trace root opens.** `generate` is a `weave.op`; it records mode, session
   id, model ids and whether enhancement is on as trace attributes. Logged
   images are thumbnailed (≤1024 px) so traces stay small.

3. **Prompt enhancement (agent 1).** If the toggle is on (default), a
   **PromptEnhancer** agent — T2I for Create, I2I for the edit modes — runs
   against the matching vLLM endpoint via the OpenAI Agents SDK
   (`OpenAIChatCompletionsModel`; vLLM has no Responses API). This follows the
   official Qwen PE protocol: vendored system prompts in
   `src/qwen2_1/pe_prompts/`, images-first content order for I2I (downscaled
   to ≤1024² data URLs), and per-task sampling — notably `presence_penalty`
   1.5 (T2I) vs 0.0 (I2I), which is not interchangeable. The model answers
   with reasoning followed by a JSON object; the code strips `</think>`, then
   a **balanced-brace scanner** extracts the last valid JSON (tolerating the
   `rewrited_prompt` typo from the training data) into
   `{rewritten_prompt, wh_ratio, ratio_follow}`. Any failure — server down,
   unparseable output, missing env URL — **fails open to the raw prompt**.

4. **Aspect resolution (Create only).** The "Auto" aspect choice maps the
   enhancer's suggested `wh_ratio` onto the nearest supported 2K resolution
   (`enhance.nearest_aspect`); falls back to 1:1.

5. **Image generation (agent 2).** A synthetic image-task agent
   (ImageGenerator / ImageEditor / RegionEditor / MultiRegionEditor per mode)
   wraps the diffusers call: `run_pipe` in `model.py` drives the singleton
   `QwenImage21Pipeline` on GPU 0. Mode is implicit in `image=` (None / single
   / list). The pipeline **loads lazily on the first generation** (~2–4 min,
   ~38 GB bf16). CFG is only passed when `true_cfg_scale > 1` *and* a negative
   prompt is set, since the pipeline ignores either alone. A random seed is
   drawn if none was given and recorded.

6. **Recipe + display.** Every parameter needed to reproduce the result —
   final prompt, `original_prompt`, PE model, wh_ratio, seed, steps, CFG,
   mode, sizes — is collected into a "recipe" dict, shown with the results
   inline and in the gallery.

7. **Dataset logging.** `distill.log_generation` appends one row per action to
   the versioned `weave.Dataset` **`image-generation-log`**: the full recipe,
   input images at PE resolution, and the full-resolution output (base64
   PNG) — accumulated as distillation/fine-tuning data. The append runs on a
   background thread (read–rebuild–republish, serialized by a lock; the
   server content-addresses media so unchanged images dedupe) and never
   raises into the app. **Weave is the system of record for outputs** —
   images are not durably stored on cluster disk.

8. **Trace root closes.** In the Weave UI the click appears as one trace
   containing the enhancer agent run and the pipeline op; the Agents tab lists
   all agents.

## Compute requirements

| Resource | Details |
|---|---|
| Slurm job | `$JOB_NAME` (default `<user>-interactive`), `sbatch --gres=gpu:h100:2 --mem=0 --cpus-per-task=16 --time=24:00:00`, wraps `sleep infinity`; services attach with `srun --overlap` |
| GPU 0 | `Qwen/Qwen-Image-2.1` via diffusers (`QwenImage21Pipeline`, bf16): ~38 GB resident, **~66 GB peak during 2K inference** — exactly one process may hold it |
| GPU 1 | 2× vLLM, bf16, `--max-num-seqs 16`, `--max-model-len 24576`, `--reasoning-parser qwen3`, `--limit-mm-per-prompt {"image": 10}` |
| PE-T2I | `Qwen/Qwen-Image-2.1-PE-T2I` (Qwen3.5-VL 9B) — port **8001**, `--gpu-memory-utilization 0.45` |
| PE-I2I | `Qwen/Qwen-Image-2.1-PE-I2I` (Qwen3.5-VL 9B) — port **8002**, `--gpu-memory-utilization 0.53` |
| marimo | port **2718** on the node IP; reached via `ssh -N -L 2718:<node_ip>:2718` |
| App venv | `$DATA_DIR/qwen2.1/.venv` — **uv-managed Python 3.10** (system python lacks dev headers → Triton JIT breaks) |
| vLLM venv | `$DATA_DIR/vllm-env` — Python 3.12, separate from the app venv |
| Storage | `$DATA_DIR/`: `qwen2.1/` (code), `hf/` (`HF_HOME` cache), `.wandb_key` (chmod 600), `wandb/` (W&B dirs), `*.log` |

Environment exported into the marimo process by `launch.sh`:
`CUDA_VISIBLE_DEVICES=0`, `HF_HOME`, `PE_T2I_URL` / `PE_I2I_URL`
(`http://<node_ip>:8001/8002`), `WANDB_API_KEY` (from `.wandb_key`),
`WANDB_DIR`/`WANDB_CACHE_DIR`/`WANDB_CONFIG_DIR`, and
`WEAVE_USE_SERVER_CACHE=false`.

## Scripts

| Script | Runs on | What it does |
|---|---|---|
| `scripts/launch.sh [--mode run\|edit] [--port N] [--no-sync]` | Mac | Full bring-up: sync → find-or-create the Slurm job (replaces a job with the wrong GPU count) → wait for RUNNING, resolve the node IP → health-check/start the PE servers → kill+restart marimo with the env above → open the tunnel. State (job id, node, pid) in `.launch/` |
| `scripts/sync.sh [--deps]` | Mac | rsync push (excludes `.venv`, `.git`, `.env`, `outputs/`, caches); copies `WANDB_API_KEY` from `.env` to the chmod-600 cluster file; on `pyproject.toml`/`uv.lock` change runs remote `uv sync` and pulls the authoritative re-locked `uv.lock` back |
| `scripts/stop.sh [--release]` | Mac | Kills the tunnel, `pkill`s marimo and vLLM inside the job; `--release` also `scancel`s the job |
| `scripts/pe_server.sh <model> <host> <port> <mem_util>` | node | One vLLM server on GPU 1 (`CUDA_VISIBLE_DEVICES=1`), launched by `launch.sh` via `srun --overlap`; flags mirror the official `prompt_rewrite/serve.sh` |

## Failure and fallback behavior

| Failure | Behavior |
|---|---|
| No `WANDB_API_KEY` / weave init error | App runs untraced; `weave.op`s execute as plain functions; dataset logging is a no-op |
| PE server down / unreachable / bad JSON | `enhance_prompt` fails open: the raw user prompt goes straight to the pipeline (`enhanced: false` in the recipe) |
| PE servers never become healthy at launch | `launch.sh` warns and continues — the app still works with raw prompts |
| Dataset append fails | Logged (`logger.exception`) and swallowed; generation already succeeded. If the previous dataset version can't be loaded, a fresh single-row version is published loudly (older versions remain recoverable) |
| Existing job has wrong GPU count | `launch.sh` cancels and resubmits (a Slurm job can't be grown) |
| marimo doesn't come up | `launch.sh` exits with a pointer to `marimo.log` |
| Guidance without negative prompt (or vice versa) | Pipeline ignores it — the app only forwards CFG when both are set |

## Known gotchas

- **One model per GPU 0.** The pipeline is ~38 GB resident and peaks ~66 GB at
  2K on an 80 GB H100 — exactly one process may hold it (it's a module-level
  singleton in `model.py` for this reason). A second loader OOMs both.
- **marimo `--watch` reloads `app.py` only.** Changes under `src/` need a
  marimo restart: `scripts/launch.sh --no-sync` (or a full `launch.sh`).
- **SQLite on network filesystems.** `/mnt/data` is networked storage; Weave's
  local server cache (sqlite) misbehaves there, hence
  `WEAVE_USE_SERVER_CACHE=false` in the launch env. Keep it.
- **`pkill -f` patterns are regexes.** The project path contains `+`, so a
  naive `pkill -f <path>` silently never matches. The scripts match on
  `bin/marimo` / `bin/vllm` instead — keep new patterns metachar-free.
- **vLLM `--gpu-memory-utilization` semantics.** It is this server's budget as
  a fraction of **TOTAL** GPU memory, and vLLM requires that much to be
  **FREE at startup**. Two servers sharing GPU 1 therefore use 0.45 then 0.53,
  and must start **sequentially** — concurrent startup races the free-memory
  profiling and the KV-cache sizing.
- **uv-managed Python only** (3.10, `UV_PYTHON_PREFERENCE=only-managed`).
  Compute nodes' system python lacks the dev headers Triton's JIT needs.
- **Node IP changes per allocation.** `launch.sh` resolves it from
  `scontrol show node` each run and wires it into the PE URLs and tunnel;
  don't hardcode it.
- **`uv.lock` is re-locked remotely.** After a remote `uv sync`, the cluster's
  lock is authoritative and `sync.sh` copies it back — don't hand-edit locally.
