# Architecture

Image Studio runs on **two CoreWeave serverless GPU sandboxes** (serverless
caps at 1 GPU per sandbox) and is used from a Mac over a tokenized public
HTTPS URL. The **studio** sandbox runs marimo with the diffusers image
pipeline in-process; the **pe** sandbox hosts two vLLM servers for the
official Qwen prompt-enhancement (PE) models, sharing its single GPU. Both
mount an instant fork of a filesystem snapshot (built once by seeding) at
`/work`. All observability goes to Weave (W&B / CoreWeave Forge).

## Components

```
┌─ Mac ───────────────────────────────────────────────┐
│  browser → https://2718-<studio-id>.…cwsandbox.com  │
│            /?access_token=<APP_TOKEN>               │
│  scripts/launch.sh · sync.sh · stop.sh              │
│   (thin wrappers around scripts/sandbox.py)         │
│  .env: CWSANDBOX_API_KEY (+ WANDB key, optional)    │
│  .launch/sandbox-state.json: ids, URLs, tokens      │
└───────┬─────────────────────────────┬───────────────┘
        │ HTTPS :2718                 │ cwsandbox SDK: create /
        │ (marimo token auth)         │ exec / upload code
        ▼                             ▼
┌─ sandbox "studio" (1 GPU) ───────┐    ┌─ sandbox "pe" (1 GPU) ──────────┐
│ marimo run app.py                │    │ vLLM: …-PE-T2I  :8001 (0.45)    │
│  --headless --watch :2718        │    │ vLLM: …-PE-I2I  :8002 (0.53)    │
│  --token (APP_TOKEN)             │    │ (Qwen3.5-VL 9B rewriters, bf16, │
│ ├ OpenAI-Agents PE agents ───────┼──▶ │  reasoning-parser qwen3,        │
│ │   HTTPS + bearer PE_API_KEY    │    │  --api-key $PE_API_TOKEN,       │
│ ├ QwenImage21Pipeline bf16       │    │  started sequentially)          │
│ │   (~38 GB resident,            │    │                                 │
│ │    ~66 GB peak @ 2K)           │    │ /work ◀─ snapshot fork          │
│ └ weave tracing ──▶ $WEAVE_PROJECT    └─────────────────────────────────┘
│                                  │
│ /work ◀─ snapshot fork           │      one seeded FSS snapshot (200 Gi):
└──────────────────────────────────┘      app/ (code + .venv py3.10) ·
                                          vllm-env/ (py3.12) · hf/ (~80 GB
                                          weights) · bin/uv · logs/
```

The local repo is the source of truth; `launch.sh`/`sync.sh` tarball it to
`/work/app` in the studio sandbox, which is what actually executes. Studio
reaches the PE servers over their **public HTTPS endpoints**: the platform
terminates TLS but does no caller auth, so vLLM enforces a bearer token
(`--api-key`) and marimo runs with token auth on — the tokens live in the
sandbox environments and in `.launch/sandbox-state.json` (chmod 600).

## Launcher lifecycle

All logic is in `scripts/sandbox.py` (a standalone uv PEP 723 script —
cwsandbox needs Python ≥3.11 while the app venv is 3.10, so run it with
`uv run --no-project --python 3.12 scripts/sandbox.py <cmd>`; the wrappers do
this for you, and `--no-project` stops uv from rebuilding the repo's 3.10 app
venv first). State persists in `.launch/sandbox-state.json`. Operational
knobs are subcommand flags and `.env` (keys, `SNAPSHOT_ID` adoption,
`SANDBOX_IMAGE` custom-image override); the topology — which sandboxes exist,
their ports, models, and resources — is deliberately the constants block at
the top of `sandbox.py`, not CLI args. That block is the configuration
surface: edit it, then reseed/relaunch.

**The architecture itself is yours to change.** The two-sandbox split (app +
pipeline on one GPU, both enhancers sharing the other) is just the layout
that mirrors the original cluster — not a constraint. To redistribute the
GPUs, edit `sandbox.py` directly; the relevant pieces are deliberately small
and legible:

- *Shape-preserving tweaks* (swap a model, move a port, resize cpu/memory,
  change the memory split between the two vLLM servers): one-line edits to
  the constants block or the `0.45`/`0.53` arguments in `cmd_up`.
- *Structural changes* (e.g. a third GPU sandbox so each enhancer gets its
  own card — removes the memory split and the sequential start; or dropping
  the enhancers entirely for a single-sandbox app): copy the ~30-line
  `_create_pe` / `_start_vllm` pattern per new sandbox and adjust the
  bootstrap order in `cmd_up`. Each sandbox is created by one `_create_*`
  function and bootstrapped by one `_start_*` function — that pairing is the
  whole pattern.
- *Multi-GPU in one sandbox* (restoring the original single-box layout)
  requires CKS placement rather than serverless — see the gotchas below;
  it's a different platform mode, not a constants edit.

After any topology change: `reseed` if the snapshot contents changed (new
models/venvs), then `stop.sh` + `launch.sh` to roll the new layout.

| Command | What happens |
|---|---|
| `sandbox.py probe` | ~1-min GPU entitlement check (GPU sandboxes are private preview): creates a 1-GPU sandbox, prints `nvidia-smi -L`, deletes it. Unentitled orgs fail at create in <2 s, nothing billed |
| `launch.sh --seed` (`seed`) | One-time, **CPU-only** sandbox (~30–60 min): installs uv under `/work/bin`, uploads the repo, builds the app venv (`uv sync`, py 3.10) and a separate vLLM venv (py 3.12), downloads all three models into `/work/hf`, snapshots `/work`, saves `snapshot_id`, deletes the seed sandbox |
| `sandbox.py reseed` | **Incremental snapshot refresh** (minutes, not another full seed): fork the current snapshot onto a CPU-only sandbox, apply only the delta (code push, `uv sync`, vLLM kernel wheels, any missing models), mid-life snapshot the running fork, save the new `snapshot_id`, delete the old snapshot once the new one is READY (`--keep-old` skips that). Running sandboxes are untouched. Full `seed` remains the path for clean rebuilds (reseed only ever adds to the mount) |
| `launch.sh` (`up`) | Reuse if both sandboxes are RUNNING (re-push code, restart marimo); else create studio+pe **in parallel** (~1 min to admit, ~4 min to restore the fork), each on a snapshot fork → start PE-T2I detached → upload code + `uv sync` + start marimo → wait for health (PE-I2I starts after T2I is healthy) → print+open the tokenized URL. Ready ~3–5 min. On failure, reaps everything by tag. `--snapshot-id <id>` (or `SNAPSHOT_ID` in `.env`) adopts an existing org snapshot into state — a new developer on a seeded team skips seeding entirely (org snapshots should be adoptable; if the restore fails, seed yourself) |
| `sync.sh` (`sync`) | Push local code to the running studio sandbox, `uv sync` (no-op unless deps changed), restart marimo |
| `stop.sh` (`down`) | Delete every sandbox tagged `image-studio` — billing stops; the snapshot is kept. `--delete-snapshot` also deletes the snapshot (next launch needs `--seed`) |
| `sandbox.py status` | List this app's sandboxes with role, state, and URLs; show launch time and snapshot id |

## Request lifecycle — what happens when a user clicks Generate

The same flow serves all four tabs (Create / Edit / Region / Multi-region);
only the input assembly differs. Everything below runs inside the marimo
process on the studio sandbox; `studio.generate(...)` is the per-click entry
point and the Weave trace root.

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
   against the matching vLLM endpoint (`PE_T2I_URL` / `PE_I2I_URL`, the pe
   sandbox's public HTTPS URLs, authenticated with the `PE_API_KEY` bearer
   token) via the OpenAI Agents SDK (`OpenAIChatCompletionsModel`; vLLM has
   no Responses API). This follows the official Qwen PE protocol: vendored
   system prompts in `src/qwen2_1/pe_prompts/`, images-first content order
   for I2I (downscaled to ≤1024² data URLs), and per-task sampling — notably
   `presence_penalty` 1.5 (T2I) vs 0.0 (I2I), which is not interchangeable.
   The model answers with reasoning followed by a JSON object; the code
   strips `</think>`, then a **balanced-brace scanner** extracts the last
   valid JSON (tolerating the `rewrited_prompt` typo from the training data)
   into `{rewritten_prompt, wh_ratio, ratio_follow}`. Any failure — server
   down, unparseable output, missing env URL — **fails open to the raw
   prompt**.

4. **Aspect resolution (Create only).** The "Auto" aspect choice maps the
   enhancer's suggested `wh_ratio` onto the nearest supported 2K resolution
   (`enhance.nearest_aspect`); falls back to 1:1.

5. **Image generation (agent 2).** A synthetic image-task agent
   (ImageGenerator / ImageEditor / RegionEditor / MultiRegionEditor per mode)
   wraps the diffusers call: `run_pipe` in `model.py` drives the singleton
   `QwenImage21Pipeline` on the studio sandbox's GPU. Mode is implicit in
   `image=` (None / single / list). The pipeline **loads lazily on the first
   generation** (~2–4 min, ~38 GB bf16). CFG is only passed when
   `true_cfg_scale > 1` *and* a negative prompt is set, since the pipeline
   ignores either alone. A random seed is drawn if none was given and
   recorded.

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
   images are not durably stored on sandbox disk.

8. **Trace root closes.** In the Weave UI the click appears as one trace
   containing the enhancer agent run and the pipeline op; the Agents tab lists
   all agents.

## Compute requirements

Both sandboxes: image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
(works on both GPU archs the pool serves), `placement_mode="serverless"`,
`max_lifetime_seconds` 24 h, tags `image-studio` + `image-studio-<role>`,
main command `sleep infinity` (servers launch detached), and a 200 Gi fork of
the seeded snapshot at `/work`.

| | studio sandbox | pe sandbox |
|---|---|---|
| Resources | 16 CPU · 64 Gi · **1 GPU** | 16 CPU · 64 Gi · **1 GPU** |
| GPU (pool is heterogeneous, not selectable) | H100 80 GB **or** RTX PRO 6000 Blackwell 96 GB | same |
| Runs | `marimo run app.py --headless --watch` + `QwenImage21Pipeline` bf16 (~38 GB resident, **~66 GB peak @ 2K** — exactly one process may hold it) | vLLM × 2, bf16, `--max-num-seqs 16`, `--max-model-len 24576`, `--reasoning-parser qwen3`, `--limit-mm-per-prompt {"image": 10}` |
| Ports (public HTTPS endpoints) | **2718** (app) | **8001** PE-T2I (`--gpu-memory-utilization 0.45`) · **8002** PE-I2I (`0.53`), started **sequentially** |
| App-level auth | marimo `--token --token-password $APP_TOKEN` → URL carries `?access_token=…` | vLLM `--api-key $PE_API_TOKEN`; studio presents it as `PE_API_KEY` bearer |
| Venv on `/work` | `app/.venv` — **uv-managed Python 3.10** | `vllm-env` — Python 3.12 |
| Logs | `/work/logs/marimo.log` | `/work/logs/pe_8001.log`, `pe_8002.log` |

Environment injected into the studio sandbox / marimo process by the
launcher: `CUDA_VISIBLE_DEVICES=0`, `HF_HOME=/work/hf`, `APP_TOKEN`,
`PE_API_KEY`, `PE_T2I_URL` / `PE_I2I_URL` (the pe sandbox's HTTPS endpoint
URLs), `WANDB_API_KEY` + `WEAVE_PROJECT` (from the local `.env`, via sandbox
`environment_variables` — never argv), `WANDB_DIR`/`WANDB_CACHE_DIR`/
`WANDB_CONFIG_DIR` (under `/work/wandb`), and `WEAVE_USE_SERVER_CACHE=false`.

## Scripts

| Script | Runs on | What it does |
|---|---|---|
| `scripts/launch.sh [--seed]` | Mac | Default: `sandbox.py up` (create-or-reuse both sandboxes, start servers, print+open the tokenized URL). `--seed`: `sandbox.py seed` (one-time snapshot build) |
| `scripts/sync.sh` | Mac | `sandbox.py sync` — tarball-push the repo to `/work/app` on the running studio sandbox (excludes `.venv .git .env outputs .launch` + caches), conditional `uv sync`, restart marimo |
| `scripts/stop.sh [--delete-snapshot]` | Mac | `sandbox.py down` — delete all `image-studio` sandboxes (billing stops; snapshot kept unless `--delete-snapshot`) |
| `uv run --no-project --python 3.12 scripts/sandbox.py <probe\|reseed\|status>` | Mac | Entitlement probe / incremental snapshot refresh / sandbox listing (no wrapper; note the explicit `--no-project --python 3.12`) |

## Failure and fallback behavior

| Failure | Behavior |
|---|---|
| No `WANDB_API_KEY` / weave init error | App runs untraced; `weave.op`s execute as plain functions; dataset logging is a no-op |
| PE server down / unreachable / bad JSON | `enhance_prompt` fails open: the raw user prompt goes straight to the pipeline (`enhanced: false` in the recipe) |
| PE servers never become healthy at launch | `up` warns and continues — the app still works with raw prompts; check `/work/logs/pe_*.log` on the pe sandbox |
| Only one of the two sandboxes still alive | `up` deletes the survivor and recreates both (a half pair can't serve) |
| Launch fails partway | `up` reaps every `image-studio` sandbox before re-raising, so nothing keeps billing |
| marimo doesn't come up | `up` exits with a pointer to `/work/logs/marimo.log` |
| Dataset append fails | Logged (`logger.exception`) and swallowed; generation already succeeded. If the previous dataset version can't be loaded, a fresh single-row version is published loudly (older versions remain recoverable) |
| Guidance without negative prompt (or vice versa) | Pipeline ignores it — the app only forwards CFG when both are set |
| Org not entitled to GPU sandboxes | Creation fails in <2 s with "not entitled" (nothing billed) — run `sandbox.py probe` first; access via your CoreWeave account team |

## Known gotchas

- **1 GPU per serverless sandbox, heterogeneous pool.** The 1-GPU cap is why
  the app is split into two sandboxes. The pool hands out H100 80 GB
  (`sm_90`) **or** RTX PRO 6000 Blackwell 96 GB (`sm_120`) — not selectable —
  so the image and wheels must support both archs (CUDA 12.8 + PyTorch ≥2.7
  cu128 builds do). Budget VRAM for the 80 GB case.
- **Snapshots capture only the mount.** Nothing outside `/work` survives into
  a fork — not the container image's changes, processes, or GPU memory.
  Everything the app needs at runtime (venvs, uv itself, `HF_HOME`, logs)
  lives under `/work`; there is no "snapshot a loaded model".
- **Fork-restore boot is ~4 min at this snapshot's size (~78 GB).** A sandbox
  created from a snapshot fork sits in CREATING while the mount restores — the
  launcher prints a note so this isn't mistaken for a hang, and treats >15 min
  in CREATING as wedged (observed live): it deletes the stuck sandbox and
  recreates.
- **The `pytorch/*-runtime` image is missing two tools the stack silently
  depends on** — the launcher's `_ensure_build_tools` apt-installs both in
  every sandbox (idempotent, ~45 s on first run), and because apt writes to
  the container layer rather than `/work`, a snapshot can never carry them —
  every fresh sandbox needs them again. **Going forward**: the repo-root
  `Dockerfile` bakes the tools into a custom image; build + push it and set
  `SANDBOX_IMAGE=<registry>/...@sha256:<digest>` in `.env` (the launcher
  picks it up — no code edit) and the runtime guards become no-ops. The two
  gaps:
  - **A C compiler (`gcc` + `libc6-dev`).** Triton JIT-compiles a small host
    C extension at the **first kernel launch** — torch imports fine, the
    model loads into VRAM fine, and only the first real generation fails
    (`RuntimeError: Failed to find C compiler`). Affects both sandboxes
    (torch/diffusers Triton ops on studio, vLLM on pe).
  - **`git`.** `pyproject.toml` pins diffusers to a git commit (the Qwen
    pipeline class is unreleased), and `uv sync` shells out to git whenever
    the built wheel isn't already in the `/work` uv cache.
  - **The CUDA toolkit (`nvcc`) is missing too** — vLLM 0.32's engine init
    runs FlashInfer JIT warmups that need nvcc unless the precompiled wheels
    (`flashinfer-cubin` + `flashinfer-jit-cache`) are installed. Deferred the
    same way: weights load and CUDA graphs capture fine, then ~3 min in the
    engine dies with `Could not find nvcc and default cuda_home=
    '/usr/local/cuda' doesn't exist`. Unlike the apt tools this fix *does*
    snapshot: `_ensure_vllm_kernels` installs the wheels into the vLLM venv's
    site-packages under `/work` (seed bakes them in; `_start_vllm` self-heals
    older snapshots).
- **Never run a server via attached `exec`** — it dies at the SDK's ~300 s
  request timeout. Servers start detached (`nohup … &`) with `sleep infinity`
  as the sandbox's main command; long steps (seeding, `uv sync`) run detached
  with sentinel-file polling (`run_long` in `sandbox.py`).
- **Public endpoints do zero platform auth.** The platform terminates TLS and
  forwards every request — so every exposed service authenticates itself:
  marimo token auth (the `?access_token=` URL) and vLLM `--api-key`. Never
  disable these.
- **Services/ports are fixed at creation.** No post-hoc port exposure —
  `sandbox.py` declares 2718 / 8001+8002 when creating each sandbox. A new
  port means recreating the sandbox.
- **Default sandbox lifetime is 600 s.** An unset `max_lifetime_seconds`
  kills a sandbox after 10 minutes; `sandbox.py` sets 24 h deliberately — it
  doubles as the billing backstop if you forget `scripts/stop.sh` (which you
  should still run: sandboxes bill per ms while up).
- **One process owns the pipeline GPU.** ~38 GB resident, ~66 GB peak at 2K —
  the pipeline is a module-level singleton in `model.py`; a second loader
  OOMs both.
- **vLLM `--gpu-memory-utilization` semantics.** It is this server's budget as
  a fraction of **TOTAL** GPU memory, and vLLM requires that much to be
  **FREE at startup**. The two PE servers share one GPU, so T2I takes 0.45,
  I2I takes 0.53, and they must start **sequentially** — concurrent startup
  races the free-memory profiling and the KV-cache sizing.
- **marimo `--watch` reloads `app.py` only.** Changes under `src/` need a
  marimo restart: `scripts/sync.sh` pushes the code and restarts it.
- **uv-managed Python only** (`UV_PYTHON_PREFERENCE=only-managed`, installed
  under `/work/uv-python`). Container pythons vary per image and a venv keyed
  to one would break on snapshot restore; managed interpreters also carry the
  dev headers Triton's JIT needs.
- **`.launch/sandbox-state.json` holds secrets** (app access token, PE bearer
  token) — chmod 600, gitignored; never commit it. Deleting it orphans the
  reuse path, but `scripts/stop.sh` can still reap by tag.
