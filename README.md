# Marimo Image Studio

A [marimo](https://marimo.io) web app for working with instruction-driven image
models — text-to-image, instruction editing, multi-reference composition, and
draw-on-image region editing — running on **CoreWeave serverless GPU sandboxes**
(two sandboxes, one GPU each) and used from your Mac through a tokenized public
HTTPS URL that the launcher prints and opens for you. The hosted model is
**Qwen-Image-2.1**, fronted by the official Qwen prompt-enhancement models and
orchestrated as agents (OpenAI Agents SDK), with every generation traced to Forge Weave. You can view the traces
[here](https://wandb.ai/wandb-smle/jb_image_editing_tool/weave).

For a deeper understanding of how the app works and the underlying model and architecture, please read below.

| Doc | What |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Components, request lifecycle, compute table, scripts, failure modes, gotchas |
| [`docs/model.md`](docs/model.md) | Qwen-Image-2.1 capabilities, resolutions, prompting tips, PE models |

## How the App Uses the CoreWeave Stack

The design leans on three layers, each doing the one job it's best at:

- **[CoreWeave Sandboxes](https://docs.coreweave.com/products/sandboxes)
  (serverless GPUs) — the compute.** Two single-GPU sandboxes are created on
  demand: one runs this marimo app with the Qwen-Image-2.1 diffusers pipeline
  in-process; the other hosts the two vLLM prompt-enhancement servers on a
  shared GPU. Model weights and venvs live in a **filesystem snapshot** that
  every launch forks instantly, so spinning the whole app up takes minutes and
  tearing it down (`scripts/stop.sh`) stops all GPU billing — there is no
  always-on infrastructure. The sandboxes' public HTTPS endpoints replace any
  VPN/SSH plumbing: the app authenticates with marimo's token auth, and the
  enhancer servers with a vLLM API key.

- **[Forge Weave](https://weave-docs.wandb.ai) (W&B) — the memory.** Every
  generation becomes a trace: the prompt-enhancement agent run (OpenAI Agents
  SDK) and the diffusion call nest under one task-named root, agents appear in
  the Agents view, and each output lands — full parameter recipe plus
  before/after images — in the versioned `image-generation-log` dataset,
  usable later as distillation/fine-tuning data. Weave is the system of
  record: generated images are **not** stored on sandbox disk at all.

- **[marimo](https://marimo.io) — the interface.** A reactive Python notebook
  served as a web app: the UI, the pipeline, and the agent orchestration are
  one Python process, so there is no separate frontend/backend to deploy. Its
  `--watch` mode hot-reloads `app.py` on every `scripts/sync.sh`, keeping the
  edit-to-live loop seconds long.

The through-line: ephemeral compute (sandboxes) + durable observability
(Weave) + a single-process app (marimo) means the entire deployment is one
command up, one command down, with nothing lost in between.

## How the App Works

A quick tour of the running app at the tokenized URL `scripts/launch.sh`
prints, captured live.

### 1. Describe what you want

![Create tab with a prompt filled in, steps and seed controls above the tabs](docs/images/create-prompt.jpg)

Type a prompt on the **✨ Create** tab; Steps and Seed sit above the tabs and
apply to every mode. You never submit this text verbatim — a PromptEnhancer
agent (a vLLM-served Qwen PE model) rewrites it before generation, and with
Aspect ratio on **Auto** the enhancer also picks the ratio that suits the scene.

### 2. Generate

![Results area cleared and showing a Generating spinner right after clicking Generate](docs/images/generating.jpg)

All four tabs share one results area below the tabs: pressing any run button
clears the previous image instantly and shows the spinner, so what you see is
always the latest run.

### 3. The result

![Generated night cabin image with Download button and seed caption](docs/images/create-result.jpg)

The finished image arrives with a **Download** button (served straight from
memory — nothing is written to sandbox disk) and a caption recording the seed
and that the prompt was enhanced. Auto aspect chose 3:2 (1248×832) for this
prompt. The whole run also lands in
[Weave](https://wandb.ai/wandb-smle/jb_image_editing_tool/weave) as a trace
plus a row in the `image-generation-log` dataset.

### 4. Edit: upload and instruct

![Edit tab with an uploaded image preview, filename, and an instruction filled in](docs/images/edit-preview.jpg)

The **🖊 Edit** tab takes 1–10 images (the first is the edit target, the rest
are references) and shows a thumbnail preview per upload. The edit itself is a
plain-language instruction — here, re-lighting the generated cabin.

### 5. The edited result

![Edited cabin image now at dawn with pink light and no snowfall](docs/images/edit-result.jpg)

Same cabin, now at dawn with the snowfall gone. Edits inherit the input
image's size and get the same seed caption, download button, and Weave trace.

### Advanced controls

![Advanced accordion expanded showing guidance, negative prompt, variations, KV cache, and prompt-enhance toggles](docs/images/advanced-options.jpg)

The **Advanced** accordion exposes true-CFG guidance with a negative prompt,
multiple variations per run, a KV-cache toggle for faster edits, and the
switch for prompt enhancement itself (off = your raw prompt goes straight to
the pipeline).

## Setup (first time)

One-time, from a fresh Mac; day-to-day use is just
[Once Setup - Run the Following](#once-setup---run-the-following).
This local repo is the **source of truth**; the sandboxes execute a copy the
scripts push to them.

1. **Clone the repo and install [uv](https://docs.astral.sh/uv/) locally:**

   ```bash
   git clone <repo-url> qwen2.1_image && cd qwen2.1_image
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   uv is the only local prerequisite — the scripts use it to run the launcher
   in its own isolated environment.

2. **CoreWeave API key + your `.env`.** Create a console token with the
   **"Sandbox User"** IAM role at
   [console.coreweave.com/tokens](https://console.coreweave.com/tokens), then
   create `.env` in the repo root (gitignored) — these values are **yours**,
   not shared with other developers:

   ```bash
   cp .env_example .env   # then fill in your values
   ```

   (`.env_example` documents every supported key, including the optional
   `SNAPSHOT_ID` for joining a team setup and `SANDBOX_IMAGE` for a custom
   image.)

   `CWSANDBOX_API_KEY` is **required** — it is how the launcher creates and
   drives the sandboxes. `WANDB_API_KEY` and `WEAVE_PROJECT` are optional:
   the launcher passes them into the app's sandbox environment, so changing
   them in `.env` and relaunching switches tracing projects; without them the
   app runs identically but untraced.

3. **GPU entitlement probe (optional but recommended).** GPU sandboxes are a
   private preview — your CoreWeave organization must be allowlisted. This
   ~1-minute check creates a 1-GPU sandbox, prints the GPU model it landed
   on, and tears it down (costs cents):

   ```bash
   uv run --no-project --python 3.12 scripts/sandbox.py probe
   ```

   (`--no-project` matters whenever you call `sandbox.py` directly: without
   it, uv rebuilds the repo's Python 3.10 app venv against 3.12 before
   running. The wrapper scripts handle this for you. If it happens, `uv sync`
   restores the venv.)

   If it fails with "not entitled to create GPU sandboxes" (within ~2
   seconds, nothing billed), ask your CoreWeave account team for GPU-sandbox
   access before continuing.

4. **One-time seeding:**

   ```bash
   scripts/launch.sh --seed
   ```

   This builds the app's persistent disk once, on a cheap CPU-only sandbox:
   both Python venvs plus ~80 GB of model weights go into a 200 Gi
   filesystem snapshot. Expect **~30–60 min**, with a one-line progress
   heartbeat per minute. Every later launch mounts an instant fork of this
   snapshot, which is why day-to-day launches are fast.

   **Joining an existing team setup?** (New developers only — if you seeded
   on this machine, your snapshot id is already in
   `.launch/sandbox-state.json` and you need none of this; `SNAPSHOT_ID` in
   `.env` is only consulted when that local state has no snapshot.) Skip
   seeding by adopting a teammate's snapshot: get the snapshot id from them
   (`sandbox.py status` prints it), then add `SNAPSHOT_ID=<id>` to your
   `.env` — or launch once with `uv run --no-project --python 3.12
   scripts/sandbox.py up --snapshot-id <id>`. The id is saved into your local
   launcher state and every later `scripts/launch.sh` uses it. Org snapshots
   should be adoptable across users in the same org; if the restore fails,
   fall back to your own `scripts/launch.sh --seed`.

5. **First launch:**

   ```bash
   scripts/launch.sh
   ```

   Expect, in order: both sandboxes created in parallel (~1 min), code
   pushed to the studio sandbox, the two prompt-enhancement vLLM servers
   loading sequentially (**~2–4 min**; `launch.sh` continues even if they
   fail — the app then uses raw prompts), marimo start, then:

   ```
   ✅ ready: https://2718-<sandbox-id>.<hash>.cwsandbox.com/?access_token=<token>
   ```

   Ready end-to-end in **~3–5 min**. On macOS the URL opens automatically;
   the `access_token` in it is required — the endpoint is on the public
   internet and marimo's token auth is the access control. The first
   generation loads the image pipeline lazily — allow **~2–4 min** (~38 GB
   bf16 onto the GPU); later generations skip this.

6. **Sanity checks:**

   ```bash
   uv run --no-project --python 3.12 scripts/sandbox.py status
   # studio  <sandbox-id>  RUNNING  https://2718-…cwsandbox.com
   # pe      <sandbox-id>  RUNNING  https://8001-…  https://8002-…
   # launched: <timestamp>
   # snapshot: <snapshot-id>
   ```

   Service logs live inside each sandbox under `/work/logs/`
   (`marimo.log` on studio; `pe_8001.log`, `pe_8002.log` on pe). To tail
   them, exec into the sandbox with the cwsandbox CLI
   (`pip install 'cwsandbox[cli]'`, then its `exec` command with the sandbox
   id from `status`). After one generation, confirm a trace landed in your
   `WEAVE_PROJECT` at `https://wandb.ai/<your-entity/your-project>/weave`.

## After Initial Setup - Run Scripts (step by step)

Day-to-day operation after [Setup](#setup-first-time), using only the
scripts, from the repo root on your Mac.

1. **Launch:**

   ```bash
   scripts/launch.sh
   ```

   It reuses the running sandboxes if both are still up; otherwise it creates
   the studio + pe pair in parallel, each on a fork of the seeded snapshot.
   Expect, in order: a printed note that restoring the ~78 GB snapshot takes
   **~4 min** per sandbox (they are not hung), the code push to the studio
   sandbox (this local repo is the source of truth), the two
   prompt-enhancement vLLM servers loading (**~2–4 min**; the launch continues
   even if they fail — the app then falls back to raw prompts), marimo
   starting, and finally the line that matters:

   ```
   ✅ ready: https://2718-<sandbox-id>.<hash>.cwsandbox.com/?access_token=<token>
   ```

   On macOS that URL opens automatically. The `access_token` is required —
   the endpoint is on the public internet and marimo's token auth is the only
   access control.

2. **Lost the URL?** Rebuild it from the state file at any time:

   ```bash
   python3 -c "import json; s = json.load(open('.launch/sandbox-state.json')); \
   print(f\"{s['studio_url']}/?access_token={s['app_token']}\")"
   ```

3. **Check health** whenever something looks off:

   ```bash
   uv run --no-project --python 3.12 scripts/sandbox.py status
   ```

   Both `studio` and `pe` should show `RUNNING` with their URLs. Service logs
   live inside each sandbox under `/work/logs/` (`marimo.log` on studio,
   `pe_8001.log` / `pe_8002.log` on pe).

4. **Push code changes** to the running app:

   ```bash
   scripts/sync.sh
   ```

   Edits to `app.py` alone hot-reload via marimo's `--watch`; anything under
   `src/` needs this sync — it re-pushes the repo, runs `uv sync` (a no-op
   unless dependencies changed), and restarts marimo. Same URL, same token.

5. **Take it down** when you're done (sandboxes bill per millisecond while
   running):

   ```bash
   scripts/stop.sh                     # delete both sandboxes — billing stops
   scripts/stop.sh --delete-snapshot   # full teardown: also drop the snapshot
   ```

6. **What persists after a plain stop:** the snapshot (billed per GiB-hr of
   storage — it is what makes the next `scripts/launch.sh` a ~3–5 min
   relaunch instead of a re-seed) and `.launch/sandbox-state.json` (the
   snapshot id and access tokens, so bookmarked URLs keep working across
   relaunches). `--delete-snapshot` removes the snapshot too; the next launch
   then needs `--seed` again.

### Updating the snapshot

The snapshot is a build artifact — refresh it whenever sandboxes should start
with something new baked in (a dependency bump in `uv.lock`, a new model in
the launcher constants, a runtime fix worth persisting):

```bash
uv run --no-project --python 3.12 scripts/sandbox.py reseed
```

`reseed` forks the current snapshot onto a cheap CPU-only sandbox, applies
only the delta (code push, `uv sync`, vLLM kernel wheels, any missing model
downloads), snapshots the result, and swaps your state over to the new id —
**minutes**, versus the ~hour of a full seed. The old snapshot is deleted once
the new one is ready (`--keep-old` keeps it); running sandboxes are untouched
and pick up the new snapshot at their next creation. Because reseed only ever
*adds* to the mount, use a full rebuild when you want to prune (removed
models, layout changes): `scripts/stop.sh --delete-snapshot`, then
`scripts/launch.sh --seed`. A failed reseed is harmless — it retries the
snapshot five times, and if the platform's snapshot service is having a bad
day your old snapshot and state stay exactly as they were; just re-run later.

**Custom image (the path going forward).** The stock PyTorch image the
launcher defaults to is missing tools the stack needs (git, gcc), which the
scripts currently re-install into every fresh sandbox at runtime (~45 s). The
repo-root `Dockerfile` bakes them in permanently. To adopt it — no code
changes required:

```bash
docker build --platform linux/amd64 -t <registry>/image-studio-base:v1 .
docker push <registry>/image-studio-base:v1
echo 'SANDBOX_IMAGE=<registry>/image-studio-base@sha256:<digest>' >> .env
```

The launcher reads `SANDBOX_IMAGE` from `.env` and uses it for every sandbox
it creates; the runtime install guards then become instant no-ops. Pin by
digest (not tag) so launches stay reproducible across the heterogeneous GPU
pool. Until you build and push, everything works on the stock image — the
guards cover the gap.

**Billing note.** Serverless sandboxes bill **per millisecond while running**
(CPU-core-hr and GiB-hr rates; GPU rates via your account team) — run
`scripts/stop.sh` when you're done. The 24 h `max_lifetime` on each sandbox is
a server-enforced auto-kill backstop, not a substitute for stopping. The
snapshot itself is billed per GiB-hr of storage; keep it (fast relaunches) or
drop it with `--delete-snapshot`.

Launcher state — sandbox ids, URLs, and the access tokens — lives in
`.launch/sandbox-state.json` (chmod 600, gitignored). Never commit it.

**Configuration.** The operational knobs are flags and `.env`: `--seed`,
snapshot adoption via `--snapshot-id`/`SNAPSHOT_ID`, `stop.sh
--delete-snapshot`, a custom container image via `SANDBOX_IMAGE`, and your
keys/Weave project in `.env`. The topology — which sandboxes exist, their
ports, models, and resources — is deliberately **not** flag-driven: it lives
in the constants block at the top of `scripts/sandbox.py`, and that block is
the configuration surface. Edit it there, then `reseed` (new models/deps) or
relaunch as appropriate.

## Tracing

When `WANDB_API_KEY` is present, every user action becomes one Weave trace in
**[wandb-smle/jb_image_editing_tool](https://wandb.ai/wandb-smle/jb_image_editing_tool/weave)**
(W&B / CoreWeave Forge): the prompt-enhancement agent run and the pipeline call
nest under a per-click root, agents appear in the Agents tab, and each output
image is appended — with its full parameter recipe and base64 input/output
images — to the versioned `image-generation-log` dataset, which doubles as
distillation/fine-tuning data. Weave is the system of record for generated
images; nothing is durably stored on sandbox disk. Without the key, the app
runs identically but untraced.
