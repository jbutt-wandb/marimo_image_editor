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
   echo 'CWSANDBOX_API_KEY=<your CoreWeave console token>' > .env
   echo 'WANDB_API_KEY=<your Forge/W&B key>' >> .env
   echo 'WEAVE_PROJECT=<your-entity/your-project>' >> .env
   ```

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
   uv run --python 3.12 scripts/sandbox.py probe
   ```

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
   uv run --python 3.12 scripts/sandbox.py status
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

## Once Setup - Run the Following

```bash
# One command, from the repo root on your Mac:
scripts/launch.sh
# 1. reuse the running sandboxes if both are still up; otherwise create the
#    studio + pe sandboxes in parallel, each on a fork of the seeded snapshot
# 2. push this repo's code to the studio sandbox (local repo is truth)
# 3. start the two prompt-enhancement vLLM servers on the pe sandbox
# 4. (re)start marimo on the studio sandbox with all env wired up
# 5. print + open the tokenized URL:
#    https://2718-<sandbox-id>.<hash>.cwsandbox.com/?access_token=<token>
```

Variants and teardown:

```bash
scripts/sync.sh                      # push code to the running studio sandbox
                                     # and restart marimo (changes under src/
                                     # need this; app.py alone hot-reloads
                                     # via --watch)
scripts/stop.sh                      # delete both sandboxes — billing stops;
                                     # the snapshot is KEPT for fast relaunch
scripts/stop.sh --delete-snapshot    # ... and delete the snapshot too
                                     # (the next launch needs --seed again)
uv run --python 3.12 scripts/sandbox.py status   # list sandboxes, URLs, snapshot
```

**Billing note.** Serverless sandboxes bill **per millisecond while running**
(CPU-core-hr and GiB-hr rates; GPU rates via your account team) — run
`scripts/stop.sh` when you're done. The 24 h `max_lifetime` on each sandbox is
a server-enforced auto-kill backstop, not a substitute for stopping. The
snapshot itself is billed per GiB-hr of storage; keep it (fast relaunches) or
drop it with `--delete-snapshot`.

Launcher state — sandbox ids, URLs, and the access tokens — lives in
`.launch/sandbox-state.json` (chmod 600, gitignored). Never commit it.

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
