#!/usr/bin/env -S uv run --python 3.12 --script
# NB: the explicit --python matters: the repo's .python-version (3.10, the app
# venv) otherwise wins over this script's requires-python and resolution fails.
# /// script
# requires-python = ">=3.11"
# dependencies = ["cwsandbox>=1.19"]
# ///
"""Image Studio launcher for CoreWeave serverless GPU sandboxes.

Replaces the old Slurm scripts (launch.sh / sync.sh / stop.sh / pe_server.sh /
config.sh, now thin wrappers or deleted). The app is split across TWO
serverless sandboxes because serverless caps at 1 GPU each:

  studio  marimo + the Qwen-Image-2.1 diffusers pipeline       port 2718
  pe      BOTH vLLM prompt-enhancer servers on one GPU          ports 8001/8002

Both mount an instant fork of a filesystem snapshot at /work that was built
once by `seed` (venvs + ~80 GB of HF weights), so a launch is roughly sandbox
creation (~60 s) + model load into VRAM (~2-4 min).

This is a standalone uv inline-metadata script (PEP 723): cwsandbox needs
Python >= 3.11 while the app venv is 3.10, so it must NOT be added to the
app's pyproject. `uv run --python 3.12 scripts/sandbox.py <cmd>` resolves its
own isolated env (the wrappers do this for you).

Subcommands:
  probe    ~1-minute GPU entitlement check (GPU sandboxes are private preview)
  seed     one-time: build the /work snapshot on a cheap CPU-only sandbox
  up       create (or reuse) both sandboxes, start servers, print/open the URL
  sync     push local code to the studio sandbox and restart marimo
  down     stop everything (reap by tag); --delete-snapshot also drops the FSS
  status   list this app's sandboxes

Auth + config come from the repo's .env (CWSANDBOX_API_KEY, WANDB_API_KEY,
WEAVE_PROJECT). Secret VALUES are never printed and never passed on argv —
they travel via sandbox environment_variables only.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import secrets as pysecrets
import shlex
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# cwsandbox installs SIGINT/SIGTERM handlers at import time — import here on
# the main thread (worker threads below only *call* the SDK, never import it).
from cwsandbox import (
    AuthStrategy,
    Endpoint,
    EndpointAuth,
    EndpointKind,
    FileSystemSnapshotOptions,
    Sandbox,
    SandboxResourceExhaustedError,
    SandboxStatus,
    SandboxTimeoutError,
    SandboxUnavailableError,
    Service,
    ServiceVisibility,
)

# --- constants ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / ".launch" / "sandbox-state.json"  # holds tokens/ids — chmod 600

AUTH = AuthStrategy.COREWEAVE_API_KEY  # <- CWSANDBOX_API_KEY from .env

# One image for everything (probe, seed, studio, pe). The serverless GPU pool
# is heterogeneous (H100 80GB sm_90 AND RTX PRO 6000 96GB sm_120) — this tag
# ships CUDA 12.8 + PyTorch 2.8 and is verified to work on both archs.
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"

APP_TAG = "image-studio"  # stable app tag; per-role tags let us reap precisely

MOUNT = "/work"
MOUNT_SIZE = "200Gi"  # weights ~80GB + two venvs + managed pythons + headroom

# A GPU request reserves ONLY the GPU — cpu/memory must be set explicitly
# (serverless requires requests == limits). 16 cpu / 64Gi mirrors the old
# Slurm job's --cpus-per-task=16.
GPU_RESOURCES = {"cpu": "16", "memory": "64Gi", "gpu": 1}
SEED_RESOURCES = {"cpu": "8", "memory": "32Gi"}  # downloads/installs: no GPU needed

MAX_LIFETIME = 24 * 3600  # mirrors the old job's --time=24:00:00; also the
#                           server-enforced billing backstop if we forget `down`
SEED_LIFETIME = 4 * 3600  # seeding is ~30-60 min; 4 h leaves slack for slow HF

APP_PORT = 2718
PE_T2I_PORT, PE_I2I_PORT = 8001, 8002

PIPELINE_MODEL = "Qwen/Qwen-Image-2.1"  # src/qwen2_1/model.py MODEL_ID
PE_T2I_MODEL = "Qwen/Qwen-Image-2.1-PE-T2I"
PE_I2I_MODEL = "Qwen/Qwen-Image-2.1-PE-I2I"

# uv must use ITS OWN managed interpreters stored under the snapshot mount —
# container pythons vary and a venv keyed to one would break on restore.
UV_ENV = (
    f"UV_CACHE_DIR={MOUNT}/cache/uv "
    f"UV_PYTHON_INSTALL_DIR={MOUNT}/uv-python "
    "UV_PYTHON_PREFERENCE=only-managed"
)
UV = f"{UV_ENV} {MOUNT}/bin/uv"

# Same exclude list as the old rsync-based sync.sh (plus .DS_Store friends).
TAR_EXCLUDES = {
    ".venv", ".git", ".env", "outputs", ".launch", "__pycache__",
    ".marimo", "__marimo__", ".ipynb_checkpoints", ".DS_Store",
}


# --- small helpers -----------------------------------------------------------

def load_dotenv() -> dict[str, str]:
    """Minimal KEY=VALUE parser for the repo's .env (no interpolation)."""
    env: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip("'\"")
    return env


def require_api_key(env: dict[str, str]) -> None:
    """Expose CWSANDBOX_API_KEY to the SDK via os.environ (never print it)."""
    if os.environ.get("CWSANDBOX_API_KEY"):
        return
    if env.get("CWSANDBOX_API_KEY"):
        os.environ["CWSANDBOX_API_KEY"] = env["CWSANDBOX_API_KEY"]
        return
    sys.exit("CWSANDBOX_API_KEY not found in .env or the environment — create a "
             "console token (Sandbox User role) at console.coreweave.com/tokens")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
    STATE_FILE.chmod(0o600)  # holds the app access token + PE bearer token


def sh(sb: Sandbox, script: str, timeout: float | None = None):
    """Attached exec of a SHORT shell command. Anything that can outlive the
    SDK's ~300 s request timeout must go through run_long() instead."""
    return sb.exec(["bash", "-c", script], timeout_seconds=timeout).result()


def run_long(sb: Sandbox, name: str, script: str, timeout: float, poll: float = 15) -> None:
    """Run a long step DETACHED and poll for completion.

    Attached exec dies at the SDK's ~300 s request timeout, so every slow step
    (uv sync, vllm install, HF downloads) runs under nohup with its exit code
    dropped into a sentinel file we poll for. Logs land in /work/logs/.
    """
    log, rc = f"{MOUNT}/logs/{name}.log", f"{MOUNT}/logs/{name}.rc"
    wrapped = f"({script}) > {log} 2>&1; echo $? > {rc}"
    sh(sb, f"mkdir -p {MOUNT}/logs && rm -f {rc} && "
           f"(nohup bash -c {shlex.quote(wrapped)} >/dev/null 2>&1 &)")
    t0 = time.monotonic()
    next_beat = 60.0
    while time.monotonic() - t0 < timeout:
        out = (sh(sb, f"cat {rc} 2>/dev/null").stdout or "").strip()
        if out == "0":
            print(f"  {name}: done ({time.monotonic() - t0:.0f}s)")
            return
        if out:
            tail = sh(sb, f"tail -n 25 {log}").stdout
            raise RuntimeError(f"{name} failed (rc={out}) — tail of {log}:\n{tail}")
        if time.monotonic() - t0 >= next_beat:  # 1-line heartbeat per minute
            last = (sh(sb, f"tail -c 2000 {log} 2>/dev/null | tail -n 1").stdout or "").strip()
            print(f"  {name}: running {time.monotonic() - t0:.0f}s  {last[:110]}")
            next_beat += 60.0
        time.sleep(poll)
    raise TimeoutError(f"{name} did not finish within {timeout}s — see {log} in the sandbox")


# Probe from INSIDE the sandbox with python (the pytorch image may lack curl).
# Any HTTP status counts as "listening" — marimo answers 401 under token auth.
_HTTP_PROBE = (
    "import sys, urllib.request, urllib.error\n"
    "try:\n"
    "    urllib.request.urlopen('http://localhost:{port}{path}', timeout=3)\n"
    "except urllib.error.HTTPError:\n"
    "    pass\n"
    "except Exception:\n"
    "    sys.exit(1)\n"
)


def wait_http(sb: Sandbox, port: int, label: str, path: str = "/",
              timeout: float = 600, interval: float = 10) -> bool:
    """Phase-2 readiness: sandbox RUNNING != app listening — poll the port."""
    print(f"Waiting for {label}", end="", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        r = sb.exec(["python", "-c", _HTTP_PROBE.format(port=port, path=path)]).result()
        if r.returncode == 0:
            print(f" up ({time.monotonic() - t0:.0f}s)")
            return True
        print(".", end="", flush=True)
        time.sleep(interval)
    print(" TIMEOUT")
    return False


def create_with_retry(label: str, wait_timeout: float = 900, **kwargs) -> Sandbox:
    """Sandbox.run (keep-alive form) + wait-until-RUNNING, with retries for the
    two transient platform failures seen live. Used by seed AND both `up`
    creators.

    - CreateSandbox can fail RESOURCE_EXHAUSTED ("runner capacity exhausted" /
      gateway errors) when the serverless pool is momentarily full. Nothing is
      created or billed on that failure — just back off and recreate.
    - A created sandbox can also wedge in CREATING past the wait timeout
      (observed live on a snapshot-fork GPU sandbox; normal fork boots are
      ~4 min). Delete the stuck one and recreate — a fresh create usually
      lands on a healthy runner.

    Backoff grows 1..5 min (~15 min total patience on top of the waits).
    """
    for attempt in range(6):
        sb = None
        try:
            sb = Sandbox.run(
                "sleep", "infinity",  # keep-alive main process; work runs via exec
                auth=AUTH,
                container_image=IMAGE,
                placement_mode="serverless",
                **kwargs,
            )
            sb.wait(timeout=wait_timeout)
            return sb
        except SandboxResourceExhaustedError as e:
            if attempt == 5:
                raise
            reason = f"capacity exhausted ({e})"
        except SandboxTimeoutError:
            if sb is not None:  # don't leave the wedged sandbox billing
                Sandbox.delete(sb.sandbox_id, auth=AUTH, missing_ok=True).result()
            if attempt == 5:
                raise
            reason = f"stuck in CREATING > {wait_timeout:.0f}s (deleted)"
        wait = 60 * (attempt + 1)
        print(f"  {label}: {reason}; retry {attempt + 1}/5 in {wait}s")
        time.sleep(wait)
    raise AssertionError("unreachable")


def make_tarball() -> bytes:
    """gzip tarball of the repo (the code the sandboxes run), minus junk/secrets."""
    buf = io.BytesIO()

    def keep(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(ti.name).parts
        return None if any(p in TAR_EXCLUDES for p in parts) else ti

    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for child in sorted(ROOT.iterdir()):
            if child.name in TAR_EXCLUDES:
                continue
            tar.add(child, arcname=child.name, filter=keep)
    return buf.getvalue()


def upload_code(sb: Sandbox) -> None:
    """Push local code into the sandbox: authenticated file API + untar.

    (No rsync here — write_file is the sandbox equivalent; the repo is a few
    MB so a full tarball each time is simpler than a delta.)
    """
    data = make_tarball()
    print(f"Uploading code ({len(data) / 1e6:.1f} MB) -> {MOUNT}/app")
    sb.write_file("/tmp/app.tar.gz", data).result()
    sh(sb, f"mkdir -p {MOUNT}/app && tar -xzf /tmp/app.tar.gz -C {MOUNT}/app && rm /tmp/app.tar.gz")


def service_url(sb: Sandbox, port: int) -> str:
    urls = {int(p): u for p, _name, u in sb.service_urls}
    return urls[port]


def reconnect(sandbox_id: str | None) -> Sandbox | None:
    """Reattach to a previously created sandbox if it is still RUNNING."""
    if not sandbox_id:
        return None
    try:
        sb = Sandbox.from_id(sandbox_id, auth=AUTH)
        return sb if sb.get_status() == SandboxStatus.RUNNING else None
    except Exception:
        return None


def reap_by_tag(delete_snapshot: bool = False) -> None:
    """Delete every sandbox carrying our app tag (also `up`'s failure path)."""
    state = load_state()
    found = Sandbox.list(tags=[APP_TAG], auth=AUTH).result()  # OperationRef -> list
    for sb in found:
        print(f"Deleting sandbox {sb.sandbox_id}")
        Sandbox.delete(sb.sandbox_id, auth=AUTH, missing_ok=True).result()
    if not found:
        print("No sandboxes to stop.")
    for key in ("studio_id", "pe_id", "studio_url", "pe_t2i_url", "pe_i2i_url", "launched_at"):
        state.pop(key, None)
    if delete_snapshot and state.get("snapshot_id"):
        print(f"Deleting snapshot {state['snapshot_id']} (next launch needs --seed again)")
        Sandbox.delete_snapshot(state.pop("snapshot_id"), auth=AUTH, missing_ok=True).result()
    save_state(state)


# --- subcommand: probe --------------------------------------------------------

def cmd_probe(_args) -> None:
    """GPU entitlement probe (GPU sandboxes are private preview, per-org
    allowlist). An unentitled org fails at create in <2 s with
    'Permission denied: this organization is not entitled to create GPU
    sandboxes' — nothing billed. Success also reveals which GPU model the
    heterogeneous pool hands us. Costs cents; run before any GPU work."""
    sb, t0 = None, time.monotonic()
    try:
        sb = Sandbox.run(
            "sleep", "600",
            auth=AUTH,
            container_image=IMAGE,
            placement_mode="serverless",
            resources={"cpu": "2", "memory": "8Gi", "gpu": 1},
            max_lifetime_seconds=600,
            tags=["gpu-access-probe"],
        )
        sb.wait(timeout=600)
        print(f"RUNNING after {time.monotonic() - t0:.1f}s")  # expect ~30-65s for GPU
        print(sb.exec(["nvidia-smi", "-L"]).result().stdout)  # which card did we get?
        print("GPU ACCESS CONFIRMED")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")  # entitlement error lands here
        sys.exit(1)
    finally:
        if sb is not None:
            sb.stop(missing_ok=True).result()


# --- subcommand: seed ---------------------------------------------------------

def cmd_seed(_args) -> None:
    """One-time snapshot build on a CPU-ONLY sandbox (downloads + installs need
    no GPU — don't pay for one). Everything the app needs at runtime must live
    UNDER /work: snapshots capture only the mount — never the image, processes,
    or GPU memory. Expect ~30-60 min (≈80 GB of HF weights)."""
    state = load_state()
    if state.get("snapshot_id"):
        sys.exit(f"snapshot {state['snapshot_id']} already exists — "
                 "run `scripts/stop.sh --delete-snapshot` first to rebuild")

    print("Creating CPU seed sandbox...")
    sb = create_with_retry(  # pool capacity can be transiently exhausted
        "seed",
        resources=SEED_RESOURCES,
        max_lifetime_seconds=SEED_LIFETIME,
        tags=[APP_TAG, f"{APP_TAG}-seed"],
        file_system_snapshot=FileSystemSnapshotOptions(mount_path=MOUNT, size=MOUNT_SIZE),
    )
    try:
        print(f"Seed sandbox {sb.sandbox_id} running.")

        # uv itself must survive into forked sandboxes -> copy the static
        # binary under /work (pip exists in the pytorch image; curl may not).
        sh(sb, "python -m pip install -q uv && mkdir -p /work/bin && "
               'cp "$(command -v uv)" /work/bin/uv', timeout=240)

        upload_code(sb)

        # The pytorch runtime image ships NO git, but pyproject pins diffusers
        # to a git commit — uv sync needs the git CLI to fetch/build it. Install
        # it container-side (seed only): the built wheel lands in the uv cache
        # under /work and survives into forks, so forked `uv sync` re-runs hit
        # the cache instead of git.
        run_long(sb, "apt-git",
                 "apt-get update && apt-get install -y --no-install-recommends git",
                 timeout=600)

        # App venv (python 3.10, from pyproject/uv.lock) at /work/app/.venv.
        run_long(sb, "app-venv", f"cd {MOUNT}/app && {UV} sync", timeout=2400)

        # Separate vLLM venv — vLLM's pins clash with the app's, same split as
        # the old cluster layout. torch cu128 wheels cover both GPU archs the
        # serverless pool serves (sm_90 H100 + sm_120 RTX PRO 6000).
        # ninja: vllm shells out to it for torch.compile.
        run_long(
            sb, "vllm-env",
            f"{UV} venv --python 3.12 {MOUNT}/vllm-env && "
            f"{UV} pip install --python {MOUNT}/vllm-env/bin/python vllm ninja",
            timeout=2400,
        )

        # Model weights into HF_HOME under the mount. Qwen-Image models are
        # public — no HF token needed. huggingface_hub comes with the app venv
        # (transitive via transformers/diffusers).
        for model in (PIPELINE_MODEL, PE_T2I_MODEL, PE_I2I_MODEL):
            tag = "hf-" + model.rsplit("/", 1)[1].lower()
            run_long(
                sb, tag,
                f"HF_HOME={MOUNT}/hf {MOUNT}/app/.venv/bin/python -c "
                f"\"from huggingface_hub import snapshot_download; "
                f"snapshot_download('{model}')\"",
                timeout=7200,
            )

        # Verify the payload BEFORE paying for the snapshot: forked sandboxes
        # assume these exact paths; the rc-sentinels above prove each step
        # exited 0 but not that it left the artifacts we expect.
        required = [
            f"{MOUNT}/app/.venv/bin/python",
            f"{MOUNT}/vllm-env/bin/vllm",
        ] + [
            # snapshot_download with HF_HOME=/work/hf lands weights here
            f"{MOUNT}/hf/hub/models--{m.replace('/', '--')}/snapshots"
            for m in (PIPELINE_MODEL, PE_T2I_MODEL, PE_I2I_MODEL)
        ]
        checks = " && ".join(f"test -e {shlex.quote(p)}" for p in required)
        if (sh(sb, f"{checks} && echo OK", timeout=120).stdout or "").strip() != "OK":
            missing = sh(sb, "; ".join(
                f"test -e {shlex.quote(p)} || echo MISSING {p}" for p in required),
                timeout=120).stdout
            raise RuntimeError(f"seed verification failed:\n{missing}")
        print("Seed verified: app venv, vllm venv, and 3 model dirs present.")
        print(sh(sb, f"du -sh {MOUNT}/hf {MOUNT}/app/.venv {MOUNT}/vllm-env 2>/dev/null; "
                     f"df -h {MOUNT} | tail -1", timeout=300).stdout)

        print("Snapshotting /work (waits for READY — can take a few minutes)...")
        # The snapshot RPC can fail with a transient gateway error even after
        # the SDK's own retries (seen live: "Service unavailable: the owning
        # gateway could not complete the CreateFileSystemSnapshot"). Don't let
        # that discard a fully built mount — retry before giving up.
        for attempt in range(5):
            try:
                snapshot_id = sb.snapshot().result()
                break
            except (SandboxUnavailableError, SandboxResourceExhaustedError) as e:
                if attempt == 4:
                    raise
                print(f"  snapshot attempt failed ({e}); retry {attempt + 1}/4 in 60s")
                time.sleep(60)
        state["snapshot_id"] = snapshot_id
        save_state(state)
        print(f"✅ snapshot ready: {snapshot_id}  (saved to {STATE_FILE})")
    finally:
        print("Deleting seed sandbox.")
        Sandbox.delete(sb.sandbox_id, auth=AUTH, missing_ok=True).result()


# --- subcommand: up -----------------------------------------------------------

def _fss_fork(snapshot_id: str) -> FileSystemSnapshotOptions:
    """Each sandbox gets an independent instant FORK of the seeded mount."""
    return FileSystemSnapshotOptions(
        mount_path=MOUNT, size=MOUNT_SIZE, file_system_snapshot_id=snapshot_id)


def _https_service(port: int, name: str) -> Service:
    # Public HTTPS endpoint: the platform terminates TLS but does ZERO caller
    # auth (OPEN). The SSH tunnel used to be the auth boundary — now each app
    # must authenticate itself (marimo token / vLLM --api-key).
    return Service(port=port, name=name, visibility=ServiceVisibility.PUBLIC,
                   endpoint=Endpoint(kind=EndpointKind.HTTPS, auth=EndpointAuth.OPEN))


def _create_pe(snapshot_id: str, pe_token: str) -> Sandbox:
    sb = create_with_retry(  # GPU pool capacity can be transiently exhausted
        "pe",
        resources=GPU_RESOURCES,
        max_lifetime_seconds=MAX_LIFETIME,
        tags=[APP_TAG, f"{APP_TAG}-pe"],
        # Secrets go through environment_variables, never argv.
        environment_variables={"PE_API_TOKEN": pe_token, "HF_HOME": f"{MOUNT}/hf"},
        # Services/ports are FIXED at creation — declare both vLLM ports now.
        services=[_https_service(PE_T2I_PORT, "pe-t2i"),
                  _https_service(PE_I2I_PORT, "pe-i2i")],
        file_system_snapshot=_fss_fork(snapshot_id),
    )
    return sb


def _create_studio(snapshot_id: str, dotenv: dict[str, str],
                   pe_token: str, app_token: str) -> Sandbox:
    env = {
        "HF_HOME": f"{MOUNT}/hf",
        # marimo reads these at startup (exec'd processes inherit container env)
        "APP_TOKEN": app_token,       # marimo --token-password
        "PE_API_KEY": pe_token,       # bearer the app presents to the PE vLLMs
        # Keep W&B artifacts/caches on the big mount, not the container disk.
        "WANDB_DIR": f"{MOUNT}/wandb",
        "WANDB_CACHE_DIR": f"{MOUNT}/wandb/cache",
        "WANDB_CONFIG_DIR": f"{MOUNT}/wandb/config",
        # Weave's local server cache misbehaves on shared/odd filesystems.
        "WEAVE_USE_SERVER_CACHE": "false",
    }
    # Weave/Forge tracing travels from local .env into the sandbox env.
    if dotenv.get("WANDB_API_KEY"):
        env["WANDB_API_KEY"] = dotenv["WANDB_API_KEY"]
    if dotenv.get("WEAVE_PROJECT"):
        env["WEAVE_PROJECT"] = dotenv["WEAVE_PROJECT"]
    else:
        print("warning: WEAVE_PROJECT not set in .env - app will run untraced",
              file=sys.stderr)

    sb = create_with_retry(  # GPU pool capacity can be transiently exhausted
        "studio",
        resources=GPU_RESOURCES,
        max_lifetime_seconds=MAX_LIFETIME,
        tags=[APP_TAG, f"{APP_TAG}-studio"],
        environment_variables=env,
        services=[_https_service(APP_PORT, "app")],
        file_system_snapshot=_fss_fork(snapshot_id),
    )
    return sb


def _ensure_build_tools(sb: Sandbox) -> None:
    """Install the two tools the pytorch *-runtime* image is missing — learned
    the hard way (each cost a debugging session), so the WHY in full:

    - gcc + libc6-dev: Triton JIT-compiles a small host C extension (its CUDA
      driver utils) at the FIRST kernel launch — so torch imports fine, the
      model loads into VRAM fine, and only the first real generation dies with
      'RuntimeError: Failed to find C compiler' (seen live on the studio
      sandbox). Needed by studio (torch/diffusers triton ops) AND pe (vLLM).
    - git: pyproject pins diffusers to a git commit (QwenImage21Pipeline is
      not in any PyPI release), and `uv sync` shells out to the git CLI
      whenever the built wheel isn't already in the /work uv cache (seen live
      in seeding and as launch bootstrap failures).

    apt installs land in the CONTAINER layer — NOT the /work snapshot (FSS
    captures only the mount) — so every fresh sandbox/fork needs this again.
    Idempotent: instant no-op when both are already present (~45 s otherwise).
    Call before any `uv sync` and before starting marimo/vLLM."""
    sh(sb, "command -v cc >/dev/null 2>&1 && command -v git >/dev/null 2>&1 || "
           "(apt-get update -qq && "
           "apt-get install -y -qq --no-install-recommends gcc libc6-dev git)",
       timeout=280)


def _start_vllm(sb: Sandbox, model: str, port: int, mem_util: float) -> None:
    """Detached vLLM server (was pe_server.sh). Flags mirror the official
    prompt_rewrite/serve.sh (reasoning parser, mm limit, bf16).

    mem_util is this server's own budget as a fraction of TOTAL GPU memory,
    and vLLM requires that much to be FREE at startup — both PE instances
    share the single GPU, so T2I takes 0.45 and I2I 0.53, started
    SEQUENTIALLY: vLLM sizes its KV cache against total memory minus what is
    already allocated, so concurrent startup makes the free-memory math race.
    """
    _ensure_build_tools(sb)
    log = f"{MOUNT}/logs/pe_{port}.log"
    # $PE_API_TOKEN expands INSIDE the sandbox from its creation env — the
    # secret never crosses our argv. PATH: vllm shells out to ninja
    # (torch.compile); the venv's bin must be on PATH.
    # The `( ... </dev/null &)` isolation matters: a child that inherits ANY of
    # the exec session's fds keeps the exec stream open and sh() then hangs to
    # the SDK's 300 s cap (bit us live on the marimo start). Same pattern as
    # run_long().
    sh(sb, f"mkdir -p {MOUNT}/logs && "
           f"(CUDA_VISIBLE_DEVICES=0 PATH={MOUNT}/vllm-env/bin:$PATH "
           f"nohup {MOUNT}/vllm-env/bin/vllm serve {model} "
           f"--host 0.0.0.0 --port {port} "
           f"--dtype bfloat16 "
           f"--max-model-len 24576 "
           f"--gpu-memory-utilization {mem_util} "
           f"--max-num-seqs 16 "
           f"--limit-mm-per-prompt '{{\"image\": 10}}' "
           f"--reasoning-parser qwen3 "
           f'--api-key "$PE_API_TOKEN" '
           f"> {log} 2>&1 </dev/null &)")


def _start_marimo(sb: Sandbox, pe_t2i_url: str, pe_i2i_url: str) -> None:
    """(Re)start marimo detached. Tokens/keys come from the sandbox's creation
    env; only the non-secret PE URLs are injected here."""
    _ensure_build_tools(sb)
    # NB: pkill -f patterns are regexes — keep them metachar-free.
    sh(sb, "pkill -f 'bin/marimo' || true")
    time.sleep(3)  # let the port free up
    # `( ... </dev/null &)` isolation as in _start_vllm/run_long: without it the
    # detached marimo inherits the exec session's fds and this sh() hangs until
    # the SDK's 300 s exec timeout (observed live).
    sh(sb, f"mkdir -p {MOUNT}/wandb/cache {MOUNT}/wandb/config {MOUNT}/logs && "
           f"cd {MOUNT}/app && "
           f"(PE_T2I_URL={shlex.quote(pe_t2i_url)} PE_I2I_URL={shlex.quote(pe_i2i_url)} "
           # the single GPU is entirely the pipeline's now (PEs live elsewhere)
           f"CUDA_VISIBLE_DEVICES=0 "
           f"nohup .venv/bin/marimo run app.py --host 0.0.0.0 --port {APP_PORT} "
           # endpoint is internet-reachable -> marimo token auth ON (the SSH
           # tunnel that used to justify --no-token is gone)
           f'--headless --watch --token --token-password "$APP_TOKEN" '
           f"> {MOUNT}/logs/marimo.log 2>&1 </dev/null &)")


def _finish_studio(studio: Sandbox, state: dict) -> None:
    """Shared tail of `up` and `sync`: wait for marimo, persist, print, open."""
    if not wait_http(studio, APP_PORT, "marimo", timeout=600):
        raise RuntimeError(
            f"marimo did not come up — check {MOUNT}/logs/marimo.log via\n"
            f"  uv run scripts/sandbox.py status  # then exec into the sandbox")
    base = service_url(studio, APP_PORT)
    url = f"{base}/?access_token={state['app_token']}"
    state["studio_url"] = base
    save_state(state)
    print(f"\n✅ ready: {url}")
    print("   stop:  scripts/stop.sh")
    if sys.platform == "darwin":
        subprocess.run(["open", url], check=False)


def cmd_up(_args) -> None:
    dotenv = load_dotenv()
    state = load_state()

    # --- reuse path (like the old find-or-create Slurm job) -------------------
    studio = reconnect(state.get("studio_id"))
    pe = reconnect(state.get("pe_id"))
    if studio and pe:
        print(f"Reusing sandboxes studio={studio.sandbox_id} pe={pe.sandbox_id}")
        upload_code(studio)
        # uv sync is a near-no-op unless pyproject/uv.lock changed since seed
        _ensure_build_tools(studio)
        run_long(studio, "uv-sync", f"cd {MOUNT}/app && {UV} sync", timeout=900)
        _start_marimo(studio, state["pe_t2i_url"], state["pe_i2i_url"])
        _finish_studio(studio, state)
        return
    if studio or pe:  # half-alive pair can't serve; start clean
        print("Only one of the two sandboxes survives — recreating both.")
        reap_by_tag()
        state = load_state()

    snapshot_id = state.get("snapshot_id")
    if not snapshot_id:
        sys.exit("No snapshot yet — run the one-time `scripts/launch.sh --seed` first.")

    # Tokens persist across relaunches so bookmarks/old tabs keep working.
    state.setdefault("pe_api_token", pysecrets.token_urlsafe(32))
    state.setdefault("app_token", pysecrets.token_urlsafe(32))
    save_state(state)

    print("Creating studio + pe sandboxes (parallel, ~60s each)...")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_pe = pool.submit(_create_pe, snapshot_id, state["pe_api_token"])
            f_studio = pool.submit(_create_studio, snapshot_id, dotenv,
                                   state["pe_api_token"], state["app_token"])
            pe, studio = f_pe.result(), f_studio.result()

        # Persist ids immediately so `down` can reap even if bootstrap fails.
        state.update({
            "studio_id": studio.sandbox_id,
            "pe_id": pe.sandbox_id,
            "pe_t2i_url": service_url(pe, PE_T2I_PORT),
            "pe_i2i_url": service_url(pe, PE_I2I_PORT),
            "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        save_state(state)
        print(f"Sandboxes up: studio={studio.sandbox_id} pe={pe.sandbox_id}")

        # Kick off the T2I vLLM FIRST — its multi-minute model load overlaps
        # with the studio bootstrap below.
        _start_vllm(pe, PE_T2I_MODEL, PE_T2I_PORT, 0.45)

        upload_code(studio)  # snapshot code may be stale; local repo is truth
        _ensure_build_tools(studio)
        run_long(studio, "uv-sync", f"cd {MOUNT}/app && {UV} sync", timeout=900)
        _start_marimo(studio, state["pe_t2i_url"], state["pe_i2i_url"])

        pe_ok = wait_http(pe, PE_T2I_PORT, "PE-T2I (first model load takes a few minutes)",
                          path="/health", timeout=900)
        if pe_ok:
            # T2I already holds ~0.45; vLLM requires util*total <= free at startup.
            _start_vllm(pe, PE_I2I_MODEL, PE_I2I_PORT, 0.53)
            pe_ok = wait_http(pe, PE_I2I_PORT, "PE-I2I", path="/health", timeout=900)
        if not pe_ok:
            # Same tolerance as the old launch.sh: the app falls back to raw
            # prompts when enhancers are absent — don't fail the launch.
            print(f"⚠️  enhancers not healthy — continuing; the app falls back "
                  f"to raw prompts.\n    check {MOUNT}/logs/pe_*.log in sandbox "
                  f"{pe.sandbox_id}", file=sys.stderr)

        _finish_studio(studio, state)
    except Exception:
        print("Launch failed — reaping sandboxes.", file=sys.stderr)
        reap_by_tag()
        raise


# --- subcommand: sync ---------------------------------------------------------

def cmd_sync(_args) -> None:
    """Push local code (source of truth) to the running studio and restart
    marimo. Replaces the old rsync-based sync.sh + `launch.sh --no-sync`."""
    state = load_state()
    studio = reconnect(state.get("studio_id"))
    if not studio:
        sys.exit("studio sandbox is not running — use scripts/launch.sh")
    upload_code(studio)
    _ensure_build_tools(studio)
    run_long(studio, "uv-sync", f"cd {MOUNT}/app && {UV} sync", timeout=900)
    _start_marimo(studio, state["pe_t2i_url"], state["pe_i2i_url"])
    _finish_studio(studio, state)


# --- subcommand: down / status --------------------------------------------------

def cmd_down(args) -> None:
    reap_by_tag(delete_snapshot=args.delete_snapshot)


def cmd_status(_args) -> None:
    state = load_state()
    any_found = False
    for role in ("studio", "pe", "seed"):
        for sb in Sandbox.list(tags=[f"{APP_TAG}-{role}"], auth=AUTH).result():
            any_found = True
            urls = "  ".join(u for _p, _n, u in sb.service_urls) or "-"
            print(f"{role:7s} {sb.sandbox_id}  {sb.get_status().name:8s} {urls}")
    if not any_found:
        print("no sandboxes")
    if state.get("launched_at"):
        print(f"launched: {state['launched_at']}")
    print(f"snapshot: {state.get('snapshot_id') or 'none (run scripts/launch.sh --seed)'}")


# --- main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe", help="1-min GPU entitlement check").set_defaults(fn=cmd_probe)
    sub.add_parser("seed", help="one-time snapshot build (venvs + weights)").set_defaults(fn=cmd_seed)
    sub.add_parser("up", help="launch (or reuse) the app sandboxes").set_defaults(fn=cmd_up)
    sub.add_parser("sync", help="push code + restart marimo").set_defaults(fn=cmd_sync)
    down = sub.add_parser("down", help="stop all app sandboxes")
    down.add_argument("--delete-snapshot", action="store_true",
                      help="also delete the FSS snapshot (next launch re-seeds)")
    down.set_defaults(fn=cmd_down)
    sub.add_parser("status", help="list this app's sandboxes").set_defaults(fn=cmd_status)
    args = parser.parse_args()

    require_api_key(load_dotenv())
    args.fn(args)


if __name__ == "__main__":
    main()
