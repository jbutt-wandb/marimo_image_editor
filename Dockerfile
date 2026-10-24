# Custom sandbox image for Image Studio — PREPARED, NOT YET BUILT.
#
# Today the launcher uses the stock pytorch runtime image and apt-installs the
# tools below into every fresh sandbox at runtime (_ensure_build_tools in
# scripts/sandbox.py, ~45 s per sandbox, repeated forever because apt writes
# to the container layer and filesystem snapshots capture only /work).
# Baking them into the image removes all runtime apt steps. To adopt it:
#
#   docker build --platform linux/amd64 -t <registry>/image-studio-base:v1 .
#   docker push <registry>/image-studio-base:v1
#   # then adopt it via .env — no code edit needed (see IMAGE in sandbox.py):
#   echo 'SANDBOX_IMAGE=<registry>/image-studio-base@sha256:<digest>' >> .env
#
# Pin by digest, not tag: the serverless GPU pool is heterogeneous (H100 sm_90
# / RTX PRO 6000 sm_120) and launches must stay reproducible on both archs.

# CUDA 12.8 + PyTorch 2.8 — verified to work on BOTH GPU archs the pool
# serves. The *-runtime* flavor is small but ships no build tools; the three
# packages below are exactly what the stack silently depends on at runtime.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# gcc + libc6-dev: Triton JIT-compiles a small host C extension at the FIRST
#   kernel launch — torch imports and the model loads into VRAM fine, then the
#   first real generation dies with "RuntimeError: Failed to find C compiler".
#   Needed by both sandboxes (torch/diffusers Triton ops on studio, vLLM on pe).
# git: pyproject pins diffusers to a git commit (QwenImage21Pipeline is in no
#   PyPI release) and `uv sync` shells out to the git CLI whenever the built
#   wheel isn't already in the /work uv cache.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev git && \
    rm -rf /var/lib/apt/lists/*

# Deliberately NOT in the image: the vLLM FlashInfer kernel wheels, both
# venvs, and the ~80 GB of model weights. Those live under /work and travel in
# the filesystem snapshot (seed/reseed bake them in) — the image only needs to
# fix the container-layer gaps a snapshot can never carry.
