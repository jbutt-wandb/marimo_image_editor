"""Qwen-Image-2.1 pipeline wrapper.

One QwenImage21Pipeline instance serves every mode: mode is selected by what
is passed as `image=` (None = text-to-image, single PIL = edit, list = multi-
reference, max 10). Only one process may hold the model (~38 GB bf16 on an
80 GB H100), so the pipeline is a module-level singleton.
"""

import random

import weave

from qwen2_1.telemetry import thumbnail_logged_values

# Recommended 2K resolutions. The pipeline falls back to a 1024 total-area
# target when width/height are omitted, so Create mode always passes these
# explicitly. Edit modes omit them to inherit the input image's size.
ASPECT_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

# Dimension divisors stay valid at half scale (all multiples of 16).
RESOLUTIONS = {"2K": 1.0, "1K": 0.5}

RGBA_PROMPT_TEMPLATE = (
    "This is an RGBA image with transparency. {prompt}. "
    "The image has alpha channel and the background is transparent."
)

MODEL_ID = "Qwen/Qwen-Image-2.1"

_pipe = None


def load_pipe(model_id=MODEL_ID, device=None):
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import QwenImage21Pipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps"
        _pipe = QwenImage21Pipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(device)
    return _pipe


def pipe_is_loaded():
    return _pipe is not None


@weave.op(
    postprocess_inputs=thumbnail_logged_values,
    postprocess_output=thumbnail_logged_values,
)
def run_pipe(
    prompt,
    images=None,
    steps=40,
    true_cfg_scale=1.0,
    negative_prompt="",
    aspect=None,
    resolution="2K",
    seed=None,
    transparent=False,
    num_images=1,
    use_kv_cache=True,
):
    """Run any mode; returns (list of PIL images, recipe dict).

    images: None | PIL.Image | list[PIL.Image] (first = edit target, rest refs)
    aspect/resolution: key into ASPECT_RATIOS / RESOLUTIONS; ignored in edit
        modes (output inherits input size)
    seed: None -> random (the drawn seed is recorded in the recipe)
    num_images: variations per run (same prompt, consecutive noise draws)
    """
    import torch

    pipe = load_pipe()

    if seed is None:
        seed = random.randrange(2**31)
    if transparent:
        prompt = RGBA_PROMPT_TEMPLATE.format(prompt=prompt.rstrip(". "))

    kwargs = dict(
        prompt=prompt,
        num_inference_steps=int(steps),
        num_images_per_prompt=int(num_images),
        use_kv_cache=bool(use_kv_cache),
        generator=torch.Generator(pipe.device.type).manual_seed(int(seed)),
    )

    # CFG only activates when true_cfg_scale > 1 AND a negative prompt is set;
    # the pipeline warns and ignores either one alone.
    if true_cfg_scale > 1 and negative_prompt.strip():
        kwargs["true_cfg_scale"] = float(true_cfg_scale)
        kwargs["negative_prompt"] = negative_prompt

    recipe = {
        "model": MODEL_ID,
        "prompt": prompt,
        "negative_prompt": negative_prompt if "negative_prompt" in kwargs else None,
        "true_cfg_scale": kwargs.get("true_cfg_scale", 1.0),
        "steps": int(steps),
        "seed": int(seed),
        "transparent": transparent,
        "num_images": int(num_images),
        "use_kv_cache": bool(use_kv_cache),
    }

    if images is None:
        scale = RESOLUTIONS[resolution]
        width, height = (round(d * scale) for d in ASPECT_RATIOS[aspect or "1:1"])
        kwargs.update(width=width, height=height)
        recipe.update(
            mode="create",
            aspect=aspect or "1:1",
            resolution=resolution,
            width=width,
            height=height,
        )
    else:
        if not isinstance(images, (list, tuple)):
            images = [images]
        if len(images) > 10:
            raise ValueError("the model supports at most 10 input images")
        kwargs["image"] = [im.convert("RGBA") for im in images]
        if len(kwargs["image"]) == 1:
            kwargs["image"] = kwargs["image"][0]
        recipe.update(mode="edit", n_images=len(images))

    return pipe(**kwargs).images, recipe
