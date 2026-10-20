import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium", app_title="Image Studio")


@app.cell
def _():
    import io

    import marimo as mo
    from PIL import Image

    from qwen2_1 import (
        ANNOTATION_COLORS,
        ASPECT_RATIOS,
        MODEL_ID,
        RESOLUTIONS,
        canvas_for_image,
        compose_region_prompt,
        marked_from_mask,
        marked_from_strokes,
        pipe_is_loaded,
        strokes_by_color,
        strokes_to_mask,
    )
    from qwen2_1 import studio
    from qwen2_1.telemetry import WEAVE_PROJECT, init_telemetry
    return (
        ANNOTATION_COLORS,
        ASPECT_RATIOS,
        Image,
        MODEL_ID,
        RESOLUTIONS,
        WEAVE_PROJECT,
        canvas_for_image,
        compose_region_prompt,
        init_telemetry,
        io,
        marked_from_mask,
        marked_from_strokes,
        mo,
        pipe_is_loaded,
        strokes_by_color,
        strokes_to_mask,
        studio,
    )


@app.cell
def _(init_telemetry):
    import logging
    import uuid

    tracing_on = init_telemetry()
    session_id = str(uuid.uuid4())
    log = logging.getLogger("qwen2_1.app")
    log.info("session start | id=%s tracing=%s", session_id, tracing_on)
    return log, session_id, tracing_on


@app.cell
def _(MODEL_ID, WEAVE_PROJECT, get_refresh, mo, pipe_is_loaded, tracing_on):
    get_refresh()
    model_select = mo.ui.dropdown(
        options=[MODEL_ID], value=MODEL_ID, label="Model"
    )
    _tracing_chip = (
        f"📡 tracing → [{WEAVE_PROJECT}](https://wandb.ai/{WEAVE_PROJECT}/weave)"
        if tracing_on
        else "⚪ tracing off (no WANDB_API_KEY)"
    )
    mo.vstack(
        [
            mo.md(
                f"""
    # Image Studio

    Generate and edit images with instruction-driven diffusion models · status:
    {"✅ model loaded" if pipe_is_loaded() else "💤 model loads on first generation (allow a couple of minutes)"}
    · {_tracing_chip}
    """
            ),
            model_select,
        ]
    )
    return (model_select,)


@app.cell
def _(mo):
    get_refresh, set_refresh = mo.state(0)
    return get_refresh, set_refresh


@app.cell
def _():
    # Last finished generation, so input tweaks don't blank the results area.
    # Plain dict on purpose: mutating it must not retrigger the results cell.
    last_result = {}
    return (last_result,)


@app.cell
def _(io, mo):
    def parse_seed(text):
        """'' -> None (random); otherwise int or a user-facing stop."""
        text = text.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            mo.stop(True, mo.md(f"⚠️ seed must be an integer, got `{text}`"))

    def upload_to_image(upload_file):
        from PIL import Image as _Image

        return _Image.open(io.BytesIO(upload_file.contents))

    def encode_pngs(images):
        """PNG-encode once per generation; re-displays reuse the bytes."""
        encoded = []
        for im in images:
            buf = io.BytesIO()
            im.save(buf, "PNG")
            encoded.append(buf.getvalue())
        return encoded

    def show_results(pngs, recipe):
        """Inline display of generated images with download buttons.
        Images live in Weave (traces + dataset), not on disk."""
        items = []
        for i, data in enumerate(pngs):
            items.append(mo.image(data))
            items.append(
                mo.download(
                    data=data,
                    filename=f"{recipe['mode']}-seed{recipe['seed']}-{i}.png",
                    label="Download",
                )
            )
        caption = f"seed `{recipe['seed']}`"
        if recipe.get("enhanced"):
            caption += " · prompt enhanced"
        return mo.vstack(items + [mo.md(caption)])
    return encode_pngs, parse_seed, show_results, upload_to_image


@app.cell
def _(mo):
    steps = mo.ui.slider(1, 60, value=40, label="Steps", show_value=True)
    seed_field = mo.ui.text(placeholder="random", label="Seed")
    guidance = mo.ui.slider(
        1.0, 5.0, step=0.5, value=1.0, label="Guidance (true CFG)", show_value=True
    )
    negative = mo.ui.text(
        placeholder="negative prompt — only active when guidance > 1",
        label="Negative",
        full_width=True,
    )
    variations = mo.ui.slider(
        1, 4, value=1, label="Variations per run", show_value=True
    )
    kv_cache = mo.ui.checkbox(
        value=True, label="KV cache (faster edits; uses extra VRAM)"
    )
    enhance_toggle = mo.ui.checkbox(
        value=True,
        label="Enhance prompts (rewrite via the PE model; falls back to your raw prompt if unavailable)",
    )
    settings_panel = mo.vstack(
        [
            mo.hstack([steps, seed_field], justify="start", gap=2),
            mo.accordion(
                {
                    "Advanced": mo.vstack(
                        [
                            guidance,
                            negative,
                            mo.md(
                                "_Guidance and negative prompt act together: the model "
                                "ignores either one on its own._"
                            ),
                            mo.hstack([variations, kv_cache], justify="start", gap=2),
                            enhance_toggle,
                        ]
                    )
                }
            ),
        ]
    )
    return (
        enhance_toggle,
        guidance,
        kv_cache,
        negative,
        seed_field,
        settings_panel,
        steps,
        variations,
    )


@app.cell
def _(ASPECT_RATIOS, RESOLUTIONS, mo):
    create_prompt = mo.ui.text_area(
        placeholder="Describe the image you want…", label="Prompt", full_width=True, rows=4
    )
    create_aspect = mo.ui.dropdown(
        options=["Auto"] + list(ASPECT_RATIOS), value="Auto", label="Aspect ratio"
    )
    create_resolution = mo.ui.dropdown(
        options=list(RESOLUTIONS), value="2K", label="Resolution"
    )
    create_transparent = mo.ui.checkbox(label="Transparent background (RGBA)")
    create_btn = mo.ui.run_button(label="Generate", kind="success")
    return create_aspect, create_btn, create_prompt, create_resolution, create_transparent


@app.cell
def _(mo):
    edit_upload = mo.ui.file(
        kind="area",
        multiple=True,
        filetypes=[".png", ".jpg", ".jpeg", ".webp"],
        label="Images (1–10 — the first is the edit target, the rest are references)",
    )
    edit_prompt = mo.ui.text_area(
        placeholder="Describe the changes you want to make…",
        label="Instruction",
        full_width=True,
        rows=4,
    )
    edit_btn = mo.ui.run_button(label="Edit", kind="success")
    return edit_btn, edit_prompt, edit_upload


@app.cell
def _(edit_upload, log, mo):
    if edit_upload.value:
        log.info(
            "edit upload | %d file(s): %s",
            len(edit_upload.value),
            ", ".join(f.name for f in edit_upload.value),
        )
        edit_preview = mo.hstack(
            [
                mo.image(f.contents, width=180, rounded=True, caption=f.name)
                for f in edit_upload.value
            ],
            justify="start",
            gap=1,
            wrap=True,
        )
    else:
        edit_preview = mo.md("")
    return (edit_preview,)


@app.cell
def _(mo):
    region_upload = mo.ui.file(
        kind="area",
        multiple=False,
        filetypes=[".png", ".jpg", ".jpeg", ".webp"],
        label="Base image to edit",
    )
    region_refs_upload = mo.ui.file(
        kind="button",
        multiple=True,
        filetypes=[".png", ".jpg", ".jpeg", ".webp"],
        label="Optional reference images",
    )
    region_brush = mo.ui.slider(2, 60, value=14, label="Brush", show_value=True)
    region_fill = mo.ui.checkbox(value=True, label="Fill enclosed areas")
    region_clear = mo.ui.run_button(label="Clear strokes")
    region_prompt = mo.ui.text_area(
        value="In the red marked region, ",
        label="Instruction",
        full_width=True,
        rows=3,
    )
    region_btn = mo.ui.run_button(label="Edit region", kind="success")
    return (
        region_brush,
        region_btn,
        region_clear,
        region_fill,
        region_prompt,
        region_refs_upload,
        region_upload,
    )


@app.cell
def _(canvas_for_image, mo, region_upload, upload_to_image):
    if region_upload.value:
        region_base = upload_to_image(region_upload.value[0])
        region_canvas_raw = canvas_for_image(region_base)
        region_canvas = mo.ui.anywidget(region_canvas_raw)
        region_canvas_view = region_canvas
    else:
        region_base = None
        region_canvas_raw = None
        region_canvas = None
        region_canvas_view = mo.md("_Upload a base image to start drawing._")
    return region_base, region_canvas, region_canvas_raw, region_canvas_view


@app.cell
def _(region_brush, region_canvas_raw, region_clear):
    if region_canvas_raw is not None:
        region_canvas_raw.brush = region_brush.value
        if region_clear.value:
            region_canvas_raw.strokes = []
    return


@app.cell
def _(ANNOTATION_COLORS, mo):
    mr_upload = mo.ui.file(
        kind="area",
        multiple=False,
        filetypes=[".png", ".jpg", ".jpeg", ".webp"],
        label="Base image to edit",
    )
    mr_color = mo.ui.radio(
        options=list(ANNOTATION_COLORS), value="red", label="Stroke color", inline=True
    )
    mr_brush = mo.ui.slider(2, 60, value=14, label="Brush", show_value=True)
    mr_fill = mo.ui.checkbox(value=True, label="Fill enclosed areas")
    mr_clear = mo.ui.run_button(label="Clear strokes")
    mr_instructions = {
        name: mo.ui.text(
            placeholder=f"what to do in the {name} region…",
            label=name,
            full_width=True,
        )
        for name in ANNOTATION_COLORS
    }
    mr_btn = mo.ui.run_button(label="Edit regions", kind="success")
    return mr_brush, mr_btn, mr_clear, mr_color, mr_fill, mr_instructions, mr_upload


@app.cell
def _(canvas_for_image, mo, mr_upload, upload_to_image):
    if mr_upload.value:
        mr_base = upload_to_image(mr_upload.value[0])
        mr_canvas_raw = canvas_for_image(mr_base)
        mr_canvas = mo.ui.anywidget(mr_canvas_raw)
        mr_canvas_view = mr_canvas
    else:
        mr_base = None
        mr_canvas_raw = None
        mr_canvas = None
        mr_canvas_view = mo.md("_Upload a base image to start drawing._")
    return mr_base, mr_canvas, mr_canvas_raw, mr_canvas_view


@app.cell
def _(ANNOTATION_COLORS, mr_brush, mr_canvas_raw, mr_clear, mr_color):
    if mr_canvas_raw is not None:
        mr_canvas_raw.brush = mr_brush.value
        mr_canvas_raw.color = ANNOTATION_COLORS[mr_color.value]
        if mr_clear.value:
            mr_canvas_raw.strokes = []
    return


@app.cell
def _(
    create_aspect,
    create_btn,
    create_prompt,
    create_resolution,
    create_transparent,
    edit_btn,
    edit_preview,
    edit_prompt,
    edit_upload,
    mo,
    mr_btn,
    mr_canvas_view,
    mr_clear,
    mr_color,
    mr_brush,
    mr_fill,
    mr_instructions,
    mr_upload,
    region_brush,
    region_btn,
    region_canvas_view,
    region_clear,
    region_fill,
    region_prompt,
    region_refs_upload,
    region_upload,
    settings_panel,
):
    _create_tab = mo.vstack(
        [
            create_prompt,
            mo.hstack(
                [create_aspect, create_resolution, create_transparent],
                justify="start",
                gap=2,
            ),
            create_btn,
        ]
    )
    _edit_tab = mo.vstack([edit_upload, edit_preview, edit_prompt, edit_btn])
    _region_tab = mo.vstack(
        [
            region_upload,
            region_canvas_view,
            mo.hstack([region_brush, region_fill, region_clear], justify="start", gap=2),
            region_refs_upload,
            region_prompt,
            region_btn,
        ]
    )
    _mr_tab = mo.vstack(
        [
            mr_upload,
            mr_canvas_view,
            mo.hstack([mr_color, mr_brush], justify="start", gap=2),
            mo.hstack([mr_fill, mr_clear], justify="start", gap=2),
            mo.md("**Instructions per color** (only colors you drew with are used):"),
            mo.vstack(list(mr_instructions.values())),
            mr_btn,
        ]
    )
    mo.vstack(
        [
            settings_panel,
            mo.ui.tabs(
                {
                    "✨ Create": _create_tab,
                    "🖊 Edit": _edit_tab,
                    "🎯 Region": _region_tab,
                    "🌈 Multi-region": _mr_tab,
                }
            ),
        ]
    )
    return


@app.cell
async def _(
    ANNOTATION_COLORS,
    compose_region_prompt,
    create_aspect,
    create_btn,
    create_prompt,
    create_resolution,
    create_transparent,
    edit_btn,
    edit_prompt,
    edit_upload,
    encode_pngs,
    enhance_toggle,
    guidance,
    kv_cache,
    last_result,
    log,
    marked_from_mask,
    marked_from_strokes,
    mo,
    mr_base,
    mr_btn,
    mr_canvas_raw,
    mr_fill,
    mr_instructions,
    negative,
    parse_seed,
    region_base,
    region_btn,
    region_canvas_raw,
    region_fill,
    region_prompt,
    region_refs_upload,
    seed_field,
    session_id,
    set_refresh,
    show_results,
    steps,
    strokes_to_mask,
    studio,
    upload_to_image,
    variations,
):
    # Single results area shared by all four tasks: a new run always replaces
    # whatever was shown before, instead of each task keeping its own output.
    _mode = next(
        (
            m
            for m, b in (
                ("create", create_btn),
                ("edit", edit_btn),
                ("region", region_btn),
                ("multi-region", mr_btn),
            )
            if b.value
        ),
        None,
    )
    # No run requested (an input merely changed): keep showing the last result.
    mo.stop(_mode is None, show_results(**last_result) if last_result else None)

    # A run just started: hide the previous result right away.
    mo.output.replace(mo.md(""))
    log.info("run requested | mode=%s", _mode)

    _params = dict(
        steps=steps.value,
        true_cfg_scale=guidance.value,
        negative_prompt=negative.value,
        seed=parse_seed(seed_field.value),
        num_images=variations.value,
        use_kv_cache=kv_cache.value,
    )
    _kwargs = {}

    if _mode == "create":
        mo.stop(not create_prompt.value.strip(), mo.md("⚠️ enter a prompt first"))
        _prompt = create_prompt.value
        _params.update(
            aspect=create_aspect.value,
            resolution=create_resolution.value,
            transparent=create_transparent.value,
        )
        _spin = "Generating…"
    elif _mode == "edit":
        mo.stop(not edit_upload.value, mo.md("⚠️ upload at least one image"))
        mo.stop(len(edit_upload.value) > 10, mo.md("⚠️ at most 10 images"))
        mo.stop(not edit_prompt.value.strip(), mo.md("⚠️ enter an instruction first"))
        _prompt = edit_prompt.value
        _kwargs["images"] = [upload_to_image(f) for f in edit_upload.value]
        _kwargs["recipe_extra"] = {"inputs": [f.name for f in edit_upload.value]}
        _spin = "Editing…"
    elif _mode == "region":
        mo.stop(region_base is None, mo.md("⚠️ upload a base image first"))
        mo.stop(
            not region_canvas_raw.strokes,
            mo.md("⚠️ draw on the image to mark a region"),
        )
        _mask = strokes_to_mask(
            region_canvas_raw.strokes,
            region_base.size,
            region_canvas_raw.width,
            fill_enclosed=region_fill.value,
        )
        _refs = [upload_to_image(f) for f in (region_refs_upload.value or [])]
        _prompt = region_prompt.value
        _kwargs["images"] = [marked_from_mask(region_base, _mask)] + _refs
        _kwargs["recipe_extra"] = {"n_refs": len(_refs)}
        _spin = "Editing region…"
    else:  # multi-region
        mo.stop(mr_base is None, mo.md("⚠️ upload a base image first"))
        mo.stop(
            not mr_canvas_raw.strokes,
            mo.md("⚠️ draw on the image to mark regions"),
        )
        _annotated, _used_hex = marked_from_strokes(
            mr_base,
            mr_canvas_raw.strokes,
            mr_canvas_raw.width,
            fill_enclosed=mr_fill.value,
        )
        _hex_to_name = {v: k for k, v in ANNOTATION_COLORS.items()}
        _instructions = {
            h: mr_instructions[_hex_to_name[h]].value
            for h in _used_hex
            if _hex_to_name.get(h) and mr_instructions[_hex_to_name[h]].value.strip()
        }
        mo.stop(
            not _instructions,
            mo.md("⚠️ write an instruction for at least one color you drew with"),
        )
        _prompt = compose_region_prompt(_instructions)
        _kwargs["images"] = [_annotated]
        _spin = "Editing regions…"

    with mo.status.spinner(_spin):
        _imgs, _recipe = await studio.generate(
            mode=_mode,
            prompt=_prompt,
            params=_params,
            enhance_enabled=enhance_toggle.value,
            session_id=session_id,
            **_kwargs,
        )
    set_refresh(lambda n: n + 1)
    last_result.clear()
    last_result.update(pngs=encode_pngs(_imgs), recipe=_recipe)
    log.info("results displayed | mode=%s images=%d", _mode, len(_imgs))
    mo.output.replace(show_results(**last_result))
    return


@app.cell
def _(WEAVE_PROJECT, mo):
    mo.md(
        f"_Generated images are not stored on the cluster — every generation is "
        f"traced and archived in [Weave](https://wandb.ai/{WEAVE_PROJECT}/weave)._"
    )
    return


if __name__ == "__main__":
    app.run()
