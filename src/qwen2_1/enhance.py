"""Shared helpers for the prompt-enhancement protocol (official
prompt_rewrite/pe_core.py): PE-resolution image encoding, the balanced-brace
last-first JSON parse (including the `rewrited_prompt` training typo), and
aspect-ratio mapping. The agents that use these live in studio.py."""

import base64
import io
import json

IMAGE_MAX_PIXELS = 1024 * 1024  # training-time cap; downscale before encoding


def _image_to_data_url(img):
    from PIL import Image

    img = img.convert("RGB")
    if img.width * img.height > IMAGE_MAX_PIXELS:
        scale = (IMAGE_MAX_PIXELS / (img.width * img.height)) ** 0.5
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
        )
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _extract_json(text):
    """Balanced-brace scan for top-level {...} spans (string-literal aware),
    tried last-first — the model emits its JSON at the end of the answer."""
    spans, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start : i + 1])
    for candidate in reversed(spans):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            # `rewrited_prompt` is a known typo in the training data
            rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
            if isinstance(rewritten, str) and rewritten.strip():
                return rewritten.strip(), obj
    return None, None


def nearest_aspect(wh_ratio, aspect_ratios):
    """Map a PE-suggested 'W:H' string onto the closest key in ASPECT_RATIOS
    (the PE vocabulary is wider than the pipeline's resolution table)."""
    try:
        w, h = wh_ratio.split(":")
        target = float(w) / float(h)
    except (ValueError, ZeroDivisionError, AttributeError):
        return None
    return min(aspect_ratios, key=lambda k: abs(target - (lambda a, b: a / b)(*map(float, k.split(":")))))
