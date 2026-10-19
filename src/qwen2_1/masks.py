"""Mask and annotation helpers.

Qwen-Image-2.1's diffusers pipeline has no mask input: region editing works by
baking a visible colored annotation into the input image and referring to it
in the prompt ("in the red marked region, ..."). These helpers convert canvas
strokes or programmatic masks into such annotated images.
"""

from PIL import Image, ImageChops, ImageDraw, ImageFilter

# Hex values must match the swatches offered by the PencilCanvas widget.
ANNOTATION_COLORS = {
    "red": "#ff0000",
    "green": "#00b140",
    "blue": "#0066ff",
    "yellow": "#ffd400",
}
_HEX_TO_NAME = {v: k for k, v in ANNOTATION_COLORS.items()}
_HEX_TO_RGB = {v: tuple(int(v[i : i + 2], 16) for i in (1, 3, 5)) for v in _HEX_TO_NAME}


def color_name(hex_color):
    return _HEX_TO_NAME.get(hex_color.lower(), hex_color)


def ellipse_mask(size, x, y, w, h):
    """Binary mask (white = region to edit); x, y, w, h are fractions of the image size."""
    m = Image.new("L", size, 0)
    cx, cy, ex, ey = x * size[0], y * size[1], w * size[0] / 2, h * size[1] / 2
    ImageDraw.Draw(m).ellipse([cx - ex, cy - ey, cx + ex, cy + ey], fill=255)
    return m


def marked_from_mask(img, mask, color=(255, 0, 0), line=None):
    """Copy of img with the mask boundary drawn in `color` — the annotation the model reads."""
    line = line or max(5, img.width // 300)
    edge = (
        mask.convert("L")
        .resize(img.size)
        .filter(ImageFilter.FIND_EDGES)
        .filter(ImageFilter.MaxFilter(line | 1))
        .point(lambda p: 255 if p else 0)
    )
    out = img.convert("RGB").copy()
    out.paste(color, (0, 0), edge)
    return out


def strokes_to_mask(strokes, size, canvas_width, fill_enclosed=True):
    """Rasterize PencilCanvas strokes ([color, brush_width, [(x, y), ...]]) into a
    white-on-black mask at the full image size. Canvas coordinates are scaled up
    by size[0] / canvas_width. With fill_enclosed, closed outlines are filled."""
    sx = size[0] / canvas_width
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for _color, wd, pts in strokes:
        scaled = [(px * sx, py * sx) for px, py in pts]
        lw = max(2, round(wd * sx))
        if len(scaled) > 1:
            d.line(scaled, fill=255, width=lw, joint="curve")
        for px, py in (scaled[0], scaled[-1]):
            d.ellipse([px - lw / 2, py - lw / 2, px + lw / 2, py + lw / 2], fill=255)
    if fill_enclosed and strokes:
        outside = m.copy()
        ImageDraw.floodfill(outside, (0, 0), 128)
        m = ImageChops.lighter(m, outside.point(lambda p: 255 if p == 0 else 0))
    return m


def strokes_by_color(strokes):
    """Group strokes into {hex_color: [stroke, ...]} preserving draw order."""
    groups = {}
    for stroke in strokes:
        groups.setdefault(stroke[0], []).append(stroke)
    return groups


def marked_from_strokes(img, strokes, canvas_width, fill_enclosed=True):
    """Annotated copy of img: each color's strokes become a boundary line in that
    color. Returns (annotated image, [hex colors used in draw order])."""
    out = img
    groups = strokes_by_color(strokes)
    for hex_color, group in groups.items():
        mask = strokes_to_mask(group, img.size, canvas_width, fill_enclosed)
        out = marked_from_mask(out, mask, color=_HEX_TO_RGB.get(hex_color, (255, 0, 0)))
    return out, list(groups)


def compose_region_prompt(instructions):
    """instructions: {hex_color: instruction}. Builds the combined multi-region
    prompt, e.g. 'In the red marked region, X. In the green marked region, Y.'"""
    parts = []
    for hex_color, text in instructions.items():
        text = text.strip().rstrip(".")
        if text:
            parts.append(f"In the {color_name(hex_color)} marked region, {text}.")
    return " ".join(parts)
