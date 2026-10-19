"""PencilCanvas: draw-on-image anywidget, ported from the Jupyter scratchpad.

Strokes are synced as [hex_color, brush_width, [(x, y), ...]] in canvas
coordinates (image scaled to `width` px wide). marimo wraps it with
mo.ui.anywidget(); masks.strokes_to_mask / marked_from_strokes rasterize the
result at full image resolution.
"""

import base64
import io

import anywidget
import traitlets


class PencilCanvas(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
      const img = new Image(), canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      let drawing = false, stroke = null;
      canvas.style.cursor = "crosshair"; canvas.style.touchAction = "none";
      const alpha = c => c + "99";
      img.onload = () => {
        canvas.width = model.get("width");
        canvas.height = Math.round(img.height * model.get("width") / img.width);
        redraw();
      };
      img.src = "data:image/png;base64," + model.get("image_b64");
      function redraw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        ctx.lineCap = ctx.lineJoin = "round";
        for (const [c, w, pts] of model.get("strokes")) {
          ctx.strokeStyle = alpha(c); ctx.lineWidth = w; ctx.beginPath();
          pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
          ctx.stroke();
        }
      }
      const pos = e => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
      canvas.addEventListener("pointerdown", e => {
        drawing = true;
        stroke = [model.get("color"), model.get("brush"), [pos(e)]];
      });
      canvas.addEventListener("pointermove", e => {
        if (!drawing) return;
        stroke[2].push(pos(e));
        const p = stroke[2];
        ctx.strokeStyle = alpha(stroke[0]); ctx.lineCap = ctx.lineJoin = "round";
        ctx.lineWidth = stroke[1]; ctx.beginPath();
        ctx.moveTo(...p[p.length - 2]); ctx.lineTo(...p[p.length - 1]); ctx.stroke();
      });
      const up = () => {
        if (!drawing) return;
        drawing = false;
        model.set("strokes", [...model.get("strokes"), stroke]); model.save_changes();
      };
      canvas.addEventListener("pointerup", up);
      canvas.addEventListener("pointerleave", up);
      model.on("change:strokes", redraw);
      model.on("change:image_b64", () => { img.src = "data:image/png;base64," + model.get("image_b64"); });
      el.appendChild(canvas);
    }
    export default { render };
    """
    image_b64 = traitlets.Unicode("").tag(sync=True)
    strokes = traitlets.List([]).tag(sync=True)
    brush = traitlets.Int(14).tag(sync=True)
    width = traitlets.Int(768).tag(sync=True)
    color = traitlets.Unicode("#ff0000").tag(sync=True)


def canvas_for_image(img, display_width=768, color="#ff0000"):
    """PencilCanvas showing img scaled to display_width (keeps aspect)."""
    img = img.convert("RGB")
    scaled = img.resize((display_width, round(img.height * display_width / img.width)))
    buf = io.BytesIO()
    scaled.save(buf, "PNG")
    return PencilCanvas(
        image_b64=base64.b64encode(buf.getvalue()).decode(),
        width=display_width,
        color=color,
    )
