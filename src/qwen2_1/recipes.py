"""Output gallery storage: each generated image is saved to outputs/ with a
JSON sidecar "recipe" recording every parameter needed to reproduce it."""

import json
import time
from pathlib import Path

OUTPUTS_DIR = Path("outputs")


def save_output(image, recipe, outputs_dir=OUTPUTS_DIR):
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    n = sum(1 for _ in outputs_dir.glob(f"{stamp}*.png"))
    path = outputs_dir / (f"{stamp}.png" if n == 0 else f"{stamp}-{n}.png")
    image.save(path)
    recipe = {**recipe, "file": path.name, "created": stamp}
    path.with_suffix(".json").write_text(json.dumps(recipe, indent=2))
    return path, recipe


def list_outputs(outputs_dir=OUTPUTS_DIR):
    """[(png Path, recipe dict)], newest first. Images without a sidecar get {}."""
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.is_dir():
        return []
    items = []
    for png in sorted(outputs_dir.glob("*.png"), reverse=True):
        sidecar = png.with_suffix(".json")
        try:
            recipe = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        except json.JSONDecodeError:
            recipe = {}
        items.append((png, recipe))
    return items
