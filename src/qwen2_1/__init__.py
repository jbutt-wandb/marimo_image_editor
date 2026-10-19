from qwen2_1.canvas import PencilCanvas, canvas_for_image
from qwen2_1.enhance import nearest_aspect
from qwen2_1.masks import (
    ANNOTATION_COLORS,
    color_name,
    compose_region_prompt,
    ellipse_mask,
    marked_from_mask,
    marked_from_strokes,
    strokes_by_color,
    strokes_to_mask,
)
from qwen2_1.model import (
    ASPECT_RATIOS,
    MODEL_ID,
    RESOLUTIONS,
    load_pipe,
    pipe_is_loaded,
    run_pipe,
)


def main() -> None:
    print("Hello from qwen2-1!")
