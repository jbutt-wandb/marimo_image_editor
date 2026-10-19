"""Distillation/fine-tuning data accumulation.

Each generation appends rows to the versioned weave.Dataset
`image-generation-log` — ONE ROW PER OUTPUT IMAGE so inputs and output stay
matched: full parameters, original + rewritten prompts, the PE-resolution
input images and the full-resolution output, all as base64 PNG strings
(pure-JSON rows keep the cheap table-append path working). Appends run on a
background thread so GPU latency is unaffected; failures are logged, never
raised — training-data capture must not break the app.
"""

import base64
import io
import logging
import threading

from qwen2_1.telemetry import telemetry_enabled, thumbnail

logger = logging.getLogger(__name__)

DATASET_NAME = "image-generation-log"
PE_INPUT_SIDE = 1024  # inputs stored at the resolution the PE model saw

_lock = threading.Lock()


def _b64_png(img, max_side=None):
    if max_side:
        img = thumbnail(img, max_side)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def log_generation(recipe, input_images, output_images):
    if not telemetry_enabled():
        return
    inputs_b64 = [_b64_png(im, PE_INPUT_SIDE) for im in input_images]
    rows = [
        {
            **recipe,
            "input_images_b64": inputs_b64,
            "output_image_b64": _b64_png(out),
            "variation_index": i,
            "n_outputs": len(output_images),
        }
        for i, out in enumerate(output_images)
    ]
    threading.Thread(target=_append, args=(rows,), daemon=True).start()
    logger.info("dataset append queued | %d row(s) -> %s", len(rows), DATASET_NAME)


def _append(rows):
    try:
        import weave

        with _lock:  # serialize read-append-publish of the dataset version
            try:
                dataset = weave.ref(DATASET_NAME).get()
            except Exception:
                # Legitimate only on the first ever publish — any other cause
                # must be loud (prior versions stay recoverable either way).
                logger.exception(
                    "could not load %s — publishing fresh version", DATASET_NAME
                )
                weave.publish(weave.Dataset(name=DATASET_NAME, rows=rows))
                logger.info("dataset append done | fresh %s", DATASET_NAME)
                return
            weave.publish(dataset.add_rows(rows))
        logger.info("dataset append done | +%d row(s)", len(rows))
    except Exception:
        logger.exception("dataset append failed for %s", DATASET_NAME)
