"""Distillation/fine-tuning data accumulation.

Every generation appends one row to the versioned weave.Dataset
`image-generation-log`: full parameters, original + rewritten prompts, the
PE-resolution input images, and the full-resolution output. Appends run on a
background thread so GPU latency is unaffected, and failures are logged, never
raised — training-data capture must not break the app.
"""

import logging
import threading

from qwen2_1.telemetry import telemetry_enabled, thumbnail

logger = logging.getLogger(__name__)

DATASET_NAME = "image-generation-log"
PE_INPUT_SIDE = 1024  # inputs stored at the resolution the PE model saw

_lock = threading.Lock()


def log_generation(recipe, input_images, output_images):
    if not telemetry_enabled():
        return
    row = {
        **{k: v for k, v in recipe.items() if not isinstance(v, (bytes,))},
        "input_images": [thumbnail(im, PE_INPUT_SIDE) for im in input_images],
        "output_image": output_images[0] if output_images else None,
        "n_outputs": len(output_images),
    }
    threading.Thread(target=_append, args=(row,), daemon=True).start()


def _append(row):
    try:
        import weave

        with _lock:  # serialize read-modify-publish of the dataset version
            # Dataset.add_rows uses the table_update endpoint, which cannot
            # serialize PIL images — only a full publish runs the media
            # serializer. Rebuild rows + republish; the server content-
            # addresses media, so unchanged images dedupe.
            try:
                existing = [dict(r) for r in weave.ref(DATASET_NAME).get().rows]
            except Exception:
                # Dropping prior rows from the latest version is only
                # legitimate on the first ever publish — any other cause
                # must be loud in the log (older versions stay recoverable).
                logger.exception(
                    "could not load %s — publishing fresh single-row version",
                    DATASET_NAME,
                )
                existing = []
            weave.publish(weave.Dataset(name=DATASET_NAME, rows=existing + [row]))
    except Exception:
        logger.exception("failed to append generation to %s", DATASET_NAME)
