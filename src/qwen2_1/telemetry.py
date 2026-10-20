"""Weave (W&B / CoreWeave Forge) tracing setup.

Tracing is strictly optional: without WANDB_API_KEY the app runs untraced.
weave ops execute as plain functions when weave.init was never called, so the
decorators elsewhere in the package are safe either way.
"""

import logging
import os
import sys

WEAVE_PROJECT = os.environ.get("WEAVE_PROJECT")  # e.g. "entity/project"; unset -> untraced
TRACE_THUMB_SIDE = 1024  # max image side logged into traces (server caps requests)

_initialized = False


def setup_logging():
    """Stage logging for the terminal (ends up in marimo.log). Idempotent."""
    pkg_logger = logging.getLogger("qwen2_1")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
        )
        pkg_logger.addHandler(handler)
        pkg_logger.setLevel(logging.INFO)
        pkg_logger.propagate = False
    return pkg_logger


def init_telemetry():
    """Init weave once if credentials exist. Returns True when tracing is on.
    Fail-open: any init error leaves the app running untraced."""
    global _initialized
    setup_logging()
    if _initialized:
        return True
    if not os.environ.get("WANDB_API_KEY"):
        logging.getLogger(__name__).info("telemetry off (no WANDB_API_KEY)")
        return False
    if not WEAVE_PROJECT:
        logging.getLogger(__name__).info("telemetry off (no WEAVE_PROJECT)")
        return False
    try:
        logging.getLogger("weave").setLevel(logging.ERROR)
        # the Agents SDK's default exporter would try to upload to OpenAI; keep
        # tracing enabled (Weave's autopatch consumes the spans) but quiet the
        # exporter's missing-key chatter
        logging.getLogger("openai.agents").setLevel(logging.ERROR)
        import weave

        weave.init(WEAVE_PROJECT)
        # Explicitly route Agents-SDK spans to Weave (the autopatch didn't take
        # in our setup) — this also replaces the SDK's default OpenAI exporter,
        # which would otherwise warn about the missing OPENAI_API_KEY.
        from agents import set_trace_processors
        from weave.integrations.openai_agents.openai_agents import (
            WeaveTracingProcessor,
        )

        set_trace_processors([WeaveTracingProcessor()])
        _initialized = True
        logging.getLogger(__name__).info("telemetry on -> %s", WEAVE_PROJECT)
        return True
    except Exception:
        logging.getLogger(__name__).exception("weave init failed; tracing off")
        return False


def telemetry_enabled():
    return _initialized


def thumbnail(img, max_side=TRACE_THUMB_SIDE):
    """Downscaled copy for logging; original is untouched."""
    from PIL import Image

    if max(img.size) <= max_side:
        return img
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def thumbnail_logged_values(value):
    """Recursively replace PIL images with thumbnails in logged inputs/outputs."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return thumbnail(value)
    if isinstance(value, (list, tuple)):
        return type(value)(thumbnail_logged_values(v) for v in value)
    if isinstance(value, dict):
        return {k: thumbnail_logged_values(v) for k, v in value.items()}
    return value
