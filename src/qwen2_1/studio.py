"""Agentic orchestration for Image Studio.

Structure (OpenAI Agents SDK + code orchestration, per SDK guidance for
deterministic flows): two specialist Agents wrap the vLLM-served prompt
enhancers — PromptEnhancer-T2I and PromptEnhancer-I2I. Python code runs the
right agent, parses its JSON, then calls the diffusers pipeline directly.
The PE models emit one JSON object as text (no tool calling), so the agent's
final_output is parsed manually rather than via output_type.

`generate()` is the per-click trace root: Weave's Agents-SDK autopatch nests
the agent run beneath it (Agents tab), and run_pipe / log_generation appear as
child ops (Traces tab).
"""

import logging
import os
from pathlib import Path

import weave
from pydantic import BaseModel

from qwen2_1.enhance import _extract_json, _image_to_data_url, nearest_aspect
from qwen2_1.model import ASPECT_RATIOS, MODEL_ID, run_pipe
from qwen2_1.recipes import save_output
from qwen2_1 import distill
from qwen2_1.telemetry import thumbnail_logged_values

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "pe_prompts"

# Official pe_core profiles — presence_penalty differs by task and is not
# interchangeable (a wrong penalty silently changes the output distribution).
PE_TASKS = {
    "t2i": {
        "agent_name": "PromptEnhancer-T2I",
        "model": "Qwen/Qwen-Image-2.1-PE-T2I",
        "url_env": "PE_T2I_URL",
        "system_prompt": "system_prompt_t2i.txt",
        "presence_penalty": 1.5,
    },
    "edit": {
        "agent_name": "PromptEnhancer-I2I",
        "model": "Qwen/Qwen-Image-2.1-PE-I2I",
        "url_env": "PE_I2I_URL",
        "system_prompt": "system_prompt_edit.txt",
        "presence_penalty": 0.0,
    },
}


class RewriteResult(BaseModel):
    rewritten_prompt: str
    wh_ratio: str = ""
    ratio_follow: str = ""


_enhancers = {}


def build_enhancer(task):
    """Agent per PE task, cached. None when the server URL isn't configured."""
    if task in _enhancers:
        return _enhancers[task]
    cfg = PE_TASKS[task]
    base_url = os.environ.get(cfg["url_env"])
    if not base_url:
        return None
    from agents import Agent, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel

    # chat-completions model class is required: vLLM has no Responses API
    client = AsyncOpenAI(base_url=base_url.rstrip("/") + "/v1", api_key="dummy")
    agent = Agent(
        name=cfg["agent_name"],
        instructions=(_PROMPTS_DIR / cfg["system_prompt"]).read_text(),
        model=OpenAIChatCompletionsModel(model=cfg["model"], openai_client=client),
        model_settings=ModelSettings(
            temperature=1.0,
            top_p=0.95,
            presence_penalty=cfg["presence_penalty"],
            max_tokens=8192,
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "seed": 42,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        ),
    )
    _enhancers[task] = agent
    return agent


@weave.op
async def enhance_prompt(prompt, images=None, task="t2i", session_id=None):
    """Run the PE agent; fail-open to the raw prompt on any error."""
    fallback = {"prompt": prompt, "wh_ratio": None, "ratio_follow": None, "enhanced": False}
    agent = build_enhancer(task)
    if agent is None:
        return fallback
    try:
        from agents import RunConfig, Runner

        if task == "edit":
            if not images:
                return fallback
            # pe_core protocol: images first (as <=1024px data URLs), text last
            content = [
                {"type": "input_image", "image_url": _image_to_data_url(im)}
                for im in images
            ] + [{"type": "input_text", "text": prompt}]
            run_input = [{"role": "user", "content": content}]
        else:
            run_input = prompt

        result = await Runner.run(
            agent,
            input=run_input,
            max_turns=1,
            run_config=RunConfig(group_id=session_id),
        )
        answer = str(result.final_output or "")
        if "</think>" in answer:
            answer = answer.rsplit("</think>", 1)[1]
        rewritten, obj = _extract_json(answer)
        if rewritten is None:
            return fallback
        parsed = RewriteResult(
            rewritten_prompt=rewritten,
            wh_ratio=str(obj.get("wh_ratio") or "").strip(),
            ratio_follow=str(obj.get("ratio_follow") or "").strip() if task == "edit" else "",
        )
        return {
            "prompt": parsed.rewritten_prompt,
            "wh_ratio": parsed.wh_ratio or None,
            "ratio_follow": parsed.ratio_follow or None,
            "enhanced": True,
        }
    except Exception:
        logger.exception("prompt enhancement failed; using raw prompt")
        return fallback


def _generate_display_name(call):
    return f"{call.inputs.get('mode', 'generate')}"


def _postprocess_inputs(inputs):
    return thumbnail_logged_values(inputs)


@weave.op(call_display_name=_generate_display_name, postprocess_inputs=_postprocess_inputs)
async def generate(
    mode,
    prompt,
    images=None,
    params=None,
    enhance_enabled=True,
    session_id=None,
    recipe_extra=None,
):
    """One user action end-to-end: enhance -> pipeline -> save -> dataset log.

    mode: create | edit | region | multi-region. For non-create modes `images`
    is the ready-to-send list (annotations already baked in; first = target).
    Returns (saved, recipe) where saved is [(Path, recipe), ...].
    """
    params = dict(params or {})
    task = "t2i" if mode == "create" else "edit"

    with weave.attributes(
        {"session_id": session_id, "mode": mode, "model": MODEL_ID,
         "pe_model": PE_TASKS[task]["model"], "enhance_enabled": enhance_enabled}
    ):
        enh = (
            await enhance_prompt(prompt, images=images, task=task, session_id=session_id)
            if enhance_enabled
            else None
        )
        final_prompt = enh["prompt"] if enh and enh["enhanced"] else prompt

        pipe_kwargs = dict(
            prompt=final_prompt,
            steps=params.get("steps", 40),
            true_cfg_scale=params.get("true_cfg_scale", 1.0),
            negative_prompt=params.get("negative_prompt", ""),
            seed=params.get("seed"),
            num_images=params.get("num_images", 1),
            use_kv_cache=params.get("use_kv_cache", True),
        )
        if mode == "create":
            aspect = params.get("aspect", "1:1")
            if aspect == "Auto":
                suggested = (enh or {}).get("wh_ratio")
                aspect = (nearest_aspect(suggested, ASPECT_RATIOS) if suggested else None) or "1:1"
            pipe_kwargs.update(
                aspect=aspect,
                resolution=params.get("resolution", "2K"),
                transparent=params.get("transparent", False),
            )
        else:
            pipe_kwargs["images"] = images

        imgs, recipe = run_pipe(**pipe_kwargs)

        recipe["mode"] = mode
        recipe["original_prompt"] = prompt
        recipe["enhanced"] = bool(enh and enh["enhanced"])
        recipe["pe_model"] = PE_TASKS[task]["model"] if recipe["enhanced"] else None
        recipe["pe_wh_ratio"] = (enh or {}).get("wh_ratio")
        recipe["pe_ratio_follow"] = (enh or {}).get("ratio_follow")
        recipe["session_id"] = session_id
        recipe.update(recipe_extra or {})

        saved = [save_output(im, recipe) for im in imgs]

        distill.log_generation(
            recipe=saved[0][1] if saved else recipe,
            input_images=images or [],
            output_images=imgs,
        )
        return saved, recipe
