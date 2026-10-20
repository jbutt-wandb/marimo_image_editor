"""Agentic orchestration for Image Studio.

Trace structure (one SDK trace per user click, consumed by Weave's
WeaveTracingProcessor so both agents render in the Agents tab):

    create image | edit image | edit image region(s)   (SDK trace root;
    │                              seed/mode/session in trace metadata)
    ├─ Agent workflow (task span, added by Runner.run)
    │    └─ PromptEnhancer-T2I|I2I     (real Agents-SDK agent run)
    │         └─ turn 1 └─ generation (tokens, rewritten prompt)
    └─ Agent workflow (task span, synthetic — mirrors Runner.run)
         └─ ImageGenerator|ImageEditor|RegionEditor|MultiRegionEditor
              └─ turn 1 └─ run_pipe   (weave.op: params + thumbnailed images)

The PromptEnhancer agents wrap the vLLM-served PE models; they emit one JSON
object as text (no tool calling), so final_output is parsed manually. The
image-task agent is synthetic — an SDK agent_span around the diffusers call,
since diffusion isn't an LLM — purely so the task shows up as an agent.
"""

import logging
import os
import random
import time
from pathlib import Path

import weave
from pydantic import BaseModel

from qwen2_1.enhance import _extract_json, _image_to_data_url, nearest_aspect
from qwen2_1.model import ASPECT_RATIOS, MODEL_ID, run_pipe
from qwen2_1 import distill

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

IMAGE_AGENT_NAMES = {
    "create": "ImageGenerator",
    "edit": "ImageEditor",
    "region": "RegionEditor",
    "multi-region": "MultiRegionEditor",
}

# Human task names for the SDK trace root (seed lives in trace metadata).
TRACE_NAMES = {
    "create": "create image",
    "edit": "edit image",
    "region": "edit image region",
    "multi-region": "edit image regions",
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


async def enhance_prompt(prompt, images=None, task="t2i"):
    """Run the PE agent; fail-open to the raw prompt on any error."""
    fallback = {"prompt": prompt, "wh_ratio": None, "ratio_follow": None, "enhanced": False}
    agent = build_enhancer(task)
    if agent is None:
        logger.warning("enhancer %s not configured; using raw prompt", task)
        return fallback
    try:
        from agents import Runner

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

        result = await Runner.run(agent, input=run_input, max_turns=1)
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


async def generate(
    mode,
    prompt,
    images=None,
    params=None,
    enhance_enabled=True,
    session_id=None,
    recipe_extra=None,
):
    """One user action end-to-end: enhance agent -> image agent -> dataset log.

    mode: create | edit | region | multi-region. For non-create modes `images`
    is the ready-to-send list (annotations already baked in; first = target).
    Returns (list of PIL images, recipe dict).
    """
    from agents import trace
    from agents.tracing import agent_span, function_span, task_span, turn_span

    params = dict(params or {})
    task = "t2i" if mode == "create" else "edit"
    seed = params.get("seed")
    if seed is None:
        seed = random.randrange(2**31)
    params["seed"] = seed

    logger.info(
        "action start | mode=%s seed=%s enhance=%s prompt=%r",
        mode, seed, enhance_enabled, prompt[:60],
    )

    with weave.attributes(
        {"session_id": session_id, "mode": mode, "model": MODEL_ID,
         "pe_model": PE_TASKS[task]["model"], "enhance_enabled": enhance_enabled}
    ):
        with trace(TRACE_NAMES[mode], group_id=session_id,
                   metadata={"mode": mode, "session_id": str(session_id),
                             "seed": str(seed),
                             "enhance_enabled": str(enhance_enabled)}):
            t0 = time.monotonic()
            enh = (
                await enhance_prompt(prompt, images=images, task=task)
                if enhance_enabled
                else None
            )
            logger.info(
                "enhance done | %.1fs enhanced=%s wh_ratio=%s",
                time.monotonic() - t0,
                bool(enh and enh["enhanced"]),
                (enh or {}).get("wh_ratio"),
            )
            final_prompt = enh["prompt"] if enh and enh["enhanced"] else prompt

            pipe_kwargs = dict(
                prompt=final_prompt,
                steps=params.get("steps", 40),
                true_cfg_scale=params.get("true_cfg_scale", 1.0),
                negative_prompt=params.get("negative_prompt", ""),
                seed=seed,
                num_images=params.get("num_images", 1),
                use_kv_cache=params.get("use_kv_cache", True),
            )
            if mode == "create":
                aspect = params.get("aspect", "1:1")
                if aspect == "Auto":
                    suggested = (enh or {}).get("wh_ratio")
                    aspect = (
                        nearest_aspect(suggested, ASPECT_RATIOS) if suggested else None
                    ) or "1:1"
                pipe_kwargs.update(
                    aspect=aspect,
                    resolution=params.get("resolution", "2K"),
                    transparent=params.get("transparent", False),
                )
            else:
                pipe_kwargs["images"] = images

            logger.info("pipeline start | mode=%s steps=%s", mode, pipe_kwargs["steps"])
            t1 = time.monotonic()
            # Synthetic agent for the GPU step. The task>agent>turn nesting
            # mirrors what Runner.run emits — Weave's Agents view only ingests
            # spans inside such an "Agent workflow" task subtree, so a bare
            # agent_span shows in traces but never in the Agents tab.
            with task_span(name="Agent workflow"):
                with agent_span(name=IMAGE_AGENT_NAMES[mode]):
                    with turn_span(turn=1, agent_name=IMAGE_AGENT_NAMES[mode]):
                        # function_span -> execute_tool in Agent Lens; never put
                        # PIL images in span data (strings only).
                        with function_span(
                            name="run_pipe",
                            input=str(
                                {k: v for k, v in pipe_kwargs.items() if k != "images"}
                            ),
                        ) as fs:
                            imgs, recipe = run_pipe(**pipe_kwargs)
                            fs.span_data.output = (
                                f"{len(imgs)} image(s) {imgs[0].size}" if imgs else "0 images"
                            )
            logger.info(
                "pipeline done | %.1fs outputs=%d size=%s",
                time.monotonic() - t1, len(imgs), imgs[0].size if imgs else None,
            )

            recipe["mode"] = mode
            recipe["original_prompt"] = prompt
            recipe["enhanced"] = bool(enh and enh["enhanced"])
            recipe["pe_model"] = PE_TASKS[task]["model"] if recipe["enhanced"] else None
            recipe["pe_wh_ratio"] = (enh or {}).get("wh_ratio")
            recipe["pe_ratio_follow"] = (enh or {}).get("ratio_follow")
            recipe["session_id"] = session_id
            recipe["created"] = time.strftime("%Y%m%d-%H%M%S")
            recipe.update(recipe_extra or {})

    distill.log_generation(recipe=recipe, input_images=images or [], output_images=imgs)
    return imgs, recipe
