# Qwen-Image-2.1

The hosted model. One diffusers `QwenImage21Pipeline` (`Qwen/Qwen-Image-2.1`,
bf16) serves every mode — the mode is auto-detected from the `image=` argument:

| `image=` | Mode |
|---|---|
| `None` | Text-to-image |
| single image | Instruction edit |
| list (≤ 10) | Multi-reference edit/composition (first = edit target, rest = references) |

The stack is ~15B+ parameters total (≈7B DiT + ≈8B VL text encoder); on an
80 GB H100 it sits at ~38 GB resident and peaks ~66 GB during 2K inference,
which is why exactly one process may hold it. A **Turbo** variant exists
(8-step distilled; not deployed here).

## Key behaviors

- **No mask parameter.** The pipeline has no inpainting-mask input. Region
  editing works by **baking a visible colored annotation into the input
  image** (boundary lines around the region) and naming it in the prompt:
  *"In the red marked region, …"*. Multi-region editing uses distinct colors
  (red/green/blue/yellow in this app), one instruction per color. The helpers
  in `src/qwen2_1/masks.py` convert canvas strokes or programmatic masks into
  such annotated images.
- **No `guidance_scale`.** Guidance is `true_cfg_scale`, and it only takes
  effect when `true_cfg_scale > 1` **and** a negative prompt is provided — the
  pipeline warns and ignores either one alone. The app enforces the coupling.
- **Resolution.** Native 2K output requires **explicit `width`/`height`**;
  omitted, the pipeline falls back to a ~1024² total-area target. Edit modes
  take no width/height — output inherits the input image size. Supported 2K
  resolutions (all dimensions divisible by 16, so a 0.5× "1K" scale stays
  valid):

  | Aspect | 2K size |
  |---|---|
  | 1:1 | 2048 × 2048 |
  | 4:3 | 2400 × 1792 |
  | 3:4 | 1792 × 2400 |
  | 3:2 | 2528 × 1696 |
  | 2:3 | 1696 × 2528 |
  | 16:9 | 2752 × 1536 |
  | 9:16 | 1536 × 2752 |

- **RGBA-native VAE.** The model can generate genuinely transparent images.
  There is no API flag — it's prompt-triggered; the app wraps the prompt:

  ```
  This is an RGBA image with transparency. {prompt}. The image has alpha
  channel and the background is transparent.
  ```

## Prompt-enhancement (PE) companion models

Qwen ships two official rewriter models — `Qwen/Qwen-Image-2.1-PE-T2I` and
`…-PE-I2I`, both Qwen3.5-VL 9B — that expand a terse user prompt into the
detailed, generation-optimized phrasing the image model was trained on. Here
they run as vLLM servers on GPU 1 and are invoked as agents before every
generation (toggleable under Advanced; see
[architecture.md](architecture.md)). They reason first, then emit a JSON
object:

```json
{"rewritten_prompt": "…", "wh_ratio": "4:3", "ratio_follow": "yes"}
```

(`rewrited_prompt` is a known typo in their training data — the parser accepts
both.) `wh_ratio` is the model's suggested aspect ratio, which the app's
"Auto" aspect maps to the nearest supported resolution above; `ratio_follow`
(I2I only) says whether the edit should keep the input's ratio. The I2I
rewriter sees the input images (downscaled to ≤1024²) before the text. If the
servers are down, the raw prompt is used unchanged.

## Prompting tips

- **Region edits: name the annotation.** The model only honors a marked region
  if the prompt references it explicitly — *"In the red marked region, replace
  the chair with a potted plant."* The annotation is a visible colored outline
  in the image; the model reads it like any other visual content.
- **Multi-region: one color per instruction.** Mark each region in a distinct
  color and chain the clauses: *"In the red marked region, X. In the green
  marked region, Y."* (The app's Multi-region tab composes this for you from
  per-color text fields.)
- **Transparency is a prompt feature.** For cutouts/logos/sprites, use the
  RGBA wrapper sentence above (the app's "Transparent background" checkbox).
  Works in T2I; the VAE decodes a real alpha channel.
- **CFG and negative prompts come as a pair.** Setting a negative prompt does
  nothing at `true_cfg_scale = 1.0`, and raising the scale does nothing
  without a negative prompt. Typical use: scale 3–4 plus a short negative
  ("blurry, low quality, extra fingers").
- **Let the enhancer do the elaboration.** Short intent-level prompts work
  well with enhancement on — the PE model adds composition, lighting and
  style detail per the official protocol. Turn it off (Advanced) when you
  want verbatim control over the prompt; the recipe records both
  `original_prompt` and the final `prompt` either way.
- **References: order matters.** In multi-image mode the first image is the
  one being edited; subsequent images are sources ("the person from the second
  image", "the logo from the third image").
