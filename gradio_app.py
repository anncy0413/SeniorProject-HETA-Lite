import html
import json
import os
import tempfile
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoTokenizer

from heta_demo import HETAAttributor

DEFAULT_MODEL_NAME = os.environ.get("HETA_MODEL", "Qwen/Qwen2.5-1.5B")
MAX_CHARS = 2000
MAX_TOKENS = 512
TOP_K = 8

EXAMPLE_PROMPTS = {
    "QA": "What causes the seasons on Earth?",
    "Long Context": (
        "In the early 1900s, scientists debated how the brain stores memory. "
        "Some believed memory was localized to specific regions, while others argued "
        "it was distributed across networks. Today, evidence shows memory involves "
        "both localized and distributed processes, depending on the type and time scale."
    ),
    "Reasoning": (
        "If a train travels 120 miles in 2 hours, and then 90 miles in 1.5 hours, "
        "what is the average speed across the entire trip?"
    ),
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');

:root {
  --bg-start: #f6f9ff;
  --bg-end: #eef3fb;
  --panel: #ffffff;
  --border: #e4ecf7;
  --text: #1f2937;
  --muted: #64748b;
  --accent: #2563eb;
  --accent-weak: #e4efff;
  --accent-strong: #1d4ed8;
  --warn: #dc2626;
}

body, .gradio-container {
  background: linear-gradient(180deg, var(--bg-start) 0%, var(--bg-end) 100%);
  color: var(--text);
  font-family: 'Manrope', 'Avenir Next', 'Segoe UI', sans-serif;
}

#heta-app {
  max-width: 1220px;
  margin: 0 auto;
  padding-bottom: 24px;
}

#heta-app h1 {
  font-size: 30px;
  font-weight: 700;
  margin-bottom: 6px;
}

#heta-app h3 {
  font-size: 16px;
  font-weight: 600;
  margin-top: 18px;
  color: var(--text);
}

#heta-app p {
  color: var(--muted);
}

#heta-app .main-row {
  gap: 18px;
}

.stack .panel {
  margin-bottom: 18px;
}

.stack .panel:last-of-type {
  margin-bottom: 0;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 14px 40px rgba(37, 99, 235, 0.08);
}

.token-preview, .heatmap {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.token-chip {
  padding: 4px 9px;
  border-radius: 999px;
  background: #f4f7ff;
  border: 1px solid #e7eefc;
  font-size: 13px;
  cursor: pointer;
  white-space: pre;
  transition: all 0.15s ease;
}

.token-chip:hover {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.14);
}

.token-selected {
  border-color: var(--accent-strong);
  background: #e9f1ff;
  box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
}

.token-target {
  border-color: #f59e0b;
  box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.18);
}

.token-highlight {
  border-color: var(--accent-strong);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
}

.heat-token {
  padding: 6px 10px;
  border-radius: 12px;
  border: 1px solid #d7e6ff;
  font-size: 13px;
  white-space: pre;
  cursor: pointer;
}

.topk-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 260px;
  overflow-y: auto;
  padding-right: 4px;
}

.topk-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  border-radius: 12px;
  border: 1px solid #e5edf7;
  background: #f7faff;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.15s ease;
}

.topk-item:hover {
  background: #eaf2ff;
  border-color: var(--accent);
}

.status-box {
  border: 1px dashed #c9dcf7;
  border-radius: 12px;
  padding: 12px 14px;
  background: #f8fbff;
  font-size: 13px;
  color: var(--muted);
}

.meta-box {
  border-radius: 12px;
  border: 1px solid #e5edf7;
  padding: 10px 12px;
  background: #f6faff;
  font-size: 13px;
  color: var(--muted);
}

.export-row {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.export-row .gr-file {
  min-height: 44px;
}

.selected-target {
  font-weight: 600;
  color: #0f172a;
}

.chip-row .gr-button {
  border-radius: 999px !important;
  padding: 6px 14px !important;
  background: #f1f5ff !important;
  border: 1px solid #e0eaff !important;
  color: var(--text) !important;
  font-size: 12px !important;
}

.chip-row .gr-button:hover {
  border-color: var(--accent) !important;
  color: var(--accent-strong) !important;
}

.gr-button-primary {
  background: linear-gradient(180deg, #2f6cf6 0%, #2563eb 100%) !important;
  border: none !important;
  box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25) !important;
}

.gr-button-secondary {
  background: #f4f7ff !important;
  border: 1px solid #e1e9f5 !important;
}

.gr-input, .gr-input textarea, .gr-input input {
  border-radius: 12px !important;
  border-color: #e2e8f0 !important;
}
"""

JS = """
function() {
  function setInputValue(elemId, value) {
    const root = document.getElementById(elemId);
    if (!root) return;
    const input = root.querySelector('input, textarea');
    if (!input) return;
    input.value = value;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }

  document.addEventListener('click', (event) => {
    const token = event.target.closest('[data-token-index]');
    if (!token) return;
    const idx = token.getAttribute('data-token-index');
    const role = token.getAttribute('data-role');
    if (role === 'preview') {
      setInputValue('target-index', idx);
    }
    if (role === 'topk' || role === 'heatmap') {
      setInputValue('highlight-index', idx);
    }
  });
}
"""


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=1)
def load_attributor(model_name: str = DEFAULT_MODEL_NAME) -> HETAAttributor:
    device = get_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return HETAAttributor(model, tokenizer, device)


def sanitize_token(token: str) -> str:
    safe = token.replace("\n", "\\n").replace("\t", "\\t")
    return html.escape(safe)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def lerp_color(start: Tuple[int, int, int], end: Tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    return rgb_to_hex(
        (
            int(start[0] + (end[0] - start[0]) * t),
            int(start[1] + (end[1] - start[1]) * t),
            int(start[2] + (end[2] - start[2]) * t),
        )
    )


def score_to_color(score: float, max_score: float) -> str:
    light = (234, 245, 255)
    deep = (47, 129, 247)
    if max_score <= 0:
        return rgb_to_hex(light)
    strength = (score / max_score) ** 0.6
    return lerp_color(light, deep, strength)


def render_token_preview(tokens: List[str], selected_index: Optional[int]) -> str:
    if not tokens:
        return "<div class='status-box'>No tokens yet. Enter a prompt to preview tokens.</div>"
    chips = []
    for idx, token in enumerate(tokens):
        classes = ["token-chip"]
        if selected_index is not None and idx == selected_index:
            classes.append("token-selected")
        token_html = sanitize_token(token)
        chips.append(
            "<span class='{}' data-role='preview' data-token-index='{}' title='Index {}'>".format(
                " ".join(classes), idx, idx
            )
            + token_html
            + "</span>"
        )
    return "<div class='token-preview'>{}</div>".format("".join(chips))


def render_heatmap(
    tokens: List[str],
    scores: List[float],
    target_index: Optional[int],
    highlight_index: Optional[int],
) -> str:
    if not tokens or not scores:
        return "<div class='status-box'>Run attribution to see the heatmap.</div>"

    max_score = max(scores) if scores else 0.0
    chips = []
    for idx, (token, score) in enumerate(zip(tokens, scores)):
        classes = ["heat-token"]
        if target_index is not None and idx == target_index:
            classes.append("token-target")
        if highlight_index is not None and idx == highlight_index:
            classes.append("token-highlight")
        token_html = sanitize_token(token)
        color = score_to_color(score, max_score)
        title = "Score: {:.6f} | Percent: {:.2f}%".format(score, score * 100.0)
        chips.append(
            "<span class='{}' data-role='heatmap' data-token-index='{}' title='{}' style='background:{};'>".format(
                " ".join(classes), idx, title, color
            )
            + token_html
            + "</span>"
        )
    return "<div class='heatmap'>{}</div>".format("".join(chips))


def render_topk(tokens: List[str], scores: List[float], target_index: Optional[int]) -> str:
    if not tokens or not scores:
        return "<div class='status-box'>Run attribution to see top contributors.</div>"
    if target_index is None:
        target_index = len(tokens)
    upper = max(0, min(target_index, len(tokens)))
    if upper == 0:
        return "<div class='status-box'>No influential tokens available for this target.</div>"
    indices = list(range(upper))
    indices.sort(key=lambda i: scores[i], reverse=True)
    top_indices = indices[: min(TOP_K, len(indices))]

    items = []
    for rank, idx in enumerate(top_indices, start=1):
        token_html = sanitize_token(tokens[idx])
        score = scores[idx] * 100.0
        items.append(
            "<div class='topk-item' data-role='topk' data-token-index='{}' title='Click to highlight in heatmap'>".format(
                idx
            )
            + "<span>#{:d} {} (index {})</span>".format(rank, token_html, idx)
            + "<span>{:.2f}%</span>".format(score)
            + "</div>"
        )
    return "<div class='topk-list'>{}</div>".format("".join(items))


def format_selected(tokens: List[str], index: Optional[int]) -> str:
    if not tokens:
        return "Selected target: (none)"
    if index is None or index < 0 or index >= len(tokens):
        return "Selected target: (invalid index)"
    token = tokens[index]
    token = token.replace("\n", "\\n").replace("\t", "\\t").replace("`", "\\`")
    return "Selected target: `{}` (index {})".format(token, index)


def format_status(state: str, detail: str = "") -> str:
    legend = "States: idle | running | queued (placeholder) | error"
    if detail:
        return "**Status:** {}\n\n{}\n\n_{}_.".format(state, detail, legend)
    return "**Status:** {}\n\n_{}_.".format(state, legend)


def prepare_prompt(prompt: str) -> Tuple:
    prompt = prompt or ""
    if not prompt.strip():
        status = format_status("error", "Prompt is empty. Enter a prompt to continue.")
        return (
            [],
            render_token_preview([], None),
            None,
            "Selected target: (none)",
            status,
            render_heatmap([], [], None, None),
            render_topk([], [], None),
            "",
            [],
            None,
            None,
            None,
        )
    if len(prompt) > MAX_CHARS:
        status = format_status(
            "error",
            "Prompt is too long. Limit is {} characters.".format(MAX_CHARS),
        )
        return (
            [],
            render_token_preview([], None),
            None,
            "Selected target: (none)",
            status,
            render_heatmap([], [], None, None),
            render_topk([], [], None),
            "",
            [],
            None,
            None,
            None,
        )

    tokenizer = load_tokenizer(DEFAULT_MODEL_NAME)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids[0].tolist()
    if len(input_ids) > MAX_TOKENS:
        status = format_status(
            "error",
            "Prompt token count is too high ({}). Limit is {} tokens.".format(
                len(input_ids), MAX_TOKENS
            ),
        )
        return (
            [],
            render_token_preview([], None),
            None,
            "Selected target: (none)",
            status,
            render_heatmap([], [], None, None),
            render_topk([], [], None),
            "",
            [],
            None,
            None,
            None,
        )

    tokens = [tokenizer.decode([tid]) for tid in input_ids]
    default_index = max(0, len(tokens) - 1)
    preview_html = render_token_preview(tokens, default_index)
    selected_text = format_selected(tokens, default_index)
    status = format_status("idle", "Token preview ready. Select a target token.")

    return (
        tokens,
        preview_html,
        default_index,
        selected_text,
        status,
        render_heatmap([], [], None, None),
        render_topk([], [], None),
        "",
        [],
        None,
        None,
        None,
    )


def sync_target_index(target_index: Any, tokens: List[str]) -> Tuple:
    if not tokens:
        return (
            target_index,
            render_token_preview([], None),
            "Selected target: (none)",
            format_status("error", "No tokens available. Preview tokens first."),
        )

    try:
        idx = int(target_index) if target_index is not None else len(tokens) - 1
    except (TypeError, ValueError):
        idx = len(tokens) - 1
        status = format_status(
            "error", "Target index must be an integer. Using last token."
        )
        return (
            idx,
            render_token_preview(tokens, idx),
            format_selected(tokens, idx),
            status,
        )

    status_detail = "Target token updated. Run attribution to refresh results."
    if idx < 0:
        idx = 0
        status_detail = "Target index adjusted to 0. Run attribution to refresh results."
    if idx >= len(tokens):
        idx = len(tokens) - 1
        status_detail = "Target index adjusted to {}. Run attribution to refresh results.".format(
            len(tokens) - 1
        )

    preview_html = render_token_preview(tokens, idx)
    selected_text = format_selected(tokens, idx)
    return (
        idx,
        preview_html,
        selected_text,
        format_status("idle", status_detail),
    )


def run_attribution(
    prompt: str,
    target_index: Any,
    quality: str,
) -> Tuple:
    prompt = prompt or ""
    if not prompt.strip():
        return (
            [],
            [],
            None,
            None,
            render_token_preview([], None),
            "Selected target: (none)",
            render_heatmap([], [], None, None),
            render_topk([], [], None),
            "",
            format_status("error", "Prompt is empty. Enter a prompt to continue."),
            None,
        )
    if len(prompt) > MAX_CHARS:
        return (
            [],
            [],
            None,
            None,
            render_token_preview([], None),
            "Selected target: (none)",
            render_heatmap([], [], None, None),
            render_topk([], [], None),
            "",
            format_status(
                "error",
                "Prompt is too long. Limit is {} characters.".format(MAX_CHARS),
            ),
            None,
        )

    tokenizer = load_tokenizer(DEFAULT_MODEL_NAME)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids[0].tolist()

    if len(input_ids) > MAX_TOKENS:
        return (
            [],
            [],
            None,
            None,
            render_token_preview([], None),
            "Selected target: (none)",
            render_heatmap([], [], None, None),
            render_topk([], [], None),
            "",
            format_status(
                "error",
                "Prompt token count is too high ({}). Limit is {} tokens.".format(
                    len(input_ids), MAX_TOKENS
                ),
            ),
            None,
        )

    tokens = [tokenizer.decode([tid]) for tid in input_ids]
    try:
        idx = int(target_index) if target_index is not None else len(tokens) - 1
    except (TypeError, ValueError):
        idx = len(tokens) - 1

    status_detail = ""
    if idx < 0 or idx >= len(tokens):
        return (
            tokens,
            [],
            None,
            None,
            render_token_preview(tokens, None),
            "Selected target: (invalid index)",
            render_heatmap([], [], None, None),
            render_topk([], [], None),
            "",
            format_status(
                "error",
                "Target index out of range. Choose 0 to {}.".format(len(tokens) - 1),
            ),
            None,
        )

    if idx == 0 and len(tokens) > 1:
        idx = 1
        status_detail = "Target index 0 adjusted to 1 due to backend constraints."

    start = time.perf_counter()
    try:
        attributor = load_attributor(DEFAULT_MODEL_NAME)
        out_tokens, scores, target_pos = attributor.attribute(prompt, idx)
    except Exception as exc:  # noqa: BLE001
        return (
            tokens,
            [],
            None,
            None,
            render_token_preview(tokens, idx),
            format_selected(tokens, idx),
            render_heatmap([], [], None, None),
            render_topk([], [], None),
            "",
            format_status("error", "Attribution failed: {}".format(exc)),
            None,
        )

    latency_ms = int((time.perf_counter() - start) * 1000)

    scores_list = [float(s) for s in scores]
    preview_html = render_token_preview(out_tokens, target_pos)
    selected_text = format_selected(out_tokens, target_pos)
    heatmap_html = render_heatmap(out_tokens, scores_list, target_pos, None)
    topk_html = render_topk(out_tokens, scores_list, target_pos)

    # Anncy: Comment quality selector will map to backend parameters such as sample counts.
    meta = "Model: {} | Quality: {} | Latency: {} ms".format(
        DEFAULT_MODEL_NAME, quality, latency_ms
    )

    export_payload = {
        "tokens": out_tokens,
        "attribution_scores": scores_list,
        "target_index": int(target_pos),
        "target_token": out_tokens[target_pos] if out_tokens else "",
        "model_name": DEFAULT_MODEL_NAME,
        "quality": quality,
        "latency_ms": latency_ms,
    }

    status_message = "Attribution complete."
    if status_detail:
        status_message += " " + status_detail

    return (
        out_tokens,
        scores_list,
        int(target_pos),
        None,
        preview_html,
        selected_text,
        heatmap_html,
        topk_html,
        meta,
        format_status("idle", status_message),
        export_payload,
    )


def update_heatmap_highlight(
    highlight_index: Any,
    tokens: List[str],
    scores: List[float],
    target_index: Optional[int],
) -> str:
    if not tokens or not scores:
        return render_heatmap([], [], None, None)
    try:
        idx = int(highlight_index) if highlight_index is not None else None
    except (TypeError, ValueError):
        idx = None
    if idx is not None and (idx < 0 or idx >= len(tokens)):
        idx = None
    return render_heatmap(tokens, scores, target_index, idx)


def create_json_file(export_payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not export_payload:
        return None
    handle, path = tempfile.mkstemp(suffix=".json", prefix="heta_export_")
    with os.fdopen(handle, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    return path


def render_heatmap_png(
    tokens: List[str],
    scores: List[float],
    target_index: Optional[int],
    highlight_index: Optional[int],
) -> Image.Image:
    font = ImageFont.load_default()
    pad_x, pad_y = 6, 4
    gap = 6
    max_width = 1100

    display_tokens = [tok.replace("\n", "\\n").replace("\t", "\\t") for tok in tokens]
    max_score = max(scores) if scores else 0.0

    rows: List[List[Tuple[str, float, int]]] = [[]]
    current_width = 0
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)

    for idx, (tok, score) in enumerate(zip(display_tokens, scores)):
        text_w, text_h = draw.textbbox((0, 0), tok, font=font)[2:]
        chip_w = text_w + pad_x * 2
        if current_width + chip_w > max_width and rows[-1]:
            rows.append([])
            current_width = 0
        rows[-1].append((tok, score, idx))
        current_width += chip_w + gap

    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + pad_y * 2
    img_height = len(rows) * (line_height + gap) + gap
    img_width = max_width + gap * 2
    image = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    y = gap
    for row in rows:
        x = gap
        for tok, score, idx in row:
            text_w, text_h = draw.textbbox((0, 0), tok, font=font)[2:]
            chip_w = text_w + pad_x * 2
            chip_h = line_height
            color = score_to_color(score, max_score)
            draw.rounded_rectangle(
                [x, y, x + chip_w, y + chip_h],
                radius=6,
                fill=color,
                outline="#cfe3ff",
                width=1,
            )
            if target_index is not None and idx == target_index:
                draw.rounded_rectangle(
                    [x, y, x + chip_w, y + chip_h],
                    radius=6,
                    outline="#e06b6b",
                    width=2,
                )
            if highlight_index is not None and idx == highlight_index:
                draw.rounded_rectangle(
                    [x - 2, y - 2, x + chip_w + 2, y + chip_h + 2],
                    radius=8,
                    outline="#1f5fbf",
                    width=2,
                )
            draw.text((x + pad_x, y + pad_y), tok, font=font, fill=(20, 30, 40))
            x += chip_w + gap
        y += line_height + gap

    return image


def export_png_file(
    tokens: List[str],
    scores: List[float],
    target_index: Optional[int],
    highlight_index: Any,
) -> Optional[str]:
    if not tokens or not scores:
        return None
    try:
        hi = int(highlight_index) if highlight_index is not None else None
    except (TypeError, ValueError):
        hi = None
    if hi is not None and (hi < 0 or hi >= len(tokens)):
        hi = None

    # Anncy: Comment PNG export of the HTML heatmap could use a canvas capture in the frontend.
    image = render_heatmap_png(tokens, scores, target_index, hi)
    handle, path = tempfile.mkstemp(suffix=".png", prefix="heta_heatmap_")
    os.close(handle)
    image.save(path)
    return path


def set_running() -> Tuple:
    # Anncy: Comment request queue position and progress would be shown here with backend support.
    return format_status("running", "Attribution running..."), gr.update(interactive=False)


def finalize_status(status_text: str) -> Tuple:
    return status_text, gr.update(interactive=True)


def build_demo() -> gr.Blocks:
    with gr.Blocks(css=CSS, js=JS, elem_id="heta-app") as demo:
        gr.Markdown(
            "# HETA Lite Demo\n"
            "Token-level attribution for decoder-only language models.\n"
            "Select a target token, run attribution, and explore the heatmap."
        )

        tokens_state = gr.State([])
        scores_state = gr.State([])
        export_state = gr.State(None)

        with gr.Row(elem_classes=["main-row"]):
            with gr.Column(scale=2, elem_classes=["stack"]):
                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Prompt")
                    prompt = gr.Textbox(
                        label="Enter a prompt",
                        lines=6,
                        placeholder="Paste a prompt or choose an example.",
                    )
                    gr.Markdown(
                        "Character limit: {} | Token limit: {}".format(
                            MAX_CHARS, MAX_TOKENS
                        )
                    )

                    with gr.Row(elem_classes=["chip-row"]):
                        example_buttons = []
                        for name in ["QA", "Long Context", "Reasoning"]:
                            btn = gr.Button(name, size="sm")
                            example_buttons.append((btn, EXAMPLE_PROMPTS[name]))

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Token Selection")
                    token_preview = gr.HTML(render_token_preview([], None))

                    with gr.Row():
                        target_index = gr.Number(
                            label="Target token index",
                            value=None,
                            precision=0,
                            interactive=True,
                            elem_id="target-index",
                        )

                    selected_display = gr.Markdown(
                        "Selected target: (none)", elem_classes=["selected-target"]
                    )

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Quality vs Latency")
                    quality = gr.Radio(
                        choices=["Fast", "Balanced", "Accurate"],
                        value="Balanced",
                        label="Quality setting",
                    )

                    with gr.Row():
                        preview_btn = gr.Button("Update Tokens", variant="secondary")
                        run_btn = gr.Button("Run Attribution", variant="primary")

            with gr.Column(scale=10, elem_classes=["stack"]):
                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Token Heatmap")
                    heatmap = gr.HTML(render_heatmap([], [], None, None))

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Top Influential Tokens")
                    topk = gr.HTML(render_topk([], [], None))

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Exports")
                    with gr.Row():
                        export_json_btn = gr.Button("Export JSON", variant="secondary")
                        export_png_btn = gr.Button("Export PNG", variant="secondary")

                    with gr.Row(elem_classes=["export-row"]):
                        export_json_file_output = gr.File(label="JSON export")
                        export_png_file_output = gr.File(label="PNG export")

                    gr.Markdown("### Run Metadata")
                    metadata = gr.Markdown("", elem_classes=["meta-box"])

                    gr.Markdown("### Request Status")
                    status = gr.Markdown(
                        format_status("idle", "Ready."), elem_classes=["status-box"]
                    )

        highlight_index = gr.Number(value=None, precision=0, elem_id="highlight-index", visible=False)

        # Preview tokens when prompt changes or button clicked
        preview_outputs = [
            tokens_state,
            token_preview,
            target_index,
            selected_display,
            status,
            heatmap,
            topk,
            metadata,
            scores_state,
            export_state,
            export_json_file_output,
            export_png_file_output,
        ]

        prompt.change(
            prepare_prompt,
            inputs=[prompt],
            outputs=preview_outputs,
        )
        preview_btn.click(
            prepare_prompt,
            inputs=[prompt],
            outputs=preview_outputs,
        )

        for btn, text in example_buttons:
            btn.click(
                lambda t=text: t,
                inputs=[],
                outputs=[prompt],
            ).then(
                prepare_prompt,
                inputs=[prompt],
                outputs=preview_outputs,
            )

        target_index.change(
            sync_target_index,
            inputs=[target_index, tokens_state],
            outputs=[target_index, token_preview, selected_display, status],
        )

        highlight_index.change(
            update_heatmap_highlight,
            inputs=[highlight_index, tokens_state, scores_state, target_index],
            outputs=[heatmap],
        )

        run_chain = run_btn.click(
            set_running,
            inputs=[],
            outputs=[status, run_btn],
        )

        run_chain = run_chain.then(
            run_attribution,
            inputs=[prompt, target_index, quality],
            outputs=[
                tokens_state,
                scores_state,
                target_index,
                highlight_index,
                token_preview,
                selected_display,
                heatmap,
                topk,
                metadata,
                status,
                export_state,
            ],
        )

        run_chain.then(
            finalize_status,
            inputs=[status],
            outputs=[status, run_btn],
        )

        export_json_btn.click(
            create_json_file,
            inputs=[export_state],
            outputs=[export_json_file_output],
        )
        export_png_btn.click(
            export_png_file,
            inputs=[tokens_state, scores_state, target_index, highlight_index],
            outputs=[export_png_file_output],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(share=True)


if __name__ == "__main__":
    main()
