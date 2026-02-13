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

MODEL_OPTIONS = {
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "GPT-J-6B": "EleutherAI/gpt-j-6B",
    "Phi-3-Medium-4K-Instruct (14B)": "microsoft/Phi-3-medium-4k-instruct",
    "Llama-3.1-70B": "meta-llama/Llama-3.1-70B-Instruct",
}
DEFAULT_MODEL_LABEL = os.environ.get("HETA_MODEL_LABEL") or os.environ.get(
    "HETA_MODEL", "Qwen2.5-3B"
)
if DEFAULT_MODEL_LABEL in MODEL_OPTIONS.values():
    DEFAULT_MODEL_LABEL = next(
        label for label, model_id in MODEL_OPTIONS.items() if model_id == DEFAULT_MODEL_LABEL
    )
if DEFAULT_MODEL_LABEL not in MODEL_OPTIONS:
    DEFAULT_MODEL_LABEL = "Qwen2.5-3B"
DEFAULT_MODEL_ID = MODEL_OPTIONS.get(DEFAULT_MODEL_LABEL, MODEL_OPTIONS["Qwen2.5-3B"])
MAX_CHARS = 2000
MAX_TOKENS = 512
TOP_K = 8
SEGMENT_SEPARATOR = " <s> "
MAX_ANSWER_TOKENS = 16

CURATED_EXAMPLE = {
    "narrative": (
        "The protagonist returns to the village after the winter storm, "
        "reflecting on her father's passing."
    ),
    "evidence": (
        "Photosynthesis primarily occurs in the leaves of the plant, "
        "where chloroplasts capture light."
    ),
    "question": "In which part of the plant does photosynthesis mainly take place?",
}
MASK_STRATEGIES = ["drop", "unk", "zero_embed"]

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

.heatmap {
  animation: fadeIn 0.25s ease;
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

.heat-token {
  transition: transform 0.12s ease, box-shadow 0.12s ease;
  border: 1px solid #f5d1d1;
}

.heat-token:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 14px rgba(185, 28, 28, 0.15);
}

.token-target {
  border-color: #b91c1c;
  box-shadow: 0 0 0 2px rgba(185, 28, 28, 0.22);
}

.token-highlight {
  border-color: #7f1d1d;
  box-shadow: 0 0 0 3px rgba(127, 29, 29, 0.2);
}

.segment-divider {
  border-left: 2px solid #fecaca;
  margin-left: 2px;
}

.answer-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.answer-chip {
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid #e7eefc;
  background: #f8fbff;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.12s ease;
}

.answer-chip:hover {
  border-color: #2563eb;
}

.answer-chip.selected {
  border-color: #b91c1c;
  background: #fee2e2;
  box-shadow: 0 0 0 2px rgba(185, 28, 28, 0.16);
}

.breakdown-tabs .tab-nav button {
  font-size: 12px;
}

.advanced-accordion {
  border: 1px solid #e4ecf7 !important;
  border-radius: 12px !important;
}

.token-pulse {
  animation: pulseRing 0.3s ease;
}

@keyframes pulseRing {
  0% { box-shadow: 0 0 0 0 rgba(185, 28, 28, 0.35); }
  100% { box-shadow: 0 0 0 10px rgba(185, 28, 28, 0); }
}

.skeleton-shimmer {
  height: 120px;
  border-radius: 12px;
  background: linear-gradient(90deg, #f8fafc 25%, #eef2ff 50%, #f8fafc 75%);
  background-size: 200% 100%;
  animation: shimmer 1.2s linear infinite;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
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

  function pulseToken(index) {
    const target = document.querySelector(`[data-heatmap-token-index="${index}"]`);
    if (!target) return;
    target.classList.remove('token-pulse');
    // Force reflow so repeated clicks re-trigger animation.
    void target.offsetWidth;
    target.classList.add('token-pulse');
    target.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
  }

  document.addEventListener('click', (event) => {
    const answerToken = event.target.closest('[data-answer-k]');
    if (answerToken) {
      const answerK = answerToken.getAttribute('data-answer-k');
      setInputValue('answer-token-k', answerK);
      return;
    }

    const token = event.target.closest('[data-token-index]');
    if (!token) return;
    const idx = token.getAttribute('data-token-index');
    const role = token.getAttribute('data-role');
    if (role === 'preview') {
      setInputValue('target-index', idx);
    }
    if (role === 'topk' || role === 'heatmap') {
      setInputValue('highlight-index', idx);
      pulseToken(idx);
    }
  });
}
"""


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model_id(model_choice: str) -> str:
    if model_choice in MODEL_OPTIONS:
        return MODEL_OPTIONS[model_choice]
    if model_choice in MODEL_OPTIONS.values():
        return model_choice
    return DEFAULT_MODEL_ID


def get_model_label(model_choice: str) -> str:
    if model_choice in MODEL_OPTIONS:
        return model_choice
    for label, model_id in MODEL_OPTIONS.items():
        if model_id == model_choice:
            return label
    return DEFAULT_MODEL_LABEL


@lru_cache(maxsize=8)
def load_tokenizer(model_name: str = DEFAULT_MODEL_ID) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=1)
def load_attributor(model_name: str = DEFAULT_MODEL_ID) -> HETAAttributor:
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
    light = (255, 247, 247)
    deep = (185, 28, 28)
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


def build_segmented_prompt(
    narrative: str, evidence: str, question: str, tokenizer: Optional[AutoTokenizer] = None
) -> Dict[str, Any]:
    if tokenizer is None:
        tokenizer = load_tokenizer(DEFAULT_MODEL_ID)

    narrative = (narrative or "").strip()
    evidence = (evidence or "").strip()
    question = (question or "").strip()
    segment_text = {
        "narrative": "[NarrativeQA] {}".format(narrative),
        "evidence": "[SciQ] {}".format(evidence),
        "question": "[Question] {}".format(question),
    }
    full_text = SEGMENT_SEPARATOR.join(
        [segment_text["narrative"], segment_text["evidence"], segment_text["question"]]
    )

    separator_ids = tokenizer(SEGMENT_SEPARATOR, add_special_tokens=False).input_ids
    narrative_ids = tokenizer(
        segment_text["narrative"], add_special_tokens=False
    ).input_ids
    evidence_ids = tokenizer(segment_text["evidence"], add_special_tokens=False).input_ids
    question_ids = tokenizer(segment_text["question"], add_special_tokens=False).input_ids

    narrative_start = 0
    narrative_end = len(narrative_ids)
    evidence_start = narrative_end + len(separator_ids)
    evidence_end = evidence_start + len(evidence_ids)
    question_start = evidence_end + len(separator_ids)
    question_end = question_start + len(question_ids)

    return {
        "full_text": full_text,
        "segments": {
            "narrative": (narrative_start, narrative_end),
            "evidence": (evidence_start, evidence_end),
            "question": (question_start, question_end),
        },
    }


def generate_answer_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_text: str,
    max_new_tokens: int = MAX_ANSWER_TOKENS,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids[0].tolist()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer_token_ids = generated[0, inputs.input_ids.shape[1] :].tolist()
    if not answer_token_ids:
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits[0, -1, :]
            answer_token_ids = [int(torch.argmax(logits).item())]
    answer_tokens = [
        tokenizer.decode([token_id], skip_special_tokens=False)
        for token_id in answer_token_ids
    ]
    answer_text = tokenizer.decode(answer_token_ids, skip_special_tokens=True)
    return {
        "prompt_token_ids": prompt_ids,
        "answer_token_ids": answer_token_ids,
        "answer_tokens": answer_tokens,
        "answer_text": answer_text.strip(),
    }


def compute_kl_information(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    target_pos: int,
    mask_strategy: str,
    return_metadata: bool = False,
) -> Any:
    seq_len = input_ids.shape[1]
    kl_scores = torch.zeros(seq_len, device=input_ids.device)
    if target_pos <= 0:
        if return_metadata:
            return kl_scores.cpu().numpy(), ""
        return kl_scores.cpu().numpy()

    pred_pos = target_pos - 1
    with torch.no_grad():
        orig_logits = model(input_ids, use_cache=False).logits[0, pred_pos, :]
        orig_probs = torch.softmax(orig_logits, dim=-1)

    replace_id = tokenizer.unk_token_id
    if replace_id is None:
        replace_id = tokenizer.mask_token_id
    if replace_id is None:
        replace_id = tokenizer.pad_token_id
    if replace_id is None:
        replace_id = tokenizer.eos_token_id
    if replace_id is None:
        replace_id = 0

    metadata_note = ""
    for pos in range(target_pos):
        try:
            if mask_strategy == "drop":
                masked_ids = torch.cat([input_ids[:, :pos], input_ids[:, pos + 1 :]], dim=1)
                masked_target_pos = target_pos - 1
                masked_pred_pos = masked_target_pos - 1
                if masked_pred_pos < 0:
                    continue
                with torch.no_grad():
                    masked_logits = model(masked_ids, use_cache=False).logits[
                        0, masked_pred_pos, :
                    ]
            elif mask_strategy == "zero_embed":
                with torch.no_grad():
                    embeds = model.get_input_embeddings()(input_ids).clone()
                    embeds[:, pos, :] = 0
                    masked_logits = model(inputs_embeds=embeds, use_cache=False).logits[
                        0, pred_pos, :
                    ]
            else:
                masked_ids = input_ids.clone()
                masked_ids[0, pos] = replace_id
                with torch.no_grad():
                    masked_logits = model(masked_ids, use_cache=False).logits[0, pred_pos, :]
        except Exception:
            # Anncy: Comment zero-embed fallback uses unk replacement when direct embedding masking fails.
            metadata_note = "zero_embed->unk fallback"
            masked_ids = input_ids.clone()
            masked_ids[0, pos] = replace_id
            with torch.no_grad():
                masked_logits = model(masked_ids, use_cache=False).logits[0, pred_pos, :]

        masked_probs = torch.softmax(masked_logits, dim=-1)
        kl = (
            orig_probs
            * (torch.log(orig_probs + 1e-10) - torch.log(masked_probs + 1e-10))
        ).sum()
        kl_scores[pos] = kl.clamp(min=0)

    kl_scores[target_pos:] = 0
    score_sum = kl_scores[:target_pos].sum()
    if score_sum > 1e-10:
        kl_scores[:target_pos] = kl_scores[:target_pos] / score_sum
    if return_metadata:
        return kl_scores.cpu().numpy(), metadata_note
    return kl_scores.cpu().numpy()


def combine_attr(
    mt_gate: np.ndarray,
    hessian_s: np.ndarray,
    kl_i: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    mt = np.clip(np.asarray(mt_gate, dtype=np.float64), 0.0, None)
    s = np.clip(np.asarray(hessian_s, dtype=np.float64), 0.0, None)
    i = np.clip(np.asarray(kl_i, dtype=np.float64), 0.0, None)
    final = mt * (beta * s + gamma * i)
    normalizer = final.sum()
    if normalizer > 1e-12:
        final = final / normalizer
    return final.astype(np.float64)


def compute_segment_mass(scores: np.ndarray, segment_ranges: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
    score_vec = np.asarray(scores, dtype=np.float64)
    n = len(score_vec)

    def range_mass(name: str) -> float:
        start, end = segment_ranges.get(name, (0, 0))
        lo = max(0, int(start))
        hi = min(n, int(end))
        if hi <= lo:
            return 0.0
        return float(score_vec[lo:hi].sum())

    narrative_mass = range_mass("narrative")
    evidence_mass = range_mass("evidence")
    question_mass = range_mass("question")
    return {
        "narrative": narrative_mass,
        "evidence": evidence_mass,
        "question": question_mass,
        "alignment": evidence_mass - narrative_mass,
    }


def render_answer_tokens(answer_tokens: List[str], answer_token_k: int) -> str:
    if not answer_tokens:
        return "<div class='status-box'>Run attribution to generate answer tokens.</div>"
    chips = []
    for idx, token in enumerate(answer_tokens[:MAX_ANSWER_TOKENS], start=1):
        classes = ["answer-chip"]
        if idx == answer_token_k:
            classes.append("selected")
        chips.append(
            "<span class='{}' data-answer-k='{}' title='Select generated token #{}'>".format(
                " ".join(classes), idx, idx
            )
            + sanitize_token(token)
            + "</span>"
        )
    return "<div class='answer-strip'>{}</div>".format("".join(chips))


def render_heatmap_strip(
    tokens: List[str],
    scores: np.ndarray,
    highlight_index: Optional[int],
    tooltips: Optional[Dict[int, Dict[str, float]]] = None,
    segment_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    target_index: Optional[int] = None,
) -> str:
    score_vec = np.asarray(scores, dtype=np.float64).tolist()
    if not tokens or not score_vec:
        return "<div class='status-box'>Run attribution to see the heatmap.</div>"

    if len(score_vec) < len(tokens):
        score_vec.extend([0.0] * (len(tokens) - len(score_vec)))
    score_vec = score_vec[: len(tokens)]
    max_score = max(score_vec) if score_vec else 0.0
    segment_starts = set()
    if segment_ranges:
        for start, _ in segment_ranges.values():
            if start > 0:
                segment_starts.add(int(start))

    chips = []
    for idx, (token, score) in enumerate(zip(tokens, score_vec)):
        classes = ["heat-token"]
        if idx in segment_starts:
            classes.append("segment-divider")
        if target_index is not None and idx == target_index:
            classes.append("token-target")
        if highlight_index is not None and idx == highlight_index:
            classes.append("token-highlight")
        tooltip_parts = []
        if tooltips and idx in tooltips:
            for key, value in tooltips[idx].items():
                tooltip_parts.append("{}: {:.6f}".format(key, float(value)))
        else:
            tooltip_parts.append("Final: {:.6f}".format(float(score)))
        tooltip_text = html.escape(" | ".join(tooltip_parts), quote=True)
        chips.append(
            "<span class='{}' data-role='heatmap' data-token-index='{}' data-heatmap-token-index='{}' title='{}' style='background:{};'>".format(
                " ".join(classes),
                idx,
                idx,
                tooltip_text,
                score_to_color(float(score), max_score),
            )
            + sanitize_token(token)
            + "</span>"
        )
    return "<div class='heatmap'>{}</div>".format("".join(chips))


def render_topk(tokens: List[str], scores: np.ndarray, k: int = 12) -> str:
    score_vec = np.asarray(scores, dtype=np.float64).tolist()
    if not tokens or not score_vec:
        return "<div class='status-box'>Run attribution to see top contributors.</div>"

    score_vec = score_vec[: len(tokens)]
    indices = sorted(range(len(score_vec)), key=lambda idx: score_vec[idx], reverse=True)
    top_indices = [idx for idx in indices if score_vec[idx] > 0][:k]
    if not top_indices:
        return "<div class='status-box'>No influential tokens available for this target.</div>"

    items = []
    for rank, idx in enumerate(top_indices, start=1):
        items.append(
            "<div class='topk-item' data-role='topk' data-token-index='{}' title='Click to highlight and scroll'>".format(
                idx
            )
            + "<span>#{:d} {} (idx {})</span>".format(rank, sanitize_token(tokens[idx]), idx)
            + "<span>{:.2f}%</span>".format(float(score_vec[idx]) * 100.0)
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


def format_segment_mass_markdown(mass: Dict[str, float]) -> str:
    return (
        "**Segment Mass**\n\n"
        "- Narrative: {:.4f}\n"
        "- SciQ evidence: {:.4f}\n"
        "- Question: {:.4f}\n"
        "- DSA-like alignment (SciQ - Narrative): {:.4f}"
    ).format(
        mass.get("narrative", 0.0),
        mass.get("evidence", 0.0),
        mass.get("question", 0.0),
        mass.get("alignment", 0.0),
    )


def build_common_tooltips(
    final_scores: np.ndarray, mt_scores: np.ndarray, s_scores: np.ndarray, kl_scores: np.ndarray
) -> Dict[int, Dict[str, float]]:
    tooltip_map: Dict[int, Dict[str, float]] = {}
    n = min(len(final_scores), len(mt_scores), len(s_scores), len(kl_scores))
    for idx in range(n):
        tooltip_map[idx] = {
            "Final": float(final_scores[idx]),
            "MT": float(mt_scores[idx]),
            "S": float(s_scores[idx]),
            "I": float(kl_scores[idx]),
        }
    return tooltip_map


def prepare_prompt(
    narrative: str,
    evidence: str,
    question: str,
    model_choice: str,
) -> Tuple:
    def preview_return(
        tokens: Optional[List[str]] = None,
        final_scores: Optional[List[float]] = None,
        mt_scores: Optional[List[float]] = None,
        s_scores: Optional[List[float]] = None,
        kl_scores: Optional[List[float]] = None,
        segment_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        full_prompt: str = "",
        target_index: Optional[int] = None,
        token_preview_html: Optional[str] = None,
        selected_text: str = "Selected target: (none)",
        answer_html: Optional[str] = None,
        answer_k: int = 1,
        final_html: Optional[str] = None,
        mt_html: Optional[str] = None,
        s_html: Optional[str] = None,
        kl_html: Optional[str] = None,
        topk_html: Optional[str] = None,
        segment_mass_html: Optional[str] = None,
        metadata_text: str = "",
        status_text: Optional[str] = None,
        export_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple:
        return (
            tokens or [],
            final_scores or [],
            mt_scores or [],
            s_scores or [],
            kl_scores or [],
            segment_ranges or {},
            full_prompt,
            target_index,
            token_preview_html if token_preview_html is not None else render_token_preview([], None),
            selected_text,
            answer_html if answer_html is not None else render_answer_tokens([], 1),
            answer_k,
            final_html if final_html is not None else render_heatmap_strip([], np.array([]), None),
            mt_html if mt_html is not None else render_heatmap_strip([], np.array([]), None),
            s_html if s_html is not None else render_heatmap_strip([], np.array([]), None),
            kl_html if kl_html is not None else render_heatmap_strip([], np.array([]), None),
            topk_html if topk_html is not None else render_topk([], np.array([])),
            segment_mass_html
            if segment_mass_html is not None
            else format_segment_mass_markdown({}),
            metadata_text,
            status_text if status_text is not None else format_status("idle", "Ready."),
            export_payload,
            None,
            None,
        )

    model_id = get_model_id(model_choice)
    model_label = get_model_label(model_choice)
    if not (narrative or "").strip() or not (evidence or "").strip() or not (question or "").strip():
        return preview_return(
            status_text=format_status("error", "Fill Narrative, SciQ evidence, and Question."),
        )

    if sum(len(part or "") for part in [narrative, evidence, question]) > MAX_CHARS:
        return preview_return(
            status_text=format_status(
                "error", "Input too long. Limit is {} characters.".format(MAX_CHARS)
            ),
        )

    try:
        tokenizer = load_tokenizer(model_id)
        prompt_bundle = build_segmented_prompt(narrative, evidence, question, tokenizer)
        prompt_ids = tokenizer(prompt_bundle["full_text"], add_special_tokens=False).input_ids
    except Exception as exc:  # noqa: BLE001
        return preview_return(
            status_text=format_status(
                "error", "Tokenizer load failed for {}: {}".format(model_label, exc)
            ),
        )

    if len(prompt_ids) > MAX_TOKENS:
        return preview_return(
            status_text=format_status(
                "error",
                "Prompt token count {} exceeds limit {}.".format(len(prompt_ids), MAX_TOKENS),
            ),
        )

    tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in prompt_ids]
    preview_html = render_token_preview(tokens, len(tokens) - 1 if tokens else None)
    return preview_return(
        tokens=tokens,
        segment_ranges=prompt_bundle["segments"],
        full_prompt=prompt_bundle["full_text"],
        target_index=len(tokens) - 1 if tokens else None,
        token_preview_html=preview_html,
        selected_text="Selected target: first answer token (k=1) before run",
        metadata_text="Model: {} | Prompt tokens: {}".format(model_label, len(tokens)),
        status_text=format_status("idle", "Segmented prompt ready."),
    )


def sync_target_index(target_index: Any, tokens: List[str]) -> Tuple:
    if not tokens:
        return (
            target_index,
            render_token_preview([], None),
            "Selected target: (none)",
            format_status("error", "No tokens available. Build prompt preview first."),
        )
    try:
        idx = int(target_index) if target_index is not None else len(tokens) - 1
    except (TypeError, ValueError):
        idx = len(tokens) - 1
    idx = max(0, min(idx, len(tokens) - 1))
    return (
        idx,
        render_token_preview(tokens, idx),
        format_selected(tokens, idx),
        format_status("idle", "Manual highlight index updated."),
    )


def run_attribution(
    narrative: str,
    evidence: str,
    question: str,
    answer_token_k: Any,
    quality: str,
    model_choice: str,
    mask_strategy: str,
    beta: float,
    gamma: float,
) -> Tuple:
    def run_return(
        tokens: Optional[List[str]] = None,
        final_scores: Optional[List[float]] = None,
        mt_scores: Optional[List[float]] = None,
        s_scores: Optional[List[float]] = None,
        kl_scores: Optional[List[float]] = None,
        segment_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        full_prompt: str = "",
        target_index: Optional[int] = None,
        token_preview_html: Optional[str] = None,
        selected_text: str = "Selected target: (none)",
        answer_html: Optional[str] = None,
        answer_k: int = 1,
        final_html: Optional[str] = None,
        mt_html: Optional[str] = None,
        s_html: Optional[str] = None,
        kl_html: Optional[str] = None,
        topk_html: Optional[str] = None,
        segment_mass_html: Optional[str] = None,
        metadata_text: str = "",
        status_text: Optional[str] = None,
        export_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple:
        return (
            tokens or [],
            final_scores or [],
            mt_scores or [],
            s_scores or [],
            kl_scores or [],
            segment_ranges or {},
            full_prompt,
            target_index,
            token_preview_html if token_preview_html is not None else render_token_preview([], None),
            selected_text,
            answer_html if answer_html is not None else render_answer_tokens([], 1),
            answer_k,
            final_html if final_html is not None else render_heatmap_strip([], np.array([]), None),
            mt_html if mt_html is not None else render_heatmap_strip([], np.array([]), None),
            s_html if s_html is not None else render_heatmap_strip([], np.array([]), None),
            kl_html if kl_html is not None else render_heatmap_strip([], np.array([]), None),
            topk_html if topk_html is not None else render_topk([], np.array([])),
            segment_mass_html
            if segment_mass_html is not None
            else format_segment_mass_markdown({}),
            metadata_text,
            status_text if status_text is not None else format_status("idle", "Ready."),
            export_payload,
            None,
            None,
        )

    model_id = get_model_id(model_choice)
    model_label = get_model_label(model_choice)
    start = time.perf_counter()

    try:
        tokenizer = load_tokenizer(model_id)
        attributor = load_attributor(model_id)
        prompt_bundle = build_segmented_prompt(narrative, evidence, question, tokenizer)
        prompt_ids = tokenizer(prompt_bundle["full_text"], add_special_tokens=False).input_ids
    except Exception as exc:  # noqa: BLE001
        return run_return(
            status_text=format_status("error", "Setup failed for {}: {}".format(model_label, exc)),
        )

    if len(prompt_ids) > MAX_TOKENS:
        return run_return(
            segment_ranges=prompt_bundle["segments"],
            full_prompt=prompt_bundle["full_text"],
            status_text=format_status(
                "error",
                "Prompt token count {} exceeds limit {}.".format(len(prompt_ids), MAX_TOKENS),
            ),
        )

    answer_bundle = generate_answer_tokens(attributor.model, tokenizer, prompt_bundle["full_text"])
    answer_ids = answer_bundle["answer_token_ids"]
    answer_tokens = answer_bundle["answer_tokens"]
    if not answer_ids:
        return run_return(
            segment_ranges=prompt_bundle["segments"],
            full_prompt=prompt_bundle["full_text"],
            status_text=format_status("error", "Generation produced no answer tokens."),
        )

    try:
        k = int(answer_token_k) if answer_token_k is not None else 1
    except (TypeError, ValueError):
        k = 1
    k = max(1, min(k, len(answer_ids)))

    full_ids = prompt_ids + answer_ids[:k]
    target_pos = len(full_ids) - 1
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=attributor.device)

    mt_gate = attributor.compute_attention_rollout(input_ids, target_pos).detach().cpu().numpy()
    # Anncy: Comment TODO replace gradient proxy with full Hessian HVP estimator used in paper experiments.
    hessian_s = (
        attributor.compute_gradient_sensitivity(input_ids, target_pos).detach().cpu().numpy()
    )
    kl_i, kl_note = compute_kl_information(
        attributor.model,
        tokenizer,
        input_ids,
        target_pos,
        mask_strategy,
        return_metadata=True,
    )
    final_scores = combine_attr(mt_gate, hessian_s, kl_i, float(beta), float(gamma))

    display_tokens = [
        tokenizer.decode([token_id], skip_special_tokens=False) for token_id in full_ids
    ]
    preview_html = render_token_preview(display_tokens, target_pos)
    selected_text = format_selected(display_tokens, target_pos)

    tooltip_map = build_common_tooltips(final_scores, mt_gate, hessian_s, kl_i)
    final_strip = render_heatmap_strip(
        display_tokens,
        final_scores,
        None,
        tooltips=tooltip_map,
        segment_ranges=prompt_bundle["segments"],
        target_index=target_pos,
    )
    mt_strip = render_heatmap_strip(
        display_tokens,
        mt_gate,
        None,
        tooltips={idx: {"MT": value} for idx, value in enumerate(mt_gate.tolist())},
        segment_ranges=prompt_bundle["segments"],
        target_index=target_pos,
    )
    s_strip = render_heatmap_strip(
        display_tokens,
        hessian_s,
        None,
        tooltips={idx: {"S": value} for idx, value in enumerate(hessian_s.tolist())},
        segment_ranges=prompt_bundle["segments"],
        target_index=target_pos,
    )
    i_strip = render_heatmap_strip(
        display_tokens,
        kl_i,
        None,
        tooltips={idx: {"I": value} for idx, value in enumerate(kl_i.tolist())},
        segment_ranges=prompt_bundle["segments"],
        target_index=target_pos,
    )

    topk_html = render_topk(display_tokens[:target_pos], final_scores[:target_pos], k=12)
    segment_mass = compute_segment_mass(final_scores, prompt_bundle["segments"])
    segment_mass_md = format_segment_mass_markdown(segment_mass)
    answer_tokens_html = render_answer_tokens(answer_tokens, k)

    latency_ms = int((time.perf_counter() - start) * 1000)
    # Anncy: Comment quality selector will map to backend runtime knobs (sampling budget / layers).
    metadata = "Model: {} | Quality: {} | Mask: {} | beta={:.2f} | gamma={:.2f} | Latency: {} ms".format(
        model_label, quality, mask_strategy, float(beta), float(gamma), latency_ms
    )
    if kl_note:
        metadata += " | KL note: {}".format(kl_note)

    export_payload = {
        "full_prompt": prompt_bundle["full_text"],
        "segments": {
            "narrative": list(prompt_bundle["segments"]["narrative"]),
            "evidence": list(prompt_bundle["segments"]["evidence"]),
            "question": list(prompt_bundle["segments"]["question"]),
        },
        "model": model_label,
        "quality": quality,
        "mask_strategy": mask_strategy,
        "beta": float(beta),
        "gamma": float(gamma),
        "answer": {
            "text": answer_bundle["answer_text"],
            "tokens": answer_tokens,
            "k": int(k),
            "target_token": answer_tokens[k - 1],
            "target_token_id": int(answer_ids[k - 1]),
        },
        "tokens": display_tokens,
        "scores": {
            "final": [float(value) for value in final_scores.tolist()],
            "mt": [float(value) for value in mt_gate.tolist()],
            "hessian_s": [float(value) for value in hessian_s.tolist()],
            "kl_i": [float(value) for value in kl_i.tolist()],
        },
        "segment_mass": segment_mass,
        "latency_ms": latency_ms,
    }

    status_detail = "Attribution complete for answer onset token k={}.".format(k)
    return run_return(
        tokens=display_tokens,
        final_scores=[float(value) for value in final_scores.tolist()],
        mt_scores=[float(value) for value in mt_gate.tolist()],
        s_scores=[float(value) for value in hessian_s.tolist()],
        kl_scores=[float(value) for value in kl_i.tolist()],
        segment_ranges=prompt_bundle["segments"],
        full_prompt=prompt_bundle["full_text"],
        target_index=int(target_pos),
        token_preview_html=preview_html,
        selected_text=selected_text,
        answer_html=answer_tokens_html,
        answer_k=int(k),
        final_html=final_strip,
        mt_html=mt_strip,
        s_html=s_strip,
        kl_html=i_strip,
        topk_html=topk_html,
        segment_mass_html=segment_mass_md,
        metadata_text=metadata,
        status_text=format_status("idle", status_detail),
        export_payload=export_payload,
    )


def update_heatmap_highlight(
    highlight_index: Any,
    tokens: List[str],
    final_scores: List[float],
    mt_scores: List[float],
    s_scores: List[float],
    kl_scores: List[float],
    target_index: Optional[int],
    segment_ranges: Dict[str, Tuple[int, int]],
) -> Tuple[str, str, str, str]:
    try:
        idx = int(highlight_index) if highlight_index is not None else None
    except (TypeError, ValueError):
        idx = None
    if idx is not None and (idx < 0 or idx >= len(tokens)):
        idx = None

    final_arr = np.asarray(final_scores, dtype=np.float64)
    mt_arr = np.asarray(mt_scores, dtype=np.float64)
    s_arr = np.asarray(s_scores, dtype=np.float64)
    kl_arr = np.asarray(kl_scores, dtype=np.float64)

    return (
        render_heatmap_strip(
            tokens,
            final_arr,
            idx,
            tooltips=build_common_tooltips(final_arr, mt_arr, s_arr, kl_arr),
            segment_ranges=segment_ranges,
            target_index=target_index,
        ),
        render_heatmap_strip(
            tokens,
            mt_arr,
            idx,
            tooltips={i: {"MT": value} for i, value in enumerate(mt_arr.tolist())},
            segment_ranges=segment_ranges,
            target_index=target_index,
        ),
        render_heatmap_strip(
            tokens,
            s_arr,
            idx,
            tooltips={i: {"S": value} for i, value in enumerate(s_arr.tolist())},
            segment_ranges=segment_ranges,
            target_index=target_index,
        ),
        render_heatmap_strip(
            tokens,
            kl_arr,
            idx,
            tooltips={i: {"I": value} for i, value in enumerate(kl_arr.tolist())},
            segment_ranges=segment_ranges,
            target_index=target_index,
        ),
    )


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
    skeleton = "<div class='skeleton-shimmer'></div>"
    return (
        format_status("running", "Attribution running..."),
        gr.update(interactive=False),
        skeleton,
        skeleton,
        skeleton,
        skeleton,
        skeleton,
    )


def finalize_status(status_text: str) -> Tuple:
    return status_text, gr.update(interactive=True)


def build_demo() -> gr.Blocks:
    with gr.Blocks(css=CSS, js=JS, elem_id="heta-app") as demo:
        gr.Markdown(
            "# HETA Lite Demo\n"
            "Paper framing: `[NarrativeQA] <s> [SciQ] <s> [Question]`.\n"
            "The demo generates the answer onset token first, then computes target-conditioned attribution."
        )

        tokens_state = gr.State([])
        final_scores_state = gr.State([])
        mt_scores_state = gr.State([])
        s_scores_state = gr.State([])
        kl_scores_state = gr.State([])
        segment_ranges_state = gr.State({})
        full_prompt_state = gr.State("")
        export_state = gr.State(None)

        with gr.Row(elem_classes=["main-row"]):
            with gr.Column(scale=2, elem_classes=["stack"]):
                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Input Builder")
                    model_choice = gr.Dropdown(
                        choices=list(MODEL_OPTIONS.keys()),
                        value=DEFAULT_MODEL_LABEL,
                        label="Select model",
                    )
                    narrative_box = gr.Textbox(
                        label="Segment A: Narrative",
                        lines=4,
                        placeholder="NarrativeQA distractor segment.",
                    )
                    evidence_box = gr.Textbox(
                        label="Segment B: SciQ evidence",
                        lines=4,
                        placeholder="SciQ evidence-bearing segment.",
                    )
                    question_box = gr.Textbox(
                        label="Question",
                        lines=3,
                        placeholder="Question used to trigger answer generation.",
                    )
                    fill_curated_example_btn = gr.Button(
                        "Fill curated example", variant="secondary"
                    )
                    gr.Markdown(
                        "Character limit: {} | Token limit: {}".format(MAX_CHARS, MAX_TOKENS)
                    )

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Token Selection")
                    token_preview = gr.HTML(render_token_preview([], None))

                    with gr.Row():
                        target_index = gr.Number(
                            label="Target token index (full sequence)",
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

                    answer_token_k = gr.Number(
                        label="Generated answer token k (1-indexed)",
                        value=1,
                        precision=0,
                        interactive=True,
                        elem_id="answer-token-k",
                    )

                    with gr.Accordion(
                        "Advanced Controls", open=False, elem_classes=["advanced-accordion"]
                    ):
                        mask_strategy = gr.Dropdown(
                            choices=MASK_STRATEGIES,
                            value="drop",
                            label="Masking strategy",
                        )
                        beta_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.5,
                            step=0.05,
                            label="beta",
                        )
                        gamma_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.5,
                            step=0.05,
                            label="gamma",
                        )
                        # Anncy: Comment queue/progress metadata would be rendered in this advanced panel with backend support.

            with gr.Column(scale=10, elem_classes=["stack"]):
                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Generated Answer")
                    answer_tokens_html = gr.HTML(render_answer_tokens([], 1))
                    segment_mass_md = gr.Markdown(format_segment_mass_markdown({}))

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Component Breakdown")
                    with gr.Tabs(elem_classes=["breakdown-tabs"]):
                        with gr.Tab("Final"):
                            final_heatmap = gr.HTML(
                                render_heatmap_strip([], np.array([]), None)
                            )
                        with gr.Tab("MT"):
                            mt_heatmap = gr.HTML(
                                render_heatmap_strip([], np.array([]), None)
                            )
                        with gr.Tab("Hessian S"):
                            s_heatmap = gr.HTML(
                                render_heatmap_strip([], np.array([]), None)
                            )
                        with gr.Tab("KL I"):
                            kl_heatmap = gr.HTML(
                                render_heatmap_strip([], np.array([]), None)
                            )

                with gr.Column(elem_classes=["panel"]):
                    gr.Markdown("### Top Influential Tokens")
                    topk = gr.HTML(render_topk([], np.array([])))

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

        highlight_index = gr.Number(
            value=None, precision=0, elem_id="highlight-index", visible=False
        )

        preview_outputs = [
            tokens_state,
            final_scores_state,
            mt_scores_state,
            s_scores_state,
            kl_scores_state,
            segment_ranges_state,
            full_prompt_state,
            target_index,
            token_preview,
            selected_display,
            answer_tokens_html,
            answer_token_k,
            final_heatmap,
            mt_heatmap,
            s_heatmap,
            kl_heatmap,
            topk,
            segment_mass_md,
            metadata,
            status,
            export_state,
            export_json_file_output,
            export_png_file_output,
        ]

        run_outputs = [
            tokens_state,
            final_scores_state,
            mt_scores_state,
            s_scores_state,
            kl_scores_state,
            segment_ranges_state,
            full_prompt_state,
            target_index,
            token_preview,
            selected_display,
            answer_tokens_html,
            answer_token_k,
            final_heatmap,
            mt_heatmap,
            s_heatmap,
            kl_heatmap,
            topk,
            segment_mass_md,
            metadata,
            status,
            export_state,
            export_json_file_output,
            export_png_file_output,
        ]

        fill_curated_example_btn.click(
            lambda: (
                CURATED_EXAMPLE["narrative"],
                CURATED_EXAMPLE["evidence"],
                CURATED_EXAMPLE["question"],
            ),
            inputs=[],
            outputs=[narrative_box, evidence_box, question_box],
        ).then(
            prepare_prompt,
            inputs=[narrative_box, evidence_box, question_box, model_choice],
            outputs=preview_outputs,
        )

        preview_btn.click(
            prepare_prompt,
            inputs=[narrative_box, evidence_box, question_box, model_choice],
            outputs=preview_outputs,
        )

        model_choice.change(
            prepare_prompt,
            inputs=[narrative_box, evidence_box, question_box, model_choice],
            outputs=preview_outputs,
        )

        target_index.change(
            sync_target_index,
            inputs=[target_index, tokens_state],
            outputs=[target_index, token_preview, selected_display, status],
        )

        highlight_index.change(
            update_heatmap_highlight,
            inputs=[
                highlight_index,
                tokens_state,
                final_scores_state,
                mt_scores_state,
                s_scores_state,
                kl_scores_state,
                target_index,
                segment_ranges_state,
            ],
            outputs=[final_heatmap, mt_heatmap, s_heatmap, kl_heatmap],
        )

        run_chain = run_btn.click(
            set_running,
            inputs=[],
            outputs=[status, run_btn, final_heatmap, mt_heatmap, s_heatmap, kl_heatmap, topk],
        )

        run_chain = run_chain.then(
            run_attribution,
            inputs=[
                narrative_box,
                evidence_box,
                question_box,
                answer_token_k,
                quality,
                model_choice,
                mask_strategy,
                beta_slider,
                gamma_slider,
            ],
            outputs=run_outputs,
        )

        run_chain.then(
            finalize_status,
            inputs=[status],
            outputs=[status, run_btn],
        )

        answer_run_chain = answer_token_k.change(
            set_running,
            inputs=[],
            outputs=[status, run_btn, final_heatmap, mt_heatmap, s_heatmap, kl_heatmap, topk],
        )

        answer_run_chain = answer_run_chain.then(
            run_attribution,
            inputs=[
                narrative_box,
                evidence_box,
                question_box,
                answer_token_k,
                quality,
                model_choice,
                mask_strategy,
                beta_slider,
                gamma_slider,
            ],
            outputs=run_outputs,
        )

        answer_run_chain.then(
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
            inputs=[tokens_state, final_scores_state, target_index, highlight_index],
            outputs=[export_png_file_output],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(share=True)


if __name__ == "__main__":
    main()
