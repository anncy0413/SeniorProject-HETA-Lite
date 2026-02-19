from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from heta_demo import HETAAttributor, MODEL_OPTIONS

SEGMENT_SEPARATOR = " <s> "
MAX_ANSWER_TOKENS = 16
DEFAULT_MODEL_LABEL = next(iter(MODEL_OPTIONS.keys()))
DEFAULT_MODEL_ID = MODEL_OPTIONS[DEFAULT_MODEL_LABEL]


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_compute_dtype(device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


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
def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@lru_cache(maxsize=1)
def load_attributor(model_name: str) -> HETAAttributor:
    device = get_device()
    dtype = get_compute_dtype(device)
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


def build_segmented_prompt(
    narrative: str, evidence: str, question: str, tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    narrative = (narrative or "").strip()
    evidence = (evidence or "").strip()
    question = (question or "").strip()
    prefix_text = {
        "narrative": "[NarrativeQA]",
        "evidence": "[SciQ]",
        "question": "[Question]",
    }
    segment_content = {
        "narrative": narrative,
        "evidence": evidence,
        "question": question,
    }
    segment_text = {
        key: (
            "{} {}".format(prefix_text[key], segment_content[key]).strip()
            if segment_content[key]
            else prefix_text[key]
        )
        for key in prefix_text
    }
    full_text = SEGMENT_SEPARATOR.join(
        [segment_text["narrative"], segment_text["evidence"], segment_text["question"]]
    )

    separator_ids = tokenizer(SEGMENT_SEPARATOR, add_special_tokens=False).input_ids
    narrative_ids = tokenizer(segment_text["narrative"], add_special_tokens=False).input_ids
    evidence_ids = tokenizer(segment_text["evidence"], add_special_tokens=False).input_ids
    question_ids = tokenizer(segment_text["question"], add_special_tokens=False).input_ids

    narrative_prefix_ids = tokenizer(
        "{} ".format(prefix_text["narrative"]), add_special_tokens=False
    ).input_ids
    evidence_prefix_ids = tokenizer(
        "{} ".format(prefix_text["evidence"]), add_special_tokens=False
    ).input_ids
    question_prefix_ids = tokenizer(
        "{} ".format(prefix_text["question"]), add_special_tokens=False
    ).input_ids

    narrative_content_ids = tokenizer(narrative, add_special_tokens=False).input_ids
    evidence_content_ids = tokenizer(evidence, add_special_tokens=False).input_ids
    question_content_ids = tokenizer(question, add_special_tokens=False).input_ids

    narrative_segment_start = 0
    narrative_segment_end = len(narrative_ids)
    evidence_segment_start = len(narrative_ids) + len(separator_ids)
    evidence_segment_end = evidence_segment_start + len(evidence_ids)
    question_segment_start = evidence_segment_start + len(evidence_ids) + len(separator_ids)
    question_segment_end = question_segment_start + len(question_ids)

    narrative_start = narrative_segment_start + len(narrative_prefix_ids)
    narrative_end = narrative_start + len(narrative_content_ids)
    evidence_start = evidence_segment_start + len(evidence_prefix_ids)
    evidence_end = evidence_start + len(evidence_content_ids)
    question_start = question_segment_start + len(question_prefix_ids)
    question_end = question_start + len(question_content_ids)

    narrative_start = min(narrative_start, narrative_segment_end)
    narrative_end = min(max(narrative_start, narrative_end), narrative_segment_end)
    evidence_start = min(evidence_start, evidence_segment_end)
    evidence_end = min(max(evidence_start, evidence_end), evidence_segment_end)
    question_start = min(question_start, question_segment_end)
    question_end = min(max(question_start, question_end), question_segment_end)

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
) -> np.ndarray:
    seq_len = input_ids.shape[1]
    kl_scores = torch.zeros(seq_len, device=input_ids.device)
    if target_pos <= 0:
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

    for pos in range(target_pos):
        if mask_strategy == "drop":
            masked_ids = torch.cat([input_ids[:, :pos], input_ids[:, pos + 1 :]], dim=1)
            masked_target_pos = target_pos - 1
            masked_pred_pos = masked_target_pos - 1
            if masked_pred_pos < 0:
                continue
            with torch.no_grad():
                masked_logits = model(masked_ids, use_cache=False).logits[0, masked_pred_pos, :]
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
    return kl_scores.cpu().numpy()


def compute_hessian_sensitivity_forward(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    target_token_id: int,
    epsilon: float = 0.05,
) -> np.ndarray:
    seq_len = input_ids.shape[1]
    scores = torch.zeros(seq_len, device=input_ids.device, dtype=torch.float32)
    if seq_len <= 1:
        return scores.cpu().numpy()

    pred_pos = seq_len - 1
    target_token = int(target_token_id)
    with torch.no_grad():
        base_embeds = model.get_input_embeddings()(input_ids)
        base_logits = model(inputs_embeds=base_embeds, use_cache=False).logits[0, pred_pos, :]
        base_log_prob = torch.log_softmax(base_logits, dim=-1)[target_token]

    plus_embeds = base_embeds.clone()
    minus_embeds = base_embeds.clone()
    eps = float(max(epsilon, 1e-4))
    for pos in range(seq_len):
        with torch.no_grad():
            base_vec = base_embeds[:, pos, :]
            norm = torch.norm(base_vec, dim=-1, keepdim=True).clamp(min=1e-6)
            delta = (eps * base_vec / norm).to(base_embeds.dtype)
            plus_embeds[:, pos, :] = base_vec + delta
            minus_embeds[:, pos, :] = base_vec - delta

            plus_logits = model(inputs_embeds=plus_embeds, use_cache=False).logits[0, pred_pos, :]
            minus_logits = model(inputs_embeds=minus_embeds, use_cache=False).logits[
                0, pred_pos, :
            ]
            lp_plus = torch.log_softmax(plus_logits, dim=-1)[target_token]
            lp_minus = torch.log_softmax(minus_logits, dim=-1)[target_token]
            second_order = torch.abs(lp_plus - 2 * base_log_prob + lp_minus) / (eps ** 2)
            scores[pos] = second_order.to(torch.float32)

            plus_embeds[:, pos, :] = base_vec
            minus_embeds[:, pos, :] = base_vec

    total = scores.sum()
    if total > 1e-10:
        scores = scores / total
    return scores.cpu().numpy()


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


def normalize_on_indices(scores: np.ndarray, keep_indices: List[int]) -> np.ndarray:
    score_vec = np.asarray(scores, dtype=np.float64).copy()
    if score_vec.size == 0:
        return score_vec
    mask = np.zeros(score_vec.shape[0], dtype=bool)
    for idx in keep_indices:
        if 0 <= int(idx) < score_vec.shape[0]:
            mask[int(idx)] = True
    score_vec[~mask] = 0.0
    total = score_vec.sum()
    if total > 1e-12:
        score_vec = score_vec / total
    return score_vec


def _build_content_indices(
    segment_ranges: Dict[str, Tuple[int, int]], prompt_len: int, context_len: int
) -> List[int]:
    indices: List[int] = []
    for start, end in segment_ranges.values():
        lo = max(0, int(start))
        hi = min(context_len, int(end))
        if hi > lo:
            indices.extend(range(lo, hi))
    if context_len > prompt_len:
        indices.extend(range(prompt_len, context_len))
    return sorted(set(indices))


def _segment_mass_abs(
    final_scores: np.ndarray, segment_ranges: Dict[str, Tuple[int, int]]
) -> Dict[str, float]:
    score_vec = np.abs(np.asarray(final_scores, dtype=np.float64))
    masses: Dict[str, float] = {}
    for name in ("narrative", "evidence", "question"):
        start, end = segment_ranges.get(name, (0, 0))
        lo = max(0, int(start))
        hi = min(score_vec.shape[0], int(end))
        masses[name] = float(score_vec[lo:hi].sum()) if hi > lo else 0.0
    denom = masses["narrative"] + masses["evidence"] + masses["question"]
    if denom > 1e-12:
        masses = {k: float(v / denom) for k, v in masses.items()}
    return masses


def run_one_example(
    model_name: str,
    narrative: str,
    evidence: str,
    question: str,
    target_k: int,
    beta: float,
    gamma: float,
    masking: str,
    quality: str,
) -> Dict[str, Any]:
    """
    Run one end-to-end attribution request using the backend flow from gradio_app.py.
    """
    start = time.perf_counter()
    model_id = get_model_id(model_name)
    model_label = get_model_label(model_name)

    tokenizer = load_tokenizer(model_id)
    attributor = load_attributor(model_id)
    prompt_bundle = build_segmented_prompt(narrative, evidence, question, tokenizer)
    full_text = prompt_bundle["full_text"]
    prompt_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    answer_bundle = generate_answer_tokens(attributor.model, tokenizer, full_text)
    answer_ids = answer_bundle["answer_token_ids"]
    answer_tokens = answer_bundle["answer_tokens"]
    if not answer_ids:
        raise RuntimeError("Generation produced no answer tokens.")

    k = int(target_k) if target_k is not None else 1
    k = max(1, min(k, len(answer_ids)))

    context_ids = prompt_ids + answer_ids[: k - 1]
    target_token_id = int(answer_ids[k - 1])
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=attributor.device)
    logical_target_pos = len(context_ids)

    mt_target_idx = max(0, input_ids.shape[1] - 1)
    mt_gate = (
        attributor.compute_attention_rollout(input_ids, mt_target_idx)
        .to(torch.float32)
        .detach()
        .cpu()
        .numpy()
    )
    hessian_s = compute_hessian_sensitivity_forward(attributor.model, input_ids, target_token_id)
    kl_i = compute_kl_information(
        attributor.model,
        tokenizer,
        input_ids,
        logical_target_pos,
        masking,
    )

    content_indices = _build_content_indices(
        prompt_bundle["segments"], prompt_len=len(prompt_ids), context_len=len(context_ids)
    )
    mt_gate = normalize_on_indices(mt_gate, content_indices)
    hessian_s = normalize_on_indices(hessian_s, content_indices)
    kl_i = normalize_on_indices(kl_i, content_indices)
    final_scores = combine_attr(mt_gate, hessian_s, kl_i, float(beta), float(gamma))
    final_scores = normalize_on_indices(final_scores, content_indices)

    tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in context_ids]
    segment_mass = _segment_mass_abs(final_scores, prompt_bundle["segments"])

    return {
        "prompt_full_text": full_text,
        "tokens": tokens,
        "answer_tokens": answer_tokens,
        "onset_token_text": answer_tokens[k - 1],
        "target_k": int(k),
        "segment_token_spans": {
            "narrative": list(prompt_bundle["segments"]["narrative"]),
            "evidence": list(prompt_bundle["segments"]["evidence"]),
            "question": list(prompt_bundle["segments"]["question"]),
        },
        "component_scores": {
            "MT": [float(x) for x in mt_gate.tolist()],
            "S": [float(x) for x in hessian_s.tolist()],
            "KL": [float(x) for x in kl_i.tolist()],
            "final": [float(x) for x in final_scores.tolist()],
        },
        "segment_mass": segment_mass,
        "run_meta": {
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "model_name": model_label,
            "beta": float(beta),
            "gamma": float(gamma),
            "masking": masking,
            "quality": quality,
        },
    }


__all__ = ["run_one_example", "MODEL_OPTIONS"]
