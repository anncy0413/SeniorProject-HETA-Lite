from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import signal
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from heta_batch_runner import MODEL_OPTIONS, run_one_example


STOP_REQUESTED = False
EPS = 1e-8


def compute_alignment_ratio(evidence_mass: float, narrative_mass: float) -> float:
    return float(evidence_mass / (narrative_mass + EPS))


def compute_alignment_logratio(evidence_mass: float, narrative_mass: float) -> float:
    return float(math.log((evidence_mass + EPS) / (narrative_mass + EPS)))


def compute_dynamic_delta(evidence_token_count: int, coeff: float, floor: float) -> float:
    count = max(1, int(evidence_token_count))
    return float(max(float(floor), float(coeff) / math.sqrt(count)))


def _request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[signal] received {signum}; will stop after current example.", flush=True)


def install_signal_handlers() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _request_stop)
        except Exception:
            pass


def write_checkpoint(
    checkpoint_path: Path,
    input_jsonl: Path,
    output_dir: Path,
    processed_total: int,
    processed_new: int,
    success_total: int,
    alignment_sum_total: float,
    last_id: str,
) -> None:
    payload = {
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "processed_total": int(processed_total),
        "processed_new_this_run": int(processed_new),
        "success_total": int(success_total),
        "alignment_sum_total": float(alignment_sum_total),
        "last_id": last_id,
    }
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HETA faithfulness batch test on HotpotQA.")
    parser.add_argument("--input_jsonl", required=True, help="Path to converted HETA-style JSONL.")
    parser.add_argument("--output_dir", required=True, help="Directory for JSONL/CSV/summary outputs.")
    parser.add_argument(
        "--model",
        default="",
        help="Model label or model id. If omitted, project default model is used.",
    )
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument(
        "--masking",
        choices=["drop", "unk", "zero_embed"],
        default="zero_embed",
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "accurate"],
        default="balanced",
    )
    parser.add_argument("--target_k", type=int, default=1)
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument(
        "--hvp_samples",
        type=int,
        default=1,
        help="Hutchinson samples for Hessian term S (lower = less VRAM).",
    )
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=0,
        help="Max context tokens for attribution. 0 = auto by quality (fast=256, balanced=384, accurate=512).",
    )
    parser.add_argument(
        "--important_top_fraction",
        type=float,
        default=0.2,
        help="Top fraction of paragraph tokens used as model-selected important tokens for DSA-style mass.",
    )
    parser.add_argument(
        "--fusion_mode",
        choices=["paper", "paper_floor", "log"],
        default="log",
        help="Final fusion mode for combining MT/S/KL.",
    )
    parser.add_argument(
        "--mt_floor",
        type=float,
        default=0.05,
        help="Lower bound for MT in paper_floor/log fusion to avoid hard zeroing.",
    )
    parser.add_argument(
        "--margin_delta",
        type=float,
        default=0.05,
        help="Relaxed success margin delta: evidence >= narrative + delta.",
    )
    parser.add_argument(
        "--ratio_threshold",
        type=float,
        default=1.2,
        help="Relaxed success ratio threshold: evidence/(narrative+eps) >= ratio.",
    )
    parser.add_argument(
        "--valid_evidence_min_tokens",
        type=int,
        default=20,
        help="Minimum evidence token count for valid_by_token_count.",
    )
    parser.add_argument(
        "--dynamic_delta_coeff",
        type=float,
        default=0.1,
        help="Adaptive delta coefficient in coeff/sqrt(evidence_token_count).",
    )
    parser.add_argument(
        "--dynamic_delta_floor",
        type=float,
        default=0.02,
        help="Adaptive delta floor lower bound.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] Skip bad JSON at line {line_no}: {exc}")


def load_completed_ids(results_jsonl: Path) -> Tuple[Set[str], int, int, float]:
    ids: Set[str] = set()
    n = 0
    n_success = 0
    sum_alignment = 0.0
    if not results_jsonl.exists():
        return ids, n, n_success, sum_alignment
    for rec in iter_jsonl(results_jsonl):
        rec_id = str(rec.get("id", ""))
        if rec_id:
            ids.add(rec_id)
        has_success = ("success_strict" in rec) or ("success" in rec)
        if has_success and "alignment" in rec:
            n += 1
            success_flag = bool(rec.get("success_strict", rec.get("success", False)))
            if success_flag:
                n_success += 1
            sum_alignment += float(rec.get("alignment", 0.0))
    return ids, n, n_success, sum_alignment


def build_top_tokens(tokens: List[str], final_scores: List[float], top_n: int = 8) -> List[Dict[str, Any]]:
    if not tokens or not final_scores:
        return []
    arr = np.asarray(final_scores, dtype=np.float64)
    if arr.size == 0:
        return []
    top_idx = np.argsort(arr)[::-1][:top_n]
    out: List[Dict[str, Any]] = []
    for idx in top_idx.tolist():
        score = float(arr[idx])
        if score <= 0:
            continue
        out.append({"index": int(idx), "token": tokens[idx], "score": score})
    return out


def build_segment_token_counts(segment_token_spans: Dict[str, List[int] | Tuple[int, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k in ("narrative", "evidence", "question"):
        span = segment_token_spans.get(k, (0, 0))
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            start, end = 0, 0
        else:
            start, end = int(span[0]), int(span[1])
        out[k] = max(0, end - start)
    return out


def segment_mass_from_scores(
    scores: List[float], segment_spans: Dict[str, List[int] | Tuple[int, int]]
) -> Dict[str, float]:
    arr = np.abs(np.asarray(scores, dtype=np.float64))
    masses: Dict[str, float] = {}
    for name in ("narrative", "evidence", "question"):
        span = segment_spans.get(name, (0, 0))
        start, end = int(span[0]), int(span[1])
        lo = max(0, start)
        hi = min(arr.shape[0], end)
        masses[name] = float(arr[lo:hi].sum()) if hi > lo else 0.0
    denom = masses["narrative"] + masses["evidence"] + masses["question"]
    if denom > 1e-12:
        masses = {k: float(v / denom) for k, v in masses.items()}
    return masses


def select_important_indices(
    scores: List[float],
    segment_spans: Dict[str, List[int] | Tuple[int, int]],
    top_fraction: float,
) -> List[int]:
    arr = np.abs(np.asarray(scores, dtype=np.float64))
    if arr.size == 0:
        return []
    candidates: List[int] = []
    for name in ("narrative", "evidence"):
        span = segment_spans.get(name, (0, 0))
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            continue
        start, end = int(span[0]), int(span[1])
        lo = max(0, start)
        hi = min(arr.shape[0], end)
        if hi > lo:
            candidates.extend(range(lo, hi))
    if not candidates:
        return []
    frac = min(1.0, max(0.01, float(top_fraction)))
    k = max(1, int(round(len(candidates) * frac)))
    cand_arr = np.asarray(candidates, dtype=np.int64)
    cand_scores = arr[cand_arr]
    order = np.argsort(cand_scores)[::-1][:k]
    selected = cand_arr[order].tolist()
    selected.sort()
    return selected


def segment_mass_on_selected(
    scores: List[float],
    segment_spans: Dict[str, List[int] | Tuple[int, int]],
    selected_indices: List[int],
) -> Dict[str, float]:
    arr = np.abs(np.asarray(scores, dtype=np.float64))
    sel = set(int(i) for i in selected_indices if 0 <= int(i) < arr.shape[0])
    masses: Dict[str, float] = {}
    for name in ("narrative", "evidence", "question"):
        span = segment_spans.get(name, (0, 0))
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            masses[name] = 0.0
            continue
        start, end = int(span[0]), int(span[1])
        lo = max(0, start)
        hi = min(arr.shape[0], end)
        if hi <= lo:
            masses[name] = 0.0
            continue
        masses[name] = float(sum(arr[idx] for idx in range(lo, hi) if idx in sel))
    denom = masses["narrative"] + masses["evidence"] + masses["question"]
    if denom > 1e-12:
        masses = {k: float(v / denom) for k, v in masses.items()}
    return masses


def write_summary_and_csv(
    results_jsonl: Path,
    summary_path: Path,
    csv_path: Path,
    margin_delta: float,
    ratio_threshold: float,
    valid_evidence_min_tokens: int,
    dynamic_delta_coeff: float,
    dynamic_delta_floor: float,
    important_top_fraction: float,
    fusion_mode: str,
    mt_floor: float,
) -> None:
    alignments: List[float] = []
    logratios: List[float] = []
    ratios: List[float] = []
    n = 0
    n_success_strict = 0
    n_success_margin = 0
    n_success_ratio = 0
    n_success_margin_dynamic = 0
    n_success_margin_005 = 0
    n_success_ratio_12 = 0
    n_valid = 0
    n_valid_success_strict = 0
    sum_alignment = 0.0
    sum_ev_share = 0.0
    sum_na_share = 0.0
    sum_ev = 0.0
    sum_na = 0.0
    sum_q = 0.0

    with csv_path.open("w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(
            csv_f,
            fieldnames=[
                "id",
                "success",
                "success_strict",
                "success_margin",
                "success_ratio",
                "success_margin_dynamic",
                "success_margin_0.05",
                "success_ratio_1.2",
                "alignment",
                "alignment_raw",
                "alignment_logratio",
                "alignment_ratio",
                "evidence_mass",
                "narrative_mass",
                "question_mass",
                "evidence_share_en",
                "narrative_share_en",
                "valid_by_token_count",
                "evidence_token_count",
                "dynamic_margin_delta",
                "margin_delta",
                "ratio_threshold",
                "latency_ms",
                "onset_token_text",
            ],
        )
        writer.writeheader()

        for rec in iter_jsonl(results_jsonl):
            segment_mass = rec.get("segment_mass", {}) or {}
            ev_raw = float(segment_mass.get("evidence", 0.0))
            na_raw = float(segment_mass.get("narrative", 0.0))
            q_raw = float(segment_mass.get("question", 0.0))
            en_denom = ev_raw + na_raw + EPS
            ev = float(rec.get("evidence_share_en", ev_raw / en_denom))
            na = float(rec.get("narrative_share_en", na_raw / en_denom))
            q = float(q_raw)

            alignment = float(rec.get("alignment", ev - na))
            alignment_ratio = float(rec.get("alignment_ratio", compute_alignment_ratio(ev, na)))
            alignment_logratio = float(
                rec.get("alignment_logratio", compute_alignment_logratio(ev, na))
            )

            counts = rec.get("segment_token_counts", {}) or {}
            if not counts:
                counts = build_segment_token_counts(rec.get("segment_token_spans", {}) or {})
            evidence_token_count = int(counts.get("evidence", 0))
            dynamic_margin_delta = float(
                rec.get(
                    "dynamic_margin_delta",
                    compute_dynamic_delta(
                        evidence_token_count,
                        coeff=dynamic_delta_coeff,
                        floor=dynamic_delta_floor,
                    ),
                )
            )
            valid_by_token_count = bool(
                rec.get(
                    "valid_by_token_count",
                    evidence_token_count >= int(valid_evidence_min_tokens),
                )
            )

            success_strict = bool(rec.get("success_strict", rec.get("success", ev > na)))
            success_margin = bool(rec.get("success_margin", ev >= na + float(margin_delta)))
            success_ratio = bool(rec.get("success_ratio", alignment_ratio >= float(ratio_threshold)))
            success_margin_dynamic = bool(
                rec.get("success_margin_dynamic", ev >= na + dynamic_margin_delta)
            )
            success_margin_005 = bool(rec.get("success_margin_0.05", ev >= na + 0.05))
            success_ratio_12 = bool(rec.get("success_ratio_1.2", alignment_ratio >= 1.2))

            writer.writerow(
                {
                    "id": rec.get("id", ""),
                    "success": int(success_strict),
                    "success_strict": int(success_strict),
                    "success_margin": int(success_margin),
                    "success_ratio": int(success_ratio),
                    "success_margin_dynamic": int(success_margin_dynamic),
                    "success_margin_0.05": int(success_margin_005),
                    "success_ratio_1.2": int(success_ratio_12),
                    "alignment": alignment,
                    "alignment_raw": float(rec.get("alignment_raw", ev_raw - na_raw)),
                    "alignment_logratio": alignment_logratio,
                    "alignment_ratio": alignment_ratio,
                    "evidence_mass": ev_raw,
                    "narrative_mass": na_raw,
                    "question_mass": q_raw,
                    "evidence_share_en": ev,
                    "narrative_share_en": na,
                    "valid_by_token_count": int(valid_by_token_count),
                    "evidence_token_count": evidence_token_count,
                    "dynamic_margin_delta": dynamic_margin_delta,
                    "margin_delta": float(margin_delta),
                    "ratio_threshold": float(ratio_threshold),
                    "latency_ms": rec.get("latency_ms", ""),
                    "onset_token_text": rec.get("onset_token_text", ""),
                }
            )

            n += 1
            n_success_strict += int(success_strict)
            n_success_margin += int(success_margin)
            n_success_ratio += int(success_ratio)
            n_success_margin_dynamic += int(success_margin_dynamic)
            n_success_margin_005 += int(success_margin_005)
            n_success_ratio_12 += int(success_ratio_12)
            n_valid += int(valid_by_token_count)
            if valid_by_token_count:
                n_valid_success_strict += int(success_strict)
            sum_alignment += alignment
            sum_ev_share += ev
            sum_na_share += na
            sum_ev += ev_raw
            sum_na += na_raw
            sum_q += q_raw
            alignments.append(alignment)
            logratios.append(alignment_logratio)
            ratios.append(alignment_ratio)

    if n == 0:
        summary = {
            "n_examples": 0,
            "success_rate": 0.0,
            "success_rate_strict": 0.0,
            "success_rate_margin": 0.0,
            "success_rate_ratio": 0.0,
            "success_rate_margin_dynamic": 0.0,
            "success_rate_margin_0.05": 0.0,
            "success_rate_ratio_1.2": 0.0,
            "valid_rate": 0.0,
            "success_rate_strict_valid": 0.0,
            "mean_alignment": 0.0,
            "median_alignment": 0.0,
            "mean_alignment_logratio": 0.0,
            "median_alignment_logratio": 0.0,
            "mean_alignment_ratio": 0.0,
            "median_alignment_ratio": 0.0,
            "mean_evidence_mass": 0.0,
            "mean_narrative_mass": 0.0,
            "mean_question_mass": 0.0,
            "mean_evidence_share_en": 0.0,
            "mean_narrative_share_en": 0.0,
            "alignment_p10": 0.0,
            "alignment_p50": 0.0,
            "alignment_p90": 0.0,
            "alignment_percentiles": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
            "metric_config": {
                "margin_delta": float(margin_delta),
                "ratio_threshold": float(ratio_threshold),
                "valid_evidence_min_tokens": int(valid_evidence_min_tokens),
                "dynamic_delta_coeff": float(dynamic_delta_coeff),
                "dynamic_delta_floor": float(dynamic_delta_floor),
                "important_top_fraction": float(important_top_fraction),
                "fusion_mode": str(fusion_mode),
                "mt_floor": float(mt_floor),
            },
        }
    else:
        p10, p50, p90 = np.percentile(np.asarray(alignments, dtype=np.float64), [10, 50, 90]).tolist()
        summary = {
            "n_examples": n,
            "success_rate": n_success_strict / n,
            "success_rate_strict": n_success_strict / n,
            "success_rate_margin": n_success_margin / n,
            "success_rate_ratio": n_success_ratio / n,
            "success_rate_margin_dynamic": n_success_margin_dynamic / n,
            "success_rate_margin_0.05": n_success_margin_005 / n,
            "success_rate_ratio_1.2": n_success_ratio_12 / n,
            "valid_rate": n_valid / n,
            "success_rate_strict_valid": (n_valid_success_strict / n_valid) if n_valid > 0 else 0.0,
            "mean_alignment": sum_alignment / n,
            "median_alignment": statistics.median(alignments),
            "mean_alignment_logratio": (sum(logratios) / n),
            "median_alignment_logratio": statistics.median(logratios),
            "mean_alignment_ratio": (sum(ratios) / n),
            "median_alignment_ratio": statistics.median(ratios),
            "mean_evidence_mass": sum_ev / n,
            "mean_narrative_mass": sum_na / n,
            "mean_question_mass": sum_q / n,
            "mean_evidence_share_en": sum_ev_share / n,
            "mean_narrative_share_en": sum_na_share / n,
            "alignment_p10": p10,
            "alignment_p50": p50,
            "alignment_p90": p90,
            "alignment_percentiles": {"p10": p10, "p50": p50, "p90": p90},
            "metric_config": {
                "margin_delta": float(margin_delta),
                "ratio_threshold": float(ratio_threshold),
                "valid_evidence_min_tokens": int(valid_evidence_min_tokens),
                "dynamic_delta_coeff": float(dynamic_delta_coeff),
                "dynamic_delta_floor": float(dynamic_delta_floor),
                "important_top_fraction": float(important_top_fraction),
                "fusion_mode": str(fusion_mode),
                "mt_floor": float(mt_floor),
            },
        }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    global STOP_REQUESTED
    args = parse_args()
    install_signal_handlers()
    if args.num_workers != 1:
        print("[warn] Only num_workers=1 is currently supported; forcing serial execution.")
    if args.model and args.model not in MODEL_OPTIONS and args.model not in MODEL_OPTIONS.values():
        print("[warn] Model is not in known dropdown options; trying it as raw model id.")

    set_seed(args.seed)
    quality_context_budget = {"fast": 256, "balanced": 384, "accurate": 512}
    effective_context_tokens = (
        int(args.max_context_tokens)
        if int(args.max_context_tokens) > 0
        else int(quality_context_budget[args.quality])
    )
    effective_hvp_samples = max(1, int(args.hvp_samples))
    print(
        "[info] runtime controls: max_context_tokens={} hvp_samples={} important_top_fraction={} fusion_mode={} mt_floor={}".format(
            effective_context_tokens,
            effective_hvp_samples,
            float(args.important_top_fraction),
            args.fusion_mode,
            float(args.mt_floor),
        ),
        flush=True,
    )

    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    results_jsonl = output_dir / "results.jsonl"
    summary_json = output_dir / "summary.json"
    results_csv = output_dir / "results.csv"
    checkpoint_json = output_dir / "checkpoint.json"

    completed_ids, base_n, base_success, base_alignment_sum = load_completed_ids(results_jsonl)
    max_examples = max(0, int(args.max_examples))
    print(
        "[info] input={} output={} resume_ids={} max_examples={}".format(
            input_jsonl,
            output_dir,
            len(completed_ids),
            max_examples,
        ),
        flush=True,
    )
    if checkpoint_json.exists():
        print(f"[info] found checkpoint file: {checkpoint_json}", flush=True)
    if max_examples and len(completed_ids) >= max_examples:
        print(
            f"[info] Resume found {len(completed_ids)} completed ids; "
            f"already >= max_examples={max_examples}. Rebuilding summary/csv only.",
            flush=True,
        )
        write_summary_and_csv(
            results_jsonl=results_jsonl,
            summary_path=summary_json,
            csv_path=results_csv,
            margin_delta=args.margin_delta,
            ratio_threshold=args.ratio_threshold,
            valid_evidence_min_tokens=args.valid_evidence_min_tokens,
            dynamic_delta_coeff=args.dynamic_delta_coeff,
            dynamic_delta_floor=args.dynamic_delta_floor,
            important_top_fraction=args.important_top_fraction,
            fusion_mode=args.fusion_mode,
            mt_floor=args.mt_floor,
        )
        return

    processed_new = 0
    success_new = 0
    alignment_sum_new = 0.0
    last_id = ""

    with results_jsonl.open("a", encoding="utf-8", buffering=1) as out_f:
        for rec in iter_jsonl(input_jsonl):
            if STOP_REQUESTED:
                print("[info] stop requested; exiting loop cleanly.", flush=True)
                break
            if max_examples and len(completed_ids) >= max_examples:
                break

            rec_id = str(rec.get("id", "")).strip()
            if not rec_id:
                rec_id = f"row_{len(completed_ids) + processed_new}"
            if rec_id in completed_ids:
                continue

            segments = rec.get("segments", {}) or {}
            narrative = (segments.get("narrative") or "").strip()
            evidence = (segments.get("evidence") or "").strip()
            question = (segments.get("question") or "").strip()
            if not (narrative and evidence and question):
                print(
                    f"[warn] Skip id={rec_id}: missing narrative/evidence/question.",
                    flush=True,
                )
                continue

            target_k = int(rec.get("meta", {}).get("target_k", args.target_k))
            print(
                "[run] id={} processed={} resume_n={}".format(
                    rec_id, processed_new + 1, base_n + processed_new
                ),
                flush=True,
            )
            one: Dict[str, Any] | None = None
            run_hvp_samples = effective_hvp_samples
            run_context_tokens = effective_context_tokens
            attempt_plan: List[Tuple[int, int]] = [(run_hvp_samples, run_context_tokens)]
            for fallback in ((1, min(run_context_tokens, 256)), (1, min(run_context_tokens, 192))):
                if fallback not in attempt_plan:
                    attempt_plan.append(fallback)

            last_error: Exception | None = None
            for attempt_idx, (attempt_hvp, attempt_ctx) in enumerate(attempt_plan, start=1):
                try:
                    one = run_one_example(
                        model_name=args.model,
                        narrative=narrative,
                        evidence=evidence,
                        question=question,
                        target_k=target_k,
                        beta=args.beta,
                        gamma=args.gamma,
                        masking=args.masking,
                        quality=args.quality,
                        hvp_samples=attempt_hvp,
                        max_context_tokens=attempt_ctx,
                        answer_text=str(rec.get("answer", "")),
                        fusion_mode=args.fusion_mode,
                        mt_floor=args.mt_floor,
                    )
                    break
                except KeyboardInterrupt:
                    STOP_REQUESTED = True
                    print("[signal] keyboard interrupt; stopping after checkpoint.", flush=True)
                    break
                except RuntimeError as exc:
                    last_error = exc
                    if "out of memory" not in str(exc).lower():
                        print(f"[warn] id={rec_id} failed: {exc}", flush=True)
                        break
                    clear_cuda_memory()
                    if attempt_idx < len(attempt_plan):
                        next_hvp, next_ctx = attempt_plan[attempt_idx]
                        print(
                            "[warn] id={} OOM on attempt {}/{} (hvp={}, ctx={}); retry hvp={} ctx={}.".format(
                                rec_id,
                                attempt_idx,
                                len(attempt_plan),
                                attempt_hvp,
                                attempt_ctx,
                                next_hvp,
                                next_ctx,
                            ),
                            flush=True,
                        )
                        continue
                    print(
                        "[warn] id={} failed after OOM retries (last hvp={}, ctx={}).".format(
                            rec_id, attempt_hvp, attempt_ctx
                        ),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    print(f"[warn] id={rec_id} failed: {exc}", flush=True)
                    break

            if STOP_REQUESTED:
                clear_cuda_memory()
                break

            if one is None:
                clear_cuda_memory()
                continue

            important_indices = select_important_indices(
                one["component_scores"]["final"],
                one["segment_token_spans"],
                top_fraction=args.important_top_fraction,
            )
            segment_mass_raw = one["segment_mass"]
            segment_mass = segment_mass_on_selected(
                one["component_scores"]["final"],
                one["segment_token_spans"],
                important_indices,
            )
            if not important_indices:
                segment_mass = segment_mass_raw
            evidence_mass = float(segment_mass.get("evidence", 0.0))
            narrative_mass = float(segment_mass.get("narrative", 0.0))
            question_mass = float(segment_mass.get("question", 0.0))
            alignment_raw = evidence_mass - narrative_mass
            en_denom = evidence_mass + narrative_mass + EPS
            evidence_share_en = evidence_mass / en_denom
            narrative_share_en = narrative_mass / en_denom
            alignment = evidence_share_en - narrative_share_en
            alignment_ratio = compute_alignment_ratio(evidence_share_en, narrative_share_en)
            alignment_logratio = compute_alignment_logratio(evidence_share_en, narrative_share_en)
            success_strict = evidence_share_en > 0.5
            success_margin = evidence_share_en >= (narrative_share_en + float(args.margin_delta))
            success_ratio = alignment_ratio >= float(args.ratio_threshold)

            segment_token_counts = build_segment_token_counts(one["segment_token_spans"])
            evidence_token_count = int(segment_token_counts.get("evidence", 0))
            dynamic_margin_delta = compute_dynamic_delta(
                evidence_token_count,
                coeff=args.dynamic_delta_coeff,
                floor=args.dynamic_delta_floor,
            )
            success_margin_dynamic = evidence_share_en >= (
                narrative_share_en + dynamic_margin_delta
            )
            valid_by_token_count = evidence_token_count >= int(args.valid_evidence_min_tokens)

            out_record = {
                "id": rec_id,
                "model": one["run_meta"]["model_name"],
                "beta": float(args.beta),
                "gamma": float(args.gamma),
                "masking": args.masking,
                "quality": args.quality,
                "target_k": int(one["target_k"]),
                "onset_token_text": one["onset_token_text"],
                "generated_onset_token_text": one.get(
                    "generated_onset_token_text", one["onset_token_text"]
                ),
                "target_source": one["run_meta"].get("target_source", "generated"),
                "target_token_id": int(one["run_meta"].get("target_token_id", -1)),
                "fusion_mode": one["run_meta"].get("fusion_mode", args.fusion_mode),
                "mt_floor": float(one["run_meta"].get("mt_floor", args.mt_floor)),
                "hvp_samples": int(one["run_meta"].get("s_hvp_samples", run_hvp_samples)),
                "max_context_tokens": int(
                    one["run_meta"].get("max_context_tokens", run_context_tokens)
                ),
                "truncated_tokens": int(one["run_meta"].get("truncated_tokens", 0)),
                "important_top_fraction": float(args.important_top_fraction),
                "important_token_count": int(len(important_indices)),
                "important_token_indices": important_indices,
                "segment_mass_scope": "important_tokens_top_fraction",
                "segment_token_spans": one["segment_token_spans"],
                "segment_token_counts": segment_token_counts,
                "segment_mass_raw": {
                    "narrative": float(segment_mass_raw.get("narrative", 0.0)),
                    "evidence": float(segment_mass_raw.get("evidence", 0.0)),
                    "question": float(segment_mass_raw.get("question", 0.0)),
                },
                "segment_mass": {
                    "narrative": narrative_mass,
                    "evidence": evidence_mass,
                    "question": question_mass,
                },
                "evidence_share_en": float(evidence_share_en),
                "narrative_share_en": float(narrative_share_en),
                "component_segment_mass": {
                    "MT": segment_mass_from_scores(
                        one["component_scores"]["MT"], one["segment_token_spans"]
                    ),
                    "S": segment_mass_from_scores(
                        one["component_scores"]["S"], one["segment_token_spans"]
                    ),
                    "KL": segment_mass_from_scores(
                        one["component_scores"]["KL"], one["segment_token_spans"]
                    ),
                    "final": segment_mass_from_scores(
                        one["component_scores"]["final"], one["segment_token_spans"]
                    ),
                },
                "component_segment_mass_selected": {
                    "MT": segment_mass_on_selected(
                        one["component_scores"]["MT"],
                        one["segment_token_spans"],
                        important_indices,
                    ),
                    "S": segment_mass_on_selected(
                        one["component_scores"]["S"],
                        one["segment_token_spans"],
                        important_indices,
                    ),
                    "KL": segment_mass_on_selected(
                        one["component_scores"]["KL"],
                        one["segment_token_spans"],
                        important_indices,
                    ),
                    "final": segment_mass_on_selected(
                        one["component_scores"]["final"],
                        one["segment_token_spans"],
                        important_indices,
                    ),
                },
                "success": bool(success_strict),
                "success_strict": bool(success_strict),
                "success_margin": bool(success_margin),
                "success_ratio": bool(success_ratio),
                "success_margin_dynamic": bool(success_margin_dynamic),
                "success_margin_0.05": bool(evidence_share_en >= (narrative_share_en + 0.05)),
                "success_ratio_1.2": bool(alignment_ratio >= 1.2),
                "alignment": float(alignment),
                "alignment_raw": float(alignment_raw),
                "alignment_ratio": float(alignment_ratio),
                "alignment_logratio": float(alignment_logratio),
                "margin_delta": float(args.margin_delta),
                "ratio_threshold": float(args.ratio_threshold),
                "dynamic_margin_delta": float(dynamic_margin_delta),
                "valid_by_token_count": bool(valid_by_token_count),
                "latency_ms": int(one["run_meta"]["latency_ms"]),
                "top_tokens": build_top_tokens(
                    one["tokens"], one["component_scores"]["final"], top_n=8
                ),
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            out_f.flush()

            if out_record["segment_token_counts"]["evidence"] == 0:
                print(
                    f"[warn] id={rec_id} evidence token span is empty; check segment mapping.",
                    flush=True,
                )

            processed_new += 1
            success_new += int(success_strict)
            alignment_sum_new += alignment
            completed_ids.add(rec_id)
            last_id = rec_id

            if processed_new % max(1, args.save_every) == 0:
                out_f.flush()
                os.fsync(out_f.fileno())
                write_checkpoint(
                    checkpoint_path=checkpoint_json,
                    input_jsonl=input_jsonl,
                    output_dir=output_dir,
                    processed_total=base_n + processed_new,
                    processed_new=processed_new,
                    success_total=base_success + success_new,
                    alignment_sum_total=base_alignment_sum + alignment_sum_new,
                    last_id=last_id,
                )

            running_n = base_n + processed_new
            if running_n > 0 and running_n % 10 == 0:
                running_success = base_success + success_new
                running_alignment_sum = base_alignment_sum + alignment_sum_new
                print(
                    "[progress] n={} success_rate={:.4f} mean_alignment={:.6f}".format(
                        running_n,
                        running_success / running_n,
                        running_alignment_sum / running_n,
                    ),
                    flush=True,
                )

            clear_cuda_memory()

        out_f.flush()
        os.fsync(out_f.fileno())

    write_checkpoint(
        checkpoint_path=checkpoint_json,
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        processed_total=base_n + processed_new,
        processed_new=processed_new,
        success_total=base_success + success_new,
        alignment_sum_total=base_alignment_sum + alignment_sum_new,
        last_id=last_id,
    )
    write_summary_and_csv(
        results_jsonl=results_jsonl,
        summary_path=summary_json,
        csv_path=results_csv,
        margin_delta=args.margin_delta,
        ratio_threshold=args.ratio_threshold,
        valid_evidence_min_tokens=args.valid_evidence_min_tokens,
        dynamic_delta_coeff=args.dynamic_delta_coeff,
        dynamic_delta_floor=args.dynamic_delta_floor,
        important_top_fraction=args.important_top_fraction,
        fusion_mode=args.fusion_mode,
        mt_floor=args.mt_floor,
    )

    print(f"[done] results_jsonl={results_jsonl}", flush=True)
    print(f"[done] summary_json={summary_json}", flush=True)
    print(f"[done] results_csv={results_csv}", flush=True)
    if STOP_REQUESTED:
        print("[done] exited early by signal; resume is safe.", flush=True)


if __name__ == "__main__":
    main()
