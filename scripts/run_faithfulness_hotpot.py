from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        if "success" in rec and "alignment" in rec:
            n += 1
            if bool(rec.get("success", False)):
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


def write_summary_and_csv(results_jsonl: Path, summary_path: Path, csv_path: Path) -> None:
    alignments: List[float] = []
    n = 0
    n_success = 0
    sum_alignment = 0.0
    sum_ev = 0.0
    sum_na = 0.0
    sum_q = 0.0

    with csv_path.open("w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(
            csv_f,
            fieldnames=[
                "id",
                "success",
                "alignment",
                "evidence_mass",
                "narrative_mass",
                "question_mass",
                "latency_ms",
                "onset_token_text",
            ],
        )
        writer.writeheader()

        for rec in iter_jsonl(results_jsonl):
            if "success" not in rec or "alignment" not in rec:
                continue
            segment_mass = rec.get("segment_mass", {}) or {}
            alignment = float(rec.get("alignment", 0.0))
            success = bool(rec.get("success", False))
            ev = float(segment_mass.get("evidence", 0.0))
            na = float(segment_mass.get("narrative", 0.0))
            q = float(segment_mass.get("question", 0.0))

            writer.writerow(
                {
                    "id": rec.get("id", ""),
                    "success": int(success),
                    "alignment": alignment,
                    "evidence_mass": ev,
                    "narrative_mass": na,
                    "question_mass": q,
                    "latency_ms": rec.get("latency_ms", ""),
                    "onset_token_text": rec.get("onset_token_text", ""),
                }
            )

            n += 1
            n_success += int(success)
            sum_alignment += alignment
            sum_ev += ev
            sum_na += na
            sum_q += q
            alignments.append(alignment)

    if n == 0:
        summary = {
            "n_examples": 0,
            "success_rate": 0.0,
            "mean_alignment": 0.0,
            "median_alignment": 0.0,
            "mean_evidence_mass": 0.0,
            "mean_narrative_mass": 0.0,
            "mean_question_mass": 0.0,
            "alignment_p10": 0.0,
            "alignment_p50": 0.0,
            "alignment_p90": 0.0,
            "alignment_percentiles": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
        }
    else:
        p10, p50, p90 = np.percentile(np.asarray(alignments, dtype=np.float64), [10, 50, 90]).tolist()
        summary = {
            "n_examples": n,
            "success_rate": n_success / n,
            "mean_alignment": sum_alignment / n,
            "median_alignment": statistics.median(alignments),
            "mean_evidence_mass": sum_ev / n,
            "mean_narrative_mass": sum_na / n,
            "mean_question_mass": sum_q / n,
            "alignment_p10": p10,
            "alignment_p50": p50,
            "alignment_p90": p90,
            "alignment_percentiles": {"p10": p10, "p50": p50, "p90": p90},
        }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    if args.num_workers != 1:
        print("[warn] Only num_workers=1 is currently supported; forcing serial execution.")
    if args.model and args.model not in MODEL_OPTIONS and args.model not in MODEL_OPTIONS.values():
        print("[warn] Model is not in known dropdown options; trying it as raw model id.")

    set_seed(args.seed)

    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    results_jsonl = output_dir / "results.jsonl"
    summary_json = output_dir / "summary.json"
    results_csv = output_dir / "results.csv"

    completed_ids, base_n, base_success, base_alignment_sum = load_completed_ids(results_jsonl)
    max_examples = max(0, int(args.max_examples))
    if max_examples and len(completed_ids) >= max_examples:
        print(
            f"[info] Resume found {len(completed_ids)} completed ids; "
            f"already >= max_examples={max_examples}. Rebuilding summary/csv only."
        )
        write_summary_and_csv(results_jsonl, summary_json, results_csv)
        return

    processed_new = 0
    success_new = 0
    alignment_sum_new = 0.0

    with results_jsonl.open("a", encoding="utf-8") as out_f:
        for rec in iter_jsonl(input_jsonl):
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
                print(f"[warn] Skip id={rec_id}: missing narrative/evidence/question.")
                continue

            target_k = int(rec.get("meta", {}).get("target_k", args.target_k))
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
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] id={rec_id} failed: {exc}")
                continue

            segment_mass = one["segment_mass"]
            evidence_mass = float(segment_mass.get("evidence", 0.0))
            narrative_mass = float(segment_mass.get("narrative", 0.0))
            question_mass = float(segment_mass.get("question", 0.0))
            alignment = evidence_mass - narrative_mass
            success = evidence_mass > narrative_mass

            out_record = {
                "id": rec_id,
                "model": one["run_meta"]["model_name"],
                "beta": float(args.beta),
                "gamma": float(args.gamma),
                "masking": args.masking,
                "quality": args.quality,
                "target_k": int(one["target_k"]),
                "onset_token_text": one["onset_token_text"],
                "segment_token_spans": one["segment_token_spans"],
                "segment_mass": {
                    "narrative": narrative_mass,
                    "evidence": evidence_mass,
                    "question": question_mass,
                },
                "success": bool(success),
                "alignment": float(alignment),
                "latency_ms": int(one["run_meta"]["latency_ms"]),
                "top_tokens": build_top_tokens(
                    one["tokens"], one["component_scores"]["final"], top_n=8
                ),
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            processed_new += 1
            success_new += int(success)
            alignment_sum_new += alignment
            completed_ids.add(rec_id)

            if processed_new % max(1, args.save_every) == 0:
                out_f.flush()
                os.fsync(out_f.fileno())

            running_n = base_n + processed_new
            if running_n > 0 and running_n % 10 == 0:
                running_success = base_success + success_new
                running_alignment_sum = base_alignment_sum + alignment_sum_new
                print(
                    "[progress] n={} success_rate={:.4f} mean_alignment={:.6f}".format(
                        running_n,
                        running_success / running_n,
                        running_alignment_sum / running_n,
                    )
                )

        out_f.flush()
        os.fsync(out_f.fileno())

    write_summary_and_csv(results_jsonl, summary_json, results_csv)

    print(f"[done] results_jsonl={results_jsonl}")
    print(f"[done] summary_json={summary_json}")
    print(f"[done] results_csv={results_csv}")


if __name__ == "__main__":
    main()
