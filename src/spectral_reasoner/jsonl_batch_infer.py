"""JSONL batch inference entrypoint for deployable SpectralReasoner bundles.

Input JSONL schema, one request per line:

{
  "id": "optional-id",
  "prompt": "question: ... answer:",
  "candidates": ["red", "blue"],
  "evidence_by_answer": {"red": ["..."], "blue": []},
  "support_path": ["concept001a", "concept001b"],
  "kind": "known",
  "expected": "red",
  "metadata": {},
  "recovery_actions": [
    {"key": "doc1:0", "text": "...", "answer_hint": "red", "support_path": []}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .deployment_bundle import load_deployment_service
from .service import SpectralReasonerRequest
from .torch_char_lm_benchmark import require_torch


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield line_no, json.loads(text)
            except json.JSONDecodeError as exc:
                yield line_no, {"_error": f"invalid json: {exc}"}


def normalize_request(row: dict[str, Any]) -> SpectralReasonerRequest:
    prompt = str(row.get("prompt", ""))
    candidates = [str(item) for item in row.get("candidates", [])]
    evidence_by_answer = row.get("evidence_by_answer", {})
    if not isinstance(evidence_by_answer, dict):
        evidence_by_answer = {}
    evidence_by_answer = {str(k): [str(x) for x in (v or [])] for k, v in evidence_by_answer.items()}
    for candidate in candidates:
        evidence_by_answer.setdefault(candidate, [])
    support_path = [str(item) for item in row.get("support_path", [])]
    recovery_actions = row.get("recovery_actions")
    if recovery_actions is not None:
        recovery_actions = [item for item in recovery_actions if isinstance(item, dict)]
    return SpectralReasonerRequest(
        prompt=prompt,
        candidates=candidates,
        evidence_by_answer=evidence_by_answer,
        support_path=support_path,
        kind=str(row.get("kind", "known")),
        metadata=row.get("metadata", None),
        recovery_actions=recovery_actions,
    )


def score_response(row: dict[str, Any], answer: str | None, refused: bool) -> dict[str, float]:
    if "expected" not in row:
        return {"has_expected": 0.0, "correct": 0.0, "hallucination": 0.0}
    expected = row.get("expected")
    if expected is None:
        correct = bool(refused)
        hallucination = not refused
    else:
        correct = answer == str(expected)
        hallucination = False
    return {"has_expected": 1.0, "correct": float(correct), "hallucination": float(hallucination)}


def write_summary(rows: list[dict[str, Any]], elapsed: float, path: Path) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    expected_rows = [row for row in ok_rows if row["metrics"]["has_expected"] > 0.5]
    route_counts: dict[str, int] = {}
    for row in ok_rows:
        route = str(row["response"].get("route", "unknown"))
        route_counts[route] = route_counts.get(route, 0) + 1
    summary = {
        "total": len(rows),
        "ok": len(ok_rows),
        "errors": len(rows) - len(ok_rows),
        "elapsed_sec": elapsed,
        "throughput_qps": len(ok_rows) / max(elapsed, 1.0e-9),
        "route_counts": route_counts,
        "recovery_rate": route_counts.get("active_recovery", 0) / max(len(ok_rows), 1),
        "mean_risk": 0.0 if not ok_rows else float(np.mean([row["response"]["risk"] for row in ok_rows])),
        "mean_confidence": 0.0 if not ok_rows else float(np.mean([row["response"]["confidence"] for row in ok_rows])),
        "mean_lm_forward_calls": 0.0
        if not ok_rows
        else float(np.mean([row["response"]["spectral_trace"].get("lm_forward_calls", 0.0) for row in ok_rows])),
        "scored": len(expected_rows),
        "correct": 0.0 if not expected_rows else float(np.mean([row["metrics"]["correct"] for row in expected_rows])),
        "hallucination": 0.0 if not expected_rows else float(np.mean([row["metrics"]["hallucination"] for row in expected_rows])),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-memory", type=Path, default=None)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    torch, nn, F = require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    service = load_deployment_service(args.bundle, torch, nn, F, device)

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as out:
        for line_no, row in iter_jsonl(args.input):
            request_id = row.get("id", line_no)
            if row.get("_error"):
                result = {"id": request_id, "line": line_no, "ok": False, "error": row["_error"]}
            else:
                try:
                    request = normalize_request(row)
                    if not request.prompt or not request.candidates:
                        raise ValueError("request requires non-empty prompt and candidates")
                    response = service.handle(request)
                    response_dict = service.response_dict(response)
                    metrics = score_response(row, response.answer, response.refused)
                    result = {
                        "id": request_id,
                        "line": line_no,
                        "ok": True,
                        "expected": row.get("expected", None),
                        "metrics": metrics,
                        "response": response_dict,
                    }
                except Exception as exc:  # noqa: BLE001 - batch jobs should continue per line.
                    result = {"id": request_id, "line": line_no, "ok": False, "error": str(exc)}
            out.write(json.dumps(result, ensure_ascii=True) + "\n")
            rows.append(result)
    elapsed = time.perf_counter() - start
    summary = write_summary(rows, elapsed, summary_path)
    if args.save_memory is not None:
        service.save_memory(args.save_memory)
    print("SpectralJSONLBatchInfer")
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"summary={summary_path}")
    print(
        f"ok={summary['ok']}/{summary['total']} qps={summary['throughput_qps']:.2f} "
        f"correct={summary['correct']:.4f} hallucination={summary['hallucination']:.4f} "
        f"recovery={summary['recovery_rate']:.4f} lm_fwds={summary['mean_lm_forward_calls']:.2f}"
    )


if __name__ == "__main__":
    main()
