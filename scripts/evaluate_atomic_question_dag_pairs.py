"""Blind pairwise semantic evaluation of saved atomic-question DAGs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "depo"), str(PROJECT_ROOT)]

from hyper_branch.client import OpenAIClient  # noqa: E402


JUDGE_PROMPT = """You are evaluating two atomic-question DAG decompositions of one multi-hop QA question.

Judge semantic executability only, not factual retrieval. A high-quality DAG preserves every required entity, relation direction, qualifier, comparison/aggregation operand, and the original question's final answer target. Its dependencies must make substitutions unambiguous. A one-node DAG is correct when it fully preserves a genuinely single-hop question; do not reward or penalize a DAG merely for having fewer or more nodes.

Return JSON exactly in this form:
{
  "winner": "A" | "B" | "tie",
  "score_a": 1-5,
  "score_b": 1-5,
  "reason": "one concise sentence",
  "errors_a": ["short error descriptions"],
  "errors_b": ["short error descriptions"]
}
"""


def _nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": node["id"],
            "question": node["question"],
            "depends_on": node["depends_on"],
        }
        for node in nodes
    ]


def _baseline_nodes(dataset: str, index: int, run_id: str) -> list[dict[str, Any]]:
    roots = list(
        (PROJECT_ROOT / "runs" / "depo_hyperbranch" / dataset / run_id).glob(
            f"{index:05d}_*"
        )
    )
    if len(roots) != 1:
        raise FileNotFoundError(f"expected one baseline directory for #{index}, got {len(roots)}")
    payload = json.loads((roots[0] / "decomposition.json").read_text(encoding="utf-8"))
    return payload["stages"]["6_atomic_question_dag"]["nodes"]


def _saved_run_nodes(dataset: str, index: int, run_id: str) -> list[dict[str, Any]]:
    path = (
        PROJECT_ROOT
        / "runs"
        / "depo_atomic_question_dag"
        / dataset
        / run_id
        / f"{index:05d}"
        / "decomposition.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["atomic_question_dag"]["nodes"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Blindly compare two saved DAG runs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--baseline-run", default="730_1000")
    parser.add_argument(
        "--reference-run",
        help="Compare with another saved decomposition-only run instead of 730_1000.",
    )
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    args = parser.parse_args()

    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / f"{args.dataset}.yaml").read_text(encoding="utf-8")
    )
    llm = OpenAIClient(
        api_key=args.api_key or os.environ["OPENAI_API_KEY"],
        model=args.llm_model,
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=0,
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
    )
    output_name = (
        f"pairwise_eval_vs_{args.reference_run}.json"
        if args.reference_run
        else "pairwise_eval.json"
    )
    output = (
        PROJECT_ROOT
        / "runs"
        / "depo_atomic_question_dag"
        / args.dataset
        / args.candidate_run
        / output_name
    )
    results: list[dict[str, Any]] = (
        json.loads(output.read_text(encoding="utf-8")) if args.resume and output.exists() else []
    )
    completed = {item["index"] for item in results if "error" not in item}

    for index in range(args.start, args.start + args.limit):
        if index in completed:
            continue
        results = [item for item in results if item["index"] != index]
        candidate_path = (
            PROJECT_ROOT
            / "runs"
            / "depo_atomic_question_dag"
            / args.dataset
            / args.candidate_run
            / f"{index:05d}"
            / "decomposition.json"
        )
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        current = _nodes(candidate["atomic_question_dag"]["nodes"])
        baseline = _nodes(
            _saved_run_nodes(args.dataset, index, args.reference_run)
            if args.reference_run
            else _baseline_nodes(args.dataset, index, args.baseline_run)
        )
        a_name, b_name = ("current", "baseline") if index % 2 == 0 else ("baseline", "current")
        a_nodes, b_nodes = (current, baseline) if a_name == "current" else (baseline, current)
        try:
            judgment = llm.chat_json(
                JUDGE_PROMPT,
                json.dumps(
                    {"question": candidate["question"], "dag_a": a_nodes, "dag_b": b_nodes},
                    ensure_ascii=False,
                ),
            )
            if judgment["winner"] not in {"A", "B", "tie"}:
                raise ValueError(f"invalid winner: {judgment['winner']}")
            winner = (
                "tie"
                if judgment["winner"] == "tie"
                else a_name if judgment["winner"] == "A" else b_name
            )
            result = {
                "index": index,
                "question": candidate["question"],
                "winner": winner,
                "current_score": judgment["score_a"] if a_name == "current" else judgment["score_b"],
                "baseline_score": judgment["score_b"] if a_name == "current" else judgment["score_a"],
                "reason": judgment["reason"],
                "current_errors": judgment["errors_a"] if a_name == "current" else judgment["errors_b"],
                "baseline_errors": judgment["errors_b"] if a_name == "current" else judgment["errors_a"],
            }
            print(f"{args.dataset} #{index}: {winner}", flush=True)
        except Exception as exc:  # noqa: BLE001
            result = {"index": index, "error": f"{type(exc).__name__}: {exc}"}
            print(f"{args.dataset} #{index} failed: {result['error']}", file=sys.stderr)
        results.append(result)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
