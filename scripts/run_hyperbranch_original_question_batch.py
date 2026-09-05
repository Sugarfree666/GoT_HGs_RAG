"""Run the original-question-only retrieval ablation for HyperBranch.

This script does not invoke DEPO.  It builds a one-node DAG containing the
original question, so all retrieval-side queries and the final answer use that
same question.  It is therefore an end-to-end ablation of DEPO decomposition
and its multi-node DAG.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT)]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from eval.eval import cal_em, cal_f1  # noqa: E402
from hyper_branch.client import OpenAIClient  # noqa: E402
from hyper_branch.pipeline import ENTITY_RECOGNITION_PROMPT, HyperBranchPipeline  # noqa: E402


def build_original_question_dag(question: str) -> dict[str, list[dict[str, Any]]]:
    """Return the one-node DAG used to remove DEPO decomposition."""
    return {
        "nodes": [
            {
                "id": "q1",
                "question": question,
                "depends_on": [],
            }
        ]
    }


def extract_original_question_entities(question: str, client: OpenAIClient) -> list[str]:
    """Extract retrieval anchors directly from the original question."""
    payload = client.chat_json(
        ENTITY_RECOGNITION_PROMPT,
        json.dumps({"question": question}, ensure_ascii=False),
    )
    entities = payload.get("entities")
    if not isinstance(entities, list) or not all(isinstance(item, str) for item in entities):
        raise ValueError("Entity recognition must return an 'entities' list of strings")
    return list(dict.fromkeys(entity.strip() for entity in entities if entity.strip()))


def gold_answers(question: dict[str, Any]) -> list[str]:
    for key in ("golden_answers", "answers", "answer"):
        value = question.get(key)
        if isinstance(value, list):
            return [str(answer).strip() for answer in value if str(answer).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    return []


def result_answer(result_file: Path) -> str:
    if not result_file.is_file():
        return ""
    nodes = json.loads(result_file.read_text(encoding="utf-8")).get("nodes", [])
    if not nodes or not isinstance(nodes[-1], dict):
        return ""
    return str(nodes[-1].get("answer", "") or "").strip()


def save_scores(
    dataset: str,
    run_id: str,
    output_dir: Path,
    indexed_questions: list[tuple[int, dict[str, Any]]],
    question_file: Path,
) -> Path:
    records: list[dict[str, Any]] = []
    for index, question in indexed_questions:
        answers = gold_answers(question)
        answer = result_answer(output_dir / f"{index:05d}" / "result.json")
        records.append(
            {
                "index": index,
                "question": question["question"],
                "golden_answers": answers,
                "answer": answer,
                "em": float(cal_em([answers], [answer])) if answers else 0.0,
                "f1": float(cal_f1([answers], [answer])) if answers else 0.0,
            }
        )

    score_dir = PROJECT_ROOT / "eval" / "results" / "depo_hyperbranch" / dataset / run_id
    score_dir.mkdir(parents=True, exist_ok=True)
    completed = sum(
        (output_dir / f"{index:05d}" / "result.json").is_file()
        for index, _ in indexed_questions
    )
    score = {
        "meta": {
            "question_file": str(question_file.resolve()),
            "runs_dir": str(output_dir.resolve()),
            "indices": [index for index, _ in indexed_questions],
            "metrics": ["em", "f1"],
            "dag_mode": "single_original_question",
            "depo_used": False,
            "entity_source": "original_question",
        },
        "counts": {"total": len(records), "completed": completed, "missing": len(records) - completed},
        "overall": {
            "em": sum(record["em"] for record in records) / len(records) if records else 0.0,
            "f1": sum(record["f1"] for record in records) / len(records) if records else 0.0,
        },
    }
    (score_dir / "test_result.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (score_dir / "test_score.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return score_dir / "test_score.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run HyperBranch with a one-node DAG containing only the original question."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--question-file",
        help="Question JSON file. Defaults to questions/<dataset>/questions.json.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    args = parser.parse_args()

    if args.start < 1:
        parser.error("--start must be >= 1")
    if args.end is not None and args.end < args.start:
        parser.error("--end must be >= --start")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    question_file = (
        Path(args.question_file)
        if args.question_file
        else PROJECT_ROOT / "questions" / args.dataset / "questions.json"
    )
    if not question_file.is_file():
        parser.error(f"question file not found: {question_file}")
    question_file = question_file.resolve()
    questions = json.loads(question_file.read_text(encoding="utf-8"))[args.start - 1 : args.end]
    if args.limit is not None:
        questions = questions[: args.limit]
    indexed_questions = [
        (args.start + offset, question) for offset, question in enumerate(questions)
    ]

    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / f"{args.dataset}.yaml").read_text(encoding="utf-8")
    )
    api_key = args.api_key or os.environ["OPENAI_API_KEY"]
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    client = OpenAIClient(
        api_key=api_key,
        model=args.llm_model,
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        base_url=base_url,
    )
    hyperbranch = HyperBranchPipeline(
        PROJECT_ROOT / config["dataset_root"],
        model=args.llm_model,
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        api_key=api_key,
        base_url=base_url,
        client=client,
    )

    run_id = args.run_id or f"ablation_original_question_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = PROJECT_ROOT / "runs" / "depo_hyperbranch" / args.dataset / run_id
    for index, item in indexed_questions:
        result_file = output_dir / f"{index:05d}" / "result.json"
        if args.resume and result_file.is_file():
            continue

        question = str(item["question"]).strip()
        try:
            topic_entities = extract_original_question_entities(question, client)
            original_question_dag = build_original_question_dag(question)
            result = hyperbranch.run(question, original_question_dag, topic_entities)
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(
                json.dumps(
                    {
                        "topic_entities": topic_entities,
                        "atomic_question_dag": original_question_dag,
                        "dag_mode": "single_original_question",
                        "topic_entity_ids": result["topic_entity_ids"],
                        "nodes": [
                            {
                                "id": node["node_id"],
                                "rewritten_question": node["question"],
                                "entities": node["entities"],
                                "entity_ids": node["entity_ids"],
                                "evidence_blocks": node["evidence_blocks"],
                                "answer": node["answer"],
                            }
                            for node in result["atomic_answers"]
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"{args.dataset} #{index}: {result['final_answer']['answer']}")
        except Exception as exc:  # noqa: BLE001
            print(f"{args.dataset} #{index} failed: {exc}", file=sys.stderr)

    score_file = save_scores(args.dataset, run_id, output_dir, indexed_questions, question_file)
    score = json.loads(score_file.read_text(encoding="utf-8"))
    print(f"saved_scores={score_file}")
    print(f"EM={score['overall']['em']:.4f}")
    print(f"F1={score['overall']['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
