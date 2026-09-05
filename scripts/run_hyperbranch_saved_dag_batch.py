from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "depo")]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from hyper_branch.client import OpenAIClient
from hyper_branch.pipeline import HyperBranchPipeline
from run_depo_hyperbranch_batch import save_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HyperBranch with saved atomic-question DAGs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-run", default="730_1000", help="Run containing saved DAGs.")
    parser.add_argument("--entities-run", required=True, help="Run containing current topic_entities in <index>/result.json.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True, help="Exclusive question index.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    return parser.parse_args()


def load_saved_dag(source_dir: Path, index: int) -> tuple[dict[str, object], Path]:
    matches = list(source_dir.glob(f"{index:05d}_*/hyperbranch_dag.json"))
    if len(matches) == 1:
        path = matches[0]
        return json.loads(path.read_text(encoding="utf-8")), path
    path = source_dir / f"{index:05d}" / "result.json"
    if path.exists():
        dag = json.loads(path.read_text(encoding="utf-8")).get("atomic_question_dag")
        if isinstance(dag, dict):
            return dag, path
    raise FileNotFoundError(f"No saved DAG found for #{index}")


def main() -> int:
    args = parse_args()
    config = yaml.safe_load((PROJECT_ROOT / "configs" / f"{args.dataset}.yaml").read_text(encoding="utf-8"))
    questions = json.loads((PROJECT_ROOT / "questions" / args.dataset / "questions.json").read_text(encoding="utf-8"))
    source_dir = PROJECT_ROOT / "runs" / "depo_hyperbranch" / args.dataset / args.source_run
    entities_dir = PROJECT_ROOT / "runs" / "depo_hyperbranch" / args.dataset / args.entities_run
    output_dir = PROJECT_ROOT / "runs" / "depo_hyperbranch" / args.dataset / args.run_id
    client = OpenAIClient(
        api_key=os.environ["OPENAI_API_KEY"],
        model=args.llm_model,
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    hyperbranch = HyperBranchPipeline(
        PROJECT_ROOT / config["dataset_root"],
        model=args.llm_model,
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),
        client=client,
    )

    indexed_questions = [(index, questions[index - 1]) for index in range(args.start, args.end)]
    for index, item in indexed_questions:
        result_path = output_dir / f"{index:05d}" / "result.json"
        if args.resume and result_path.exists():
            continue
        try:
            question = item["question"].strip()
            dag, dag_path = load_saved_dag(source_dir, index)
            entity_path = entities_dir / f"{index:05d}" / "result.json"
            entities = json.loads(entity_path.read_text(encoding="utf-8"))["topic_entities"]
            result = hyperbranch.run(question, dag, entities)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(
                    {
                        "source_dag": str(dag_path),
                        "source_entities": str(entity_path),
                        "topic_entities": entities,
                        "nodes": [
                            {
                                "id": node["node_id"],
                                "rewritten_question": node["question"],
                                "entities": node["entities"],
                                "evidence_blocks": node["evidence_blocks"],
                                "answer": node["answer"],
                            }
                            for node in result["atomic_answers"]
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"{args.dataset} #{index}: {result['final_answer']['answer']}", flush=True)
        except Exception as exc:
            print(f"{args.dataset} #{index} failed: {exc}", file=sys.stderr, flush=True)
    score_file = save_scores(args.dataset, args.run_id, output_dir, indexed_questions)
    score = json.loads(score_file.read_text(encoding="utf-8"))
    print(f"saved_scores={score_file}")
    print(f"EM={score['overall']['em']:.4f}")
    print(f"F1={score['overall']['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
