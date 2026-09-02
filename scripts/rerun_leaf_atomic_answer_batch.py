"""Re-answer saved HyperBranch leaf nodes without changing their DAG or evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT)]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from hyper_branch.client import OpenAIClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-run", default="full_test1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True, help="Exclusive index.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / "configs" / f"{args.dataset}.yaml").read_text(encoding="utf-8"))
    questions = json.loads((PROJECT_ROOT / "questions" / args.dataset / "questions.json").read_text(encoding="utf-8"))
    prompt = (PROJECT_ROOT / "prompts" / "atomic_answer.md").read_text(encoding="utf-8").strip()
    client = OpenAIClient(
        api_key=os.environ["OPENAI_API_KEY"],
        model=args.llm_model,
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    source_dir = PROJECT_ROOT / "runs" / "depo_hyperbranch" / args.dataset / args.source_run
    output_dir = PROJECT_ROOT / "runs" / "depo_hyperbranch" / args.dataset / args.run_id

    for index in range(args.start, args.end):
        output_path = output_dir / f"{index:05d}" / "result.json"
        if args.resume and output_path.exists():
            continue
        try:
            source_path = source_dir / f"{index:05d}" / "result.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            leaf = source["nodes"][-1]
            question = questions[index - 1]["question"].strip()
            response = client.chat_json(
                prompt,
                json.dumps(
                    {
                        "original_question": question,
                        "atomic_question": leaf["rewritten_question"],
                        # The saved leaf question has dependency placeholders substituted already.
                        "dependency_context": [],
                        "evidence_blocks": leaf["evidence_blocks"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                max_tokens=900,
            )
            answer = str(response["answer"]).strip()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "source_result": str(source_path),
                        "question": question,
                        "gold_answer": questions[index - 1].get("answer"),
                        "leaf_node_id": leaf["id"],
                        "atomic_question": leaf["rewritten_question"],
                        "previous_answer": leaf["answer"],
                        "answer": answer,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"{args.dataset} #{index}: {leaf['answer']} -> {answer}", flush=True)
        except Exception as exc:
            print(f"{args.dataset} #{index} failed: {exc}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
