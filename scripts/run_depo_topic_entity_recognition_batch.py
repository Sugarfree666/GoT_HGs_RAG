"""Run only DEPO topic-entity recognition for one dataset."""

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
sys.path[:0] = [str(PROJECT_ROOT / "depo"), str(PROJECT_ROOT)]

from entity_masking_preprocessor import TOPIC_ENTITY_RECOGNITION_PROMPT
from hyper_branch.client import OpenAIClient


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DEPO topic-entity recognition only; no parsing, DAG, retrieval, or answering."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    args = parser.parse_args()

    question_file = PROJECT_ROOT / "questions" / args.dataset / "questions.json"
    questions = json.loads(question_file.read_text(encoding="utf-8"))[args.start - 1 : args.end]
    if args.limit is not None:
        questions = questions[: args.limit]

    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / f"{args.dataset}.yaml").read_text(encoding="utf-8")
    )
    llm = OpenAIClient(
        api_key=args.api_key or os.environ["OPENAI_API_KEY"],
        model=args.llm_model,
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
    )

    records: list[dict[str, Any]] = []
    for index, item in enumerate(questions, start=args.start):
        question = str(item["question"]).strip()
        entities = llm.chat_json(
            TOPIC_ENTITY_RECOGNITION_PROMPT,
            json.dumps({"question": question}, ensure_ascii=False),
        )["entities"]
        records.append({"index": index, "question": question, "entities": entities})
        print(f"{args.dataset} #{index}: {entities}")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "runs" / "depo_topic_entity_recognition" / args.dataset / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"
    output_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results={output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
