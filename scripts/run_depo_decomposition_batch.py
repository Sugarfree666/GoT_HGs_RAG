"""Run only DEPO entity recognition, PAS parsing, and atomic-question DAG generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT / "depo"), str(PROJECT_ROOT)]

from hyper_branch.client import OpenAIClient


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DEPO atomic-question DAG decomposition for one dataset."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    args = parser.parse_args()

    api_key = args.api_key or os.environ["OPENAI_API_KEY"]
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    questions = json.loads(
        (PROJECT_ROOT / "questions" / args.dataset / "questions.json").read_text(encoding="utf-8")
    )[args.start - 1 : args.end]
    if args.limit is not None:
        questions = questions[: args.limit]

    from hanlp_sdp_parser import HanLPSDPParser
    from pipeline import run_depo

    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / f"{args.dataset}.yaml").read_text(encoding="utf-8")
    )
    llm = OpenAIClient(
        api_key=api_key,
        model=args.llm_model,
        embedding_model=config["embedding_model"],
        timeout_seconds=config["timeout_seconds"],
        temperature=config["temperature"],
        base_url=base_url,
    )
    sdp_parser = HanLPSDPParser()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "runs" / "depo_decomposition" / args.dataset / run_id
    for offset, item in enumerate(questions, start=1):
        index = args.start + offset - 1
        result_file = output_dir / f"{index:05d}" / "result.json"
        if args.resume and result_file.exists():
            continue

        question = item["question"].strip()
        decomposition = run_depo(question, sdp_parser, llm)
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(
            json.dumps(
                {
                    "question": question,
                    "gold_answer": item["answer"],
                    "entities": decomposition["entities"],
                    "atomic_question_dag": decomposition["atomic_question_dag"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{args.dataset} #{index}: {len(decomposition['atomic_question_dag']['nodes'])} nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
