"""Run DEPO decomposition only: no retrieval or HyperBranch execution."""

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

from hanlp_sdp_parser import HanLPSDPParser  # noqa: E402
from hyper_branch.client import OpenAIClient  # noqa: E402
from atomic_question_dag import generate_atomic_question_dag  # noqa: E402
from pipeline import extract_question_structure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate question structures and atomic-question DAGs only."
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

    if args.start < 1 or (args.end is not None and args.end < args.start):
        parser.error("require --start >= 1 and --end >= --start")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    questions = json.loads(
        (PROJECT_ROOT / "questions" / args.dataset / "questions.json").read_text(
            encoding="utf-8"
        )
    )[args.start - 1 : args.end]
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
    parser_model = HanLPSDPParser()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "runs" / "depo_atomic_question_dag" / args.dataset / run_id

    for offset, item in enumerate(questions):
        index = args.start + offset
        result_file = output_dir / f"{index:05d}" / "decomposition.json"
        if args.resume and result_file.exists():
            continue
        question = str(item["question"]).strip()
        try:
            structure = extract_question_structure(question, parser_model, llm)
            dag = generate_atomic_question_dag(
                llm,
                question,
                structure["entities"],
                structure["question_structure"],
            )
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(
                json.dumps(
                    {
                        "index": index,
                        "question": question,
                        "entities": structure["entities"],
                        "question_structure": structure["question_structure"],
                        "atomic_question_dag": dag,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"{args.dataset} #{index}: "
                f"nodes={len(dag['nodes'])}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{args.dataset} #{index} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
