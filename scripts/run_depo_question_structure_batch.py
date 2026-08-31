"""Run DEPO only through question-structure extraction for one dataset."""

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

from hanlp_sdp_parser import HanLPSDPParser  # noqa: E402
from hyper_branch.client import OpenAIClient  # noqa: E402
from pipeline import extract_question_structure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run DEPO through question-structure extraction only and write one "
            "Markdown file with each question and its extracted structure."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output-file")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--limit", type=int)
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

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        Path(args.output_file)
        if args.output_file
        else PROJECT_ROOT
        / "runs"
        / "depo_question_structure"
        / args.dataset
        / run_id
        / "question_structure.md"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    questions = _load_questions(args.dataset, args.start, args.end, args.limit)
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
    sdp_parser = HanLPSDPParser()

    entries: list[dict[str, Any]] = []
    for index, item in questions:
        question = str(item["question"]).strip()
        try:
            structure_result = extract_question_structure(question, sdp_parser, llm)
            question_structure = _format_paths(structure_result["question_structure"])
            entries.append(
                {
                    "index": index,
                    "question": question,
                    "question_structure": question_structure,
                }
            )
            status = "ok" if question_structure else "empty"
            print(f"{args.dataset} #{index}: {status} branches={len(question_structure)}")
        except Exception as exc:  # noqa: BLE001
            entries.append(
                {
                    "index": index,
                    "question": question,
                    "question_structure": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"{args.dataset} #{index}: error {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        output_file.write_text(_render_markdown(entries), encoding="utf-8")

    print(f"results={output_file}")
    return 0


def _load_questions(
    dataset: str,
    start: int,
    end: int | None,
    limit: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    question_file = PROJECT_ROOT / "questions" / dataset / "questions.json"
    questions = json.loads(question_file.read_text(encoding="utf-8"))[start - 1 : end]
    if limit is not None:
        questions = questions[:limit]
    return list(enumerate(questions, start=start))


def _format_paths(paths: list[list[str]]) -> list[str]:
    return [
        " -- ".join(token for token in (part.strip() for part in path) if token)
        for path in paths
        if any(part.strip() for part in path)
    ]


def _render_markdown(entries: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for entry in entries:
        lines = [
            f"## #{entry['index']}",
            "",
            f"Question: {entry['question']}",
            "",
            "Question structure:",
        ]
        if entry.get("error"):
            lines.append(f"- ERROR: {entry['error']}")
        elif entry["question_structure"]:
            lines.extend(f"- {branch}" for branch in entry["question_structure"])
        else:
            lines.append("- EMPTY")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


if __name__ == "__main__":
    raise SystemExit(main())
