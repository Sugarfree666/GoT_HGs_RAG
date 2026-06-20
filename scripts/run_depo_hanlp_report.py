from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))

from entity_masking_preprocessor import EntityMaskingPreprocessor
from hanlp_sdp_parser import HanLPSDPParser
from io_utils import read_questions
from llm_client import LLMClient
from main import run_hanlp_sdp_pipeline
from models import QuestionRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DEPO HanLP-SDP pipeline on a questions dataset and write a compact "
            "Step4-only Markdown report."
        )
    )
    parser.add_argument(
        "--questions-file",
        default="questions/2wikimultihopqa/questions.json",
        help="Questions JSON file. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--output",
        default="runs/result_depo_hanlp.md",
        help="Markdown report path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-based inclusive start question index. Default: 1.",
    )
    parser.add_argument(
        "--end",
        type=int,
        help="1-based inclusive end question index.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of questions to run after applying --start/--end.",
    )
    parser.add_argument("--api-key", help="OpenAI API key. Used only if OPENAI_API_KEY is not set.")
    parser.add_argument("--base-url", help="OpenAI base URL. Used only if OPENAI_BASE_URL is not set.")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model for explicit entity extraction.")
    parser.add_argument(
        "--hanlp-model",
        help="HanLP pretrained constant name from hanlp.pretrained.mtl/sdp, or a local model path.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Tri-SDP debug JSON files. The Markdown report remains compact.",
    )
    parser.add_argument(
        "--debug-dir",
        default="debug/hanlp_sdp",
        help="Directory for HanLP Tri-SDP debug JSON files when --debug is enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    questions_file = _repo_path(args.questions_file)
    output_path = _repo_path(args.output)
    debug_dir = str(_repo_path(args.debug_dir))

    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    records = _select_records(read_questions(questions_file), start=args.start, end=args.end, limit=args.limit)
    if not records:
        print("No questions selected.", file=sys.stderr)
        return 2

    llm_client = LLMClient(api_key=api_key, base_url=base_url, model=args.llm_model)
    preprocessor = EntityMaskingPreprocessor(llm_client)
    parser = HanLPSDPParser(args.hanlp_model)

    report_lines = _report_header(
        questions_file=questions_file,
        output_path=output_path,
        start=args.start,
        end=args.end,
        limit=args.limit,
        debug=args.debug,
        debug_dir=Path(debug_dir),
    )

    total = len(records)
    for offset, (question_index, record) in enumerate(records, start=1):
        print(f"[{offset}/{total}] Running question {question_index}...")
        try:
            result = run_hanlp_sdp_pipeline(
                record=record,
                index=question_index,
                preprocessor=preprocessor,
                parser=parser,
                debug=args.debug,
                debug_dir=debug_dir,
            )
            report_lines.extend(_format_question_result(question_index, record, result))
        except Exception as exc:  # Keep long batch runs inspectable even if one item fails.
            print(f"Question {question_index} failed: {exc}", file=sys.stderr)
            report_lines.extend(_format_question_error(question_index, record, exc))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Report written to {output_path}")
    return 0


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def _select_records(
    records: list[QuestionRecord],
    *,
    start: int,
    end: int | None,
    limit: int | None,
) -> list[tuple[int, QuestionRecord]]:
    if start < 1:
        raise ValueError("--start must be >= 1.")
    if end is not None and end < start:
        raise ValueError("--end must be >= --start.")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be >= 1.")

    indexed_records = list(enumerate(records, start=1))
    selected = indexed_records[start - 1 : end]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _report_header(
    *,
    questions_file: Path,
    output_path: Path,
    start: int,
    end: int | None,
    limit: int | None,
    debug: bool,
    debug_dir: Path,
) -> list[str]:
    selected_range = f"{start}-{end}" if end is not None else f"{start}-end"
    lines = [
        "# DEPO HanLP-SDP Step4 Report",
        "",
        f"- Questions file: `{questions_file}`",
        f"- Output file: `{output_path}`",
        f"- Selected range: `{selected_range}`",
    ]
    if limit is not None:
        lines.append(f"- Limit: `{limit}`")
    if debug:
        lines.append(f"- Debug dir: `{debug_dir}`")
    lines.append("")
    return lines


def _format_question_result(index: int, record: QuestionRecord, result: dict[str, Any]) -> list[str]:
    preprocess_result = result["preprocess_result"]
    token_reasoning_structure = result["token_reasoning_structure"]
    title = f"## Question {index}"
    if record.qid:
        title += f" ({record.qid})"

    lines = [
        title,
        "",
        "### Original Question",
        "",
        record.question,
        "",
        "### Explicit Entities",
        "",
    ]
    if preprocess_result.explicit_entities.entities:
        for entity in preprocess_result.explicit_entities.entities:
            lines.append(f"- {entity.text}")
    else:
        lines.append("(none)")

    lines.extend(
        [
            "",
            "### Step4 Token Reasoning Structure",
            "",
            "[Graph]",
            "```text",
        ]
    )
    if token_reasoning_structure.edges:
        for edge in token_reasoning_structure.edges:
            lines.append(f"{edge.source_text} ---- {edge.target_text}")
    else:
        lines.append("(none)")
    lines.extend(["```", "", "[Paths]", "```text"])
    if token_reasoning_structure.paths:
        for path in token_reasoning_structure.paths:
            lines.append(f"{path.path_id}: {' ---- '.join(path.nodes)}")
    else:
        lines.append("(none)")
    lines.append("```")

    metadata = _format_step4_metadata(token_reasoning_structure)
    if metadata:
        lines.extend(["", *metadata])
    lines.append("")
    return lines


def _format_question_error(index: int, record: QuestionRecord, exc: Exception) -> list[str]:
    return [
        f"## Question {index}",
        "",
        "### Original Question",
        "",
        record.question,
        "",
        "### Explicit Entities",
        "",
        "(not available: pipeline failed)",
        "",
        "### Step4 Token Reasoning Structure",
        "",
        f"ERROR: {type(exc).__name__}: {exc}",
        "",
    ]


def _format_step4_metadata(token_reasoning_structure: Any) -> list[str]:
    lines: list[str] = []
    if token_reasoning_structure.answer_anchor:
        lines.append(f"answer_anchor: {token_reasoning_structure.answer_anchor}")
    if token_reasoning_structure.entity_anchors:
        lines.append(f"entity_anchors: {', '.join(token_reasoning_structure.entity_anchors)}")
    if token_reasoning_structure.constraints:
        lines.append(f"constraints: {_format_constraints(token_reasoning_structure.constraints)}")
    if token_reasoning_structure.candidate_sets:
        lines.append(f"candidate_sets: {_format_candidate_sets(token_reasoning_structure.candidate_sets)}")
    debug_file = getattr(token_reasoning_structure, "debug_file", "")
    if debug_file:
        lines.append(f"Debug file: {debug_file}")
    return lines


def _format_constraints(constraints: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for constraint in constraints:
        text = str(constraint.get("text") or "")
        target = str(constraint.get("target") or "")
        constraint_type = str(constraint.get("type") or "")
        if target:
            rendered.append(f"{constraint_type}:{text}->{target}")
        else:
            rendered.append(f"{constraint_type}:{text}")
    return "; ".join(rendered)


def _format_candidate_sets(candidate_sets: list[list[str]]) -> str:
    return "; ".join(", ".join(candidate_set) for candidate_set in candidate_sets)


if __name__ == "__main__":
    raise SystemExit(main())
