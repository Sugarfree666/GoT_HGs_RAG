from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METHOD = "depo_step1_entity_recognition"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DEPO Step1 entity recognition and question normalization only. "
            "This script does not run HanLP, graph repair, path search, or Step5."
        )
    )
    parser.add_argument("--dataset", help="Dataset subdirectory under questions/, e.g. 2wikimultihopqa.")
    parser.add_argument("--questions-file", help="Specific questions JSON or JSONL file. Overrides --dataset.")
    parser.add_argument("--all-datasets", action="store_true", help="Process every questions/*/questions.json file.")
    parser.add_argument("--questions-root", default="questions", help="Root directory containing dataset folders.")
    parser.add_argument(
        "--output-root",
        default="runs/depo_entity_recognition",
        help="Root directory for JSONL and Markdown entity-recognition reports.",
    )
    parser.add_argument("--run-id", help="Output run id under output-root/dataset/. Defaults to current timestamp.")
    parser.add_argument("--start", type=int, default=1, help="1-based inclusive start index in each input file.")
    parser.add_argument("--end", type=int, help="1-based inclusive end index in each input file.")
    parser.add_argument("--limit", type=int, help="Maximum questions after applying --start/--end.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model used by DEPO Step1.")
    parser.add_argument("--quiet", action="store_true", help="Only print per-question progress and output locations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    try:
        question_files = _resolve_question_files(args)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not question_files:
        print("No questions files found.", file=sys.stderr)
        return 2

    try:
        from llm_client import LLMClient
        from mask_span_extractor import ExplicitEntityExtractor
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = _repo_path(args.output_root)
    extractor = ExplicitEntityExtractor(
        LLMClient(api_key=api_key, base_url=base_url, model=args.llm_model)
    )

    for questions_file in question_files:
        dataset = _dataset_name(questions_file)
        try:
            items = _slice_items(
                _read_question_items(questions_file),
                start=args.start,
                end=args.end,
                limit=args.limit,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"{dataset}: {exc}", file=sys.stderr)
            return 2
        if not items:
            print(f"No questions selected for {dataset}.", file=sys.stderr)
            continue

        output_dir = output_root / dataset / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "entity_recognition.jsonl"
        report_path = output_dir / "entity_recognition.md"
        report_lines = _report_header(
            dataset=dataset,
            questions_file=questions_file,
            run_id=run_id,
            start=args.start,
            end=args.end,
            limit=args.limit,
        )

        print(
            f"Running DEPO Step1 entity recognition: dataset={dataset}, "
            f"questions={len(items)}, output={output_dir}"
        )
        with results_path.open("w", encoding="utf-8") as results_file:
            for offset, item in enumerate(items, start=1):
                print(f"[run {offset}/{len(items)}] {dataset} #{item['index']} {item['question']}")
                try:
                    result = extractor.extract(item["question"])
                    payload = build_result_payload(
                        dataset=dataset,
                        questions_file=questions_file,
                        item=item,
                        result=result,
                    )
                except Exception as exc:  # Keep batch evaluation inspectable after individual failures.
                    payload = build_error_payload(
                        dataset=dataset,
                        questions_file=questions_file,
                        item=item,
                        exc=exc,
                    )

                results_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                results_file.flush()
                report_lines.extend(build_markdown_result(payload))
                report_lines.append("")
                if not args.quiet:
                    _print_result(payload)

        report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
        print(f"Entity-recognition JSONL written to {results_path}")
        print(f"Entity-recognition Markdown report written to {report_path}")

    return 0


def build_result_payload(
    *,
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    question = item["question"]
    entities = [_entity_payload(question, entity) for entity in result.entities]
    return {
        "method": METHOD,
        "status": "ok",
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": question,
        "gold_answer": item.get("answer"),
        "explicit_entities": entities,
        "normalized_question": result.normalized_question or question,
        "normalization_changed": bool(result.normalization_changed),
        "normalization_note": result.normalization_note,
        "warnings": list(result.warnings),
        "raw_llm_payload": result.raw_payload,
    }


def build_error_payload(
    *,
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "method": METHOD,
        "status": "error",
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "gold_answer": item.get("answer"),
        "explicit_entities": [],
        "normalized_question": item["question"],
        "normalization_changed": False,
        "normalization_note": "",
        "warnings": [f"Entity recognition failed: {type(exc).__name__}: {exc}"],
        "raw_llm_payload": None,
    }


def _entity_payload(question: str, entity: Any) -> dict[str, Any]:
    surface = str(entity.text)
    matched_spans = [
        {"start_char": match.start(), "end_char": match.end()}
        for match in re.finditer(re.escape(surface), question)
    ]
    return {
        "surface": surface,
        "type": entity.semantic_type_hint or "Entity",
        "start_char": entity.start_char,
        "end_char": entity.end_char,
        "matched_spans": matched_spans,
        "reason": entity.reason,
    }


def build_markdown_result(payload: dict[str, Any]) -> list[str]:
    title = f"## Question {payload['index']}"
    if payload.get("qid"):
        title += f" ({payload['qid']})"
    lines = [title, "", "### Original Question", "", str(payload["question"]), ""]
    if payload["status"] == "error":
        lines.extend(["### Error", "", *[f"- {warning}" for warning in payload["warnings"]]])
        return lines

    lines.extend(["### Explicit Entities", ""])
    entities = payload["explicit_entities"]
    if entities:
        for entity in entities:
            spans = ", ".join(
                f"[{span['start_char']}:{span['end_char']}]"
                for span in entity["matched_spans"]
            )
            lines.append(f"- `{entity['surface']}` ({entity['type']}; {spans})")
    else:
        lines.append("(none)")

    lines.extend(
        [
            "",
            "### Normalized Question",
            "",
            str(payload["normalized_question"]),
            "",
            f"- Changed: `{payload['normalization_changed']}`",
        ]
    )
    if payload.get("normalization_note"):
        lines.append(f"- Note: {payload['normalization_note']}")
    lines.extend(["", "### Warnings", ""])
    warnings = payload.get("warnings") or []
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("(none)")
    return lines


def _print_result(payload: dict[str, Any]) -> None:
    if payload["status"] == "error":
        print(f"  error: {payload['warnings'][0]}")
        return
    entities = payload["explicit_entities"]
    if entities:
        for entity in entities:
            spans = ", ".join(
                f"[{span['start_char']}:{span['end_char']}]"
                for span in entity["matched_spans"]
            )
            print(f"  entity: {entity['surface']!r} type={entity['type']} spans={spans}")
    else:
        print("  entities: (none)")
    print(f"  normalized: {payload['normalized_question']}")
    print(f"  normalization_changed: {payload['normalization_changed']}")
    for warning in payload.get("warnings") or []:
        print(f"  warning: {warning}")


def _report_header(
    *,
    dataset: str,
    questions_file: Path,
    run_id: str,
    start: int,
    end: int | None,
    limit: int | None,
) -> list[str]:
    lines = [
        f"# DEPO Step1 Entity Recognition: {dataset}",
        "",
        f"- Run id: `{run_id}`",
        f"- Questions file: `{questions_file}`",
        f"- Range: `{start}-{end if end is not None else 'end'}`",
        "- Scope: explicit entity recognition and normalized_question only",
        "",
    ]
    if limit is not None:
        lines.insert(-1, f"- Limit: `{limit}`")
    return lines


def _resolve_question_files(args: argparse.Namespace) -> list[Path]:
    if args.questions_file:
        return [_repo_path(args.questions_file)]
    questions_root = _repo_path(args.questions_root)
    if args.dataset:
        path = questions_root / args.dataset / "questions.json"
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {path}")
        return [path]
    if args.all_datasets:
        return sorted(questions_root.glob("*/questions.json"))
    raise ValueError("Specify --dataset, --questions-file, or --all-datasets.")


def _read_question_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        raw_items = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw_items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_items, list):
            raise ValueError(f"{path} must contain a JSON list.")

    items: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_items, start=1):
        if isinstance(raw, str):
            question = raw.strip()
            qid = None
            answer = None
        elif isinstance(raw, dict):
            question = str(raw.get("question", raw.get("query", ""))).strip()
            qid_value = raw.get("id", raw.get("qid"))
            qid = str(qid_value) if qid_value is not None else None
            answer = raw.get("answer")
        else:
            raise ValueError(f"Unsupported question item at index {index}: {raw!r}")
        if not question:
            raise ValueError(f"Question at index {index} is empty.")
        items.append(
            {
                "index": index,
                "qid": qid,
                "question": question,
                "answer": answer,
                "raw": raw,
            }
        )
    return items


def _slice_items(
    items: list[dict[str, Any]],
    *,
    start: int,
    end: int | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if start < 1:
        raise ValueError("--start must be >= 1.")
    if end is not None and end < start:
        raise ValueError("--end must be >= --start.")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be >= 1.")
    selected = items[start - 1 : end]
    return selected[:limit] if limit is not None else selected


def _dataset_name(questions_file: Path) -> str:
    return questions_file.parent.name or questions_file.stem


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


if __name__ == "__main__":
    raise SystemExit(main())
