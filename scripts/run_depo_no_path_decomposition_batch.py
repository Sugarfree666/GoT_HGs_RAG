from __future__ import annotations

import argparse
import json
import os
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

from run_depo_decomposition_batch import (  # noqa: E402
    _dag_node_count,
    _dag_valid,
    _dataset_name,
    _manifest_item,
    _question_dir_name,
    _read_question_items,
    _repo_path,
    _resolve_question_files,
    _slice_items,
    _step5_actions,
    _summary_error_lines,
    _summary_question_lines,
    _write_json,
    build_error_markdown,
)


METHOD = "depo_no_path_atomic_dag"
STEP5_MODE = "no_path"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DEPO no-path ablation over questions datasets and save "
            "Step5 atomic decomposition artifacts."
        )
    )
    parser.add_argument("--dataset", help="Dataset subdirectory under questions/, e.g. 2wikimultihopqa.")
    parser.add_argument("--questions-file", help="Specific questions file path. Overrides --dataset.")
    parser.add_argument("--all-datasets", action="store_true", help="Process every questions/*/questions.json file.")
    parser.add_argument("--questions-root", default="questions", help="Root directory containing dataset folders.")
    parser.add_argument(
        "--output-root",
        default="runs/depo_no_path_decomposition",
        help="Root output directory for DEPO no-path decomposition artifacts.",
    )
    parser.add_argument("--run-id", help="Output run id under output-root/dataset/. Defaults to current timestamp.")
    parser.add_argument("--start", type=int, default=1, help="1-based inclusive start index in each input file.")
    parser.add_argument("--end", type=int, help="1-based inclusive end index in each input file.")
    parser.add_argument("--limit", type=int, help="Maximum number of questions after applying --start/--end.")
    parser.add_argument("--resume", action="store_true", help="Skip questions whose decomposition.json already exists.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model used by no-path DEPO Step5.")
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
        from atomic_question_dag import NoPathAtomicDAGGenerator
        from llm_client import LLMClient
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = _repo_path(args.output_root)

    llm_client = LLMClient(api_key=api_key, base_url=base_url, model=args.llm_model)
    generator = NoPathAtomicDAGGenerator(llm_client)

    for questions_file in question_files:
        dataset = _dataset_name(questions_file)
        dataset_output_dir = output_root / dataset / run_id
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_output_dir / "manifest.jsonl"
        summary_path = dataset_output_dir / "summary.md"

        items = _slice_items(_read_question_items(questions_file), start=args.start, end=args.end, limit=args.limit)
        print(
            f"Running DEPO no-path decomposition: dataset={dataset}, "
            f"questions={len(items)}, output={dataset_output_dir}"
        )

        summary_lines = _summary_header(
            dataset=dataset,
            questions_file=questions_file,
            run_id=run_id,
            start=args.start,
            end=args.end,
            limit=args.limit,
        )

        with manifest_path.open("a", encoding="utf-8") as manifest:
            for offset, item in enumerate(items, start=1):
                question_dir = dataset_output_dir / _question_dir_name(item["index"], item.get("qid"), item["question"])
                decomposition_path = question_dir / "decomposition.json"
                if args.resume and decomposition_path.exists():
                    print(f"[skip] {dataset} #{item['index']} {item['question']}")
                    manifest_item = {
                        "method": METHOD,
                        "dataset": dataset,
                        "index": item["index"],
                        "qid": item.get("qid"),
                        "question": item["question"],
                        "gold_answer": item.get("answer"),
                        "status": "skipped",
                        "dag_valid": None,
                        "dag_node_count": None,
                        "output_dir": str(question_dir),
                    }
                    manifest.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
                    manifest.flush()
                    continue

                question_dir.mkdir(parents=True, exist_ok=True)
                print(f"[run {offset}/{len(items)}] {dataset} #{item['index']} {item['question']}")
                try:
                    atomic_question_dag = generator.generate(original_question=item["question"])
                    payload = build_decomposition_payload(
                        dataset=dataset,
                        questions_file=questions_file,
                        item=item,
                        atomic_question_dag=atomic_question_dag,
                    )
                    _write_json(decomposition_path, payload)
                    (question_dir / "decomposition.md").write_text(build_markdown_report(payload), encoding="utf-8")
                    manifest_item = _manifest_item(payload, question_dir)
                    summary_lines.extend(_summary_question_lines(payload, question_dir))
                    print(
                        f"[ok]  {dataset} #{item['index']} "
                        f"nodes={_dag_node_count(payload)} valid={_dag_valid(payload)} -> {question_dir}"
                    )
                except Exception as exc:  # Keep long batch jobs inspectable even if one item fails.
                    payload = build_error_payload(dataset, questions_file, item, exc)
                    _write_json(question_dir / "error.json", payload)
                    (question_dir / "error.md").write_text(build_error_markdown(payload), encoding="utf-8")
                    manifest_item = {
                        "method": METHOD,
                        "dataset": dataset,
                        "index": item["index"],
                        "qid": item.get("qid"),
                        "question": item["question"],
                        "gold_answer": item.get("answer"),
                        "status": "error",
                        "dag_valid": None,
                        "dag_node_count": None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "output_dir": str(question_dir),
                    }
                    summary_lines.extend(_summary_error_lines(payload, question_dir))
                    print(f"[err] {dataset} #{item['index']} {type(exc).__name__}: {exc}")

                manifest.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
                manifest.flush()
                summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")

        summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")
        print(f"Summary written to {summary_path}")

    return 0


def build_decomposition_payload(
    *,
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    atomic_question_dag: Any,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "method": METHOD,
        "step5_mode": STEP5_MODE,
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "raw_question_item": item.get("raw"),
        "gold_answer": item.get("answer"),
        "stages": {
            "5_step5_action_trace": {
                "input": {
                    "original_question": item["question"],
                },
                "actions": _step5_actions(atomic_question_dag),
            },
            "6_atomic_question_dag": atomic_question_dag.to_dict() if atomic_question_dag is not None else None,
        },
    }


def build_error_payload(
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "status": "error",
        "method": METHOD,
        "step5_mode": STEP5_MODE,
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "raw_question_item": item.get("raw"),
        "gold_answer": item.get("answer"),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    stages = payload["stages"]
    dag = stages.get("6_atomic_question_dag") or {}
    action_trace = stages.get("5_step5_action_trace") or {}
    lines: list[str] = [
        f"# DEPO No-Path Decomposition #{payload['index']}",
        "",
        f"- Dataset: `{payload['dataset']}`",
    ]
    if payload.get("qid"):
        lines.append(f"- QID: `{payload['qid']}`")
    lines.append(f"- Question: {payload['question']}")
    if payload.get("gold_answer") is not None:
        lines.append(f"- Gold answer: {payload['gold_answer']}")
    lines.append(f"- Step5 mode: `{payload.get('step5_mode', STEP5_MODE)}`")
    lines.append("")

    lines.append("## 1. Step5 Action Trace")
    actions = action_trace.get("actions") or []
    if actions:
        for action in actions:
            lines.append(f"- {action.get('id')}: {action.get('question')}")
            consume = action.get("consume") or []
            lines.append(f"  - consume: {' ---- '.join(consume) if consume else '(none)'}")
            lines.append(f"  - produce: {action.get('produce') or ''}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## 2. Atomic Question DAG")
    if dag is None:
        lines.append("(not generated)")
    elif not dag.get("valid", False):
        lines.append("Invalid DAG")
        for error in dag.get("validation_errors", []) or []:
            lines.append(f"- {error}")
    else:
        for node in dag.get("nodes", []) or []:
            lines.append(f"- {node.get('id')}: {node.get('question')}")
            depends_on = node.get("depends_on") or []
            lines.append(f"  - depends_on: {', '.join(depends_on) if depends_on else '(none)'}")
    lines.append("")
    return "\n".join(lines)


def _summary_header(
    *,
    dataset: str,
    questions_file: Path,
    run_id: str,
    start: int,
    end: int | None,
    limit: int | None,
) -> list[str]:
    lines = [
        f"# DEPO No-Path Decomposition Run: {dataset}",
        "",
        f"- Run id: `{run_id}`",
        f"- Questions file: `{questions_file}`",
        f"- Range: `{start}-{end if end is not None else 'end'}`",
    ]
    if limit is not None:
        lines.append(f"- Limit: `{limit}`")
    lines.append("")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
