from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DEPO parser preprocessing and CoreNLP dependency parsing over questions/*.json."
        )
    )
    parser.add_argument("--dataset", help="Dataset subdirectory under questions/, e.g. 2wikimultihopqa.")
    parser.add_argument("--questions-file", help="Specific questions.json path. Overrides --dataset.")
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Process every questions/*/questions.json file.",
    )
    parser.add_argument("--questions-root", default="questions", help="Root directory containing dataset folders.")
    parser.add_argument(
        "--output-root",
        default="runs/depo_decomposition",
        help="Root output directory for decomposition artifacts.",
    )
    parser.add_argument(
        "--run-id",
        help="Output run id under output-root/dataset/. Defaults to current timestamp.",
    )
    parser.add_argument("--limit", type=int, help="Maximum number of questions per dataset.")
    parser.add_argument("--start", type=int, default=1, help="1-based start index within each questions file.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a question if its decomposition.json already exists.",
    )
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model used by DEPO stages.")
    parser.add_argument(
        "--corenlp-url",
        default="http://localhost:9000",
        help="Endpoint used by the managed CoreNLP server.",
    )
    parser.add_argument("--corenlp-memory", default="4G", help="Java heap memory for managed CoreNLP.")
    parser.add_argument("--corenlp-home", help="Stanford CoreNLP directory containing jar files.")
    parser.add_argument(
        "--corenlp-timeout-ms",
        type=int,
        default=60000,
        help="CoreNLP annotation timeout in milliseconds.",
    )
    parser.add_argument("--debug", action="store_true", help="Keep full verbose fields in outputs.")
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
        print("No questions.json files found.", file=sys.stderr)
        return 2

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)

    try:
        from corenlp_parser import CoreNLPParser
        from hanlp_sdp_preprocessor import HanLPSDPPreprocessor
        from llm_client import LLMClient
        from main import run_corenlp_dependency_pipeline
        from models import QuestionRecord

        llm_client = LLMClient(api_key=api_key, base_url=base_url, model=args.model)
        preprocessor = HanLPSDPPreprocessor(llm_client)

        with CoreNLPParser(
            args.corenlp_url,
            timeout_ms=args.corenlp_timeout_ms,
            memory=args.corenlp_memory,
            corenlp_home=args.corenlp_home,
        ) as parser:
            for questions_file in question_files:
                dataset = questions_file.parent.name
                dataset_output_dir = output_root / dataset / run_id
                dataset_output_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = dataset_output_dir / "manifest.jsonl"
                question_items = _read_question_items(questions_file)
                selected_items = _slice_items(question_items, start=args.start, limit=args.limit)
                print(
                    f"Running DEPO decomposition: dataset={dataset}, "
                    f"questions={len(selected_items)}, output={dataset_output_dir}"
                )

                with manifest_path.open("a", encoding="utf-8") as manifest:
                    for item in selected_items:
                        record = QuestionRecord(question=item["question"], qid=item.get("qid"))
                        question_dir = dataset_output_dir / _question_dir_name(item["index"], record.qid, record.question)
                        decomposition_path = question_dir / "decomposition.json"
                        if args.resume and decomposition_path.exists():
                            print(f"[skip] {dataset} #{item['index']} {record.question}")
                            manifest.write(
                                json.dumps(
                                    {
                                        "dataset": dataset,
                                        "index": item["index"],
                                        "qid": record.qid,
                                        "question": record.question,
                                        "status": "skipped",
                                        "output_dir": str(question_dir),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            manifest.flush()
                            continue

                        question_dir.mkdir(parents=True, exist_ok=True)
                        print(f"[run] {dataset} #{item['index']} {record.question}")
                        try:
                            result = run_corenlp_dependency_pipeline(
                                record=record,
                                index=item["index"],
                                preprocessor=preprocessor,
                                parser=parser,
                                debug=args.debug,
                            )
                            payload = build_decomposition_payload(
                                dataset=dataset,
                                questions_file=questions_file,
                                item=item,
                                result=result,
                                debug=args.debug,
                            )
                            _write_json(decomposition_path, payload)
                            (question_dir / "decomposition.md").write_text(
                                build_markdown_report(payload),
                                encoding="utf-8",
                            )
                            manifest_item = _manifest_item(payload, question_dir)
                            print(f"[ok]  {dataset} #{item['index']} -> {question_dir}")
                        except Exception as exc:  # Keep batch jobs moving across bad items.
                            payload = build_error_payload(dataset, questions_file, item, exc)
                            _write_json(question_dir / "error.json", payload)
                            (question_dir / "error.md").write_text(build_error_markdown(payload), encoding="utf-8")
                            manifest_item = {
                                "dataset": dataset,
                                "index": item["index"],
                                "qid": item.get("qid"),
                                "question": item["question"],
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "output_dir": str(question_dir),
                            }
                            print(f"[err] {dataset} #{item['index']} {type(exc).__name__}: {exc}")

                        manifest.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
                        manifest.flush()
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


def build_decomposition_payload(
    *,
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    result: dict[str, Any],
    debug: bool,
) -> dict[str, Any]:
    del debug
    preprocess_result = result["preprocess_result"]
    dependency_parse = result["dependency_parse"]

    return {
        "status": "ok",
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "raw_question_item": item.get("raw"),
        "gold_answer": item.get("answer"),
        "stages": {
            "1_explicit_entities": _dataclass_to_jsonable(preprocess_result.explicit_entities),
            "2_entity_masking": {
                "masked_question": preprocess_result.masked_question,
                "corenlp_input_sentence": result.get("corenlp_input_sentence") or preprocess_result.masked_question,
                "mask_mappings": [_dataclass_to_jsonable(mapping) for mapping in preprocess_result.mask_mappings],
                "warnings": list(preprocess_result.warnings),
            },
            "3_corenlp_dependency_parse": {
                "tokens": [_dataclass_to_jsonable(token) for token in dependency_parse.tokens],
                "edges": [_dataclass_to_jsonable(edge) for edge in dependency_parse.edges],
                "edge_display": [edge.display() for edge in dependency_parse.edges],
            },
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
    lines: list[str] = []
    lines.append(f"# DEPO CoreNLP Dependency Parse #{payload['index']}")
    lines.append("")
    lines.append(f"- Dataset: `{payload['dataset']}`")
    if payload.get("qid"):
        lines.append(f"- QID: `{payload['qid']}`")
    lines.append(f"- Question: {payload['question']}")
    if payload.get("gold_answer") is not None:
        lines.append(f"- Gold answer: {payload['gold_answer']}")
    lines.append("")

    lines.append("## 2. Explicit Entities")
    explicit_entities = (stages.get("1_explicit_entities") or {}).get("entities", [])
    if explicit_entities:
        for entity in explicit_entities:
            lines.append(
                f"- {entity.get('text')} ({entity.get('semantic_type_hint')}) "
                f"span=({entity.get('start_char')}, {entity.get('end_char')})"
            )
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## 2. Entity Masking")
    masking = stages.get("2_entity_masking") or {}
    for mapping in masking.get("mask_mappings", []):
        lines.append(f"- {mapping.get('placeholder')} -> {mapping.get('original_text')}")
    if not masking.get("mask_mappings"):
        lines.append("(none)")
    lines.append("")
    lines.append(f"Masked question: {masking.get('masked_question', '')}")
    lines.append(f"CoreNLP input sentence: {masking.get('corenlp_input_sentence', masking.get('masked_question', ''))}")
    for warning in masking.get("warnings", []) or []:
        lines.append(f"- warning: {warning}")
    lines.append("")

    lines.append("## 3. CoreNLP Dependency Parse")
    for edge in stages["3_corenlp_dependency_parse"]["edge_display"]:
        lines.append(f"- {edge}")
    if not stages["3_corenlp_dependency_parse"]["edge_display"]:
        lines.append("(none)")
    lines.append("")
    return "\n".join(lines)


def build_error_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# DEPO Decomposition Error #{payload['index']}",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Question: {payload['question']}",
    ]
    if payload.get("gold_answer") is not None:
        lines.append(f"- Gold answer: {payload['gold_answer']}")
    lines.extend(
        [
            f"- Error type: `{payload['error_type']}`",
            "",
            "```text",
            str(payload["error"]),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_question_files(args: argparse.Namespace) -> list[Path]:
    if args.questions_file:
        return [Path(args.questions_file)]
    questions_root = Path(args.questions_root)
    if args.dataset:
        return [questions_root / args.dataset / "questions.json"]
    if args.all_datasets:
        return sorted(questions_root.glob("*/questions.json"))
    raise ValueError("Specify --dataset, --questions-file, or --all-datasets.")


def _read_question_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")

    items: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if isinstance(item, str):
            question = item.strip()
            raw = item
            qid = None
            answer = None
        elif isinstance(item, dict):
            question = str(item.get("question", "")).strip()
            raw = item
            qid_value = item.get("id", item.get("qid"))
            qid = str(qid_value) if qid_value is not None else None
            answer = item.get("answer")
        else:
            raise ValueError(f"Unsupported question item at index {index}: {item!r}")
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


def _slice_items(items: list[dict[str, Any]], *, start: int, limit: int | None) -> list[dict[str, Any]]:
    if start < 1:
        raise ValueError("--start must be >= 1.")
    selected = items[start - 1 :]
    return selected[:limit] if limit is not None else selected


def _question_dir_name(index: int, qid: str | None, question: str) -> str:
    prefix = f"{index:05d}"
    if qid:
        prefix += f"_{_slug(qid, max_len=48)}"
    return f"{prefix}_{_slug(question, max_len=80)}"


def _slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return (slug[:max_len].strip("-") or "question")


def _dataclass_to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return _dataclass_to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _dataclass_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_dataclass_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _manifest_item(payload: dict[str, Any], question_dir: Path) -> dict[str, Any]:
    stages = payload["stages"]
    dependency_parse = stages.get("3_corenlp_dependency_parse") or {}
    return {
        "dataset": payload["dataset"],
        "index": payload["index"],
        "qid": payload.get("qid"),
        "question": payload["question"],
        "gold_answer": payload.get("gold_answer"),
        "status": "ok",
        "dependency_edge_count": len(dependency_parse.get("edges", [])),
        "output_dir": str(question_dir),
    }


if __name__ == "__main__":
    raise SystemExit(main())
