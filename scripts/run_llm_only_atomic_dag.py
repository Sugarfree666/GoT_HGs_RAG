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


METHOD = "llm_only_atomic_question_dag"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DEPO Step5 atomic-question DAGs using only an original question "
            "and supplied topic entities. HanLP, Step4 paths, and the rest of DEPO are not run."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="One original question to decompose.")
    source.add_argument(
        "--input-file",
        help=(
            "JSON or JSONL records containing question plus topic_entities, explicit_entities, "
            "or entities. This accepts Step1 entity_recognition.jsonl output."
        ),
    )
    parser.add_argument(
        "--entity",
        action="append",
        default=[],
        help="Topic entity for --question mode. Repeat this option for multiple entities.",
    )
    parser.add_argument(
        "--entities-json",
        help='JSON list of topic entities for --question mode, for example: ["Film A", "Film B"].',
    )
    parser.add_argument("--start", type=int, default=1, help="1-based inclusive input-file start index.")
    parser.add_argument("--end", type=int, help="1-based inclusive input-file end index.")
    parser.add_argument("--limit", type=int, help="Maximum records after applying --start/--end.")
    parser.add_argument("--output", help="Write the single-question result JSON to this path.")
    parser.add_argument(
        "--output-dir",
        help=(
            "Write input-file results.jsonl and summary.md here. Defaults to "
            "runs/llm_only_atomic_dag/<input-file-stem>/<timestamp>."
        ),
    )
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model used for the decomposition.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        _validate_mode_args(args)
        api_key = os.getenv("OPENAI_API_KEY") or args.api_key
        base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

        from atomic_question_dag import LLMOnlyAtomicDAGGenerator
        from llm_client import LLMClient
    except (ModuleNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    generator = LLMOnlyAtomicDAGGenerator(
        LLMClient(api_key=api_key, base_url=base_url, model=args.model)
    )
    if args.question is not None:
        return _run_single_question(args, generator)
    return _run_input_file(args, generator)


def _validate_mode_args(args: argparse.Namespace) -> None:
    if args.question is not None:
        if args.entities_json and args.entity:
            raise ValueError("Use either --entity or --entities-json, not both.")
        if args.output_dir:
            raise ValueError("--output-dir is only valid with --input-file.")
        return

    if args.entity or args.entities_json:
        raise ValueError("--entity and --entities-json are only valid with --question.")
    if args.output:
        raise ValueError("--output is only valid with --question.")


def _run_single_question(args: argparse.Namespace, generator: Any) -> int:
    try:
        topic_entities = _single_question_entities(args)
        result = generator.generate(
            original_question=args.question,
            explicit_entities=topic_entities,
        )
        payload = _result_payload(
            index=1,
            question=args.question,
            topic_entities=topic_entities,
            result=result,
        )
    except Exception as exc:
        payload = _error_payload(index=1, question=args.question, topic_entities=[], exc=exc)

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = _repo_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Result written to {output_path}", file=sys.stderr)
    return 0 if payload["status"] == "ok" else 1


def _run_input_file(args: argparse.Namespace, generator: Any) -> int:
    try:
        input_path = _repo_path(args.input_file)
        records = _slice_records(
            _read_input_records(input_path),
            start=args.start,
            end=args.end,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_dir = (
        _repo_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / "runs"
        / "llm_only_atomic_dag"
        / input_path.stem
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.md"
    summary_lines = [
        "# LLM-only Atomic Question DAG Run",
        "",
        f"- Input: `{input_path}`",
        f"- Records: `{len(records)}`",
        f"- Model: `{args.model}`",
        "",
    ]
    failures = 0
    with results_path.open("w", encoding="utf-8") as results_file:
        for offset, record in enumerate(records, start=1):
            print(f"[run {offset}/{len(records)}] #{record['index']} {record['question']}")
            try:
                result = generator.generate(
                    original_question=record["question"],
                    explicit_entities=record["topic_entities"],
                )
                payload = _result_payload(
                    index=record["index"],
                    question=record["question"],
                    topic_entities=record["topic_entities"],
                    result=result,
                    qid=record.get("qid"),
                )
            except Exception as exc:  # Keep long experiment runs inspectable after one failed call.
                failures += 1
                payload = _error_payload(
                    index=record["index"],
                    question=record["question"],
                    topic_entities=record["topic_entities"],
                    exc=exc,
                    qid=record.get("qid"),
                )

            results_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            results_file.flush()
            summary_lines.extend(_summary_lines(payload))

    summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Results written to {results_path}")
    print(f"Summary written to {summary_path}")
    return 1 if failures else 0


def _single_question_entities(args: argparse.Namespace) -> list[str]:
    if not args.entities_json:
        return list(args.entity)
    try:
        value = json.loads(args.entities_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--entities-json must be a JSON list: {exc.msg}") from exc
    if not isinstance(value, list) or any(not isinstance(entity, str) for entity in value):
        raise ValueError("--entities-json must be a JSON list of strings.")
    return value


def _read_input_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.casefold() == ".jsonl":
        raw_records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw_payload, list):
            raw_records = raw_payload
        elif isinstance(raw_payload, dict):
            raw_records = raw_payload.get("records", raw_payload.get("items", [raw_payload]))
        else:
            raise ValueError(f"{path} must contain a JSON object, JSON list, or JSONL records.")

    if not isinstance(raw_records, list):
        raise ValueError(f"{path} records/items must be a list.")

    records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records, start=1):
        if not isinstance(raw_record, dict):
            raise ValueError(f"Input record {index} must be a JSON object.")
        question = str(raw_record.get("original_question", raw_record.get("question", ""))).strip()
        if not question:
            raise ValueError(f"Input record {index} has no non-empty question/original_question.")
        records.append(
            {
                "index": index,
                "qid": _record_qid(raw_record),
                "question": question,
                "topic_entities": _record_topic_entities(raw_record, index),
            }
        )
    return records


def _record_qid(record: dict[str, Any]) -> str | None:
    value = record.get("qid", record.get("id"))
    return str(value) if value is not None else None


def _record_topic_entities(record: dict[str, Any], index: int) -> list[str]:
    value = record.get("topic_entities")
    if value is None:
        value = record.get("explicit_entities")
    if value is None:
        value = record.get("entities", [])
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"Input record {index} topic entities must be a list or string.")

    entities: list[str] = []
    for entity_index, entity in enumerate(value, start=1):
        if isinstance(entity, str):
            entities.append(entity)
            continue
        if isinstance(entity, dict):
            surface = entity.get("surface", entity.get("text", entity.get("original_text")))
            if isinstance(surface, str):
                entities.append(surface)
                continue
        raise ValueError(
            f"Input record {index} entity {entity_index} must be a string or an object with surface/text."
        )
    return entities


def _slice_records(
    records: list[dict[str, Any]],
    *,
    start: int,
    end: int | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if start < 1:
        raise ValueError("--start must be >= 1.")
    if end is not None and end < start:
        raise ValueError("--end must be >= --start.")
    selected = records[start - 1 : end]
    return selected[:limit] if limit is not None else selected


def _result_payload(
    *,
    index: int,
    question: str,
    topic_entities: list[str],
    result: Any,
    qid: str | None = None,
) -> dict[str, Any]:
    return {
        "method": METHOD,
        "status": "ok",
        "index": index,
        "qid": qid,
        "original_question": question,
        "topic_entities": list(topic_entities),
        "atomic_question_dag": result.to_dict(),
    }


def _error_payload(
    *,
    index: int,
    question: str,
    topic_entities: list[str],
    exc: Exception,
    qid: str | None = None,
) -> dict[str, Any]:
    return {
        "method": METHOD,
        "status": "error",
        "index": index,
        "qid": qid,
        "original_question": question,
        "topic_entities": list(topic_entities),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _summary_lines(payload: dict[str, Any]) -> list[str]:
    title = f"## {payload['index']}. {payload['original_question']}"
    lines = [title, "", f"- Topic entities: {json.dumps(payload['topic_entities'], ensure_ascii=False)}"]
    if payload["status"] == "error":
        return [*lines, f"- Error: `{payload['error_type']}: {payload['error']}`", ""]

    dag = payload["atomic_question_dag"]
    lines.extend(
        [
            f"- DAG valid: `{dag['valid']}`",
            f"- DAG nodes: `{len(dag['nodes'])}`",
            "",
        ]
    )
    lines.extend(f"- {node['id']}: {node['question']}" for node in dag["nodes"])
    return [*lines, ""]


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


if __name__ == "__main__":
    raise SystemExit(main())
