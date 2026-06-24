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


DIRECT_LLM_DECOMPOSITION_SYSTEM = """
You are an expert question decomposition module.

Your task is to convert a complex question directly into an Atomic Question DAG.

You are given:

* the original question

Use the original question as the authoritative source of meaning.
Do not rely on any external knowledge.
Do not answer the question.

Atomic question definition:
An atomic question asks for one missing answer using one semantic operation. It should be directly answerable once its dependencies are resolved. It must not contain an unresolved nested relation that should be asked first.

Core principles:

1. Generate the complete set of atomic questions needed to answer the original question.
2. Preserve all entities, constraints, comparison conditions, and answer intent from the original question.
3. Use natural questions, not symbolic triples.
4. If a question depends on a previous answer, refer to it as q1's answer, q2's answer, etc.
5. Every dependency mentioned in question text must also appear in depends_on.
6. Do not answer the questions.
7. Do not invent entities, relations, dates, or constraints not present in the original question.
8. Do not leave unresolved placeholders such as ENTITYA or ENTITYB if the original question contains the restored entity names.
9. Since no token path evidence is provided, set support to null for every node.
10. Return valid JSON only.

Output format:
{
"nodes": [
{
"id": "q1",
"question": "atomic question?",
"depends_on": [],
"support": null
}
]
}

Example 1 input:
{
"original_question": "What nationality is the performer of song When The Stars Go Blue?"
}

Example 1 output:
{
"nodes": [
{
"id": "q1",
"question": "Who is the performer of When The Stars Go Blue?",
"depends_on": [],
"support": null
},
{
"id": "q2",
"question": "What is the nationality of q1's answer?",
"depends_on": ["q1"],
"support": null
}
]
}

Example 2 input:
{
"original_question": "Which country is the composer of film Thunder On The Hill from?"
}

Example 2 output:
{
"nodes": [
{
"id": "q1",
"question": "Who is the composer of Thunder On The Hill?",
"depends_on": [],
"support": null
},
{
"id": "q2",
"question": "Which country is q1's answer from?",
"depends_on": ["q1"],
"support": null
}
]
}

Example 3 input:
{
"original_question": "Which film whose director is younger, Dangerously They Live or Salad By The Roots?"
}

Example 3 output:
{
"nodes": [
{
"id": "q1",
"question": "Who directed Dangerously They Live?",
"depends_on": [],
"support": null
},
{
"id": "q2",
"question": "When was q1's answer born?",
"depends_on": ["q1"],
"support": null
},
{
"id": "q3",
"question": "Who directed Salad By The Roots?",
"depends_on": [],
"support": null
},
{
"id": "q4",
"question": "When was q3's answer born?",
"depends_on": ["q3"],
"support": null
},
{
"id": "q5",
"question": "Which film has the younger director, Dangerously They Live or Salad By The Roots, based on q2's answer and q4's answer?",
"depends_on": ["q2", "q4"],
"support": null
}
]
}

Now generate the Atomic Question DAG for the given input JSON.
Return only the JSON object.

""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a direct-LLM Atomic Subquestion DAG decomposition baseline over "
            "questions/*.json and save per-question artifacts."
        )
    )
    parser.add_argument("--dataset", help="Dataset subdirectory under questions/, e.g. 2wikimultihopqa.")
    parser.add_argument("--questions-file", help="Specific questions.json path. Overrides --dataset.")
    parser.add_argument("--all-datasets", action="store_true", help="Process every questions/*/questions.json file.")
    parser.add_argument("--questions-root", default="questions", help="Root directory containing dataset folders.")
    parser.add_argument(
        "--output-root",
        default="runs/llm_decomposition",
        help="Root output directory for direct-LLM decomposition artifacts.",
    )
    parser.add_argument("--run-id", help="Output run id under output-root/dataset/. Defaults to current timestamp.")
    parser.add_argument("--limit", type=int, help="Maximum number of questions per dataset.")
    parser.add_argument("--start", type=int, default=1, help="1-based start index within each questions file.")
    parser.add_argument("--resume", action="store_true", help="Skip questions with existing decomposition.json.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model used for direct decomposition.")
    parser.add_argument("--max-retries", type=int, default=3, help="JSON retry count for each LLM call.")
    parser.add_argument("--debug", action="store_true", help="Keep raw LLM payloads in outputs.")
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

    try:
        from llm_client import LLMClient
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2

    llm_client = LLMClient(api_key=api_key, base_url=base_url, model=args.model)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)

    for questions_file in question_files:
        dataset = questions_file.parent.name
        dataset_output_dir = output_root / dataset / run_id
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_output_dir / "manifest.jsonl"
        question_items = _slice_items(_read_question_items(questions_file), start=args.start, limit=args.limit)
        print(
            f"Running direct LLM decomposition: dataset={dataset}, "
            f"questions={len(question_items)}, output={dataset_output_dir}"
        )

        with manifest_path.open("a", encoding="utf-8") as manifest:
            for item in question_items:
                question_dir = dataset_output_dir / _question_dir_name(item["index"], item.get("qid"), item["question"])
                decomposition_path = question_dir / "decomposition.json"
                if args.resume and decomposition_path.exists():
                    print(f"[skip] {dataset} #{item['index']} {item['question']}")
                    manifest.write(
                        json.dumps(
                            {
                                "method": "direct_llm_atomic_dag",
                                "dataset": dataset,
                                "index": item["index"],
                                "qid": item.get("qid"),
                                "question": item["question"],
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
                print(f"[run] {dataset} #{item['index']} {item['question']}")
                try:
                    dag, raw_payload, warnings = decompose_question_direct(
                        llm_client,
                        item["question"],
                        max_retries=args.max_retries,
                    )
                    payload = build_decomposition_payload(
                        dataset=dataset,
                        questions_file=questions_file,
                        item=item,
                        dag=dag,
                        raw_payload=raw_payload,
                        warnings=warnings,
                        debug=args.debug,
                    )
                    _write_json(decomposition_path, payload)
                    _write_json(question_dir / "llm_dag_raw.json", dag)
                    (question_dir / "decomposition.md").write_text(build_markdown_report(payload), encoding="utf-8")
                    manifest_item = _manifest_item(payload, question_dir)
                    print(f"[ok]  {dataset} #{item['index']} nodes={len(dag.get('nodes', []))} -> {question_dir}")
                except Exception as exc:
                    payload = build_error_payload(dataset, questions_file, item, exc)
                    _write_json(question_dir / "error.json", payload)
                    (question_dir / "error.md").write_text(build_error_markdown(payload), encoding="utf-8")
                    manifest_item = {
                        "method": "direct_llm_atomic_dag",
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

    return 0


def decompose_question_direct(
    llm_client: Any,
    question: str,
    *,
    max_retries: int = 3,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    payload = llm_client.chat_json(
        DIRECT_LLM_DECOMPOSITION_SYSTEM,
        build_direct_decomposition_prompt(question),
        max_retries=max_retries,
    )
    dag, warnings = normalize_atomic_dag_payload(payload)
    return dag, payload, warnings


def build_direct_decomposition_prompt(question: str) -> str:
    payload = {"original_question": question}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_atomic_dag_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    raw_nodes = payload.get("nodes")
    if raw_nodes is None:
        raw_nodes = payload.get("atomic_questions") or payload.get("subquestions")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("Direct LLM decomposition payload must contain a non-empty nodes list.")

    warnings: list[str] = []
    nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, raw_node in enumerate(raw_nodes, start=1):
        if not isinstance(raw_node, dict):
            warnings.append(f"Dropped non-object node at position {index}.")
            continue
        node_id = str(raw_node.get("node_id") or raw_node.get("id") or f"q{index}").strip()
        if not re.fullmatch(r"q\d+", node_id):
            warnings.append(f"Renamed invalid node_id {node_id!r} to q{index}.")
            node_id = f"q{index}"
        if node_id in seen_ids:
            replacement = f"q{index}"
            warnings.append(f"Renamed duplicate node_id {node_id!r} to {replacement}.")
            node_id = replacement
        seen_ids.add(node_id)

        question = str(raw_node.get("question") or raw_node.get("subquestion") or "").strip()
        if not question:
            warnings.append(f"Dropped node {node_id} because question is empty.")
            seen_ids.remove(node_id)
            continue
        dependencies = _normalize_dependencies(raw_node.get("dependencies") or raw_node.get("depends_on"), seen_ids, node_id, warnings)
        nodes.append({"node_id": node_id, "question": question, "dependencies": dependencies})

    if not nodes:
        raise ValueError("Direct LLM decomposition produced no usable atomic question nodes.")
    return {"nodes": nodes, "warnings": warnings} if warnings else {"nodes": nodes}, warnings


def build_decomposition_payload(
    *,
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    dag: dict[str, Any],
    raw_payload: dict[str, Any],
    warnings: list[str],
    debug: bool,
) -> dict[str, Any]:
    stages: dict[str, Any] = {
        "1_original_question": item["question"],
        "2_direct_llm_atomic_subquestion_dag": dag,
    }
    if debug:
        stages["raw_llm_payload"] = raw_payload
    if warnings:
        stages["normalization_warnings"] = warnings

    return {
        "status": "ok",
        "method": "direct_llm_atomic_dag",
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "raw_question_item": item.get("raw"),
        "gold_answer": item.get("answer"),
        "stages": stages,
        "subquestions": dag.get("nodes", []),
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# Direct LLM Decomposition #{payload['index']}",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Question: {payload['question']}",
    ]
    if payload.get("gold_answer") is not None:
        lines.append(f"- Gold answer: {payload['gold_answer']}")
    lines.extend(["", "## Atomic Subquestion DAG"])
    dag = payload["stages"]["2_direct_llm_atomic_subquestion_dag"]
    for node in dag.get("nodes", []):
        dependencies = node.get("dependencies") or []
        dep_text = f" depends_on={', '.join(dependencies)}" if dependencies else " depends_on=none"
        lines.append(f"- {node.get('node_id')}: {node.get('question')}{dep_text}")
    if dag.get("warnings"):
        lines.extend(["", "## Warnings"])
        for warning in dag["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def build_error_payload(dataset: str, questions_file: Path, item: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "status": "error",
        "method": "direct_llm_atomic_dag",
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


def build_error_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Direct LLM Decomposition Error #{payload['index']}",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Question: {payload['question']}",
    ]
    if payload.get("gold_answer") is not None:
        lines.append(f"- Gold answer: {payload['gold_answer']}")
    lines.extend(["", f"- Error type: `{payload['error_type']}`", "", "```text", str(payload["error"]), "```", ""])
    return "\n".join(lines)


def _normalize_dependencies(raw_dependencies: Any, seen_ids: set[str], node_id: str, warnings: list[str]) -> list[str]:
    if raw_dependencies is None:
        return []
    if isinstance(raw_dependencies, str):
        dependencies = [raw_dependencies]
    elif isinstance(raw_dependencies, list):
        dependencies = raw_dependencies
    else:
        warnings.append(f"Ignored invalid dependencies for {node_id}: expected list or string.")
        return []

    valid: list[str] = []
    for dependency in dependencies:
        dependency_id = str(dependency).strip()
        if not dependency_id or dependency_id == node_id:
            continue
        if dependency_id not in seen_ids:
            warnings.append(f"Ignored dependency {dependency_id!r} for {node_id}; it does not reference an earlier node.")
            continue
        if dependency_id not in valid:
            valid.append(dependency_id)
    return valid


def _resolve_question_files(args: argparse.Namespace) -> list[Path]:
    if args.questions_file:
        path = Path(args.questions_file)
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {path}")
        return [path]
    questions_root = Path(args.questions_root)
    if args.dataset:
        path = questions_root / args.dataset / "questions.json"
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {path}")
        return [path]
    if args.all_datasets:
        return sorted(questions_root.glob("*/questions.json"))
    raise ValueError("Specify --dataset, --questions-file, or --all-datasets.")


def _read_question_items(path: Path) -> list[dict[str, Any]]:
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
        items.append({"index": index, "qid": qid, "question": question, "answer": answer, "raw": raw})
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _manifest_item(payload: dict[str, Any], question_dir: Path) -> dict[str, Any]:
    dag = payload["stages"]["2_direct_llm_atomic_subquestion_dag"]
    return {
        "method": "direct_llm_atomic_dag",
        "dataset": payload["dataset"],
        "index": payload["index"],
        "qid": payload.get("qid"),
        "question": payload["question"],
        "gold_answer": payload.get("gold_answer"),
        "status": "ok",
        "atomic_question_count": len(dag.get("nodes", [])),
        "warning_count": len(dag.get("warnings", [])),
        "output_dir": str(question_dir),
    }


if __name__ == "__main__":
    raise SystemExit(main())
