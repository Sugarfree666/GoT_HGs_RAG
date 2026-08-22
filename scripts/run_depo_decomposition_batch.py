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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DEPO HanLP-SDP pipeline over questions datasets and save "
            "Step5 atomic decomposition artifacts."
        )
    )
    parser.add_argument("--dataset", help="Dataset subdirectory under questions/, e.g. 2wikimultihopqa.")
    parser.add_argument("--questions-file", help="Specific questions file path. Overrides --dataset.")
    parser.add_argument("--all-datasets", action="store_true", help="Process every questions/*/questions.json file.")
    parser.add_argument("--questions-root", default="questions", help="Root directory containing dataset folders.")
    parser.add_argument(
        "--output-root",
        default="runs/depo_decomposition",
        help="Root output directory for DEPO decomposition artifacts.",
    )
    parser.add_argument("--run-id", help="Output run id under output-root/dataset/. Defaults to current timestamp.")
    parser.add_argument("--start", type=int, default=1, help="1-based inclusive start index in each input file.")
    parser.add_argument("--end", type=int, help="1-based inclusive end index in each input file.")
    parser.add_argument("--limit", type=int, help="Maximum number of questions after applying --start/--end.")
    parser.add_argument("--resume", action="store_true", help="Skip questions already recorded in manifest.jsonl.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model used by DEPO Step2 and Step5.")
    parser.add_argument(
        "--hanlp-model",
        help="HanLP pretrained constant name from hanlp.pretrained.mtl/sdp, or a local model path.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable HanLP PAS Step4 debug JSON files.")
    parser.add_argument(
        "--debug-dir",
        default="debug/hanlp_sdp",
        help="Directory for HanLP PAS debug JSON files when --debug is enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    try:
        question_files = _resolve_question_files(args)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not question_files:
        print("No questions files found.", file=sys.stderr)
        return 2

    try:
        from atomic_question_dag import restore_global_best_paths
        from entity_masking_preprocessor import EntityMaskingPreprocessor
        from hanlp_sdp_parser import HanLPSDPParser
        from llm_client import LLMClient
        from main import run_hanlp_sdp_pipeline
        from models import QuestionRecord
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = _repo_path(args.output_root)
    debug_dir = str(_repo_path(args.debug_dir))

    llm_client = LLMClient(api_key=api_key, base_url=base_url, model=args.llm_model)
    preprocessor = EntityMaskingPreprocessor(llm_client)
    parser = HanLPSDPParser(args.hanlp_model)

    for questions_file in question_files:
        dataset = _dataset_name(questions_file)
        dataset_output_dir = output_root / dataset / run_id
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_output_dir / "manifest.jsonl"
        summary_path = dataset_output_dir / "summary.md"
        report_path = dataset_output_dir / "decomposition.md"
        processed_keys = (
            _processed_manifest_keys(manifest_path)
            if args.resume and report_path.exists()
            else set()
        )

        items = _slice_items(_read_question_items(questions_file), start=args.start, end=args.end, limit=args.limit)
        print(
            f"Running DEPO decomposition: dataset={dataset}, "
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
        if args.resume and report_path.exists():
            report_lines = report_path.read_text(encoding="utf-8").splitlines()
            if report_lines and report_lines[-1]:
                report_lines.append("")
        else:
            report_lines = _run_report_header(
                dataset=dataset,
                questions_file=questions_file,
                run_id=run_id,
                start=args.start,
                end=args.end,
                limit=args.limit,
            )

        manifest_mode = "a" if args.resume else "w"
        with manifest_path.open(manifest_mode, encoding="utf-8") as manifest:
            for offset, item in enumerate(items, start=1):
                record = QuestionRecord(question=item["question"], qid=item.get("qid"))
                manifest_key = _manifest_key(dataset, item["index"], record.qid)
                if args.resume and manifest_key in processed_keys:
                    print(f"[skip] {dataset} #{item['index']} {record.question}")
                    manifest_item = {
                        "method": "depo_hanlp_sdp_atomic_dag",
                        "dataset": dataset,
                        "index": item["index"],
                        "qid": record.qid,
                        "question": record.question,
                        "status": "skipped",
                        "output_file": str(report_path),
                    }
                    manifest.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
                    manifest.flush()
                    continue

                print(f"[run {offset}/{len(items)}] {dataset} #{item['index']} {record.question}")
                try:
                    result = run_hanlp_sdp_pipeline(
                        record=record,
                        index=item["index"],
                        preprocessor=preprocessor,
                        parser=parser,
                        debug=args.debug,
                        debug_dir=debug_dir,
                        llm_client=llm_client,
                    )
                    question_structure = restore_global_best_paths(
                        result["token_reasoning_structure"].paths,
                        result["preprocess_result"].mask_mappings,
                    )
                    payload = build_decomposition_payload(
                        dataset=dataset,
                        questions_file=questions_file,
                        item=item,
                        result=result,
                        question_structure=question_structure,
                    )
                    report_lines.extend(build_markdown_report(payload, heading_level=2).splitlines())
                    report_lines.append("")
                    manifest_item = _manifest_item(payload, report_path)
                    summary_lines.extend(_summary_question_lines(payload, report_path))
                    print(
                        f"[ok]  {dataset} #{item['index']} "
                        f"nodes={_dag_node_count(payload)} valid={_dag_valid(payload)} -> {report_path}"
                    )
                except Exception as exc:  # Keep long batch jobs inspectable even if one item fails.
                    payload = build_error_payload(dataset, questions_file, item, exc)
                    report_lines.extend(build_error_markdown(payload, heading_level=2).splitlines())
                    report_lines.append("")
                    manifest_item = {
                        "method": "depo_hanlp_sdp_atomic_dag",
                        "dataset": dataset,
                        "index": item["index"],
                        "qid": item.get("qid"),
                        "question": item["question"],
                        "status": "sample",
                        "error_type": type(exc).__name__,
                        "sample": str(exc),
                        "output_file": str(report_path),
                    }
                    summary_lines.extend(_summary_error_lines(payload, report_path))
                    print(f"[err] {dataset} #{item['index']} {type(exc).__name__}: {exc}")

                manifest.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
                manifest.flush()
                report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
                summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")

        report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
        summary_path.write_text("\n".join(summary_lines).rstrip() + "\n", encoding="utf-8")
        print(f"Decomposition report written to {report_path}")
        print(f"Summary written to {summary_path}")

    return 0


def build_decomposition_payload(
    *,
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    result: dict[str, Any],
    question_structure: list[list[str]],
) -> dict[str, Any]:
    preprocess_result = result["preprocess_result"]
    token_reasoning_structure = result["token_reasoning_structure"]
    atomic_question_dag = result.get("atomic_question_dag")

    return {
        "status": "ok",
        "method": "depo_hanlp_sdp_atomic_dag",
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "raw_question_item": item.get("raw"),
        "gold_answer": item.get("answer"),
        "stages": {
            "1_explicit_entities": preprocess_result.explicit_entities.to_dict(),
            "2_entity_masking": {
                "masked_question": preprocess_result.masked_question,
                "hanlp_input_sentence": result.get("hanlp_input_sentence") or preprocess_result.masked_question,
                "mask_mappings": [mapping.to_dict() for mapping in preprocess_result.mask_mappings],
            },
            "3_hanlp_sdp_parsing": {
                "model": result["hanlp_sdp_result"].model,
                "tokens": list(result["hanlp_sdp_result"].tokens),
                "available_keys": list(result["hanlp_sdp_result"].available_keys),
                "mask_token_checks": dict(result["hanlp_sdp_result"].mask_token_checks),
                "sdp_graphs": result["hanlp_sdp_result"].sdp_graphs,
                "edges": _hanlp_sdp_edges(result["hanlp_sdp_result"]),
                "warnings": list(result["hanlp_sdp_result"].warnings),
            },
            "4_token_reasoning_structure": _compact_token_reasoning(token_reasoning_structure),
            "5_step5_action_trace": {
                "input": {
                    "original_question": item["question"],
                    "question_entities": [
                        entity.text
                        for entity in preprocess_result.explicit_entities.entities
                    ],
                    "question_structure": [
                        " -- ".join(str(node).strip() for node in branch if str(node).strip())
                        for branch in question_structure
                        if any(str(node).strip() for node in branch)
                    ],
                },
                "atomic_questions": _step5_atomic_questions(atomic_question_dag),
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
        "status": "sample",
        "method": "depo_hanlp_sdp_atomic_dag",
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "raw_question_item": item.get("raw"),
        "gold_answer": item.get("answer"),
        "error_type": type(exc).__name__,
        "sample": str(exc),
    }


def build_markdown_report(payload: dict[str, Any], *, heading_level: int = 1) -> str:
    stages = payload["stages"]
    dag = stages.get("6_atomic_question_dag") or {}
    action_trace = stages.get("5_step5_action_trace") or {}
    token_reasoning = stages.get("4_token_reasoning_structure") or {}
    h1 = "#" * heading_level
    h2 = "#" * (heading_level + 1)
    h3 = "#" * (heading_level + 2)
    lines: list[str] = [
        f"{h1} DEPO Decomposition #{payload['index']}",
        "",
        f"- Dataset: `{payload['dataset']}`",
    ]
    if payload.get("qid"):
        lines.append(f"- QID: `{payload['qid']}`")
    lines.append(f"- Question: {payload['question']}")
    if payload.get("gold_answer") is not None:
        lines.append(f"- Gold answer: {payload['gold_answer']}")
    lines.append("")

    lines.append(f"{h2} 1. Explicit Entities")
    entities = (stages.get("1_explicit_entities") or {}).get("entities", [])
    if entities:
        for entity in entities:
            lines.append(f"- {entity.get('text')} span=({entity.get('start_char')}, {entity.get('end_char')})")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(f"{h2} 2. Entity Masking")
    masking = stages.get("2_entity_masking") or {}
    for mapping in masking.get("mask_mappings", []):
        lines.append(f"- {mapping.get('placeholder')} -> {mapping.get('original_text')}")
    if not masking.get("mask_mappings"):
        lines.append("(none)")
    lines.append("")
    lines.append(f"Masked question: {masking.get('masked_question', '')}")
    lines.append("")

    lines.append(f"{h2} 3. HanLP SDP Graph")
    hanlp = stages.get("3_hanlp_sdp_parsing") or {}
    lines.append(f"Model: {hanlp.get('model') or '(unknown)'}")
    tokens = hanlp.get("tokens") or []
    if tokens:
        lines.append(f"Tokens: {' '.join(str(token) for token in tokens)}")
    mask_checks = hanlp.get("mask_token_checks") or {}
    if mask_checks:
        lines.append("")
        lines.append("Mask token checks:")
        for placeholder, status in mask_checks.items():
            lines.append(f"- {placeholder}: {status}")
    edges = hanlp.get("edges") or []
    if edges:
        lines.append("")
        lines.append("Readable SDP edges:")
        for formalism in _hanlp_sdp_formalisms(edges):
            lines.append(f"- {formalism}")
            for edge in edges:
                if edge.get("formalism") == formalism:
                    lines.append(f"  - {_hanlp_sdp_edge_display(edge)}")
    else:
        lines.append("(no readable SDP edges)")
    lines.append("")

    lines.append(f"{h2} 4. Token Reasoning Structure")
    lines.append(f"{h3} Repaired Evidence Graph")
    repaired_graph_lines = _repaired_evidence_graph_lines(token_reasoning)
    if repaired_graph_lines:
        lines.extend(repaired_graph_lines)
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(f"{h3} Question Structure")
    input_payload = action_trace.get("input") or {}
    question_structure = input_payload.get("question_structure") or []
    if question_structure:
        for branch_index, branch in enumerate(question_structure, start=1):
            branch_text = branch if isinstance(branch, str) else " -- ".join(branch)
            lines.append(f"- Branch {branch_index}: {branch_text}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(f"{h2} 5. Step5 Atomic Questions")
    atomic_questions = _atomic_questions_from_payload(action_trace)
    if atomic_questions:
        for question in atomic_questions:
            depends_on = question.get("depends_on") or []
            lines.append(f"- {question.get('id')}: {question.get('question')}")
            lines.append(f"  - depends_on: {', '.join(depends_on) if depends_on else '(none)'}")
            lines.append(f"  - operation: {question.get('operation') or ''}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append(f"{h2} 6. Atomic Question DAG")
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

def build_error_markdown(payload: dict[str, Any], *, heading_level: int = 1) -> str:
    h1 = "#" * heading_level
    lines = [
        f"{h1} DEPO Decomposition Error #{payload['index']}",
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
            str(payload["sample"]),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _repaired_evidence_graph_lines(token_reasoning: dict[str, Any]) -> list[str]:
    edges = token_reasoning.get("repaired_evidence_edges") or token_reasoning.get("graph_edges") or []
    lines: list[str] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source_id = str(edge.get("source") or edge.get("source_id") or "?")
        target_id = str(edge.get("target") or edge.get("target_id") or "?")
        source_text = str(edge.get("source_text") or edge.get("source") or source_id)
        target_text = str(edge.get("target_text") or edge.get("target") or target_id)
        metadata = _format_repaired_edge_metadata(edge)
        suffix = f" ({metadata})" if metadata else ""
        lines.append(f"{source_text}[{source_id}] -- {target_text}[{target_id}]{suffix}")
    return lines


def _format_repaired_edge_metadata(edge: dict[str, Any]) -> str:
    edge_cost = edge.get("edge_cost")
    if edge_cost is None:
        return "cost=blocked"
    return f"cost={edge_cost}"

# 
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
        raw_items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_payload, list):
            raise ValueError(f"{path} must contain a JSON list.")
        raw_items = raw_payload

    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        if isinstance(item, str):
            question = item.strip()
            raw = item
            qid = None
            answer = None
        elif isinstance(item, dict):
            question = str(item.get("question", item.get("query", ""))).strip()
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


def _step5_actions(atomic_question_dag: Any) -> list[dict[str, Any]]:
    if atomic_question_dag is None:
        return []
    raw_payload = getattr(atomic_question_dag, "raw_payload", None)
    if not isinstance(raw_payload, dict):
        return []
    actions = raw_payload.get("actions")
    if not isinstance(actions, list):
        return []
    return [dict(action) for action in actions if isinstance(action, dict)]


def _step5_atomic_questions(atomic_question_dag: Any) -> list[dict[str, Any]]:
    if atomic_question_dag is None:
        return []
    raw_payload = getattr(atomic_question_dag, "raw_payload", None)
    if not isinstance(raw_payload, dict):
        return []
    return _atomic_questions_from_payload(raw_payload)


def _atomic_questions_from_payload(raw_payload: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_payload, dict):
        return []

    questions = raw_payload.get("atomic_questions")
    if isinstance(questions, list):
        return [dict(question) for question in questions if isinstance(question, dict)]

    raw_dag = raw_payload.get("atomic_question_dag")
    if isinstance(raw_dag, dict):
        questions = raw_dag.get("atomic_questions")
        if isinstance(questions, list):
            return [dict(question) for question in questions if isinstance(question, dict)]
    if isinstance(raw_dag, list):
        return [dict(question) for question in raw_dag if isinstance(question, dict)]
    return []


def _hanlp_sdp_edges(hanlp_sdp_result: Any) -> list[dict[str, Any]]:
    return [
        {
            "formalism": edge.formalism,
            "head_idx": edge.head_idx,
            "head": edge.head,
            "relation": edge.relation,
            "dep_idx": edge.dep_idx,
            "dep": edge.dep,
            "display": edge.display() if hasattr(edge, "display") else _hanlp_sdp_edge_display(edge.__dict__),
        }
        for edge in getattr(hanlp_sdp_result, "edges", []) or []
    ]


def _hanlp_sdp_formalisms(edges: list[dict[str, Any]]) -> list[str]:
    formalisms = {str(edge.get("formalism") or "") for edge in edges}
    return sorted((formalism for formalism in formalisms if formalism), key=_hanlp_sdp_formalism_sort_key)


def _hanlp_sdp_formalism_sort_key(formalism: str) -> tuple[int, str]:
    order = {"sdp/dm": 0, "sdp/pas": 1, "sdp/psd": 2}
    return (order.get(formalism, 100), formalism)


def _hanlp_sdp_edge_display(edge: dict[str, Any]) -> str:
    head_idx = int(edge.get("head_idx") or 0)
    dep_idx = int(edge.get("dep_idx") or 0)
    head = str(edge.get("head") or "")
    dep = str(edge.get("dep") or "")
    relation = str(edge.get("relation") or "")
    head_label = "ROOT[0]" if head_idx == 0 else f"{head}[{head_idx}]"
    dep_label = f"{dep}[{dep_idx}]" if dep_idx > 0 else dep
    return f"{head_label} --{relation}--> {dep_label}"


def _compact_token_reasoning(token_reasoning_structure: Any) -> dict[str, Any]:
    debug_payload = getattr(token_reasoning_structure, "debug_payload", {}) or {}
    repaired_evidence_edges = list(debug_payload.get("repaired_evidence_edges") or [])
    if not repaired_evidence_edges:
        repaired_evidence_edges = [
            edge.to_dict() if hasattr(edge, "to_dict") else dict(edge)
            for edge in getattr(token_reasoning_structure, "edges", []) or []
            if hasattr(edge, "to_dict") or isinstance(edge, dict)
        ]
    return {
        "path_type": token_reasoning_structure.path_type,
        "answer_anchor": token_reasoning_structure.answer_anchor,
        "answer_anchor_id": token_reasoning_structure.answer_anchor_id,
        "entity_anchors": list(token_reasoning_structure.entity_anchors),
        "candidate_sets": [list(candidate_set) for candidate_set in token_reasoning_structure.candidate_sets],
        "constraints": list(token_reasoning_structure.constraints),
        "global_selection": dict(token_reasoning_structure.global_selection or {}),
        "warnings": list(token_reasoning_structure.warnings),
        "debug_file": token_reasoning_structure.debug_file,
        "graph_edges": [
            edge.to_dict() if hasattr(edge, "to_dict") else dict(edge)
            for edge in token_reasoning_structure.edges
        ],
        "repaired_evidence_edges": repaired_evidence_edges,
        "virtual_edges": list(debug_payload.get("virtual_edges") or []),
        "paths": [path.to_dict() for path in token_reasoning_structure.paths],
    }


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
        f"# DEPO Decomposition Run: {dataset}",
        "",
        f"- Run id: `{run_id}`",
        f"- Questions file: `{questions_file}`",
        f"- Range: `{start}-{end if end is not None else 'end'}`",
    ]
    if limit is not None:
        lines.append(f"- Limit: `{limit}`")
    lines.append("")
    return lines


def _run_report_header(
    *,
    dataset: str,
    questions_file: Path,
    run_id: str,
    start: int,
    end: int | None,
    limit: int | None,
) -> list[str]:
    lines = [
        f"# DEPO Decomposition Run: {dataset}",
        "",
        f"- Run id: `{run_id}`",
        f"- Questions file: `{questions_file}`",
        f"- Range: `{start}-{end if end is not None else 'end'}`",
    ]
    if limit is not None:
        lines.append(f"- Limit: `{limit}`")
    lines.append("")
    return lines


def _summary_question_lines(payload: dict[str, Any], output_file: Path) -> list[str]:
    dag = (((payload.get("stages") or {}).get("6_atomic_question_dag")) or {})
    lines = [
        f"## {payload['index']}. {payload['question']}",
        "",
        f"- Output: `{output_file}`",
        f"- DAG valid: `{dag.get('valid')}`",
        f"- DAG nodes: `{len(dag.get('nodes', []) or [])}`",
        "",
    ]
    if dag.get("nodes"):
        for node in dag["nodes"]:
            lines.append(f"- {node.get('id')}: {node.get('question')}")
    lines.append("")
    return lines


def _summary_error_lines(payload: dict[str, Any], output_file: Path) -> list[str]:
    return [
        f"## {payload['index']}. {payload['question']}",
        "",
        f"- Output: `{output_file}`",
        f"- Status: sample",
        f"- Error: `{payload['error_type']}: {payload['sample']}`",
        "",
    ]


def _manifest_item(payload: dict[str, Any], output_file: Path) -> dict[str, Any]:
    dag = (((payload.get("stages") or {}).get("6_atomic_question_dag")) or {})
    return {
        "method": payload["method"],
        "dataset": payload["dataset"],
        "index": payload["index"],
        "qid": payload.get("qid"),
        "question": payload["question"],
        "gold_answer": payload.get("gold_answer"),
        "status": payload["status"],
        "dag_valid": dag.get("valid"),
        "dag_node_count": len(dag.get("nodes", []) or []),
        "output_file": str(output_file),
    }


def _processed_manifest_keys(manifest_path: Path) -> set[tuple[str, int, str | None]]:
    if not manifest_path.exists():
        return set()
    keys: set[tuple[str, int, str | None]] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("status") not in {"ok", "sample"}:
            continue
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        keys.add(_manifest_key(str(item.get("dataset") or ""), index, item.get("qid")))
    return keys


def _manifest_key(dataset: str, index: int, qid: str | None) -> tuple[str, int, str | None]:
    return (dataset, index, str(qid) if qid is not None else None)


def _question_dir_name(index: int, qid: str | None, question: str) -> str:
    prefix = f"{index:05d}"
    if qid:
        prefix += f"_{_slug(qid, max_len=48)}"
    return f"{prefix}_{_slug(question, max_len=80)}"


def _slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return (slug[:max_len].strip("-") or "question")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    return value


def _dag_valid(payload: dict[str, Any]) -> Any:
    return (((payload.get("stages") or {}).get("6_atomic_question_dag")) or {}).get("valid")


def _dag_node_count(payload: dict[str, Any]) -> int:
    return len(((((payload.get("stages") or {}).get("6_atomic_question_dag")) or {}).get("nodes")) or [])


def _dataset_name(questions_file: Path) -> str:
    if questions_file.name == "questions.json":
        return questions_file.parent.name
    return questions_file.parent.name if questions_file.parent.name != "questions" else questions_file.stem


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


if __name__ == "__main__":
    raise SystemExit(main())
