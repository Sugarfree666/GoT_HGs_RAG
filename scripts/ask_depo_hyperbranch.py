from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_depo_2wiki import (  # noqa: E402
    HyperBranchClientAdapter,
    HyperBranchExperimentRunner,
    QuestionRecord,
    _build_depo_components,
    _build_parser,
    _combined_evidence,
    _depo_result_to_artifact,
    _ensure_dag,
    _execution_bindings_for_node,
    _extract_evidence,
    _normalize_base_url,
    _resolve_path,
    _substitute_variables,
    _synthetic_evidence_subgraph,
    _synthetic_task_frame,
    _synthetic_thought_graph,
    _write_json,
    load_config,
    run_depo_pipeline,
    short_text,
    slugify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one custom question through DEPO + HyperBranch and print debug artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--question", default="", help="Question to run. If omitted, prompt in the console.")
    parser.add_argument("--config", default="configs/2wikimultihopqa.yaml")
    parser.add_argument("--output-dir", default="runs/depo_custom")
    parser.add_argument("--model", default="", help="Override chat model for both DEPO and HyperBranch.")
    parser.add_argument("--embedding-model", default="", help="Override embedding model for HyperBranch.")
    parser.add_argument("--api-key", default="", help="OpenAI-compatible API key. Prefer OPENAI_API_KEY.")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL, e.g. https://api.example.com/v1.")
    parser.add_argument("--evidence-top-k", type=int, default=3, help="Evidence snippets to print per subquestion.")
    parser.add_argument("--corenlp-url", default="http://localhost:9000")
    parser.add_argument("--corenlp-memory", default="4G")
    parser.add_argument("--corenlp-home", default="")
    parser.add_argument("--corenlp-timeout-ms", type=int, default=60000)
    parser.add_argument(
        "--corenlp-backend",
        choices=("auto", "stanza", "java"),
        default="java",
        help="Use stanza CoreNLPClient, direct Java server management, or auto fallback.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    question = args.question.strip() or input("Question: ").strip()
    if not question:
        raise ValueError("Question is empty.")

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = _normalize_base_url(args.base_url)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY or pass --api-key.")

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{slugify(question, 48)}"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(_resolve_path(args.config), PROJECT_ROOT)
    config.runtime.base_run_dir = output_dir
    if args.model:
        config.llm.model = args.model
    if args.embedding_model:
        config.llm.embedding_model = args.embedding_model

    hb_runner = HyperBranchExperimentRunner(config=config, output_dir=output_dir)
    if hb_runner.client is None:
        raise RuntimeError("DEPO decomposition requires an online LLM client.")
    depo_llm = HyperBranchClientAdapter(hb_runner.client)
    depo_components = _build_depo_components(depo_llm)

    _print_title("Question")
    print(question)
    print(f"Run dir: {run_dir}")

    try:
        parser_cm = _build_parser(args, output_dir)
        with parser_cm as parser_backend:
            print("\n[1/4] Running DEPO decomposition...")
            depo_result = run_depo_pipeline(
                record=QuestionRecord(question=question),
                index=1,
                parser=parser_backend,
                debug=False,
                **depo_components,
            )
            dag = _ensure_dag(depo_result)
            depo_artifact = _depo_result_to_artifact(depo_result)
            _write_json(artifacts_dir / "depo_decomposition.json", depo_artifact)

            _print_semantic_ast(depo_artifact.get("semantic_ast", {}))
            _print_atomic_dag(depo_artifact.get("subquestion_dag", dag.to_dict()))

            print("\n[2/4] Running HyperBranch for atomic subquestions...")
            variable_bindings: dict[str, str] = {}
            subquestion_results: list[dict[str, Any]] = []
            for index, node in enumerate(dag.nodes, start=1):
                node_dict = node.to_dict()
                execution_bindings = _execution_bindings_for_node(node_dict, variable_bindings)
                executable_question = _substitute_variables(node.question, execution_bindings)
                sub_run_dir = run_dir / "subquestions" / node.id
                print(f"\nSubquestion {index}/{len(dag.nodes)}: {node.id}")
                print(f"  planned: {node.question}")
                if executable_question != node.question:
                    print(f"  executable: {executable_question}")

                hb_result = hb_runner.run(executable_question, sub_run_dir)
                answer = str(hb_result.get("final_answer", {}).get("answer", "") or "").strip()
                if node.output and node.output != "FINAL" and answer:
                    variable_bindings[node.output] = answer

                sub_result = {
                    "node": node_dict,
                    "question": node.question,
                    "executable_question": executable_question,
                    "answer": answer,
                    "output_variable": node.output,
                    "hyperbranch_run_dir": str(sub_run_dir),
                    "final_answer": hb_result.get("final_answer", {}),
                    "evidence": _extract_evidence(hb_result),
                    "status": "success",
                }
                subquestion_results.append(sub_result)
                _print_subquestion_result(sub_result, max(0, args.evidence_top_k))

            print("\n[3/4] Synthesizing final answer...")
            final_answer = hb_runner.synthesize_final_answer(question, dag, subquestion_results)
            combined_evidence = _combined_evidence(subquestion_results, limit=80)

            _write_json(artifacts_dir / "subquestion_results.json", subquestion_results)
            _write_json(artifacts_dir / "final_answer.json", final_answer)
            _write_json(artifacts_dir / "evidence_subgraph.json", _synthetic_evidence_subgraph(combined_evidence))
            _write_json(
                artifacts_dir / "thought_graph.json",
                _synthetic_thought_graph(question, final_answer, subquestion_results, combined_evidence),
            )
            _write_json(artifacts_dir / "task_frame.json", _synthetic_task_frame(question, dag))

            print("\n[4/4] Done.")
            _print_final_answer(final_answer)
            _print_artifacts(run_dir)
        return 0
    except Exception as exc:
        _write_json(
            artifacts_dir / "error.json",
            {
                "question": question,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print("\nError")
        print(f"{type(exc).__name__}: {exc}")
        print(f"Artifacts: {run_dir}")
        return 1


def _print_title(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def _print_semantic_ast(ast_payload: Any) -> None:
    _print_title("Semantic AST")
    if not isinstance(ast_payload, dict) or not ast_payload:
        print("(empty)")
        return

    operator = ast_payload.get("primary_operator") or {}
    if isinstance(operator, dict):
        print(f"Status: {ast_payload.get('status', '')}")
        print(f"Operator: {operator.get('operator', '') or '(none)'}")
        print(f"Operator inputs: {_format_list(operator.get('inputs', []))}")
        print(f"Operator output: {operator.get('output', '') or '(none)'}")
        cue_text = operator.get("cue_text", "") or ast_payload.get("cue_text", "")
        if cue_text:
            print(f"Cue text: {cue_text}")

    cue_frame = ast_payload.get("detected_cue_frame") or {}
    if cue_frame:
        print(f"Cue frame: {json.dumps(cue_frame, ensure_ascii=False)}")

    nodes = ast_payload.get("nodes", [])
    print("Nodes:")
    if isinstance(nodes, list) and nodes:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            pieces = [
                str(node.get("id", "")),
                str(node.get("label", "")),
                f"kind={node.get('kind', '')}",
                f"type={node.get('semantic_type', '')}",
                f"source={node.get('source', '')}",
            ]
            print(f"  - {' | '.join(piece for piece in pieces if piece)}")
            details = []
            for key in ("branch_of", "expected_value_slot", "relation_hint", "grounding_text"):
                value = node.get(key)
                if value:
                    details.append(f"{key}={value}")
            if details:
                print(f"    {'; '.join(details)}")
    else:
        print("  (none)")

    edges = ast_payload.get("edges", [])
    print("Edges:")
    if isinstance(edges, list) and edges:
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            relation = edge.get("relation_hint", "") or edge.get("edge_type", "")
            print(
                "  - "
                f"{edge.get('source', '')} -> {edge.get('target', '')}"
                f" | type={edge.get('edge_type', '')}"
                f" | relation={relation}"
            )
    else:
        print("  (none)")

    _print_messages("Warnings", ast_payload.get("warnings", []))
    _print_messages("Validation warnings", ast_payload.get("validation_warnings", []))
    _print_messages("Repair actions", ast_payload.get("fallback_repair_actions", []))


def _print_atomic_dag(dag_payload: Any) -> None:
    _print_title("Atomic Subquestion DAG")
    if not isinstance(dag_payload, dict) or not dag_payload:
        print("(empty)")
        return

    nodes = dag_payload.get("nodes", [])
    print("Nodes:")
    if isinstance(nodes, list) and nodes:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            print(
                "  - "
                f"{node.get('id', '')}"
                f" | type={node.get('type', '')}"
                f" | output={node.get('output', '')}"
                f" | depends_on={_format_list(node.get('depends_on', []))}"
            )
            print(f"    question: {node.get('question', '')}")
            candidate_bindings = node.get("candidate_bindings", [])
            if candidate_bindings:
                print(f"    candidate_bindings: {json.dumps(candidate_bindings, ensure_ascii=False)}")
    else:
        print("  (none)")

    edges = dag_payload.get("edges", [])
    print("Edges:")
    if isinstance(edges, list) and edges:
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            print(f"  - {edge.get('source', '')} -> {edge.get('target', '')} | variable={edge.get('variable', '')}")
    else:
        print("  (none)")

    _print_messages("Warnings", dag_payload.get("warnings", []))


def _print_subquestion_result(result: dict[str, Any], evidence_top_k: int) -> None:
    final_answer = result.get("final_answer", {})
    if not isinstance(final_answer, dict):
        final_answer = {}
    print(f"  answer: {result.get('answer', '') or '(empty)'}")
    summary = str(final_answer.get("reasoning_summary", "") or "").strip()
    if summary:
        print(f"  reasoning: {short_text(summary, 220)}")
    gaps = final_answer.get("remaining_gaps", [])
    if gaps:
        print(f"  gaps: {_format_list(gaps)}")

    evidence = result.get("evidence", [])
    if not isinstance(evidence, list) or not evidence_top_k:
        return
    print("  evidence:")
    for item in evidence[:evidence_top_k]:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "") or "(no chunk_id)")
        score = item.get("score")
        score_text = _format_score(score)
        content = _one_line(str(item.get("content", "") or ""))
        print(f"    - {chunk_id} | score={score_text} | {short_text(content, 260)}")


def _print_final_answer(final_answer: dict[str, Any]) -> None:
    _print_title("Final Answer")
    print(f"Answer: {final_answer.get('answer', '') or '(empty)'}")
    print(f"Confidence: {_format_score(final_answer.get('confidence'))}")
    summary = str(final_answer.get("reasoning_summary", "") or "").strip()
    if summary:
        print(f"Reasoning summary: {summary}")
    gaps = final_answer.get("remaining_gaps", [])
    if gaps:
        print(f"Remaining gaps: {_format_list(gaps)}")


def _print_artifacts(run_dir: Path) -> None:
    _print_title("Artifacts")
    print(f"Run dir: {run_dir}")
    print(f"DEPO: {run_dir / 'artifacts' / 'depo_decomposition.json'}")
    print(f"Subquestions: {run_dir / 'artifacts' / 'subquestion_results.json'}")
    print(f"Final answer: {run_dir / 'artifacts' / 'final_answer.json'}")


def _print_messages(title: str, messages: Any) -> None:
    if not messages:
        return
    print(f"{title}:")
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, (dict, list)):
                print(f"  - {json.dumps(item, ensure_ascii=False)}")
            else:
                print(f"  - {item}")
        return
    print(f"  - {messages}")


def _format_list(value: Any) -> str:
    if value is None:
        return "[]"
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def _format_score(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _one_line(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


if __name__ == "__main__":
    raise SystemExit(main())
