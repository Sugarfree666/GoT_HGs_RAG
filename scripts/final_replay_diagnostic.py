from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval import cal_em, cal_f1  # type: ignore
from hyper_branch.atomic.composer import FinalAnswerComposer, _postprocess_final_answer
from hyper_branch.config import LLMConfig
from hyper_branch.llm.client import OpenAICompatibleClient
from hyper_branch.llm.prompts import PromptManager
from hyper_branch.llm.service import OpenAIAtomicLLMService


COUNTRY_TO_DEMONYM = {
    "america": "american",
    "austria": "austrian",
    "canada": "canadian",
    "china": "chinese",
    "denmark": "danish",
    "france": "french",
    "germany": "german",
    "italy": "italian",
    "romania": "romanian",
    "russia": "russian",
    "spain": "spanish",
    "united states": "american",
    "united states of america": "american",
}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = read_manifest(run_dir)
    wanted_ids = parse_ids(args.ids)
    new_answers = read_new_answers(args.new_answers_json)
    service = make_service() if args.online_replay else None
    diagnostics = []

    for row in rows:
        idx = int(row.get("index", 0))
        if wanted_ids and idx not in wanted_ids:
            continue
        if args.max_index and idx > args.max_index:
            continue
        atomic_results, dag_nodes = load_artifacts(Path(row["output_dir"]))
        old_answer = str(row.get("final_answer") or "")
        if idx in new_answers:
            new_answer = new_answers[idx]
        elif service is not None:
            new_answer = replay_final_answer(service, row, atomic_results, dag_nodes)
        else:
            new_answer = old_answer
        diagnostics.append(build_diagnostic(row, old_answer, new_answer, atomic_results))

    if args.summary:
        print(json.dumps(summarize(diagnostics), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        write_output(diagnostics, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose final-answer replay failures.")
    parser.add_argument("run_dir", help="Path to a depo_hyperbranch run directory containing manifest.jsonl.")
    parser.add_argument("--ids", default="", help="Comma-separated 1-based example ids to diagnose.")
    parser.add_argument(
        "--new-answers-json",
        default="",
        help="Optional JSON/JSONL mapping replayed answers. Supports {id: answer} or rows with index/new_answer/answer.",
    )
    parser.add_argument("--online-replay", action="store_true", help="Replay final answers with the current LLM prompt.")
    parser.add_argument("--max-index", type=int, default=0, help="Only diagnose examples with index <= this value.")
    parser.add_argument("--summary", action="store_true", help="Print aggregate metrics instead of JSONL diagnostics.")
    parser.add_argument("--output", default="", help="Optional output path. Defaults to stdout JSONL.")
    return parser.parse_args()


def read_manifest(run_dir: Path) -> list[dict[str, Any]]:
    manifest = run_dir / "manifest.jsonl"
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_ids(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def read_new_answers(path: str) -> dict[int, str]:
    if not path:
        return {}
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if source.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return {int(key): str(value) for key, value in payload.items()}
        rows = payload
    answers: dict[int, str] = {}
    for row in rows:
        idx = int(row.get("index", row.get("id", row.get("idx", 0))))
        if idx:
            answers[idx] = str(row.get("new_answer", row.get("answer", "")))
    return answers


def make_service() -> OpenAIAtomicLLMService:
    if not os.environ.get("OPENAI_BASE_URL") and os.environ.get("OPAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["OPAI_BASE_URL"]
    config = LLMConfig(
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        model=os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
        timeout_seconds=120,
        max_retries=2,
        retry_backoff_seconds=1.0,
        temperature=0.0,
    )
    return OpenAIAtomicLLMService(OpenAICompatibleClient(config), PromptManager(ROOT / "prompts"))


def load_artifacts(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_path = output_dir / "hyperbranch_result.json"
    dag_path = output_dir / "hyperbranch_dag.json"
    if not result_path.exists() or not dag_path.exists():
        return [], []
    result = json.loads(result_path.read_text(encoding="utf-8"))
    dag = json.loads(dag_path.read_text(encoding="utf-8"))
    atomic_results = [atomic_payload(item) for item in result.get("atomic_results", [])]
    dag_nodes = [
        {
            "node_id": item.get("node_id") or item.get("id") or "",
            "question": item.get("question", ""),
            "dependencies": item.get("dependencies") or item.get("depends_on") or [],
            "metadata": {
                "operation": item.get("operation", ""),
                "output_type": item.get("output_type", ""),
                "support_step_ids": item.get("support_step_ids", []),
            },
        }
        for item in dag.get("nodes", [])
    ]
    return atomic_results, dag_nodes


def atomic_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": item.get("node_id", ""),
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "confidence": item.get("confidence", 0.0),
        "reasoning_summary": trim(item.get("reasoning_summary", "")),
        "used_dependencies": list(item.get("used_dependencies", [])) if isinstance(item.get("used_dependencies", []), list) else [],
        "used_hyperedge_ids": list(item.get("used_hyperedge_ids", [])) if isinstance(item.get("used_hyperedge_ids", []), list) else [],
        "top_evidence": [evidence_payload(ev) for ev in item.get("evidence", [])[:3]],
    }


def evidence_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "hyperedge_id": trim(item.get("hyperedge_id", "")),
        "hyperedge_text": trim(item.get("hyperedge_text", "")),
        "branch_support": item.get("branch_support", []),
        "score_breakdown": item.get("score_breakdown", {}),
        "evidence_texts": [trim(text) for text in item.get("evidence_texts", [])[:2]],
    }


def trim(value: Any, limit: int = 650) -> Any:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def replay_final_answer(
    service: OpenAIAtomicLLMService,
    row: dict[str, Any],
    atomic_results: list[dict[str, Any]],
    dag_nodes: list[dict[str, Any]],
) -> str:
    if not atomic_results:
        return str(row.get("final_answer") or "")
    raw = service.compose_final_answer(str(row.get("question") or ""), dag_nodes, atomic_results)
    composer = FinalAnswerComposer()
    payload = composer._coerce_single_payload(raw, str(row.get("question") or ""), atomic_results)
    final = _postprocess_final_answer(
        payload=payload,
        original_question=str(row.get("question") or ""),
        atomic_results=atomic_results,
        dag_nodes=dag_nodes,
    )
    return str(final.get("answer") or "")


def build_diagnostic(
    row: dict[str, Any],
    old_answer: str,
    new_answer: str,
    atomic_results: list[dict[str, Any]],
) -> dict[str, Any]:
    gold = str(row.get("gold_answer") or "")
    atomic_answers = [str(item.get("answer") or "") for item in atomic_results]
    evidence_texts = collect_evidence_texts(atomic_results)
    evidence_blob = " ".join(evidence_texts)
    atomic_blob = " ".join(atomic_answers)
    old_em = cal_em([[gold]], [old_answer])
    new_em = cal_em([[gold]], [new_answer])
    old_f1 = cal_f1([[gold]], [old_answer])
    new_f1 = cal_f1([[gold]], [new_answer])
    changed_country_to_demonym = has_country_to_demonym_rewrite(old_answer, new_answer, atomic_answers)
    diagnostic = {
        "id": str(row.get("index", "")),
        "gold": gold,
        "old_answer": old_answer,
        "new_answer": new_answer,
        "old_em": old_em,
        "new_em": new_em,
        "old_f1": old_f1,
        "new_f1": new_f1,
        "atomic_answers": atomic_answers,
        "top_evidence_texts": evidence_texts[:6],
        "gold_in_atomic_answers": contains_surface(atomic_blob, gold),
        "gold_in_evidence": contains_surface(evidence_blob, gold),
        "old_contains_gold": contains_surface(old_answer, gold),
        "new_contains_gold": contains_surface(new_answer, gold),
        "new_answer_is_extractive": is_extractive(new_answer, row, atomic_answers, evidence_blob),
        "changed_country_to_demonym": changed_country_to_demonym,
    }
    diagnostic["category"] = categorize(diagnostic, old_answer, new_answer, atomic_blob, evidence_blob)
    return diagnostic


def collect_evidence_texts(atomic_results: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for result in atomic_results:
        for evidence in result.get("top_evidence", []):
            for key in ("hyperedge_text",):
                text = str(evidence.get(key, "") or "").strip()
                if text:
                    texts.append(text)
            for text in evidence.get("evidence_texts", []):
                text = str(text or "").strip()
                if text:
                    texts.append(text)
    deduped: list[str] = []
    for text in texts:
        if text not in deduped:
            deduped.append(text)
    return deduped


def contains_surface(container: str, value: str) -> bool:
    container = str(container or "").lower()
    value = str(value or "").lower()
    return bool(value and value in container)


def is_extractive(new_answer: str, row: dict[str, Any], atomic_answers: list[str], evidence_blob: str) -> bool:
    answer = str(new_answer or "").strip()
    if not answer:
        return False
    if answer.upper() == "INSUFFICIENT_EVIDENCE" or answer.lower() in {"yes", "no"}:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", answer) or re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2}|[1-9][0-9]{2})", answer):
        return True
    sources = [str(row.get("question") or ""), *atomic_answers, evidence_blob]
    return any(contains_surface(source, answer) for source in sources)


def has_country_to_demonym_rewrite(old_answer: str, new_answer: str, atomic_answers: list[str]) -> bool:
    new_norm = norm(new_answer)
    candidates = [old_answer, *atomic_answers]
    for candidate in candidates:
        candidate_norm = norm(candidate)
        if COUNTRY_TO_DEMONYM.get(candidate_norm) == new_norm:
            return True
    return False


def categorize(
    diagnostic: dict[str, Any],
    old_answer: str,
    new_answer: str,
    atomic_blob: str,
    evidence_blob: str,
) -> str:
    if diagnostic["old_em"] == 1.0 and diagnostic["new_em"] == 0.0:
        return "llm_regression"
    if diagnostic["changed_country_to_demonym"]:
        return "answer_type_mismatch"
    if diagnostic["old_contains_gold"] and norm(old_answer) != norm(diagnostic["gold"]):
        return "final_crop"
    if not diagnostic["gold_in_atomic_answers"] and not diagnostic["gold_in_evidence"]:
        if norm(old_answer) and norm(new_answer) and not diagnostic["new_answer_is_extractive"]:
            return "alias_mapping"
        return "evidence_missing"
    if not diagnostic["gold_in_atomic_answers"] and diagnostic["gold_in_evidence"]:
        return "atomic_error"
    if diagnostic["gold_in_atomic_answers"] and diagnostic["new_em"] == 0.0:
        return "llm_regression"
    if not contains_surface(atomic_blob, diagnostic["gold"]) and not contains_surface(evidence_blob, diagnostic["gold"]):
        return "evidence_missing"
    return "answer_type_mismatch"


def norm(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def write_output(diagnostics: list[dict[str, Any]], output: str) -> None:
    lines = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in diagnostics]
    if output:
        Path(output).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    else:
        for line in lines:
            print(line)


def summarize(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(diagnostics)
    if count == 0:
        return {"count": 0}
    category_counts: dict[str, int] = {}
    for item in diagnostics:
        category = str(item.get("category", ""))
        category_counts[category] = category_counts.get(category, 0) + 1
    fixed = [item for item in diagnostics if item["old_em"] == 0.0 and item["new_em"] == 1.0]
    regressed = [item for item in diagnostics if item["old_em"] == 1.0 and item["new_em"] == 0.0]
    return {
        "count": count,
        "old_em": sum(float(item["old_em"]) for item in diagnostics) / count,
        "new_em": sum(float(item["new_em"]) for item in diagnostics) / count,
        "old_f1": sum(float(item["old_f1"]) for item in diagnostics) / count,
        "new_f1": sum(float(item["new_f1"]) for item in diagnostics) / count,
        "fixed_count": len(fixed),
        "regressed_count": len(regressed),
        "changed_country_to_demonym_count": sum(1 for item in diagnostics if item["changed_country_to_demonym"]),
        "non_extractive_new_answer_count": sum(1 for item in diagnostics if not item["new_answer_is_extractive"]),
        "category_counts": category_counts,
        "fixed_ids": [item["id"] for item in fixed],
        "regressed_ids": [item["id"] for item in regressed],
    }


if __name__ == "__main__":
    main()
