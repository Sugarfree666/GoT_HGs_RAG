from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyper_branch.config import Config, load_config
from hyper_branch.llm import OpenAICompatibleClient
from hyper_branch.logging_utils import TraceStore, configure_logging, create_run_dir
from hyper_branch.utils import extract_json_payload


SYSTEM_PROMPT = """You are an LLM-only QA baseline.

Return JSON only:
{
  "answer": "...",
  "reasoning_summary": "...",
  "confidence": 0.0
}

Rules:
- Answer the question directly using only your internal knowledge.
- Do not use retrieval, tools, documents, or hidden evidence.
- The answer field must be the clean shortest answer span suitable for QA evaluation.
- For yes/no questions, answer exactly "yes" or "no".
- If you do not know, set answer to "INSUFFICIENT_EVIDENCE" and confidence to 0.0.
- Keep reasoning_summary brief.
- Keep confidence between 0 and 1.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LLM-only baseline on 2WikiMultiHopQA questions.")
    parser.add_argument("--config", default="configs/2wikimultihopqa.yaml", help="Config used for LLM settings.")
    parser.add_argument("--question-file", default="questions/2wikimultihopqa/questions.json")
    parser.add_argument("--output-dir", default="runs/LLM_only")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", default="", help="Optional API key override. Prefer env vars.")
    parser.add_argument("--base-url", default="", help="Optional OpenAI-compatible base URL override.")
    parser.add_argument("--resume", action="store_true", help="Skip questions that already have a final_answer.json.")
    parser.add_argument("--allow-failure", action="store_true", help="Continue after per-question failures.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()
    config = load_config(Path(args.config), project_root)
    _apply_overrides(config, args)
    questions = _load_questions(Path(args.question_file), args.start_index, args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for offset, item in enumerate(questions):
        question_index = args.start_index + offset
        question = str(item.get("question", "") or "").strip()
        if not question:
            failures += 1
            if not args.allow_failure:
                raise ValueError(f"Question at index {question_index} is missing a non-empty 'question' field.")
            continue

        run_dir = _existing_success_run(output_dir, question) if args.resume else None
        if run_dir is not None:
            print(f"[{question_index}] skipped existing {run_dir}")
            continue

        run_dir = create_run_dir(output_dir, question)
        trace_store = TraceStore(run_dir)
        logger = configure_logging(run_dir, config.runtime.log_level, verbose_console=args.verbose)
        logger.info("Starting HyperBranch pipeline for question: %s", question)
        logger.info("Running LLM-only baseline with model=%s", config.llm.model)
        try:
            client = OpenAICompatibleClient(config.llm, trace_store=trace_store)
            final_answer = answer_question(client, question, temperature=args.temperature)
            trace_store.save_artifact(
                "artifacts/llm_only_input.json",
                {"question_index": question_index, "question": question, "model": config.llm.model},
            )
            trace_store.save_artifact("artifacts/final_answer.json", final_answer)
            print(f"[{question_index}] success answer={final_answer.get('answer', '')} run_dir={run_dir}")
        except Exception as exc:
            failures += 1
            logger.error("Pipeline failed for question: %s", question)
            trace_store.save_artifact(
                "artifacts/error.json",
                {
                    "question": question,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "config": {"model": config.llm.model, "base_url_env": config.llm.base_url_env},
                },
            )
            print(f"[{question_index}] failed error={type(exc).__name__}: {exc} run_dir={run_dir}")
            if not args.allow_failure:
                _close_logger(logger)
                return 1
        finally:
            _close_logger(logger)

    print(f"completed total={len(questions)} failures={failures} output_dir={output_dir}")
    return 0 if args.allow_failure or failures == 0 else 1


def answer_question(client: OpenAICompatibleClient, question: str, *, temperature: float) -> dict[str, Any]:
    response_text = client.chat_text(
        stage="llm_only_answer",
        system_prompt=SYSTEM_PROMPT,
        user_payload={"question": question},
        max_tokens=500,
        temperature=temperature,
    )
    parsed = extract_json_payload(response_text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM-only answer did not return a JSON object.")
    answer = str(parsed.get("answer", "") or "").strip()
    if not answer:
        answer = "INSUFFICIENT_EVIDENCE"
    confidence = _clamp_float(parsed.get("confidence", 0.0))
    return {
        "answer": answer,
        "reasoning_summary": str(parsed.get("reasoning_summary", "") or "").strip(),
        "confidence": confidence,
        "remaining_gaps": list(parsed.get("remaining_gaps", [])) if isinstance(parsed.get("remaining_gaps"), list) else [],
    }


def _load_questions(path: Path, start_index: int, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return payload[start_index : start_index + limit]


def _apply_overrides(config: Config, args: argparse.Namespace) -> None:
    config.llm.model = args.model
    config.llm.temperature = args.temperature
    config.runtime.base_run_dir = Path(args.output_dir).resolve()
    if args.api_key:
        os.environ[config.llm.api_key_env] = args.api_key
    if args.base_url:
        os.environ[config.llm.base_url_env] = args.base_url.rstrip("/")


def _existing_success_run(output_dir: Path, question: str) -> Path | None:
    for run_dir in sorted((path for path in output_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime):
        if not (run_dir / "artifacts" / "final_answer.json").exists():
            continue
        log_path = run_dir / "run.log"
        if not log_path.exists():
            continue
        marker = f"Starting HyperBranch pipeline for question: {question}"
        if marker in log_path.read_text(encoding="utf-8", errors="replace"):
            return run_dir
    return None


def _clamp_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
