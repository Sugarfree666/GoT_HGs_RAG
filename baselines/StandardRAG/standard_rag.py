from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyper_branch.config import Config, load_config
from hyper_branch.data.vector_store import VectorStore
from hyper_branch.llm import OpenAICompatibleClient
from hyper_branch.logging_utils import TraceStore, configure_logging, create_run_dir
from hyper_branch.utils import extract_json_payload, pretty_json


DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 5
TEXT_EMBEDDING_3_SMALL_DIM = 1536


SYSTEM_PROMPT = """You are a Standard RAG question answering baseline.

Return valid JSON only with exactly these keys:
{"answer": "...", "confidence": 0.0, "reasoning_summary": "..."}

Rules:
- Use only the provided retrieved chunk context.
- Do not use graph, hypergraph, entity, relation, hyperedge, path, or decomposition reasoning.
- Answer the original question directly.
- The answer must be the shortest evaluation-ready span, while preserving required specificity.
- For dates, output the full date if the context supports it; otherwise output the most specific date supported.
- For yes/no questions, answer exactly "yes" or "no".
- For comparison or selection questions, output only the selected answer.
- For count questions, output only the number.
- Do not include citations, explanations, or uncertainty language in the "answer" field.
- If the context is insufficient, make the best grounded answer possible and set a lower confidence.
- confidence must be a number from 0 to 1.
"""


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    content: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class PreparedPaths:
    dataset_name: str
    dataset_root: Path
    question_file: Path
    corpus_path: Path
    index_path: Path
    output_path: Path
    runs_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a flat dense Standard RAG baseline over raw chunks/passages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/run_standard_rag.py --dataset 2wikimultihopqa "
            "--question-file questions/2wikimultihopqa/hyperrag_query_test.json "
            "--top-k 5 --limit 100 "
            "--runs-dir runs/StandardRAG/2wikimultihopqa_hyperrag_query_test "
            "--output-path runs/StandardRAG/2wikimultihopqa_hyperrag_query_test/generated_answer.json\n\n"
            "Evaluate with existing tooling:\n"
            "  python eval/get_score.py --question-file questions/2wikimultihopqa/hyperrag_query_test.json "
            "--runs-dir runs/StandardRAG/2wikimultihopqa_hyperrag_query_test "
            "--limit 100 --skip-rsim --skip-gen "
            "--output-dir eval/results/2wikimultihopqa/standard_rag"
        ),
    )
    parser.add_argument("--config", default="configs/2wikimultihopqa.yaml", help="Config used for paths and LLM settings.")
    parser.add_argument("--dataset", default="", help="Dataset name under datasets/. Overrides config dataset root.")
    parser.add_argument("--dataset-root", default="", help="Explicit dataset root. Overrides --dataset and config.")
    parser.add_argument("--question-file", default="", help="Input questions JSON/JSONL path.")
    parser.add_argument("--corpus-path", default="", help="Raw chunk/passage corpus path.")
    parser.add_argument("--index-path", default="", help="Chunk/passage vector index path.")
    parser.add_argument("--output-path", default="", help="JSON or JSONL result file path.")
    parser.add_argument("--runs-dir", default="", help="Per-question run directory for eval/get_score.py.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks/passages to retrieve.")
    parser.add_argument("--start-index", type=int, default=0, help="Starting question index.")
    parser.add_argument("--limit", type=int, default=0, help="Number of questions to run. 0 means all remaining.")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL, help="Answer generation model.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model for questions/chunks.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Answer generation temperature.")
    parser.add_argument("--max-answer-tokens", type=int, default=500)
    parser.add_argument("--embedding-batch-size", type=int, default=64, help="Batch size when building a missing index.")
    parser.add_argument("--api-key", default="", help="Optional API key override. Prefer OPENAI_API_KEY.")
    parser.add_argument("--base-url", default="", help="Optional OpenAI-compatible base URL override.")
    parser.add_argument("--resume", action="store_true", help="Skip existing successful per-question runs.")
    parser.add_argument("--allow-failure", action="store_true", help="Continue after per-question failures.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)

    project_root = Path.cwd()
    config = load_config((project_root / args.config).resolve(), project_root)
    _apply_overrides(config, args)
    paths = _prepare_paths(project_root, config, args)

    _validate_openai_env(config)
    client = OpenAICompatibleClient(config.llm)
    chunks = load_corpus_chunks(paths.corpus_path)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_store = load_or_build_chunk_store(
        paths.index_path,
        chunks,
        client,
        config.llm.embedding_model,
        args.embedding_batch_size,
    )
    validate_chunk_index_coverage(chunk_store, chunk_by_id)
    questions = load_questions(paths.question_file, args.start_index, args.limit)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    paths.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "standard_rag "
        f"dataset={paths.dataset_name} questions={len(questions)} "
        f"top_k={args.top_k} index={paths.index_path}"
    )

    results: list[dict[str, Any]] = []
    failures = 0
    for local_index, item in enumerate(questions):
        question_index = args.start_index + local_index
        try:
            result = run_one_question(
                item=item,
                question_index=question_index,
                chunk_store=chunk_store,
                chunk_by_id=chunk_by_id,
                client=client,
                config=config,
                paths=paths,
                top_k=args.top_k,
                max_answer_tokens=args.max_answer_tokens,
                resume=args.resume,
                verbose=args.verbose,
            )
            results.append(result)
            status = result.get("run_status", "success")
            print(f"[{question_index}] {status} answer={result.get('predicted_answer', '')}")
        except Exception as exc:
            failures += 1
            result = failure_record(item, question_index, exc)
            results.append(result)
            print(f"[{question_index}] failed error={type(exc).__name__}: {exc}")
            if not args.allow_failure:
                save_results(paths.output_path, results)
                return 1

    save_results(paths.output_path, results)
    print(f"saved_results={paths.output_path}")
    print(f"completed total={len(questions)} failures={failures} runs_dir={paths.runs_dir}")
    return 0 if args.allow_failure or failures == 0 else 1


def run_one_question(
    *,
    item: dict[str, Any],
    question_index: int,
    chunk_store: VectorStore,
    chunk_by_id: dict[str, ChunkRecord],
    client: OpenAICompatibleClient,
    config: Config,
    paths: PreparedPaths,
    top_k: int,
    max_answer_tokens: int,
    resume: bool,
    verbose: bool,
) -> dict[str, Any]:
    question = str(item.get("question", "") or "").strip()
    if not question:
        raise ValueError(f"Question at index {question_index} is missing a non-empty 'question' field.")

    run_dir = _existing_success_run(paths.runs_dir, question) if resume else None
    if run_dir is not None:
        final_answer = _load_json(run_dir / "artifacts" / "final_answer.json")
        evidence = _load_json(run_dir / "artifacts" / "retrieval.json")
        retrieved_contexts = evidence if isinstance(evidence, list) else []
        answer = str(final_answer.get("answer", "") if isinstance(final_answer, dict) else "").strip()
        return build_result_record(
            item=item,
            question_index=question_index,
            question=question,
            answer=answer,
            final_answer=final_answer if isinstance(final_answer, dict) else {},
            retrieved_contexts=retrieved_contexts,
            run_dir=run_dir,
            run_status="success",
        )

    run_dir = create_run_dir(paths.runs_dir, question)
    logger = configure_logging(run_dir, config.runtime.log_level, verbose_console=verbose)
    trace_store = TraceStore(run_dir)
    previous_trace_store = client.trace_store
    client.trace_store = trace_store
    try:
        logger.info("Starting HyperBranch pipeline for question: %s", question)
        logger.info(
            "Running Standard RAG baseline with model=%s embedding_model=%s top_k=%s",
            config.llm.model,
            config.llm.embedding_model,
            top_k,
        )
        retrieved_contexts = retrieve_contexts(
            question=question,
            chunk_store=chunk_store,
            chunk_by_id=chunk_by_id,
            client=client,
            top_k=top_k,
        )
        final_answer = answer_with_context(
            question=question,
            retrieved_contexts=retrieved_contexts,
            client=client,
            max_tokens=max_answer_tokens,
        )
        trace_store.save_artifact(
            "artifacts/standard_rag_input.json",
            {
                "qid": infer_qid(item, question_index),
                "question_index": question_index,
                "question": question,
                "dataset": paths.dataset_name,
                "corpus_path": str(paths.corpus_path),
                "index_path": str(paths.index_path),
                "top_k": top_k,
                "model": config.llm.model,
                "embedding_model": config.llm.embedding_model,
            },
        )
        trace_store.save_artifact("artifacts/retrieval.json", retrieved_contexts)
        trace_store.save_artifact("artifacts/evidence_subgraph.json", build_evidence_subgraph(retrieved_contexts))
        trace_store.save_artifact("artifacts/final_answer.json", final_answer)
        return build_result_record(
            item=item,
            question_index=question_index,
            question=question,
            answer=str(final_answer.get("answer", "") or "").strip(),
            final_answer=final_answer,
            retrieved_contexts=retrieved_contexts,
            run_dir=run_dir,
            run_status="success",
        )
    except Exception as exc:
        logger.error("Pipeline failed for question: %s", question)
        trace_store.save_artifact(
            "artifacts/error.json",
            {
                "question": question,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "baseline": "standard_rag",
            },
        )
        raise
    finally:
        client.trace_store = previous_trace_store
        _close_logger(logger)


def retrieve_contexts(
    *,
    question: str,
    chunk_store: VectorStore,
    chunk_by_id: dict[str, ChunkRecord],
    client: OpenAICompatibleClient,
    top_k: int,
) -> list[dict[str, Any]]:
    query_vector = client.embed_texts([question], stage="standard_rag_query_embedding")[0]
    matches = chunk_store.query(query_vector, top_k=top_k)
    contexts: list[dict[str, Any]] = []
    for rank, match in enumerate(matches, start=1):
        chunk = chunk_by_id.get(match.item_id)
        content = chunk.content if chunk is not None else str(match.metadata.get("content", "") or "")
        metadata = dict(chunk.metadata) if chunk is not None else dict(match.metadata)
        contexts.append(
            {
                "rank": rank,
                "chunk_id": match.item_id,
                "score": match.score,
                "content": content,
                "metadata": metadata,
            }
        )
    return contexts


def answer_with_context(
    *,
    question: str,
    retrieved_contexts: list[dict[str, Any]],
    client: OpenAICompatibleClient,
    max_tokens: int,
) -> dict[str, Any]:
    response_text = client.chat_text(
        stage="standard_rag_answer",
        system_prompt=SYSTEM_PROMPT,
        user_payload={
            "question": question,
            "retrieved_contexts": [
                {
                    "rank": item["rank"],
                    "chunk_id": item["chunk_id"],
                    "content": item["content"],
                }
                for item in retrieved_contexts
            ],
        },
        max_tokens=max_tokens,
        temperature=0.0,
    )
    try:
        parsed = extract_json_payload(response_text)
    except ValueError:
        parsed = {"answer": response_text.strip(), "confidence": 0.0, "reasoning_summary": ""}
    if not isinstance(parsed, dict):
        raise ValueError("Standard RAG answer did not return a JSON object.")

    answer = str(parsed.get("answer", "") or "").strip()
    if not answer:
        answer = "INSUFFICIENT_EVIDENCE"
    return {
        "answer": answer,
        "reasoning_summary": str(parsed.get("reasoning_summary", "") or "").strip(),
        "confidence": _clamp_float(parsed.get("confidence", 0.0)),
        "remaining_gaps": [],
    }


def load_or_build_chunk_store(
    index_path: Path,
    chunks: list[ChunkRecord],
    client: OpenAICompatibleClient,
    embedding_model: str,
    batch_size: int,
) -> VectorStore:
    if index_path.exists():
        store = VectorStore.from_json(index_path, name="chunks", label_fields=("__id__",))
        if store.matrix.shape[1] != TEXT_EMBEDDING_3_SMALL_DIM:
            raise ValueError(
                f"Chunk index {index_path} has dimension {store.matrix.shape[1]}, "
                f"expected {TEXT_EMBEDDING_3_SMALL_DIM} for {DEFAULT_EMBEDDING_MODEL}."
            )
        return store

    if not chunks:
        raise ValueError(f"Cannot build chunk index because corpus is empty: {index_path}")
    if batch_size <= 0:
        raise ValueError("--embedding-batch-size must be > 0")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.content for chunk in batch]
        batch_vectors = client.embed_texts(texts, stage="standard_rag_index_embeddings")
        for chunk, vector in zip(batch, batch_vectors, strict=True):
            rows.append({"__id__": chunk.chunk_id})
            vectors.append(np.asarray(vector, dtype=np.float32))
        print(f"indexed_chunks={min(start + batch_size, len(chunks))}/{len(chunks)}")

    matrix = np.vstack(vectors).astype("<f4", copy=False)
    payload = {
        "embedding_dim": int(matrix.shape[1]),
        "embedding_model": embedding_model,
        "data": rows,
        "matrix": base64.b64encode(matrix.tobytes(order="C")).decode("ascii"),
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return VectorStore.from_json(index_path, name="chunks", label_fields=("__id__",))


def validate_chunk_index_coverage(store: VectorStore, chunk_by_id: dict[str, ChunkRecord]) -> None:
    if not store.row_ids:
        raise ValueError("Chunk vector index is empty.")
    if store.matrix.shape[1] != TEXT_EMBEDDING_3_SMALL_DIM:
        raise ValueError(
            f"Chunk index has dimension {store.matrix.shape[1]}, "
            f"expected {TEXT_EMBEDDING_3_SMALL_DIM} for {DEFAULT_EMBEDDING_MODEL}."
        )
    covered = 0
    for row_id, row in zip(store.row_ids, store.rows, strict=True):
        if row_id in chunk_by_id or str(row.get("content", "") or "").strip():
            covered += 1
    if covered == 0:
        raise ValueError(
            "Chunk vector index has no IDs that match the corpus and does not contain inline content."
        )
    if covered < len(store.row_ids):
        print(
            "warning "
            f"chunk_index_coverage={covered}/{len(store.row_ids)}; "
            "unmatched rows will have empty retrieved content"
        )


def load_corpus_chunks(path: Path) -> list[ChunkRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    records.append(json.loads(text))
        return _records_to_chunks(records)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            chunks: list[ChunkRecord] = []
            for key, value in payload.items():
                if isinstance(value, dict):
                    content = _extract_content(value)
                    metadata = {k: v for k, v in value.items() if k != "content"}
                else:
                    content = str(value)
                    metadata = {}
                if content.strip():
                    chunks.append(ChunkRecord(chunk_id=str(key), content=content.strip(), metadata=metadata))
            return chunks
        if isinstance(payload, list):
            return _records_to_chunks(payload)
        raise ValueError(f"Unsupported JSON corpus payload in {path}: {type(payload).__name__}")

    chunks = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        text = line.strip()
        if text:
            chunks.append(ChunkRecord(chunk_id=f"chunk-{index}", content=text, metadata={"line_index": index}))
    return chunks


def _records_to_chunks(records: list[Any]) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for index, record in enumerate(records):
        if isinstance(record, dict):
            content = _extract_content(record)
            chunk_id = _extract_chunk_id(record, index, content)
            metadata = {key: value for key, value in record.items() if key not in {"content", "text", "passage"}}
        else:
            content = str(record)
            chunk_id = f"chunk-{index}"
            metadata = {"record_index": index}
        if content.strip():
            chunks.append(ChunkRecord(chunk_id=chunk_id, content=content.strip(), metadata=metadata))
    return chunks


def _extract_content(record: dict[str, Any]) -> str:
    for key in ("content", "text", "passage", "chunk", "body"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"Corpus record is missing a text field: {list(record.keys())}")


def _extract_chunk_id(record: dict[str, Any], index: int, content: str) -> str:
    for key in ("__id__", "chunk_id", "id", "doc_id", "_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"chunk-{index}"


def load_questions(path: Path, start_index: int, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Question file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        payload: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    payload.append(json.loads(text))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(payload).__name__}")
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    selected = payload[start_index:] if limit == 0 else payload[start_index : start_index + limit]
    return [_coerce_question_record(item, start_index + offset) for offset, item in enumerate(selected)]


def _coerce_question_record(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"qid": str(index), "question": item}
    raise ValueError(f"Question record at index {index} must be an object or string.")


def build_result_record(
    *,
    item: dict[str, Any],
    question_index: int,
    question: str,
    answer: str,
    final_answer: dict[str, Any],
    retrieved_contexts: list[dict[str, Any]],
    run_dir: Path | None,
    run_status: str,
) -> dict[str, Any]:
    retrieved_knowledge = "\n\n".join(
        str(context.get("content", "") or "").strip()
        for context in retrieved_contexts
        if str(context.get("content", "") or "").strip()
    )
    retrieved_for_eval = [
        {
            "chunk_id": context.get("chunk_id", ""),
            "content": context.get("content", ""),
            "score": context.get("score"),
            "rank": context.get("rank"),
            "source_node_ids": [],
            "source_edge_ids": [],
            "thought_status": "standard_rag",
        }
        for context in retrieved_contexts
    ]
    reasoning_summary = str(final_answer.get("reasoning_summary", "") or "").strip()
    return {
        "qid": infer_qid(item, question_index),
        "question": question,
        "golden_answers": extract_gold_answers(item),
        "context": list(item.get("context", [])) if isinstance(item.get("context"), list) else [],
        "nhops": item.get("nhops"),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "run_status": run_status,
        "answer": answer,
        "predicted_answer": answer,
        "generation": f"<answer>{answer}</answer>" if answer else "",
        "generation_explanation": reasoning_summary,
        "confidence": _clamp_float(final_answer.get("confidence", 0.0)),
        "retrieved": retrieved_for_eval,
        "retrieved_contexts": retrieved_contexts,
        "retrieved_knowledge": retrieved_knowledge,
    }


def build_evidence_subgraph(retrieved_contexts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "evidence": [
            {
                "chunk_id": context.get("chunk_id", ""),
                "content": context.get("content", ""),
                "score": context.get("score"),
                "rank": context.get("rank"),
                "source_node_ids": [],
                "source_edge_ids": [],
                "thought_id": "standard_rag",
            }
            for context in retrieved_contexts
        ]
    }


def failure_record(item: dict[str, Any], question_index: int, exc: Exception) -> dict[str, Any]:
    question = str(item.get("question", "") or "").strip()
    return {
        "qid": infer_qid(item, question_index),
        "question": question,
        "golden_answers": extract_gold_answers(item),
        "run_dir": None,
        "run_status": "failed",
        "answer": "",
        "predicted_answer": "",
        "generation": "",
        "generation_explanation": "",
        "retrieved": [],
        "retrieved_contexts": [],
        "retrieved_knowledge": "",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def save_results(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    path.write_text(pretty_json(records), encoding="utf-8")


def _prepare_paths(project_root: Path, config: Config, args: argparse.Namespace) -> PreparedPaths:
    dataset_root = config.dataset.root
    dataset_name = dataset_root.name
    if args.dataset:
        dataset_name = args.dataset
        dataset_root = _resolve_path(project_root, Path("datasets") / args.dataset)
    if args.dataset_root:
        dataset_root = _resolve_path(project_root, args.dataset_root)
        dataset_name = dataset_root.name

    question_file = _resolve_path(project_root, args.question_file) if args.question_file else _default_question_file(project_root, dataset_name)
    corpus_path = _resolve_path(project_root, args.corpus_path) if args.corpus_path else dataset_root / config.dataset.text_chunk_file
    index_path = _resolve_path(project_root, args.index_path) if args.index_path else dataset_root / config.dataset.chunk_vdb_file

    default_runs_dir = project_root / "runs" / "StandardRAG" / dataset_name
    runs_dir = _resolve_path(project_root, args.runs_dir) if args.runs_dir else default_runs_dir
    output_path = (
        _resolve_path(project_root, args.output_path)
        if args.output_path
        else runs_dir / "generated_answer.json"
    )
    return PreparedPaths(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        question_file=question_file,
        corpus_path=corpus_path,
        index_path=index_path,
        output_path=output_path,
        runs_dir=runs_dir,
    )


def _default_question_file(project_root: Path, dataset_name: str) -> Path:
    base = project_root / "questions" / dataset_name
    for name in ("hyperrag_query_test.json", "questions.json"):
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / "questions.json"


def _apply_overrides(config: Config, args: argparse.Namespace) -> None:
    config.llm.model = args.model
    config.llm.embedding_model = args.embedding_model
    config.llm.temperature = args.temperature
    if args.api_key:
        os.environ[config.llm.api_key_env] = args.api_key
    if args.base_url:
        os.environ[config.llm.base_url_env] = args.base_url.rstrip("/")


def _validate_openai_env(config: Config) -> None:
    if not os.getenv(config.llm.api_key_env, "").strip():
        raise RuntimeError(
            f"Environment variable {config.llm.api_key_env} is required. "
            "Set OPENAI_API_KEY or pass --api-key."
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.max_answer_tokens <= 0:
        raise ValueError("--max-answer-tokens must be > 0")
    if args.embedding_model != DEFAULT_EMBEDDING_MODEL:
        raise ValueError("Standard RAG baseline is fixed to text-embedding-3-small.")
    if args.model != DEFAULT_CHAT_MODEL:
        raise ValueError("Standard RAG baseline is fixed to gpt-4o-mini.")


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _existing_success_run(output_dir: Path, question: str) -> Path | None:
    if not output_dir.exists():
        return None
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


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _clamp_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def infer_qid(item: dict[str, Any], fallback_index: int) -> str:
    for key in ("qid", "question_id", "id", "_id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(fallback_index)


def extract_gold_answers(item: dict[str, Any]) -> list[str]:
    for key in ("golden_answers", "answers", "answer"):
        value = item.get(key)
        if isinstance(value, list):
            return [str(answer).strip() for answer in value if str(answer).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
    return []


if __name__ == "__main__":
    raise SystemExit(main())
