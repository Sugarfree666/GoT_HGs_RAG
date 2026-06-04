from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FALLBACK_TEXT_FIELDS = ("context", "sentence", "content")


@dataclass(slots=True)
class ConstructConfig:
    input_path: Path
    output_dir: Path
    input_format: str = "auto"
    text_field: str = "text"
    fallback_text_fields: tuple[str, ...] = DEFAULT_FALLBACK_TEXT_FIELDS
    limit: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"
    language: str | None = None
    max_concurrency: int | None = None
    llm_max_async: int = 16
    embedding_max_async: int = 16
    resume: bool = True


def construct_hypergraph(config: ConstructConfig) -> dict[str, Any]:
    _validate_positive("max_concurrency", config.max_concurrency, allow_none=True)
    _validate_positive("llm_max_async", config.llm_max_async)
    _validate_positive("embedding_max_async", config.embedding_max_async)
    contexts = read_contexts(
        config.input_path,
        input_format=config.input_format,
        text_field=config.text_field,
        fallback_text_fields=config.fallback_text_fields,
        limit=config.limit,
    )
    if not contexts:
        raise ValueError(f"No non-empty contexts were read from {config.input_path}.")

    _apply_openai_environment(config)

    from .hypergraphrag import HyperGraphRAG
    from .hypergraphrag.llm import openai_complete, openai_embedding

    llm_kwargs = _compact_dict(
        {
            "base_url": config.base_url,
            "api_key": config.api_key,
            "temperature": config.temperature,
        }
    )
    embedding_func = _make_openai_embedding_func(
        openai_embedding,
        model=config.embedding_model,
        base_url=config.base_url,
        api_key=config.api_key,
        concurrent_limit=config.embedding_max_async,
    )
    addon_params = {}
    if config.language:
        addon_params["language"] = config.language
    if config.max_concurrency is not None:
        addon_params["max_concurrency"] = config.max_concurrency
    if config.resume:
        addon_params["resume"] = True

    rag = HyperGraphRAG(
        working_dir=str(config.output_dir),
        llm_model_func=openai_complete,
        llm_model_name=config.llm_model,
        llm_model_kwargs=llm_kwargs,
        embedding_func=embedding_func,
        chunk_token_size=config.chunk_token_size,
        chunk_overlap_token_size=config.chunk_overlap_token_size,
        tiktoken_model_name=config.tiktoken_model_name,
        llm_model_max_async=config.llm_max_async,
        embedding_func_max_async=config.embedding_max_async,
        addon_params=addon_params,
    )
    rag.insert(contexts)

    return {
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "document_count": len(contexts),
        "graphml_file": str(config.output_dir / "graph_chunk_entity_relation.graphml"),
        "text_chunk_file": str(config.output_dir / "kv_store_text_chunks.json"),
        "hyperedge_vdb_file": str(config.output_dir / "vdb_hyperedges.json"),
    }


def read_contexts(
    path: Path,
    *,
    input_format: str = "auto",
    text_field: str = "text",
    fallback_text_fields: tuple[str, ...] = DEFAULT_FALLBACK_TEXT_FIELDS,
    limit: int | None = None,
) -> list[str]:
    resolved_format = _resolve_input_format(path, input_format)
    records = _read_records(path, resolved_format)
    contexts: list[str] = []
    for record in records:
        text = _extract_text(record, text_field, fallback_text_fields)
        if not text:
            continue
        contexts.append(text)
        if limit is not None and len(contexts) >= limit:
            break
    return contexts


def _read_records(path: Path, input_format: str) -> Iterable[Any]:
    if input_format == "txt":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield line
        return

    if input_format == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        return

    if input_format == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield from _iter_json_payload(payload)
        return

    raise ValueError(f"Unsupported input format: {input_format}")


def _iter_json_payload(payload: Any) -> Iterable[Any]:
    if isinstance(payload, list):
        yield from payload
        return
    if isinstance(payload, dict):
        for key in ("data", "documents", "contexts", "texts", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                yield from value
                return
        yield payload
        return
    yield payload


def _extract_text(record: Any, text_field: str, fallback_text_fields: tuple[str, ...]) -> str:
    if isinstance(record, dict):
        for field in (text_field, *fallback_text_fields):
            text = _coerce_text(record.get(field))
            if text:
                return text
        return ""
    return _coerce_text(record)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_coerce_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for field in ("text", "content", "sentence"):
            text = _coerce_text(value.get(field))
            if text:
                return text
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value).strip()


def _make_openai_embedding_func(
    openai_embedding_func: Any,
    *,
    model: str,
    base_url: str | None,
    api_key: str | None,
    concurrent_limit: int,
) -> Any:
    from .hypergraphrag.utils import EmbeddingFunc

    raw_embedding_func = getattr(openai_embedding_func, "func", openai_embedding_func)

    async def embedding(texts: list[str]):
        return await raw_embedding_func(
            texts,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    return EmbeddingFunc(
        embedding_dim=getattr(openai_embedding_func, "embedding_dim", 1536),
        max_token_size=getattr(openai_embedding_func, "max_token_size", 8192),
        func=embedding,
        concurrent_limit=concurrent_limit,
    )


def _resolve_input_format(path: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix in {".txt", ".text"}:
        return "txt"
    raise ValueError(f"Cannot infer input format from suffix: {path.suffix}")


def _apply_openai_environment(config: ConstructConfig) -> None:
    if config.api_key:
        os.environ["OPENAI_API_KEY"] = config.api_key
    if config.base_url:
        os.environ["OPENAI_BASE_URL"] = config.base_url
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Set it in the environment or pass --api-key.")


def _compact_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _validate_positive(name: str, value: int | None, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if value is None or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct a HyperBranch hypergraph with the HyperRAG method.")
    parser.add_argument("--input", required=True, help="Input corpus file: .jsonl, .json, or .txt.")
    parser.add_argument("--output", required=True, help="Output dataset directory for GraphML, KV, and vector files.")
    parser.add_argument("--input-format", choices=("auto", "jsonl", "json", "txt"), default="auto")
    parser.add_argument("--text-field", default="text", help="Primary text field for JSON/JSONL records.")
    parser.add_argument(
        "--fallback-text-fields",
        default=",".join(DEFAULT_FALLBACK_TEXT_FIELDS),
        help="Comma-separated fallback text fields for JSON/JSONL records.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of contexts to insert.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key. Prefer OPENAI_API_KEY.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL. Prefer OPENAI_BASE_URL.")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature for graph construction.")
    parser.add_argument("--chunk-token-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap-token-size", type=int, default=100)
    parser.add_argument("--tiktoken-model-name", default="gpt-4o-mini")
    parser.add_argument("--language", default=None, help="Optional extraction language hint.")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Max chunk extraction tasks in flight.")
    parser.add_argument("--llm-max-async", type=int, default=16, help="Max concurrent LLM calls.")
    parser.add_argument("--embedding-max-async", type=int, default=16, help="Max concurrent embedding calls.")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Enable checkpoint files for resumable entity extraction. Enabled by default.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable entity extraction checkpoint files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ConstructConfig(
        input_path=Path(args.input).resolve(),
        output_dir=Path(args.output).resolve(),
        input_format=args.input_format,
        text_field=args.text_field,
        fallback_text_fields=tuple(
            field.strip() for field in args.fallback_text_fields.split(",") if field.strip()
        ),
        limit=args.limit,
        api_key=args.api_key,
        base_url=args.base_url,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap_token_size,
        tiktoken_model_name=args.tiktoken_model_name,
        language=args.language,
        max_concurrency=args.max_concurrency,
        llm_max_async=args.llm_max_async,
        embedding_max_async=args.embedding_max_async,
        resume=args.resume,
    )
    summary = construct_hypergraph(config)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
