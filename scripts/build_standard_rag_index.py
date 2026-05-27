from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.StandardRAG.standard_rag import (
    DEFAULT_EMBEDDING_MODEL,
    _resolve_path,
    _validate_openai_env,
    load_corpus_chunks,
    load_or_build_chunk_store,
    validate_chunk_index_coverage,
)
from hyper_branch.config import load_config
from hyper_branch.data.vector_store import VectorStore
from hyper_branch.llm import OpenAICompatibleClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reusable flat chunk/passage vector index for Standard RAG."
    )
    parser.add_argument("--config", default="configs/2wikimultihopqa.yaml")
    parser.add_argument("--dataset", default="", help="Dataset name under datasets/.")
    parser.add_argument("--dataset-root", default="", help="Explicit dataset root.")
    parser.add_argument("--corpus-path", default="", help="Raw chunk/passage corpus path.")
    parser.add_argument("--index-path", default="", help="Output vdb_chunks-style JSON path.")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--api-key", default="", help="Optional API key override. Prefer OPENAI_API_KEY.")
    parser.add_argument("--base-url", default="", help="Optional OpenAI-compatible base URL override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be > 0")

    project_root = Path.cwd()
    config = load_config(_resolve_path(project_root, args.config), project_root)
    config.llm.embedding_model = DEFAULT_EMBEDDING_MODEL
    if args.api_key:
        os.environ[config.llm.api_key_env] = args.api_key
    if args.base_url:
        os.environ[config.llm.base_url_env] = args.base_url.rstrip("/")

    dataset_root = config.dataset.root
    if args.dataset:
        dataset_root = _resolve_path(project_root, Path("datasets") / args.dataset)
    if args.dataset_root:
        dataset_root = _resolve_path(project_root, args.dataset_root)

    corpus_path = _resolve_path(project_root, args.corpus_path) if args.corpus_path else dataset_root / config.dataset.text_chunk_file
    index_path = _resolve_path(project_root, args.index_path) if args.index_path else dataset_root / config.dataset.chunk_vdb_file

    chunks = load_corpus_chunks(corpus_path)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if index_path.exists():
        store = VectorStore.from_json(index_path, name="chunks", label_fields=("__id__",))
        validate_chunk_index_coverage(store, chunk_by_id)
        print(f"existing_index={index_path}")
        print(f"chunks={len(chunks)} vectors={len(store.rows)} dim={store.matrix.shape[1]}")
        return 0

    _validate_openai_env(config)
    client = OpenAICompatibleClient(config.llm)
    store = load_or_build_chunk_store(
        index_path,
        chunks,
        client,
        config.llm.embedding_model,
        args.embedding_batch_size,
    )
    validate_chunk_index_coverage(store, chunk_by_id)
    print(f"built_index={index_path}")
    print(f"chunks={len(chunks)} vectors={len(store.rows)} dim={store.matrix.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
