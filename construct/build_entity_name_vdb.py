from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


if __package__ in {None, ""}:  # Support: python construct/build_entity_name_vdb.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from construct.builder import _make_openai_embedding_func
    from construct.hypergraphrag.llm import openai_embedding
    from construct.hypergraphrag.storage import NanoVectorDBStorage
    from construct.hypergraphrag.utils import compute_mdhash_id
else:
    from .builder import _make_openai_embedding_func
    from .hypergraphrag.llm import openai_embedding
    from .hypergraphrag.storage import NanoVectorDBStorage
    from .hypergraphrag.utils import compute_mdhash_id


_DEFAULT_GRAPHML_FILE = "graph_chunk_entity_relation.graphml"
_OUTPUT_FILE = "vdb_entity_names.json"
_STAGING_DIR = ".entity_name_vdb_build"
_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(slots=True)
class EntityNameVDBConfig:
    dataset_dir: Path
    graphml_path: Path | None = None
    api_key: str | None = None
    base_url: str | None = None
    embedding_batch_size: int = 128
    upsert_batch_size: int = 8192
    embedding_max_async: int = 16
    rebuild: bool = False

    @property
    def resolved_graphml_path(self) -> Path:
        return (self.graphml_path or self.dataset_dir / _DEFAULT_GRAPHML_FILE).resolve()

    @property
    def output_path(self) -> Path:
        return (self.dataset_dir / _OUTPUT_FILE).resolve()


def iter_graphml_entity_names(path: Path) -> Iterator[str]:
    """Yield entity node IDs without loading the full GraphML into memory."""
    key_names: dict[str, str] = {}
    graph_element: ET.Element | None = None
    for event, element in ET.iterparse(path, events=("start", "end")):
        tag = _local_xml_name(element.tag)
        if event == "start" and tag == "graph":
            graph_element = element
            continue
        if event == "start" and tag == "key":
            key_id = str(element.attrib.get("id", ""))
            attr_name = str(element.attrib.get("attr.name", key_id))
            if key_id:
                key_names[key_id] = attr_name
            continue
        if event != "end" or tag != "node":
            if event == "end" and tag == "edge":
                element.clear()
                if graph_element is not None:
                    graph_element.remove(element)
            continue

        role = ""
        for child in element:
            if _local_xml_name(child.tag) != "data":
                continue
            key_id = str(child.attrib.get("key", ""))
            if key_names.get(key_id, key_id) == "role":
                role = str(child.text or "").strip().lower()
                break
        if role == "entity":
            entity_name = str(element.attrib.get("id", "")).strip()
            if entity_name:
                yield entity_name
        element.clear()
        if graph_element is not None:
            graph_element.remove(element)


def build_entity_name_vdb(
    config: EntityNameVDBConfig,
    *,
    embedding_func: Any | None = None,
) -> dict[str, Any]:
    return asyncio.run(_build_entity_name_vdb(config, embedding_func=embedding_func))


async def _build_entity_name_vdb(
    config: EntityNameVDBConfig,
    *,
    embedding_func: Any | None = None,
) -> dict[str, Any]:
    _validate_config(config)
    dataset_dir = config.dataset_dir.resolve()
    graphml_path = config.resolved_graphml_path
    output_path = config.output_path
    dataset_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, str]] = {}
    for entity_name in iter_graphml_entity_names(graphml_path):
        record_id = compute_mdhash_id(entity_name, prefix="en-")
        existing = records.get(record_id)
        if existing is not None and existing["entity_name"] != entity_name:
            raise ValueError(f"Entity-name hash collision: {existing['entity_name']!r} and {entity_name!r}.")
        records[record_id] = {
            "content": entity_name,
            "entity_name": entity_name,
        }
    if not records:
        raise ValueError(f"No entity nodes were found in {graphml_path}.")

    if embedding_func is None:
        embedding_func = _openai_embedding_func(config)

    staging_dir = dataset_dir / _STAGING_DIR
    staging_path = staging_dir / _OUTPUT_FILE
    if config.rebuild and staging_path.exists():
        staging_path.unlink()

    if not config.rebuild and not staging_path.exists() and output_path.exists():
        final_storage = _new_storage(dataset_dir, config, embedding_func)
        missing_from_final = await final_storage.filter_keys(list(records))
        if not missing_from_final:
            stored_count = len(final_storage.client_storage.get("data", []))
            return _summary(
                dataset_dir=dataset_dir,
                graphml_path=graphml_path,
                output_path=output_path,
                entity_count=len(records),
                existing_count=len(records),
                embedded_count=0,
                stored_count=stored_count,
            )
        del final_storage
        gc.collect()
        staging_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, staging_path)

    staging_dir.mkdir(parents=True, exist_ok=True)
    storage = _new_storage(staging_dir, config, embedding_func)
    missing_ids = await storage.filter_keys(list(records))
    missing_records = [(record_id, record) for record_id, record in records.items() if record_id in missing_ids]
    existing_count = len(records) - len(missing_records)
    embedded_count = 0

    try:
        for start in range(0, len(missing_records), config.upsert_batch_size):
            batch = dict(missing_records[start : start + config.upsert_batch_size])
            await storage.upsert(batch)
            embedded_count += len(batch)
    except Exception:
        if staging_path.exists() or embedded_count:
            await storage.index_done_callback()
        raise

    await storage.index_done_callback()

    stored_ids = {
        str(row.get("__id__", ""))
        for row in storage.client_storage.get("data", [])
        if row.get("__id__")
    }
    absent_ids = set(records).difference(stored_ids)
    if absent_ids:
        raise RuntimeError(f"Entity-name vector DB is missing {len(absent_ids)} expected records after saving.")

    os.replace(staging_path, output_path)
    try:
        staging_dir.rmdir()
    except OSError:
        pass

    return _summary(
        dataset_dir=dataset_dir,
        graphml_path=graphml_path,
        output_path=output_path,
        entity_count=len(records),
        existing_count=existing_count,
        embedded_count=embedded_count,
        stored_count=len(stored_ids),
    )


def _new_storage(dataset_dir: Path, config: EntityNameVDBConfig, embedding_func: Any) -> NanoVectorDBStorage:
    return NanoVectorDBStorage(
        namespace="entity_names",
        global_config={
            "working_dir": str(dataset_dir),
            "embedding_batch_num": config.embedding_batch_size,
        },
        embedding_func=embedding_func,
        meta_fields={"entity_name"},
    )


def _summary(
    *,
    dataset_dir: Path,
    graphml_path: Path,
    output_path: Path,
    entity_count: int,
    existing_count: int,
    embedded_count: int,
    stored_count: int,
) -> dict[str, Any]:
    return {
        "dataset_dir": str(dataset_dir),
        "graphml_path": str(graphml_path),
        "output_path": str(output_path),
        "embedding_model": _EMBEDDING_MODEL,
        "entity_count": entity_count,
        "existing_count": existing_count,
        "embedded_count": embedded_count,
        "stored_count": stored_count,
    }


def _openai_embedding_func(config: EntityNameVDBConfig) -> Any:
    if config.api_key:
        os.environ["OPENAI_API_KEY"] = config.api_key
    if config.base_url:
        os.environ["OPENAI_BASE_URL"] = config.base_url
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Set it in the environment or pass --api-key.")
    return _make_openai_embedding_func(
        openai_embedding,
        model=_EMBEDDING_MODEL,
        base_url=config.base_url,
        api_key=config.api_key,
        concurrent_limit=config.embedding_max_async,
    )


def _validate_config(config: EntityNameVDBConfig) -> None:
    for name, value in (
        ("embedding_batch_size", config.embedding_batch_size),
        ("upsert_batch_size", config.upsert_batch_size),
        ("embedding_max_async", config.embedding_max_async),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not config.resolved_graphml_path.is_file():
        raise FileNotFoundError(f"GraphML file does not exist: {config.resolved_graphml_path}")


def _local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build vdb_entity_names.json by embedding only the entity node IDs from an existing HyperBranch GraphML."
        )
    )
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory that will receive vdb_entity_names.json.")
    parser.add_argument(
        "--graphml",
        default=None,
        help=f"GraphML path. Defaults to <dataset-dir>/{_DEFAULT_GRAPHML_FILE}.",
    )
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key. Prefer OPENAI_API_KEY.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL. Prefer OPENAI_BASE_URL.")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--upsert-batch-size", type=int, default=8192)
    parser.add_argument("--embedding-max-async", type=int, default=16)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Build a fresh staged index; the existing final file remains available until replacement succeeds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_entity_name_vdb(
        EntityNameVDBConfig(
            dataset_dir=Path(args.dataset_dir),
            graphml_path=Path(args.graphml) if args.graphml else None,
            api_key=args.api_key,
            base_url=args.base_url,
            embedding_batch_size=args.embedding_batch_size,
            upsert_batch_size=args.upsert_batch_size,
            embedding_max_async=args.embedding_max_async,
            rebuild=args.rebuild,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
