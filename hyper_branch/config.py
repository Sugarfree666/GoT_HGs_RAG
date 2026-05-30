from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    root: Path
    graphml_file: str | None = None
    full_doc_file: str = "kv_store_full_docs.json"
    text_chunk_file: str = "kv_store_text_chunks.json"
    hyperedge_vdb_file: str = "vdb_hyperedges.json"
    entity_vdb_file: str = "vdb_entity_names.json"
    entity_vdb_fallback_file: str = "vdb_entities.json"
    chunk_vdb_file: str = "vdb_chunks.json"


@dataclass(slots=True)
class RuntimeConfig:
    base_run_dir: Path
    log_level: str = "INFO"


@dataclass(slots=True)
class RetrievalConfig:
    relation_top_k: int = 10
    semantic_top_k: int = 10
    evidence_top_k: int = 5
    max_anchor_hyperedges_per_entity: int | None = None
    anchor_entity_top_k: int = 3
    anchor_entity_llm_min_confidence: float = 0.6
    anchor_weight: float = 0.4
    relation_weight: float = 0.4
    semantic_weight: float = 0.2


@dataclass(slots=True)
class LLMConfig:
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    temperature: float = 0.2
    use_mock: bool = False


@dataclass(slots=True)
class PromptConfig:
    directory: Path


@dataclass(slots=True)
class Config:
    project_root: Path
    dataset: DatasetConfig
    runtime: RuntimeConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    prompts: PromptConfig


def load_config(config_path: Path, project_root: Path) -> Config:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset = raw.get("dataset", {})
    runtime = raw.get("runtime", {})
    retrieval = raw.get("retrieval", {})
    llm = raw.get("llm", {})
    prompts = raw.get("prompts", {})

    dataset_cfg = DatasetConfig(
        root=_resolve_path(project_root, dataset.get("root", "datasets/agriculture")),
        graphml_file=dataset.get("graphml_file"),
        full_doc_file=dataset.get("full_doc_file", "kv_store_full_docs.json"),
        text_chunk_file=dataset.get("text_chunk_file", "kv_store_text_chunks.json"),
        hyperedge_vdb_file=dataset.get("hyperedge_vdb_file", "vdb_hyperedges.json"),
        entity_vdb_file=dataset.get("entity_vdb_file", "vdb_entity_names.json"),
        entity_vdb_fallback_file=dataset.get("entity_vdb_fallback_file", "vdb_entities.json"),
        chunk_vdb_file=dataset.get("chunk_vdb_file", "vdb_chunks.json"),
    )
    runtime_cfg = RuntimeConfig(
        base_run_dir=_resolve_path(project_root, runtime.get("base_run_dir", "runs")),
        log_level=str(runtime.get("log_level", "INFO")).upper(),
    )
    retrieval_cfg = RetrievalConfig(
        relation_top_k=int(retrieval.get("relation_top_k", 10)),
        semantic_top_k=int(retrieval.get("semantic_top_k", 10)),
        evidence_top_k=int(retrieval.get("evidence_top_k", 5)),
        max_anchor_hyperedges_per_entity=_optional_int(retrieval.get("max_anchor_hyperedges_per_entity")),
        anchor_entity_top_k=int(retrieval.get("anchor_entity_top_k", 3)),
        anchor_entity_llm_min_confidence=float(retrieval.get("anchor_entity_llm_min_confidence", 0.6)),
        anchor_weight=float(retrieval.get("anchor_weight", _nested_weight(retrieval, "anchor", 0.4))),
        relation_weight=float(retrieval.get("relation_weight", _nested_weight(retrieval, "relation", 0.4))),
        semantic_weight=float(retrieval.get("semantic_weight", _nested_weight(retrieval, "semantic", 0.2))),
    )
    llm_cfg = LLMConfig(
        api_key_env=str(llm.get("api_key_env", "OPENAI_API_KEY")),
        base_url_env=str(llm.get("base_url_env", "OPENAI_BASE_URL")),
        model=str(llm.get("model", "gpt-4o-mini")),
        embedding_model=str(llm.get("embedding_model", "text-embedding-3-small")),
        timeout_seconds=int(llm.get("timeout_seconds", 120)),
        max_retries=int(llm.get("max_retries", 3)),
        retry_backoff_seconds=float(llm.get("retry_backoff_seconds", 2.0)),
        temperature=float(llm.get("temperature", 0.2)),
        use_mock=bool(llm.get("use_mock", False)),
    )
    prompt_cfg = PromptConfig(directory=_resolve_path(project_root, prompts.get("dir", "prompts")))
    return Config(
        project_root=project_root,
        dataset=dataset_cfg,
        runtime=runtime_cfg,
        retrieval=retrieval_cfg,
        llm=llm_cfg,
        prompts=prompt_cfg,
    )


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _nested_weight(retrieval: dict[str, Any], key: str, default: float) -> float:
    weights = retrieval.get("fusion_weights", {})
    if isinstance(weights, dict):
        return float(weights.get(key, default))
    return default
