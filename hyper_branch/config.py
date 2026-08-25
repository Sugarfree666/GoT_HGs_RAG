"""加载 YAML 配置，并转换为 HyperBranch 使用的类型化路径和运行参数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class DatasetConfig:
    root: Path
    graphml_file: str = "graph_chunk_entity_relation.graphml"
    text_chunk_file: str = "kv_store_text_chunks.json"
    hyperedge_vdb_file: str = "vdb_hyperedges.json"
    entity_vdb_file: str = "vdb_entity_names.json"


@dataclass(slots=True)
class RetrievalConfig:
    local_hyperedge_top_k: int = 3
    entity_link_vector_top_k: int = 1
    entity_link_vector_min_score: float = 0.6


@dataclass(slots=True)
class LLMConfig:
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    timeout_seconds: int = 120
    temperature: float = 0.2


@dataclass(slots=True)
class Config:
    dataset: DatasetConfig
    retrieval: RetrievalConfig
    llm: LLMConfig


def load_config(config_path: Path, project_root: Path) -> Config:
    """读取一份 YAML、补齐默认值，并从项目根目录解析相对路径。"""

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset = raw.get("dataset", {})
    retrieval = raw.get("retrieval", {})
    llm = raw.get("llm", {})

    # 原始 YAML 只在这里处理；其余模块只消费类型化配置。
    dataset_cfg = DatasetConfig(
        root=_resolve_path(project_root, dataset.get("root", "datasets/agriculture")),
        graphml_file=str(dataset.get("graphml_file", "graph_chunk_entity_relation.graphml")),
        text_chunk_file=dataset.get("text_chunk_file", "kv_store_text_chunks.json"),
        hyperedge_vdb_file=dataset.get("hyperedge_vdb_file", "vdb_hyperedges.json"),
        entity_vdb_file=dataset.get("entity_vdb_file", "vdb_entity_names.json"),
    )
    retrieval_cfg = RetrievalConfig(
        local_hyperedge_top_k=int(retrieval.get("local_hyperedge_top_k", 3)),
        entity_link_vector_top_k=int(retrieval.get("entity_link_vector_top_k", 1)),
        entity_link_vector_min_score=float(retrieval.get("entity_link_vector_min_score", 0.6)),
    )
    llm_cfg = LLMConfig(
        api_key_env=str(llm.get("api_key_env", "OPENAI_API_KEY")),
        base_url_env=str(llm.get("base_url_env", "OPENAI_BASE_URL")),
        model=str(llm.get("model", "gpt-4o-mini")),
        embedding_model=str(llm.get("embedding_model", "text-embedding-3-small")),
        timeout_seconds=int(llm.get("timeout_seconds", 120)),
        temperature=float(llm.get("temperature", 0.2)),
    )
    return Config(
        dataset=dataset_cfg,
        retrieval=retrieval_cfg,
        llm=llm_cfg,
    )


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    """稳定地解析配置路径，同时保留显式的绝对路径。"""

    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()
