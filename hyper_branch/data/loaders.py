"""加载一份 HyperBranch 数据集：GraphML、来源文本和预计算向量库。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import DatasetConfig
from .graph import KnowledgeHypergraph
from .vector_store import VectorStore


@dataclass(slots=True)
class DatasetBundle:
    """一次运行内由所有原子问题共享的数据集资源。"""

    graph: KnowledgeHypergraph
    text_chunks: dict[str, dict[str, Any]]
    entity_store: VectorStore
    hyperedge_store: VectorStore

    def get_chunk_text(self, chunk_id: str) -> str:
        return str(self.text_chunks.get(chunk_id, {}).get("content", ""))


class HypergraphDatasetLoader:
    """校验并加载检索器所需的全部磁盘资源。"""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def load(self) -> DatasetBundle:
        """一次加载图和向量索引，并返回用于记录产物的数据集摘要。"""

        root = self.config.root
        graph_path = self._resolve_graph_path(root)

        # 文本记录和全部向量库必须对应同一份图快照。
        text_chunks = self._load_json(root / self.config.text_chunk_file)
        entity_vdb_path = root / self.config.entity_vdb_file
        #加载超图
        graph = KnowledgeHypergraph.from_graphml(graph_path)
        #加载实体向量库
        entity_store = VectorStore.from_json(entity_vdb_path, name="entity_names", label_fields=("entity_name",))
        #加载超边向量库
        hyperedge_store = VectorStore.from_json(
            root / self.config.hyperedge_vdb_file,
            name="hyperedges",
            label_fields=("hyperedge_name",),
        )
        return DatasetBundle(
            graph=graph,
            text_chunks=text_chunks,
            entity_store=entity_store,
            hyperedge_store=hyperedge_store,
        )

    def _resolve_graph_path(self, root: Path) -> Path:
        """优先使用显式 GraphML 文件，否则选标准文件名或最新文件。"""

        return root / self.config.graphml_file

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))
