"""数据加载和检索模块共用的小型图、向量和证据值对象。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .utils import normalize_label


@dataclass(slots=True)
class VectorMatch:
    item_id: str
    label: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["display_label"] = normalize_label(self.label)
        return payload


@dataclass(slots=True)
class GraphNode:
    node_id: str
    role: str
    weight: float = 0.0
    source_ids: list[str] = field(default_factory=list)
    entity_type: str | None = None
    description: str | None = None

    @property
    def display_label(self) -> str:
        return normalize_label(self.node_id)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["display_label"] = self.display_label
        return payload


@dataclass(slots=True)
class GraphEdge:
    edge_id: str
    source: str
    target: str
    role: str
    weight: float = 0.0
    source_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_display"] = normalize_label(self.source)
        payload["target_display"] = normalize_label(self.target)
        return payload


@dataclass(slots=True)
class EvidenceItem:
    evidence_id: str
    chunk_id: str
    content: str
    score: float
    source_node_ids: list[str] = field(default_factory=list)
    source_edge_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_nodes"] = [normalize_label(node_id) for node_id in self.source_node_ids]
        return payload
