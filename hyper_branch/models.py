"""数据加载和检索模块共用的小型图、向量和证据值对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class VectorMatch:
    item_id: str
    label: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class GraphNode:
    node_id: str
    role: str
    source_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    role: str
