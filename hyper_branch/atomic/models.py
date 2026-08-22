"""原子执行器内部使用的类型化 DAG、分析、候选和答案记录。"""


from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AtomicQuestionNode:
    node_id: str
    question: str
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "question": self.question,
            "dependencies": list(self.dependencies),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class AtomicQuestionAnalysis:
    entities: list[str] = field(default_factory=list)
    answer_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": list(self.entities),
            "answer_type": self.answer_type,
        }

@dataclass(slots=True)
class FusedHyperedgeCandidate:
    hyperedge_id: str
    hyperedge_text: str
    branch_support: set[str] = field(default_factory=set)
    semantic_score: float = 0.0
    entity_ids: list[str] = field(default_factory=list)
    entity_records: list[dict[str, Any]] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    chunk_texts: list[str] = field(default_factory=list)
    evidence_texts: list[str] = field(default_factory=list)
    rank: int | None = None
    score_breakdown: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hyperedge_id": self.hyperedge_id,
            "hyperedge_text": self.hyperedge_text,
            "branch_support": sorted(self.branch_support),
            "semantic_score": self.semantic_score,
            "entity_ids": list(self.entity_ids),
            "entity_records": [dict(item) for item in self.entity_records],
            "chunk_ids": list(self.chunk_ids),
            "chunk_texts": list(self.chunk_texts),
            "evidence_texts": list(self.evidence_texts),
            "rank": self.rank,
            "score_breakdown": dict(self.score_breakdown),
        }


@dataclass(slots=True)
class AtomicAnswerResult:
    node_id: str
    question: str
    answer: str
    used_dependencies: list[str] = field(default_factory=list)
    used_hyperedge_ids: list[str] = field(default_factory=list)
    insufficient: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "question": self.question,
            "answer": self.answer,
            "used_dependencies": list(self.used_dependencies),
            "used_hyperedge_ids": list(self.used_hyperedge_ids),
            "insufficient": self.insufficient,
        }


@dataclass(slots=True)
class DagExecutionResult:
    atomic_results: list[AtomicAnswerResult]
    final_answer: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "atomic_results": [item.to_dict() for item in self.atomic_results],
            "final_answer": dict(self.final_answer),
        }
