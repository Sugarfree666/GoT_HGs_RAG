from __future__ import annotations

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
    anchor_score: float = 0.0
    relation_score: float = 0.0
    semantic_score: float = 0.0
    fusion_score: float = 0.0
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
            "anchor_score": self.anchor_score,
            "relation_score": self.relation_score,
            "semantic_score": self.semantic_score,
            "fusion_score": self.fusion_score,
            "entity_ids": list(self.entity_ids),
            "entity_records": [dict(item) for item in self.entity_records],
            "chunk_ids": list(self.chunk_ids),
            "chunk_texts": list(self.chunk_texts),
            "evidence_texts": list(self.evidence_texts),
            "rank": self.rank,
            "score_breakdown": dict(self.score_breakdown),
        }


@dataclass(slots=True)
class EvidencePathCandidate:
    path_type: str
    path_texts: list[str]
    hyperedge_ids: list[str]
    context_ids: list[str]
    anchor_entity_id: str
    bridge_entity_id: str
    seed_hyperedge_id: str
    seed_hyperedge_rank: int
    seed_hyperedge_score: float
    path_score: float = 0.0
    path_rank: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    structural_key: tuple[str, ...] = field(default_factory=tuple)

    @property
    def rank(self) -> int | None:
        return self.path_rank

    @property
    def hyperedge_id(self) -> str:
        return self.seed_hyperedge_id

    @property
    def hyperedge_text(self) -> str:
        return self.path_texts[-1] if self.path_texts else ""

    @property
    def semantic_score(self) -> float:
        return self.path_score

    @property
    def fusion_score(self) -> float:
        return self.path_score

    @property
    def entity_ids(self) -> list[str]:
        values = [self.anchor_entity_id, self.bridge_entity_id]
        return [value for value in values if value]

    @property
    def chunk_ids(self) -> list[str]:
        return list(self.context_ids)

    @property
    def chunk_texts(self) -> list[str]:
        if self.path_type != "3he":
            return []
        return self.path_texts[1:2]

    @property
    def evidence_texts(self) -> list[str]:
        return list(self.path_texts)

    @property
    def score_breakdown(self) -> dict[str, Any]:
        return {
            "seed_hyperedge_id": self.seed_hyperedge_id,
            "seed_hyperedge_rank": self.seed_hyperedge_rank,
            "seed_hyperedge_score": self.seed_hyperedge_score,
            "path_score": self.path_score,
            "path_type": self.path_type,
            "anchor_entity_id": self.anchor_entity_id,
            "bridge_entity_id": self.bridge_entity_id,
        }

    def serialized_text(self) -> str:
        return "\n".join(str(text or "").strip() for text in self.path_texts if str(text or "").strip())

    def to_answer_payload(self) -> dict[str, Any]:
        return {"path": list(self.path_texts)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_type": self.path_type,
            "path_texts": list(self.path_texts),
            "path": list(self.path_texts),
            "hyperedge_ids": list(self.hyperedge_ids),
            "context_ids": list(self.context_ids),
            "anchor_entity_id": self.anchor_entity_id,
            "bridge_entity_id": self.bridge_entity_id,
            "seed_hyperedge_id": self.seed_hyperedge_id,
            "seed_hyperedge_rank": self.seed_hyperedge_rank,
            "seed_hyperedge_score": self.seed_hyperedge_score,
            "path_score": self.path_score,
            "path_rank": self.path_rank,
            "provenance": dict(self.provenance),
            "structural_key": list(self.structural_key),
        }


@dataclass(slots=True)
class AtomicAnswerResult:
    node_id: str
    question: str
    analysis: AtomicQuestionAnalysis
    evidence: list[Any]
    answer: str
    reasoning_summary: str
    used_dependencies: list[str] = field(default_factory=list)
    used_hyperedge_ids: list[str] = field(default_factory=list)
    insufficient: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "question": self.question,
            "analysis": self.analysis.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "answer": self.answer,
            "reasoning_summary": self.reasoning_summary,
            "used_dependencies": list(self.used_dependencies),
            "used_hyperedge_ids": list(self.used_hyperedge_ids),
            "insufficient": self.insufficient,
        }


@dataclass(slots=True)
class DagExecutionResult:
    original_question: str
    atomic_results: list[AtomicAnswerResult]
    final_answer: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "atomic_results": [item.to_dict() for item in self.atomic_results],
            "final_answer": dict(self.final_answer),
            "artifacts": dict(self.artifacts),
        }
