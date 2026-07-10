from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


BranchName = Literal["anchor", "relation", "semantic"]
PathLabel = Literal["ANSWER", "EXPAND", "DROP"]


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
    relations: list[str] = field(default_factory=list)
    relation_query: str = ""
    answer_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": list(self.entities),
            "relations": list(self.relations),
            "relation_query": self.relation_query,
            "answer_type": self.answer_type,
        }


@dataclass(slots=True)
class BranchHit:
    hyperedge_id: str
    branch: BranchName
    raw_score: float
    hyperedge_text: str
    entity_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hyperedge_id": self.hyperedge_id,
            "branch": self.branch,
            "raw_score": self.raw_score,
            "hyperedge_text": self.hyperedge_text,
            "entity_ids": list(self.entity_ids),
            "chunk_ids": list(self.chunk_ids),
            "metadata": dict(self.metadata),
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
    chunk_ids: list[str] = field(default_factory=list)
    evidence_texts: list[str] = field(default_factory=list)
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
            "chunk_ids": list(self.chunk_ids),
            "evidence_texts": list(self.evidence_texts),
            "score_breakdown": dict(self.score_breakdown),
        }


@dataclass(slots=True)
class HypergraphPathStep:
    from_entity_id: str
    hyperedge_id: str
    hyperedge_text: str
    to_entity_id: str
    semantic_score: float
    semantic_rank: int
    entity_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    chunk_texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_entity_id": self.from_entity_id,
            "hyperedge_id": self.hyperedge_id,
            "hyperedge_text": self.hyperedge_text,
            "to_entity_id": self.to_entity_id,
            "semantic_score": self.semantic_score,
            "semantic_rank": self.semantic_rank,
            "entity_ids": list(self.entity_ids),
            "chunk_ids": list(self.chunk_ids),
            "chunk_texts": list(self.chunk_texts),
        }


@dataclass(slots=True)
class HypergraphReasoningPath:
    path_id: str
    anchor_entity_id: str
    entity_ids: list[str]
    hyperedge_ids: list[str]
    steps: list[HypergraphPathStep]
    hop_count: int
    label: PathLabel | None = None
    label_reason: str = ""
    answer_entity_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "anchor_entity_id": self.anchor_entity_id,
            "entity_ids": list(self.entity_ids),
            "hyperedge_ids": list(self.hyperedge_ids),
            "steps": [step.to_dict() for step in self.steps],
            "hop_count": self.hop_count,
            "label": self.label,
            "label_reason": self.label_reason,
            "answer_entity_ids": list(self.answer_entity_ids),
        }

    @property
    def tail_entity_id(self) -> str:
        return self.entity_ids[-1] if self.entity_ids else self.anchor_entity_id


@dataclass(slots=True)
class AtomicWalkResult:
    selected_paths: list[HypergraphReasoningPath] = field(default_factory=list)
    evidence_mode: str = "insufficient"
    answer_paths_found: bool = False
    insufficient: bool = True
    hop_artifacts: list[dict[str, Any]] = field(default_factory=list)
    anchor_entities: list[dict[str, Any]] = field(default_factory=list)
    resolved_anchor_entity_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_paths": [path.to_dict() for path in self.selected_paths],
            "selected_path_ids": [path.path_id for path in self.selected_paths],
            "evidence_mode": self.evidence_mode,
            "answer_paths_found": self.answer_paths_found,
            "insufficient": self.insufficient,
            "hop_artifacts": list(self.hop_artifacts),
            "anchor_entities": list(self.anchor_entities),
            "resolved_anchor_entity_ids": list(self.resolved_anchor_entity_ids),
        }


@dataclass(slots=True)
class AtomicAnswerResult:
    node_id: str
    question: str
    analysis: AtomicQuestionAnalysis
    evidence: list[FusedHyperedgeCandidate]
    answer: str
    confidence: float
    reasoning_summary: str
    used_dependencies: list[str] = field(default_factory=list)
    used_hyperedge_ids: list[str] = field(default_factory=list)
    paths: list[HypergraphReasoningPath] = field(default_factory=list)
    used_path_ids: list[str] = field(default_factory=list)
    walk_artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "question": self.question,
            "analysis": self.analysis.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "used_dependencies": list(self.used_dependencies),
            "used_hyperedge_ids": list(self.used_hyperedge_ids),
            "paths": [path.to_dict() for path in self.paths],
            "used_path_ids": list(self.used_path_ids),
            "walk_artifacts": dict(self.walk_artifacts),
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
