"""原子执行器内部使用的类型化 DAG、分析、候选和答案记录。"""


from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AtomicQuestionNode:
    node_id: str
    question: str
    dependencies: list[str] = field(default_factory=list)

@dataclass(slots=True)
class FusedHyperedgeCandidate:
    hyperedge_id: str
    hyperedge_text: str
    chunk_ids: list[str] = field(default_factory=list)
    chunk_texts: list[str] = field(default_factory=list)
    first_hop_hyperedge_ids: list[str] = field(default_factory=list)

@dataclass(slots=True)
class AtomicAnswerResult:
    node_id: str
    question: str
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "question": self.question,
            "answer": self.answer,
        }


@dataclass(slots=True)
class DagExecutionResult:
    #所有原子问题答案
    atomic_results: list[AtomicAnswerResult]
    final_answer: dict[str, Any]
