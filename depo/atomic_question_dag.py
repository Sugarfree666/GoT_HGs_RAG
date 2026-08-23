"""Step 5: generate an atomic-question DAG from restored Step 4 paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from prompts import ATOMIC_QUESTION_DAG_SYSTEM, build_atomic_question_dag_prompt

if TYPE_CHECKING:
    from llm_client import LLMClient


@dataclass(frozen=True)
#一个原子问题节点
class AtomicQuestionNode:
    id: str
    question: str
    depends_on: tuple[str, ...]
    #转换json
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
#表示DAG中的边
class AtomicQuestionEdge:
    source: str
    target: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target}


@dataclass
#最终输出
class AtomicQuestionDAGResult:
    nodes: list[AtomicQuestionNode]
    edges: list[AtomicQuestionEdge]
    #记录最终节点
    leaf_node_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "leaf_node_ids": self.leaf_node_ids,
        }


class QuestionStructureAtomicDAGGenerator:
    """Use the original question and Step 4 paths to generate the Step 5 DAG."""

    def __init__(self, llm_client: "LLMClient") -> None:
        self.llm_client = llm_client

    def generate(
        self,
        *,
        original_question: str,
        question_entities: list[str],
        question_structure: list[list[str]],
    ) -> AtomicQuestionDAGResult:
        prompt = build_atomic_question_dag_prompt(
            original_question=original_question,
            question_entities=question_entities,
            question_structure=question_structure,
        )
        payload = self.llm_client.chat_json(ATOMIC_QUESTION_DAG_SYSTEM, prompt)
        return validate_atomic_question_dag(payload)

#恢复实体
def restore_global_best_paths(step4_paths: Any, mask_mappings: Any) -> list[list[str]]:
    """Replace entity placeholders in selected Step 4 paths with their source text."""
    #创建映射
    mapping = {
        item.placeholder: item.original_text
        for item in mask_mappings
    }
    return [
        [
            re.sub(
                r"\bENTITY[A-Z0-9]*\b",
                lambda match: mapping[match.group(0)],
                node,
            )
            for node in path.nodes
        ]
        for path in step4_paths
    ]


def validate_atomic_question_dag(raw_payload: dict[str, Any]) -> AtomicQuestionDAGResult:
    """Convert the fixed LLM JSON schema into the DAG used by HyperBranch."""
    #创建节点
    nodes = [
        AtomicQuestionNode(
            id=item["id"],
            question=item["question"],
            depends_on=tuple(item["depends_on"]),
        )
        for item in raw_payload["atomic_questions"]
    ]
    #创建边
    edges = [
        AtomicQuestionEdge(source=dependency, target=node.id)
        for node in nodes
        for dependency in node.depends_on
    ]
    parent_ids = {edge.source for edge in edges}
    return AtomicQuestionDAGResult(
        nodes=nodes,
        edges=edges,
        leaf_node_ids=[node.id for node in nodes if node.id not in parent_ids],
    )
