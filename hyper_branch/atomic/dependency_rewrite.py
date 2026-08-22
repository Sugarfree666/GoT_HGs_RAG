"""Replace explicit qN answer references in a DEPO atomic question."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DependencyReplacement:
    node_id: str
    span: str
    answer: str

    def to_dict(self) -> dict[str, str]:
        return {"node_id": self.node_id, "span": self.span, "answer": self.answer}


@dataclass(slots=True)
class DependencyQuestionRewrite:
    original_question: str
    retrieval_question: str
    replacements: list[DependencyReplacement] = field(default_factory=list)
    primary_anchor_entities: list[str] = field(default_factory=list)

    @property
    def whether_rewritten(self) -> bool:
        return bool(self.replacements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "resolved_question": self.retrieval_question,
            "replacements": [replacement.to_dict() for replacement in self.replacements],
        }


def resolve_dependency_question(
    question: str,
    dependency_answers: list[dict[str, Any]],
) -> DependencyQuestionRewrite:
    rewritten = question
    replacements: list[DependencyReplacement] = []
    anchors: list[str] = []
    for dependency in dependency_answers:
        node_id = dependency["node_id"]
        answer = str(dependency["answer"]).strip()
        if not answer:
            continue
        pattern = rf"\b{re.escape(node_id)}(?:\s+answer|['\u2019]s\s+answer)\b"
        matches = list(re.finditer(pattern, rewritten, flags=re.IGNORECASE))
        for match in reversed(matches):
            rewritten = f"{rewritten[:match.start()]}{answer}{rewritten[match.end():]}"
            replacements.insert(0, DependencyReplacement(node_id, match.group(0), answer))
        if matches and answer not in anchors:
            anchors.append(answer)
    return DependencyQuestionRewrite(question, rewritten, replacements, anchors)
