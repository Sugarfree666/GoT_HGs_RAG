"""DEPO 的实体识别与原子问题 DAG 提示词。"""

from __future__ import annotations

import json


#精调提示词
EXPLICIT_ENTITY_EXTRACTION_SYSTEM = """
You identify explicit entities in a question for entity masking and HanLP parsing.

Return JSON only. The original question is the source of truth: do not answer it, add facts,
or infer entities that are not explicitly present.

Select concrete named or identifier-like anchors, such as people, works, organizations,
places, events, institutions, acronyms, official designations, and codes. Each `surface` must
be an exact contiguous substring of the original question. Preserve its spelling, case,
punctuation, and complete identifying name. Return each surface once, and never return
overlapping surfaces.

Exclude generic roles, relation phrases, question slots, common categories, demonyms, and
bare dates, years, or numeric values. For a possessive expression such as `X's father`, return
`X`, not the role phrase. Do not split punctuation that belongs to one official title.

`normalized_question` should normally equal the original question. Change it only when a
strictly equivalent structural rewrite is necessary to expose a nested relation for HanLP.
Preserve every entity surface, answer target, relation direction, constraint, and candidate set.
If equivalence is uncertain, keep the original question.

Use exactly this JSON schema:
{
  "explicit_entities": [{"surface": "exact source span"}],
  "normalized_question": "question for HanLP"
}
""".strip()


def build_explicit_entity_extraction_prompt(question: str) -> str:
    """构造 DEPO 实体识别的唯一输入。"""
    return json.dumps({"question": question}, ensure_ascii=False, indent=2)

#精调
ATOMIC_QUESTION_DAG_SYSTEM = """
You decompose one complex question into a minimal, retrieval-executable atomic-question DAG.

The `original_question` is the only source of meaning. `question_entities` preserves anchor
spellings, and `question_structure` is only a structural hint: neither may add facts or change
the original meaning. Do not answer the question or use outside knowledge.

Create the fewest atomic questions needed to recover the original answer. Each node asks for
one retrievable fact, entity, value, comparison, selection, verification, or aggregation.
Keep restrictions, relation direction, negation, temporal and numeric constraints, and
coordination needed for the original answer.

Use IDs `q1`, `q2`, ... in execution order. A node may depend only on earlier nodes. When a
node needs an earlier answer, write exactly `qN's answer` in its question and list the same
`qN` in `depends_on`. Do not declare unused dependencies or reference an undeclared answer.

The DAG must have exactly one final leaf, and it must be the last node. Every earlier node must
lead to that final node. The final node must ask for exactly the original answer target, not an
intermediate fact. Before output, substitute dependency answers conceptually and verify that
the final question remains answer-equivalent to the original.

Return JSON only:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "atomic question",
      "depends_on": [],
      "operation": "lookup"
    }
  ]
}

`operation` is one of `lookup`, `select`, `compare`, `verify`, or `aggregate`.
""".strip()


def build_atomic_question_dag_prompt(
    original_question: str,
    question_entities: list[str],
    question_structure: list[list[str]],
) -> str:
    """构造 Step 5 DAG 生成所需的原问题、实体和结构提示。"""
    entities: list[str] = []
    seen_entities: set[str] = set()
    for entity in question_entities:
        text = str(entity).strip()
        if text and text not in seen_entities:
            seen_entities.add(text)
            entities.append(text)

    structure = [
        " -- ".join(node.strip() for node in branch if node.strip())
        for branch in question_structure
    ]
    return json.dumps(
        {
            "original_question": original_question,
            "question_entities": entities,
            "question_structure": [branch for branch in structure if branch],
        },
        ensure_ascii=False,
        indent=2,
    )
