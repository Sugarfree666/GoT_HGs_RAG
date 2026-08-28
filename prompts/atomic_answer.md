You answer one atomic question using the supplied JSON input.

The input contains:
`original_question`: the original complex question from which the atomic-question DAG was decomposed. It provides the global semantic context and shows how the current atomic question contributes to answering the overall question.
atomic_question`: the current node in the atomic-question DAG. This is the question you must answer.
`dependency_context`: prerequisite nodes of the atomic-question DAG, including each prerequisite question and its answer. These provide the intermediate results needed to interpret or answer the current atomic question.
`evidence_blocks`: retrieved evidence blocks ordered by relevance, containing hyperedges and their source text. A hyperedge is a natural-language factual statement that may connect multiple entities in one relation.
Instructions:
1. Answer only `atomic_question`. 
2. Understand `atomic_question` as one step in the reasoning process of `original_question`. Use `original_question` when necessary to determine the intended meaning, answer target, relation direction, or constraints of the current atomic question. 
3. Use facts that match the correct entity, relation, direction, and constraints. Ignore irrelevant information.
4. When `first_hop_hyperedge_text` and `hyperedge_text` appear together, treat them as a possible evidence chain in that order.
5. Use the supplied evidence and dependency context first. Use your own knowledge to fill in the missing information when they are insufficient to answer the atomic question.
6. Return only the shortest answer that fully answers the atomic question. Preserve exact names, titles, dates, and numbers from the evidence, but do not copy surrounding evidence or add explanations.
Return strict JSON only:
{
  "answer": "..."
}
Do not include reasoning, explanations, citations, evidence IDs, or additional fields.
