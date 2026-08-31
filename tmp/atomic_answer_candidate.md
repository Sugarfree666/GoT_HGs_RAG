You answer one atomic question using the supplied JSON input.
The input contains:
- `original_question`: global context for disambiguation.
- `atomic_question`: the question you must answer.
- `dependency_context`: answers to prerequisite questions.
- `evidence_blocks`: retrieved evidence blocks ordered by relevance, containing hyperedges and their source text.
Instructions:
1. Answer `atomic_question`. Use `original_question` only to preserve the intended answer target, relation, and constraints when the atomic question is ambiguous or awkward after dependency substitution.
2. Prioritize the supplied evidence and usable dependency answers.
3. Use facts that match the correct entity, relation, direction, and constraints. Ignore irrelevant information.
4. When `first_hop_hyperedge_text` and `hyperedge_text` appear together, treat them as a possible evidence chain in that order.
5. For a comparison or choice, resolve the relation from the relevant values and return the selected option or entity, not an intermediate value.
6. If the supplied context is insufficient, use reliable knowledge to give the best direct answer rather than describing missing evidence.
7. Return only the shortest answer that fully answers the atomic question. Preserve exact names, titles, dates, and numbers from the evidence, but do not copy surrounding evidence or add explanations.
Return strict JSON only:
{
  "answer": "..."
}
Do not include reasoning, explanations, citations, evidence IDs, or additional fields.
