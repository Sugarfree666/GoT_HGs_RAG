You are selecting the next 1-2 frontier entities to expand in a multi-branch hypergraph RAG system.

Return JSON only:
{
  "selected_entity_ids": ["..."],
  "reason": "..."
}

Requirements:
- Select at most 2 entity ids from `candidate_entities`.
- Prefer fresh entities that are most likely to close the missing constraints, relation gaps, or current focus.
- Avoid generic entities unless they clearly help answer the question.
- Use the candidate descriptions and their supporting frontier hyperedges to judge expansion value.
- Do not invent entity ids that are not in `candidate_entities`.
