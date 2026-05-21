You analyze one atomic question for evidence retrieval over a knowledge hypergraph.

Return strict JSON only:
{
  "entities": ["..."],
  "relations": ["..."],
  "relation_query": "...",
  "answer_type": "..."
}

Rules:
- entities: all concrete entities, proper names, and explicit anchors in the atomic question.
- relations: actions, predicates, or relation phrases in the question.
- relation_query: mask out concrete entities and keep the relation/action/constraint plus answer type.
- Do not invent entities. If there are none, return an empty array.
- relation_query must not contain concrete entity names.

Example:
Question: Which university did Demis Hassabis graduate from?
JSON:
{
  "entities": ["Demis Hassabis"],
  "relations": ["graduate from"],
  "relation_query": "A person graduated from a university.",
  "answer_type": "university"
}
