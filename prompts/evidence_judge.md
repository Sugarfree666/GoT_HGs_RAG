You are deciding whether the current compressed evidence view is sufficient to answer the question.

Return JSON only:
{
  "enough": false,
  "confidence": 0.0,
  "reason": "...",
  "missing_requirements": ["..."],
  "next_focus": ["..."]
}

Requirements:
- Return `enough=true` only if the current evidence view is already sufficient to answer the question directly.
- Use the frontier hyperedges, their branch agreement pattern, the core chunk evidence, and the coverage summary.
- If not enough, clearly state which requirements are still missing and what the next iteration should focus on.
- Keep `confidence` between 0 and 1.
