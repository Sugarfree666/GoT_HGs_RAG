You are generating the final answer for an iterative multi-branch hypergraph RAG pipeline.

Return JSON only:
{
  "answer": "...",
  "reasoning_summary": "...",
  "confidence": 0.0,
  "remaining_gaps": ["..."]
}

Requirements:
- Answer the user's question directly.
- `answer` must be the shortest grounded answer span, not an explanatory sentence.
- Prefer 1 to 5 words in `answer`; never exceed 8 words.
- Never start `answer` with phrases like `The answer is`, `By`, `It is`, or any full-sentence framing.
- Put explanation only in `reasoning_summary`.
- When the evidence supports a short phrase in `coverage_summary.answer_hypotheses`, `coverage_summary.target.text`, or the frontier evidence, copy that short phrase instead of paraphrasing.
- For this dataset, many `How do/does ... contribute ...` questions still expect the target concept or outcome itself. Example: return `COMMUNITY HEALTH`, not `They improve community health by ...`.
- Use the current compressed evidence view and the thought graph summary.
- If evidence is incomplete, say so in `remaining_gaps` instead of fabricating certainty.
- `reasoning_summary` should briefly explain which evidence pattern led to the answer.
- Keep `confidence` between 0 and 1.
