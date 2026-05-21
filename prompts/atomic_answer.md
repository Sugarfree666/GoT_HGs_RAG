Answer one atomic question using only the provided evidence and dependency answers.

Return strict JSON only:
{
  "answer": "...",
  "confidence": 0.0,
  "reasoning_summary": "...",
  "used_hyperedge_ids": ["..."],
  "insufficient": false
}

Rules:
- Use only the given dependency answers and top evidence.
- If the evidence is insufficient, set answer to "INSUFFICIENT_EVIDENCE", confidence to 0.0, and insufficient to true.
- Keep reasoning_summary concise. Do not describe a multi-step reasoning trace.
- used_hyperedge_ids must be selected from the provided evidence.
