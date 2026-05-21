Compose the final answer from an original complex question and ordered atomic results.

Return strict JSON only:
{
  "answer": "...",
  "reasoning_summary": "...",
  "confidence": 0.0,
  "atomic_answer_trace": [
    {"node_id": "...", "question": "...", "answer": "...", "used_hyperedge_ids": ["..."]}
  ],
  "remaining_gaps": []
}

Rules:
- Base the final answer only on atomic answers and their top evidence.
- Preserve the atomic answer trace in dependency order.
- If important atomic answers are insufficient, list them in remaining_gaps and lower confidence.
- Keep reasoning_summary concise.
