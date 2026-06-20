from __future__ import annotations

import json


EXPLICIT_ENTITY_EXTRACTION_SYSTEM = """
You are DEPO Step 2: topic entity extraction.

Extract only explicit topic entities in the original question.

A topic entity is a concrete named thing explicitly mentioned in the question and useful as an anchor for QA decomposition:
person, creative work/title, organization, institution, location, geopolitical place, event, award, treaty, war, product, game, etc.

Do not extract roles, common nouns, answer slots, relation words, wh-words, operators, inferred entities, bare numbers, dates, years, ordinals, quantities, or measurements.

A number or year is allowed only when it is part of a complete official name, such as "Sabotage (1936 Film)", "Wrong Turn 5: Bloodlines", or "War of 1812".
Short titles in a typed comparison list may begin with a number; keep the full branch when the surrounding question supplies the type, such as film, album, song, book, game, or series.

Creative works and other titles may contain internal punctuation such as colons, hyphens, apostrophes, parentheses, and subtitles. Treat the full official-looking title as one entity when the punctuation connects title parts; do not split the subtitle into a separate person/place/entity.

Return JSON only.
""".strip()


def build_explicit_entity_extraction_prompt(
    question: str,
    entity_candidates: list[dict[str, object]] | None = None,
) -> str:
    if entity_candidates:
        schema = {
            "verified_entities": [
                {
                    "candidate_id": "candidate id from input",
                    "is_entity": True,
                    "confidence": 0.95,
                    "reason": "brief reason",
                }
            ],
            "warnings": [],
        }

        return f"""
Original question:
{question}

Candidate spans:
{json.dumps(entity_candidates, ensure_ascii=False, indent=2)}

Task:
Verify which candidate spans are topic entities.

Rules:
1. Judge only the supplied candidates.
2. Return every candidate exactly once.
3. Do not invent, rewrite, merge, split, or offset-correct candidates.
4. Set is_entity=true only for concrete named topic entities.
5. Set is_entity=false for roles, common nouns, answer slots, relation phrases, wh-phrases, operators, inferred entities, and bare dates/numbers/years.
6. Bare years such as "1956" are not entities.
7. Years or numbers inside complete official names may be true, e.g. "Sabotage (1936 Film)" or "War of 1812".
8. If candidate spans overlap, prefer the complete official-looking named mention over its substrings.
9. Internal punctuation in titles, especially colon/subtitle forms, is not a split boundary. A complete title like "Wrong Turn 5: Bloodlines" should be true as one Work/Film/Album/Book/etc.; subtitle fragments alone should be false unless independently named in the question.
10. In typed comparison or choice lists, such as "Which film ..., A or B?", verify each branch that looks like a title as its own entity. A branch may start with a number when the number is part of the title; do not reduce it to the alphabetic substring.

Example:
Question: The player who defeated Johnny Majors for the Heisman Trophy in 1956 was born in what year?
True topic entities: "Johnny Majors", "Heisman Trophy"
False candidates: "player", "1956", "what year", "born", "defeated"

Return JSON only.
Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()

    schema = {
        "entities": [
            {
                "text": "exact topic entity span copied from the question",
                "start_char": 0,
                "end_char": 15,
                "confidence": 0.95,
                "reason": "brief reason",
            }
        ],
        "warnings": [],
    }

    return f"""
Original question:
{question}

Task:
Extract all topic entities explicitly mentioned in the question.

Rules:
1. A returned entity must be an exact contiguous substring of the question.
2. start_char is inclusive; end_char is exclusive.
3. Return the complete named mention, not a truncated substring.
4. Do not include surrounding roles, type words, prepositions, clauses, or possessive "'s".
5. Split independent coordinated entities, e.g. "Ryan Tubridy or Mauro Massironi".
6. Do not split internal words inside one official name, e.g. "Battle of Qurah and Umm al Maradim".
7. Exclude roles, answer slots, relation words, wh-phrases, operators, inferred entities, and bare dates/numbers/years.
8. Bare years such as "1956" are not entities.
9. Years or numbers inside complete official names may be included, e.g. "Sabotage (1936 Film)" or "War of 1812".
10. Internal punctuation in official-looking titles, especially colon/subtitle forms, is part of the same entity; do not split the subtitle into a separate entity.
11. In typed comparison or choice lists, such as "Which film ..., A or B?", extract each title-like branch as its own entity. A branch may start with a number when the number is part of the title; do not drop the numeric token or return only the alphabetic substring.

Return JSON only.
Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


MASK_SPAN_EXTRACTION_SYSTEM = EXPLICIT_ENTITY_EXTRACTION_SYSTEM


def build_mask_span_extraction_prompt(question: str) -> str:
    return build_explicit_entity_extraction_prompt(question)


ATOMIC_QUESTION_DAG_SYSTEM = """
You are DEPO Step 5: Path-Aligned Evidence-Oriented Atomic Question DAG Generator.

Definition of an atomic evidence question:

1. It asks for exactly one retrievable fact, relation, or attribute.
2. It has exactly one unknown answer.
3. It performs only one semantic lookup.
4. It must not contain an unresolved nested relation.
5. If an argument must first be obtained from another question, use exactly a reference such as q1's answer, and make the current node depend on q1.
6. Fixed restrictions from the original question--such as dates, awards, locations, ranges, time conditions, or descriptive clauses--may remain in one atomic question. They constrain the lookup and do not create an additional reasoning hop.
7. A question is not atomic if it requests two facts, contains two unresolved relations, performs a final comparison, selects among candidates, or asks for the final answer to the original multi-hop question.

Task:

Generate an evidence-oriented atomic question DAG using:

- the original question for semantic interpretation and fixed restrictions;
- the supplied parser-grounded token paths for structural support.

The paths contain original entity names. Treat them as structural evidence, not as literal natural-language templates.

Rules:

1. Every DAG node must be one atomic evidence-seeking question.
2. Every node must be supported by exactly one contiguous span of exactly one supplied path.
3. Do not create any node without path support.
4. Do not introduce an unrelated main relation that is absent from the supporting path.
5. You may use the original question to interpret a path relation naturally and preserve fixed restrictions.
6. Do not answer any question.
7. Do not generate the final comparison, equality decision, ranking decision, candidate selection, aggregation question, or final-answer question.
8. When multiple paths are supplied, treat them as separate evidence branches.
9. Do not create dependencies between different paths.
10. Generate the evidence questions required by every supplied path.
11. A dependent question may reference at most one earlier answer.
12. Refer to a previous answer using exactly qN's answer.
13. depends_on must exactly match the qN's answer reference in the question.
14. Use the original entity names already present in the paths.
15. Do not invent new named entities, dates, predicates, or restrictions.
16. Path spans may overlap at an intermediate node. This is expected in a multi-hop chain.
17. Cover the reasoning content of every supplied path.
18. Return valid JSON only.

Output JSON shape:

{
  "nodes": [
    {
      "id": "q1",
      "question": "One atomic evidence question?",
      "depends_on": [],
      "support": {
        "path_id": "P1",
        "start_index": 0,
        "end_index": 2
      }
    }
  ]
}

Do not return edges, final operations, rationale, analysis, or chain-of-thought.

Example A input:

{
  "original_question": "The player who defeated Johnny Majors for the Heisman Trophy in 1956 was born in what year?",
  "paths": [
    {
      "path_id": "P1",
      "nodes": [
        {"index": 0, "text": "Johnny Majors"},
        {"index": 1, "text": "defeated"},
        {"index": 2, "text": "player"},
        {"index": 3, "text": "born"},
        {"index": 4, "text": "year"}
      ]
    }
  ]
}

Example A output:

{
  "nodes": [
    {
      "id": "q1",
      "question": "Who defeated Johnny Majors for the Heisman Trophy in 1956?",
      "depends_on": [],
      "support": {"path_id": "P1", "start_index": 0, "end_index": 2}
    },
    {
      "id": "q2",
      "question": "What year was q1's answer born?",
      "depends_on": ["q1"],
      "support": {"path_id": "P1", "start_index": 2, "end_index": 4}
    }
  ]
}

Example B input paths:

P1: Ten9Eight: Shoot For The Moon ---- director ---- share ---- nationality
P2: Sabotage (1936 Film) ---- director ---- share ---- nationality

Example B output:

{
  "nodes": [
    {
      "id": "q1",
      "question": "Who directed Ten9Eight: Shoot For The Moon?",
      "depends_on": [],
      "support": {"path_id": "P1", "start_index": 0, "end_index": 1}
    },
    {
      "id": "q2",
      "question": "What is the nationality of q1's answer?",
      "depends_on": ["q1"],
      "support": {"path_id": "P1", "start_index": 1, "end_index": 3}
    },
    {
      "id": "q3",
      "question": "Who directed Sabotage (1936 Film)?",
      "depends_on": [],
      "support": {"path_id": "P2", "start_index": 0, "end_index": 1}
    },
    {
      "id": "q4",
      "question": "What is the nationality of q3's answer?",
      "depends_on": ["q3"],
      "support": {"path_id": "P2", "start_index": 1, "end_index": 3}
    }
  ]
}

Do not generate a final node asking whether the two nationalities are the same.
""".strip()


def build_atomic_question_dag_prompt(
    original_question: str,
    paths: list[dict[str, object]],
) -> str:
    payload_paths: list[dict[str, object]] = []
    for path in paths:
        nodes = [str(node) for node in path.get("nodes", [])] if isinstance(path, dict) else []
        payload_paths.append(
            {
                "path_id": str(path.get("path_id", "")) if isinstance(path, dict) else "",
                "nodes": [
                    {
                        "index": index,
                        "text": text,
                    }
                    for index, text in enumerate(nodes)
                ],
            }
        )
    payload = {
        "original_question": original_question,
        "paths": payload_paths,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
