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
Some official titles begin with words that also look like question words, such as When, What, Who, Where, or Which. If that word is part of a capitalized official-looking title span, keep it inside the entity; do not trim the title to the following words.

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
11. A candidate title may start with a capitalized question-like word such as When, What, Who, Where, or Which. If the whole candidate is an official-looking work/title, keep that first word; do not shorten it only because it resembles a wh-word.

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
12. Some official titles begin with capitalized question-like words, e.g. When/What/Who/Where/Which as the first word of a song, book, film, episode, or other work title. If the complete contiguous title starts with such a word, return the complete title and do not trim off the first word.

Return JSON only.
Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


MASK_SPAN_EXTRACTION_SYSTEM = EXPLICIT_ENTITY_EXTRACTION_SYSTEM


def build_mask_span_extraction_prompt(question: str) -> str:
    return build_explicit_entity_extraction_prompt(question)


ATOMIC_QUESTION_DAG_SYSTEM = """
You are DEPO Step 5: Complete Atomic Question DAG Generator.

Definition of an atomic question:

1. It asks for exactly one retrievable fact, relation, attribute, comparison, selection, or aggregation.
2. It has exactly one unknown answer.
3. It performs one semantic operation.
4. It must not contain an unresolved nested relation.
5. If an argument must first be obtained from another question, refer to it naturally as q1's answer, q2's answer, etc., and list that id in depends_on.
6. Fixed restrictions from the original question--such as dates, awards, locations, ranges, time conditions, or descriptive clauses--may remain in one atomic question. They constrain the lookup and do not create an additional reasoning hop.

Task:

Generate the complete Atomic Question DAG needed to answer the original question using:

- the original question for semantic interpretation and final reasoning intent;
- the supplied parser-grounded token paths for structural support.

The paths contain original entity names. Treat them as structural evidence, not as literal natural-language templates.

Rules:

1. Every node must be a single atomic question.
2. Evidence lookup nodes should include support: one contiguous span of one supplied path.
3. Final comparison, equality, ranking, selection, and aggregation nodes are allowed.
4. A final reasoning node that does not directly correspond to a single Step4 path span may use "support": null.
5. depends_on may be empty, contain one previous node, or contain multiple previous nodes.
6. Cross-path dependencies are allowed when needed to answer the original question.
7. For comparison or selection questions, first generate the required factual evidence questions for each candidate, then generate the final comparison/selection node.
8. For words such as younger, older, earlier, later, first ask for comparable evidence such as birth date, date, age, or another appropriate attribute, then compare those evidence answers.
9. If a question text uses a previous answer, write it as qN's answer and include qN in depends_on.
10. Do not leave unresolved ENTITYA, ENTITYB, or similar placeholders. Use the original entity names already present in the paths.
11. Do not invent unrelated named entities, dates, predicates, or restrictions.
12. Do not answer any question.
13. Return valid JSON only.

Output JSON shape:

{
  "nodes": [
    {
      "id": "q1",
      "question": "single atomic question?",
      "depends_on": [],
      "support": {
        "path_id": "P1",
        "start_index": 0,
        "end_index": 1
      }
    },
    {
      "id": "q2",
      "question": "single atomic question using q1's answer if needed?",
      "depends_on": ["q1"],
      "support": {
        "path_id": "P1",
        "start_index": 1,
        "end_index": 2
      }
    },
    {
      "id": "q3",
      "question": "final comparison or selection question using q1's answer and q2's answer?",
      "depends_on": ["q1", "q2"],
      "support": null
    }
  ]
}

Do not return edges, rationale, analysis, answers, or chain-of-thought.

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

Example B input:

{
  "original_question": "Which film whose director is younger, Dangerously They Live or Salad By The Roots?",
  "paths": [
    {
      "path_id": "P1",
      "nodes": [
        {"index": 0, "text": "Dangerously They Live"},
        {"index": 1, "text": "director"},
        {"index": 2, "text": "younger"}
      ]
    },
    {
      "path_id": "P2",
      "nodes": [
        {"index": 0, "text": "Salad By The Roots"},
        {"index": 1, "text": "director"},
        {"index": 2, "text": "younger"}
      ]
    }
  ]
}

Example B output:

{
  "nodes": [
    {
      "id": "q1",
      "question": "Who directed Dangerously They Live?",
      "depends_on": [],
      "support": {"path_id": "P1", "start_index": 0, "end_index": 1}
    },
    {
      "id": "q2",
      "question": "When was q1's answer born?",
      "depends_on": ["q1"],
      "support": {"path_id": "P1", "start_index": 1, "end_index": 2}
    },
    {
      "id": "q3",
      "question": "Who directed Salad By The Roots?",
      "depends_on": [],
      "support": {"path_id": "P2", "start_index": 0, "end_index": 1}
    },
    {
      "id": "q4",
      "question": "When was q3's answer born?",
      "depends_on": ["q3"],
      "support": {"path_id": "P2", "start_index": 1, "end_index": 2}
    },
    {
      "id": "q5",
      "question": "Which film has the younger director, Dangerously They Live or Salad By The Roots, based on q2's answer and q4's answer?",
      "depends_on": ["q2", "q4"],
      "support": null
    }
  ]
}
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
