from __future__ import annotations

import json


EXPLICIT_ENTITY_EXTRACTION_SYSTEM = """
You are DEPO Step 2: topic entity extraction and parser-friendly question normalization.

Extract only explicit topic entities in the original question, and produce one lightly normalized question for parser input.

A topic entity is a concrete named thing explicitly mentioned in the question and useful as an anchor for QA decomposition:
person, creative work/title, organization, institution, location, geopolitical place, event, award, treaty, war, product, game, etc.

Do not extract roles, common nouns, answer slots, relation words, wh-words, operators, inferred entities, bare numbers, dates, years, ordinals, quantities, or measurements.

A number or year is allowed only when it is part of a complete official name, such as "Sabotage (1936 Film)", "Wrong Turn 5: Bloodlines", or "War of 1812".
Short titles in a typed comparison list may begin with a number; keep the full branch when the surrounding question supplies the type, such as film, album, song, book, game, or series.

Creative works and other titles may contain internal punctuation such as colons, hyphens, apostrophes, parentheses, and subtitles. Treat the full official-looking title as one entity when the punctuation connects title parts; do not split the subtitle into a separate person/place/entity.
Some official titles begin with words that also look like question words, such as When, What, Who, Where, or Which. If that word is part of a capitalized official-looking title span, keep it inside the entity; do not trim the title to the following words.

Do not answer the question.
Do not decompose it into atomic questions.
Do not change the meaning or delete any constraint.
Keep every entity surface form exactly as it appears in the original question.
Only make minimal grammar/parser-friendly edits, such as repairing wh-question word order, adding a missing auxiliary, or smoothing parser-unfriendly phrasing.
The normalized question must still be one single-sentence question.
If the original question is already grammatical and natural, keep it unchanged.

Return JSON only.
""".strip()


def build_explicit_entity_extraction_prompt(
    question: str,
    entity_candidates: list[dict[str, object]] | None = None,
) -> str:
    if entity_candidates:
        schema = {
            "explicit_entities": [
                {
                    "surface": "exact entity string from the original question",
                    "type": "Person | Location | Organization | Work | Event | Other",
                }
            ],
            "normalized_question": "one grammatical parser-friendly question preserving the original meaning",
            "normalization_changed": True,
            "normalization_note": "brief description of the minimal edit, or empty string if unchanged",
            "warnings": [],
        }

        return f"""
Original question:
{question}

Candidate spans (Deterministic entity candidates):
{json.dumps(entity_candidates, ensure_ascii=False, indent=2)}

Task:
Verify which candidate spans are topic entities, and lightly normalize the original question for dependency parsing.
Legacy verified_entities responses are accepted by the parser for compatibility, but you must return the explicit_entities schema below.

Rules:
1. Judge only the supplied candidates.
2. Return explicit_entities as selected entity surfaces copied exactly from the original question.
3. Do not invent, rewrite, merge, split, or offset-correct entity surfaces.
4. Select only concrete named topic entities.
5. Exclude roles, common nouns, answer slots, relation phrases, wh-phrases, operators, inferred entities, and bare dates/numbers/years.
6. Bare years such as "1956" are not entities.
7. Years or numbers inside complete official names may be true, e.g. "Sabotage (1936 Film)" or "War of 1812".
8. If candidate spans overlap, prefer the complete official-looking named mention over its substrings.
9. Internal punctuation in titles, especially colon/subtitle forms, is not a split boundary. A complete title like "Wrong Turn 5: Bloodlines" should be true as one Work/Film/Album/Book/etc.; subtitle fragments alone should be false unless independently named in the question.
10. In typed comparison or choice lists, such as "Which film ..., A or B?", verify each branch that looks like a title as its own entity. A branch may start with a number when the number is part of the title; do not reduce it to the alphabetic substring.
11. A candidate title may start with a capitalized question-like word such as When, What, Who, Where, or Which. If the whole candidate is an official-looking work/title, keep that first word; do not shorten it only because it resembles a wh-word.
12. Do not answer the question or decompose it.
13. Do not remove any restriction or change the semantics.
14. normalized_question must be a single question sentence.
15. Only apply light parser-friendly grammar repair. For example, "Which country the composer of film Thunder On The Hill is from?" becomes "Which country is the composer of film Thunder On The Hill from?"
16. If the original question is already grammatical and natural, return it unchanged and set normalization_changed=false.

Example:
Question: The player who defeated Johnny Majors for the Heisman Trophy in 1956 was born in what year?
True topic entities: "Johnny Majors", "Heisman Trophy"
False candidates: "player", "1956", "what year", "born", "defeated"

Return JSON only.
Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()

    schema = {
        "explicit_entities": [
            {
                "surface": "exact entity string from the original question",
                "type": "Person | Location | Organization | Work | Event | Other",
            }
        ],
        "normalized_question": "one grammatical parser-friendly question preserving the original meaning",
        "normalization_changed": True,
        "normalization_note": "brief description of the minimal edit, or empty string if unchanged",
        "warnings": [],
    }

    return f"""
Original question:
{question}

Task:
Extract all topic entities explicitly mentioned in the question, and lightly normalize the original question for dependency parsing.

Rules:
1. A returned entity surface must be an exact contiguous substring of the original question.
2. Return the complete named mention, not a truncated substring.
3. Do not include surrounding roles, type words, prepositions, clauses, or possessive "'s".
4. Split independent coordinated entities, e.g. "Ryan Tubridy or Mauro Massironi".
5. Do not split internal words inside one official name, e.g. "Battle of Qurah and Umm al Maradim".
6. Exclude roles, answer slots, relation words, wh-phrases, operators, inferred entities, and bare dates/numbers/years.
7. Bare years such as "1956" are not entities.
8. Years or numbers inside complete official names may be included, e.g. "Sabotage (1936 Film)" or "War of 1812".
9. Internal punctuation in official-looking titles, especially colon/subtitle forms, is part of the same entity; do not split the subtitle into a separate entity.
10. In typed comparison or choice lists, such as "Which film ..., A or B?", extract each title-like branch as its own entity. A branch may start with a number when the number is part of the title; do not drop the numeric token or return only the alphabetic substring.
11. Some official titles begin with capitalized question-like words, e.g. When/What/Who/Where/Which as the first word of a song, book, film, episode, or other work title. If the complete contiguous title starts with such a word, return the complete title and do not trim off the first word.
12. Do not answer the question or decompose it.
13. Do not remove any restriction or change the semantics.
14. normalized_question must be a single question sentence.
15. Only apply light parser-friendly grammar repair. For example, "Which country the composer of film Thunder On The Hill is from?" becomes "Which country is the composer of film Thunder On The Hill from?"
16. If the original question is already grammatical and natural, return it unchanged and set normalization_changed=false.

Return JSON only.
Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


MASK_SPAN_EXTRACTION_SYSTEM = EXPLICIT_ENTITY_EXTRACTION_SYSTEM


def build_mask_span_extraction_prompt(question: str) -> str:
    return build_explicit_entity_extraction_prompt(question)


ATOMIC_QUESTION_DAG_SYSTEM = """
You are an expert question decomposition module.

Your task is to convert a complex question into an Atomic Question DAG.

You are given:

* the original question
* one or more parser-grounded token paths

Use the original question to understand the full meaning.
Use the token paths as optional structural evidence. They may help identify reasoning order, but the original question is authoritative.
The paths are not natural-language sentences; they are compact evidence chains.

Atomic question definition:
An atomic question asks for one missing answer using one semantic operation. It should be directly answerable once its dependencies are resolved. It must not contain an unresolved nested relation that should be asked first.

Core principles:

1. Generate the complete set of atomic questions needed to answer the original question.
2. Preserve all entities, constraints, comparison conditions, and answer intent from the original question.
3. Use natural questions, not symbolic triples.
4. If a question depends on a previous answer, refer to it as q1's answer, q2's answer, etc.
5. Every dependency mentioned in question text must also appear in depends_on.
6. Evidence lookup questions should use support from one contiguous span of a supplied path.
7. Final comparison, selection, equality, ranking, or aggregation questions may use support: null.
8. Do not answer the questions.
9. Do not invent entities, relations, dates, or constraints not present in the original question or paths.
10. Do not leave unresolved placeholders such as ENTITYA or ENTITYB if the path already contains the restored entity name.
11. Return valid JSON only.

Output format:
{
"nodes": [
{
"id": "q1",
"question": "atomic question?",
"depends_on": [],
"support": {
"path_id": "P1",
"start_index": 0,
"end_index": 1
}
}
]
}

Support format:

* path_id: the id of the supporting path
* start_index: the first token index in the path span
* end_index: the last token index in the path span
* use support: null only for final reasoning nodes that do not correspond to one contiguous path span

Example 1 input:
{
"original_question": "What nationality is the performer of song When The Stars Go Blue?",
"paths": [
{
"path_id": "P1",
"nodes": [
{"index": 0, "text": "When The Stars Go Blue"},
{"index": 1, "text": "performer"},
{"index": 2, "text": "nationality"}
]
}
]
}

Example 1 output:
{
"nodes": [
{
"id": "q1",
"question": "Who is the performer of When The Stars Go Blue?",
"depends_on": [],
"support": {"path_id": "P1", "start_index": 0, "end_index": 1}
},
{
"id": "q2",
"question": "What is the nationality of q1's answer?",
"depends_on": ["q1"],
"support": {"path_id": "P1", "start_index": 1, "end_index": 2}
}
]
}

Example 2 input:
{
"original_question": "Which country is the composer of film Thunder On The Hill from?",
"paths": [
{
"path_id": "P1",
"nodes": [
{"index": 0, "text": "Thunder On The Hill"},
{"index": 1, "text": "composer"},
{"index": 2, "text": "country"}
]
}
]
}

Example 2 output:
{
"nodes": [
{
"id": "q1",
"question": "Who is the composer of Thunder On The Hill?",
"depends_on": [],
"support": {"path_id": "P1", "start_index": 0, "end_index": 1}
},
{
"id": "q2",
"question": "Which country is q1's answer from?",
"depends_on": ["q1"],
"support": {"path_id": "P1", "start_index": 1, "end_index": 2}
}
]
}

Example 3 input:
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

Example 3 output:
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

Now generate the Atomic Question DAG for the given input JSON.
Return only the JSON object.

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
