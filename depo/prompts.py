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
You are DEPO Step 5: path contraction action trace generation.

Your task is to convert a complex question into a complete path contraction action trace.
Do not output the final Atomic Question DAG. A deterministic program will convert your action trace into the DAG.

You are given exactly:

* original_question: the full natural-language question
* explicit_entities: original entity surface strings extracted from the question
* global_best_path: the single mandatory structural backbone selected by Step 4, with ENTITY placeholders already restored

How to use the inputs:

1. Use global_best_path as the mandatory structural backbone.
2. Use original_question for full semantics, constraints, answer intent, and grammatical realization.
3. Use global_best_path for relation order and dependency structure.
4. Generate a complete contraction action trace, not a final DAG.
5. Do not output support spans, path indices, nodes, edges, depends_on, start_index, or end_index.
6. Do not introduce relations absent from both the original question and global_best_path.
7. For "When" questions, ask about the event relation present in the original question/path, not unrelated attributes such as birth date unless the original question/path explicitly requires birth.
8. Do not leave ENTITY placeholders unresolved.
9. Do not answer the questions.
10. Return valid JSON only.

Action trace rules:

* actions must be a non-empty array.
* id must be q1, q2, q3, ... in order.
* consume lists the current residual path fragment consumed by that atomic question. It may contain previous produced values such as q1_answer.
* produce must be qN_answer for action qN.
* question is the natural-language atomic question to show downstream.
* If an action depends on a previous answer, refer to it as q1's answer, q2's answer, etc. in question text, or include q1_answer, q2_answer, etc. in consume.
* Do not output depends_on. The program will derive dependencies from qN_answer and qN's answer references in question/consume.

Output format:
{
"actions": [
{
"id": "q1",
"consume": ["path node 1", "path node 2"],
"produce": "q1_answer",
"question": "natural-language atomic question?"
},
{
"id": "q2",
"consume": ["path node 3", "relation", "q1_answer"],
"produce": "q2_answer",
"question": "natural-language atomic question using q1's answer?"
}
]
}

Example input:
{
"original_question": "When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?",
"explicit_entities": ["Messi", "Copa del Rey", "Barcelona"],
"global_best_path": ["Barcelona", "signed", "get", "person", "compared", "goals", "Messi"]
}

Expected output:
{
"actions": [
{
"id": "q1",
"consume": ["person", "compared", "goals", "Messi"],
"produce": "q1_answer",
"question": "Who is the person that Messi's goals in Copa del Rey were compared to?"
},
{
"id": "q2",
"consume": ["Barcelona", "signed", "get", "q1_answer"],
"produce": "q2_answer",
"question": "When did q1's answer get signed by Barcelona?"
}
]
}

Do not generate "When was q1's answer born?" for that example, because "born" is absent from both the original question and global_best_path.

Now generate the contraction action trace for the given input JSON.
Return only the JSON object.
""".strip()


def build_atomic_question_dag_prompt(
    original_question: str,
    explicit_entities: list[str],
    global_best_path: list[str],
) -> str:
    payload = {
        "original_question": original_question,
        "explicit_entities": [str(entity) for entity in explicit_entities],
        "global_best_path": [str(node) for node in global_best_path],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
