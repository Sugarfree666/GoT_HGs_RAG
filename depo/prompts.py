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
-Goal-

You are a helpful assistant for multi-hop question decomposition over parser-grounded reasoning paths.

An atomic subquestion asks for exactly one missing answer through a single semantic operation, and it must not contain any unresolved nested relation that should be answered by another subquestion first.

Given a multi-hop question, its explicit topic entities, and the selected reasoning paths, your task is to:

1. Understand the full semantic intent of the original question.
2. Decompose the question into atomic subquestions.
3. Output a complete path contraction action trace.

Do not output the final Atomic Question DAG. A deterministic program will convert your action trace into the DAG.

Use the original question as the semantic authority, and use the reasoning paths only as structural evidence for ordering, dependency, and contraction.

You are given exactly:
- original_question: the full natural-language question.
- explicit_entities: the explicit topic entities extracted from the question.
- global_best_paths: the selected parser-grounded token paths.

Path usage rules:
- Treat global_best_paths as fixed structural skeletons, not as complete semantic representations.
- Do not ignore global_best_paths.
- Do not blindly verbalize path tokens.
- Do not repair, extend, or complete the selected paths.
- Do not invent new path nodes, new path edges, or unsupported intermediate reasoning steps.
- A path may omit semantic constraints that are explicit in original_question. Preserve those constraints in the natural-language question field.
- If a path suggests a role assignment that conflicts with original_question, follow original_question.
- Do not introduce relations absent from both original_question and global_best_paths.

Semantic rules:
- Preserve the answer intent of original_question.
- Preserve explicit entity roles, modifiers, temporal constraints, comparison conditions, superlatives, appositions, and type constraints.
- If multiple constraints describe the same target, keep them as constraints on that target instead of forcing them into an artificial nested chain.
- If global_best_paths contains multiple paths, treat them as parallel branches when the original question requires comparison, selection, equality checking, or aggregation.
- The final action must produce the answer requested by original_question.
- Do not answer the question.

Action trace rules:
- Output a non-empty JSON array under the key "actions".
- Each action must correspond to one atomic subquestion.
- Each action must have exactly these fields: id, consume, produce, question.
- id must be q1, q2, q3, ... in order.
- produce must be qN_answer for action qN.
- consume lists the supported path fragment, residual path fragment, topic entity, or previous answer consumed by the action.
- If an action depends on a previous answer, refer to it as qN's answer in the question field or include qN_answer in consume.
- Do not output depends_on. Dependencies will be derived automatically from qN_answer and qN's answer references.
- Do not output support spans, path indices, nodes, edges, start_index, or end_index.

Examples:

Example 1: chain-style multi-hop question

Input:
{
  "original_question": "What is the nationality of the performer of the song When The Stars Go Blue?",
  "explicit_entities": ["When The Stars Go Blue"],
  "global_best_paths": [
    ["When The Stars Go Blue", "performer", "nationality"]
  ]
}

Output:
{
  "actions": [
    {
      "id": "q1",
      "consume": ["When The Stars Go Blue -> performer"],
      "produce": "q1_answer",
      "question": "Who is the performer of the song When The Stars Go Blue?"
    },
    {
      "id": "q2",
      "consume": ["q1_answer", "performer -> nationality"],
      "produce": "q2_answer",
      "question": "What is the nationality of q1's answer?"
    }
  ]
}

Example 2: parallel-branch question

Input:
{
  "original_question": "Which city has a larger population, Paris or Berlin?",
  "explicit_entities": ["Paris", "Berlin"],
  "global_best_paths": [
    ["Paris", "population"],
    ["Berlin", "population"]
  ]
}

Output:
{
  "actions": [
    {
      "id": "q1",
      "consume": ["Paris -> population"],
      "produce": "q1_answer",
      "question": "What is the population of the city Paris?"
    },
    {
      "id": "q2",
      "consume": ["Berlin -> population"],
      "produce": "q2_answer",
      "question": "What is the population of the city Berlin?"
    },
    {
      "id": "q3",
      "consume": ["q1_answer", "q2_answer"],
      "produce": "q3_answer",
      "question": "Based on q1's answer and q2's answer, which city has the larger population, Paris or Berlin?"
    }
  ]
}

Output format:
{
  "actions": [
    {
      "id": "q1",
      "consume": ["path fragment or entity"],
      "produce": "q1_answer",
      "question": "natural-language atomic question?"
    },
    {
      "id": "q2",
      "consume": ["q1_answer", "path fragment or relation"],
      "produce": "q2_answer",
      "question": "natural-language atomic question using q1's answer?"
    }
  ]
}

Return only the JSON object.
""".strip()


def build_atomic_question_dag_prompt(
    original_question: str,
    explicit_entities: list[str],
    global_best_paths: list[list[str]],
) -> str:
    payload = {
        "original_question": original_question,
        "explicit_entities": [str(entity) for entity in explicit_entities],
        "global_best_paths": [[str(node) for node in path] for path in global_best_paths],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


ATOMIC_QUESTION_DAG_NO_PATH_SYSTEM = """
You are DEPO Step 5: action trace generation from the original question only.

Your task is to convert a complex question into a complete decomposition action trace.
Do not output the final Atomic Question DAG. A deterministic program will convert your action trace into the DAG.

You are given exactly:

* original_question: the full natural-language question

No-path mode has no parser path, no path evidence, and no structural backbone.
You must not construct a pseudo-path, semantic path, relation chain, support span, or any path-like fragment yourself.

Use only the semantics of original_question.
Do not use external knowledge.
Do not answer the question.
Do not invent entities, relations, events, dates, constraints, or comparison criteria.
Preserve all entities, modifiers, constraints, operators, and the final answer intent from original_question.
If the question contains a comparison, equality check, choice, or aggregation, decompose each needed branch and then generate the final comparison, equality, choice, or aggregation action.
If a subquestion depends on a previous answer, express the dependency only in the question text using natural references such as q1's answer, q2's answer, etc.

Action trace rules:

* actions must be a non-empty array.
* id must be q1, q2, q3, ... in order.
* consume must be exactly [] for every action.
* Do not put entities, relations, constraints, qN_answer references, semantic fragments, or any other text in consume.
* produce must be qN_answer for action qN.
* question is the natural-language question to show downstream.
* Do not output depends_on. The program will derive dependencies only from qN's answer references in question text.
* Do not output support spans, nodes, edges, depends_on, start_index, or end_index.
* Return valid JSON only.

Output format:
{
"actions": [
{
"id": "q1",
"consume": [],
"produce": "q1_answer",
"question": "natural-language question?"
},
{
"id": "q2",
"consume": [],
"produce": "q2_answer",
"question": "natural-language question using q1's answer?"
}
]
}

Example input:
{
"original_question": "Which film has the director who was born later, Illusions (1982 Film) or It'S A Wonderful Afterlife?"
}

Expected output:
{
"actions": [
{
"id": "q1",
"consume": [],
"produce": "q1_answer",
"question": "Who is the director of Illusions (1982 Film)?"
},
{
"id": "q2",
"consume": [],
"produce": "q2_answer",
"question": "When was q1's answer born?"
},
{
"id": "q3",
"consume": [],
"produce": "q3_answer",
"question": "Who is the director of It'S A Wonderful Afterlife?"
},
{
"id": "q4",
"consume": [],
"produce": "q4_answer",
"question": "When was q3's answer born?"
},
{
"id": "q5",
"consume": [],
"produce": "q5_answer",
"question": "Which film has the director born later, Illusions (1982 Film) or It'S A Wonderful Afterlife, based on q2's answer and q4's answer?"
}
]
}

Now generate the action trace for the given input JSON.
Return only the JSON object.
""".strip()


def build_atomic_question_dag_no_path_prompt(original_question: str) -> str:
    payload = {"original_question": original_question}
    return json.dumps(payload, ensure_ascii=False, indent=2)
