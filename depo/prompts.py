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
You are DEPO Step 5: semantic-reasoning-path-guided Atomic Question DAG generation.

Your task is to convert parser-grounded token paths into semantic reasoning paths, then generate atomic questions from those semantic paths.
Do not answer the question. Do not use external knowledge. Return valid JSON only.

You are given exactly:
- original_question: the full natural-language question. This is the semantic authority.
- explicit_entities: named entities explicitly mentioned in the original question.
- global_best_paths: token-level structural evidence selected by Step 4, with entity placeholders already restored.

You must not require or assume any additional input fields. Do not ask for masked_question, normalized_question, SDP edges, anchor paths, path scores, candidate sets, debug structures, or hidden context.

Required output schema:
{
  "semantic_reasoning_paths": [
    {
      "branch_id": "p1",
      "source_token_path": ["token copied from global_best_paths", "..."],
      "semantic_nodes": [
        {
          "id": "p1_n1",
          "label": "meaningful semantic object",
          "kind": "entity | intermediate_variable | value_slot | constraint | operator"
        }
      ],
      "semantic_edges": [
        {
          "id": "p1_e1",
          "source": "p1_n1",
          "target": "p1_n2",
          "relation": "one executable semantic relation",
          "support_tokens": ["tokens copied from source_token_path"]
        }
      ],
      "terminal_node_id": "p1_nK"
    }
  ],
  "atomic_questions": [
    {
      "id": "q1",
      "question": "natural-language atomic question?",
      "depends_on": [],
      "operation": "lookup | compare | select | verify | intersect | aggregate",
      "semantic_edge_ids": ["p1_e1"]
    }
  ]
}

Core requirements:
1. Use original_question as the semantic authority. Preserve entities, constraints, modifiers, operators, comparison conditions, and final answer intent.
2. Use global_best_paths as mandatory parser-grounded structural evidence. They guide the decomposition, but they are noisy token paths, not the final semantic paths.
3. First transduce each token-level path into a semantic reasoning path. Then generate atomic questions from semantic edges and operator nodes.
4. Do not copy token paths directly as semantic paths. Remove function-word noise and recover relation direction from original_question.
5. Do not output final DAG edges. The deterministic program derives DAG edges only from atomic_questions.depends_on.

Semantic reasoning path rules:
1. branch_id values must be p1, p2, p3, ... in the same order as global_best_paths.
2. source_token_path must copy the corresponding global_best_paths entry exactly.
3. Semantic nodes must be meaningful reasoning objects: explicit entities, intermediate variables, value slots, constraints, or operators.
4. Do not make these function words semantic nodes unless they are truly operator metadata: who, what, which, where, when, why, is, was, were, be, do, did, does, have, the, a, an, of, in, by, from, to, with, for, at, punctuation.
5. Convert predicate and event tokens into semantic relations or value slots:
   - born + when -> birth_date
   - born + where -> birthplace
   - died + when -> death_date
   - died + where -> death_place
   - died + why -> cause_of_death
   - located + country -> country
   - graduated + university -> university
   - performer + song -> performer of song
   - director + film -> director of film
6. Preserve common noun anchors when they define intermediate variables or answer types: performer, director, author, composer, spouse, child, company, city, country, region, county, team, league, university, school, award, event, body of water.
7. Preserve operators as semantic operator nodes or final compare/select/verify/intersect/aggregate questions: same, different, both, either, common, later, earlier, older, younger, before, after, largest, smallest, highest, lowest, most, fewest, first, last.
8. Every semantic edge must express one executable one-hop relation and cite support_tokens copied from its source_token_path.
9. Do not include unresolved ENTITY placeholders anywhere.

Atomic question rules:
1. ids must be q1, q2, q3, ... in order.
2. Each lookup question asks for exactly one missing answer and corresponds to one semantic edge.
3. Do not ask multi-hop questions. Do not merge two unresolved lookups into one question.
4. If a question depends on previous answers, put those ids in depends_on and refer to them naturally in the question, such as q1's answer. Do not use {{q1}} or any braced reference.
5. operation must be one of: lookup, compare, select, verify, intersect, aggregate.
6. lookup questions must include semantic_edge_ids. Final compare/select/verify/intersect/aggregate questions may use semantic_edge_ids=[] when they only combine previous answers.
7. Generate lookup branches before comparison, selection, equality, intersection, or aggregation questions.
8. The final atomic question must preserve the final answer intent of original_question.

Example:
Input:
{
  "original_question": "Do director of film Ten9Eight: Shoot For The Moon and director of film Sabotage (1936 Film) share the same nationality?",
  "explicit_entities": ["Ten9Eight: Shoot For The Moon", "Sabotage (1936 Film)"],
  "global_best_paths": [
    ["Ten9Eight: Shoot For The Moon", "director", "nationality"],
    ["Sabotage (1936 Film)", "director", "nationality"]
  ]
}

Correct output:
{
  "semantic_reasoning_paths": [
    {
      "branch_id": "p1",
      "source_token_path": ["Ten9Eight: Shoot For The Moon", "director", "nationality"],
      "semantic_nodes": [
        {"id": "p1_n1", "label": "Ten9Eight: Shoot For The Moon", "kind": "entity"},
        {"id": "p1_n2", "label": "director of Ten9Eight: Shoot For The Moon", "kind": "intermediate_variable"},
        {"id": "p1_n3", "label": "nationality of director", "kind": "value_slot"}
      ],
      "semantic_edges": [
        {"id": "p1_e1", "source": "p1_n1", "target": "p1_n2", "relation": "director of film", "support_tokens": ["Ten9Eight: Shoot For The Moon", "director"]},
        {"id": "p1_e2", "source": "p1_n2", "target": "p1_n3", "relation": "nationality of director", "support_tokens": ["director", "nationality"]}
      ],
      "terminal_node_id": "p1_n3"
    },
    {
      "branch_id": "p2",
      "source_token_path": ["Sabotage (1936 Film)", "director", "nationality"],
      "semantic_nodes": [
        {"id": "p2_n1", "label": "Sabotage (1936 Film)", "kind": "entity"},
        {"id": "p2_n2", "label": "director of Sabotage (1936 Film)", "kind": "intermediate_variable"},
        {"id": "p2_n3", "label": "nationality of director", "kind": "value_slot"}
      ],
      "semantic_edges": [
        {"id": "p2_e1", "source": "p2_n1", "target": "p2_n2", "relation": "director of film", "support_tokens": ["Sabotage (1936 Film)", "director"]},
        {"id": "p2_e2", "source": "p2_n2", "target": "p2_n3", "relation": "nationality of director", "support_tokens": ["director", "nationality"]}
      ],
      "terminal_node_id": "p2_n3"
    }
  ],
  "atomic_questions": [
    {"id": "q1", "question": "Who directed Ten9Eight: Shoot For The Moon?", "depends_on": [], "operation": "lookup", "semantic_edge_ids": ["p1_e1"]},
    {"id": "q2", "question": "What is the nationality of q1's answer?", "depends_on": ["q1"], "operation": "lookup", "semantic_edge_ids": ["p1_e2"]},
    {"id": "q3", "question": "Who directed Sabotage (1936 Film)?", "depends_on": [], "operation": "lookup", "semantic_edge_ids": ["p2_e1"]},
    {"id": "q4", "question": "What is the nationality of q3's answer?", "depends_on": ["q3"], "operation": "lookup", "semantic_edge_ids": ["p2_e2"]},
    {"id": "q5", "question": "Do q2's answer and q4's answer indicate that the two directors share the same nationality?", "depends_on": ["q2", "q4"], "operation": "verify", "semantic_edge_ids": []}
  ]
}

Now generate the semantic reasoning paths and atomic questions for the given input JSON.
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

