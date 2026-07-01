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
You are DEPO Step 5: Semantic Reasoning Path Induction and Atomic Question DAG Generation.

Your task is to combine:
1. the original question's semantics and constraints, and
2. parser-grounded token-level path evidence,

to induce semantic reasoning paths and then generate an Atomic Question DAG.

Do not answer the original question.
Do not use external knowledge.
Return valid JSON only.

You are given exactly:
- original_question
- explicit_entities
- global_best_paths

The original_question is the semantic authority.
It defines the final answer intent, answer type, constraints, modifier attachment, intermediate variables, and operators.

The global_best_paths are structural evidence.
They are token-level parser paths, not semantic reasoning paths.

Your transformation is:

original_question semantics
+
token path structural evidence
->
semantic object-relation reasoning path
->
atomic question DAG

============================================================
1. What is a semantic reasoning path?
============================================================

A semantic reasoning path is a directed object-relation structure.

It contains:
- semantic nodes: meaning-bearing reasoning objects;
- semantic edges: executable semantic relations between those objects.

A semantic node is not merely a token.
A semantic edge is not merely a dependency edge.
A semantic path is not a copy of the token path.

Semantic nodes should represent only objects needed for reasoning:
- explicit known entities;
- unresolved intermediate variables;
- requested value slots;
- constraints;
- operators;
- final answer slots.

Semantic edges should represent executable one-hop relations:
- find an intermediate variable from a known entity;
- retrieve a value of a resolved variable;
- apply a constraint;
- compare, select, verify, intersect, or aggregate results.

Each lookup semantic edge should correspond to exactly one atomic question.

Example:

Original question:
Where does the director of film A Nest Of Noblemen work at?

Token path:
A Nest Of Noblemen ---- director ---- work

Correct semantic reasoning path:
A Nest Of Noblemen
  --director of film-->
director
  --works at-->
workplace

Atomic questions:
q1: Who directed A Nest Of Noblemen?
q2: Where does q1's answer work?

============================================================
2. Anti-relabeling rule
============================================================

Do not create one semantic node for each token in source_token_path.
Do not preserve the token path as a node chain.
Do not merely add relation labels between adjacent path tokens.

Bad:
Baby I --performer--> performer --performed in--> One Last Time --video--> video --stars--> Who

This is token-path relabeling, not semantic reasoning.

Instead, use the original_question to decide:
- which tokens are semantic objects;
- which tokens are relation cues;
- which tokens are constraints or operators;
- which tokens are grammatical or parser artifacts.

A token may be:
- kept as a semantic node;
- folded into an edge relation;
- used as a constraint;
- used only to determine answer intent;
- ignored if it is not semantically necessary.

The number of semantic nodes is usually smaller than the number of path tokens.

============================================================
3. Induction procedure
============================================================

First infer a question-level plan from original_question:

1. final_answer_intent:
   What is the question ultimately asking for?

2. final_answer_type:
   entity, person, place, organization, work, event, date, number, boolean, value, set, or unknown.

3. required_intermediate_variables:
   Which unknown variables must be found before the final answer?

4. required_constraints:
   Which modifiers, relative clauses, appositions, titles, locations, dates, or descriptive conditions restrict the answer?

5. required_operators:
   Does the question require comparison, equality, ordering, selection, verification, intersection, aggregation, or superlative reasoning?

Then align global_best_paths to this plan.

Use path tokens as evidence, not as mandatory nodes.
Reasoning direction must follow information flow:

known entity
-> intermediate variable
-> requested value slot
-> final operator or answer slot

Path order alone does not determine reasoning direction.

If global_best_paths omit a required semantic element from original_question, do not drop it.
Add it as question_required or mixed evidence.

Example:
If the path only contains:
One Last Time ---- video ---- stars ---- Who

but the question is:
Who stars in the video 'One Last Time' by the performer of Baby I?

then the semantic path must still include:
Baby I --performer of song--> performer of Baby I

because this variable is required by the original question.

============================================================
4. Semantic node rules
============================================================

Create a semantic node only if it is required for reasoning.

A node is required if removing it would change:
- what entity is queried;
- what intermediate variable must be resolved;
- what value is requested;
- what constraint must hold;
- what operator is applied;
- or what final answer type is expected.

Do not create ordinary semantic nodes for pure grammatical material, punctuation, auxiliary wording, surface wh words, prepositions, or parser artifacts.

A wh expression may determine final_answer_intent or answer type, but it should not usually become a semantic object node.

A predicate token may become a relation, value slot, constraint, or operator; it should not automatically become a node.

============================================================
5. Semantic edge rules
============================================================

Each semantic edge must:
1. connect semantic object nodes;
2. express a specific executable relation or operation;
3. introduce at most one unresolved target;
4. preserve necessary constraints;
5. be supported by the token path, original question, or both;
6. be convertible into one atomic question if it is a lookup edge.

Avoid vague relation labels such as "related to", "associated with", "connected to", or "about" unless the original question itself uses such a vague relation.

Use condition_node_ids when an edge requires an additional known or previously resolved constraint.

Example:
video 'One Last Time'
  --stars in video constrained by performer-->
person who stars in the video

where condition_node_ids points to:
performer of Baby I

============================================================
6. Atomic Question DAG rules
============================================================

Generate atomic questions from semantic edges.

Rules:
1. Each lookup atomic question should correspond to one lookup semantic edge.
2. Each lookup atomic question must ask for exactly one missing answer.
3. Do not ask multi-hop questions.
4. Do not merge two unresolved variables into one question.
5. If a question depends on a previous answer, its question text must explicitly mention every dependency as q1's answer, q2's answer, etc.
6. Do not use braced placeholders such as {{q1}}.
7. Do not leave unresolved ENTITY placeholders.
8. Preserve exact entity surface forms from original_question.
9. Preserve all necessary constraints.
10. The final leaf question must match final_answer_intent and final_answer_type.
11. depends_on may only reference previous q ids.
12. Do not add a follow-up question that merely restates or generalizes the answer already produced by its dependency.

A question must never refer to its own answer.

Bad:
q1: Who stars in the video 'One Last Time' by q1's answer?

Good:
q1: Who is the performer of Baby I?
q2: Who stars in the video 'One Last Time' by q1's answer?

If depends_on contains qN:
- the question text must explicitly mention qN's answer.
- every dependency in depends_on must be mentioned exactly as qN's answer.
- do not write a dependent question that can be read without the dependency.

When qN's answer already has the needed semantic type, do not wrap it in a redundant or type-changing noun phrase.

Bad:
q1: What county is Fort Deposit located in?
q2: What is the capital of the county of q1's answer?

Good:
q1: What county is Fort Deposit located in?
q2: What is the capital of q1's answer?

For possessive-wh questions such as "Whose X ...?", the final answer is the possessor or associated entity of X, not X itself.

Example:
Whose sister played Susie in miracle on 34th street?

Correct:
q1: Who played Susie in miracle on 34th street?
q2: Whose sister is q1's answer?

Incorrect:
q1: Who is the sister of the person who played Susie?

============================================================
7. Output schema
============================================================

Return exactly one JSON object:

{
  "question_plan": {
    "final_answer_intent": "what the original question ultimately asks",
    "final_answer_type": "entity | person | place | organization | work | event | date | number | boolean | value | set | unknown",
    "required_intermediate_variables": ["intermediate variable description"],
    "required_constraints": ["constraint description"],
    "required_operators": ["operator description"]
  },
  "semantic_reasoning_paths": [
    {
      "branch_id": "p1",
      "source_token_path": ["token copied from global_best_paths"],
      "semantic_nodes": [
        {
          "id": "p1_n1",
          "label": "semantic object label",
          "kind": "entity | intermediate_variable | value_slot | constraint | operator | answer_slot",
          "output_type": "entity | person | place | organization | work | event | date | number | boolean | value | set | unknown",
          "origin": "explicit_entity | path_evidence | question_required | derived_variable | operator",
          "path_evidence": ["tokens copied from source_token_path"],
          "question_evidence": ["phrase(s) from original_question"]
        }
      ],
      "semantic_edges": [
        {
          "id": "p1_e1",
          "source": "p1_n1",
          "target": "p1_n2",
          "condition_node_ids": [],
          "relation": "specific executable semantic relation",
          "edge_type": "lookup | constraint | compare | select | verify | intersect | aggregate",
          "evidence_status": "path_grounded | question_required | mixed | operator",
          "support_tokens": ["tokens copied from source_token_path"],
          "question_evidence": ["phrase(s) from original_question"],
          "atomic_question_hint": "one-hop atomic question corresponding to this edge"
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
      "semantic_edge_ids": ["p1_e1"],
      "output_node_id": "p1_n2",
      "output_type": "entity | person | place | organization | work | event | date | number | boolean | value | set | unknown"
    }
  ]
}

Schema requirements:
- branch_id values must be p1, p2, p3, ... in order.
- source_token_path must copy the corresponding global_best_paths entry exactly.
- semantic node ids must be p1_n1, p1_n2, ... inside each path.
- semantic edge ids must be p1_e1, p1_e2, ... inside each path.
- edge source, target, and condition_node_ids must refer to existing semantic node ids.
- terminal_node_id must refer to an existing semantic node id.
- support_tokens and path_evidence must be copied from source_token_path.
- support_tokens may be empty only when evidence_status is question_required or operator.
- path_evidence may be empty only when origin is question_required or operator.
- lookup atomic questions must cite at least one semantic_edge_id.
- q ids must be q1, q2, q3, ... in reasoning order.
- depends_on may only reference previous q ids.
- Return only JSON. Do not include markdown, comments, explanations, or extra fields.

============================================================
8. Final self-check
============================================================

Before returning, verify:

1. Did I infer the final answer intent from original_question?
2. Did I preserve all required constraints?
3. Did I include necessary semantics missing from global_best_paths?
4. Did I build semantic object nodes rather than raw token nodes?
5. Did I avoid one-token-one-node relabeling?
6. Does each lookup edge map to exactly one atomic question?
7. Does each atomic question ask for exactly one missing answer?
8. Are all qN's answer references legal and backward-pointing?
9. Does the final leaf question answer the original question?

Return valid JSON only.
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
