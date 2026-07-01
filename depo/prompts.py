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

to produce true semantic reasoning paths and then an Atomic Question DAG.

Do not answer the original question.
Do not use external knowledge.
Return valid JSON only.

You are given exactly:
- original_question
- explicit_entities
- global_best_paths

Do not assume any other input.

============================================================
Core idea
============================================================

The original question provides:
- final answer intent;
- expected answer type;
- explicit entities;
- intermediate variables;
- constraints;
- modifier attachment;
- comparison, selection, verification, aggregation, and superlative logic.

The token paths provide:
- parser-grounded token-level structural evidence;
- useful anchor tokens;
- useful relation cues;
- useful constraint/operator cues.

A token path is NOT a semantic reasoning path.

Your job is:

token path
+ original question semantics
->
semantic reasoning path
->
atomic question DAG

============================================================
What is a semantic reasoning path?
============================================================

A semantic reasoning path is a directed object-relation structure.

It contains:

1. Semantic nodes:
   Meaning-bearing reasoning objects.

2. Semantic edges:
   Executable semantic relations between those objects.

A semantic node is not a token.
A semantic edge is not a dependency edge.

A semantic node should represent one of:
- an explicit known entity;
- an intermediate variable to be found;
- a requested value slot;
- a necessary constraint;
- an operator;
- a final answer slot.

A semantic edge should represent one executable relation or operation, such as:
- identify a variable from a known entity;
- retrieve a value of a resolved variable;
- apply a constraint;
- compare two values;
- select an item;
- verify a condition;
- intersect or aggregate branch results.

Each lookup semantic edge should be convertible into exactly one atomic question.

Example:

Original question:
Where does the director of film A Nest Of Noblemen work at?

Token path:
A Nest Of Noblemen ---- director ---- work

Correct semantic reasoning path:
A Nest Of Noblemen
  ---director of film--->
director
  ---works at--->
workplace

Atomic questions:
q1: Who directed A Nest Of Noblemen?
q2: Where does q1's answer work?

This is correct because:
- "A Nest Of Noblemen" is a known entity node.
- "director" is an intermediate variable node.
- "workplace" is a value/answer slot node.
- "director of film" is one executable lookup edge.
- "works at" is one executable lookup edge.

============================================================
Critical anti-relabeling rule
============================================================

Do NOT create one semantic node for each token in source_token_path.

Do NOT preserve the token path as a semantic node chain.

Do NOT merely add relation labels between adjacent path tokens.

Bad:

source_token_path:
Baby I ---- performer ---- One Last Time ---- video ---- stars ---- Who

bad semantic path:
Baby I
  --performer of song-->
performer
  --performed in-->
One Last Time
  --video of song-->
video
  --stars in video-->
Who

This is token-path relabeling, not semantic reasoning.

Correct behavior:
Use the original question to identify the true semantic objects and relations.

For:
Who stars in the video 'One Last Time' by the performer of Baby I?

A better semantic path is:
Baby I
  --performer of song-->
performer of Baby I
One Last Time video, constrained by performer of Baby I
  --stars in video-->
person who stars in the video

This yields:
q1: Who is the performer of Baby I?
q2: Who stars in the video 'One Last Time' by q1's answer?

If a required relation is missing from global_best_paths but present in original_question, include it as question_required evidence.

============================================================
How to convert token paths into semantic reasoning paths
============================================================

First derive a question-level semantic plan from original_question.

Identify:
1. final_answer_intent:
   What is the original question ultimately asking for?

2. final_answer_type:
   person, place, organization, work, event, date, number, boolean, value, set, or unknown.

3. required intermediate variables:
   What unknown entities or values must be resolved before the final answer?

4. required constraints:
   Which modifiers, relative clauses, appositions, titles, locations, dates, superlatives, or prepositional phrases restrict the answer?

5. required operators:
   same/different, comparison, ordering, selection, verification, intersection, aggregation, superlative, etc.

Then align global_best_paths to this plan.

For each path:
- use path tokens as evidence;
- do not copy them as nodes;
- decide which path tokens are semantic objects;
- decide which path tokens are relation cues;
- decide which path tokens are constraints/operators;
- fold grammatical tokens into relations or discard them;
- correct path direction using original_question information flow.

Reasoning direction should follow:

known entity
-> intermediate variable
-> requested value slot
-> final operator / final answer

Path order alone is not semantic direction.

============================================================
Semantic node rules
============================================================

Create a semantic node only if it is required for reasoning.

A node is required if removing it would change:
- the entity being queried;
- the intermediate variable to resolve;
- the value being retrieved;
- the constraint being preserved;
- the final answer type;
- or the operator being applied.

Do not create semantic nodes for:
- pure grammar;
- punctuation;
- wh surface words as ordinary objects;
- auxiliary verbs;
- prepositions by themselves;
- path tokens that only function as relation cues;
- parser artifacts.

A wh word can influence final_answer_intent or final_answer_type, but it should not normally become a semantic node.

A surface predicate can become:
- a semantic relation;
- a value slot;
- a constraint;
- or an operator,
but it should not automatically become a semantic node.

============================================================
Semantic edge rules
============================================================

Each semantic edge must:
1. connect semantic object nodes;
2. express a specific executable relation or operation;
3. introduce at most one new unresolved target;
4. preserve relevant constraints;
5. be supported by source_token_path, original_question, or both;
6. be convertible into one atomic question if it is a lookup edge.

Use condition_node_ids when an edge requires an additional already-known or previously-resolved constraint.

Example:
For "video One Last Time by the performer of Baby I":

Node A: One Last Time video
Node B: performer of Baby I
Node C: person who stars in the video

Edge:
source = Node A
condition_node_ids = [Node B]
target = Node C
relation = "stars in video constrained by performer"
atomic question = "Who stars in the video 'One Last Time' by q1's answer?"

============================================================
Question-required semantics
============================================================

global_best_paths may omit necessary semantics from original_question.

Do not drop missing required semantics.

If a necessary variable or constraint appears in original_question but not in source_token_path:
- create a semantic node with origin = "question_required";
- create an edge with evidence_status = "question_required" or "mixed";
- use support_tokens = [] when no source_token_path token supports it;
- include question_evidence from original_question.

Example:
If global_best_path is:
One Last Time ---- video ---- stars ---- Who

but original_question is:
Who stars in the video 'One Last Time' by the performer of Baby I?

Then "performer of Baby I" is required even though it is missing from the path.
Create a question_required edge:
Baby I -> performer of Baby I

============================================================
Atomic Question DAG generation
============================================================

Generate atomic questions from semantic edges.

Rules:
1. Each lookup atomic question should correspond to one semantic edge.
2. Each lookup atomic question asks for exactly one missing answer.
3. Do not ask multi-hop questions.
4. Do not merge two unresolved variables into one question.
5. If a question depends on a previous answer, its question text must explicitly mention every dependency as q1's answer, q2's answer, etc.
6. Do not use braced placeholders such as {{q1}}.
7. Do not leave unresolved ENTITY placeholders.
8. Preserve exact entity surface forms from original_question.
9. Preserve all constraints.
10. Preserve final answer intent and final answer type.
11. Every depends_on id must refer only to a previous q id.
12. The final leaf question must match final_answer_intent.
13. Do not add a follow-up question that merely restates or generalizes the answer already produced by its dependency.

For operator questions:
- generate lookup branches first;
- then generate final compare/select/verify/intersect/aggregate question;
- operator questions may have semantic_edge_ids = [] only if they combine previous answers and do not require new retrieval.

============================================================
Dependency and reference rules
============================================================

If a question text mentions qN's answer:
- N must be smaller than the current question id.
- qN must appear in depends_on.

If depends_on contains qN:
- the question text must explicitly mention qN's answer.
- every dependency in depends_on must be mentioned exactly as qN's answer.
- do not write a dependent question that can be read without the dependency.

A question must never refer to its own answer.
Bad:
q1: Who stars in the video 'One Last Time' by q1's answer?

Correct:
q1: Who is the performer of Baby I?
q2: Who stars in the video 'One Last Time' by q1's answer?

When qN's answer already has the needed semantic type, do not wrap it in another noun phrase that duplicates or changes its type.

Bad:
q2: What is the capital of the county of q1's answer?
when q1's answer is already a county.

Good:
q2: What is the capital of q1's answer?

Avoid dangling generic follow-up questions.
Bad:
q1: Which city shares a county with Helvetia?
q2: How long are the council terms of q1's answer?
q3: What is the length of council terms?

Correct:
q1: Which city shares a county with Helvetia?
q2: How long are the council terms of q1's answer?

============================================================
Possessive-WH rule
============================================================

For questions of the form "Whose X ...?", the final answer is the possessor/associated entity of X, not X itself.

Do not convert:
Whose sister played Susie in miracle on 34th street?

into:
Who is the sister of the person who played Susie?

Correct decomposition:
q1: Who played Susie in miracle on 34th street?
q2: Whose sister is q1's answer?

The semantic path should represent:
Susie + miracle on 34th street
  --played by in work-->
actor
actor
  --is sister of whose person-->
possessor / final answer

============================================================
Output schema
============================================================

Return exactly one JSON object:

{
  "question_plan": {
    "final_answer_intent": "what the original question ultimately asks",
    "final_answer_type": "person | place | organization | work | event | date | number | boolean | value | set | unknown",
    "required_constraints": ["constraint phrase or description"],
    "required_intermediate_variables": ["intermediate variable description"]
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
          "path_evidence": ["tokens from source_token_path when available; original-question evidence is allowed when needed"],
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
          "support_tokens": ["tokens from source_token_path when available; may be empty or question-level when needed"],
          "question_evidence": ["phrase(s) from original_question"],
          "atomic_question_hint": "one-hop atomic question corresponding to this edge"
        }
      ],
      "terminal_node_id": "p1_nK",
      "folded_or_discarded_tokens": [
        {
          "token": "token copied from source_token_path",
          "reason": "folded_into_relation | grammatical | wh_answer_intent | noisy_parser_artifact | duplicate"
        }
      ]
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
      "output_type": "person | place | organization | work | event | date | number | boolean | value | set | unknown"
    }
  ]
}

Schema rules:
1. branch_id values must be p1, p2, p3, ... in order.
2. source_token_path must copy the corresponding global_best_paths entry exactly.
3. semantic node ids must be p1_n1, p1_n2, ... inside p1; p2_n1, p2_n2, ... inside p2.
4. semantic edge ids must be p1_e1, p1_e2, ... inside p1; p2_e1, p2_e2, ... inside p2.
5. semantic edge source, target, and condition_node_ids must refer to existing semantic node ids.
6. terminal_node_id must refer to an existing semantic node id.
7. support_tokens and path_evidence should prefer source_token_path tokens when available, but may include original-question evidence when the token path omits required semantics.
8. support_tokens may be empty when evidence comes only from original_question; cite question_evidence in that case.
9. path_evidence may be empty when origin is question_required/operator or when the evidence is represented in question_evidence.
10. A lookup atomic question must cite at least one semantic_edge_id.
11. Each cited semantic_edge_id must exist.
12. output_node_id must refer to the target node of the cited lookup edge when possible.
13. q ids must be q1, q2, q3, ... in reasoning order.
14. depends_on may only reference previous q ids.
15. Return only JSON. Do not include markdown, explanations, or comments.

============================================================
Self-check before returning
============================================================

Before returning, verify:

1. Did I build semantic object nodes, not token nodes?
2. Did I avoid one-token-one-node relabeling?
3. Did I avoid merely labeling adjacent token edges?
4. Did I preserve final answer intent and final answer type?
5. Did I preserve all required constraints?
6. Did I include required question semantics missing from global_best_paths?
7. Does every lookup edge correspond to one atomic question?
8. Does every lookup atomic question ask for exactly one missing answer?
9. Are all qN's answer references legal and backward-pointing?
10. Does the final leaf answer the original question?

Now generate the JSON object for the given input.
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

