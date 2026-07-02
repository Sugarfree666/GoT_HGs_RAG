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
It defines:
- final answer intent;
- final answer type;
- explicit entities;
- intermediate variables;
- constraints;
- modifier attachment;
- candidate alternatives;
- comparison, selection, verification, aggregation, and other operator semantics.

The global_best_paths are structural evidence.
They are token-level parser paths.
They are not semantic reasoning paths.

Your transformation is:

original_question semantics
+
token-level structural evidence
->
semantic object-relation reasoning paths
->
atomic question DAG

============================================================
1. Semantic Reasoning Path
============================================================

A semantic reasoning path is a directed object-relation structure.

It contains:
- semantic nodes: meaning-bearing reasoning objects;
- semantic edges: semantic relations, constraints, or operator links between those objects.

A semantic node is not merely a token.
A semantic edge is not merely a dependency edge.
A semantic path is not a copy of the token path.

A semantic node should represent only something needed for reasoning:
- a known entity;
- an unresolved intermediate variable;
- a requested value slot;
- a necessary constraint;
- an operator;
- a final answer slot.

A semantic edge should represent a meaningful semantic relation or transition:
- identify an intermediate variable from a known object;
- retrieve or describe a value of a resolved object;
- apply or preserve a constraint;
- compare values;
- select an answer;
- verify a condition;
- combine branch results.

Semantic edges are reasoning-path structure, not an atomic-question execution trace.
Do not require one semantic edge to map to one atomic question.
Do not require every atomic question to cite a semantic edge.
The Atomic Question DAG should be generated from the semantic reasoning paths together with the original_question.

============================================================
2. Evidence Roles
============================================================

Use the two evidence sources differently.

original_question:
- determines meaning;
- determines what must be preserved;
- determines reasoning direction;
- determines the final answer type;
- determines whether a token is an entity, variable, value, constraint, relation cue, or operator cue.

global_best_paths:
- provide token-level structural evidence;
- suggest useful anchors and relation cues;
- suggest branch structure;
- suggest candidate alternatives;
- may be incomplete, noisy, direction-ambiguous, or over-inclusive.

Do not let token order alone determine semantic direction.
Reasoning direction should follow information flow:

known object
-> intermediate variable
-> requested value
-> operator or final answer

============================================================
3. Anti-Relabeling Principle
============================================================

Do not create one semantic node for each token in source_token_path.
Do not preserve the token path as a semantic node chain.
Do not merely add relation labels between adjacent tokens.

A token or span may be:
- kept as a semantic node if it denotes a required reasoning object;
- folded into an edge relation if it expresses a relation;
- represented as a constraint if it restricts a variable or answer;
- represented as an operator if it controls comparison, selection, verification, or aggregation;
- used only to determine final answer intent;
- discarded if it is purely grammatical, duplicate, or parser noise.

The number of semantic nodes is usually smaller than the number of path tokens.

============================================================
4. Question-Level Plan
============================================================

Before building semantic reasoning paths, infer a question-level plan from original_question.

Identify:

1. final_answer_intent
   What is the original question ultimately asking for?

2. final_answer_type
   Choose one:
   entity, person, place, organization, work, event, date, number, boolean, value, set, unknown

3. required_intermediate_variables
   Unknown objects or values that must be resolved before the final answer.

4. required_constraints
   Conditions, modifiers, descriptions, locations, times, appositions, relative clauses, candidate restrictions, or other restrictions that must be preserved.

5. required_operators
   Any comparison, equality test, ordering, selection, verification, intersection, aggregation, or superlative operation required by the question.

The semantic paths and the final atomic DAG must preserve this plan.

============================================================
5. Building Semantic Paths
============================================================

For each evidence path in global_best_paths:

1. Copy the path exactly into source_token_path.

2. Determine the primary semantic role of this path:
   - single_path: one ordinary reasoning branch;
   - candidate: one branch for a candidate alternative.

3. Identify the primary candidate if the path represents one candidate alternative.

4. Convert the token path into a semantic object-relation path:
   - keep only required semantic objects as nodes;
   - convert relation cues into edge relations;
   - preserve required constraints;
   - preserve required operators;
   - ignore or fold non-semantic tokens;
   - correct direction using the original_question.

5. If a required semantic element appears in original_question but is missing from the token path, include it with question_required or mixed evidence.

6. If a path contains an explicit entity that belongs to another parallel candidate branch because of coordination or parser artifacts, do not use it as a semantic object in the current candidate branch unless the original_question explicitly relates both entities within that same branch.

============================================================
6. Parallel Candidate Branches
============================================================

When global_best_paths represent parallel candidate alternatives, build separate semantic reasoning paths for the alternatives.

Do not merge multiple candidates into one semantic path.

For each candidate branch:
- build one semantic path centered on that candidate;
- resolve the branch-specific intermediate variables;
- resolve the branch-specific value or object needed by the original question's operator;
- make the branch terminal the evidence value or branch output needed for later final answering.

Do not build an extra pure operator semantic path such as p3 only to compare branch outputs.
The downstream HyperBranch answer stage will combine the original question with the retrieved branch facts.
Represent comparison or selection semantics as branch_role, edge_type, relation wording, node labels, or folded operator evidence inside the candidate paths.

============================================================
7. Intermediate Variable Preservation
============================================================

Preserve intermediate variables in the semantic reasoning path when they are needed to explain the reasoning.

If the question requires:

known object
-> intermediate variable
-> attribute or value

then the semantic path should make the intermediate variable visible instead of hiding it completely.

This is a semantic-path requirement, not a one-edge-one-question requirement.
The final Atomic Question DAG may use one or more questions as needed, as long as the questions are answerable, dependencies are explicit, and the final answer intent is preserved.

============================================================
8. Comparison and Selection
============================================================

For comparison, ordering, superlative, equality, difference, or selection:

1. Resolve each candidate or branch first.
2. Resolve the value being compared in each branch.
3. Do not add a final operator semantic path or final operator atomic question only to answer the original question.
4. Leave the final comparison, selection, equality check, or verification to the downstream HyperBranch answer stage.

The compared value is evidence for the decision.
It is not necessarily the final answer.

Use operator words to decide which branch facts must be retrieved.
For example:
- younger or older between people usually requires birth date or age; prefer birth date when it is natural to ask;
- born first, born earlier, born later requires birth date;
- died first, died earlier, died later requires death date;
- same nationality requires nationality for each compared person;
- larger, smaller, highest, lowest, most, fewest requires the relevant numeric value for each branch.

Do not make operator words such as younger, older, first, same, larger, or highest into dangling value nodes unless they are clearly the requested answer slot.
Do not use branch relations such as "is younger than" unless both compared objects are already explicit inputs to that one question.
Prefer value nodes such as birth date, death date, nationality, population, count, height, date, or score.

Bad semantic candidate branch:
A --has director--> director of A --is younger than--> younger

Good semantic candidate branch:
A --has director--> director of A --birth date for younger comparison--> birth date of director of A

============================================================
9. Semantic Node Criteria
============================================================

Create a semantic node only if it is required for reasoning.

A node is required if removing it would change:
- what is being queried;
- which intermediate variable must be resolved;
- which value is requested;
- which constraint is preserved;
- which operator is applied;
- or what final answer type is expected.

Do not create ordinary semantic object nodes for:
- purely grammatical material;
- punctuation;
- auxiliary wording;
- prepositions by themselves;
- parser artifacts;
- surface question words that only indicate answer intent.

Question words may influence final_answer_intent and final_answer_type, but they should not normally become semantic object nodes.

A predicate token may become:
- an edge relation;
- a value slot;
- a constraint;
- or an operator.

It should not automatically become a node.
In candidate comparison branches, an operator token such as younger, older, first, same, or larger should normally be folded into the relation or folded_or_discarded_tokens, while the terminal node should be the concrete compared value.

============================================================
10. Semantic Edge Criteria
============================================================

Each semantic edge must:

1. connect existing semantic object nodes;
2. express a specific semantic relation, constraint, or operation;
3. preserve necessary constraints;
4. be supported by the token path, the original question, or both;
5. help explain the reasoning path used to build the final Atomic Question DAG.

Avoid vague relations unless the original question itself is vague.

Use condition_node_ids when an edge requires an additional already-known or previously-resolved constraint.

An edge target should represent the next semantic object, value, constraint, operator output, or final answer slot in the reasoning path.

An operator edge should have a target node representing the selected answer, boolean judgment, aggregated result, or final answer slot.

============================================================
11. Atomic Question DAG
============================================================

Generate the Atomic Question DAG from:
- the original_question;
- explicit_entities;
- the semantic_reasoning_paths you induced from global_best_paths.

The semantic reasoning paths guide the decomposition, but they are not an execution trace.
An atomic question may be supported by one semantic edge, multiple semantic edges, a whole semantic path segment, or question-required semantics.
semantic_edge_ids are optional debug/provenance fields only; do not rely on them as the contract.

Rules:

1. Each lookup atomic question should ask for one missing answer whenever possible.
2. Do not create unnecessary dangling questions.
3. If a question depends on a previous answer, mention that dependency as q1's answer, q2's answer, etc.
4. Do not use braced placeholders such as {{q1}}.
5. Do not leave unresolved ENTITY placeholders.
6. Preserve exact entity surface forms from original_question.
7. Preserve all necessary constraints.
8. depends_on may only reference previous q ids.
9. A question must never refer to its own answer.
10. Final leaf questions should provide the retrieved evidence facts needed by the original question.
11. Do not add a final atomic question that merely asks the original comparison, selection, or verification question.
12. Do not collapse unresolved branch reasoning into one direct comparison question.

Bad for "Which film has a director who is younger, A or B?":
q1: Is the director of A younger than the director of B?

Better:
q1: Who directed A?
q2: When was q1's answer born?
q3: Who directed B?
q4: When was q3's answer born?

No final q5 is needed; the downstream HyperBranch answer stage will use the original question and the retrieved facts.

If depends_on contains qN, the question text must explicitly mention qN's answer.
Do not write a depends_on entry that is not used in the question text.

Bad:
q1: Who directed A?
q2: What is the age of the director of A?
depends_on: ["q1"]

Good:
q1: Who directed A?
q2: When was q1's answer born?
depends_on: ["q1"]

Do not create dangling questions.
Every non-final question should either support a later question or be part of the final answer.

When a previous answer already has the needed semantic type, do not wrap it in a redundant or type-changing phrase.

============================================================
12. Output Schema
============================================================

Return exactly one JSON object:

{
  "semantic_reasoning_paths": [
    {
      "branch_id": "p1",
      "source_token_path": ["token copied from global_best_paths"],
      "branch_role": "candidate | single_path",
      "primary_candidate": "explicit entity for this branch, or empty string",
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
          "relation": "specific semantic relation, constraint, or operator transition",
          "edge_type": "lookup | constraint | compare | select | verify | intersect | aggregate",
          "evidence_status": "path_grounded | question_required | mixed | operator",
          "support_tokens": ["tokens copied from source_token_path"],
          "question_evidence": ["phrase(s) from original_question"],
          "atomic_question_hint": "optional natural-language question hint"
        }
      ],
      "terminal_node_id": "p1_nK",
      "folded_or_discarded_tokens": [
        {
          "token": "token copied from source_token_path",
          "reason": "folded_into_relation | grammatical | wh_answer_intent | noisy_parser_artifact | duplicate | coordination_artifact"
        }
      ]
    }
  ],
  "atomic_question_dag": {
    "atomic_questions": [
      {
        "id": "q1",
        "question": "natural-language atomic question?",
        "depends_on": [],
        "operation": "lookup | compare | select | verify | intersect | aggregate",
        "output_type": "entity | person | place | organization | work | event | date | number | boolean | value | set | unknown"
      }
    ]
  }
}

Schema requirements:
- The top-level object must contain semantic_reasoning_paths and atomic_question_dag.
- Do not output question_plan as a top-level field; use the question-level plan internally.
- branch_id values must be p1, p2, p3, ... in order.
- semantic_reasoning_paths should normally contain one path per global_best_paths entry.
- source_token_path must copy the corresponding global_best_paths entry exactly.
- Do not add extra pure operator paths such as p3 when there are only two candidate evidence paths.
- candidate branches must not merge multiple candidates into one semantic path.
- semantic node ids must be p1_n1, p1_n2, ... inside each path.
- semantic edge ids must be p1_e1, p1_e2, ... inside each path.
- edge source, target, and condition_node_ids must refer to existing semantic node ids.
- terminal_node_id must refer to an existing semantic node id.
- support_tokens and path_evidence must be copied from source_token_path.
- support_tokens may be empty only when evidence_status is question_required or operator.
- path_evidence may be empty only when origin is question_required or operator.
- atomic_question_dag.atomic_questions must be a non-empty list.
- atomic questions do not need semantic_edge_ids.
- If semantic_edge_ids or output_node_id are included, treat them only as optional debug provenance, not as required bindings.
- q ids must be q1, q2, q3, ... in reasoning order.
- depends_on may only reference previous q ids.
- Return only JSON. Do not include markdown, comments, or explanations.

============================================================
13. Final Self-Check
============================================================

Before returning, verify:

1. Did I infer the final answer intent from original_question?
2. Did I preserve all required constraints?
3. Did I include necessary semantics missing from global_best_paths?
4. Did I generate separate semantic paths for parallel candidate branches?
5. Did I avoid merging multiple candidates into one semantic path?
6. Did I build semantic object nodes rather than raw token nodes?
7. Did I avoid one-token-one-node relabeling?
8. Did I preserve intermediate variables when they are needed to explain the reasoning?
9. Did I avoid forcing one semantic edge to become one atomic question?
10. Does each atomic question ask for a clear missing answer or perform a clear operator step?
11. For comparison or selection, did I retrieve the compared values for each branch?
12. Did I avoid adding a final compare/select/verify question that merely answers the original question?
13. Are all qN's answer references legal and backward-pointing?
14. Do the final leaf questions provide evidence facts for the downstream HyperBranch answer stage?

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
