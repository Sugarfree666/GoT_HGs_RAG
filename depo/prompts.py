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
Person, place, and organization mentions may also contain disambiguating parentheticals or appositive titles. Preserve the complete surface form when the modifier is part of identifying the entity, such as "Christopher Newton (Criminal)" or "John Ernest, Duke Of Saxe-Eisenach".
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
10. Parenthetical or appositive disambiguators attached to a named entity are part of the entity when they identify the mention, e.g. "Christopher Newton (Criminal)" and "John Ernest, Duke Of Saxe-Eisenach". Do not split the appositive into an independent entity when it is only a title/disambiguator.
11. In typed comparison or choice lists, such as "Which film ..., A or B?", verify each branch that looks like a title as its own entity. A branch may start with a number when the number is part of the title; do not reduce it to the alphabetic substring.
12. A candidate title may start with a capitalized question-like word such as When, What, Who, Where, or Which. If the whole candidate is an official-looking work/title, keep that first word; do not shorten it only because it resembles a wh-word.
13. Do not answer the question or decompose it.
14. Do not remove any restriction or change the semantics.
15. normalized_question must be a single question sentence.
16. Only apply light parser-friendly grammar repair. For example, "Which country the composer of film Thunder On The Hill is from?" becomes "Which country is the composer of film Thunder On The Hill from?"
17. If the original question is already grammatical and natural, return it unchanged and set normalization_changed=false.

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
10. Parenthetical or appositive disambiguators attached to a named entity are part of the entity when they identify the mention, e.g. "Christopher Newton (Criminal)" and "John Ernest, Duke Of Saxe-Eisenach". Do not split the appositive into an independent entity when it is only a title/disambiguator.
11. In typed comparison or choice lists, such as "Which film ..., A or B?", extract each title-like branch as its own entity. A branch may start with a number when the number is part of the title; do not drop the numeric token or return only the alphabetic substring.
12. Some official titles begin with capitalized question-like words, e.g. When/What/Who/Where/Which as the first word of a song, book, film, episode, or other work title. If the complete contiguous title starts with such a word, return the complete title and do not trim off the first word.
13. Do not answer the question or decompose it.
14. Do not remove any restriction or change the semantics.
15. normalized_question must be a single question sentence.
16. Only apply light parser-friendly grammar repair. For example, "Which country the composer of film Thunder On The Hill is from?" becomes "Which country is the composer of film Thunder On The Hill from?"
17. If the original question is already grammatical and natural, return it unchanged and set normalization_changed=false.

Return JSON only.
Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


MASK_SPAN_EXTRACTION_SYSTEM = EXPLICIT_ENTITY_EXTRACTION_SYSTEM


def build_mask_span_extraction_prompt(question: str) -> str:
    return build_explicit_entity_extraction_prompt(question)


ATOMIC_QUESTION_DAG_SYSTEM = r"""
You are DEPO Step 5: Retrieval-Executable Atomic Question DAG Generator.

Your task is to decompose the original question into the smallest DAG whose nodes can be executed reliably by the downstream retrieval-and-answering system.

Do not answer the original question.
Do not use external knowledge.
Return exactly one valid JSON object with one top-level key: "atomic_questions".
Do not output explanations, plans, reasoning traces, markdown, comments, or warnings.

You are given:
- original_question: the semantic authority.
- topic_entities: explicit named entities from the original question.
- step4_paths: token-level structural hints selected by Step4.

The original_question is authoritative. step4_paths are only structural hints. They may be incomplete, noisy, reversed, or redundant. Use them to detect latent variables, relation chains, candidate branches, constraints, and operators, but never copy token order mechanically.
DAG nodes do not need path support from step4_paths when the original_question requires the node.

============================================================
PRIMARY OBJECTIVE: RETRIEVAL-EXECUTABLE ATOMICITY
============================================================

Generate the smallest RETRIEVAL-EXECUTABLE DAG, not merely the smallest syntactic DAG.

A lookup node is retrieval-executable only when, after every qN's answer reference is replaced by its established answer, the question:
1. asks for exactly one missing entity, attribute, value, set, or fact;
2. applies one target relation or attribute to a concrete named anchor or to one entity-like dependency answer;
3. contains enough explicit wording to identify that anchor and relation;
4. does not hide another unresolved entity/value inside a possessive phrase, role phrase, relative clause, location phrase, or nested relational description;
5. does not also perform comparison, selection, verification, or aggregation.

A lookup node may contain modifiers needed to disambiguate the same one-hop relation. It must be split when an unnamed intermediate object must first be identified and then queried by another relation.

Use the LATENT-BRIDGE TEST:
- Ask whether part of the question denotes an unknown person/place/organization/work/event/value that must be resolved before the outer requested relation can be evaluated.
- If yes, create a lookup node for that intermediate object and a later node that explicitly uses qN's answer.
- If no, keep the direct relation in one lookup node.

============================================================
MANDATORY SPLIT PATTERNS
============================================================

Split all outer-attribute questions over an unnamed role or relation result.

1. Possessive role or kinship bridge
- "Where did Coulson Wallop's father study?"
  q1: "Who is Coulson Wallop's father?"
  q2: "Where did q1's answer study?"

- "Where did Sylvia Burka's husband die?"
  q1: "Who is Sylvia Burka's husband?"
  q2: "Where did q1's answer die?"

2. Role-of-entity bridge
- "Where was the director of The Outlaw Express born?"
  q1: "Who directed The Outlaw Express?"
  q2: "Where was q1's answer born?"

- "What nationality is the performer of When The Stars Go Blue?"
  q1: "Who performed When The Stars Go Blue?"
  q2: "What is the nationality of q1's answer?"

3. Nested relative or location bridge
- "Which region is immediately north of the region where Israel is located?"
  q1: "Which region is Israel located in?"
  q2: "Which region is immediately north of q1's answer?"

- "When was the region immediately north of the region where Israel is located created?"
  q1: "Which region is Israel located in?"
  q2: "Which region is immediately north of q1's answer?"
  q3: "When was q2's answer created?"

4. Nested target followed by another attribute
- "How long are the city council terms of the city that shares a county with Helvetia?"
  q1: "Which city shares a county with Helvetia?"
  q2: "How long are the city council terms of q1's answer?"

5. Multi-relation chains
If the semantic chain is A --r1--> B --r2--> C, and B is not explicitly named, use two lookup nodes. Continue similarly for longer chains. Do not compress two entity-changing relations into one lookup node.

============================================================
DO NOT OVER-SPLIT
============================================================

Keep one lookup node when the requested answer is directly returned by one relation from a known anchor and there is no hidden intermediate object that is queried again.

Valid one-node lookups:
- "Who is Coulson Wallop's father?"
- "What is the label of Vilaiyaadu Mankatha?"
- "Which city shares a county with Helvetia?"
- "When was The Outlaw Express released?"

The phrase "shares a county with Helvetia" directly defines the requested city. The county is not separately requested and is not later queried, so it need not become its own node.

Do not create nodes for grammatical words, generic categories, relation labels, or values that are never consumed.

============================================================
DEPENDENCY AND ANCHOR SAFETY
============================================================

1. Use ids q1, q2, q3, ... in order.
2. depends_on may contain only earlier q ids.
3. If depends_on contains qN, the question text must explicitly contain exactly the reference "qN's answer".
4. If the question text contains "qN's answer", depends_on must contain qN.
5. Never use vague references such as "the person", "that city", "the group", "it", or "they" for a previous answer.
6. A dependent lookup must remain answerable after dependency substitution.
7. A lookup node that uses a dependency as its retrieval anchor should consume an entity-like output: entity, person, place, organization, work, or event.
8. Date, year, number, boolean, and ordinary scalar/value outputs should normally feed compare, select, verify, or aggregate nodes, not another entity lookup.
9. Independent evidence branches must not depend on each other.
10. Preserve exact topic-entity surface forms when used.
11. Do not output unresolved ENTITYA/ENTITYB placeholders.

============================================================
OPERATIONS
============================================================

lookup:
- retrieves one relation or attribute from a known anchor;
- returns one entity/value/set needed by a later node or by the final answer.

compare:
- compares already retrieved values and returns a relation or judgment;
- must not introduce a new factual lookup.

select:
- chooses the requested candidate/entity using already retrieved branch values;
- must preserve the exact candidate surfaces from the original question.

verify:
- returns a boolean based on already retrieved facts.

aggregate:
- counts, sums, lists, minimizes, maximizes, or otherwise aggregates an already retrieved set or values.

For comparison/candidate questions:
- build independent lookup branches for each candidate;
- retrieve the exact compared attribute for each branch;
- use one final compare/select/verify node that depends on every required branch.

For "lived longer", retrieve sufficient lifespan information for every candidate, not birth dates alone.

============================================================
CONSTRAINT COVERAGE AND BRANCH COMPLETENESS
============================================================

Preserve every restrictive condition that changes which entity is being asked about.

If several branches jointly identify one target:
- resolve each required branch, or retain a branch directly in the target lookup when it is genuinely one-hop;
- combine all required branches before asking the final attribute;
- never leave a required branch as a separate unused leaf.

If branches are candidate alternatives, keep them independent until the terminal compare/select/verify node.

Every non-final node must be an ancestor of the final node.
There must be exactly one leaf node.
The unique leaf must be the final qN node.
The unique leaf must return the answer type and intent requested by the original question.

============================================================
OUTPUT SCHEMA
============================================================

Return exactly:

{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "atomic question?",
      "depends_on": [],
      "operation": "lookup | compare | select | verify | aggregate",
      "output_type": "entity | person | place | organization | work | event | date | number | boolean | value | set | unknown"
    }
  ]
}

Do not output fields other than id, question, depends_on, operation, and output_type.
Every question must be one grammatical question ending in exactly one question mark.

============================================================
EXAMPLES
============================================================

Example A — Possessive bridge
Input question:
"Where did Coulson Wallop's father study?"

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Who is Coulson Wallop's father?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    },
    {
      "id": "q2",
      "question": "Where did q1's answer study?",
      "depends_on": ["q1"],
      "operation": "lookup",
      "output_type": "place"
    }
  ]
}

Example B — Nested location chain
Input question:
"When was the region immediately north of the region where Israel is located created?"

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Which region is Israel located in?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "place"
    },
    {
      "id": "q2",
      "question": "Which region is immediately north of q1's answer?",
      "depends_on": ["q1"],
      "operation": "lookup",
      "output_type": "place"
    },
    {
      "id": "q3",
      "question": "When was q2's answer created?",
      "depends_on": ["q2"],
      "operation": "lookup",
      "output_type": "date"
    }
  ]
}

Example C — Direct lookup, no over-splitting
Input question:
"Which city shares a county with Helvetia?"

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Which city shares a county with Helvetia?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "place"
    }
  ]
}

Example D — Parallel candidate comparison
Input question:
"Which film has the younger director, Dangerously They Live or Salad By The Roots?"

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Who directed Dangerously They Live?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    },
    {
      "id": "q2",
      "question": "When was q1's answer born?",
      "depends_on": ["q1"],
      "operation": "lookup",
      "output_type": "date"
    },
    {
      "id": "q3",
      "question": "Who directed Salad By The Roots?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    },
    {
      "id": "q4",
      "question": "When was q3's answer born?",
      "depends_on": ["q3"],
      "operation": "lookup",
      "output_type": "date"
    },
    {
      "id": "q5",
      "question": "Based on q2's answer and q4's answer, which film has the younger director: Dangerously They Live or Salad By The Roots?",
      "depends_on": ["q2", "q4"],
      "operation": "select",
      "output_type": "work"
    }
  ]
}

============================================================
FINAL SILENT AUDIT
============================================================

Before returning JSON, silently verify all of the following:
1. The output has only the atomic_questions top-level key.
2. Each lookup passes the retrieval-executable one-link test.
3. Every hidden role, kinship, nested location, or relative-clause bridge is split when an outer relation queries its result.
4. No direct one-hop lookup is unnecessarily split.
5. Every qN reference and depends_on entry matches exactly.
6. Every dependent lookup has a usable entity anchor after substitution.
7. Scalar dependencies feed operator nodes rather than unsupported entity lookups.
8. All restrictive constraints and candidate branches are preserved.
9. There is exactly one leaf, it is the final qN node, and every earlier node reaches it.
10. The final leaf preserves the original answer intent and answer type.
11. No unresolved placeholders or invented facts appear.
12. The result is valid JSON only.
""".strip()



def build_atomic_question_dag_prompt(
    original_question: str,
    explicit_entities: list[str],
    global_best_paths: list[list[str]],
) -> str:
    payload = {
        "original_question": original_question,
        "topic_entities": [str(entity) for entity in explicit_entities],
        "step4_paths": [[str(node) for node in path] for path in global_best_paths],
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
