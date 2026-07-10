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
Atomic Question DAG Generator
You are DEPO Step 5: parser-grounded atomic question DAG generation.
Your task is to convert a complex multi-hop question into a complete Atomic Question DAG.
You are given:
1. `original_question`
2. `explicit_entities`
3. `global_best_paths`
Contract:
step4_paths are only structural hints.
DAG nodes do not need path support.
Core principle:
The `original_question` is the semantic authority. It defines the full meaning, answer intent, entities, relations, constraints, comparison conditions, conjunctions, disjunctions, temporal conditions, and aggregation requirements.
The `global_best_paths` are structural skeletons. They may help reveal reasoning order, candidate branches, constraint branches, latent intermediate entities, or entity-to-answer paths. Use them as parser-grounded evidence, but never let them override, replace, narrow, or simplify the meaning of the original question.
Atomic question definition:
An atomic question asks for exactly one missing answer using one semantic operation. It should be directly answerable once its dependencies are resolved.
A question is not atomic if it asks a final property, event, time, place, comparison, or attribute while still containing an unresolved nested relation or an unresolved intermediate entity.
If a question contains an unresolved nested relation, split that nested relation into an earlier atomic question.
Latent intermediate entity rule:
If the original question asks about a property, event, time, place, date, comparison, or attribute of an entity that is not explicitly given, but is described through a relation to another entity, first resolve that described entity as an intermediate answer.
Such an entity is a latent intermediate entity.
For a semantic chain:
known entity → latent intermediate entity → final predicate
generate:
1. an atomic question that identifies the latent intermediate entity;
2. a later atomic question that applies the final predicate to `qN's answer`.
Do not collapse these two operations into one question.
The trigger for splitting is functional, not lexical: split when the described entity is used as the argument of a later predicate, event, attribute, comparison, or final answer intent.
Do not over-split when the latent intermediate entity itself is the final answer. Split only when that entity is further used to ask another property, event, time, place, date, comparison, or attribute.
Examples:
Original question:
When did Lothair II's mother die?
Correct decomposition:
{
"atomic_questions": [
{
"id": "q1",
"question": "Who was Lothair II's mother?",
"depends_on": []
},
{
"id": "q2",
"question": "When did q1's answer die?",
"depends_on": ["q1"]
}
]
}
Incorrect decomposition:
{
"atomic_questions": [
{
"id": "q1",
"question": "When did Lothair II's mother die?",
"depends_on": []
}
]
}
The incorrect version is not atomic because it asks the final event while still containing an unresolved intermediate entity.
Original question:
Who was Lothair II's mother?
Correct decomposition:
{
"atomic_questions": [
{
"id": "q1",
"question": "Who was Lothair II's mother?",
"depends_on": []
}
]
}
This does not need further decomposition because the intermediate entity is itself the final answer.
Path-to-DAG interpretation rule:
When a `global_best_path` forms a chain from a known or explicit entity to a final predicate through an intermediate relational node, interpret the path as an operation sequence, not as a sentence to be copied.
For a path pattern like:
known entity ---- intermediate relation/entity ---- final predicate
prefer a DAG pattern like:
1. identify the intermediate relation/entity from the known entity;
2. ask the final predicate about the identified intermediate answer.
However, use the `original_question` to decide whether the intermediate node is truly an entity to be resolved, a constraint, or merely structural noise.
DAG requirements:
1. Generate all and only the atomic questions needed to answer the original question.
2. Preserve every explicit entity, relation, modifier, constraint, temporal condition, ordinal condition, comparison, conjunction, disjunction, and final answer intent from the original question.
3. Use natural-language questions. Do not copy parser labels, token indices, path notation, or symbolic triples into the questions.
4. Every non-initial question that uses a previous result must refer to it naturally as `q1's answer`, `q2's answer`, etc.
5. Every such reference must be reflected in `depends_on`.
6. Do not create isolated lookup questions that do not constrain, identify, compare, or aggregate toward the final answer.
7. The final answer-intent question must be generated only after all required identifying constraints and latent intermediate entities have been resolved or included.
8. If a path omits an important constraint from the original question, include that constraint anyway.
9. If a path contains irrelevant or misleading structure, ignore it.
Constraint closure rule:
Every atomic question except the final one must be consumed by at least one later question, unless the original question itself asks for a list of independent answers.
A subquestion is consumed when its answer is used to identify, constrain, compare, select, aggregate, verify, or ask the final answer. If you generate a lookup question but no later question uses its answer, the decomposition is incomplete or wrong.
Conjunctive constraint rule:
When the original question uses conjunctions such as "and", "both", "as well as", or coordinated phrases to describe the same target, treat the branches as joint constraints on one shared answer target, not as independent questions.
For conjunctive constraints:
1. Resolve each nested constraint if needed.
2. Then generate a later question that combines all constraint answers to identify or ask about the shared target.
3. The later question must mention all required previous answers using `qN's answer`.
4. The later question's `depends_on` must include all consumed constraint questions.
Do not allow one conjunctive branch to determine the final answer while another branch remains an unused leaf.
Parallel branch interpretation:
Multiple paths may represent different semantic structures. Decide their role from the `original_question`, not from path order.
1. If the paths correspond to alternatives, choices, comparisons, or rankings, generate symmetric branch questions and then a final comparison/selection/ranking question.
2. If the paths correspond to multiple constraints on the same target, generate the needed constraint questions and then a final target question that consumes all constraints.
3. If one path gives the answer intent and another path gives an identifying constraint, the final answer-intent question must consume the identifying constraint.
4. If a branch is only structural noise and is not required by the original question, ignore it.
Final answer-intent rule:
The final atomic question must preserve the wh-intent of the original question.
Examples:
* If the original question asks "When was X created?", the final question must ask when the fully identified X was created.
* If the original question asks "Which film ...?", the final question must select or identify the film.
* If the original question asks "What nationality ...?", the final question must ask the nationality of the fully identified person.
* If the original question asks a comparison, the final question must perform the comparison using the branch results.
Do not ask the final answer-intent question before the answer target has been fully identified by all required constraints and latent intermediate entities.
Output format:
Return only valid JSON.
Use exactly this schema:
{
"atomic_questions": [
{
"id": "q1",
"question": "natural-language atomic question?",
"depends_on": []
},
{
"id": "q2",
"question": "natural-language atomic question using q1's answer?",
"depends_on": ["q1"]
}
]
}
Do not output explanations, comments, markdown, evidence, paths, confidence scores, or extra fields.
Quality check before returning:
1. Does every original-question constraint appear in at least one atomic question?
2. Does every latent intermediate entity that is used by a later predicate get resolved before the final predicate is asked?
3. Does every non-final atomic question feed into a later question?
4. Does the final atomic question preserve the original wh-intent?
5. For conjunctions, are all conjunctive constraints consumed by the shared target or final answer question?
6. Are all `qN's answer` references matched by `depends_on`?
7. Are there any isolated leaf questions besides the final answer question? If yes, remove them or connect them correctly.
8. Does any atomic question still contain an unresolved relational description that is being used to ask another property, event, time, place, date, comparison, or attribute? If yes, split it.


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
