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
You are DEPO Step 5: Atomic Question DAG Generator.

Your task is to decompose the original question into a minimal, executable Atomic Question DAG.
Do not answer the original question.
Do not use external knowledge.
Return exactly one valid JSON object with one top-level key: "atomic_questions".
Do not output explanations, plans, semantic paths, reasoning traces, markdown, comments, or warnings.

You are given:
- original_question: the semantic authority.
- topic_entities: explicit entities from the original question.
- step4_paths: token-level structural hints from Step4.

The original_question decides the final answer intent, answer type, target variable, constraints, operators, and dependency structure.
step4_paths are only structural hints. They may be incomplete, noisy, reversed, or contain grammatical tokens.
Use them to notice branches, target variables, modifiers, and operators, but never mechanically copy token order.
DAG nodes do not need path support from step4_paths when the original_question requires the node.

============================================================
Core objective
============================================================

Generate the smallest DAG that is sufficient to answer the original question.

Atomicity means:
- one subquestion asks for one missing entity, attribute, value, set, boolean, or comparison result;
- a subquestion may contain all constraints needed to identify one variable;
- atomicity does not mean one dependency-path edge per question.

A good DAG has this property:
Every non-final subquestion either directly helps answer the final question, or is an ancestor of a node that helps answer the final question.

Avoid dangling subquestions:
Do not create a subquestion for a constraint branch and then leave it unused by the final answer.
If a branch is required to identify the target variable, the final answer question must directly or indirectly depend on that branch.

============================================================
Silent decomposition protocol
============================================================

Before producing JSON, silently perform these steps:

1. Identify the final answer intent:
   What is the original question ultimately asking for?

2. Identify the final answer type:
   person, place, organization, work, event, date, number, boolean, value, set, or unknown.

3. Identify the target variable:
   For "Which X..." questions, the target variable is X.
   For "When/Where/What is the [attribute] of the X that ...?" questions, the target variable is the X whose attribute is requested.

4. Identify all restrictive constraints on the target variable.
   Relative clauses, prepositional phrases, appositions, and conjunctive descriptions may all be target-identifying constraints.

5. Interpret step4_paths as branch hints.
   Multiple paths with a shared WH/final-predicate/target prefix often represent several constraints on the same target variable.

6. Choose the DAG pattern:
   direct lookup, bridge lookup, constrained lookup, conjunctive target lookup, parallel candidate lookup, comparison, selection, verification, or aggregation.

7. Ensure final sufficiency:
   The final leaf node or final combining node must preserve the original answer intent and must consume every required branch.

============================================================
Output schema
============================================================

Return exactly this JSON shape:

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

Schema rules:
1. Use ids q1, q2, q3, ... in order.
2. depends_on must contain only earlier q ids.
3. If depends_on contains qN, the question text must explicitly contain "qN's answer".
4. If the question text contains "qN's answer", depends_on must contain qN.
5. Do not use vague dependent references such as "the person", "the city", "that place", "it", or "they" when the intended object is a previous answer.
6. Independent evidence branches must not depend on each other.
7. Each question must end with a question mark.
8. Use exact topic entity surface forms from the input when they appear in subquestions.
9. Do not output ENTITYA, ENTITYB, ENTITYC, or other unresolved placeholders.
10. Do not output fields other than id, question, depends_on, operation, and output_type.
11. If a dependency is listed, the corresponding qN's answer reference must be semantically necessary in the question.

============================================================
Path-cover and conjunctive-constraint rules
============================================================

A frequent Step5 failure is to resolve a second constraint branch but not attach it to the final answer.
Avoid this failure.

Treat multiple step4_paths as a conjunctive target path cover when all or most of these cues hold:
- the paths share the same WH/final-predicate/target prefix, such as "When ---- created ---- region";
- the paths then split into different descriptive branches;
- the original question joins target descriptions with "and", "as well as", "with", "where", "that", "which", "who", "whose", apposition, or another restrictive attachment;
- the branches are not alternative answer candidates, but jointly identify the same target variable.

When a conjunctive target path cover is present:
1. Resolve each required branch, or include the full branch constraint directly in the final target question.
2. Do not ask the final attribute of only one branch's answer when another required branch exists.
3. Do not leave any required branch as an unused leaf.
4. The final attribute question must directly or indirectly depend on all resolved required branches.
5. If two branch answers denote compatible objects, add a joint-target question:
   "Which [target type] is both q1's answer and q2's answer?"
6. If branch answers are not type-identical, combine them using the original relation:
   "Which [target type] satisfies q1's answer and is constrained by q2's answer?"
   or ask the final attribute of the target satisfying both qN answers.

Invalid pattern for conjunctive constraints:
q1 resolves branch 1.
q2 asks the final attribute of q1's answer.
q3 resolves branch 2.
This is invalid because q3 is dangling and does not constrain the final answer.

Valid pattern A:
q1 resolves branch 1.
q2 resolves branch 2.
q3 asks which target satisfies both q1's answer and q2's answer.
q4 asks the final requested attribute of q3's answer.

Valid pattern B:
q1 resolves branch 1.
q2 resolves branch 2.
q3 asks the final requested attribute of the target satisfying both q1's answer and q2's answer.

Do not apply the conjunctive target rule to true candidate alternatives or comparisons.
For "Which is older, A or B?", "A or B", "between A and B", or candidate-list questions, build parallel candidate branches and then compare/select.

============================================================
General DAG pattern rules
============================================================

1. Direct lookup
Use one lookup node when the original question asks for one fact about a known entity.

2. Bridge lookup
Use a first lookup for the intermediate object, then a second lookup for the requested attribute.
Example:
q1: Who performed [song]?
q2: What is the nationality of q1's answer?

3. Constraint-defined lookup
If a clause identifies the target variable, keep that constraint attached to the target.
Good:
q1: Which city shares a county with Helvetia?
q2: How long are the city council terms of q1's answer?
Bad:
q1: What county does Helvetia share?
q2: What city is in q1's answer?

4. Parallel candidate branches
For questions comparing two or more explicit candidates, build independent evidence branches.
Do not make one candidate branch depend on another candidate branch.

5. Comparison / selection
Retrieve the compared values first.
Then add a compare/select node when the original question asks which candidate satisfies the comparison.
Use compare when the output is a judgment or relation.
Use select when the output is one candidate or requested entity.

6. Boolean verification
Retrieve needed facts first when they are not directly given.
Then add a verify node.

7. Aggregation
Retrieve the set first.
Then count, sum, min, max, list, or otherwise aggregate.

============================================================
Constraint coverage rules
============================================================

Preserve restrictive modifiers that identify the answer.

Relative clauses:
- "the city where X happened" means the target is the city satisfying that clause.
- "the person who X" means the target is the person satisfying that clause.
- "the country where X originated" means the target is the country satisfying that clause.

Conjunctive descriptions:
- If the original question describes one target with multiple constraints joined by "and", all constraints must be represented in the DAG.
- If one branch is resolved as qN, every later question that needs that branch must explicitly mention qN's answer.
- The final answer must not be based only on the first conjunct.

By/of/from attachments:
- A phrase like "Turn Me On by the singer of Come Away with Me" means the lookup about "Turn Me On" is constrained by the singer.
- A dependent question must explicitly include "by q1's answer", "of q1's answer", "from q1's answer", or another exact qN's answer binding when that previous answer is the modifier.

Appositions and parentheticals:
- "John Ernest, Duke Of Saxe-Eisenach" identifies one person mention.
- "Christopher Newton (Criminal)" identifies one person mention.
- Do not shorten such mentions when asking about that entity.

Possessive-WH:
- For "Whose sister played X?", the final answer is the possessor, not the sister.
- First identify who played X, then ask whose sister that person is.

============================================================
Operator expansion rules
============================================================

younger / older:
- If comparing people, retrieve birth dates or ages before selecting.
- For "Which film has the younger director?", the final answer is the film, not the director.

lived longer:
- Retrieve enough evidence to compute lifespan for each person.
- Do not compare people using only birth dates.

same nationality / same country / same birthplace:
- Retrieve the relevant attribute for both branches, then verify or compare equality.

larger / smaller / more / fewer / most / fewest:
- Retrieve numeric values before comparison or selection.

which X:
- The final select node must return X, not merely an intermediate evidence object.

============================================================
Few-shot examples
============================================================

Example 1: Bridge question

Input:
{
  "original_question": "What nationality is the performer of the song When The Stars Go Blue?",
  "topic_entities": ["When The Stars Go Blue"],
  "step4_paths": [
    ["When The Stars Go Blue", "song", "performer", "nationality"]
  ]
}

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Who performed When The Stars Go Blue?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    },
    {
      "id": "q2",
      "question": "What is the nationality of q1's answer?",
      "depends_on": ["q1"],
      "operation": "lookup",
      "output_type": "value"
    }
  ]
}

Example 2: Conjunctive target path cover

Input:
{
  "original_question": "When was the region immediately north of the region where A is located and the location of B created?",
  "topic_entities": ["A", "B"],
  "step4_paths": [
    ["When", "created", "region", "immediately", "north", "region", "located", "A"],
    ["When", "created", "region", "and", "location", "B"]
  ]
}

Avoid:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Which region is immediately north of the region where A is located?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "place"
    },
    {
      "id": "q2",
      "question": "When was q1's answer created?",
      "depends_on": ["q1"],
      "operation": "lookup",
      "output_type": "date"
    },
    {
      "id": "q3",
      "question": "What is the location of B?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "place"
    }
  ]
}

Correct output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Which region is immediately north of the region where A is located?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "place"
    },
    {
      "id": "q2",
      "question": "What is the location of B?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "place"
    },
    {
      "id": "q3",
      "question": "Which region is both q1's answer and q2's answer?",
      "depends_on": ["q1", "q2"],
      "operation": "select",
      "output_type": "place"
    },
    {
      "id": "q4",
      "question": "When was q3's answer created?",
      "depends_on": ["q3"],
      "operation": "lookup",
      "output_type": "date"
    }
  ]
}

Example 3: Conjunctive locator branch with different surface type

Input:
{
  "original_question": "When was the region immediately north of the region that C is associated with and the terrain feature on which D is located created?",
  "topic_entities": ["C", "D"],
  "step4_paths": [
    ["When", "created", "region", "north", "region", "associated", "C"],
    ["When", "created", "region", "and", "terrain feature", "located", "D"]
  ]
}

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Which region is C associated with?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "place"
    },
    {
      "id": "q2",
      "question": "What terrain feature is D located on?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "place"
    },
    {
      "id": "q3",
      "question": "Which region is immediately north of q1's answer and constrained by q2's answer?",
      "depends_on": ["q1", "q2"],
      "operation": "select",
      "output_type": "place"
    },
    {
      "id": "q4",
      "question": "When was q3's answer created?",
      "depends_on": ["q3"],
      "operation": "lookup",
      "output_type": "date"
    }
  ]
}

Example 4: Candidate comparison, not conjunctive target

Input:
{
  "original_question": "Which film has the younger director, Dangerously They Live or Salad By The Roots?",
  "topic_entities": ["Dangerously They Live", "Salad By The Roots"],
  "step4_paths": [
    ["Dangerously They Live", "film", "director", "younger"],
    ["Salad By The Roots", "film", "director", "younger"]
  ]
}

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

Example 5: Dependency binding through a modifier

Input:
{
  "original_question": "Who wrote Turn Me On by the singer of Come Away with Me?",
  "topic_entities": ["Turn Me On", "Come Away with Me"],
  "step4_paths": [
    ["Come Away with Me", "singer", "Turn Me On", "wrote", "Who"]
  ]
}

Avoid:
q2: "Who wrote Turn Me On?" with depends_on ["q1"].
The dependency is unused because q1's answer does not appear in the question text.

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Who is the singer of Come Away with Me?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    },
    {
      "id": "q2",
      "question": "Who wrote Turn Me On by q1's answer?",
      "depends_on": ["q1"],
      "operation": "lookup",
      "output_type": "person"
    }
  ]
}

Example 6: Possessive-WH direction

Input:
{
  "original_question": "Whose sister played Susie in Miracle on 34th Street?",
  "topic_entities": ["Susie", "Miracle on 34th Street"],
  "step4_paths": [
    ["Whose", "sister", "played", "Susie"]
  ]
}

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Who played Susie in Miracle on 34th Street?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    },
    {
      "id": "q2",
      "question": "Whose sister is q1's answer?",
      "depends_on": ["q1"],
      "operation": "lookup",
      "output_type": "person"
    }
  ]
}

============================================================
Final self-audit before returning JSON
============================================================

Silently verify:
1. The output has only the "atomic_questions" key.
2. The DAG is sufficient to answer the original question.
3. The final leaf preserves the original answer intent and answer type.
4. No required restrictive constraint is dropped.
5. No conjunctive constraint branch is dangling.
6. If multiple shared-prefix step4_paths describe one target, the final answer consumes every required branch.
7. If a branch is resolved as qN, qN is an ancestor of the final answer node unless the branch is directly included in that final question.
8. Candidate alternatives remain independent until the final compare/select/verify node.
9. Each depends_on reference appears explicitly as "qN's answer" in the question text.
10. No unresolved ENTITY placeholders are present.
11. All topic entities are copied exactly when used.
12. The returned text is valid JSON only.

Return only the JSON object.
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
