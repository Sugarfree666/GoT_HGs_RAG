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
Do not output explanations, plans, semantic paths, reasoning traces, markdown, comments, or warnings.
Return exactly one valid JSON object with one top-level key: "atomic_questions".

You are given:
- original_question: the full natural-language question. This is the semantic authority.
- topic_entities: explicit/topic entities extracted from the original question.
- step4_paths: token-level structural hints produced by Step4.

The original_question decides:
- the final answer intent;
- the final answer type;
- the target variable;
- intermediate variables;
- constraints and modifier attachment;
- comparison, selection, verification, and aggregation semantics;
- whether a final combine/compare/select/verify node is needed.

step4_paths are only structural hints.
They may help reveal candidate branches, bridge variables, operators, modifiers, or noisy parser structure.
They do not determine the DAG.
They do not need to support every DAG node.
DAG nodes do not need path support.
They may be incomplete, noisy, reversed, or contain grammatical tokens.
Never mechanically copy step4_paths.
Never convert adjacent path tokens into relations.
Ignore parser-noise tokens such as get, has, is, are, of, the, a, how, which, long when they are only grammatical or WH markers.

============================================================
Core objective
============================================================

Generate the smallest DAG that is sufficient to answer the original question.

Atomicity means:
- one subquestion asks for one missing entity, attribute, value, set, boolean, or comparison result;
- a subquestion may contain constraints from the original question if those constraints are necessary to identify one variable;
- atomicity does not mean one dependency-path edge per question.

Good atomic question:
- "Which city shares a county with Helvetia?"
This asks for one target city under a necessary constraint.

Bad atomic question:
- "What county does Helvetia share?"
This is underspecified and asks for the wrong intermediate object when the original question asks about a city.

============================================================
Silent decomposition protocol
============================================================

Before producing JSON, silently perform these steps:

1. Identify the requested answer intent.
   Ask: What is the original question ultimately asking for?

2. Identify the requested answer type.
   Examples: person, place, organization, work, date, number, boolean, value, set.

3. Identify the target variable.
   For "Which film/person/city..." questions, the target variable is the film/person/city requested by the WH phrase.

4. Identify topic entities and candidate branches.
   Use topic_entities exactly as written when they appear in subquestions.

5. Interpret step4_paths only as hints.
   Treat path tokens as possible anchors, branches, intermediate variables, modifiers, or operators.
   Do not preserve token order if it conflicts with the original question.

6. Choose the DAG pattern:
   - direct lookup
   - bridge lookup
   - constraint-defined lookup
   - parallel candidate lookup
   - comparison / selection
   - boolean verification
   - aggregation / counting
   - set intersection or filtering

7. Expand hidden operator requirements.
   Comparisons usually require evidence values before comparison.
   Verification usually requires facts before verification.
   Aggregation usually requires a set before aggregation.

8. Write executable subquestions.
   Each subquestion should be natural, retriever-friendly, and answerable independently after replacing qN's answer references with actual answers.

9. Check dependency binding.
   Every dependency must be explicitly visible in the question text using the exact phrase qN's answer.

10. Check final sufficiency.
   The leaf node or leaf nodes must provide enough information to answer the original question.

11. Check coverage of restrictive and conjunctive modifiers.
   If the target entity is described by multiple constraints joined by "and", "as well as", "with", "where", "that", "which", "who", "whose", or appositions, preserve all constraints needed to identify the target.
   Do not silently drop the second half of a conjunction because it is not visible in step4_paths.

12. Check appositive and parenthetical entity identity.
   If a topic entity or original question mention includes a parenthetical, comma title, subtitle, or other disambiguator, keep the complete identifying phrase in the subquestion when that entity is queried.
   If topic_entities split a single appositive mention into pieces, reconstruct the complete mention from original_question when writing the atomic question.

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
5. Do not use vague dependent references such as "the person", "the director", "the city", "that place", "it", or "they" when the intended object is a previous answer.
6. Independent branches must not depend on each other.
7. Each question must end with a question mark.
8. Use exact topic entity surface forms from the input.
9. Do not output ENTITYA, ENTITYB, ENTITYC, or other unresolved placeholders.
10. Do not output fields other than id, question, depends_on, operation, and output_type.
11. If a dependency is listed, the corresponding qN's answer reference must be semantically necessary in the question. Do not add unused depends_on entries.

============================================================
DAG pattern rules
============================================================

1. Direct lookup
Use one lookup node when the original question asks for one fact about a known entity.

Example shape:
q1: What is the [attribute] of [entity]?

2. Bridge lookup
Use a first lookup for the intermediate object, then a second lookup for the requested attribute/value.

Example shape:
q1: Who performed [song]?
q2: What is the nationality of q1's answer?

3. Constraint-defined lookup
If a relative clause identifies the target variable, keep the constraint inside the lookup question.
Do not split off an underspecified intermediate question.

Good:
q1: Which city shares a county with Helvetia?
q2: How long are the city council terms of q1's answer?

Bad:
q1: What county does Helvetia share?
q2: What city is in q1's answer?

4. Parallel candidate branches
For questions comparing two or more explicit candidates, build parallel evidence branches.
Do not make one candidate branch depend on another candidate branch.

5. Comparison / selection
Retrieve the compared values first.
Then add a compare/select node when the original question asks which candidate satisfies the comparison.

Use:
- compare when the output is a comparison judgment or relation.
- select when the output is one of the original candidates or a requested entity type.

6. Boolean verification
Retrieve the needed facts first when they are not directly given.
Then add a verify node if the original question is yes/no.

7. Aggregation
Retrieve the set first.
Then count, sum, min, max, list, or otherwise aggregate.

============================================================
Constraint coverage rules
============================================================

Many errors come from dropping restrictive modifiers. Avoid that by treating constraints as part of the target variable whenever they identify the answer.

Relative clauses:
- "the city where X happened" means the target is the city satisfying that clause.
- "the person who X" means the target is the person satisfying that clause.
- "the country where X originated" means the target is the country satisfying that clause.

Conjunctive descriptions:
- If the original question says "the region immediately north of A and the location/site of B", the target region must satisfy both constraints, or the DAG must resolve enough facts to identify that joint target.
- Do not answer only the "north of A" part and ignore "location/site of B".

By/of/from attachments:
- A phrase like "Turn Me On by the singer of Come Away with Me" means the lookup about "Turn Me On" is constrained by q1's answer.
- A dependent question must explicitly include "by q1's answer", "of q1's answer", "from q1's answer", or another exact qN's answer binding when that previous answer is the modifier.

Appositions:
- "John Ernest, Duke Of Saxe-Eisenach" identifies one person mention.
- "Christopher Newton (Criminal)" identifies one person mention.
- Do not shorten these to "John Ernest" or "Christopher Newton" when asking about that entity.

Possessive-WH:
- For "Whose sister played X?", the final answer is the possessor, not the sister.
- First identify who played X, then ask whose sister that person is.

Do not over-decompose constraints when a single constrained lookup is clearer and still asks for one target variable.

============================================================
Operator expansion rules
============================================================

younger / older:
- If comparing people, retrieve birth dates or ages before selecting.
- The final answer type must remain the requested type.
- For "Which film has the younger director?", the final selected answer is a film, not a director.

born earlier / born later:
- Retrieve birth dates for the relevant people.

earlier / later for events or releases:
- Retrieve dates or release dates.

larger / smaller / more / fewer / most / fewest:
- Retrieve numeric values for each candidate or branch.

longer / shorter / how long:
- Retrieve duration or length values.

lived longer:
- Retrieve enough evidence to compute each lifespan.
- Usually retrieve birth date and death date for each person, or directly retrieve lifespan duration for each person if that is a natural retrievable fact.
- Do not compare people using only birth dates.

same nationality / same country / same birthplace:
- Retrieve the relevant attribute for both branches, then verify or compare equality.

goals / points / population / area / distance / age:
- Ask for the numeric value explicitly.
- Prefer "How many..." for counts.
- Avoid vague questions like "What are Messi's goals?" when the original asks for a number.

which X:
- The final select node must return X, not merely the intermediate evidence object.

============================================================
Path-hint usage rules
============================================================

Use step4_paths to help notice:
- explicit candidate branches;
- bridge nouns such as director, author, performer, composer, birthplace;
- target attributes such as nationality, population, birth date, release date;
- operators such as younger, larger, longer, same, compare, first, most;
- constraint cues such as shares, located, part, member, spouse, parent.

But:
- do not ask about every token in the path;
- do not create questions for grammatical tokens;
- do not follow the path direction if the original question implies a different semantic direction;
- do not use path tokens as relation labels;
- do not force a node to exist only because a token appears in the path.

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

Example 2: Split mixed path into independent evidence branches

Input:
{
  "original_question": "How many goals did the person Barcelona signed score compared with Messi?",
  "topic_entities": ["Barcelona", "Messi"],
  "step4_paths": [
    ["Barcelona", "signed", "get", "person", "compared", "goals", "Messi"]
  ]
}

Avoid:
- making Messi's goals depend on the person signed by Barcelona;
- asking "What are Messi's goals?" when the requested evidence is a number;
- losing the goals of the person signed by Barcelona.

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Who did Barcelona sign?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    },
    {
      "id": "q2",
      "question": "How many goals did q1's answer score?",
      "depends_on": ["q1"],
      "operation": "lookup",
      "output_type": "number"
    },
    {
      "id": "q3",
      "question": "How many goals did Messi score?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "number"
    },
    {
      "id": "q4",
      "question": "Based on q2's answer and q3's answer, how do the goals scored by the person Barcelona signed compare with Messi's goals?",
      "depends_on": ["q2", "q3"],
      "operation": "compare",
      "output_type": "value"
    }
  ]
}

Example 3: Relative clause as target constraint

Input:
{
  "original_question": "How long are the city council terms for the city that shares a county with Helvetia?",
  "topic_entities": ["Helvetia"],
  "step4_paths": [
    ["Helvetia", "shares", "county", "city", "council", "terms", "long", "How"]
  ]
}

Avoid:
- asking "What county does Helvetia share?";
- treating "county" as the requested answer;
- splitting the relative clause away from the target city.

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Which city shares a county with Helvetia?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "place"
    },
    {
      "id": "q2",
      "question": "How long are the city council terms of q1's answer?",
      "depends_on": ["q1"],
      "operation": "lookup",
      "output_type": "value"
    }
  ]
}

Example 4: Candidate comparison with hidden scalar evidence

Input:
{
  "original_question": "Which film has the younger director, Dangerously They Live or Salad By The Roots?",
  "topic_entities": ["Dangerously They Live", "Salad By The Roots"],
  "step4_paths": [
    ["Dangerously They Live", "film", "director", "younger"],
    ["Salad By The Roots", "film", "director", "younger"]
  ]
}

Avoid:
- selecting the younger director instead of the film;
- comparing directors without retrieving birth dates or ages;
- collapsing both candidates into one unresolved question.

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

Example 5: Boolean verification

Input:
{
  "original_question": "Did the director of The Matrix also direct Cloud Atlas?",
  "topic_entities": ["The Matrix", "Cloud Atlas"],
  "step4_paths": [
    ["The Matrix", "director", "also", "direct", "Cloud Atlas"]
  ]
}

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Who directed The Matrix?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    },
    {
      "id": "q2",
      "question": "Who directed Cloud Atlas?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    },
    {
      "id": "q3",
      "question": "Based on q1's answer and q2's answer, did the director of The Matrix also direct Cloud Atlas?",
      "depends_on": ["q1", "q2"],
      "operation": "verify",
      "output_type": "boolean"
    }
  ]
}

Example 6: Attribute equality

Input:
{
  "original_question": "Do the authors of The Hobbit and Dune have the same nationality?",
  "topic_entities": ["The Hobbit", "Dune"],
  "step4_paths": [
    ["The Hobbit", "authors", "same", "nationality", "Dune"]
  ]
}

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Who is the author of The Hobbit?",
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
    },
    {
      "id": "q3",
      "question": "Who is the author of Dune?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    },
    {
      "id": "q4",
      "question": "What is the nationality of q3's answer?",
      "depends_on": ["q3"],
      "operation": "lookup",
      "output_type": "value"
    },
    {
      "id": "q5",
      "question": "Based on q2's answer and q4's answer, do the authors of The Hobbit and Dune have the same nationality?",
      "depends_on": ["q2", "q4"],
      "operation": "verify",
      "output_type": "boolean"
    }
  ]
}

Example 7: Possessive-WH direction

Input:
{
  "original_question": "Whose sister played Susie in Miracle on 34th Street?",
  "topic_entities": ["Susie", "Miracle on 34th Street"],
  "step4_paths": [
    ["Whose", "sister", "played", "Susie"]
  ]
}

Avoid:
- "Who is the sister of q1's answer?"
- This reverses the question and returns the sister instead of the possessor.

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

Example 8: Dependency binding through a modifier

Input:
{
  "original_question": "Who wrote Turn Me On by the singer of Come Away with Me?",
  "topic_entities": ["Turn Me On", "Come Away with Me"],
  "step4_paths": [
    ["Come Away with Me", "singer", "Turn Me On", "wrote", "Who"]
  ]
}

Avoid:
- q2: "Who wrote Turn Me On?" with depends_on ["q1"].
- The question text does not bind q1's answer, so the dependency is semantically unused.

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

Example 9: Lifespan comparison

Input:
{
  "original_question": "Who lived longer, Ludwig Elsbett or Pamela Ann Rymer?",
  "topic_entities": ["Ludwig Elsbett", "Pamela Ann Rymer"],
  "step4_paths": [
    ["Ludwig Elsbett", "lived", "longer", "Pamela Ann Rymer"]
  ]
}

Avoid:
- comparing only birth dates.

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "When was Ludwig Elsbett born?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "date"
    },
    {
      "id": "q2",
      "question": "When did Ludwig Elsbett die?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "date"
    },
    {
      "id": "q3",
      "question": "When was Pamela Ann Rymer born?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "date"
    },
    {
      "id": "q4",
      "question": "When did Pamela Ann Rymer die?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "date"
    },
    {
      "id": "q5",
      "question": "Based on q1's answer, q2's answer, q3's answer, and q4's answer, who lived longer: Ludwig Elsbett or Pamela Ann Rymer?",
      "depends_on": ["q1", "q2", "q3", "q4"],
      "operation": "compare",
      "output_type": "person"
    }
  ]
}

Example 10: Preserve appositive identity

Input:
{
  "original_question": "Who is the father-in-law of John Ernest, Duke Of Saxe-Eisenach?",
  "topic_entities": ["John Ernest", "Duke Of Saxe-Eisenach"],
  "step4_paths": [
    ["John Ernest"]
  ]
}

Output:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Who is the father-in-law of John Ernest, Duke Of Saxe-Eisenach?",
      "depends_on": [],
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
3. The final leaf preserves the original answer intent.
4. The final leaf preserves the requested answer type.
5. step4_paths were used only as hints, not copied mechanically.
6. No semantic paths or support fields are present.
7. No unresolved ENTITY placeholders are present.
8. Each dependent question explicitly mentions qN's answer.
9. Independent candidate branches do not depend on each other.
10. Comparison/select/verify nodes depend on evidence values, not only unresolved intermediate entities.
11. Relative clauses are attached to the correct target variable.
12. No underspecified question is present.
13. All topic entities are copied exactly when used.
14. Parenthetical and appositive entity disambiguators from original_question are preserved.
15. Restrictive conjuncts and relative clauses needed to identify the target are preserved.
16. The returned text is valid JSON only.

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
