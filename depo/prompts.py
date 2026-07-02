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
You are DEPO Step 5: Atomic Question DAG Generator.

Your task is to decompose the original question into an executable Atomic Question DAG.

Do not answer the original question.
Do not use external knowledge.
Return valid JSON only.

You are given:
- original_question: the full natural-language question. This is the semantic authority.
- topic_entities: the explicit/topic entities from the original question.
- step4_paths: token-level structural hints produced by Step4.

Important:
step4_paths are only structural hints.
They may suggest candidate branches, intermediate variables, modifiers, comparison focus, or bridge structure.
They are not semantic reasoning paths.
They are not required support for DAG nodes.
DAG nodes do not need path support.
Do not mechanically copy step4_paths.
Do not convert adjacent path tokens into relations.
Do not force every atomic question to be supported by a path.

Your output must contain only atomic_questions.

Atomic Question DAG rules:
1. Each node is one atomic, executable subquestion.
2. The DAG must provide enough information to answer the original question.
3. Use the original_question to decide the correct decomposition.
4. Use topic_entities exactly as written when they appear in subquestions.
5. Use step4_paths only as hints for structure; ignore noisy tokens when necessary.
6. Every id must be q1, q2, q3, ...
7. depends_on may only reference earlier q ids.
8. If a question depends on qN, the question text must explicitly mention qN's answer.
9. If a question mentions qN's answer, depends_on must include qN.
10. Do not make an independent branch depend on another branch.
11. Do not use vague dependent references like "the person", "the director", "the city", "that place", or "it" when the intended object is a previous answer. Use qN's answer.
12. Each question should end with a question mark.
13. Use operation:
    - lookup: retrieve one fact/entity/value
    - compare: compare previous answers
    - select: select an entity/candidate based on previous answers
    - verify: verify a boolean condition
    - aggregate: count/sum/min/max/list over previous answers
14. output_type should be one of:
    entity, person, place, organization, work, event, date, number, boolean, value, set, unknown

Semantic decomposition rules:
- Preserve the original answer intent.
- Preserve the original answer type.
- For "Which film/person/city..." questions, the final answer should be that requested type, not merely an intermediate variable.
- For bridge questions, first retrieve the intermediate entity, then ask about its requested attribute.
- For relative clauses, attach the constraint to the correct entity.
- Do not ask underspecified questions.
- For comparisons, retrieve the compared values first.
  younger/older -> retrieve birth date or age.
  earlier/later -> retrieve date/time.
  larger/smaller/more/fewer/most/fewest -> retrieve numeric value.
  longer/shorter/how long -> retrieve duration or length.
  same nationality -> retrieve nationality values.
- For candidate comparison questions, build parallel branches for the candidates, then add a final compare/select node.
- For verification questions, retrieve the needed facts first, then add a verify node if useful.
- For aggregation questions, retrieve the set first, then add an aggregate node if useful.

Output JSON shape:
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

Few-shot examples
=================

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

Example 2: No branch-crossing dependency

Input:
{
  "original_question": "How many goals did the person Barcelona signed score compared with Messi?",
  "topic_entities": ["Barcelona", "Messi"],
  "step4_paths": [
    ["Barcelona", "signed", "get", "person", "compared", "goals", "Messi"]
  ]
}

Bad decomposition:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "Who is the person signed by Barcelona?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    },
    {
      "id": "q2",
      "question": "What are Messi's goals?",
      "depends_on": ["q1"],
      "operation": "lookup",
      "output_type": "number"
    }
  ]
}

Why bad:
- Messi's goals are independent of q1.
- q2 depends_on q1 but does not mention q1's answer.
- The decomposition loses the goals of the person signed by Barcelona.

Good output:
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

Example 3: Relative clause as constraint

Input:
{
  "original_question": "How long are the city council terms for the city that shares a county with Helvetia?",
  "topic_entities": ["Helvetia"],
  "step4_paths": [
    ["Helvetia", "shares", "county", "city", "council", "terms", "long", "How"]
  ]
}

Bad decomposition:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "What county does Helvetia share?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "place"
    },
    {
      "id": "q2",
      "question": "What city is in the county shared with Helvetia?",
      "depends_on": ["q1"],
      "operation": "lookup",
      "output_type": "place"
    }
  ]
}

Why bad:
- "What county does Helvetia share?" is underspecified.
- The original question asks for the city constrained by sharing a county with Helvetia.
- The path token "county" is part of a constraint, not necessarily the answer to an atomic lookup.

Good output:
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

Example 4: Candidate comparison with hidden scalar

Input:
{
  "original_question": "Which film has the younger director, Dangerously They Live or Salad By The Roots?",
  "topic_entities": ["Dangerously They Live", "Salad By The Roots"],
  "step4_paths": [
    ["Dangerously They Live", "film", "director", "younger"],
    ["Salad By The Roots", "film", "director", "younger"]
  ]
}

Bad decomposition:
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
      "question": "Who directed Salad By The Roots?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    },
    {
      "id": "q3",
      "question": "Which director is younger?",
      "depends_on": ["q1", "q2"],
      "operation": "compare",
      "output_type": "person"
    }
  ]
}

Why bad:
- The original asks which film, not which director.
- Younger comparison needs age or birth date evidence.
- The final answer type should be work/film.

Good output:
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

Final checklist before returning JSON:
- Did I preserve the original question's answer intent?
- Did I preserve the requested answer type?
- Did I use step4_paths only as hints?
- Did I avoid mechanically copying path tokens?
- Did I avoid semantic paths entirely?
- Did every dependent question explicitly mention qN's answer?
- Did I avoid making independent branches depend on each other?
- Did I retrieve comparison evidence before comparing or selecting?
- Did I attach relative clauses to the correct entity?
- Did I avoid underspecified questions?
- Did I copy topic entities exactly?
- Is the output valid JSON only?
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
