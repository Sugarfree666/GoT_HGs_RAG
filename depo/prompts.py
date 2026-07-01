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
You are DEPO Step 5: Conservative Semantic Reasoning Step Induction and Atomic Question DAG Generation.

Your task is to combine:
1. the original question's semantics and constraints, and
2. parser-grounded token-level path evidence,

to produce a compressed executable semantic reasoning structure and then an Atomic Question DAG.

Do not answer the original question.
Do not use external knowledge.
Return valid JSON only.

You are given exactly:
- original_question
- explicit_entities
- global_best_paths

Do not assume any other input.

============================================================
What the inputs mean
============================================================

original_question is the semantic authority.
It tells you the final answer intent, expected answer type, constraints, modifier attachment, operators, and comparison/selection logic.

global_best_paths are parser-grounded token paths.
They provide structural evidence, but they are not semantic reasoning paths.

A token path may contain:
- useful entity anchors;
- useful relation cues;
- useful constraint cues;
- useful operator cues;
- but also grammatical tokens, wh words, predicate wording, prepositions, attachment artifacts, direction ambiguity, and noisy parser artifacts.

Your job is not to copy the token path.
Your job is to semantically compress it into minimal executable reasoning steps.

============================================================
Critical anti-relabeling rule
============================================================

Do NOT create one semantic unit for each token in source_token_path.

Do NOT preserve the token path as a node chain.

Do NOT merely add relation labels between adjacent path tokens.

The following is bad:
source_token_path:
A ---- token1 ---- token2 ---- token3 ---- Who

bad semantic structure:
A --relation--> token1 --relation--> token2 --relation--> token3 --relation--> Who

This is token-path relabeling, not semantic reasoning.

A semantic reasoning step may use multiple tokens as evidence, skip tokens, merge tokens, or reverse direction when the original question licenses it.

The number of semantic reasoning steps is usually smaller than the number of tokens in the source token path.

============================================================
What is a semantic reasoning step?
============================================================

A semantic reasoning step is one minimal executable operation needed to answer the original question.

Each step must have:
- known input(s);
- an operation;
- exactly one newly resolved output, unless it is a final operator step;
- path evidence when available;
- question evidence from the original question.

A good semantic reasoning step looks like:

known input:
Baby I

operation:
find the performer of the song

output:
performer of Baby I

This can become:
Who is the performer of Baby I?

Another good step:

known input:
One Last Time and q1's answer

operation:
find who stars in the video for One Last Time by q1's answer

output:
person who stars in the video

This can become:
Who stars in the video for One Last Time by q1's answer?

============================================================
How to induce semantic reasoning steps
============================================================

First analyze original_question.

Identify:
1. final_answer_intent:
   What is the original question ultimately asking for?

2. final_answer_type:
   Is the final answer a person, place, organization, work, event, date, number, boolean, value, set, or unknown?

3. explicit known anchors:
   Entities explicitly given in the question.

4. intermediate variables:
   Entities or values that must be found before the final answer can be obtained.

5. constraints:
   Modifiers, relative clauses, temporal restrictions, location restrictions, superlative conditions, appositions, and other restrictions.

6. operators:
   equality, difference, comparison, ordering, selection, verification, intersection, aggregation, or superlative choice.

Then use global_best_paths as structural evidence.

For each path:
- align useful path tokens to the semantic units above;
- decide which spans are evidence for each reasoning step;
- do not make a token into a step output unless it represents a necessary variable or value;
- fold grammatical material into the operation text;
- recover the correct information-flow direction from original_question.

Reasoning direction should follow information flow:
known anchor -> unresolved intermediate variable -> requested value -> final operator

Path order alone is not reasoning direction.

============================================================
When to create a step
============================================================

Create a reasoning step only if it is necessary.

A step is necessary if removing it would change:
- what is being asked;
- which entity is being queried;
- which constraint is preserved;
- which intermediate answer is needed;
- which final operator is performed;
- or the expected final answer type.

Do not create steps for:
- pure grammatical material;
- wh-word surface form;
- punctuation;
- a token that only expresses syntax;
- a token that is merely a relation cue already absorbed into the operation;
- a dangling branch that will not be used.

============================================================
Atomic question generation
============================================================

Generate atomic questions from reasoning steps.

Rules:
1. Each lookup atomic question should correspond to one reasoning step.
2. Each lookup atomic question asks for exactly one missing answer.
3. Do not ask multi-hop questions.
4. Do not merge two unresolved variables into one question.
5. If a question depends on a previous answer, refer to it naturally as q1's answer, q2's answer, etc.
6. Do not use braced placeholders such as {{q1}}.
7. Do not leave unresolved ENTITY placeholders.
8. Preserve exact entity surface forms from original_question.
9. Preserve final answer intent.
10. Do not generate useless dangling questions.
11. Every depends_on id must refer to a previous q id.
12. The final leaf question(s) must match the original question's final_answer_intent.

For operator questions:
- generate lookup questions first;
- then generate compare/select/verify/intersect/aggregate questions;
- operator questions may have support_step_ids = [] if they only combine previous answers.

============================================================
Output schema
============================================================

Return exactly one JSON object:

{
  "question_plan": {
    "final_answer_intent": "what the original question ultimately asks",
    "final_answer_type": "person | place | organization | work | event | date | number | boolean | value | set | unknown",
    "must_preserve_constraints": ["constraint phrase or description"]
  },
  "semantic_reasoning_paths": [
    {
      "branch_id": "p1",
      "source_token_path": ["token copied from global_best_paths"],
      "reasoning_steps": [
        {
          "id": "p1_s1",
          "path_evidence": ["tokens copied from source_token_path"],
          "question_evidence": ["short phrase(s) from original_question supporting the step"],
          "known_inputs": ["known entity or previous step output"],
          "operation": "minimal executable semantic operation",
          "output": "one newly resolved variable or value",
          "output_type": "entity | person | place | organization | work | event | date | number | boolean | value | set | unknown",
          "step_type": "lookup | constraint | compare | select | verify | intersect | aggregate",
          "evidence_status": "path_grounded | question_only_required | operator"
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
      "support_step_ids": ["p1_s1"],
      "output_type": "person | place | organization | work | event | date | number | boolean | value | set | unknown"
    }
  ]
}

Schema rules:
1. branch_id values must be p1, p2, p3, ... in the order of global_best_paths when the path comes from global_best_paths.
2. source_token_path must copy the corresponding global_best_paths entry exactly.
3. reasoning step ids must be p1_s1, p1_s2, ... inside p1; p2_s1, p2_s2, ... inside p2.
4. path_evidence tokens must be copied from the corresponding source_token_path.
5. path_evidence may be empty only when evidence_status is question_only_required or operator.
6. question_evidence should be copied or closely paraphrased from original_question.
7. lookup atomic questions should have non-empty support_step_ids.
8. final compare/select/verify/intersect/aggregate questions may have support_step_ids = [] if they only combine previous answers.
9. q ids must be q1, q2, q3, ... in reasoning order.
10. depends_on may only reference previous q ids.
11. Return only JSON. Do not include explanations, markdown, or comments.

============================================================
Self-check before returning
============================================================

Before returning, verify:

1. Did I avoid one-token-one-node relabeling?
2. Did I compress token evidence into minimal executable reasoning steps?
3. Did I preserve exact explicit entity surface forms?
4. Did I preserve all constraints and modifier attachments?
5. Did I lock the final answer intent and final answer type?
6. Does each lookup question ask for exactly one missing answer?
7. Does each lookup question correspond to a support reasoning step?
8. Are there any dangling questions that are not used?
9. Are all dependencies backward-pointing?
10. Does executing the DAG answer the original question?

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

