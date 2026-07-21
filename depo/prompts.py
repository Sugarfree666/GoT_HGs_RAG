from __future__ import annotations

import json


NESTED_OF_NORMALIZATION_POLICY = """
Normalization is deliberately narrow.
normalized_question must usually equal the original question.

Use only one of the following strictly equivalent rewrite patterns.

1. Nested nominal "of" relation chain
Rewrite only when all conditions hold:
1. Outside explicit entity names, the question contains two or more mutually nested nominal "of" relations.
2. Those "of" relations express the query relation chain, not merely an entity name or a fixed phrase.
3. The rewrite is strictly equivalent and makes the implicit predicate-argument chain explicit for HanLP PAS.

Target rewrites:
- Convert common role nouns in a nested "of" chain into explicit predicates:
  "the director of X" -> "the person who directed X"
  "the performer of song X" -> "the person who performed the song X"
  "the composer of work X" -> "the person who composed the work X"
  "the author/writer of work X" -> "the person who wrote the work X"
- Convert birth-location attributes over an unnamed person into a direct birth question:
  "the place of birth of the performer of song X" -> "Where was the person who performed the song X born?"
  Preserve "country/city/place" answer type when it is explicitly requested.

2. Implicit attribute ownership
Rewrite a question only when all conditions hold:
1. It asks for a nominalized attribute of a target person, object, role, or nested relation result.
2. The attribute and target are connected only by a copula or compressed interrogative order, rather than an explicit predicate, preposition, or possessive construction that already expresses the queried attribute ownership. The target itself may contain a possessive role or relation, such as "Lamprocles's father".
3. Moving the existing copula before the attribute and adding only needed function words makes the ownership explicit without changing the answer set.

Prefer: "WH + copula + the attribute + of + target".
This is a general structural rule, not an attribute-word whitelist. Do not add facts, roles, or semantic predicates; only reorder existing words and add articles or prepositions when necessary.

Examples that should change:
- What nationality is Lamprocles's father?
  -> What is the nationality of Lamprocles's father?
- What profession is ENTITYA's husband?
  -> What is the profession of ENTITYA's husband?
- What religion was ENTITYA's mother?
  -> What was the religion of ENTITYA's mother?
- What date is ENTITYA's birth?
  -> What is the date of ENTITYA's birth?

Examples that should change:
- Who is the child of the director of film An Event?
  -> Who is the child of the person who directed the film An Event?
- What is the place of birth of the performer of song Changed It?
  -> Where was the person who performed the song Changed It born?

Examples that must stay unchanged:
- What is the capital of France?  (single-layer "of")
- Who was born first out of A and B?  (fixed phrase "out of")
- What event followed War of 1812?  ("of" inside an entity name)
- Which film has the director born later, A or B?  (parallel candidate question; Step4 handles candidate attachment)
- What country is ENTITYA located in?  ("located in" already states the relation)
- What is ENTITYA?  (there is no queried attribute noun)

Do not normalize for style, fluency, articles, general parser-friendliness, arbitrary wh-order repair, or parallel candidate attachment. The exact reorder in the implicit attribute-ownership pattern above is allowed; all other changes remain forbidden.
If strict equivalence is uncertain, return the original question and set normalization_changed=false.
""".strip()


EXPLICIT_ENTITY_EXTRACTION_SYSTEM = f"""
You are DEPO Step 2: topic entity extraction and narrow structural question normalization for HanLP PAS parsing.

Extract only explicit topic entities in the original question. Also return normalized_question, but its scope is intentionally limited.
The original question is the semantic authority; normalized_question is only a parser input. Preserve the exact answer set.

A topic entity is a concrete named thing explicitly mentioned in the question and useful as an anchor for QA decomposition:
person, creative work/title, organization, institution, location, geopolitical place, event, award, treaty, war, product, game, etc.

Do not extract roles, common nouns, answer slots, relation words, wh-words, operators, inferred entities, bare numbers, dates, years, ordinals, quantities, or measurements.

Entity selection procedure (perform it silently before producing JSON):
1. Apply the anchor gate before selecting a span: emit it only if the span itself names one particular person, work, organization, location, event, product, or other identifiable object. Apply the distinctiveness test: a valid surface identifies one particular object by itself, while a role, occupation, class, category, or description could denote many different objects and must be rejected. Do not emit a variable in the relation chain merely because resolving it is necessary to answer the question. Return an empty entity list only when no explicit named anchor passes this test; never use an empty list to avoid classifying an unfamiliar but clearly named surface.
2. Scan the whole original question from left to right for every explicit named object that passes the anchor gate. Scan subordinate clauses, possessives, quotations, comparisons, and long relation chains too; the grammatical focus of the question is not the only possible entity.
3. For each accepted candidate, keep the complete official-looking name or title, but remove only adjacent grammatical wrappers, role words, and possessive relations. A capitalized name component that looks like an article or initial is still part of a name when it is attached to the named surface, such as "A Lim" or "The Who". Do not use capitalization as the only test: source questions can lowercase a real name or title.
4. Emit each selected surface once, in left-to-right textual order. A surface denotes every exact occurrence of that surface in the original question. Before returning it, check all of those occurrences: selected surfaces must be mutually non-overlapping everywhere, not only at their first occurrence. Never emit both a full entity and one of its nested substrings.
5. Copy the surface exactly as written, even when the question contains a typo, unusual casing, malformed punctuation, or a non-English spelling. Do not correct, translate, expand, or normalize an entity surface.

Do not promote a descriptive label into an entity merely because it is capitalized or appears important to the question. Occupations, offices, demonyms, generic facilities, and administrative categories remain non-entities unless an actual proper name is present. For example, "prime minister", "Jamaican cricketer", "British racing driver", "Payload Specialist", "indoor arena", "Governor", "city", "region", "state", and "country" are descriptions, not topic entities. In contrast, a complete named institution, award, event, or location may legitimately contain category words, such as "Gujarat Legislative Assembly" or "Wexner Graduate Fellowships".

Treat relation variables as non-entities even when they occur in a long multi-hop question or appear with an article. Never emit bare or descriptive forms such as "city", "the city", "country", "the country", "region", "place", "the birthplace", "author", "the author", "the performer", "Governor", "headquarters", "group", "film company", "the president", "basilica", "museum", "series", "British racing driver", "English musician", "American", "2005 American drama-comedy film", or "seventh-most populated city". Descriptions such as "a Jamaican cricketer", "an American rock band", "an English actress", "a Payload Specialist", "an American professional Hawaiian surfer", and "a poetry and fiction writer" are likewise not entities. These are answer variables, roles, descriptions, or relation endpoints, not named anchors. A title or office becomes an entity only when its complete selected surface itself identifies a particular entity, such as "Prince Of Orange", "Crown Princess Of Denmark", or "University Of Kentucky".

A number or year is allowed only when it is part of a complete official name, such as "Sabotage (1936 Film)", "Wrong Turn 5: Bloodlines", or "War of 1812".
Short titles in a typed comparison list may begin with a number; keep the full branch when the surrounding question supplies the type, such as film, album, song, book, game, or series.

Creative works and other titles may contain internal punctuation such as colons, hyphens, apostrophes, parentheses, and subtitles. Treat the full official-looking title as one entity when the punctuation connects title parts; do not split the subtitle into a separate person/place/entity. A parenthetical disambiguator attached to any named entity remains inside that entity, for example "B Boy (Song)", "Sabotage (1936 Film)", or "Christopher Newton (Criminal)".

An internal "and" or "or" can belong to one official name or title. Keep it inside one entity when it joins parts of a single name-like construction, such as an event name of the form "Battle of X and Y" or a title such as "Love, Honor And Oh-Baby!". Split it only when it syntactically coordinates independent named mentions. Do not use external knowledge to invent a larger name.

Entity-boundary rules (apply these before returning any surface):
- Exclude grammatical type heads that introduce a title. For "the director of film The Private Life Of Cinema", return only "The Private Life Of Cinema", never "film The Private Life Of Cinema". Likewise, return "B Boy (Song)" rather than "song B Boy (Song)" and "A Nest Of Noblemen" rather than "film A Nest Of Noblemen". A type word stays only when it is capitalized inside the official title itself, especially in a parenthetical disambiguator.
- Exclude generic roles and relation phrases even when they are the semantic focus of the question. Never return "the director", "the performer", "the author", "the father", or "the husband" as an entity.
- Stop an entity before a possessive relation. For "Lamprocles's father", return "Lamprocles" only; do not return "Lamprocles's father" or "father". The same applies to possessive role and kinship suffixes such as director, father, mother, husband, wife, child, daughter, son, spouse, author, performer, and composer. For example, return "marty mcfly", not "marty mcfly's daughter", and return "Empress Wang", not "Empress Wang's husband".
- A comma-separated personal name and capitalized rank/designation are separate topic entities, not one merged entity. Return "Maurice" and "Prince Of Orange" separately from "Maurice, Prince Of Orange"; similarly split "Beatrice I, Countess Of Burgundy", "John Ernest, Duke Of Saxe-Eisenach", and "Mary, Crown Princess Of Denmark". Do not merge, trim, or otherwise rewrite either selected surface.

These rules override the general preference to keep an official-looking title whole. They do not split punctuation internal to a creative-work title, and they do not permit ordinary lowercase roles to become entities.
Some official titles begin with words that also look like question words, such as When, What, Who, Where, or Which. If that word is part of a capitalized official-looking title span, keep it inside the entity; do not trim the title to the following words.

Do not answer the question.
Do not decompose it into atomic questions.
Do not use external knowledge.
Do not change the meaning, answer type, relation direction, comparison direction, negation, quantifier, candidate set, or any restriction.
Keep every entity surface form exactly as it appears in the original question.

{NESTED_OF_NORMALIZATION_POLICY}

Hard semantic constraints:
- Preserve all explicit entities verbatim in normalized_question.
- Preserve and/or, both/either/neither, same, all, superlatives, ordinals, time, quantity, range, negation, and restrictive clauses.
- Do not replace with near-synonyms that change meaning: "born later" is not "younger"; "country of birth" is not "nationality".
- Do not introduce new named entities or ENTITYA/ENTITYB placeholders.

FINAL ENTITY OUTPUT GATE:
Before returning JSON, remove every surface that fails any condition below. This gate takes priority over the desire to return more entities.
1. The surface itself must identify one particular named object in the question. Do not return unknown relation variables, answer slots, roles, categories, descriptive phrases, demonyms, or bare numbers and years.
2. Reject generic forms such as "battle", "city", "the birthplace", "the performer", "Governor", "basilica", "museum", "state", "Payload Specialist", "Space Shuttle", "English musician", "poetry and fiction writer", "American", and "1956". A phrase is not an entity merely because it can be assigned a semantic type.
3. For a possessive X's role, retain X only when X is a named entity; never return the role or the entire possessive relation.
4. Copy source characters exactly. Never correct a typo, accent, capitalization, punctuation, or spacing.
5. All exact occurrences of all selected surfaces must be mutually non-overlapping. If a short candidate occurs inside a longer selected candidate anywhere in the question, omit the short candidate even when it also appears separately elsewhere.
6. Return an empty explicit_entities list only when the question contains no explicit named anchor. Do not omit a named person, official-looking title, named institution, event, or location merely because it is unfamiliar or its type is uncertain.

NORMALIZATION FIREWALL:
- Entity extraction and normalization are independent. Do not alter the original question to make entity boundaries easier.
- Follow the normalization policy above literally. Unless every listed trigger holds, return the original question and set normalization_changed=false.
- Never remove, resolve, shorten, or rewrite a possessive relation, role phrase, single relation, named entity, or location constraint. If strict equivalence is uncertain, return the original question unchanged.

The normalized question must be one natural English question ending with one question mark.
Before returning JSON, silently verify selected entities occur verbatim in both questions, the answer set is unchanged, no relation direction or restriction changed, no placeholders or external facts were introduced, and normalization_changed matches the actual text change.

Return JSON only.
""".strip()


def build_explicit_entity_extraction_prompt(question: str) -> str:

    schema = {
        "explicit_entities": [
            {
                "surface": "exact entity string from the original question",
                "type": "Person | Location | Organization | Work | Event | Other",
            }
        ],
        "normalized_question": "the original question, or a strictly equivalent allowed structural rewrite",
        "normalization_changed": True,
        "normalization_note": "brief description of the structural rewrite, or empty string if unchanged",
        "warnings": [],
    }

    return f"""
Original question:
{question}

Task:
Return the Step 2 JSON object. Directly identify explicit named topic entities, then apply the system normalization policy independently.

Entity rules:
1. Use the anchor gate and distinctiveness test: select a surface only when it itself identifies one particular named person, work, organization, location, event, product, or other object. Reject a phrase that could describe many things rather than identify one, even when it is capitalized or has modifiers. Do not select a relation variable, answer slot, role, category, description, demonym, bare number, or bare year.
2. Scan the entire question, including subordinate clauses, possessives, quotations, comparisons, and relation chains. Return accepted surfaces once each in their original left-to-right order. After rejecting relation variables, make a recall pass for named people, titles, institutions, events, and locations that occur away from the question focus.
3. Each surface is an exact case-preserving contiguous substring. Never correct spelling, accents, case, punctuation, or spacing. Source text can contain typos and lowercase names or titles.
4. Keep complete titles, including meaningful punctuation and parenthetical disambiguators. Remove only grammatical type heads that introduce a title: use "B Boy (Song)", not "song B Boy (Song)".
5. Do not output generic path nodes, even when they are capitalized or semantically important. Examples to exclude: "city", "country", "region", "place", "the birthplace", "the performer", "Governor", "headquarters", "group", "museum", "state", "series", "British racing driver", "Jamaican cricketer", "American rock band", "English actress", "Payload Specialist", "American professional Hawaiian surfer", and "poetry and fiction writer". In contrast, retain complete names such as "University Of Kentucky" or a lowercased title/name such as "marty mcfly".
6. Stop before possessive roles: output "Lamprocles", not "Lamprocles's father"; output "Empress Wang", not "Empress Wang's husband"; output "marty mcfly", not "marty mcfly's daughter".
7. Split a comma-separated personal name from its capitalized rank/designation: "Maurice, Prince Of Orange" -> ["Maurice", "Prince Of Orange"] and "Beatrice I, Countess Of Burgundy" -> ["Beatrice I", "Countess Of Burgundy"]. Do not return the unsplit full span in either case.
8. Do not return overlapping selections. A surface represents every one of its exact occurrences in the question, so omit a shorter surface if it occurs inside a longer selected surface anywhere.
9. Treat an exact source span as a possible title/name when it is the base of a possessive or the complement of a relation that conventionally takes a work or named object. For example, select "Vilaiyaadu Mankatha" from "Vilaiyaadu Mankatha's record label", "III" from "the performer of III", and "Crucifixion" from "Crucifixion's creator". Keep that named base span even if it is a single word, a Roman numeral, all caps, lowercased, or unfamiliar. This recall rule never authorizes a generic role or descriptive noun phrase.

Boundary decision examples:
- "The View from the Bottom is the fifth studio album by an American rock band" -> return "The View from the Bottom" only; "an American rock band" is a description.
- "Bill Nelson flew as a Payload Specialist on a Space Shuttle" -> return "Bill Nelson" only; the occupation and vehicle category are not entities.
- "What age was Georgia Middleman ... in the seventh-most populated city in the United States?" -> return "Georgia Middleman" and "United States"; the ordinal city description is not an entity.
- "Crucifixion's creator" -> return "Crucifixion", never the complete possessive phrase. Every selected surface must stop before a possessive marker when what follows is a generic relation or role.
- "Battle of Qurah and Umm al Maradim" is one event name. An "and" inside a complete event/work/place name is not a coordination boundary.
- "Beatrice I, Countess Of Burgundy's husband" -> return "Beatrice I" and "Countess Of Burgundy"; do not return the unsplit comma span.

Normalization:
- Keep normalized_question equal to the original question unless every system normalization condition holds. Never normalize a possessive relation, a single role relation, a named entity, or any constraint merely to simplify parsing.
- Preserve every selected entity verbatim in normalized_question. Do not answer, infer external facts, change relation direction, or add placeholders.

Final entity check: remove every output that is only a generic noun phrase or that changes even one source character. Keep a valid named anchor even when its type is uncertain; return [] only when no explicit named anchor remains. In particular, do not turn a generic role, occupation, nationality description, answer variable, or relation endpoint into an entity to increase recall. Before emitting each surface, apply the distinctiveness test and reject it when it is an article-led description, a role/occupation, a generic vehicle/facility/class, a place/category variable, a bare year/ordinal, a demonym, or a possessive phrase that includes a relation word. Thus reject "English musician", "American rock band", "Payload Specialist", "Space Shuttle", "museum", "the birthplace", "the president", "2005", and "American"; keep actual named titles and people. A leading article may remain only when it is genuinely part of an official title, such as "The View from the Bottom". Do this check immediately before JSON output.

Output JSON only, with exactly this shape:
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
