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

A topic entity is a concrete named or identifier-like thing explicitly mentioned in the question and useful as an anchor for QA decomposition:
person, fictional character or named role, creative work/title, organization, institution, government body, location, named building or facility, named region, geopolitical place, event, award, competition, treaty, war, product, game, acronym, code, designation, etc.

Do not extract roles, common nouns, answer slots, relation words, wh-words, operators, inferred entities, bare numbers, dates, years, ordinals, quantities, or measurements.

Entity selection procedure (perform it silently before producing JSON):
1. Apply the anchor gate before selecting a span: emit a surface when it plausibly names or identifies a concrete person, fictional character, work, organization, institution, government body, named location, named region, named building/facility, event, award, competition, product, code, acronym, or other anchor in the question. Prefer recall for exact named or identifier-like surfaces, even when they are unfamiliar, lowercase, all-caps, short, numeric, or punctuated. Reject only pure roles, occupations, classes, categories, answer slots, and relation variables that do not name a concrete anchor. Do not emit a variable in the relation chain merely because resolving it is necessary to answer the question. Return an empty entity list only when no explicit named or identifier-like anchor is present.
2. Scan the whole original question from left to right for every explicit named or identifier-like anchor that passes the anchor gate. Scan subordinate clauses, possessives, quotations, comparisons, and long relation chains too; the grammatical focus of the question is not the only possible entity.
3. For each accepted candidate, keep the complete official-looking name or title, but remove only adjacent grammatical wrappers, role words, and possessive relations. A capitalized name component that looks like an article or initial is still part of a name when it is attached to the named surface, such as "A Lim" or "The Who". Do not use capitalization as the only test: source questions can lowercase a real name or title.
4. Emit each selected surface once, in left-to-right textual order. A surface denotes every exact occurrence of that surface in the original question. Before returning it, check all of those occurrences: selected surfaces must be mutually non-overlapping everywhere, not only at their first occurrence. Never emit both a full entity and one of its nested substrings.
5. Copy the surface exactly as written, even when the question contains a typo, unusual casing, malformed punctuation, or a non-English spelling. Do not correct, translate, expand, or normalize an entity surface.

Do not promote a descriptive label into an entity merely because it is capitalized or appears important to the question. Occupations, offices, demonyms, generic facilities, and administrative categories remain non-entities unless an actual proper name, identifier, or official designation is present. For example, "prime minister", "Jamaican cricketer", "British racing driver", "Payload Specialist", "indoor arena", "Governor", "governor general of India", "city", "region", "state", "country", "league", and "body of water" are descriptions or relation variables, not topic entities. In contrast, a complete named institution, award, event, competition, code, character, official designation, named building/facility, or location may legitimately contain category words, such as "Gujarat Legislative Assembly", "Wexner Graduate Fellowships", "House of Representatives", "US Senate", "Cabinet", "NATO", "Mantua Cathedral", "Birmingham", "Near East", "Indy Car Race", "FA Cup", "MLB MVP", "Susie", or "ISO 3166-2:CV".

Treat relation variables as non-entities even when they occur in a long multi-hop question or appear with an article. Never emit bare or descriptive forms such as "city", "the city", "country", "the country", "region", "place", "the birthplace", "author", "the author", "the performer", "Governor", "governor general of India", "headquarters", "group", "film company", "league", "body of water", "the president", "basilica", "museum", "series", "British racing driver", "English musician", "American", "Italian", "1999 draft", "2005 American drama-comedy film", or "seventh-most populated city". Descriptions such as "a Jamaican cricketer", "an American rock band", "an English actress", "a Payload Specialist", "an American professional Hawaiian surfer", and "a poetry and fiction writer" are likewise not entities. These are answer variables, roles, descriptions, or relation endpoints, not named anchors. A title or office becomes an entity only when its complete selected surface itself identifies a particular entity, such as "Prince Of Orange", "Crown Princess Of Denmark", or "University Of Kentucky".

A number, year, Roman numeral, or code is allowed when it is part of a complete official-looking name or identifier, such as "Sabotage (1936 Film)", "Wrong Turn 5: Bloodlines", "War of 1812", "1979-80 European Cup", "1894-95 FA Cup", "Shrek 2", "III", "ISO 3166-2:CV", or "MLB MVP". A bare calendar year before a generic event or type noun, such as "1999 draft", is a time constraint rather than an entity unless the full span is an official name.
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
Before returning JSON, remove surfaces that are clearly not named or identifier-like anchors. This gate prevents pure relation variables and descriptions; it must not suppress plausible exact anchors merely because they are unfamiliar, short, lowercase, all-caps, numeric, or punctuated.
1. The surface must be a named or identifier-like anchor in the question. Do not return unknown relation variables, answer slots, roles, categories, descriptive phrases, demonyms, or standalone numbers and years that are not part of an identifier/name.
2. Reject generic forms such as "battle", "city", "the birthplace", "the performer", "Governor", "governor general of India", "basilica", "museum", "state", "league", "film company", "body of water", "Payload Specialist", "Space Shuttle", "English musician", "poetry and fiction writer", "American", "Italian", "1956", and "1999". A phrase is not an entity merely because it can be assigned a semantic type. However, keep official-looking names and identifiers that include category words or numbers, such as "House of Representatives", "US Senate", "Cabinet", "NATO", "Civil War", "Mexican-American war", "Korean conflict", "Indy Car Race", "FA Cup", "European Cup", "1979-80 European Cup", "1894-95 FA Cup", "MLB MVP", "ISO 3166-2:CV", "Shrek 2", "III", "Auctor", "KZAR", "Darling Mills Creek", "Mantua Cathedral", "Birmingham", "Near East", "Susie", and "Manchuria".
3. For a possessive X's role, retain X only when X is a named entity; never return the role or the entire possessive relation.
4. Copy source characters exactly. Never correct a typo, accent, capitalization, punctuation, or spacing.
5. All exact occurrences of all selected surfaces must be mutually non-overlapping. If a short candidate occurs inside a longer selected candidate anywhere in the question, omit the short candidate even when it also appears separately elsewhere.
6. Return an empty explicit_entities list only when the question contains no plausible named or identifier-like anchor. Make a final recall pass for named places, named regions, named buildings/facilities, named characters, short titles, acronyms, codes, awards, competitions, named events, government bodies, institutions, rivers/creeks, stations, places, and lowercase or unfamiliar names. Do not omit a named person, official-looking title, named institution, government body, event, identifier, acronym, named building/facility, named region, or location merely because it is unfamiliar or its type is uncertain.

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
1. Use the anchor gate and distinctiveness test: select a surface when it plausibly identifies a named or identifier-like person, work, organization, location, event, award, competition, product, code, acronym, designation, or other anchor. Prefer recall for exact surfaces that look like names, titles, acronyms, codes, or official designations, even when they are unfamiliar, lowercase, all-caps, short, numeric, or punctuated. Reject only phrases that are clearly relation variables, answer slots, roles, generic categories, descriptions, demonyms, standalone numbers, or standalone years.
2. Scan the entire question, including subordinate clauses, possessives, quotations, comparisons, and relation chains. Return accepted surfaces once each in their original left-to-right order. After rejecting relation variables, make a recall pass for named people, titles, institutions, events, awards, competitions, locations, acronyms, codes, and official designations that occur away from the question focus.
3. Each surface is an exact case-preserving contiguous substring. Never correct spelling, accents, case, punctuation, or spacing. Source text can contain typos and lowercase names or titles.
4. Keep complete titles, including meaningful punctuation and parenthetical disambiguators. Remove only grammatical type heads that introduce a title: use "B Boy (Song)", not "song B Boy (Song)".
5. Do not output generic path nodes, even when they are capitalized or semantically important. Examples to exclude: "city", "country", "region", "place", "the birthplace", "the performer", "Governor", "governor general of India", "headquarters", "group", "film company", "league", "body of water", "museum", "state", "series", "British racing driver", "Jamaican cricketer", "American rock band", "English actress", "Payload Specialist", "American professional Hawaiian surfer", "poetry and fiction writer", "American", "Italian", and a bare year such as "1999". In contrast, retain complete or official-looking names and identifiers such as "University Of Kentucky", "House of Representatives", "US Senate", "Cabinet", "NATO", "Mantua Cathedral", "Birmingham", "Near East", "Civil War", "Mexican-American war", "Korean conflict", "Indy Car Race", "FA Cup", "1894-95 FA Cup", "MLB MVP", "ISO 3166-2:CV", "KZAR", "Darling Mills Creek", "Shrek 2", "Susie", "Auctor", or a lowercased title/name such as "marty mcfly".
6. Stop before possessive roles: output "Lamprocles", not "Lamprocles's father"; output "Empress Wang", not "Empress Wang's husband"; output "marty mcfly", not "marty mcfly's daughter".
7. Split a comma-separated personal name from its capitalized rank/designation: "Maurice, Prince Of Orange" -> ["Maurice", "Prince Of Orange"] and "Beatrice I, Countess Of Burgundy" -> ["Beatrice I", "Countess Of Burgundy"]. Do not return the unsplit full span in either case.
8. Do not return overlapping selections. A surface represents every one of its exact occurrences in the question, so omit a shorter surface if it occurs inside a longer selected surface anywhere.
9. Treat an exact source span as a possible title/name when it is the base of a possessive or the complement of a relation that conventionally takes a work, named object, named place, named facility, government body, or named character. For example, select "Vilaiyaadu Mankatha" from "Vilaiyaadu Mankatha's record label", "III" from "the performer of III", "Crucifixion" from "Crucifixion's creator", "Auctor" from "the language Auctor comes from", "KZAR" from "where KZAR is located", "Birmingham" from "in Birmingham", "Mantua Cathedral" from "Mantua Cathedral is dedicated to", "Near East" from "the disgrace of Near East", and "Susie" from "played Susie". Keep that named base span even if it is a single word, a Roman numeral, all caps, lowercased, or unfamiliar. This recall rule never authorizes a generic role or descriptive noun phrase.

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

Final entity check: remove every output that is only a generic noun phrase or that changes even one source character. Keep a plausible named or identifier-like anchor even when its type is uncertain; return [] only when no explicit named or identifier-like anchor remains. In particular, do not turn a generic role, occupation, nationality description, answer variable, or relation endpoint into an entity to increase recall. Before emitting each surface, apply the distinctiveness test and reject it when it is an article-led description, a role/occupation, a generic vehicle/facility/class, a place/category variable, a bare year/ordinal, a demonym, or a possessive phrase that includes a relation word. Thus reject "English musician", "American rock band", "Payload Specialist", "Space Shuttle", "museum", "the birthplace", "the president", "film company", "league", "city", "1999", and "American"; keep actual named titles, people, characters, institutions, government bodies, events, places, named buildings/facilities, named regions, acronyms, codes, and official-looking designations. A leading article may remain only when it is genuinely part of an official title, such as "The View from the Bottom". Do this check immediately before JSON output.

Output JSON only, with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


MASK_SPAN_EXTRACTION_SYSTEM = EXPLICIT_ENTITY_EXTRACTION_SYSTEM


def build_mask_span_extraction_prompt(question: str) -> str:
    return build_explicit_entity_extraction_prompt(question)


ATOMIC_QUESTION_DAG_SYSTEM = r"""
You are an expert in decomposing complex questions into atomic-question DAGs.

Your task is to convert an `original_question` and its `question_structure` into the smallest semantically complete and retrieval-executable Atomic Question DAG.

Return exactly one JSON object and no other text.

## Inputs

### `original_question`

The original question is the only source of semantic meaning.

Use it to determine:

* the exact entities and candidates;
* relation meanings and directions;
* the final answer target;
* comparison or selection criteria;
* conjunction, disjunction, negation, and quantifiers;
* temporal, numeric, superlative, and restrictive conditions.

Preserve named entities and candidate surface forms exactly as written in the original question.

### `question_structure`

The question structure contains one or more structural branches.

Example:

Branch 1:
Changed It -- song -- performer -- birth -- place

Branch 2:
Another Song -- song -- performer -- birth -- place

Each branch is an approximate structural skeleton of the question. Adjacent nodes are separated by `--`.

The separator means only that two nodes are structurally connected. It does not specify:

* factual direction;
* subject-object direction;
* grammatical dependency direction;
* answer dependency;
* exact surface-word order.

The left-to-right order represents an approximate traversal through the question structure, usually from a known mention toward relations, intermediate results, conditions, or the query target.

Use the question structure to identify likely relation chains, intermediate results, parallel branches, and decomposition order.

The question structure may omit function words, contain redundant nodes, collapse nearby relations, or have imperfect local ordering. When it conflicts with the original question, always follow the original question.

Do not introduce, remove, shorten, or replace an entity only because of its form in the question structure.

## Decomposition Rules

1. First identify the exact answer requested by the original question.

2. Use the question structure to determine the relation chain or parallel branches that lead to that answer.

3. Create an intermediate atomic question when an unnamed entity, person, place, work, organization, event, value, or relation result must be found before the next relation can be evaluated.

4. Keep a direct one-hop question as one atomic question.

5. Do not combine multiple sequential unknown relations into one atomic question.

6. Multiple modifiers or restrictions that jointly identify the same target may remain in one atomic question.

7. For independent candidates or comparison subjects, create independent branches and combine them in one final node.

8. Stop as soon as the exact answer requested by the original question can be produced.

9. Remove every node that does not contribute directly or indirectly to the final node.

## Semantic Preservation

Preserve every answer-changing part of the original question, including:

* relation direction and participant roles;
* all named entities and candidates;
* conjunction and disjunction;
* comparison direction;
* negation and quantifiers;
* temporal and numeric conditions;
* superlatives;
* restrictive clauses;
* the final answer target.

Do not replace one relation with a related but different relation.

For example:

* nationality is not country of birth;
* born later is not younger;
* owner is not possessed object;
* agent is not patient;
* source is not destination.

Do not invent entities, facts, relations, restrictions, candidates, or intermediate hops.

Do not output unresolved placeholders such as `ENTITYA` or `ENTITYB`.

## Atomic Questions

Each lookup node must ask for one new entity, attribute, value, set, or fact using:

* a named anchor from the original question;
* one or more earlier answers;
* or both.

Every atomic question must be understandable as a standalone retrieval query after its answer references are resolved.

## Dependencies

Use ordered IDs:

`q1`, `q2`, `q3`, ...

A node may depend only on earlier nodes.

When a question uses an earlier answer:

* refer to it using exactly `qN's answer`;
* include `qN` in `depends_on`.

Every ID in `depends_on` must appear in the question as `qN's answer`.

Every `qN's answer` reference in the question must appear in `depends_on`.

The final node must be the only leaf node.

Every earlier node must contribute directly or indirectly to the final node.

## Operations

Use only the following operations:

* `lookup`: retrieve an entity, fact, attribute, value, or set;
* `select`: choose the requested entity or candidate using earlier answers;
* `compare`: perform a requested comparison over earlier answers;
* `verify`: return a boolean answer when the original question asks yes or no;
* `aggregate`: perform a required numeric or set operation.

## Output Schema

Return exactly:

{
"atomic_questions": [
{
"id": "q1",
"question": "atomic natural-language question?",
"depends_on": [],
"operation": "lookup"
}
]
}

Do not add any other top-level key or node field.

## Examples

### Example 1: Direct one-hop question

Input:

Original question:
When was The Outlaw Express released?

Question structure:

Branch 1:
The Outlaw Express -- released -- When

Output:

{"atomic_questions":[{"id":"q1","question":"When was The Outlaw Express released?","depends_on":[],"operation":"lookup"}]}

### Example 2: Sequential intermediate result

Input:

Original question:
What is the place of birth of the performer of song Changed It?

Question structure:

Branch 1:
Changed It -- song -- performer -- birth -- place

Output:

{"atomic_questions":[{"id":"q1","question":"Who performed the song Changed It?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"Where was q1's answer born?","depends_on":["q1"],"operation":"lookup"}]}

### Example 3: Parallel candidate comparison

Input:

Original question:
Which film has the director born later, Dangerously They Live or Salad By The Roots?

Question structure:

Branch 1:
Dangerously They Live -- director -- born -- later

Branch 2:
Salad By The Roots -- director -- born -- later

Output:

{"atomic_questions":[{"id":"q1","question":"Who directed Dangerously They Live?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"When was q1's answer born?","depends_on":["q1"],"operation":"lookup"},{"id":"q3","question":"Who directed Salad By The Roots?","depends_on":[],"operation":"lookup"},{"id":"q4","question":"When was q3's answer born?","depends_on":["q3"],"operation":"lookup"},{"id":"q5","question":"Based on q2's answer and q4's answer, which film has the director who was born later: Dangerously They Live or Salad By The Roots?","depends_on":["q2","q4"],"operation":"select"}]}

Before returning the JSON, silently verify:

1. The final node answers exactly the original question.
2. Every answer-changing relation and restriction is preserved.
3. The question structure is used wherever it agrees with the original question.
4. Every intermediate node is necessary and atomic.
5. Every dependency matches its answer reference.
6. Every earlier node contributes to the final node.
7. The final node is the only leaf.

""".strip()


def build_atomic_question_dag_prompt(
    original_question: str,
    question_structure: list[list[str]],
) -> str:
    lines = [
        "Original question:",
        str(original_question),
        "",
        "Question structure:",
        "",
    ]
    branch_index = 1
    for branch in question_structure:
        nodes = []
        for node in branch:
            text = str(node).strip()
            if text:
                nodes.append(text)
        if not nodes:
            continue
        lines.extend([f"Branch {branch_index}:", " -- ".join(nodes), ""])
        branch_index += 1
    return "\n".join(lines).rstrip()
