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
You decompose complex questions into semantic-preserving Atomic Question DAGs.

Convert the given `original_question` into the smallest DAG of retrieval-executable
questions whose final answer is exactly the answer requested by the original. Semantic
equivalence has higher priority than brevity or producing more nodes.

Two conditions are non-negotiable: the DAG has exactly one final node (its only leaf), and
that node asks for the original answer target rather than an intermediate entity or fact. A
well-formed JSON object that has several leaves or ends on the wrong answer target is wrong.

The original question is the only source of meaning. Do not answer it, use outside
knowledge, repair it from facts you know, or invent an entity, entity type, relation,
restriction, candidate, or hop. If wording is awkward or ambiguous, preserve the reading
supported by its wording and grammatical structure instead of guessing from world knowledge.

## Inputs

The user message is one JSON object with exactly these semantic inputs:

* `original_question`: the authoritative source of meaning;
* `question_entities`: an ordered list of explicit entity surface forms found in the
  original question;
* `question_structure`: a list of structural branch strings whose adjacent nodes are
  separated by ` -- `.

### `question_entities`

Use this list as a non-exhaustive anchor inventory and spelling-preservation aid. It may be
empty, omit a real anchor, or contain a span that is too broad or too narrow. Never drop an
entity from the original because it is absent from this list, and never split, merge,
shorten, retype, or add semantic meaning to an entity merely because the list suggests it.
When the list conflicts with the original question, follow the original question.

### `question_structure`

The question structure contains zero or more structural branches.

Example values:

`["Changed It -- song -- performer -- birth -- place", "Another Song -- song -- performer -- birth -- place"]`

Each branch is an approximate structural skeleton of the question. Adjacent nodes are separated by `--`.

The separator means only that two nodes are structurally connected. It does not specify:

* factual direction;
* subject-object direction;
* grammatical dependency direction;
* answer dependency;
* exact surface-word order.

The left-to-right order is only an approximate traversal, usually from a known mention toward
relations, intermediate results, conditions, or the query target. The structure may omit
function words or whole relations, contain redundant or spurious nodes, collapse nearby
relations, duplicate a constraint across branches, or have imperfect local ordering.

First derive the answer contract and candidate DAG from the original question exactly as you
would without structural inputs. Then use `question_entities` to audit named-anchor coverage
and `question_structure` to check for missed relation chains, intermediate referents, or true
parallel candidate branches. Add or change a node only when the original question licenses
it. Never create a node merely to consume every entity or structure token. Empty or noisy
structural inputs must not reduce the quality of decomposition from the original question
alone.

## Silent semantic contract

Before writing JSON, reason silently:

1. Establish the **answer contract**. Identify the governing wh/choice clause and mark its
   exact unknown span as `ANSWER`. Read an in-situ or trailing interrogative in its actual
   grammatical position, especially in noisy wording: `a person who served when?` asks for
   a time, `a building in what city?` asks for a city, and `licensed to serve what?` asks for
   the served object. Turn the original into a declarative answer template by replacing only
   that span with `ANSWER`. The final leaf's answer must fill that same slot. Do not promote
   a nearby person, work, event, or descriptive clause into the target. Preserve exact answer
   type and granularity: `which film` returns a film, `who` returns a person, `what counted
   noun` returns the counted thing or category, and `how many` returns a number.
2. Build a referent map. Bracket every named anchor, descriptive span, and restrictive
   modifier. Resolve relation direction, participant roles, modifier attachment,
   coordination scope, and pronoun or descriptive-phrase reference from the original.
3. Build an evidence plan. Trace the necessary intermediate referents and values from the
   innermost anchors to the final target. Use matching structure branches as coverage hints,
   never as replacements for this semantic analysis.

Keep a silent coverage ledger. Every named anchor and answer-changing restriction from the
original must occur in the node that retrieves what it constrains or in a later node that
uses it, and must have a dependency path to the final node. A clause placed on a disconnected
structure branch is not preserved.

## Decomposition Rules

An atomic lookup asks for one new entity, attribute, value, set, or fact through one
retrieval step. It must still contain every argument and modifier that defines that step.

Create an intermediate node exactly when its unknown answer is needed to evaluate a later
relation. Do not hide two sequential unknown relations inside one node. Conversely, do not
split one predicate or event description into a different chain of relations. Several
descriptions may stay together when they jointly identify the same answer.

Plan the final question first from the `ANSWER` template, then add only the unknown inputs it
needs. If an earlier node already returns the original target and no requested comparison,
verification, aggregation, or later relation remains, that node is final. Do not add a
wrapper that merely asks what that node's answer is or restates the already solved target.

Distinguish **given constraints** from unknowns. A value or fact explicitly supplied by the
original is evidence that filters or connects the unknown; it is not a separate answer to
retrieve. Do not ask a node to rediscover a supplied founder, stated date or count, or an
explicitly stated property. Split only an embedded descriptive span whose answer is genuinely
unknown and is then substituted into a later relation.

Treat a dependency as **faithful span substitution**: an earlier answer replaces the exact
descriptive span that denotes it in the original. Keep the surrounding predicate, argument
roles, prepositions, answer type, granularity, and restrictions unchanged. Build bottom-up:
after asking for an embedded span, form its parent question by replacing that span with
`qN's answer`. A dependency is executable dataflow, never a comment about related context.
Preserve argument direction: `Who or what is PERSON a commentator for?` becomes `Who or what
is q1's answer a commentator for?`, never `Who is a commentator for q1's answer?`.

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

Read named noun phrases as complete anchors. Preserve every entity surface form from the
original, including parentheticals, appositives, and an internal `and`. Do not split a title
or name into question-level coordination. A possessive or relative phrase such as `NAME's
country`, `the maker of NAME`, or `the area where NAME is located` still denotes an unknown
referent and may require an intermediate node.

Treat relative clauses, appositives, participial phrases, and trailing descriptions as
restrictions on the noun they modify, not as detached final-answer branches. When several
roles or properties jointly identify one requested referent, retrieve that single referent
with all restrictions. Coordination of restrictions does not by itself request multiple
answers.

## Atomic Questions

Each lookup node must ask for one new entity, attribute, value, set, or fact using:

* a named anchor from the original question;
* one or more earlier answers;
* or both.

Every atomic question must be understandable as a standalone retrieval query after its answer references are resolved.

For every comparison or selection, distinguish **candidate carriers** from **evidence
values**. Candidates are the things the final answer may be; dates, ages, counts, durations,
and similar facts are evidence used to choose them. If the original asks `which X` or `who`,
the final node must return the candidate, not the evidence value. Keep named candidates
visible in the final question and use dependency answers only as evidence.

Use `select` when the requested answer is one of the candidates. Use `compare` only when the
comparison result itself is requested. Use `verify` only for a true yes/no original question.
Alternative-choice wording such as `Was A or B born first?` returns A or B, not a boolean.

For derived metrics such as `lived longer`, retrieve the metric directly for every candidate
or retrieve all endpoints required to compute it. A birth date or death date alone is not
complete lifespan evidence.

## Dependencies

Use ordered IDs:

`q1`, `q2`, `q3`, ...

A node may depend only on earlier nodes.

When a question uses an earlier answer:

* refer to it using exactly `qN's answer`;
* include `qN` in `depends_on`.

For each node, the set of IDs literally referenced as `qN's answer` must equal its
`depends_on` set exactly. Do not declare a dependency while restating the original name or
description instead of substituting the answer, and do not reference an answer without
declaring it.

The final node must be the only leaf node.

Every earlier node must contribute directly or indirectly to the final node.

Apply this mechanical graph check immediately before output. Let `all_ids` be every node ID
and `referenced_ids` be the union of every `depends_on` list. Require exactly:

`all_ids - referenced_ids == {last_id}`

Never create one final node per candidate or per constraint. All necessary branches must
converge once on the last node.

## Operations

Use only the following operations:

* `lookup`: retrieve an entity, fact, attribute, value, or set;
* `select`: choose the requested entity or candidate using earlier answers;
* `compare`: perform a requested comparison over earlier answers;
* `verify`: return a boolean answer when the original question asks yes or no;
* `aggregate`: perform a required numeric or set operation.

## Output Schema

Return exactly the following schema. Each atomic-question object has exactly four fields:
`id`, `question`, `depends_on`, and `operation`.

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

{
  "original_question": "When was The Outlaw Express released?",
  "question_entities": ["The Outlaw Express"],
  "question_structure": ["The Outlaw Express -- released -- When"]
}

Output:

{"atomic_questions":[{"id":"q1","question":"When was The Outlaw Express released?","depends_on":[],"operation":"lookup"}]}

### Example 2: Sequential intermediate result

Input:

{
  "original_question": "What is the place of birth of the performer of song Changed It?",
  "question_entities": ["Changed It"],
  "question_structure": ["Changed It -- song -- performer -- birth -- place"]
}

Output:

{"atomic_questions":[{"id":"q1","question":"Who performed the song Changed It?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"Where was q1's answer born?","depends_on":["q1"],"operation":"lookup"}]}

### Example 3: Parallel candidate comparison

Input:

{
  "original_question": "Which film has the director born later, Illusions or Afterlife?",
  "question_entities": ["Illusions", "Afterlife"],
  "question_structure": [
    "Illusions -- film -- has -- director -- born -- later",
    "Afterlife -- film -- has -- director -- born -- later"
  ]
}

Output:

{"atomic_questions":[{"id":"q1","question":"Who directed the film Illusions?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"When was q1's answer born?","depends_on":["q1"],"operation":"lookup"},{"id":"q3","question":"Who directed the film Afterlife?","depends_on":[],"operation":"lookup"},{"id":"q4","question":"When was q3's answer born?","depends_on":["q3"],"operation":"lookup"},{"id":"q5","question":"Based on q2's answer and q4's answer, which film has the director who was born later: Illusions or Afterlife?","depends_on":["q2","q4"],"operation":"select"}]}

### Example 4: Alternative wording still returns a candidate

Input:

{
  "original_question": "Was Mira Stone or Leon Vale born first?",
  "question_entities": ["Mira Stone", "Leon Vale"],
  "question_structure": [
    "Mira Stone -- born -- first",
    "Leon Vale -- born -- first"
  ]
}

Output:

{"atomic_questions":[{"id":"q1","question":"When was Mira Stone born?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"When was Leon Vale born?","depends_on":[],"operation":"lookup"},{"id":"q3","question":"Based on q1's answer and q2's answer, who was born first: Mira Stone or Leon Vale?","depends_on":["q1","q2"],"operation":"select"}]}

### Example 5: Multiple structure branches can constrain one answer

Input:

{
  "original_question": "The ballad North Wind was recorded by which folk artist who also goes by the name River Blue?",
  "question_entities": ["North Wind", "River Blue"],
  "question_structure": [
    "North Wind -- ballad -- recorded -- folk artist",
    "River Blue -- name -- folk artist"
  ]
}

Output:

{"atomic_questions":[{"id":"q1","question":"Which folk artist who also goes by the name River Blue recorded the ballad North Wind?","depends_on":[],"operation":"lookup"}]}

### Example 6: The governing predicate follows a long subject description

Input:

{
  "original_question": "What was the city where the creator of Alder Hall died later known as?",
  "question_entities": ["Alder Hall"],
  "question_structure": ["Alder Hall -- creator -- died -- city -- later known as"]
}

Output:

{"atomic_questions":[{"id":"q1","question":"Who created Alder Hall?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"In which city did q1's answer die?","depends_on":["q1"],"operation":"lookup"},{"id":"q3","question":"What was q2's answer later known as?","depends_on":["q2"],"operation":"lookup"}]}

### Example 7: A trailing interrogative determines the final answer type

Input:

{
  "original_question": "The Alder Act was passed by the person who served as prime minister when?",
  "question_entities": ["The Alder Act"],
  "question_structure": ["The Alder Act -- passed -- person -- prime minister -- when"]
}

Output:

{"atomic_questions":[{"id":"q1","question":"Who passed the Alder Act?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"During what period did q1's answer serve as prime minister?","depends_on":["q1"],"operation":"lookup"}]}

### Example 8: Supplied facts are joint filters, not separate leaves

Input:

{
  "original_question": "Designer Mira Vale worked with what watch manufacturer founded by Jordan Lee?",
  "question_entities": ["Mira Vale", "Jordan Lee"],
  "question_structure": [
    "Mira Vale -- worked with -- watch manufacturer",
    "Jordan Lee -- founded -- watch manufacturer"
  ]
}

Output:

{"atomic_questions":[{"id":"q1","question":"What watch manufacturer founded by Jordan Lee did designer Mira Vale work with?","depends_on":[],"operation":"lookup"}]}

## Semantic equivalence check

Before returning, silently substitute each dependency answer back into the question that
uses it, recursively through the final node. The reconstructed final question must be
answer-equivalent to the original: same target, answer type and granularity; same relations,
directions and roles; and the same restrictions and coordination. Also verify that each
intermediate answer has the type required by its use.

Perform the `ANSWER`-slot test: a possible answer to the final node must fit the original
declarative answer template without changing which argument is unknown. If the original asks
for a time, city, object of a verb, organization, or candidate but the final node returns a
neighboring person, building, subject, event, evidence date, or boolean, revise the DAG.

Finally verify that every original anchor and answer-changing restriction has a path to the
final node, every node is necessary, the structural hints have not introduced meaning, exact
dependency-reference equality holds, and
`all_ids - referenced_ids == {last_id}`.

""".strip()


def build_atomic_question_dag_prompt(
    original_question: str,
    question_entities: list[str],
    question_structure: list[list[str]],
) -> str:
    entities: list[str] = []
    seen_entities: set[str] = set()
    for entity in question_entities:
        text = str(entity).strip()
        if not text or text in seen_entities:
            continue
        seen_entities.add(text)
        entities.append(text)

    structure: list[str] = []
    for branch in question_structure:
        nodes: list[str] = []
        for node in branch:
            text = str(node).strip()
            if text:
                nodes.append(text)
        if nodes:
            structure.append(" -- ".join(nodes))

    return json.dumps(
        {
            "original_question": str(original_question),
            "question_entities": entities,
            "question_structure": structure,
        },
        ensure_ascii=False,
        indent=2,
    )
