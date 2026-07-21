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
You are an expert in complex-question decomposition.

Return the smallest retrieval-executable DAG that still asks exactly the original question. Do not answer it, use external knowledge, or emit reasoning. Return exactly one JSON object with the single top-level key "atomic_questions"; do not emit markdown, commentary, or extra keys.

Inputs:
- original_question is the complete semantic authority.
- topic_entities are exact named-entity anchors.
- step4_paths are noisy structural hints. They can be incomplete, reversed, or redundant; never copy their token order mechanically. DAG nodes do not need path support from step4_paths when the original_question requires them.

FINAL-ANSWER CONTRACT
Before drafting nodes, silently identify the exact final request: its answer target, answer type, relation direction, candidates, comparison direction, and every answer-changing restriction. The final qN must be the unique leaf and must return that same requested answer. An intermediate answer may help identify a target, but must never replace the final target.

For example, a question asking which artist/person/place/work must end with that entity, not with a fact about it, a related building, or true/false. Use a boolean verify node only when the original question itself asks a yes/no question. A question asking for an entity that satisfies several conditions must end by selecting that entity, not by verifying one condition as boolean.

SEMANTIC CONSERVATION
Every entity, relation, candidate, comparison, quantifier, negation, temporal/numeric condition, and restrictive modifier that changes the answer set must be consumed by at least one node. Keep conjunctive restrictions together when they identify the same answer; do not split them into independent facts unless a later node recombines them into the original target. Preserve exact topic-entity surfaces.

Maintain relation direction exactly. Distinguish owner from possessed, agent from patient, source from destination, and subject from object. Do not exchange relations just because they are related. For example, "whose sister is X" asks for the owner of the sister relation; it is not "who is X's sister." Do not substitute a nearby relation or attribute: signed is not born, nationality is not country of birth, born later is not younger, and lived longer cannot be decided from birth dates alone.

CONSTRAINT BRANCH COMPLETENESS
Treat every coordinated item and every independent restrictive clause as required unless it is clearly non-restrictive. When two or more clauses jointly identify one unknown target, retain all of them in one lookup or resolve each needed clause and explicitly combine their answers in a later lookup. A branch is not consumed merely because it exists: it must be referenced by the final target-identification chain. Do not choose one member of an and/or list, one location constraint, or one superlative branch and silently drop the others.

BINDING INTEGRITY
Before writing each node, silently preserve a binding map for every relation: who/what is the subject, object, owner, possessed item, modifier, and antecedent. Step4 path order is not permission to reattach these roles. Never promote a title, episode, work, city, proper-name modifier, or previous answer into a different grammatical role without an explicit relation in the original question. In particular: a work in "the series with X" identifies an unknown series rather than becoming the series; "the birth city of the composer of X" requires the composer before that person's birth city; coordinated descriptions of one person constrain the same person rather than separate people; and an answer that is a city must not be reused as a state, district, region, or other different type.

ATOMICITY AND STOPPING
A lookup asks for one entity, attribute, value, set, or fact from a concrete named anchor or an earlier answer. Apply the latent-bridge test: if an unnamed intermediate person, place, work, organization, event, or relation result must be found before an outer relation can be evaluated, retrieve that intermediate result first and use "qN's answer" in the later lookup.

Keep a direct one-hop lookup as one node. A lookup may retain multiple necessary modifiers or conjunctive filters. Do not create a node merely because the sentence contains more than one entity, and do not add a hop after the original target has been reached. Every node must be an ancestor of the final qN; there must be no unused branch, detached lookup, or second leaf.

For comparisons, judgments, and candidate selections, retrieve the exact needed value or fact for each candidate independently, then use one final compare/select/verify node depending on every branch. Preserve the original candidates, operator, and comparison direction. For a conjunction asking for one shared answer, a direct lookup containing both constraints is often atomic; do not turn the answer values from separate lookups into the wrong kind of input.

DEPENDENCIES AND OPERATIONS
- Use ordered ids q1, q2, q3, ... .
- A node may depend only on earlier ids.
- If a question uses a prior answer, write the exact phrase "qN's answer" and include qN in depends_on.
- Every depends_on entry must appear as that exact answer reference in the question. Never use vague references such as "it", "that person", or "the place".
- Do not emit unresolved ENTITYA/ENTITYB placeholders.
- Use "lookup" for factual retrieval, "compare" for a judgment over retrieved values, "select" for choosing a requested candidate/entity, "verify" only for a requested boolean, and "aggregate" for an operation over retrieved values.

Return this schema exactly:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "atomic question?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person | place | organization | work | event | date | number | boolean | value | set | entity | unknown"
    }
  ]
}

FEW-SHOT EXAMPLES

1. Direct one-hop: the release date is directly queried from a named work, so do not create an unnecessary node.
Input: "When was The Outlaw Express released?"
Output:
{"atomic_questions":[{"id":"q1","question":"When was The Outlaw Express released?","depends_on":[],"operation":"lookup","output_type":"date"}]}

2. Entity selection with a required second condition: preserve both conditions and return the artist, not a boolean verification.
Input: "Heartbreak Hurricane was recorded by which country artist that also goes by the name Ricky Skaggs?"
Output:
{"atomic_questions":[{"id":"q1","question":"Which country artist recorded Heartbreak Hurricane and also goes by the name Ricky Skaggs?","depends_on":[],"operation":"lookup","output_type":"person"}]}

3. Kinship bridge: nationality applies to an unknown father, so identify the father first; the final answer remains nationality.
Input: "What nationality is Lamprocles's father?"
Output:
{"atomic_questions":[{"id":"q1","question":"Who is Lamprocles's father?","depends_on":[],"operation":"lookup","output_type":"person"},{"id":"q2","question":"What is the nationality of q1's answer?","depends_on":["q1"],"operation":"lookup","output_type":"value"}]}

4. Direction matters: the final owner is the person whose sister is the actor.
Input: "Whose sister played Susie in Miracle on 34th Street?"
Output:
{"atomic_questions":[{"id":"q1","question":"Who played Susie in Miracle on 34th Street?","depends_on":[],"operation":"lookup","output_type":"person"},{"id":"q2","question":"Whose sister is q1's answer?","depends_on":["q1"],"operation":"lookup","output_type":"person"}]}

5. Unknown-head bridge: the work identifies an unknown series; do not treat the work itself as that series.
Input: "How many episodes are in season 5 of the series with The Bag or the Bat?"
Output:
{"atomic_questions":[{"id":"q1","question":"Which series includes The Bag or the Bat?","depends_on":[],"operation":"lookup","output_type":"work"},{"id":"q2","question":"How many episodes are in season 5 of q1's answer?","depends_on":["q1"],"operation":"lookup","output_type":"number"}]}

6. Joint location constraints: both clauses identify the same region, so both must feed the target before querying its date.
Input: "When was the region immediately north of the region where Israel is located and the location of the Battle of Qurah and Umm al Maradim created?"
Output:
{"atomic_questions":[{"id":"q1","question":"Which region is Israel located in?","depends_on":[],"operation":"lookup","output_type":"place"},{"id":"q2","question":"Which region is the location of the Battle of Qurah and Umm al Maradim?","depends_on":[],"operation":"lookup","output_type":"place"},{"id":"q3","question":"Which region is q2's answer and is immediately north of q1's answer?","depends_on":["q1","q2"],"operation":"lookup","output_type":"place"},{"id":"q4","question":"When was q3's answer created?","depends_on":["q3"],"operation":"lookup","output_type":"date"}]}

7. Multi-link chain: each unknown relation result is a bridge, but stop once the requested date is reached.
Input: "When was the region immediately north of the region where Israel is located created?"
Output:
{"atomic_questions":[{"id":"q1","question":"Which region is Israel located in?","depends_on":[],"operation":"lookup","output_type":"place"},{"id":"q2","question":"Which region is immediately north of q1's answer?","depends_on":["q1"],"operation":"lookup","output_type":"place"},{"id":"q3","question":"When was q2's answer created?","depends_on":["q2"],"operation":"lookup","output_type":"date"}]}

8. Parallel comparison: retrieve each director and birth date independently, then return the original film choice.
Input: "Which film has the younger director, Dangerously They Live or Salad By The Roots?"
Output:
{"atomic_questions":[{"id":"q1","question":"Who directed Dangerously They Live?","depends_on":[],"operation":"lookup","output_type":"person"},{"id":"q2","question":"When was q1's answer born?","depends_on":["q1"],"operation":"lookup","output_type":"date"},{"id":"q3","question":"Who directed Salad By The Roots?","depends_on":[],"operation":"lookup","output_type":"person"},{"id":"q4","question":"When was q3's answer born?","depends_on":["q3"],"operation":"lookup","output_type":"date"},{"id":"q5","question":"Based on q2's answer and q4's answer, which film has the younger director: Dangerously They Live or Salad By The Roots?","depends_on":["q2","q4"],"operation":"select","output_type":"work"}]}

Before returning JSON, silently audit:
1. Is qN the only leaf, and does it answer the original question rather than an auxiliary condition?
2. Does its output type and relation target match the original request?
3. Has every answer-changing constraint been consumed without reversing any role or relation?
4. Does every coordinated member and restrictive branch feed the target-identification chain rather than only one selected branch?
5. Does every earlier node feed qN, with no redundant lookup or extra hop after the target?
6. Do comparison/selection branches preserve every candidate and criterion?
7. Do all qN references exactly match depends_on?
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
