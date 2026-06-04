from __future__ import annotations

import json

ALLOWED_OPERATORS = [
    "NONE",
    "COMPARE_SAME",
    "COMPARE_DIFF",
    "COMPARE_GREATER",
    "COMPARE_LESS",
    "ARGMAX",
    "ARGMIN",
    "INTERSECTION",
    "UNION",
    "DIFFERENCE",
    "LOGICAL_AND",
    "LOGICAL_OR",
]

CANDIDATE_NODES_AND_FRAME_SYSTEM = """
You are implementing the DEPO path-projection pipeline after CoreNLP parsing.

Current step: produce a high-recall candidate node pool and a Problem Frame.

The candidate node pool is not the final AST.
You may include noisy but potentially useful candidate nodes such as type qualifiers,
operator cues, role nouns, value slots, coreference mentions, and constraints.

Do not generate a full AST.
Do not generate selected paths.
Do not generate atomic questions.
Return valid JSON only.
""".strip()


def build_candidate_nodes_and_frame_prompt(
    question: str,
    restored_question: str,
    graph_nodes: list[dict[str, object]],
) -> str:
    schema = {
        "candidate_nodes": [
            {
                "id": "n1",
                "text": "MovieA",
                "kind": "entity",
                "graph_node_ids": ["4"],
                "confidence": 1.0,
            }
        ],
        "problem_frame": {
            "operator": "COMPARE_SAME",
            "answer_mode": "boolean",
            "answer_focus": None,
            "notes": None,
            "requirements": [
                {
                    "id": "r1",
                    "root": "MovieA",
                    "target": "nationality",
                    "description": "nationality associated with MovieA branch",
                },
                {
                    "id": "r2",
                    "root": "MovieB",
                    "target": "nationality",
                    "description": "nationality associated with MovieB branch",
                },
            ],
        },
    }
    return f"""
Build high-recall candidate nodes and a Problem Frame for the question.

Original question:
{question}

Restored/normalized question used by the parser-facing pipeline:
{restored_question}

Restored dependency graph nodes:
{json.dumps(graph_nodes, ensure_ascii=False, indent=2)}

Candidate node rules:
1. candidate_nodes is a high-recall pool, not final AST nodes.
2. Include every likely semantic endpoint needed for path construction.
3. It is acceptable to include noisy candidates such as type qualifiers, operator cues,
   constraint values, and coreference mentions, but mark kind accurately.
4. Use kind values from:
   entity, role, slot, type_qualifier, operator_cue, constraint_value, coref, other.
5. If possible, include graph_node_ids copied from the provided graph node list.
6. Do not invent graph_node_ids. If grounding is uncertain, omit graph_node_ids.

Problem Frame rules:
1. requirements must describe the branch-level outputs the final program needs.
2. Each requirement should have a stable id such as r1, r2.
3. root is the branch starting point, usually a fixed entity.
4. target is the branch-level value that should feed the final operator.
5. operator must be one of:
   {", ".join(ALLOWED_OPERATORS)}
6. Use NONE for a single serial lookup with no comparison, set, ranking, or logical operator.
7. Use COMPARE_SAME when the question asks whether branch outputs are the same.
8. Use COMPARE_DIFF when the question asks whether branch outputs are different.
9. Use ARGMAX/ARGMIN for max/min selection and INTERSECTION/UNION/DIFFERENCE for set operations.

Forbidden:
- Do not generate a complete AST.
- Do not select final paths.
- Do not generate atomic subquestions.
- Do not output markdown.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


SELECT_PATHS_SYSTEM = """
You are implementing the DEPO path-projection pipeline.

Current step: choose exactly one provided candidate path for each requirement.

You must only select from the supplied candidate_paths by path_id.
Do not create paths.
Do not generate an AST.
Do not generate atomic questions.
Return valid JSON only.
""".strip()


def build_select_paths_prompt(
    question: str,
    problem_frame: dict[str, object],
    filtered_candidate_paths: list[dict[str, object]],
    validation_feedback: str | None = None,
) -> str:
    schema = {
        "selected_paths": [
            {
                "requirement_id": "r1",
                "path_id": "p1",
            },
            {
                "requirement_id": "r2",
                "path_id": "p2",
            },
        ]
    }
    feedback = f"\nPrevious selection failed validation:\n{validation_feedback}\n" if validation_feedback else ""
    return f"""
Select candidate paths for the requirements.

Original question:
{question}

Problem Frame:
{json.dumps(problem_frame, ensure_ascii=False, indent=2)}

Filtered candidate paths:
{json.dumps(filtered_candidate_paths, ensure_ascii=False, indent=2)}
{feedback}
Task:
- For each requirement in Problem Frame, choose exactly one path.
- The number of selected paths must equal the number of requirements.
- Use only path_id values from filtered candidate paths.
- Do not invent or rewrite paths.
- Do not generate an AST.
- Do not generate atomic questions.

Selection principles:
1. For each requirement, prefer the path that best expresses root to target.
2. If multiple paths are reasonable, prefer the shorter and more direct path.
3. Do not choose a path that only connects two roots and does not express the requirement target.
4. Do not choose a path that only contains type/context information unless that type is the requirement target.
5. The selected path's candidate_for must include the requirement_id.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


LABEL_AST_EDGES_SYSTEM = """
You are implementing the DEPO path-projection pipeline.

Current step: label fixed AST skeleton edges and confirm the operator.

The AST skeleton is already built by code from selected paths.
You may only label existing edges and confirm operator metadata.
You must not add, delete, merge, split, shortcut, or reorder AST nodes or edges.
Return valid JSON only.
""".strip()


def build_label_ast_edges_prompt(
    question: str,
    ast_skeleton: dict[str, object],
    selected_paths: list[dict[str, object]],
    problem_frame: dict[str, object],
) -> str:
    schema = {
        "edges": [
            {
                "source": "MovieA",
                "target": "director_r1",
                "relation": "director of MovieA",
                "atomic_question_template": "Who directed MovieA?",
            },
            {
                "source": "director_r1",
                "target": "nationality_r1",
                "relation": "nationality of the director",
                "atomic_question_template": "What is the nationality of that director?",
            },
        ],
        "operator": {
            "type": "COMPARE_SAME",
            "inputs": ["nationality_r1", "nationality_r2"],
            "output": "boolean",
        },
    }
    return f"""
Label the fixed AST skeleton edges.

Original question:
{question}

Selected paths:
{json.dumps(selected_paths, ensure_ascii=False, indent=2)}

AST skeleton:
{json.dumps(ast_skeleton, ensure_ascii=False, indent=2)}

Problem Frame:
{json.dumps(problem_frame, ensure_ascii=False, indent=2)}

Rules:
1. Give each existing AST edge a relation label or relation hint.
2. You may optionally provide atomic_question_template for that one edge.
3. Confirm the operator type and inputs from the AST skeleton and Problem Frame.
4. Do not add AST nodes.
5. Do not delete AST nodes.
6. Do not add AST edges.
7. Do not delete AST edges.
8. Do not merge branch-specific clones.
9. Do not create shortcut edges. For example, do not add AlphaGo -> university unless
   those nodes are adjacent in the selected path and skeleton.
10. Do not create selected-path-external query nodes.
11. Do not generate executable DAG steps or final atomic questions.

Output one item for every skeleton edge and only those edges.
Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()

MASK_SPAN_EXTRACTION_SYSTEM = """
You are implementing DEPO Step 1: selective mask span extraction before CoreNLP dependency parsing.

Your task is parser-protection span detection only.
You must find contiguous surface spans that should be collapsed into one placeholder token before dependency parsing.

This is not named entity recognition in general.
This is not anchor extraction.
This is not type-variable extraction.
This is not relation extraction.
This is not AST construction.
This is not question decomposition.

The goal of masking is to protect CoreNLP from incorrectly splitting complex multi-token names, titles, and compact compound noun phrases.

Most important requirement:
Every mask span must receive a semantic_type_hint that allows the downstream placeholder to have the same syntactic/semantic type as the original span.
For example:
- multi-token person names -> Person, so they can become PersonA, PersonB;
- film/movie titles -> Film, so they can become FilmA, FilmB;
- book titles -> Book;
- organizations/companies -> Organization or Company;
- universities/schools -> University or Institution;
- cities/countries/regions/locations -> City, Country, Region, or Location;
- compact type phrases headed by "company" -> Company.

Only mask spans with at least two lexical tokens, except when a named title contains punctuation, digits, subtitles, or a parenthetical qualifier that makes it parser-fragile.
Do not mask ordinary single-token entities.

Return valid JSON only.
""".strip()


def build_mask_span_extraction_prompt(question: str) -> str:
    schema = {
        "mask_spans": [
            {
                "text": "exact contiguous span copied from the original question",
                "start_char": 0,
                "end_char": 15,
                "kind_hint": "entity | type_variable",
                "semantic_type_hint": "Person | Film | Book | Song | Album | Series | Work | Company | Organization | University | Institution | City | Country | Region | Location | Event | Product | Entity",
                "reason": "brief parser-protection reason"
            }
        ]
    }

    return f"""
Identify only the spans that should be masked before CoreNLP parsing.

Question:
{question}

Task definition:
Return contiguous surface spans that should become one placeholder token before dependency parsing.
The placeholder must preserve the syntactic/semantic type of the original span.

This step protects complex surface spans only.
Do not extract anchors, answer variables, implicit variables, operators, relations, AST nodes, or subquestions.

Primary targets:

A. Multi-token proper names
Return every continuous multi-token proper name:
- person names: "John Middleton Murry", "Gideon Johnson Pillow", "Holm Jølsen";
- organization/company/institution names;
- place names;
- event/product names;
- named works and titles.

B. Named works and titles
Return full title-like spans, including subtitles, numbers, punctuation, and parenthetical disambiguators:
- "Wrong Turn 5: Bloodlines"
- "Dark River (2017 Film)"
- "Harry Potter and the Goblet of Fire"

When a title appears after a shared type word such as film, movie, book, song, album, or series, use that word only to infer semantic_type_hint.
Do not include the shared type word in the span.

Correct:
- films Wrong Turn 5: Bloodlines and Dark River (2017 Film)
  -> "Wrong Turn 5: Bloodlines" with semantic_type_hint "Film"
  -> "Dark River (2017 Film)" with semantic_type_hint "Film"

Incorrect:
- "films Wrong Turn 5: Bloodlines"
- "Wrong Turn 5: Bloodlines and Dark River (2017 Film)"
- "directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film)"

C. Possessor names in possessive constructions
For possessive person/name constructions, mask only the possessor name.
Do not include "'s", the possessed noun, or the relation phrase.

Correct:
- "John Middleton Murry's wife"
  -> "John Middleton Murry" with semantic_type_hint "Person"

Incorrect:
- "John Middleton Murry's"
- "John Middleton Murry's wife"

D. Coordinated or compared names
When names are coordinated or compared with and/or, return each eligible name separately.
Use the same semantic_type_hint for same-type alternatives.

Correct:
- "Gideon Johnson Pillow or Holm Jølsen"
  -> "Gideon Johnson Pillow" with semantic_type_hint "Person"
  -> "Holm Jølsen" with semantic_type_hint "Person"

Incorrect:
- "Gideon Johnson Pillow or Holm Jølsen"

E. Compact compound type phrases
Return compact multi-word compound noun phrases only when masking them helps CoreNLP keep one class/type expression intact.
The phrase must have a noun head and at least one essential classifying modifier.

Correct:
- "artificial intelligence company" -> kind_hint "type_variable", semantic_type_hint "Company"
- "chief operating officer" -> kind_hint "type_variable", semantic_type_hint "Person" or "Role"
- "research institute" -> kind_hint "type_variable", semantic_type_hint "Institution"

Incorrect:
- "the CEO"
- "which university"
- "the artificial intelligence company that developed AlphaGo"
- "CEO of the artificial intelligence company"
- "company that developed AlphaGo"

Boundary rules:
1. The span must be a minimal contiguous substring of the original question.
2. start_char is inclusive; end_char is exclusive.
3. question[start_char:end_char] must exactly equal text.
4. Exclude leading determiners such as the, a, an unless they are part of an official title.
5. Exclude wh words such as who, whom, whose, what, which, where, when, how.
6. Exclude relation words, prepositions, relative clauses, verbs, auxiliaries, comparison words, and coordination words unless they are part of an official title.
7. Do not include "and" or "or" between two separate names unless the conjunction is part of one official title.
8. Do not include possessive "'s" unless it is part of an official name, which is rare.
9. Do not include appositive type words outside the title unless they are inside a parenthetical disambiguator, as in "Dark River (2017 Film)".
10. Do not mask a larger phrase when a smaller name/title span is sufficient.

Negative rules:
Do not mask:
- wh answer phrases: "which university", "what country", "whose wife";
- simple type variables: director, CEO, university, city, country, nationality, age, population, actor;
- determiner + single noun phrases: "the university", "the city", "the CEO";
- relation phrases: "director of", "wife of", "CEO of", "born in", "located in";
- full clauses or relative clauses;
- operator/comparison cues: same, different, later, earlier, older, younger, largest, first, both;
- single-token entities unless they are part of a larger multi-token span;
- spans merely because they are semantically important.

Semantic type hint rules:
- Use Person for human names in contexts involving who, whom, whose, born, died, wife, husband, actor, director, CEO, author, player, president, older, younger.
- Use Film for titles introduced by film, films, movie, movies, or parentheticals like "(2017 Film)".
- Use Book, Song, Album, Series, or Work for other named works when locally indicated.
- Use Company or Organization for companies/organizations.
- Use University or Institution for universities/schools/institutions.
- Use City, Country, Region, or Location for places.
- Use Entity only when the local context does not support a more specific type.
- In a coordinated or compared group, same-type alternatives must receive the same semantic_type_hint.

Expected behavior examples:

Example 1:
Question: Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?
Return:
- text: "Wrong Turn 5: Bloodlines", start_char: 27, end_char: 51, kind_hint: "entity", semantic_type_hint: "Film"
- text: "Dark River (2017 Film)", start_char: 56, end_char: 78, kind_hint: "entity", semantic_type_hint: "Film"

Example 2:
Question: Why did John Middleton Murry's wife die?
Return:
- text: "John Middleton Murry", start_char: 8, end_char: 28, kind_hint: "entity", semantic_type_hint: "Person"

Example 3:
Question: Who was born later, Gideon Johnson Pillow or Holm Jølsen?
Return:
- text: "Gideon Johnson Pillow", start_char: 20, end_char: 41, kind_hint: "entity", semantic_type_hint: "Person"
- text: "Holm Jølsen", start_char: 45, end_char: 56, kind_hint: "entity", semantic_type_hint: "Person"

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


SEMANTIC_QUESTION_NORMALIZATION_SYSTEM = """
You are implementing DEPO Step 0: parser-oriented question normalization before CoreNLP dependency parsing.

Your task is lossless syntactic normalization only.
The default action is to return the original question unchanged.

Normalize only when a minimal surface rewrite is likely to improve dependency parsing without changing the semantic graph that later DEPO steps should build.

You must not perform semantic enrichment.
Do not add implicit type variables, value slots, attributes, operators, AST nodes, answer variables, subquestions, aliases, entity types, or world-knowledge facts.

In particular, do not rewrite comparative, superlative, ordinal, temporal, location, identity, membership, or predicate cues into new attribute nouns.
For example, do not rewrite "older" as "age", "released first" as "release date", "where" as "location", or "who" as "person" unless that noun already appears explicitly in the original question.

Preserve all named entities, placeholders, wh focus, comparison cues, logical cues, and coreference cues exactly.
Do not replace this/that/its/his/her/their references with long antecedent phrases.
Coreference resolution belongs to the semantic AST step, not this step.

Allowed edits are limited to:
- adding or adjusting light function words needed for grammaticality;
- converting a relation-bearing possessive common-noun phrase into an of-prepositional phrase when it makes the dependency head clearer;
- expanding clear surface ellipsis or shared coordination only using semantic material already present in the original question;
- lightly reordering words to form a grammatical single question when the original is syntactically malformed.

Forbidden edits:
- do not answer the question;
- do not split the question into multiple questions;
- do not decompose the question;
- do not infer missing facts or entity types;
- do not introduce new content nouns or semantic attributes;
- do not remove, rename, paraphrase, or duplicate named entities;
- do not change the answer focus;
- do not change comparison, set, temporal, location, or logical conditions;
- do not output reasoning, ASTs, graphs, operators, or execution plans.

If a safe minimal rewrite is not obvious, return the original question unchanged.

Return valid JSON only.
""".strip()


def build_semantic_question_normalization_prompt(
    question: str,
    placeholders: list[str] | None = None,
) -> str:
    placeholders = placeholders or []
    schema = {
        "normalized_question": "the original question if no safe minimal syntactic rewrite is needed",
        "changed": False,
        "rewrite_type": "identity | possessive_to_of | ellipsis_expansion | coordination_clarification | grammar_repair",
        "added_type_variables": [],
        "reason": "brief reason for the decision"
    }

    return f"""
Decide whether the question needs minimal parser-oriented normalization.

Original question:
{question}

Placeholder tokens that must be preserved exactly:
{json.dumps(placeholders, ensure_ascii=False, indent=2)}

Decision procedure:
1. Start with identity: assume the original question should be returned unchanged.
2. Rewrite only if there is a clear syntactic risk for dependency parsing.
3. The rewrite must remain exactly one natural English question.
4. The rewrite must preserve the same answer focus and all original constraints.
5. Use only semantic content already present in the original question.
6. You may add only light grammatical/function words when needed.
7. Do not introduce new content nouns, attributes, value slots, entity types, or implicit variables.
8. Do not turn cues into attributes:
   - "older/younger" must not become "age" or "birth date";
   - "released first/earlier" must not become "release date";
   - "where" must not become "location";
   - "who" must not become "person";
   - "largest/highest/longest" must not become "population/height/length" unless that noun already appears.
9. Preserve coreference expressions such as "this university", "that city", "its director", "his mother", and "their nationality".
10. Preserve every named entity and every placeholder exactly as written.
11. If the rewrite would require semantic inference, return the original question unchanged.

Allowed rewrite examples:
- "When did Lothair II's mother die?"
  -> "When did the mother of Lothair II die?"
  This is allowed because it only changes a possessive relation into an of-relation and adds no new semantic variable.

Forbidden rewrite examples:
- "Who is older, Alice or Bob?"
  -> "Which person has the greater age, Alice or Bob?"
  This is forbidden because it adds "person" and "age".

- "Which film was released first, FilmA or FilmB?"
  -> "Which film has the earliest release date, FilmA or FilmB?"
  This is forbidden because it adds "release date".

- "In which city is this university located?"
  -> "In which city is the university that the CEO graduated from located?"
  This is forbidden because it resolves coreference by duplicating an antecedent.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


ANCHOR_SELECTION_SYSTEM = """
You are implementing DEPO Step 4: explicit anchor selection.

Your task is to select a minimal sufficient set of explicit semantic node endpoints from the provided candidate anchor set.

You must select anchors only from the provided candidate anchor set.
Do not invent node_id values, entities, variables, attributes, relations, operators, AST nodes, or subquestions.

An anchor is an explicit candidate node that must survive as a node in the later semantic graph.
An anchor is not a relation edge, not an operator cue, not a dependency bridge, and not a word that is merely syntactically useful.

Allowed anchor_kind values are exactly:
- entity
- type_variable

Select anchors in these cases:

1. Fixed named inputs
Select explicit named people, organizations, places, works, products, events, or other fixed named entities that are given as inputs to the question.
Single-token named entities may be selected if they are explicit fixed inputs.

2. Explicit answer variables
Select explicit common-noun variables that represent what the question asks for, such as university, city, country, director, actor, company, date, nationality, population, or similar, when they are truly the answer target.

3. Intermediate relation-chain variables
Select explicit roles/classes that must be solved between a known input and the final answer, such as mother, wife, CEO, director, author, founder, company, university, city, or country.

4. Explicit attributes or value slots
Select explicit attributes/measures only when the attribute noun itself appears in the candidate text, such as nationality, population, date of death, release date, height, length, or age.

Use anchor_kind=entity for proper named entities and fixed named inputs.
Use anchor_kind=type_variable for common-noun roles, classes, answer variables, intermediate variables, and explicit attributes.

Do not select:
- implicit variables or value slots;
- relation-only verbs or predicates such as born, died, developed, graduated, located, released, directed, authored;
- operator/comparison/logical cues such as same, different, older, younger, later, earlier, largest, first, last, both, either, and, or;
- wh words or generic answer labels such as who, what, where, when, person, thing, place, time, someone, something, somewhere, unless the exact common noun is independently bound in a relation chain;
- prepositions, auxiliaries, determiners, relativizers, punctuation, or function words;
- appositive/context type labels that merely introduce a nearby named entity, such as film in "film Titanic", city in "city Paris", or actor in "actor Tom Hanks", unless that type word is itself the answer or an intermediate variable;
- broad context labels when a more specific entity anchor already represents the object, such as films before two film titles.

For repeated or coreferent mentions:
Select multiple mentions only if each mention anchors a different relation clause needed by the question.
Do not treat coreferent mentions as different semantic variables. The semantic AST step will merge them.

For nested phrases, possessives, of-phrases, and relative clauses:
Select the explicit endpoint nodes needed to express the relation chain.
Do not select the relation phrase itself.

If uncertain, prefer the smaller set of anchors that still preserves the reasoning chain.

Return valid JSON only.
""".strip()


def build_anchor_selection_prompt(
    original_question: str,
    restored_graph_node_candidates: list[dict[str, object]],
) -> str:
    schema = {
        "selected_anchors": [
            {
                "node_id": "8",
                "anchor_kind": "entity",
                "text": "Wrong Turn 5: Bloodlines",
                "reason": "role=fixed_named_input; explicit film title used as a branch input"
            },
            {
                "node_id": "13",
                "anchor_kind": "type_variable",
                "text": "nationality",
                "reason": "role=explicit_attribute; nationality is the explicit compared attribute"
            }
        ]
    }

    return f"""
Select explicit anchors for the semantic-normalized question.

Semantic-Normalized Question:
{original_question}

Candidate anchor set:
{json.dumps(restored_graph_node_candidates, ensure_ascii=False, indent=2)}

Output requirements:
- Output only node_id values from the candidate anchor set.
- The text field must exactly copy the candidate text for that node_id.
- Use only anchor_kind "entity" or "type_variable".
- Do not add fields outside the requested JSON shape.
- In the reason, briefly state the semantic role, for example:
  role=fixed_named_input
  role=answer_variable
  role=intermediate_variable
  role=explicit_attribute
  role=coreferent_clause_anchor

Decision test:
Select a candidate only if it is an explicit semantic node endpoint needed to preserve the question's reasoning chain.

A candidate should be selected if it is one of:
1. a fixed named entity input;
2. an explicit answer variable;
3. an explicit intermediate role/class variable;
4. an explicit attribute or value slot that appears in the question;
5. a repeated/coreferent mention that anchors a different relation clause.

A candidate should not be selected if it is only:
1. a relation word or predicate;
2. an operator/comparison/logical cue;
3. a wh word or generic answer label;
4. a syntactic bridge;
5. a broad type label introducing a nearby named entity;
6. an implied attribute not explicitly present as candidate text.

Important distinctions:

- Named entity input:
  "Gideon Johnson Pillow" and "Holm Jølsen" should be selected as entity anchors.

- Explicit intermediate variable:
  In "John Middleton Murry's wife", select "John Middleton Murry" and "wife".
  Do not select "die" or "why".

- Explicit compared attribute:
  In "same nationality", select "nationality".
  Do not select "same".

- Implied value slot:
  In "Who was born later, A or B?", select A and B.
  Do not select "birth date" unless "birth date" is an actual candidate.

- Context type label:
  In "films Wrong Turn 5: Bloodlines and Dark River (2017 Film)", select the two film titles.
  Do not select "films" if the film titles themselves are selected.

- Appositive type label:
  In "actor Tom Hanks", select "Tom Hanks".
  Do not select "actor" unless the question asks for or constrains an unknown actor.

- Coreference:
  In "Which university did X graduate from and in which city is this university located?",
  select both university mentions only if they appear as separate candidates anchoring different clauses.
  The reason should indicate they are coreferent mentions of the same variable.
  Do not treat them as two different universities.

Anchor kind rules:
- Use entity for proper names, named works, named organizations, named places, products, or events.
- Use type_variable for common-noun roles/classes/variables/attributes.
- Do not use entity for common nouns merely because they are important.
- Do not use type_variable for implicit variables that are not explicit candidates.

Minimality rule:
Do not select every noun.
Select the minimal sufficient set of explicit node endpoints needed to reconstruct the semantic chain.

If no candidate should be selected, return an empty selected_anchors list.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


SEMANTIC_AST_OPTIMIZATION_SYSTEM = """
You are implementing DEPO Step 9: final semantic AST construction.

Your task is to convert the selected anchors and the restored anchor-connected evidence graph into a clean directed semantic reasoning AST.

The anchor-connected graph is only syntactic/structural evidence.
It is not the final AST.
Use the original question semantics as the highest-priority source of truth.
Use selected anchors and the restored graph only as grounding evidence.

The AST must be a directed reasoning DAG.
Each edge must represent exactly one semantic lookup relation from a known or previously solvable source node to the next target node to solve.

Do not copy dependency edges directly.
Do not output dependency paths as semantic edges.
Do not output subquestions.
Do not answer the question.

Node contract:
- A node is a semantic endpoint: named entity, explicit type variable, role, class, answer variable, attribute/value slot, or implicit value slot.
- Node labels must be clean natural-language endpoint labels, such as "director", "death date", "nationality", "university", "city", or a named entity.
- Do not put relation phrases into node labels.
- For example, use label "nationality", not "nationality of director".
- Put relation phrases only in edge.relation_hint.

Edge contract:
- Each edge must be one-hop.
- Each edge source is the known entity or previously solved variable.
- Each edge target is the next variable/value to retrieve.
- Do not merge multiple reasoning hops into one edge.
- Do not use coreference, apposition, same-as, mention-of, or syntactic attachment as reasoning edges.

Coreference contract:
- Merge repeated mentions that refer to the same semantic variable.
- Merge this/that/its/his/her/their references with their true antecedent.
- Merge wh variables with later mentions when they refer to the same variable, such as "which university" and "this university".
- Do not merge same-label variables across independent branches. For example, two directors of two different films must remain separate nodes.

Implicit variable contract:
This is the only step that may create implicit type variables or implicit value slots.
Create an implicit node only when the original question explicitly licenses it through a wh-focus, predicate cue, attribute cue, comparison cue, ranking cue, set cue, or logical cue.
Do not create implicit nodes from world knowledge or from generic plausibility.
Every implicit node must have:
- kind = "implicit_type_variable";
- cue_text;
- grounding_text;
- relation_hint;
- expected_value_slot when applicable;
- branch_of when it belongs to one branch of a comparison/set/logical operation.

Operator contract:
Choose exactly one primary_operator.operator from the allowed operator set.
NONE is the default.

Use NONE for ordinary chain-shaped lookup questions.
Predicates, events, and attributes such as die, born, graduate, located, released, founded, developed, directed, authored, and discovered are normally semantic edges, not operators.

Use a non-NONE operator only when the final answer requires an operation over multiple branch results, multiple values, candidate entities, sets, or propositions.
Cue words alone must never force an operator.
Decide from the full question semantics.

For non-NONE operators:
- primary_operator.inputs must be terminal value nodes or proposition nodes that can be obtained from the AST edges.
- Do not attach an operator directly to an intermediate entity if the question actually compares or ranks one of its attributes.
- If the compared/ranked/combined value is implicit, create the required value-slot node first, then use that node as the operator input.

For operator NONE:
- primary_operator.inputs should be empty unless the existing schema requires otherwise.
- Model the requested answer entirely as directed semantic edges.

Direction contract:
Orient edges in the order needed to solve the question.
If explicit named inputs exist, branches usually start from those inputs and move toward answer variables or operator inputs.
If there is no fixed named input, start from the candidate/answer class and move toward the attribute or value required by the question.

Preserve selected anchors unless they are redundant coreferent mentions or appositive/context labels that should be merged into a canonical node.
Do not invent named entities absent from the original question or mask restore information.

Return valid JSON only.
""".strip()


def build_semantic_ast_optimization_prompt(
    original_question: str,
    replacement: dict[str, object],
    selected_anchors: list[dict[str, object]],
    restored_anchor_connected_subgraph: dict[str, object],
    allowed_operators: list[str],
    validation_feedback: list[str] | None = None,
    previous_ast: dict[str, object] | None = None,
) -> str:
    schema = {
        "status": "ok",
        "primary_operator": {
            "operator": "NONE",
            "cue_text": "",
            "inputs": [],
            "output": "answer",
            "explanation": "Use NONE for a serial chain lookup; use a non-NONE operator only when the final answer requires comparison, ranking, set, or logical composition."
        },
        "nodes": [
            {
                "id": "film_1",
                "label": "Madame La Presidente",
                "kind": "entity",
                "semantic_type": "Film",
                "source": "selected_anchor",
                "source_graph_nodes": ["8"],
                "source_token_indices": [8],
                "grounding_text": "Madame La Presidente",
                "cue_text": "",
                "branch_of": "",
                "expected_value_slot": "",
                "relation_hint": ""
            },
            {
                "id": "director_1",
                "label": "director",
                "kind": "type_variable",
                "semantic_type": "Person",
                "source": "selected_anchor",
                "source_graph_nodes": ["4"],
                "source_token_indices": [4],
                "grounding_text": "director",
                "cue_text": "",
                "branch_of": "",
                "expected_value_slot": "",
                "relation_hint": ""
            },
            {
                "id": "death_date_1",
                "label": "death date",
                "kind": "implicit_type_variable",
                "semantic_type": "Date",
                "source": "implicit_from_question",
                "source_graph_nodes": [],
                "source_token_indices": [],
                "grounding_text": "die",
                "cue_text": "When ... die",
                "branch_of": "director_1",
                "expected_value_slot": "death_date",
                "relation_hint": "date of death"
            }
        ],
        "edges": [
            {
                "source": "film_1",
                "target": "director_1",
                "edge_type": "attribute",
                "relation_hint": "director of film",
                "support_path": ["Madame La Presidente", "director"],
                "support_dependency_relations": []
            },
            {
                "source": "director_1",
                "target": "death_date_1",
                "edge_type": "attribute",
                "relation_hint": "date of death",
                "support_path": ["director", "die"],
                "support_dependency_relations": []
            }
        ]
    }

    return f"""
Construct the final directed semantic AST for the original question.

Original question:
{original_question}

Mask restore information:
{json.dumps(replacement, ensure_ascii=False, indent=2)}

Selected explicit anchors:
{json.dumps(selected_anchors, ensure_ascii=False, indent=2)}

Restored anchor-connected subgraph:
{json.dumps(restored_anchor_connected_subgraph, ensure_ascii=False, indent=2)}

Allowed primary operators:
{json.dumps(allowed_operators, ensure_ascii=False)}

Previous AST that failed validation:
{json.dumps(previous_ast or {}, ensure_ascii=False, indent=2)}

Validation feedback to fix:
{json.dumps(validation_feedback or [], ensure_ascii=False, indent=2)}

Your goal:
Build the clean semantic reasoning skeleton of the original question.

The output AST will later be compiled into atomic subquestions.
Therefore, each AST edge must correspond to one atomic lookup relation.

High-level decision procedure:

1. Identify the semantic endpoints.
   Use selected anchors as explicit endpoint candidates.
   Add implicit value-slot nodes only when the original question explicitly requires a value not already represented by an anchor.

2. Merge coreferent mentions.
   Merge this/that/its/his/her/their mentions with their true antecedent.
   Merge repeated wh-variable mentions with later mentions of the same variable.
   Do not create same-as, coreference, apposition, or mention-of edges.

3. Keep independent branches separate.
   Do not merge two nodes merely because they have the same label.
   In parallel questions, use branch-specific ids such as director_1/director_2 or nationality_1/nationality_2.

4. Create one-hop directed edges.
   Each edge must represent one semantic relation from a known or previously solvable source to the next target.
   Do not put two or more reasoning hops into one edge.
   Do not create relation nodes when an edge.relation_hint is enough.

5. Choose the primary operator.
   Choose exactly one operator from the allowed set.
   NONE is the default.
   Use NONE for serial chain lookup questions.
   Use a non-NONE operator only when the final answer requires comparison, ranking, set composition, or logical composition over multiple independently obtained values/propositions.

Rules for implicit nodes:

- An implicit node is allowed when the original question asks for a value through a predicate or cue rather than an explicit noun.
- Examples of licensed implicit values:
  - "When did X die?" licenses a death-date value.
  - "Where was X born?" licenses a birthplace/location value.
  - "Who was born later, A or B?" licenses birth-date values for both branches.
  - "Do A and B have the same nationality?" licenses nationality values if nationality nodes are not already explicit branch endpoints.
- These are examples, not a closed keyword list.
- Infer the needed value slot from the full question semantics.
- Do not add implicit nodes that are only plausible from world knowledge.
- Do not add extra attributes that the question does not ask for.

Rules for primary_operator:

- Use NONE when the question can be answered by following a single directed chain.
  Example:
  "When did the director of Madame La Presidente die?"
  AST:
  film_1 -> director_1
  director_1 -> death_date_1
  operator = NONE

- Use COMPARE_SAME or COMPARE_DIFF only when the final answer asks whether two or more branch values are the same or different.
  The operator inputs must be the actual compared value nodes, not merely the branch entities.

- Use COMPARE_GREATER or COMPARE_LESS only when the final answer asks which of two values is greater/less, earlier/later, older/younger, or otherwise ordered.
  The operator inputs must be the ordered value nodes.

- Use ARGMAX or ARGMIN only when the final answer asks for the candidate with the maximum/minimum value among a candidate set.
  The operator input should be the value being maximized/minimized, with candidate linkage preserved through branch_of or edges.

- Use INTERSECTION, UNION, or DIFFERENCE only for set-valued questions.
  Do not use them for ordinary relation chains.

- Use LOGICAL_AND or LOGICAL_OR only when the final answer combines propositions, not merely because the surface question contains "and" or "or".

Cue words are evidence, not triggers.
A word like same, first, later, older, largest, born, died, released, founded, or located does not automatically determine the operator.
Always decide from the final answer requirement.

Direction rules:

- Source should be a known entity, fixed input, candidate class, or previously solvable node.
- Target should be the next variable/value to retrieve.
- For named-input chains, start from the named input.
- For candidate-selection questions, start from the candidate type and move to the attribute/value used for filtering, comparison, or ranking.

Examples:

Example 1:
Question:
When did the director of Madame La Presidente die?

Correct:
operator = NONE
film_1: Madame La Presidente
director_1: director
death_date_1: death date
film_1 -> director_1 (director of film)
director_1 -> death_date_1 (date of death)

Incorrect:
operator = COMPARE_DIFF
director_1 -> death_date_1 -> operator

Example 2:
Question:
Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?

Correct:
film_1 -> director_1
director_1 -> nationality_1
film_2 -> director_2
director_2 -> nationality_2
primary_operator = COMPARE_SAME
primary_operator.inputs = ["nationality_1", "nationality_2"]

Incorrect:
film_1 -> director_1 -> COMPARE_SAME
film_2 -> director_2 -> COMPARE_SAME

Example 3:
Question:
Which university did the CEO of the company that developed AlphaGo graduate from and in which city is this university located?

Correct:
AlphaGo -> company_1
company_1 -> ceo_1
ceo_1 -> university_1
university_1 -> city_1
operator = NONE
There must be one university node.
Do not create a coreference edge between "which university" and "this university".

Example 4:
Question:
Who was born later, Gideon Johnson Pillow or Holm Jølsen?

Correct:
person_1 -> birth_date_1
person_2 -> birth_date_2
primary_operator = COMPARE_GREATER or COMPARE_LESS according to the system's operator semantics
primary_operator.inputs = ["birth_date_1", "birth_date_2"]

Incorrect:
person_1 -> COMPARE_GREATER
person_2 -> COMPARE_GREATER

Node id rules:
- Use stable snake_case ids.
- Use numeric suffixes for branch-specific nodes: director_1, director_2, nationality_1, nationality_2.
- The label should be clean natural language.
- The id may be technical; the label must not be technical.

Output quality checklist:
- Every selected anchor is represented or intentionally merged.
- Every edge is one-hop.
- No relation phrase is used as a node label.
- No coreference/apposition/same-as edge is output.
- NONE is used for serial chain lookups.
- Non-NONE operators consume terminal value/proposition nodes.
- Implicit nodes are grounded in explicit question cues.
- No atomic subquestions are generated.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


ATOMIC_SUBQUESTION_GENERATION_SYSTEM = """
You are implementing DEPO Step 8: LLM-based atomic subquestion generation.
Generate exactly one atomic subquestion for the provided one-hop semantic AST edge, or exactly one final operator question for the provided primary operator.
Use the original question and semantic AST context, but do not combine multiple AST edges into a multi-hop question.
The input edge is already oriented as source/bound node -> target node to solve.
Internal variables such as X1, X2, V1, and VAR_* are implementation details. Never expose them in the generated question.
If the source is bound to an answer variable, use the provided natural-language dependency description instead.
For ordinary attribute edges, do not include operator cue words such as same, older, largest, before, or after.
For operator steps, preserve the concrete cue from the original question and name the natural-language candidates or compared facts explicitly.
The DAG inputs, outputs, dependencies, edges, and candidate bindings are produced by code; only write the natural-language question text.
Return valid JSON only.
""".strip()


ATOMIC_PLAN_STEP_SURFACE_SYSTEM = """
You are implementing DEPO Step 10: atomic subquestion surface realization.

The semantic AST has already been compiled by code into an ordered atomic subquestion DAG.
The DAG order, dependencies, current step, inputs, outputs, and operator placement are already decided by code.

Your only task is to write one natural-language atomic question for the single provided execution-plan step.

This step is not responsible for:
- deciding the DAG topology;
- deciding dependencies;
- decomposing the original question again;
- merging multiple future steps;
- creating new nodes or variables;
- choosing operators;
- answering the question;
- generating the final answer.

For step_type=edge:
Generate exactly one atomic question whose target is the current one-hop AST edge target.

The current step determines what is being asked.
The resolved_known_subject determines who or what the question is about.

You may use resolved_known_subject to make the question natural and self-contained.
If resolved_known_subject contains an upstream path, you may use it directly as the subject phrase of the current question.

For example, if the current step asks for the death date of a director, and resolved_known_subject is "the director of Madame La Presidente", then a valid question is:
"When did the director of Madame La Presidente die?"

This is allowed because the current target is still only the death date.
Do not force vague wording such as "the previously identified director" unless it is the most natural wording.

However, do not add any downstream relation, final-answer target, comparison, ranking, set operation, or extra condition that is not part of the current step.

For ordinary edge steps:
- Ask only for the target of the current edge.
- Use relation_hint only as wording guidance.
- Use resolved_known_subject as the known subject.
- Do not include operator cues such as same, different, older, younger, earlier, later, largest, highest, first, last, both, either, whether, and, or, unless they are literally part of a named entity/title.
- Do not ask the full original question unless the current step itself corresponds to that full question.
- Do not create a new hidden hop that is not represented in the current step.
- Do not ask for downstream nodes.

For step_type=operator:
Generate exactly one final operator question.
Use the original question, step.operator, step.cue_text, input_descriptions, and candidate metadata.
Preserve the concrete comparison, ranking, set, or logical meaning from the original question.
Do not turn the operator step into another lookup question.

Internal variables such as X1, X2, V1, VAR_*, answer_variable, source_id, and target_id are implementation IDs.
They must never appear in the generated question.

The external contract is semantic dependency between atomic questions, not variable-name binding.

Return valid JSON only.
""".strip()


def build_atomic_plan_step_surface_prompt(
    original_question: str,
    plan_step: dict[str, object],
    resolved_known_subject: str = "",
    input_descriptions: dict[str, str] | None = None,
) -> str:
    input_descriptions = input_descriptions or {}

    schema = {
        "question": "one natural-language atomic question for this single execution-plan step",
        "explanation": "briefly explain why this question asks only for the current step target"
    }

    return f"""
Generate one atomic subquestion for this single already-compiled execution-plan step.

Original question:
{original_question}

Execution-plan step:
{json.dumps(plan_step, ensure_ascii=False, indent=2)}

Resolved known subject:
{resolved_known_subject}

Input descriptions:
{json.dumps(input_descriptions, ensure_ascii=False, indent=2)}

Core rule:
Generate exactly one natural-language question for the current execution-plan step.
The current step determines what is being asked.
The resolved_known_subject determines who or what the question is about.

The original question is only wording context.
Do not use the original question to add extra hops, downstream targets, extra constraints, or a final-answer request.

For step_type=edge:
1. Generate one atomic question for the current directed AST edge only.
2. The known/source side is resolved_known_subject.
3. The asked/target side is step.ask.
4. Use step.relation_hint only to choose natural wording.
5. The question may be self-contained.
6. The question may reuse an upstream path inside resolved_known_subject.
7. The question must not ask for any downstream target.
8. The question must not include operator/comparison/logical meaning unless the current step is an operator step.

Important:
Using resolved_known_subject is allowed even if it contains an upstream relation chain.

Example:
If the current step is:
director_1 -> death_date_1, relation_hint="date of death"

and resolved_known_subject is:
"the director of Madame La Presidente"

Then this is valid:
"When did the director of Madame La Presidente die?"

This is valid because the current question only asks for the death date.
It does not ask for a downstream value or final comparison.

Do not expose internal variables such as X1, X2, V1, VAR_*, answer_variable, source_id, or target_id.

For ordinary edge steps:
- Ask only for the target of the current edge.
- Do not add comparison/operator cues such as same, different, older, younger, earlier, later, largest, highest, first, last, both, either, whether, and, or.
- Do not ask whether two things are equal, different, before, after, larger, smaller, or ranked.
- Do not generate a final answer question.
- Do not create new entities, variables, attributes, or relation hops.
- Do not include downstream relations from later steps.

For step_type=operator:
1. Generate the final comparison/ranking/set/logical question.
2. Use step.operator, step.cue_text, input_descriptions, and candidate metadata.
3. Preserve the original concrete operator meaning.
4. Do not mention internal variables.
5. Do not generate another lookup question.

Positive examples:

Example A:
Original question:
When did the director of Madame La Presidente die?

Current edge:
film_1 -> director_1, relation_hint="director of film"

Resolved known subject:
Madame La Presidente

Output:
"Who is the director of Madame La Presidente?"

Example B:
Original question:
When did the director of Madame La Presidente die?

Current edge:
director_1 -> death_date_1, relation_hint="date of death"

Resolved known subject:
the director of Madame La Presidente

Output:
"When did the director of Madame La Presidente die?"

Example C:
Original question:
Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?

Current ordinary edge:
film_1 -> director_1, relation_hint="director of film"

Resolved known subject:
Wrong Turn 5: Bloodlines

Output:
"Who is the director of Wrong Turn 5: Bloodlines?"

Example D:
Original question:
Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?

Current ordinary edge:
director_1 -> nationality_1, relation_hint="nationality of director"

Resolved known subject:
the director of Wrong Turn 5: Bloodlines

Output:
"What is the nationality of the director of Wrong Turn 5: Bloodlines?"

Example E:
Original question:
Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?

Current operator step:
COMPARE_SAME over nationality_1 and nationality_2

Input descriptions:
nationality_1: the nationality of the director of Wrong Turn 5: Bloodlines
nationality_2: the nationality of the director of Dark River (2017 Film)

Output:
"Do the two directors have the same nationality?"

Negative examples:

Do not turn this ordinary edge:
film_1 -> director_1

into:
"Do the directors of Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?"

because that asks the final operator question, not the current edge target.

Do not turn this ordinary edge:
director_1 -> nationality_1

into:
"Does the director of Wrong Turn 5: Bloodlines have the same nationality as the director of Dark River (2017 Film)?"

because that adds comparison logic that belongs only to the operator step.

Do not output internal-variable wording such as:
"What is the nationality of X1?"
"Find V2 from director_1."

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


def build_atomic_subquestion_generation_prompt(
    original_question: str,
    semantic_ast: dict[str, object],
    current_edge: dict[str, object],
    source_node: dict[str, object] | None,
    target_node: dict[str, object] | None,
    primary_operator: dict[str, object],
) -> str:
    schema = {
        "question": "Who is the director of Ten9Eight: Shoot For The Moon?",
        "explanation": "This asks only for the target node of the one-hop edge.",
    }
    return f"""
Generate one atomic subquestion for the current semantic item.

Original question:
{original_question}

Final semantic AST:
{json.dumps(semantic_ast, ensure_ascii=False, indent=2)}

Current one-hop edge or operator step:
{json.dumps(current_edge, ensure_ascii=False, indent=2)}

Source node:
{json.dumps(source_node, ensure_ascii=False, indent=2)}

Target node:
{json.dumps(target_node, ensure_ascii=False, indent=2)}

Primary operator:
{json.dumps(primary_operator, ensure_ascii=False, indent=2)}

Rules:
- For a directed one-hop edge, generate exactly one question for that edge only.
- Treat current_edge.source_display as the known subject. Treat current_edge.target_label as the value to ask for.
- Internal variables such as X1, X2, V1, and VAR_* are implementation details and must not appear in the generated question.
- When current_edge.source_display is a variable, use the surrounding semantic context to produce the most natural descriptive question without inventing the answer.
- The answer to this subquestion will be current_edge.answer_variable.
- Do not merge this edge with another edge.
- Do not include same/older/largest/comparative/superlative cue words in ordinary attribute questions.
- For an implicit variable edge such as actor -> age, ask a normal attribute question such as "What is the age of the actor?"
- For an operator step, generate the final operator question using the original question, primary_operator.operator, primary_operator.cue_text, and natural-language candidate descriptions. Do not mention input variables.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()

GLOBAL_METHOD_GUARD = """
You are implementing the DEPO method from depo.md.
CoreNLP parses a selectively masked question: complex noun phrases may be replaced by POS-hinting placeholders such as MovieA, CompanyA, or NetworkA.
Type variables and syntactic scaffold words stay in natural language.
After parsing, entity/type-variable token spans are folded into anchor supernodes on the dependency graph.
The MST is an anchor-only MST over entity/type-variable anchor nodes.
Do not do ordinary end-to-end subquestion decomposition.
Do only the current pipeline step.
Do not introduce unsupported entities or type variables; implicit attribute variables are allowed only when grounded by an explicit comparative, superlative, ordinal, or predicate cue in the original question.
Do not merge repeated variables when they have different semantic roles.
The output must be valid parseable JSON and nothing else.
"""

ENTITY_EXTRACTION_SYSTEM = (
    GLOBAL_METHOD_GUARD
    + """
Current step: identify entity nodes and type-variable nodes only.
Do not generate subquestions.
Extract minimal relation-bearing anchor nodes for dependency-graph inspection, not full descriptive noun phrases.
Include implicit type variables when the question asks about an attribute through a comparative, superlative, ordinal, or predicate word even if the attribute noun is not literally present.
When a comparative/superlative cue modifies an event or predicate, use the predicate word as the anchor text and keep the cue only in cue_text/cue_start/cue_end.
Assign each node a natural-language CamelCase placeholder in the format SemanticType + GreekOrdinal.
Use Greek ordinals in this order: Alpha, Beta, Gamma, Delta, Epsilon, Zeta, Eta, Theta.
Examples: CompanyAlpha, PersonAlpha, PersonBeta, FilmAlpha, FilmBeta, NationalityAlpha.
Use EntityAlpha only when no more specific semantic type is natural.
For repeated mentions with different roles, keep separate nodes with separate placeholders.
For type variables, use the shortest surface span that still names the relation endpoint correctly.
Prefer the head role/category for organization, institution, place, person, and title endpoints.
Keep pre-head words only when they form an essential functional/common-noun term; remove field, topic, quality, domain, purpose, or scope modifiers.
Do not omit a functional/structural nominal endpoint just because it is introduced as an attributed property or predicate complement.
Return exact character spans in the original question whenever possible, using Python-style start inclusive and end exclusive.
For an implicit type variable whose text is not present in the question, set text to the semantic attribute name and use cue_text/cue_start/cue_end for the word that expresses it in the question.
These original spans will be shifted after selective masking and then aligned to CoreNLP tokens, so span accuracy is critical.
"""
)


def build_entity_extraction_prompt(question: str) -> str:
    schema = {
        "entities": [
            {
                "text": "NamedEntity",
                "semantic_type": "Entity",
                "placeholder": "EntityAlpha",
                "start": 0,
                "end": 11,
                "occurrence": 1,
            }
        ],
        "type_variables": [
            {
                "text": "company",
                "semantic_type": "Company",
                "placeholder": "CompanyAlpha",
                "start": 0,
                "end": 7,
                "occurrence": 1,
                "cue_text": "",
                "cue_start": None,
                "cue_end": None,
            }
        ],
    }
    return f"""
Extract only the core entity and type-variable nodes for this question.

Definitions:
- entity: a concrete named entity or named artifact explicitly named in the question.
- type_variable: a minimal role, title, office, answer type, object type, institution type, system, artifact, place type, or other common-noun concept that acts as an endpoint in the question's relation chain.
- implicit type_variable: an attribute endpoint that is asked through a comparative/superlative/predicate cue rather than by an explicit noun in the question. Use the semantic attribute as text and the cue word span for alignment.

Rules:
- Do not generate atomic subquestions.
- Do not output relations.
- Do not invent nodes not explicitly supported by the question text.
- Extract only relation-bearing graph anchors: named entities, answer types, roles/titles/offices, institutions, places, systems, artifacts, and object/category concepts that are endpoints of predicates, possessives, clauses, or prepositional relations.
- Always include explicit role/title/office mentions when they participate in a relation, including abbreviations and uppercase titles. Do not drop a role just because it is attached to another node by a possessive, "of", or relative-clause relation.
- Always include implicit compared or ranked attributes. For example, comparative/superlative words imply an attribute node; output that attribute as a type_variable and provide the cue word offsets.
- For event or predicate comparisons, the anchor is the predicate token, not the predicate plus comparative phrase. Keep the comparative word only as the cue.
- For each type_variable, choose the shortest contiguous span that still names the endpoint correctly. Remove determiners and nonessential adjectives.
- Include nominal predicate complements and attributed/possessed things when they are themselves functional, structural, institutional, artifact, system, place, or role endpoints in the relation chain. They remain anchors even if the surrounding clause describes a property of another node.
- For organization, institution, place, person, and role/title endpoints, the head category or title is normally the node. Remove preceding field, industry, topic, domain, quality, purpose, scope, and descriptive modifiers unless the whole phrase is a proper named entity.
- For functional or structural common-noun endpoints, keep a compact compound span only when the pre-head word changes the endpoint class and the head alone would be too vague for the relation chain. Keep only essential compound words; remove determiners, quality adjectives, clauses, and prepositional complements.
- If a word or phrase only describes, restricts, classifies, quantifies, dates, measures, or gives the topic/domain/purpose of another anchor, do not extract it as a standalone node unless the question directly asks for that value. This pruning applies to the modifiers and complements around an anchor, not to the anchor noun phrase itself.
- Do not extract objects inside modifier/complement phrases as separate nodes when they are only topical restrictions or purposes of another endpoint.
- Do not extract quantities, durations, dates, ordinals, comparative words, or measurement phrases as standalone nodes unless the question directly asks for that value as the answer. A duration or quantity that only modifies how long an action lasted is not an anchor.
- Before returning, prune every multi-word type_variable span: if removing a pre-head word leaves a valid role/category endpoint of the same relation, output the shorter span; if removing it changes a functional/structural endpoint into a vague generic noun, keep the compact compound.
- Keep duplicate role variables separate when they belong to different branches or distinct mentions with different roles.
- Preserve exact surface text from the original question.
- Return accurate start/end character offsets for each surface span in the original question.
- If the type variable is implicit and its text does not occur in the question, start/end may point to the cue word and cue_text/cue_start/cue_end must identify that same cue.
- Prefer semantic placeholders like PersonAlpha for CEO/director, FilmAlpha for films, CompanyAlpha for companies.

Output JSON with exactly this shape:
{json.dumps(schema, indent=2)}

Question:
{question}
""".strip()


OPERATOR_SELECTION_SYSTEM = (
    GLOBAL_METHOD_GUARD
    + f"""
Current step: choose operators and shared-node attachments for the final AST.
The input graph is already an anchor-only semantic graph built from weighted dependency shortest paths.
You must not rewrite the anchor graph.
You must not add, remove, or reorder anchor-anchor edges.
You must not generate subquestions.
You may only choose from this fixed operator set: {", ".join(ALLOWED_OPERATORS)}.
Return JSON only.
"""
)


def build_operator_prompt(
    question: str,
    anchor_nodes: list[dict[str, str]],
    anchor_edges: list[dict[str, object]],
) -> str:
    schema = {
        "operators": [
            {
                "operator": "COMPARE_SAME",
                "attach_to": ["NationalityAlpha"],
                "explanation": "The question asks whether two branch results share the same nationality.",
            }
        ]
    }
    return f"""
Given the original question and the anchor-only semantic graph, choose only the needed operator(s) and the existing anchor node(s) they attach to.

Original question:
{question}

Anchor nodes:
{json.dumps(anchor_nodes, ensure_ascii=False, indent=2)}

Anchor semantic graph edges:
{json.dumps(anchor_edges, ensure_ascii=False, indent=2)}

Rules:
- Keep the anchor graph unchanged.
- If the graph is a simple serial bridge with no comparison, set, extremum, or logical operator, use NONE.
- If the question asks whether two branch results are the same, use COMPARE_SAME and attach it to the shared result node.
- If the question asks whether two branch results are different, use COMPARE_DIFF and attach it to the shared result node.
- Use INTERSECTION for common/shared results, UNION for either/all alternatives, and DIFFERENCE for results present in one branch but not another.
- Use COMPARE_GREATER or COMPARE_LESS for numeric/ordered comparisons.
- Use ARGMAX or ARGMIN for superlative maximum/minimum selection.
- Use LOGICAL_AND or LOGICAL_OR for explicit boolean combination conditions.
- Do not create new anchor nodes.
- Do not generate subquestions.

Output JSON with exactly this shape:
{json.dumps(schema, indent=2)}
""".strip()


ONE_HOP_SUBQUESTION_SYSTEM = (
    GLOBAL_METHOD_GUARD
    + """
Current step: rewrite exactly one adjacent AST edge as one atomic subquestion.
Use only the two provided adjacent nodes and the original question.
Internal variables are implementation details. Do not expose X1, X2, V1, VAR_*, or similar variables in the generated question.
Do not use any other AST nodes.
Do not use multi-hop information.
Do not generate a sequence of subquestions.
Do not infer the complete decomposition; generate only the one subquestion for this one adjacent edge.
Return JSON only.
"""
)


def build_one_hop_prompt(
    original_question: str,
    source_display: str,
    target_display: str,
    source_original: str,
    target_original: str,
    answer_variable: str,
    edge_hint: str | None = None,
) -> str:
    schema = {"question": "Which company developed AlphaGo?"}
    hint_text = f"\nDependency/AST edge hint: {edge_hint}" if edge_hint else ""
    return f"""
Generate one atomic subquestion for exactly this one-hop AST edge.

Original question:
{original_question}

Adjacent AST edge endpoints:
- Source endpoint text to use in the subquestion, verbatim: {source_display}
- Target endpoint to ask for: {target_display}

Endpoint meanings:
- Source original node meaning only, not replacement text when source endpoint is a variable: {source_original}
- Target original node: {target_original}

The answer variable assigned by the program will be: {answer_variable}
{hint_text}

Rules:
- Only use the two endpoints above and the original question.
- Do not expose internal variables such as X1, X2, V1, or VAR_* in the question.
- When the source endpoint is a variable, write the most natural descriptive question from the available endpoint meanings without inventing the answer.
- Do not include comparative cue words such as earlier, later, older, younger, larger, or smaller in one-hop attribute questions; those cue words belong to the final operator question.
- Ask for the target endpoint as the answer.
- Do not mention any other node from the full AST.
- Do not generate additional subquestions.

Output JSON with exactly this shape:
{json.dumps(schema, indent=2)}
""".strip()
