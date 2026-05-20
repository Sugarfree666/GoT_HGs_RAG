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

MASK_SPAN_EXTRACTION_SYSTEM = """
You are implementing the DEPO selective complex-span masking step.
Your job is parser-protection span detection before CoreNLP parsing.
This is a surface-span task, not semantic node extraction.
Identify only complex contiguous spans that should become one placeholder token so the dependency parser does not split them incorrectly.
This step protects multi-token surface spans only.
Do not perform anchor extraction.
Do not extract answer variables, wh phrases, selected anchors, implicit variables, operators, relations, AST nodes, or subquestions.
Extract only multi-word proper-name/title spans and compact multi-word compound type or functional noun phrases.
Do not mask wh answer phrases such as which/what/whose + a single type noun.
Do not mask single-token entities, even if they are mixed-case, abbreviated, or well-known names.
Keep simple type variables such as director, CEO, university, city, nationality, age, population, country, and actor unmasked unless they are part of a larger multi-word phrase.
Return valid JSON only.
""".strip()


def build_mask_span_extraction_prompt(question: str) -> str:
    schema = {
        "mask_spans": [
            {
                "text": "multi word span",
                "start_char": 0,
                "end_char": 15,
                "kind_hint": "entity",
                "semantic_type_hint": "Entity",
                "reason": "brief parser-protection reason",
            }
        ]
    }
    return f"""
Identify only spans that should be masked before CoreNLP parsing.

This is parser-protection span detection only.
It is not entity importance detection, anchor extraction, type-variable extraction, answer-variable extraction, relation extraction, or question decomposition.
The output is only a list of contiguous surface spans in the original question that need to be collapsed into one placeholder token before dependency parsing.
In short, target multi-word named entities/titles and compact multi-word compound type/function noun phrases with at least two content lexical units.
Here, valid multi-word type variables must be compound-head type phrases, not wh answer phrases.
Every accepted span must have at least two lexical units after ignoring determiners and wh words.

Core target:
- Mask every multi-word named entity, every multi-word title, and every compact multi-word compound type/function noun phrase.
- The span must contain at least two content lexical units. Determiners and wh words such as the/a/an/which/what/whose do not count. Middle initials such as "W." count as part of a multi-word person name.
- Do not mask a single-token entity by itself, even if it is mixed-case, has digits, is an acronym, or appears parser-fragile.
- Do not mask a span merely because it is semantically important to the question.

Extract exactly these two categories:

Category A: Proper-name spans.
- Continuous multi-token person names, organization names, institution names, place names, event names, product names, and named works.
- These are mandatory mask spans whenever present.
- A personal name pattern such as First Last, First Middle Last, First M. Last, or multi-token name particles must be returned.
- In a comparison or coordination with "and/or", apply this to every option independently.

Category B: Compound-head type spans.
- A compact noun phrase whose rightmost word is the semantic head, and whose earlier words are essential compound/classifying modifiers of that head.
- The head must be an entity/type/function/object/role class noun.
- The phrase should still name a reusable type or object class after masking.
- Valid type spans normally have at least one content modifier before the head, not just a determiner before a single noun.
- Valid type spans are compound-head phrases such as artificial intelligence company, chief operating officer, distribution network, research institute, local food system, or mixed-use space.
- Invalid type spans are wh/determiner + one simple type noun, such as which university, which city, what country, whose director, the CEO, the university, or a city.
- If a quality/evaluative adjective appears before a compound head phrase, drop the quality adjective and keep the compact compound head phrase.

Span boundary rules:
- Use exact original question character offsets, start inclusive and end exclusive.
- Return the minimal contiguous span that should become one placeholder token.
- For named entities and titles, keep the full official/name-like surface form.
- For type variables and functional noun phrases, exclude leading determiners such as the/a/an unless they are part of a named title.
- Exclude leading wh words such as which/what/whose/who/whom/where/when/how. A wh phrase is an answer cue, not a parser-protection span.
- For compound-head type spans, include only essential compound/classifying modifiers and the head noun.
- Do not include quality/evaluative adjectives, relative clauses, participial clauses, prepositional complements, comparison words, or coordination words unless they are part of a proper name/title.
- If two candidate spans overlap, prefer the larger coherent entity/title, or the compact functional noun phrase for type variables.
- Scan the whole question. In coordinated alternatives, extract every eligible multi-word entity, not just one side.

Negative boundary rules:
- Do not mask wh answer phrases or wh + single-noun variables: which university, which city, what country, whose director, where, when, who, what.
- Do not mask determiner + single noun spans: the university, the city, a company, the CEO.
- Do not mask a single simple type variable: university, city, CEO, company, director, nationality, population, region, country, actor.
- Do not mask a single-token named entity or product/work/person/place name.
- Do not mask modifier-only phrases that do not include the functional/type head noun.
- Do not mask purpose/topic phrases after prepositions such as for/about/with unless that prepositional object itself is the main type being asked for.
- Do not mask durations, quantities, measurements, or temporal expressions.
- Do not mask adjective-only phrases or descriptive property phrases.
- Do not mask spans containing verbs, gerunds, or participles unless the span is an official named title.
- Do not mask material/topic phrases that only specify the domain, purpose, or contents of another head noun.
- Do not mask a phrase just because it is multi-word; it must satisfy Category A or Category B.

Semantic type hints:
- Choose a semantic_type_hint that preserves the original POS/semantic role for placeholder generation.
- Person names in human contexts such as who/whom/whose, older/younger, actor, CEO, director, author, player, or president should use semantic_type_hint: Person.
- Location names should use City/Country/Region/Location when the question context asks for places.
- Organizations and institutions should use Company/Organization/University/Institution when supported by the span or context.
- Named works should use Film/Book/Song/Album/Series/Work when supported by local wording.
- Multi-word type variables should use kind_hint: type_variable and a semantic_type_hint for their head class.

Do not mask simple one-word type variables by default:
director, CEO, university, city, company, nationality, age, population, country, region, actor.

Decision procedure:
1. Ignore semantic importance first. Ask only: would collapsing this exact surface span help CoreNLP keep a complex name or compound noun phrase intact?
2. First scan for Category A proper-name spans. Return every multi-token proper name. Do not skip names just because they are syntactically simple.
3. Then scan for Category B compound-head type spans. Keep only compact modifier + head noun phrases.
4. For proper names, include all adjacent name tokens and initials in the person/place/organization/title name.
5. For type/function phrases, the head noun must be included, nonessential quality adjectives should be excluded, and determiners/wh words do not count as modifiers.
6. Drop single-token named entities and single-word type variables.
7. Drop wh + single noun, determiner + single noun, modifier-only phrases, duration/quantity phrases, verb phrases, and prepositional purpose/context phrases.
8. For coordinated alternatives, apply the same criteria independently to each side.

Forbidden outputs:
- selected anchors
- implicit type variables
- operators
- final AST
- subquestions
- decomposition of coordination

Use exact original question character offsets, start inclusive and end exclusive.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Question:
{question}
""".strip()


SEMANTIC_QUESTION_NORMALIZATION_SYSTEM = """
You are implementing a semantic question normalization step before dependency parsing.
First decide whether the input question needs normalization.
If the question is already a clear, single, dependency-parser-friendly question, return the original question unchanged.
If normalization is needed, rewrite the input as one semantically equivalent question whose syntax is clearer and more explicit for a dependency parser.
Do not answer the question.
Do not decompose the question into atomic questions.
Do not output reasoning, ASTs, graphs, operators, execution plans, or subquestions.
Do not introduce new named entities.
Preserve every placeholder token exactly if placeholders are present.
Preserve explicit type variables, answer type, and question meaning.
Only make an implicit type variable explicit when it is licensed by a clear cue in the original question.
If a safe rewrite is not possible, return the original question unchanged.
Return valid JSON only.
""".strip()


def build_semantic_question_normalization_prompt(
    question: str,
    placeholders: list[str] | None = None,
) -> str:
    placeholders = placeholders or []
    schema = {
        "normalized_question": "the original question if no rewrite is needed, otherwise one semantically equivalent normalized single-sentence question",
        "changed": True,
        "added_type_variables": [
            {
                "text": "explicit type variable added by the rewrite",
                "trigger_text": "exact cue in the original question",
                "reason": "brief justification",
            }
        ],
    }
    return f"""
Decide whether the question needs semantic normalization for dependency parsing.

Original question:
{question}

Placeholder tokens that must be preserved exactly:
{json.dumps(placeholders, ensure_ascii=False, indent=2)}

Rewrite goals:
- If the original question is already clear, complete, and parser-friendly, set normalized_question to the exact original question and changed to false.
- If the original question has ellipsis, compressed coordination, unclear attachment, implicit comparison attributes, or a structure likely to confuse dependency parsing, provide a normalized question.
- Keep the output as exactly one question.
- Remove ellipsis and expand shared coordinated structure when doing so does not change meaning.
- Make relationships between entities, type variables, comparison targets, logical constraints, and answer focus syntactically explicit.
- If a comparative, superlative, ordinal, location, time, identity, membership, or predicate cue clearly implies an attribute/type variable, the rewrite may name that variable explicitly.
- Keep all explicit type variables from the input.
- Keep all placeholders exactly as written, with the same number of occurrences.
- Keep all named entities exactly as written.
- Prefer a direct, parser-friendly clause structure over compressed coordination.
- For coordination or comparison, make each compared or coordinated item attach clearly to the same relation or attribute.
- Keep the result a natural English question, not a statement, fragment, list, or template.

Strict restrictions:
- Do not answer the question.
- Do not split the input into multiple questions.
- Do not output an AST, graph, decomposition, solving order, or reasoning trace.
- Do not introduce a named entity not present in the original question.
- Do not infer missing facts, categories, locations, dates, genders, nationalities, occupations, or aliases from world knowledge.
- Do not remove or rename any placeholder.
- Do not change the answer type.
- Do not change the comparison, set, temporal, location, or logical condition being asked.
- If the rewrite would require guessing unstated facts or entity types, return the original question unchanged and changed=false.

Added type-variable reporting:
- If the rewrite adds an explicit type-variable noun that was only implicit in the original, list it in added_type_variables.
- The trigger_text must be an exact cue span from the original question.
- Do not list variables that already appeared explicitly in the input.
- If normalized_question is exactly the original question, added_type_variables must be an empty list.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


ANCHOR_SELECTION_SYSTEM = """
You are implementing DEPO Step 4: explicit anchor selection.
You must select anchors only from the provided candidate anchor set.
You will see only the semantic-normalized question and the candidate anchor set.
Do not require masked text, dependency tokens, dependency edges, POS tags, parser labels, placeholders, ASTs, or subquestions.
The candidate text already shows the surface text to judge.
Allowed anchor kinds are exactly: entity, type_variable.
Select every explicit entity or explicit type-variable endpoint that must be preserved as a node in the semantic graph.
Anchors are not only final answers. Select intermediate entities, roles, classes, and attributes that are required to preserve relation chains and constraints.
Select single-token named entities when they are explicit fixed inputs to a relation or constraint.
Select compact multi-word type variables and single-word type variables when they are answer variables, intermediate constraint variables, relation-chain endpoints, or explicit attributes.
For nested phrases, possessive/of-prepositional phrases, and relative clauses, select each explicit endpoint needed to express the chain, not just the final answer variable.
Do not select generic answer labels that merely restate who/what/where/when, such as person, thing, place, time, entity, item, one, someone, something, or somewhere, unless that exact class must be independently bound in a relation chain.
Use entity only for proper named entities or fixed named inputs. Use type_variable for common-noun roles, classes, institutions, organizations, places, and attributes such as CEO, company, university, city, actor, director, nationality, population, and compact noun phrases.
Do not select implicit_type_variable, operator, cue, comparative cue, superlative cue, coordination cue, logical cue, function word, or predicate-only verb.
Do not select words such as same, different, older, younger, larger, smaller, largest, highest, first, last, before, after, and, or, both, either.
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
                "text": "Ten9Eight: Shoot For The Moon",
                "reason": "Film entity explicitly mentioned in the question",
            },
            {
                "node_id": "13",
                "anchor_kind": "type_variable",
                "text": "nationality",
                "reason": "Explicit attribute being compared",
            },
        ]
    }
    return f"""
Select explicit anchors for the semantic-normalized question.

Semantic-Normalized Question:
{original_question}

Candidate anchor set:
{json.dumps(restored_graph_node_candidates, ensure_ascii=False, indent=2)}

Rules:
- Output node_id values from the candidate anchor set.
- Use only the Semantic-Normalized Question and the candidate anchor set.
- Do not infer from missing parser metadata. Do not invent entities, variables, attributes, or relations.
- Select all explicit semantic endpoints needed to reconstruct the question's relation chain.
- Select entity anchors for explicit named people, organizations, places, works, products, events, or other fixed named inputs, including single-token names.
- Select type_variable anchors for explicit answer variables, intermediate role/class variables, constrained noun phrases, and explicit attributes or measures.
- Do not use anchor_kind=entity for common-noun roles or classes just because they are important. CEO, company, university, city, actor, director, region, nationality, population, and multi-word class phrases should use anchor_kind=type_variable unless they are part of an official proper name.
- Select relation-chain endpoints inside of-phrases, possessives, prepositional constraints, and relative clauses when they name an entity or type variable.
- Select intermediate variables even when they are not the final answer, if removing them would lose a constraint needed to reach the answer.
- Select repeated mentions separately when they are separate candidate nodes and play separate roles in the question.
- Do not select implicit variables. If a comparative cue implies age but age is not a candidate text, do not create or select age.
- Do not select operators or cues. For "same nationality", select nationality, not same.
- Do not select generic answer labels. For "Who is the older person, PersonA or PersonB?", select PersonA and PersonB only; do not select person.
- Do not select predicate-only verbs, auxiliaries, determiners, wh words, prepositions, relativizers, punctuation, function words, comparative/superlative words, coordination words, or purely syntactic bridge words.
- Relation phrases belong later in semantic AST edge relation_hint; Step 4 selects the endpoint nodes, not relation text.
- Use anchor_kind=entity for proper named entities and fixed named inputs. Use anchor_kind=type_variable for type variables, roles, classes, answer variables, and explicit attributes.
- The text field of each selected item must exactly copy the candidate text for that node_id. Do not output implied attributes such as age unless age is itself a candidate text.
- If no candidate should be selected, return an empty selected_anchors list.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


SEMANTIC_AST_OPTIMIZATION_SYSTEM = """
You are implementing DEPO Step 9: final semantic AST optimization.
The anchor connected graph is only syntactic/structural evidence, not the final reasoning AST.
Use original question semantics as the highest-priority source of truth.
Do not copy dependency edges or evidence paths directly as semantic edges.
The directed AST is a reasoning DAG: each edge must point from an already-known or already-bound node to the next node that should be solved.
Coreference, same-as, apposition, and repeated mentions are not reasoning hops. Merge them into one canonical semantic node instead of outputting an edge.
Merge this/that/the/its/his/her/their mentions with their antecedent. Merge a wh-variable with later mentions that refer to the same variable, such as "which university" and "this university".
This is the only step that may create implicit type variables and choose a primary operator.
Choose exactly one primary_operator from the allowed operator set. Use NONE when there is no comparison, superlative, set, or logical cue.
Do not invent entities that are not present in the original question or mask mapping.
Do not generate subquestions.
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
            "operator": "COMPARE_SAME",
            "cue_text": "same",
            "inputs": ["nationality_1", "nationality_2"],
            "output": "answer",
            "explanation": "The question asks whether two nationalities are the same.",
        },
        "nodes": [
            {
                "id": "movie_1",
                "label": "Ten9Eight: Shoot For The Moon",
                "kind": "entity",
                "semantic_type": "Film",
                "source": "selected_anchor",
                "source_graph_nodes": ["8"],
                "source_token_indices": [8],
                "grounding_text": "Ten9Eight: Shoot For The Moon",
                "cue_text": "",
                "branch_of": "",
                "expected_value_slot": "",
                "relation_hint": "",
            }
        ],
        "edges": [
            {
                "source": "movie_1",
                "target": "director_1",
                "edge_type": "attribute",
                "relation_hint": "director of film",
                "support_path": ["Ten9Eight: Shoot For The Moon", "film", "director"],
                "support_dependency_relations": ["appos", "nmod:of"],
            }
        ],
    }
    return f"""
Optimize the restored anchor connected subgraph into a directed semantic AST.
The anchor connected graph is only evidence. The final AST must be the semantic reasoning skeleton of the original question, not a transcription of dependency edges.

Original question:
{original_question}

Mask restore information:
{json.dumps(replacement, ensure_ascii=False, indent=2)}

Selected explicit anchors:
{json.dumps(selected_anchors, ensure_ascii=False, indent=2)}

Restored anchor connected subgraph:
{json.dumps(restored_anchor_connected_subgraph, ensure_ascii=False, indent=2)}

Allowed primary operators:
{json.dumps(allowed_operators, ensure_ascii=False)}

Previous AST that failed validation:
{json.dumps(previous_ast or {}, ensure_ascii=False, indent=2)}

Validation feedback to fix:
{json.dumps(validation_feedback or [], ensure_ascii=False, indent=2)}

Rules:
- Choose exactly one primary_operator.operator from the allowed set.
- If validation feedback is non-empty, the previous AST operator inputs do not match expected value slots inferred from the original question; regenerate the AST by inserting implicit_type_variable nodes before the operator inputs.
- Use NONE when no operator cue is present.
- Operator choice must be grounded in the original question cue, e.g. same -> COMPARE_SAME, different -> COMPARE_DIFF, older -> COMPARE_GREATER on age, largest/highest/most -> ARGMAX.
- You may add implicit type variables only when grounded by a cue in the original question.
- Implicit variables must include cue_text and grounding_text.
- Before attaching branch endpoints to an operator, infer the expected value slot required by the original question.
- A value slot is the actual attribute, date, measure, category, count, role value, or comparable value that must be retrieved before the operator can be applied.
- The operator should consume value slot nodes, not merely the current branch endpoints.
- Do not only check whether the operator input is a bare entity. Even if the operator input is an intermediate variable such as director, actor, CEO, city, university, company, country, person, film, organization, or type variable, the AST is incomplete if the original cue requires an additional value slot such as birth_date, nationality, release_date, population, founding_date, height, length, count, or category.
- For each branch: identify the current branch endpoint; infer the expected value slot from the original question cue; if the endpoint is not already that slot, create an implicit_type_variable; connect current endpoint -> implicit_type_variable; make the operator consume the implicit_type_variable.
- Every implicit_type_variable must include cue_text, grounding_text, branch_of, relation_hint, and expected_value_slot.
- Cue-to-value-slot guidance:
  released first/earliest/earlier -> expected_value_slot=release_date, operator=ARGMIN or COMPARE_LESS.
  released later/latest/last -> expected_value_slot=release_date, operator=ARGMAX or COMPARE_GREATER.
  born earlier -> expected_value_slot=birth_date, operator=COMPARE_LESS.
  born later -> expected_value_slot=birth_date, operator=COMPARE_GREATER.
  died earlier/later -> expected_value_slot=death_date, operator=COMPARE_LESS/COMPARE_GREATER.
  founded earlier/later -> expected_value_slot=founding_date, operator=COMPARE_LESS/COMPARE_GREATER.
  published first/earlier -> expected_value_slot=publication_date, operator=ARGMIN or COMPARE_LESS.
  launched first/earlier -> expected_value_slot=launch_date, operator=ARGMIN or COMPARE_LESS.
  older -> expected_value_slot=age or birth_date, operator=COMPARE_GREATER if using age and COMPARE_LESS if using birth_date.
  younger -> expected_value_slot=age or birth_date, operator=COMPARE_LESS if using age and COMPARE_GREATER if using birth_date.
  same/different nationality/director/author/country -> expected_value_slot=that noun, operator=COMPARE_SAME/COMPARE_DIFF.
  largest/smallest population -> expected_value_slot=population, operator=ARGMAX/ARGMIN.
  highest mountain -> expected_value_slot=height or elevation, operator=ARGMAX.
  lowest point -> expected_value_slot=elevation, operator=ARGMIN.
  longest/shortest river -> expected_value_slot=length, operator=ARGMAX/ARGMIN.
  most/fewest awards -> expected_value_slot=award_count, operator=ARGMAX/ARGMIN.
  most populous city -> expected_value_slot=population, operator=ARGMAX.
- Bad: FilmA -> director_1 -> COMPARE_LESS and FilmB -> director_2 -> COMPARE_LESS when the question says "director who was born earlier". Good: FilmA -> director_1 -> birth_date_1 -> COMPARE_LESS and FilmB -> director_2 -> birth_date_2 -> COMPARE_LESS.
- Bad: FilmA -> director_1 -> COMPARE_SAME and FilmB -> director_2 -> COMPARE_SAME when the question says "same nationality". Good: FilmA -> director_1 -> nationality_1 -> COMPARE_SAME and FilmB -> director_2 -> nationality_2 -> COMPARE_SAME.
- Original question semantics outrank the anchor connected graph. Use the graph only as evidence for endpoints and local wording.
- Do not turn dependency relations, coreference arcs, apposition arcs, same-as mentions, or pronoun/determiner mentions into semantic edges.
- Merge coreferent mentions into one node. In particular, merge this/that/the/its/his/her/their mentions with the antecedent they refer to.
- Merge repeated mentions of the same wh-variable with later references. For example, "which university" and "this university" must be one node such as university_1.
- Never output edges whose relation_hint is same as, is the same as, corefers with, refers to, appositive, or mention-of; perform node merging instead.
- Correct semantic roles before writing edges: a graduate-from edge should be Person/Role -> University; located-in should be Institution/Organization/Place -> City/Country/Location; developed should connect product/system and developer; CEO-of should be Company/Organization -> CEO/Person.
- You may split parallel branches and copy shared variables, e.g. nationality -> nationality_1 and nationality_2.
- Node id may carry branch suffixes such as director_1 or nationality_2, but node label must be clean natural-language text such as director or nationality.
- Do not put edge/relation phrases into node labels. For example, label the node nationality, not nationality of director.
- Put phrases such as director of film or nationality of director only in edge.relation_hint.
- Convert the undirected evidence graph into directed semantic edges whose direction follows inference, not surface syntax.
- Direction rule: source is a known constant/entity or a previously solved variable; target is the next variable/value to solve.
- If explicit named entities exist, start each branch from those entities and move toward answer variables or operator inputs.
- If no explicit entity exists, start from the answer candidate type or comparison subject, then move toward attributes used for constraints/operators.
- For "Which university did the CEO of the company that developed AlphaGo graduate from?", the reasoning direction is AlphaGo -> company -> CEO -> university, not university -> CEO -> company -> AlphaGo.
- For "Which university did the CEO of the company that developed AlphaGo graduate from and in which city is this university located?", use AlphaGo -> company -> CEO -> university -> city. There must be only one university node; do not add university -> university same-as/coreference edges.
- For "Do film A and film B share the same nationality?", use film A -> director_1 -> nationality_1 and film B -> director_2 -> nationality_2.
- For "Which country has the largest population?", use country -> population, with ARGMAX over population.
- The primary operator will be represented as an operator node by the system; primary_operator.inputs must name the branch endpoint node ids consumed by the operator.
- Keep selected anchors unless you provide an explicit reason in the node/edge choices.
- Do not create entities that are absent from selected anchors or mask mappings.
- Do not generate atomic subquestions.

Output JSON with exactly this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


ATOMIC_SUBQUESTION_GENERATION_SYSTEM = """
You are implementing DEPO Step 8: LLM-based atomic subquestion generation.
Generate exactly one atomic subquestion for the provided one-hop semantic AST edge, or exactly one final operator question for the provided primary operator.
Use the original question and semantic AST context, but do not combine multiple AST edges into a multi-hop question.
The input edge is already oriented as source/bound node -> target node to solve.
If the source is bound to an answer variable such as X1, use that variable in the question instead of expanding the original source label.
For ordinary attribute edges, do not include operator cue words such as same, older, largest, before, or after.
For operator steps, preserve the concrete cue from the original question and mention the operator input variables directly.
The DAG inputs, outputs, dependencies, edges, and candidate bindings are produced by code; only write the natural-language question text.
Return valid JSON only.
""".strip()


ATOMIC_PLAN_STEP_SURFACE_SYSTEM = """
You are implementing DEPO Step 8 surface realization for one deterministic execution-plan step.
The semantic AST has already been compiled into a variable-bound execution DAG by code.
Do not re-plan, reorder, infer hidden hops, merge steps, or use any node not present in this single plan step.
For an edge step, generate one atomic question from exactly one AST one-hop relation; the answer is answer_variable.
Use step.known as the known subject exactly; if it is X1, X2, or another variable, the exact variable must appear in the question.
For an operator step, generate the final atomic operator question from the original question, step.operator, step.cue_text, step.inputs, and step.operator_branches.
Preserve the concrete comparison/logical meaning from the original question; do not replace it with generic greater/less wording unless the original cue is actually greater/less.
Mention every variable in step.inputs exactly, and do not ask a new attribute edge question in an operator step.
Do not decide execution dependencies, outputs, DAG edges, or candidate bindings; those are deterministic fields outside this prompt.
Do not include comparative/superlative/operator cue words in ordinary edge questions.
Return valid JSON only.
""".strip()


def build_atomic_plan_step_surface_prompt(
    original_question: str,
    plan_step: dict[str, object],
) -> str:
    schema = {
        "question": "What is the nationality of X1?",
        "answer_variable": "X2",
        "explanation": "This surfaces only the provided execution step.",
    }
    return f"""
Generate one atomic subquestion from this already-compiled execution-plan step.

Original question:
{original_question}

Execution-plan step:
{json.dumps(plan_step, ensure_ascii=False, indent=2)}

Rules:
- Do not infer a different step from the original question.
- Do not use the full AST or any unstated path.
- For step_type=edge, ask only for the one-hop relation step.known -> step.ask using step.relation_hint as wording guidance.
- The answer to an edge step will be step.answer_variable.
- If step.known is an answer variable such as X1, X2, or X1_nationality, that exact variable must appear in the question.
- If step.known is a variable, do not expand it back into step.known_node_label or the original entity/path.
- For ordinary edge steps, do not include operator cue words such as same, different, older, younger, largest, highest, first, before, or after.
- For step_type=operator, generate the final operator question, not another one-hop attribute question.
- For step_type=operator, use step.operator_branches only to understand what each input variable represents; the question itself must mention the variables from step.inputs directly.
- For step_type=operator, preserve the original cue/attribute/event in step.cue_text and the original question. If the cue is a concrete property or event, ask about that property or event rather than asking whether one input is abstractly greater/less than another.
- For step_type=operator, include every variable in step.inputs exactly once or clearly as a comparison/set/logical input.

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
        "answer_variable": "X1",
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
- If current_edge.source_display is X1, X2, or another variable, that exact variable must appear in the generated question.
- When current_edge.source_display is a variable, do not also expand it back into the original path or source label. For example, ask "What is the nationality of X1?", not "For X1, what is the nationality of the director of FilmA?"
- The answer to this subquestion will be current_edge.answer_variable.
- Do not merge this edge with another edge.
- Do not include same/older/largest/comparative/superlative cue words in ordinary attribute questions.
- For an implicit variable edge such as actor -> age, ask a normal attribute question such as "What is the age of the actor?"
- For an operator step, generate the final operator question using the original question, primary_operator.operator, primary_operator.cue_text, and current_edge.inputs. Mention those input variables directly and preserve the concrete comparison/logical cue instead of using generic greater/less wording.

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
Respect variable binding exactly: if an endpoint is an intermediate answer variable such as X1 or X2, use that variable verbatim in the generated question.
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
- If the source endpoint is an answer variable such as X1, X2, or X1_nationality, the exact variable string must appear in the question.
- When the source endpoint is a variable, do not expand it back to the source original node text.
- Do not include comparative cue words such as earlier, later, older, younger, larger, or smaller in one-hop attribute questions; those cue words belong to the final operator question.
- Ask for the target endpoint as the answer.
- Do not mention any other node from the full AST.
- Do not generate additional subquestions.

Output JSON with exactly this shape:
{json.dumps(schema, indent=2)}
""".strip()
