# DEPO Decomposition #10

- Dataset: `2wikimultihopqa`
- Question: What nationality is the director of film Blood Street?
- Gold answer: Chinese

## 1. Semantic-Normalized Question
What nationality is the director of the film Blood Street?

## 2. Mask Spans
- Blood Street? (entity, Film)

## 3. Selective Masked Question
What nationality is the director of the film MovieA

## 4. CoreNLP Dependency Parse
- nationality[2] --det--> What[1]
- is[3] --obj--> nationality[2]
- director[5] --det--> the[4]
- is[3] --nsubj--> director[5]
- MovieA[9] --case--> of[6]
- MovieA[9] --det--> the[7]
- MovieA[9] --compound--> film[8]
- director[5] --nmod:of--> MovieA[9]

## 5. Undirected Dependency Graph
- What[1] --det-- nationality[2]
- nationality[2] --obj-- is[3]
- is[3] --nsubj-- director[5]
- the[4] --det-- director[5]
- director[5] --nmod:of-- Blood Street?[9]
- of[6] --case-- Blood Street?[9]
- the[7] --det-- Blood Street?[9]
- film[8] --compound-- Blood Street?[9]

## 6. Entity Start Nodes
- e1: Blood Street? graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Blood Street? -- director -- is -- nationality -- What
- e1_p2 (e1): Blood Street? -- director -- is -- nationality
- e1_p3 (e1): Blood Street? -- director
- e1_p4 (e1): Blood Street? -- director -- is
- e1_p5 (e1): Blood Street? -- director -- the
- e1_p6 (e1): Blood Street? -- film
- e1_p7 (e1): Blood Street? -- of
- e1_p8 (e1): Blood Street? -- the

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Blood Street, reaches director, and covers the nationality predicate, fully supporting the question intent.
- e1: e1_p2 score=85.0 valid=True terminal=nationality
  Reason: The path starts from Blood Street, reaches director, and covers the nationality predicate, mostly supporting the question intent.
- e1: e1_p3 score=30.0 valid=True terminal=none
  Reason: The path only connects Blood Street to director, missing necessary predicates and answer intent.
- e1: e1_p4 score=70.0 valid=True terminal=nationality
  Reason: The path starts from Blood Street, reaches director, and includes the is predicate, but misses the nationality cue.
- e1: e1_p5 score=25.0 valid=True terminal=none
  Reason: The path connects Blood Street to director and then to the determiner 'the', missing key predicates and answer intent.
- e1: e1_p6 score=30.0 valid=True terminal=none
  Reason: The path connects Blood Street to film, missing the director and key predicates.
- e1: e1_p7 score=10.0 valid=True terminal=none
  Reason: The path connects Blood Street to the preposition 'of', missing all necessary predicates and answer intent.
- e1: e1_p8 score=10.0 valid=True terminal=none
  Reason: The path connects Blood Street to the determiner 'the', missing all necessary predicates and answer intent.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p2'} mean_path_score=85.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- blood_street -> director (director of Blood Street)
- director -> nationality (nationality of the director)
### ast_ps2 (ps2)
- blood_street -> director (director of Blood Street)
- director -> nationality (nationality of the director)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST decomposes each film into director and nationality branches without generating a final comparison question.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- blood_street: Blood Street (entity)
- director: director (type_variable)
- nationality: nationality (value_slot)

Edges:
- blood_street -> director (director of Blood Street)
- director -> nationality (nationality of the director)

## 11. Atomic Subquestion DAG
- None: Who is the director of Blood Street?
- None: What is the nationality of the director of Blood Street?
