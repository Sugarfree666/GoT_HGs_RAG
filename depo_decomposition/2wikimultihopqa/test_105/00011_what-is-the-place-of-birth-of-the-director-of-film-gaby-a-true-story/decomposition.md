# DEPO Decomposition #11

- Dataset: `2wikimultihopqa`
- Question: What is the place of birth of the director of film Gaby: A True Story?
- Gold answer: Mexico City

## 1. Semantic-Normalized Question
What is the place of birth of the director of the film Gaby: A True Story?

## 2. Mask Spans
- Gaby: A True Story? (entity, Film)

## 3. Selective Masked Question
What is the place of birth of the director of the film MovieA

## 4. CoreNLP Dependency Parse
- What[1] --cop--> is[2]
- place[4] --det--> the[3]
- What[1] --nsubj--> place[4]
- birth[6] --case--> of[5]
- place[4] --nmod:of--> birth[6]
- director[9] --case--> of[7]
- director[9] --det--> the[8]
- birth[6] --nmod:of--> director[9]
- MovieA[13] --case--> of[10]
- MovieA[13] --det--> the[11]
- MovieA[13] --compound--> film[12]
- director[9] --nmod:of--> MovieA[13]

## 5. Undirected Dependency Graph
- What[1] --cop-- is[2]
- What[1] --nsubj-- place[4]
- the[3] --det-- place[4]
- place[4] --nmod:of-- birth[6]
- of[5] --case-- birth[6]
- birth[6] --nmod:of-- director[9]
- of[7] --case-- director[9]
- the[8] --det-- director[9]
- director[9] --nmod:of-- Gaby: A True Story?[13]
- of[10] --case-- Gaby: A True Story?[13]
- the[11] --det-- Gaby: A True Story?[13]
- film[12] --compound-- Gaby: A True Story?[13]

## 6. Entity Start Nodes
- e1: Gaby: A True Story? graph_node_ids=['13']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Gaby: A True Story? -- director -- birth -- place -- What
- e1_p2 (e1): Gaby: A True Story? -- director -- birth -- place -- What -- is
- e1_p3 (e1): Gaby: A True Story? -- director -- birth -- place
- e1_p4 (e1): Gaby: A True Story? -- director -- birth -- place -- the
- e1_p5 (e1): Gaby: A True Story? -- director -- birth
- e1_p6 (e1): Gaby: A True Story? -- director -- birth -- of
- e1_p7 (e1): Gaby: A True Story? -- director
- e1_p8 (e1): Gaby: A True Story? -- director -- of
- e1_p9 (e1): Gaby: A True Story? -- director -- the
- e1_p10 (e1): Gaby: A True Story? -- film
- e1_p11 (e1): Gaby: A True Story? -- of
- e1_p12 (e1): Gaby: A True Story? -- the

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=place_of_birth
  Reason: The path effectively connects the film to its director and the concept of birth, covering the necessary elements to answer the question.
- e1: e1_p2 score=95.0 valid=True terminal=place_of_birth
  Reason: This path includes the auxiliary 'is', enhancing its executability while maintaining coverage of all necessary elements for the question.
- e1: e1_p3 score=85.0 valid=True terminal=place_of_birth
  Reason: While the path covers the essential elements, it lacks the auxiliary 'is', which slightly reduces its executability.
- e1: e1_p4 score=80.0 valid=True terminal=place_of_birth
  Reason: This path includes a determiner 'the', which does not contribute to the semantic chain, affecting its overall score.
- e1: e1_p5 score=70.0 valid=True terminal=place_of_birth
  Reason: The path stops too early, missing the necessary 'place' and 'is', which are crucial for answering the question.
- e1: e1_p6 score=60.0 valid=True terminal=place_of_birth
  Reason: The inclusion of 'of' does not contribute meaningfully to the path, and it lacks key elements needed for a complete answer.
- e1: e1_p7 score=40.0 valid=True terminal=place_of_birth
  Reason: This path is too short and lacks essential components to form a meaningful answer.
- e1: e1_p8 score=50.0 valid=True terminal=place_of_birth
  Reason: The path includes 'of', which does not add value, and misses key elements needed for a complete answer.
- e1: e1_p9 score=45.0 valid=True terminal=place_of_birth
  Reason: The inclusion of 'the' does not contribute meaningfully to the path, and it lacks key elements needed for a complete answer.
- e1: e1_p10 score=30.0 valid=True terminal=place_of_birth
  Reason: This path only connects the film to the concept of 'film', which is not relevant to the question.
- e1: e1_p11 score=20.0 valid=True terminal=place_of_birth
  Reason: The path only includes 'of', which does not contribute to answering the question.
- e1: e1_p12 score=25.0 valid=True terminal=place_of_birth
  Reason: This path only includes 'the', which does not contribute to answering the question.

## 8.1 Top-2 Paths per Entity
- e1: e1_p2, e1_p1

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p2'} mean_path_score=95.0
- ps2: {'e1': 'e1_p1'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- gaby_a_true_story -> director (director of Gaby: A True Story)
- director -> birthplace (place of birth of the director)
### ast_ps2 (ps2)
- gaby_a_true_story -> director (director of Gaby: A True Story)
- director -> birthplace (place of birth of the director)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST decomposes each film into director and birthplace branches without generating a final comparison question.
- ast_ps2: score=0.9 valid=True reason=This AST also effectively connects the film to its director and birthplace, but it has a slightly lower path score.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- gaby_a_true_story: Gaby: A True Story (entity)
- director: director (type_variable)
- birthplace: birthplace (value_slot)

Edges:
- gaby_a_true_story -> director (director of Gaby: A True Story)
- director -> birthplace (place of birth of the director)

## 11. Atomic Subquestion DAG
- None: Who is the director of Gaby: A True Story?
- None: What is the birthplace of the director of Gaby: A True Story?
