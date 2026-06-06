# DEPO Decomposition #20

- Dataset: `2wikimultihopqa`
- Question: Where was the director of film The Private Life Of Cinema born?
- Gold answer: Montreal, Quebec

## 1. Semantic-Normalized Question
Where was the director of the film The Private Life Of Cinema born?

## 2. Mask Spans
- The Private Life Of Cinema born? (entity, Film)

## 3. Selective Masked Question
Where was the director of the film MovieA

## 4. CoreNLP Dependency Parse
- was[2] --advmod--> Where[1]
- director[4] --det--> the[3]
- was[2] --nsubj--> director[4]
- MovieA[8] --case--> of[5]
- MovieA[8] --det--> the[6]
- MovieA[8] --compound--> film[7]
- director[4] --nmod:of--> MovieA[8]

## 5. Undirected Dependency Graph
- Where[1] --advmod-- was[2]
- was[2] --nsubj-- director[4]
- the[3] --det-- director[4]
- director[4] --nmod:of-- The Private Life Of Cinema born?[8]
- of[5] --case-- The Private Life Of Cinema born?[8]
- the[6] --det-- The Private Life Of Cinema born?[8]
- film[7] --compound-- The Private Life Of Cinema born?[8]

## 6. Entity Start Nodes
- e1: The Private Life Of Cinema born? graph_node_ids=['8']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): The Private Life Of Cinema born? -- director -- was -- Where
- e1_p2 (e1): The Private Life Of Cinema born? -- director
- e1_p3 (e1): The Private Life Of Cinema born? -- director -- was
- e1_p4 (e1): The Private Life Of Cinema born? -- director -- the
- e1_p5 (e1): The Private Life Of Cinema born? -- film
- e1_p6 (e1): The Private Life Of Cinema born? -- of
- e1_p7 (e1): The Private Life Of Cinema born? -- the

## 8. LLM Path Scores
- e1: e1_p1 score=70.0 valid=True terminal=birth_location
  Reason: The path starts from the film, identifies the director, and includes the 'where' cue, but it lacks a direct link to the birth location.
- e1: e1_p2 score=50.0 valid=True terminal=birth_location
  Reason: The path identifies the director but does not connect to the birth location or include the 'where' cue.
- e1: e1_p3 score=80.0 valid=True terminal=birth_location
  Reason: The path starts from the film, identifies the director, and includes the 'where' cue, but it still lacks a direct link to the birth location.
- e1: e1_p4 score=40.0 valid=True terminal=birth_location
  Reason: The path identifies the director but does not connect to the birth location or include the 'where' cue.
- e1: e1_p5 score=30.0 valid=True terminal=birth_location
  Reason: The path identifies the film but does not connect to the director or the birth location.
- e1: e1_p6 score=20.0 valid=True terminal=birth_location
  Reason: The path includes a preposition but does not connect to the director or the birth location.
- e1: e1_p7 score=25.0 valid=True terminal=birth_location
  Reason: The path includes a determiner but does not connect to the director or the birth location.

## 8.1 Top-2 Paths per Entity
- e1: e1_p3, e1_p1

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p3'} mean_path_score=80.0
- ps2: {'e1': 'e1_p1'} mean_path_score=70.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- the_private_life_of_cinema -> director (director of The Private Life Of Cinema)
- director -> birthplace (birthplace of the director)
### ast_ps2 (ps2)
- the_private_life_of_cinema -> director (director of The Private Life Of Cinema)
- director -> birthplace (birthplace of the director)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively identifies the director of the film and their birthplace, aligning with the original question's intent and allowing for one-hop atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- the_private_life_of_cinema: The Private Life Of Cinema (entity)
- director: director (type_variable)
- birthplace: birthplace (value_slot)

Edges:
- the_private_life_of_cinema -> director (director of The Private Life Of Cinema)
- director -> birthplace (birthplace of the director)

## 11. Atomic Subquestion DAG
- None: Who is the director of The Private Life Of Cinema?
- None: Where was the director of The Private Life Of Cinema born?
