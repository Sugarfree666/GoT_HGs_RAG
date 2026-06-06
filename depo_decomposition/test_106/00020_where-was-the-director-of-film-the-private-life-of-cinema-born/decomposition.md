# DEPO Decomposition #20

- Dataset: `2wikimultihopqa`
- Question: Where was the director of film The Private Life Of Cinema born?
- Gold answer: Montreal, Quebec

## 1. Semantic-Normalized Question
Where was the director of the film The Private Life Of Cinema born?

## 2. Explicit Entities
- The Private Life Of Cinema (Film) span=(35, 61)

## 3. Entity Masking
- FilmA -> The Private Life Of Cinema

Where was the director of the film FilmA born?

## 4. CoreNLP Dependency Parse
- was[2] --advmod--> Where[1]
- born[9] --aux:pass--> was[2]
- director[4] --det--> the[3]
- born[9] --nsubj:pass--> director[4]
- FilmA[8] --case--> of[5]
- FilmA[8] --det--> the[6]
- FilmA[8] --compound--> film[7]
- director[4] --nmod:of--> FilmA[8]
- born[9] --punct--> ?[10]

## 5. Undirected Dependency Graph
- Where[1] --advmod-- was[2]
- was[2] --aux:pass-- born[9]
- the[3] --det-- director[4]
- director[4] --nsubj:pass-- born[9]
- director[4] --nmod:of-- The Private Life Of Cinema[8]
- of[5] --case-- The Private Life Of Cinema[8]
- the[6] --det-- The Private Life Of Cinema[8]
- film[7] --compound-- The Private Life Of Cinema[8]
- born[9] --punct-- ?[10]

## 6. Entity Start Nodes from Explicit Entities
- e1: The Private Life Of Cinema graph_node_ids=['8']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): The Private Life Of Cinema -- director -- born -- was -- Where
- e1_p2 (e1): The Private Life Of Cinema -- director -- born
- e1_p3 (e1): The Private Life Of Cinema -- director -- born -- was
- e1_p4 (e1): The Private Life Of Cinema -- director -- born -- ?
- e1_p5 (e1): The Private Life Of Cinema -- director
- e1_p6 (e1): The Private Life Of Cinema -- director -- the
- e1_p7 (e1): The Private Life Of Cinema -- film
- e1_p8 (e1): The Private Life Of Cinema -- of
- e1_p9 (e1): The Private Life Of Cinema -- the

## 8. LLM Path Scores
- e1: e1_p1 score=85.0 valid=True terminal=birth_location
  Reason: The path starts from The Private Life Of Cinema, includes the director and the born predicate, and covers the where cue.
- e1: e1_p2 score=75.0 valid=True terminal=birth_location
  Reason: The path starts from The Private Life Of Cinema and includes the director and born, but it misses the where cue.
- e1: e1_p3 score=90.0 valid=True terminal=birth_location
  Reason: The path starts from The Private Life Of Cinema, includes the director, the born predicate, and the where cue.
- e1: e1_p4 score=70.0 valid=True terminal=birth_location
  Reason: The path starts from The Private Life Of Cinema and includes the director and born, but it ends with a punctuation mark and misses the where cue.
- e1: e1_p5 score=30.0 valid=True terminal=birth_location
  Reason: The path only includes The Private Life Of Cinema and the director, missing all necessary cues and predicates.
- e1: e1_p6 score=40.0 valid=True terminal=birth_location
  Reason: The path includes The Private Life Of Cinema, the director, and a determiner, but misses all necessary cues and predicates.
- e1: e1_p7 score=20.0 valid=True terminal=birth_location
  Reason: The path only includes The Private Life Of Cinema and the film, missing all necessary cues and predicates.
- e1: e1_p8 score=10.0 valid=True terminal=birth_location
  Reason: The path only includes The Private Life Of Cinema and a preposition, missing all necessary cues and predicates.
- e1: e1_p9 score=10.0 valid=True terminal=birth_location
  Reason: The path only includes The Private Life Of Cinema and a determiner, missing all necessary cues and predicates.

## 8.1 Top-2 Paths per Entity
- e1: e1_p3, e1_p1

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p3'} mean_path_score=90.0
- ps2: {'e1': 'e1_p1'} mean_path_score=85.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- the_private_life_of_cinema -> director (director of The Private Life Of Cinema)
- director -> birthplace (place where the director was born)
### ast_ps2 (ps2)
- the_private_life_of_cinema -> director (director of The Private Life Of Cinema)
- director -> birthplace (place where the director was born)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST decomposes the question into the necessary components: the film, the director, and the birthplace, covering all required cues and allowing for executable atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- the_private_life_of_cinema: The Private Life Of Cinema (entity)
- director: director (type_variable)
- birthplace: birthplace (value_slot)

Edges:
- the_private_life_of_cinema -> director (director of The Private Life Of Cinema)
- director -> birthplace (place where the director was born)

## 11. Atomic Subquestion DAG
- None: Who is the director of The Private Life Of Cinema?
- None: Where was the director of The Private Life Of Cinema born?
