# DEPO Decomposition #13

- Dataset: `2wikimultihopqa`
- Question: What nationality is the performer of song When The Stars Go Blue?
- Gold answer: America

## 1. Semantic-Normalized Question
What nationality is the performer of the song When The Stars Go Blue?

## 2. Explicit Entities
- The Stars Go Blue (Entity) span=(51, 68)

## 3. Entity Masking
- EntityA -> The Stars Go Blue

What nationality is the performer of the song When EntityA?

## 4. CoreNLP Dependency Parse
- nationality[2] --det--> What[1]
- is[3] --obj--> nationality[2]
- performer[5] --det--> the[4]
- is[3] --nsubj--> performer[5]
- song[8] --case--> of[6]
- song[8] --det--> the[7]
- performer[5] --nmod:of--> song[8]
- EntityA[10] --advmod--> When[9]
- is[3] --dep--> EntityA[10]
- is[3] --punct--> ?[11]

## 5. Undirected Dependency Graph
- What[1] --det-- nationality[2]
- nationality[2] --obj-- is[3]
- is[3] --nsubj-- performer[5]
- is[3] --dep-- The Stars Go Blue[10]
- is[3] --punct-- ?[11]
- the[4] --det-- performer[5]
- performer[5] --nmod:of-- song[8]
- of[6] --case-- song[8]
- the[7] --det-- song[8]
- When[9] --advmod-- The Stars Go Blue[10]

## 6. Entity Start Nodes from Explicit Entities
- e1: The Stars Go Blue graph_node_ids=['10']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): The Stars Go Blue -- is -- performer -- song
- e1_p2 (e1): The Stars Go Blue -- is -- performer -- song -- of
- e1_p3 (e1): The Stars Go Blue -- is -- performer -- song -- the
- e1_p4 (e1): The Stars Go Blue -- is -- nationality -- What
- e1_p5 (e1): The Stars Go Blue -- is -- nationality
- e1_p6 (e1): The Stars Go Blue -- is -- performer
- e1_p7 (e1): The Stars Go Blue -- is -- performer -- the
- e1_p8 (e1): The Stars Go Blue -- When
- e1_p9 (e1): The Stars Go Blue -- is
- e1_p10 (e1): The Stars Go Blue -- is -- ?

## 8. LLM Path Scores
- e1: e1_p1 score=75.0 valid=True terminal=nationality
  Reason: The path starts from The Stars Go Blue and reaches performer, but it does not cover the nationality aspect needed for the answer.
- e1: e1_p2 score=80.0 valid=True terminal=nationality
  Reason: The path includes the necessary nodes but still lacks a direct connection to the nationality aspect.
- e1: e1_p3 score=70.0 valid=True terminal=nationality
  Reason: The path is somewhat relevant but does not adequately connect to the nationality needed for the answer.
- e1: e1_p4 score=90.0 valid=True terminal=nationality
  Reason: The path effectively connects The Stars Go Blue to nationality and includes the necessary question cue.
- e1: e1_p5 score=85.0 valid=True terminal=nationality
  Reason: The path connects The Stars Go Blue to nationality but lacks additional context that could enhance its completeness.
- e1: e1_p6 score=60.0 valid=True terminal=nationality
  Reason: The path is too short and does not connect to the nationality aspect, making it less useful.
- e1: e1_p7 score=65.0 valid=True terminal=nationality
  Reason: The path includes performer but does not connect to nationality, making it incomplete.
- e1: e1_p8 score=30.0 valid=True terminal=nationality
  Reason: The path is too short and does not provide relevant information for the question.
- e1: e1_p9 score=25.0 valid=True terminal=nationality
  Reason: The path is too short and does not connect to any relevant concepts for the question.
- e1: e1_p10 score=20.0 valid=True terminal=nationality
  Reason: The path ends with punctuation and does not provide any useful information for the question.

## 8.1 Top-2 Paths per Entity
- e1: e1_p4, e1_p5

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p4'} mean_path_score=90.0
- ps2: {'e1': 'e1_p5'} mean_path_score=85.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- the_stars_go_blue -> performer (performer of The Stars Go Blue)
- performer -> nationality (nationality of the performer)
### ast_ps2 (ps2)
- The_Stars_Go_Blue -> performer (performer of The Stars Go Blue)
- performer -> nationality (nationality of the performer)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively connects the song 'The Stars Go Blue' to its performer and their nationality, covering all necessary aspects of the original question.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- the_stars_go_blue: The Stars Go Blue (entity)
- performer: performer (type_variable)
- nationality: nationality (value_slot)

Edges:
- the_stars_go_blue -> performer (performer of The Stars Go Blue)
- performer -> nationality (nationality of the performer)

## 11. Atomic Subquestion DAG
- None: Who is the performer of The Stars Go Blue?
- None: What is the nationality of the performer of When The Stars Go Blue?
