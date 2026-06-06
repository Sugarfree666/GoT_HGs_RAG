# DEPO Decomposition #9

- Dataset: `hotpotqa`
- Question: Are Colocasia and Coronilla both flowering plants?
- Gold answer: yes

## 1. Semantic-Normalized Question
Are Colocasia and Coronilla both flowering plants?

## 2. Mask Spans
(none)

## 3. Selective Masked Question
Are Colocasia and Coronilla both flowering plants?

## 4. CoreNLP Dependency Parse
- plants[7] --cop--> Are[1]
- plants[7] --nsubj--> Colocasia[2]
- Coronilla[4] --cc--> and[3]
- Colocasia[2] --conj:and--> Coronilla[4]
- plants[7] --nsubj--> Coronilla[4]
- plants[7] --det--> both[5]
- plants[7] --compound--> flowering[6]
- plants[7] --punct--> ?[8]

## 5. Undirected Dependency Graph
- Are[1] --cop-- plants[7]
- Colocasia[2] --nsubj-- plants[7]
- Colocasia[2] --conj:and-- Coronilla[4]
- and[3] --cc-- Coronilla[4]
- Coronilla[4] --nsubj-- plants[7]
- both[5] --det-- plants[7]
- flowering[6] --compound-- plants[7]
- plants[7] --punct-- ?[8]

## 6. Entity Start Nodes
- e1: Colocasia graph_node_ids=['2']
- e2: Coronilla graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Colocasia -- plants -- both
- e1_p2 (e1): Colocasia -- plants -- flowering
- e1_p3 (e1): Colocasia -- plants
- e1_p4 (e1): Colocasia -- plants -- Are
- e1_p5 (e1): Colocasia -- plants -- ?
- e1_p6 (e1): Colocasia -- plants -- Coronilla
- e1_p7 (e1): Colocasia -- Coronilla
- e1_p8 (e1): Colocasia -- Coronilla -- plants -- both
- e1_p9 (e1): Colocasia -- Coronilla -- plants -- flowering
- e1_p10 (e1): Colocasia -- Coronilla -- plants
- e1_p11 (e1): Colocasia -- Coronilla -- plants -- Are
- e1_p12 (e1): Colocasia -- Coronilla -- plants -- ?
- e1_p13 (e1): Colocasia -- plants -- Coronilla -- and
- e1_p14 (e1): Colocasia -- Coronilla -- and
- e2_p1 (e2): Coronilla -- plants -- both
- e2_p2 (e2): Coronilla -- plants -- flowering
- e2_p3 (e2): Coronilla -- plants
- e2_p4 (e2): Coronilla -- plants -- Are
- e2_p5 (e2): Coronilla -- plants -- ?
- e2_p6 (e2): Coronilla -- and
- e2_p7 (e2): Coronilla -- plants -- Colocasia
- e2_p8 (e2): Coronilla -- Colocasia
- e2_p9 (e2): Coronilla -- Colocasia -- plants -- both
- e2_p10 (e2): Coronilla -- Colocasia -- plants -- flowering
- e2_p11 (e2): Coronilla -- Colocasia -- plants
- e2_p12 (e2): Coronilla -- Colocasia -- plants -- Are
- e2_p13 (e2): Coronilla -- Colocasia -- plants -- ?

## 8. LLM Path Scores
- e1: e1_p1 score=75.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia and reaches 'both', but it misses the key cue 'flowering'.
- e1: e1_p2 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, reaches 'plants', and includes the key cue 'flowering', supporting the answer intent.
- e1: e1_p3 score=55.0 valid=True terminal=flowering
  Reason: The path is too short and does not cover the necessary cues for the answer intent.
- e1: e1_p4 score=30.0 valid=True terminal=flowering
  Reason: The path ends with an auxiliary and does not provide sufficient coverage for the answer intent.
- e1: e1_p5 score=30.0 valid=True terminal=flowering
  Reason: The path ends with punctuation and lacks necessary coverage for the answer intent.
- e1: e1_p6 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, reaches 'plants', and includes Coronilla, supporting the answer intent.
- e1: e1_p7 score=55.0 valid=True terminal=flowering
  Reason: The path is too short and does not cover the necessary cues for the answer intent.
- e1: e1_p8 score=95.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, includes Coronilla, reaches 'plants', and covers 'both', supporting the answer intent.
- e1: e1_p9 score=95.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, includes Coronilla, reaches 'plants', and covers 'flowering', supporting the answer intent.
- e1: e1_p10 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, includes Coronilla, and reaches 'plants', but misses the cue 'both'.
- e1: e1_p11 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, includes Coronilla, reaches 'plants', and covers the auxiliary 'Are', but misses the cue 'both'.
- e1: e1_p12 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, includes Coronilla, reaches 'plants', and covers the punctuation '?', but misses the cue 'both'.
- e1: e1_p13 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, includes Coronilla, reaches 'plants', and covers the conjunction 'and', but misses the cue 'both'.
- e1: e1_p14 score=55.0 valid=True terminal=flowering
  Reason: The path is too short and does not cover the necessary cues for the answer intent.
- e2: e2_p1 score=75.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla and reaches 'both', but it misses the key cue 'flowering'.
- e2: e2_p2 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, reaches 'plants', and includes the key cue 'flowering', supporting the answer intent.
- e2: e2_p3 score=55.0 valid=True terminal=flowering
  Reason: The path is too short and does not cover the necessary cues for the answer intent.
- e2: e2_p4 score=30.0 valid=True terminal=flowering
  Reason: The path ends with an auxiliary and does not provide sufficient coverage for the answer intent.
- e2: e2_p5 score=30.0 valid=True terminal=flowering
  Reason: The path ends with punctuation and lacks necessary coverage for the answer intent.
- e2: e2_p6 score=30.0 valid=True terminal=flowering
  Reason: The path is too short and does not cover the necessary cues for the answer intent.
- e2: e2_p7 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, reaches 'plants', and includes Colocasia, supporting the answer intent.
- e2: e2_p8 score=55.0 valid=True terminal=flowering
  Reason: The path is too short and does not cover the necessary cues for the answer intent.
- e2: e2_p9 score=95.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, includes Colocasia, reaches 'plants', and covers 'both', supporting the answer intent.
- e2: e2_p10 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, includes Colocasia, and reaches 'plants', but misses the cue 'both'.
- e2: e2_p11 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, includes Colocasia, reaches 'plants', and covers the auxiliary 'Are', but misses the cue 'both'.
- e2: e2_p12 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, includes Colocasia, reaches 'plants', and covers the punctuation '?', but misses the cue 'both'.
- e2: e2_p13 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, includes Colocasia, reaches 'plants', and covers the conjunction 'and', but misses the cue 'both'.

## 8.1 Top-2 Paths per Entity
- e1: e1_p8, e1_p9
- e2: e2_p9, e2_p10

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p8', 'e2': 'e2_p9'} mean_path_score=95.0
- ps2: {'e1': 'e1_p8', 'e2': 'e2_p10'} mean_path_score=92.5
- ps3: {'e1': 'e1_p9', 'e2': 'e2_p9'} mean_path_score=95.0
- ps4: {'e1': 'e1_p9', 'e2': 'e2_p10'} mean_path_score=92.5

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- colocasia -> plants (type of plant)
- coronilla -> plants_2 (type of plant)
- plants -> both_e1 (both)
- plants_2 -> both_e2 (both)
### ast_ps2 (ps2)
- colocasia -> plants_e1 (type of plant)
- coronilla -> plants_e2 (type of plant)
- plants_e1 -> flowering_e1 (is flowering)
- plants_e2 -> flowering_e2 (is flowering)
### ast_ps3 (ps3)
- colocasia -> plants_e1 (plants of Colocasia)
- coronilla -> plants_e2 (plants of Coronilla)
- plants_e1 -> flowering_e1 (type of plants)
- plants_e2 -> flowering_e2 (type of plants)
- plants_e1 -> both_e1 (both types of plants)
- plants_e2 -> both_e2 (both types of plants)
### ast_ps4 (ps4)
- colocasia -> plants_e1 (plants of Colocasia)
- coronilla -> plants_e2 (plants of Coronilla)
- plants_e1 -> flowering_e1 (characteristic of plants)
- plants_e2 -> flowering_e2 (characteristic of plants)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively covers both entities, Colocasia and Coronilla, and their relationship to being flowering plants, allowing for the decomposition into atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- colocasia: Colocasia (entity)
- coronilla: Coronilla (entity)
- plants: plants (value_slot)
- plants_2: plants (value_slot)
- both_e1: both (value_slot)
- both_e2: both (value_slot)

Edges:
- colocasia -> plants (type of plant)
- coronilla -> plants_2 (type of plant)
- plants -> both_e1 (both)
- plants_2 -> both_e2 (both)

## 11. Atomic Subquestion DAG
- None: What type of plant is Colocasia?
- None: Are the plants of Colocasia both flowering plants?
- None: What type of plant is Coronilla?
- None: Are the plants of Coronilla both flowering plants?
