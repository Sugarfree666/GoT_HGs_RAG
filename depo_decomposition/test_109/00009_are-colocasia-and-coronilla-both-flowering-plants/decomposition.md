# DEPO Decomposition #9

- Dataset: `hotpotqa`
- Question: Are Colocasia and Coronilla both flowering plants?
- Gold answer: yes

## 1. Semantic-Normalized Question
Are Colocasia and Coronilla both flowering plants?

## 2. Explicit Entities
- Colocasia (Entity) span=(4, 13)
- Coronilla (Entity) span=(18, 27)

## 3. Entity Masking
- EntityA -> Colocasia
- EntityB -> Coronilla

Are EntityA and EntityB both flowering plants?

## 4. CoreNLP Dependency Parse
- plants[7] --cop--> Are[1]
- plants[7] --nsubj--> EntityA[2]
- EntityB[4] --cc--> and[3]
- EntityA[2] --conj:and--> EntityB[4]
- plants[7] --nsubj--> EntityB[4]
- plants[7] --dep--> both[5]
- plants[7] --compound--> flowering[6]
- plants[7] --punct--> ?[8]

## 5. Undirected Dependency Graph
- Are[1] --cop-- plants[7]
- Colocasia[2] --nsubj-- plants[7]
- Colocasia[2] --conj:and-- Coronilla[4]
- and[3] --cc-- Coronilla[4]
- Coronilla[4] --nsubj-- plants[7]
- both[5] --dep-- plants[7]
- flowering[6] --compound-- plants[7]
- plants[7] --punct-- ?[8]

## 6. Entity Start Nodes from Explicit Entities
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
  Reason: The path starts from Colocasia and reaches 'plants' and 'both', but it misses the 'flowering' cue.
- e1: e1_p2 score=75.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia and reaches 'plants' and 'flowering', but it misses the 'both' cue.
- e1: e1_p3 score=55.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia and reaches 'plants', but it does not cover the necessary cues 'both' and 'flowering'.
- e1: e1_p4 score=30.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia and reaches 'plants' and 'Are', but it does not cover the necessary cues 'both' and 'flowering'.
- e1: e1_p5 score=30.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia and reaches 'plants' and '?', but it does not cover the necessary cues 'both' and 'flowering'.
- e1: e1_p6 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, reaches 'plants', and includes 'Coronilla' while covering the 'both' cue.
- e1: e1_p7 score=55.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia and reaches 'Coronilla', but it does not cover the necessary cues 'both' and 'flowering'.
- e1: e1_p8 score=95.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, reaches 'Coronilla', covers 'plants', and includes the 'both' cue.
- e1: e1_p9 score=95.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, reaches 'Coronilla', covers 'plants', and includes the 'flowering' cue.
- e1: e1_p10 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, reaches 'Coronilla', and covers 'plants', but it does not cover the necessary cues 'both' and 'flowering'.
- e1: e1_p11 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, reaches 'Coronilla', covers 'plants', and includes the 'Are' cue, but it does not cover the necessary 'both' cue.
- e1: e1_p12 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, reaches 'Coronilla', covers 'plants', and includes the '?' cue, but it does not cover the necessary 'both' cue.
- e1: e1_p13 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, reaches 'Coronilla', covers 'plants', and includes the 'and' cue, but it does not cover the necessary 'both' cue.
- e1: e1_p14 score=75.0 valid=True terminal=flowering
  Reason: The path starts from Colocasia, reaches 'Coronilla', and includes 'and', but it does not cover the necessary 'both' cue.
- e2: e2_p1 score=75.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla and reaches 'plants' and 'both', but it misses the 'flowering' cue.
- e2: e2_p2 score=75.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla and reaches 'plants' and 'flowering', but it misses the 'both' cue.
- e2: e2_p3 score=55.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla and reaches 'plants', but it does not cover the necessary cues 'both' and 'flowering'.
- e2: e2_p4 score=30.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla and reaches 'plants' and 'Are', but it does not cover the necessary cues 'both' and 'flowering'.
- e2: e2_p5 score=30.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla and reaches 'plants' and '?', but it does not cover the necessary cues 'both' and 'flowering'.
- e2: e2_p6 score=30.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla and reaches 'and', but it does not cover the necessary cues 'both' and 'flowering'.
- e2: e2_p7 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, reaches 'plants', and includes 'Colocasia' while covering the 'both' cue.
- e2: e2_p8 score=55.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla and reaches 'Colocasia', but it does not cover the necessary cues 'both' and 'flowering'.
- e2: e2_p9 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, reaches 'Colocasia', covers 'plants', and includes the 'both' cue.
- e2: e2_p10 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, reaches 'Colocasia', covers 'plants', and includes the 'flowering' cue, but it does not cover the necessary 'both' cue.
- e2: e2_p11 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, reaches 'Colocasia', covers 'plants', and includes the 'Are' cue, but it does not cover the necessary 'both' cue.
- e2: e2_p12 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, reaches 'Colocasia', covers 'plants', and includes the '?' cue, but it does not cover the necessary 'both' cue.
- e2: e2_p13 score=90.0 valid=True terminal=flowering
  Reason: The path starts from Coronilla, reaches 'Colocasia', covers 'plants', and includes the 'and' cue, but it does not cover the necessary 'both' cue.

## 8.1 Top-2 Paths per Entity
- e1: e1_p8, e1_p9
- e2: e2_p10, e2_p11

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p8', 'e2': 'e2_p10'} mean_path_score=92.5
- ps2: {'e1': 'e1_p8', 'e2': 'e2_p11'} mean_path_score=92.5
- ps3: {'e1': 'e1_p9', 'e2': 'e2_p10'} mean_path_score=92.5
- ps4: {'e1': 'e1_p9', 'e2': 'e2_p11'} mean_path_score=92.5

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Are Colocasia and Coronilla both flowering plants?
- ps1
  - e1_p8: Colocasia -> Coronilla -> plants -> both
  - e2_p10: Coronilla -> Colocasia -> plants -> flowering
- ps2
  - e1_p8: Colocasia -> Coronilla -> plants -> both
  - e2_p11: Coronilla -> Colocasia -> plants
- ps3
  - e1_p9: Colocasia -> Coronilla -> plants -> flowering
  - e2_p10: Coronilla -> Colocasia -> plants -> flowering
- ps4
  - e1_p9: Colocasia -> Coronilla -> plants -> flowering
  - e2_p11: Coronilla -> Colocasia -> plants

Output:
- selected_path_set_ids: ['ps1', 'ps3']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Is Colocasia a flowering plant? depends_on=[] support=['e1_p8']
- q2: Is Coronilla a flowering plant? depends_on=[] support=['e2_p10']

## 10. Atomic Subquestion DAG
- None: Is Colocasia a flowering plant?
- None: Is Coronilla a flowering plant?
