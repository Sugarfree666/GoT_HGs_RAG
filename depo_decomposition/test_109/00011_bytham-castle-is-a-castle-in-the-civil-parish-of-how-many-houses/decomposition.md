# DEPO Decomposition #11

- Dataset: `hotpotqa`
- Question: Bytham Castle is a castle in the civil parish of how many houses?
- Gold answer: 300

## 1. Semantic-Normalized Question
Bytham Castle is a castle in the civil parish of how many houses?

## 2. Explicit Entities
- Bytham Castle (Location) span=(0, 13)

## 3. Entity Masking
- LocationA -> Bytham Castle

LocationA is a castle in the civil parish of how many houses?

## 4. CoreNLP Dependency Parse
- castle[4] --nsubj--> LocationA[1]
- castle[4] --cop--> is[2]
- castle[4] --det--> a[3]
- parish[8] --case--> in[5]
- parish[8] --det--> the[6]
- parish[8] --amod--> civil[7]
- castle[4] --nmod:in--> parish[8]
- houses[12] --case--> of[9]
- many[11] --advmod--> how[10]
- houses[12] --amod--> many[11]
- parish[8] --nmod:of--> houses[12]
- castle[4] --punct--> ?[13]

## 5. Undirected Dependency Graph
- Bytham Castle[1] --nsubj-- castle[4]
- is[2] --cop-- castle[4]
- a[3] --det-- castle[4]
- castle[4] --nmod:in-- parish[8]
- castle[4] --punct-- ?[13]
- in[5] --case-- parish[8]
- the[6] --det-- parish[8]
- civil[7] --amod-- parish[8]
- parish[8] --nmod:of-- houses[12]
- of[9] --case-- houses[12]
- how[10] --advmod-- many[11]
- many[11] --amod-- houses[12]

## 6. Entity Start Nodes from Explicit Entities
- e1: Bytham Castle graph_node_ids=['1']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Bytham Castle -- castle -- parish -- houses -- many -- how
- e1_p2 (e1): Bytham Castle -- castle -- parish -- houses -- many
- e1_p3 (e1): Bytham Castle -- castle -- parish -- civil
- e1_p4 (e1): Bytham Castle -- castle -- parish -- houses
- e1_p5 (e1): Bytham Castle -- castle -- parish -- houses -- of
- e1_p6 (e1): Bytham Castle -- castle -- parish
- e1_p7 (e1): Bytham Castle -- castle -- parish -- in
- e1_p8 (e1): Bytham Castle -- castle -- parish -- the
- e1_p9 (e1): Bytham Castle -- castle
- e1_p10 (e1): Bytham Castle -- castle -- is
- e1_p11 (e1): Bytham Castle -- castle -- a
- e1_p12 (e1): Bytham Castle -- castle -- ?

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=houses_count
  Reason: The path starts from Bytham Castle, covers the necessary roles leading to houses, and includes the how many cue for the count.
- e1: e1_p2 score=85.0 valid=True terminal=houses_count
  Reason: The path starts from Bytham Castle and leads to houses, covering the how many cue, but lacks the final modifier 'how' for full clarity.
- e1: e1_p3 score=70.0 valid=True terminal=houses_count
  Reason: The path starts from Bytham Castle and reaches civil, but it does not cover the necessary cues for the count question.
- e1: e1_p4 score=80.0 valid=True terminal=houses_count
  Reason: The path starts from Bytham Castle and leads to houses, but it misses the how many cue for the count.
- e1: e1_p5 score=60.0 valid=True terminal=houses_count
  Reason: The path starts from Bytham Castle and leads to houses, but it includes 'of', which is not useful for the answer intent.
- e1: e1_p6 score=50.0 valid=True terminal=houses_count
  Reason: The path starts from Bytham Castle and leads to parish, but it does not reach houses or cover the necessary cues.
- e1: e1_p7 score=40.0 valid=True terminal=houses_count
  Reason: The path starts from Bytham Castle and leads to parish, but it ends with 'in', which does not contribute to the answer intent.
- e1: e1_p8 score=30.0 valid=True terminal=houses_count
  Reason: The path starts from Bytham Castle and leads to parish, but it ends with 'the', which does not support the answer intent.
- e1: e1_p9 score=20.0 valid=True terminal=houses_count
  Reason: The path starts from Bytham Castle but only leads to 'castle', which does not support the answer intent.
- e1: e1_p10 score=25.0 valid=True terminal=houses_count
  Reason: The path starts from Bytham Castle and leads to 'is', which does not contribute to the answer intent.
- e1: e1_p11 score=15.0 valid=True terminal=houses_count
  Reason: The path starts from Bytham Castle and leads to 'a', which does not support the answer intent.
- e1: e1_p12 score=10.0 valid=True terminal=houses_count
  Reason: The path starts from Bytham Castle and ends with '?', which does not support the answer intent.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p2'} mean_path_score=85.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Bytham Castle is a castle in the civil parish of how many houses?
- ps1
  - e1_p1: Bytham Castle -> castle -> parish -> houses -> many -> how
- ps2
  - e1_p2: Bytham Castle -> castle -> parish -> houses -> many

Output:
- selected_path_set_ids: ['ps1']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: What is the castle in the civil parish of Bytham Castle? depends_on=[] support=['e1_p1']
- q2: How many houses are in the civil parish of q1's answer? depends_on=['q1'] support=['e1_p1']

## 10. Atomic Subquestion DAG
- None: What is the castle in the civil parish of Bytham Castle?
- None: How many houses are in the civil parish of q1's answer?
