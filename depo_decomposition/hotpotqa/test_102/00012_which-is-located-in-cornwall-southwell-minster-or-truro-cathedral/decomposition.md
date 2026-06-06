# DEPO Decomposition #12

- Dataset: `hotpotqa`
- Question: Which is located in Cornwall, Southwell Minster or Truro Cathedral?
- Gold answer: The Cathedral of the Blessed Virgin Mary, Truro

## 1. Semantic-Normalized Question
Which is located in Cornwall, Southwell Minster or Truro Cathedral?

## 2. Mask Spans
- Southwell Minster (entity, Location)
- Truro Cathedral (entity, Location)

## 3. Selective Masked Question
Which is located in Cornwall, SomeEntityA or SomeEntityB?

## 4. CoreNLP Dependency Parse
- located[3] --nsubj:pass--> Which[1]
- located[3] --aux:pass--> is[2]
- Cornwall[5] --case--> in[4]
- located[3] --obl:in--> Cornwall[5]
- Cornwall[5] --punct--> ,[6]
- located[3] --obl:in--> SomeEntityA[7]
- Cornwall[5] --conj:or--> SomeEntityA[7]
- SomeEntityB[9] --cc--> or[8]
- located[3] --obl:in--> SomeEntityB[9]
- Cornwall[5] --conj:or--> SomeEntityB[9]
- located[3] --punct--> ?[10]

## 5. Undirected Dependency Graph
- Which[1] --nsubj:pass-- located[3]
- is[2] --aux:pass-- located[3]
- located[3] --obl:in-- Cornwall[5]
- located[3] --obl:in-- Southwell Minster[7]
- located[3] --obl:in-- Truro Cathedral[9]
- located[3] --punct-- ?[10]
- in[4] --case-- Cornwall[5]
- Cornwall[5] --punct-- ,[6]
- Cornwall[5] --conj:or-- Southwell Minster[7]
- Cornwall[5] --conj:or-- Truro Cathedral[9]
- or[8] --cc-- Truro Cathedral[9]

## 6. Entity Start Nodes
- e1: Southwell Minster graph_node_ids=['7']
- e2: Truro Cathedral graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Southwell Minster -- Cornwall -- located -- Which
- e1_p2 (e1): Southwell Minster -- located -- Cornwall
- e1_p3 (e1): Southwell Minster -- Cornwall -- located
- e1_p4 (e1): Southwell Minster -- located -- Cornwall -- in
- e1_p5 (e1): Southwell Minster -- located -- Cornwall -- ,
- e1_p6 (e1): Southwell Minster -- Cornwall -- located -- is
- e1_p7 (e1): Southwell Minster -- Cornwall -- located -- ?
- e1_p8 (e1): Southwell Minster -- located -- Which
- e1_p9 (e1): Southwell Minster -- located
- e1_p10 (e1): Southwell Minster -- Cornwall
- e1_p11 (e1): Southwell Minster -- located -- is
- e1_p12 (e1): Southwell Minster -- located -- ?
- e1_p13 (e1): Southwell Minster -- Cornwall -- in
- e1_p14 (e1): Southwell Minster -- Cornwall -- ,
- e1_p15 (e1): Southwell Minster -- located -- Cornwall -- Truro Cathedral
- e1_p16 (e1): Southwell Minster -- Cornwall -- located -- Truro Cathedral
- e1_p17 (e1): Southwell Minster -- located -- Truro Cathedral
- e1_p18 (e1): Southwell Minster -- Cornwall -- Truro Cathedral
- e1_p19 (e1): Southwell Minster -- Cornwall -- Truro Cathedral -- located -- Which
- e1_p20 (e1): Southwell Minster -- located -- Truro Cathedral -- Cornwall
- e1_p21 (e1): Southwell Minster -- Cornwall -- Truro Cathedral -- located
- e1_p22 (e1): Southwell Minster -- located -- Cornwall -- Truro Cathedral -- or
- e1_p23 (e1): Southwell Minster -- located -- Truro Cathedral -- Cornwall -- in
- e1_p24 (e1): Southwell Minster -- located -- Truro Cathedral -- Cornwall -- ,
- e1_p25 (e1): Southwell Minster -- Cornwall -- located -- Truro Cathedral -- or
- e1_p26 (e1): Southwell Minster -- Cornwall -- Truro Cathedral -- located -- is
- e1_p27 (e1): Southwell Minster -- Cornwall -- Truro Cathedral -- located -- ?
- e1_p28 (e1): Southwell Minster -- located -- Truro Cathedral -- or
- e1_p29 (e1): Southwell Minster -- Cornwall -- Truro Cathedral -- or
- e2_p1 (e2): Truro Cathedral -- Cornwall -- located -- Which
- e2_p2 (e2): Truro Cathedral -- located -- Cornwall
- e2_p3 (e2): Truro Cathedral -- Cornwall -- located
- e2_p4 (e2): Truro Cathedral -- located -- Cornwall -- in
- e2_p5 (e2): Truro Cathedral -- located -- Cornwall -- ,
- e2_p6 (e2): Truro Cathedral -- Cornwall -- located -- is
- e2_p7 (e2): Truro Cathedral -- Cornwall -- located -- ?
- e2_p8 (e2): Truro Cathedral -- located -- Which
- e2_p9 (e2): Truro Cathedral -- located
- e2_p10 (e2): Truro Cathedral -- Cornwall
- e2_p11 (e2): Truro Cathedral -- located -- is
- e2_p12 (e2): Truro Cathedral -- located -- ?
- e2_p13 (e2): Truro Cathedral -- Cornwall -- in
- e2_p14 (e2): Truro Cathedral -- Cornwall -- ,
- e2_p15 (e2): Truro Cathedral -- or
- e2_p16 (e2): Truro Cathedral -- located -- Cornwall -- Southwell Minster
- e2_p17 (e2): Truro Cathedral -- Cornwall -- located -- Southwell Minster
- e2_p18 (e2): Truro Cathedral -- located -- Southwell Minster
- e2_p19 (e2): Truro Cathedral -- Cornwall -- Southwell Minster
- e2_p20 (e2): Truro Cathedral -- Cornwall -- Southwell Minster -- located -- Which
- e2_p21 (e2): Truro Cathedral -- located -- Southwell Minster -- Cornwall
- e2_p22 (e2): Truro Cathedral -- Cornwall -- Southwell Minster -- located
- e2_p23 (e2): Truro Cathedral -- located -- Southwell Minster -- Cornwall -- in
- e2_p24 (e2): Truro Cathedral -- located -- Southwell Minster -- Cornwall -- ,
- e2_p25 (e2): Truro Cathedral -- Cornwall -- Southwell Minster -- located -- is
- e2_p26 (e2): Truro Cathedral -- Cornwall -- Southwell Minster -- located -- ?

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p2 score=75.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster and reaches Cornwall, covering the located predicate but missing the which cue.
- e1: e1_p3 score=75.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, and covers the located predicate, including the which cue.
- e1: e1_p4 score=70.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster and reaches Cornwall, covering the located predicate but missing the which cue.
- e1: e1_p5 score=30.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster and reaches Cornwall, but ends with punctuation, missing the which cue.
- e1: e1_p6 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p7 score=30.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster and reaches Cornwall, but ends with punctuation, missing the which cue.
- e1: e1_p8 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p9 score=30.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster and reaches Cornwall, but stops too early, missing the which cue.
- e1: e1_p10 score=30.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster and reaches Cornwall, but stops too early, missing the which cue.
- e1: e1_p11 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p12 score=30.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster and reaches Cornwall, but stops too early, missing the which cue.
- e1: e1_p13 score=70.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster and reaches Cornwall, covering the located predicate but missing the which cue.
- e1: e1_p14 score=30.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster and reaches Cornwall, but stops too early, missing the which cue.
- e1: e1_p15 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p16 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p17 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p18 score=70.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster and reaches Cornwall, covering the located predicate but missing the which cue.
- e1: e1_p19 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p20 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p21 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p22 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p23 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p24 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p25 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p26 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p27 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p28 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e1: e1_p29 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p1 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p2 score=75.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral and reaches Cornwall, covering the located predicate but missing the which cue.
- e2: e2_p3 score=75.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, and covers the located predicate, including the which cue.
- e2: e2_p4 score=70.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral and reaches Cornwall, covering the located predicate but missing the which cue.
- e2: e2_p5 score=30.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral and reaches Cornwall, but ends with punctuation, missing the which cue.
- e2: e2_p6 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p7 score=30.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral and reaches Cornwall, but ends with punctuation, missing the which cue.
- e2: e2_p8 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p9 score=30.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral and reaches Cornwall, but stops too early, missing the which cue.
- e2: e2_p10 score=30.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral and reaches Cornwall, but stops too early, missing the which cue.
- e2: e2_p11 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p12 score=30.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral and reaches Cornwall, but stops too early, missing the which cue.
- e2: e2_p13 score=70.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral and reaches Cornwall, covering the located predicate but missing the which cue.
- e2: e2_p14 score=30.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral and reaches Cornwall, but stops too early, missing the which cue.
- e2: e2_p15 score=30.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral and reaches Cornwall, but stops too early, missing the which cue.
- e2: e2_p16 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p17 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p18 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p19 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p20 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p21 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p22 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p23 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p24 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p25 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.
- e2: e2_p26 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, covers the located predicate, and includes the which cue.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p11
- e2: e2_p1, e2_p11

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1', 'e2': 'e2_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p1', 'e2': 'e2_p11'} mean_path_score=90.0
- ps3: {'e1': 'e1_p11', 'e2': 'e2_p1'} mean_path_score=90.0
- ps4: {'e1': 'e1_p11', 'e2': 'e2_p11'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- southwell_minster -> location (located in)
- truro_cathedral -> location_2 (located in)
### ast_ps2 (ps2)
- southwell_minster -> cornwall (located in)
- truro_cathedral -> cornwall_2 (located in)
### ast_ps3 (ps3)
- southwell_minster -> cornwall_1 (located in)
- truro_cathedral -> cornwall_2 (located in)
### ast_ps4 (ps4)
- southwell_minster -> location_e1 (located in)
- truro_cathedral -> location_e2 (located in)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively covers the locations of both Southwell Minster and Truro Cathedral in Cornwall, allowing for clear decomposition into atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- southwell_minster: Southwell Minster (entity)
- location: location (value_slot)
- truro_cathedral: Truro Cathedral (entity)
- location_2: location (value_slot)

Edges:
- southwell_minster -> location (located in)
- truro_cathedral -> location_2 (located in)

## 11. Atomic Subquestion DAG
- None: Where is Southwell Minster located?
- None: Where is Truro Cathedral located?
