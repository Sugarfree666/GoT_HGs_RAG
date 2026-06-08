# DEPO Decomposition #12

- Dataset: `hotpotqa`
- Question: Which is located in Cornwall, Southwell Minster or Truro Cathedral?
- Gold answer: The Cathedral of the Blessed Virgin Mary, Truro

## 1. Semantic-Normalized Question
Which is located in Cornwall, Southwell Minster or Truro Cathedral?

## 2. Explicit Entities
- Cornwall (Location) span=(20, 28)
- Southwell Minster (Organization) span=(30, 47)
- Truro Cathedral (Organization) span=(51, 66)

## 3. Entity Masking
- LocationA -> Cornwall
- OrganizationA -> Southwell Minster
- OrganizationB -> Truro Cathedral

Which is located in LocationA, OrganizationA or OrganizationB?

## 4. CoreNLP Dependency Parse
- located[3] --nsubj:pass--> Which[1]
- located[3] --aux:pass--> is[2]
- LocationA[5] --case--> in[4]
- located[3] --obl:in--> LocationA[5]
- LocationA[5] --punct--> ,[6]
- located[3] --obl:in--> OrganizationA[7]
- LocationA[5] --conj:or--> OrganizationA[7]
- OrganizationB[9] --cc--> or[8]
- located[3] --obl:in--> OrganizationB[9]
- LocationA[5] --conj:or--> OrganizationB[9]
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

## 6. Entity Start Nodes from Explicit Entities
- e1: Cornwall graph_node_ids=['5']
- e2: Southwell Minster graph_node_ids=['7']
- e3: Truro Cathedral graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Cornwall -- located -- Which
- e1_p2 (e1): Cornwall -- located
- e1_p3 (e1): Cornwall -- located -- is
- e1_p4 (e1): Cornwall -- located -- ?
- e1_p5 (e1): Cornwall -- in
- e1_p6 (e1): Cornwall -- ,
- e1_p7 (e1): Cornwall -- located -- Southwell Minster
- e1_p8 (e1): Cornwall -- located -- Truro Cathedral
- e1_p9 (e1): Cornwall -- Southwell Minster
- e1_p10 (e1): Cornwall -- Truro Cathedral
- e1_p11 (e1): Cornwall -- Southwell Minster -- located -- Which
- e1_p12 (e1): Cornwall -- Truro Cathedral -- located -- Which
- e1_p13 (e1): Cornwall -- Southwell Minster -- located
- e1_p14 (e1): Cornwall -- Truro Cathedral -- located
- e1_p15 (e1): Cornwall -- located -- Truro Cathedral -- or
- e1_p16 (e1): Cornwall -- Southwell Minster -- located -- is
- e1_p17 (e1): Cornwall -- Southwell Minster -- located -- ?
- e1_p18 (e1): Cornwall -- Truro Cathedral -- located -- is
- e1_p19 (e1): Cornwall -- Truro Cathedral -- located -- ?
- e1_p20 (e1): Cornwall -- Truro Cathedral -- or
- e1_p21 (e1): Cornwall -- Southwell Minster -- located -- Truro Cathedral
- e1_p22 (e1): Cornwall -- Truro Cathedral -- located -- Southwell Minster
- e1_p23 (e1): Cornwall -- Southwell Minster -- located -- Truro Cathedral -- or
- e2_p1 (e2): Southwell Minster -- located -- Which
- e2_p2 (e2): Southwell Minster -- located
- e2_p3 (e2): Southwell Minster -- located -- is
- e2_p4 (e2): Southwell Minster -- located -- ?
- e2_p5 (e2): Southwell Minster -- located -- Cornwall
- e2_p6 (e2): Southwell Minster -- located -- Truro Cathedral
- e2_p7 (e2): Southwell Minster -- Cornwall
- e2_p8 (e2): Southwell Minster -- Cornwall -- located -- Which
- e2_p9 (e2): Southwell Minster -- Cornwall -- located
- e2_p10 (e2): Southwell Minster -- located -- Cornwall -- in
- e2_p11 (e2): Southwell Minster -- located -- Cornwall -- ,
- e2_p12 (e2): Southwell Minster -- located -- Truro Cathedral -- or
- e2_p13 (e2): Southwell Minster -- Cornwall -- located -- is
- e2_p14 (e2): Southwell Minster -- Cornwall -- located -- ?
- e2_p15 (e2): Southwell Minster -- Cornwall -- in
- e2_p16 (e2): Southwell Minster -- Cornwall -- ,
- e2_p17 (e2): Southwell Minster -- located -- Cornwall -- Truro Cathedral
- e2_p18 (e2): Southwell Minster -- located -- Truro Cathedral -- Cornwall
- e2_p19 (e2): Southwell Minster -- Cornwall -- located -- Truro Cathedral
- e2_p20 (e2): Southwell Minster -- Cornwall -- Truro Cathedral
- e2_p21 (e2): Southwell Minster -- Cornwall -- Truro Cathedral -- located -- Which
- e2_p22 (e2): Southwell Minster -- Cornwall -- Truro Cathedral -- located
- e2_p23 (e2): Southwell Minster -- located -- Cornwall -- Truro Cathedral -- or
- e2_p24 (e2): Southwell Minster -- located -- Truro Cathedral -- Cornwall -- in
- e2_p25 (e2): Southwell Minster -- located -- Truro Cathedral -- Cornwall -- ,
- e2_p26 (e2): Southwell Minster -- Cornwall -- located -- Truro Cathedral -- or
- e2_p27 (e2): Southwell Minster -- Cornwall -- Truro Cathedral -- located -- is
- e2_p28 (e2): Southwell Minster -- Cornwall -- Truro Cathedral -- located -- ?
- e2_p29 (e2): Southwell Minster -- Cornwall -- Truro Cathedral -- or
- e3_p1 (e3): Truro Cathedral -- located -- Which
- e3_p2 (e3): Truro Cathedral -- located
- e3_p3 (e3): Truro Cathedral -- located -- is
- e3_p4 (e3): Truro Cathedral -- located -- ?
- e3_p5 (e3): Truro Cathedral -- or
- e3_p6 (e3): Truro Cathedral -- located -- Cornwall
- e3_p7 (e3): Truro Cathedral -- located -- Southwell Minster
- e3_p8 (e3): Truro Cathedral -- Cornwall
- e3_p9 (e3): Truro Cathedral -- Cornwall -- located -- Which
- e3_p10 (e3): Truro Cathedral -- Cornwall -- located
- e3_p11 (e3): Truro Cathedral -- located -- Cornwall -- in
- e3_p12 (e3): Truro Cathedral -- located -- Cornwall -- ,
- e3_p13 (e3): Truro Cathedral -- Cornwall -- located -- is
- e3_p14 (e3): Truro Cathedral -- Cornwall -- located -- ?
- e3_p15 (e3): Truro Cathedral -- Cornwall -- in
- e3_p16 (e3): Truro Cathedral -- Cornwall -- ,
- e3_p17 (e3): Truro Cathedral -- located -- Cornwall -- Southwell Minster
- e3_p18 (e3): Truro Cathedral -- located -- Southwell Minster -- Cornwall
- e3_p19 (e3): Truro Cathedral -- Cornwall -- located -- Southwell Minster
- e3_p20 (e3): Truro Cathedral -- Cornwall -- Southwell Minster
- e3_p21 (e3): Truro Cathedral -- Cornwall -- Southwell Minster -- located -- Which
- e3_p22 (e3): Truro Cathedral -- Cornwall -- Southwell Minster -- located
- e3_p23 (e3): Truro Cathedral -- located -- Southwell Minster -- Cornwall -- in
- e3_p24 (e3): Truro Cathedral -- located -- Southwell Minster -- Cornwall -- ,
- e3_p25 (e3): Truro Cathedral -- Cornwall -- Southwell Minster -- located -- is
- e3_p26 (e3): Truro Cathedral -- Cornwall -- Southwell Minster -- located -- ?

## 8. LLM Path Scores
- e1: e1_p1 score=75.0 valid=True terminal=location
  Reason: The path starts from Cornwall and includes the located predicate, but it does not reach the answer slot.
- e1: e1_p2 score=55.0 valid=True terminal=location
  Reason: The path starts from Cornwall and includes the located predicate, but it does not cover the answer intent.
- e1: e1_p3 score=75.0 valid=True terminal=location
  Reason: The path starts from Cornwall, includes the located predicate, and covers the answer intent, but does not reach the answer slot.
- e1: e1_p4 score=30.0 valid=True terminal=location
  Reason: The path starts from Cornwall but ends with a punctuation mark, failing to reach the answer slot.
- e1: e1_p5 score=30.0 valid=True terminal=location
  Reason: The path starts from Cornwall but does not cover the answer intent and ends prematurely.
- e1: e1_p6 score=30.0 valid=True terminal=location
  Reason: The path starts from Cornwall but ends with a punctuation mark, failing to reach the answer slot.
- e1: e1_p7 score=90.0 valid=True terminal=location
  Reason: The path starts from Cornwall, includes the located predicate, and reaches Southwell Minster, covering the answer intent.
- e1: e1_p8 score=90.0 valid=True terminal=location
  Reason: The path starts from Cornwall, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e1: e1_p9 score=75.0 valid=True terminal=location
  Reason: The path starts from Cornwall and reaches Southwell Minster, but does not cover the answer intent.
- e1: e1_p10 score=75.0 valid=True terminal=location
  Reason: The path starts from Cornwall and reaches Truro Cathedral, but does not cover the answer intent.
- e1: e1_p11 score=95.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Southwell Minster, includes the located predicate, and covers the answer intent.
- e1: e1_p12 score=95.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Truro Cathedral, includes the located predicate, and covers the answer intent.
- e1: e1_p13 score=90.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Southwell Minster, includes the located predicate, and covers the answer intent.
- e1: e1_p14 score=90.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Truro Cathedral, includes the located predicate, and covers the answer intent.
- e1: e1_p15 score=90.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Truro Cathedral, includes the located predicate, and covers the answer intent.
- e1: e1_p16 score=90.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Southwell Minster, includes the located predicate, and covers the answer intent.
- e1: e1_p17 score=90.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Southwell Minster, includes the located predicate, and covers the answer intent.
- e1: e1_p18 score=90.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Truro Cathedral, includes the located predicate, and covers the answer intent.
- e1: e1_p19 score=90.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Truro Cathedral, includes the located predicate, and covers the answer intent.
- e1: e1_p20 score=90.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Truro Cathedral, includes the located predicate, and covers the answer intent.
- e1: e1_p21 score=95.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Southwell Minster, includes the located predicate, and covers the answer intent.
- e1: e1_p22 score=95.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Truro Cathedral, includes the located predicate, and covers the answer intent.
- e1: e1_p23 score=90.0 valid=True terminal=location
  Reason: The path starts from Cornwall, reaches Southwell Minster, includes the located predicate, and covers the answer intent.
- e2: e2_p1 score=75.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, but does not reach the answer slot.
- e2: e2_p2 score=55.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster and includes the located predicate, but it does not cover the answer intent.
- e2: e2_p3 score=75.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, but does not reach the answer slot.
- e2: e2_p4 score=30.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster but ends with a punctuation mark, failing to reach the answer slot.
- e2: e2_p5 score=30.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster but does not cover the answer intent and ends prematurely.
- e2: e2_p6 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p7 score=30.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster but does not cover the answer intent and ends prematurely.
- e2: e2_p8 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e2: e2_p9 score=75.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster and reaches Cornwall, but does not cover the answer intent.
- e2: e2_p10 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e2: e2_p11 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e2: e2_p12 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p13 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e2: e2_p14 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e2: e2_p15 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e2: e2_p16 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e2: e2_p17 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p18 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p19 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p20 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p21 score=95.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p22 score=95.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p23 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p24 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p25 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p26 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p27 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p28 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e2: e2_p29 score=90.0 valid=True terminal=location
  Reason: The path starts from Southwell Minster, includes the located predicate, and reaches Truro Cathedral, covering the answer intent.
- e3: e3_p1 score=75.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, but does not reach the answer slot.
- e3: e3_p2 score=55.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral and includes the located predicate, but it does not cover the answer intent.
- e3: e3_p3 score=75.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, but does not reach the answer slot.
- e3: e3_p4 score=30.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral but ends with a punctuation mark, failing to reach the answer slot.
- e3: e3_p5 score=30.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral but does not cover the answer intent and ends prematurely.
- e3: e3_p6 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e3: e3_p7 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Southwell Minster, covering the answer intent.
- e3: e3_p8 score=30.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral but does not cover the answer intent and ends prematurely.
- e3: e3_p9 score=95.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, reaches Cornwall, includes the located predicate, and covers the answer intent.
- e3: e3_p10 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e3: e3_p11 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e3: e3_p12 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e3: e3_p13 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e3: e3_p14 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e3: e3_p15 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e3: e3_p16 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Cornwall, covering the answer intent.
- e3: e3_p17 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Southwell Minster, covering the answer intent.
- e3: e3_p18 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Southwell Minster, covering the answer intent.
- e3: e3_p19 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Southwell Minster, covering the answer intent.
- e3: e3_p20 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Southwell Minster, covering the answer intent.
- e3: e3_p21 score=95.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Southwell Minster, covering the answer intent.
- e3: e3_p22 score=95.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Southwell Minster, covering the answer intent.
- e3: e3_p23 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Southwell Minster, covering the answer intent.
- e3: e3_p24 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Southwell Minster, covering the answer intent.
- e3: e3_p25 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Southwell Minster, covering the answer intent.
- e3: e3_p26 score=90.0 valid=True terminal=location
  Reason: The path starts from Truro Cathedral, includes the located predicate, and reaches Southwell Minster, covering the answer intent.

## 8.1 Top-2 Paths per Entity
- e1: e1_p11, e1_p12
- e2: e2_p21, e2_p22
- e3: e3_p21, e3_p22

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p11', 'e2': 'e2_p21', 'e3': 'e3_p21'} mean_path_score=95.0
- ps2: {'e1': 'e1_p11', 'e2': 'e2_p21', 'e3': 'e3_p22'} mean_path_score=95.0
- ps3: {'e1': 'e1_p11', 'e2': 'e2_p22', 'e3': 'e3_p21'} mean_path_score=95.0
- ps4: {'e1': 'e1_p11', 'e2': 'e2_p22', 'e3': 'e3_p22'} mean_path_score=95.0
- ps5: {'e1': 'e1_p12', 'e2': 'e2_p21', 'e3': 'e3_p21'} mean_path_score=95.0
- ps6: {'e1': 'e1_p12', 'e2': 'e2_p21', 'e3': 'e3_p22'} mean_path_score=95.0
- ps7: {'e1': 'e1_p12', 'e2': 'e2_p22', 'e3': 'e3_p21'} mean_path_score=95.0
- ps8: {'e1': 'e1_p12', 'e2': 'e2_p22', 'e3': 'e3_p22'} mean_path_score=95.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Which is located in Cornwall, Southwell Minster or Truro Cathedral?
- ps1
  - e1_p11: Cornwall -> Southwell Minster -> located -> Which
  - e2_p21: Southwell Minster -> Cornwall -> Truro Cathedral -> located -> Which
  - e3_p21: Truro Cathedral -> Cornwall -> Southwell Minster -> located -> Which
- ps2
  - e1_p11: Cornwall -> Southwell Minster -> located -> Which
  - e2_p21: Southwell Minster -> Cornwall -> Truro Cathedral -> located -> Which
  - e3_p22: Truro Cathedral -> Cornwall -> Southwell Minster -> located
- ps3
  - e1_p11: Cornwall -> Southwell Minster -> located -> Which
  - e2_p22: Southwell Minster -> Cornwall -> Truro Cathedral -> located
  - e3_p21: Truro Cathedral -> Cornwall -> Southwell Minster -> located -> Which
- ps4
  - e1_p11: Cornwall -> Southwell Minster -> located -> Which
  - e2_p22: Southwell Minster -> Cornwall -> Truro Cathedral -> located
  - e3_p22: Truro Cathedral -> Cornwall -> Southwell Minster -> located

Output:
- selected_path_set_ids: ['ps1', 'ps3']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Which is located in Cornwall, Southwell Minster? depends_on=[] support=['e1_p11']
- q2: Which is located in Cornwall, Truro Cathedral? depends_on=[] support=['e3_p21']

## 10. Atomic Subquestion DAG
- None: Which is located in Cornwall, Southwell Minster?
- None: Which is located in Cornwall, Truro Cathedral?
