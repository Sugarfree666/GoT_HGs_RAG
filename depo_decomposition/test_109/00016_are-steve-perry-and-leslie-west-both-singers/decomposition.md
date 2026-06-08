# DEPO Decomposition #16

- Dataset: `hotpotqa`
- Question: Are Steve Perry and Leslie West both singers?
- Gold answer: yes

## 1. Semantic-Normalized Question
Are Steve Perry and Leslie West both singers?

## 2. Explicit Entities
- Steve Perry (Person) span=(4, 15)
- Leslie West (Person) span=(20, 31)

## 3. Entity Masking
- PersonA -> Steve Perry
- PersonB -> Leslie West

Are PersonA and PersonB both singers?

## 4. CoreNLP Dependency Parse
- singers[6] --cop--> Are[1]
- singers[6] --nsubj--> PersonA[2]
- PersonB[4] --cc--> and[3]
- PersonA[2] --conj:and--> PersonB[4]
- singers[6] --nsubj--> PersonB[4]
- singers[6] --dep--> both[5]
- singers[6] --punct--> ?[7]

## 5. Undirected Dependency Graph
- Are[1] --cop-- singers[6]
- Steve Perry[2] --nsubj-- singers[6]
- Steve Perry[2] --conj:and-- Leslie West[4]
- and[3] --cc-- Leslie West[4]
- Leslie West[4] --nsubj-- singers[6]
- both[5] --dep-- singers[6]
- singers[6] --punct-- ?[7]

## 6. Entity Start Nodes from Explicit Entities
- e1: Steve Perry graph_node_ids=['2']
- e2: Leslie West graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Steve Perry -- singers -- both
- e1_p2 (e1): Steve Perry -- singers
- e1_p3 (e1): Steve Perry -- singers -- Are
- e1_p4 (e1): Steve Perry -- singers -- ?
- e1_p5 (e1): Steve Perry -- singers -- Leslie West
- e1_p6 (e1): Steve Perry -- Leslie West
- e1_p7 (e1): Steve Perry -- Leslie West -- singers -- both
- e1_p8 (e1): Steve Perry -- Leslie West -- singers
- e1_p9 (e1): Steve Perry -- Leslie West -- singers -- Are
- e1_p10 (e1): Steve Perry -- Leslie West -- singers -- ?
- e1_p11 (e1): Steve Perry -- singers -- Leslie West -- and
- e1_p12 (e1): Steve Perry -- Leslie West -- and
- e2_p1 (e2): Leslie West -- singers -- both
- e2_p2 (e2): Leslie West -- singers
- e2_p3 (e2): Leslie West -- singers -- Are
- e2_p4 (e2): Leslie West -- singers -- ?
- e2_p5 (e2): Leslie West -- and
- e2_p6 (e2): Leslie West -- singers -- Steve Perry
- e2_p7 (e2): Leslie West -- Steve Perry
- e2_p8 (e2): Leslie West -- Steve Perry -- singers -- both
- e2_p9 (e2): Leslie West -- Steve Perry -- singers
- e2_p10 (e2): Leslie West -- Steve Perry -- singers -- Are
- e2_p11 (e2): Leslie West -- Steve Perry -- singers -- ?

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, reaches singers, and includes the both cue, directly supporting the question intent.
- e1: e1_p2 score=75.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry and reaches singers, but it misses the both cue which is important for the question intent.
- e1: e1_p3 score=55.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry and reaches singers, but it includes the auxiliary 'Are' which does not contribute to the semantic chain.
- e1: e1_p4 score=30.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry and reaches singers, but ends with a punctuation mark, which does not support a complete semantic chain.
- e1: e1_p5 score=90.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, reaches singers, and includes Leslie West, directly supporting the question intent.
- e1: e1_p6 score=55.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry and reaches Leslie West, but it does not include the singers node, which is crucial for the question intent.
- e1: e1_p7 score=95.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, passes through Leslie West, reaches singers, and includes the both cue, directly supporting the question intent.
- e1: e1_p8 score=90.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, passes through Leslie West, and reaches singers, directly supporting the question intent.
- e1: e1_p9 score=95.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, passes through Leslie West, reaches singers, and includes the both cue, directly supporting the question intent.
- e1: e1_p10 score=30.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, passes through Leslie West, reaches singers, but ends with a punctuation mark, which does not support a complete semantic chain.
- e1: e1_p11 score=90.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, reaches singers, and includes Leslie West, directly supporting the question intent.
- e1: e1_p12 score=55.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry and reaches Leslie West, but it does not include the singers node, which is crucial for the question intent.
- e2: e2_p1 score=90.0 valid=True terminal=singers
  Reason: The path starts from Leslie West, reaches singers, and includes the both cue, directly supporting the question intent.
- e2: e2_p2 score=75.0 valid=True terminal=singers
  Reason: The path starts from Leslie West and reaches singers, but it misses the both cue which is important for the question intent.
- e2: e2_p3 score=55.0 valid=True terminal=singers
  Reason: The path starts from Leslie West and reaches singers, but it includes the auxiliary 'Are' which does not contribute to the semantic chain.
- e2: e2_p4 score=30.0 valid=True terminal=singers
  Reason: The path starts from Leslie West and reaches singers, but ends with a punctuation mark, which does not support a complete semantic chain.
- e2: e2_p5 score=0.0 valid=False
  Reason: The path starts from Leslie West and ends with 'and', failing to reach any relevant nodes.
- e2: e2_p6 score=90.0 valid=True terminal=singers
  Reason: The path starts from Leslie West, reaches singers, and includes Steve Perry, directly supporting the question intent.
- e2: e2_p7 score=55.0 valid=True terminal=singers
  Reason: The path starts from Leslie West and reaches Steve Perry, but it does not include the singers node, which is crucial for the question intent.
- e2: e2_p8 score=95.0 valid=True terminal=singers
  Reason: The path starts from Leslie West, passes through Steve Perry, reaches singers, and includes the both cue, directly supporting the question intent.
- e2: e2_p9 score=90.0 valid=True terminal=singers
  Reason: The path starts from Leslie West, passes through Steve Perry, and reaches singers, directly supporting the question intent.
- e2: e2_p10 score=95.0 valid=True terminal=singers
  Reason: The path starts from Leslie West, passes through Steve Perry, reaches singers, and includes the both cue, directly supporting the question intent.
- e2: e2_p11 score=30.0 valid=True terminal=singers
  Reason: The path starts from Leslie West, passes through Steve Perry, reaches singers, but ends with a punctuation mark, which does not support a complete semantic chain.

## 8.1 Top-2 Paths per Entity
- e1: e1_p7, e1_p9
- e2: e2_p10, e2_p8

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p7', 'e2': 'e2_p10'} mean_path_score=95.0
- ps2: {'e1': 'e1_p7', 'e2': 'e2_p8'} mean_path_score=95.0
- ps3: {'e1': 'e1_p9', 'e2': 'e2_p10'} mean_path_score=95.0
- ps4: {'e1': 'e1_p9', 'e2': 'e2_p8'} mean_path_score=95.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Are Steve Perry and Leslie West both singers?
- ps1
  - e1_p7: Steve Perry -> Leslie West -> singers -> both
  - e2_p10: Leslie West -> Steve Perry -> singers -> Are
- ps2
  - e1_p7: Steve Perry -> Leslie West -> singers -> both
  - e2_p8: Leslie West -> Steve Perry -> singers -> both
- ps3
  - e1_p9: Steve Perry -> Leslie West -> singers -> Are
  - e2_p10: Leslie West -> Steve Perry -> singers -> Are
- ps4
  - e1_p9: Steve Perry -> Leslie West -> singers -> Are
  - e2_p8: Leslie West -> Steve Perry -> singers -> both

Output:
- selected_path_set_ids: ['ps1']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Is Steve Perry a singer? depends_on=[] support=['e1_p7']
- q2: Is Leslie West a singer? depends_on=[] support=['e1_p7']

## 10. Atomic Subquestion DAG
- None: Is Steve Perry a singer?
- None: Is Leslie West a singer?
