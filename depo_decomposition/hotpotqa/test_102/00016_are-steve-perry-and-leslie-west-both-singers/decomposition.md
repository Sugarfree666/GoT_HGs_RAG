# DEPO Decomposition #16

- Dataset: `hotpotqa`
- Question: Are Steve Perry and Leslie West both singers?
- Gold answer: yes

## 1. Semantic-Normalized Question
Are Steve Perry and Leslie West both singers?

## 2. Mask Spans
- Steve Perry (entity, Person)
- Leslie West (entity, Person)

## 3. Selective Masked Question
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

## 6. Entity Start Nodes
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
  Reason: The path starts from Steve Perry and reaches singers, but it does not include the both cue and ends with a copula which is less useful.
- e1: e1_p4 score=30.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry and reaches singers, but it ends with punctuation and lacks necessary cues.
- e1: e1_p5 score=90.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, reaches singers, and includes Leslie West, covering the both cue effectively.
- e1: e1_p6 score=75.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry and reaches Leslie West, but it misses the both cue which is important for the question intent.
- e1: e1_p7 score=95.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, includes Leslie West, reaches singers, and covers the both cue effectively.
- e1: e1_p8 score=90.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, includes Leslie West, reaches singers, and covers the both cue effectively.
- e1: e1_p9 score=95.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, includes Leslie West, reaches singers, and covers the both cue effectively.
- e1: e1_p10 score=90.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, includes Leslie West, reaches singers, but misses the both cue.
- e1: e1_p11 score=75.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, includes Leslie West, reaches singers, but misses the both cue.
- e1: e1_p12 score=70.0 valid=True terminal=singers
  Reason: The path starts from Steve Perry, includes Leslie West, but ends with a conjunction which is less useful.
- e2: e2_p1 score=90.0 valid=True terminal=singers
  Reason: The path starts from Leslie West, reaches singers, and includes the both cue, directly supporting the question intent.
- e2: e2_p2 score=75.0 valid=True terminal=singers
  Reason: The path starts from Leslie West and reaches singers, but it misses the both cue which is important for the question intent.
- e2: e2_p3 score=55.0 valid=True terminal=singers
  Reason: The path starts from Leslie West and reaches singers, but it does not include the both cue and ends with a copula which is less useful.
- e2: e2_p4 score=30.0 valid=True terminal=singers
  Reason: The path starts from Leslie West and reaches singers, but it ends with punctuation and lacks necessary cues.
- e2: e2_p5 score=30.0 valid=True terminal=singers
  Reason: The path starts from Leslie West but does not reach the singers and lacks necessary cues.
- e2: e2_p6 score=90.0 valid=True terminal=singers
  Reason: The path starts from Leslie West, reaches singers, and includes Steve Perry, covering the both cue effectively.
- e2: e2_p7 score=75.0 valid=True terminal=singers
  Reason: The path starts from Leslie West and reaches Steve Perry, but it misses the both cue which is important for the question intent.
- e2: e2_p8 score=95.0 valid=True terminal=singers
  Reason: The path starts from Leslie West, includes Steve Perry, reaches singers, and covers the both cue effectively.
- e2: e2_p9 score=90.0 valid=True terminal=singers
  Reason: The path starts from Leslie West, includes Steve Perry, reaches singers, and covers the both cue effectively.
- e2: e2_p10 score=90.0 valid=True terminal=singers
  Reason: The path starts from Leslie West, includes Steve Perry, reaches singers, but misses the both cue.
- e2: e2_p11 score=75.0 valid=True terminal=singers
  Reason: The path starts from Leslie West, includes Steve Perry, reaches singers, but misses the both cue.

## 8.1 Top-2 Paths per Entity
- e1: e1_p7, e1_p9
- e2: e2_p8, e2_p1

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p7', 'e2': 'e2_p8'} mean_path_score=95.0
- ps2: {'e1': 'e1_p7', 'e2': 'e2_p1'} mean_path_score=92.5
- ps3: {'e1': 'e1_p9', 'e2': 'e2_p8'} mean_path_score=95.0
- ps4: {'e1': 'e1_p9', 'e2': 'e2_p1'} mean_path_score=92.5

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- steve_perry -> singers_e1 (singers of Steve Perry)
- leslie_west -> singers_e2 (singers of Leslie West)
### ast_ps2 (ps2)
- steve_perry -> singers (singers of Steve Perry)
- leslie_west -> leslie_west_singers (singers of Leslie West)
### ast_ps3 (ps3)
- steve_perry -> singers_e1 (singer of Steve Perry)
- leslie_west -> singers_e2 (singer of Leslie West)
### ast_ps4 (ps4)
- steve_perry -> singers_e1 (occupation of Steve Perry)
- leslie_west -> singers_e2 (occupation of Leslie West)
- singers_e1 -> both_e1 (both are)
- singers_e2 -> both_e2 (both are)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively covers both entities, Steve Perry and Leslie West, as singers, and includes the necessary cues for the original question, allowing for one-hop atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- steve_perry: Steve Perry (entity)
- leslie_west: Leslie West (entity)
- singers_e1: singers (value_slot)
- singers_e2: singers (value_slot)
- leslie_west_2: Leslie West (entity)
- steve_perry_2: Steve Perry (entity)
- singers_2: singers (value_slot)

Edges:
- steve_perry -> singers_e1 (singers of Steve Perry)
- leslie_west -> singers_e2 (singers of Leslie West)

## 11. Atomic Subquestion DAG
- None: What are the singers of Steve Perry?
- None: Who are the singers of Leslie West?
