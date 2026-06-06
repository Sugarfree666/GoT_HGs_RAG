# DEPO Decomposition #11

- Dataset: `hotpotqa`
- Question: Bytham Castle is a castle in the civil parish of how many houses?
- Gold answer: 300

## 1. Semantic-Normalized Question
Bytham Castle is a castle in the civil parish of how many houses?

## 2. Mask Spans
- Bytham Castle (entity, BythamCastle)

## 3. Selective Masked Question
SomeEntityA is a castle in the civil parish of how many houses?

## 4. CoreNLP Dependency Parse
- castle[4] --nsubj--> SomeEntityA[1]
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

## 6. Entity Start Nodes
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
- e1: e1_p1 score=90.0 valid=True terminal=count
  Reason: The path starts from Bytham Castle, covers the necessary roles leading to houses, and includes the how many cue for counting.
- e1: e1_p2 score=85.0 valid=True terminal=count
  Reason: The path starts from Bytham Castle and leads to houses, covering the how many cue, but lacks the final modifier 'how'.
- e1: e1_p3 score=70.0 valid=True terminal=count
  Reason: The path covers the entity and leads to civil but does not reach the answer target of houses or the how many cue.
- e1: e1_p4 score=75.0 valid=True terminal=count
  Reason: The path leads to houses but misses the how many cue, which is essential for the answer intent.
- e1: e1_p5 score=60.0 valid=True terminal=count
  Reason: The path reaches houses but ends with a preposition, missing the necessary cue for counting.
- e1: e1_p6 score=50.0 valid=True terminal=count
  Reason: The path stops too early at parish and does not reach the answer target or the how many cue.
- e1: e1_p7 score=40.0 valid=True terminal=count
  Reason: The path ends with a preposition and does not reach the answer target or the how many cue.
- e1: e1_p8 score=30.0 valid=True terminal=count
  Reason: The path ends with a determiner and does not reach the answer target or the how many cue.
- e1: e1_p9 score=20.0 valid=True terminal=count
  Reason: The path is too short and does not provide any useful information towards the answer.
- e1: e1_p10 score=25.0 valid=True terminal=count
  Reason: The path ends with an auxiliary and does not reach the answer target or the how many cue.
- e1: e1_p11 score=15.0 valid=True terminal=count
  Reason: The path is too short and does not provide any useful information towards the answer.
- e1: e1_p12 score=10.0 valid=True terminal=count
  Reason: The path ends with punctuation and does not provide any useful information towards the answer.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p2'} mean_path_score=85.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- bytham_castle -> castle (castle of Bytham Castle)
- castle -> parish (in parish of castle)
- parish -> houses (of parish)
- houses -> count (how many houses)
### ast_ps2 (ps2)
- bytham_castle -> castle (castle of Bytham Castle)
- castle -> parish (in parish of castle)
- parish -> houses (of parish)
- houses -> count (how many houses)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively covers the necessary branches leading to the count of houses in the parish of Bytham Castle, aligning with the original question's intent and allowing for one-hop atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- bytham_castle: Bytham Castle (entity)
- castle: castle (type_variable)
- parish: parish (type_variable)
- houses: houses (type_variable)
- count: count (value_slot)

Edges:
- bytham_castle -> castle (castle of Bytham Castle)
- castle -> parish (in parish of castle)
- parish -> houses (of parish)
- houses -> count (how many houses)

## 11. Atomic Subquestion DAG
- None: What is the castle of Bytham Castle?
- None: In which parish is Bytham Castle located?
- None: How many houses are in the parish of Bytham Castle?
- None: How many houses are in the parish of the castle of Bytham Castle?
