# DEPO Decomposition #2

- Dataset: `hotpotqa`
- Question: The fictional private detective that appears in "The Adventure of the Seven Clocks" what written by whom?
- Gold answer: Sir Arthur Conan Doyle

## 1. Semantic-Normalized Question
The fictional private detective that appears in 'The Adventure of the Seven Clocks' was written by whom?

## 2. Explicit Entities
- The Adventure of the Seven Clocks (Work) span=(49, 82)

## 3. Entity Masking
- WorkA -> The Adventure of the Seven Clocks

The fictional private detective that appears in 'WorkA' was written by whom?

## 4. CoreNLP Dependency Parse
- detective[4] --det--> The[1]
- detective[4] --amod--> fictional[2]
- detective[4] --amod--> private[3]
- appears[6] --nsubj--> detective[4]
- written[12] --nsubj:pass--> detective[4]
- detective[4] --ref--> that[5]
- detective[4] --acl:relcl--> appears[6]
- WorkA[9] --case--> in[7]
- WorkA[9] --punct--> '[8]
- appears[6] --obl:in--> WorkA[9]
- WorkA[9] --punct--> '[10]
- written[12] --aux:pass--> was[11]
- whom[14] --case--> by[13]
- written[12] --obl:agent--> whom[14]
- written[12] --punct--> ?[15]

## 5. Undirected Dependency Graph
- The[1] --det-- detective[4]
- fictional[2] --amod-- detective[4]
- private[3] --amod-- detective[4]
- detective[4] --nsubj/acl:relcl-- appears[6]
- detective[4] --nsubj:pass-- written[12]
- detective[4] --ref-- that[5]
- appears[6] --obl:in-- The Adventure of the Seven Clocks[9]
- in[7] --case-- The Adventure of the Seven Clocks[9]
- '[8] --punct-- The Adventure of the Seven Clocks[9]
- The Adventure of the Seven Clocks[9] --punct-- '[10]
- was[11] --aux:pass-- written[12]
- written[12] --obl:agent-- whom[14]
- written[12] --punct-- ?[15]
- by[13] --case-- whom[14]

## 6. Entity Start Nodes from Explicit Entities
- e1: The Adventure of the Seven Clocks graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): The Adventure of the Seven Clocks -- appears -- detective -- fictional
- e1_p2 (e1): The Adventure of the Seven Clocks -- appears -- detective -- private
- e1_p3 (e1): The Adventure of the Seven Clocks -- appears -- detective -- written
- e1_p4 (e1): The Adventure of the Seven Clocks -- appears -- detective -- written -- was
- e1_p5 (e1): The Adventure of the Seven Clocks -- appears -- detective -- written -- whom
- e1_p6 (e1): The Adventure of the Seven Clocks -- appears -- detective -- written -- ?
- e1_p7 (e1): The Adventure of the Seven Clocks -- appears -- detective -- written -- whom -- by
- e1_p8 (e1): The Adventure of the Seven Clocks -- appears -- detective
- e1_p9 (e1): The Adventure of the Seven Clocks -- appears -- detective -- The
- e1_p10 (e1): The Adventure of the Seven Clocks -- appears -- detective -- that
- e1_p11 (e1): The Adventure of the Seven Clocks -- appears
- e1_p12 (e1): The Adventure of the Seven Clocks -- in
- e1_p13 (e1): The Adventure of the Seven Clocks -- '
- e1_p14 (e1): The Adventure of the Seven Clocks -- '

## 8. LLM Path Scores
- e1: e1_p1 score=55.0 valid=True terminal=author
  Reason: The path does not reach the necessary predicate 'written' and lacks coverage for the answer intent.
- e1: e1_p2 score=55.0 valid=True terminal=author
  Reason: The path does not reach the necessary predicate 'written' and lacks coverage for the answer intent.
- e1: e1_p3 score=75.0 valid=True terminal=author
  Reason: The path reaches the predicate 'written' but lacks the wh cue for the answer intent.
- e1: e1_p4 score=85.0 valid=True terminal=author
  Reason: The path includes the predicate 'written' and the auxiliary 'was', supporting the answer intent well.
- e1: e1_p5 score=90.0 valid=True terminal=author
  Reason: The path effectively reaches the predicate 'written' and includes the wh cue 'whom', fully supporting the answer intent.
- e1: e1_p6 score=55.0 valid=True terminal=author
  Reason: The path does not reach the necessary predicate 'written' and lacks coverage for the answer intent.
- e1: e1_p7 score=90.0 valid=True terminal=author
  Reason: The path effectively reaches the predicate 'written', includes the wh cue 'whom', and supports the answer intent well.
- e1: e1_p8 score=30.0 valid=True terminal=author
  Reason: The path stops too early and does not reach the necessary predicate 'written' or the answer intent.
- e1: e1_p9 score=30.0 valid=True terminal=author
  Reason: The path stops too early and does not reach the necessary predicate 'written' or the answer intent.
- e1: e1_p10 score=30.0 valid=True terminal=author
  Reason: The path stops too early and does not reach the necessary predicate 'written' or the answer intent.
- e1: e1_p11 score=0.0 valid=False terminal=author
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p12 score=0.0 valid=False terminal=author
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p13 score=0.0 valid=False terminal=author
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p14 score=0.0 valid=False terminal=author
  Reason: The path is too short and does not provide any useful information.

## 8.1 Top-2 Paths per Entity
- e1: e1_p5, e1_p7

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p5'} mean_path_score=90.0
- ps2: {'e1': 'e1_p7'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: The fictional private detective that appears in "The Adventure of the Seven Clocks" what written by whom?
- ps1
  - e1_p5: The Adventure of the Seven Clocks -> appears -> detective -> written -> whom
- ps2
  - e1_p7: The Adventure of the Seven Clocks -> appears -> detective -> written -> whom -> by

Output:
- selected_path_set_ids: ['ps1', 'ps2']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who is the detective in The Adventure of the Seven Clocks? depends_on=[] support=['e1_p5']
- q2: Who wrote q1's answer? depends_on=['q1'] support=['e1_p7']

## 10. Atomic Subquestion DAG
- None: Who is the detective in The Adventure of the Seven Clocks?
- None: Who wrote q1's answer?
