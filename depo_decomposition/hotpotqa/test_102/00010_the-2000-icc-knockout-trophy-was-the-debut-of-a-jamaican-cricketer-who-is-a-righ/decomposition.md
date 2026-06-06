# DEPO Decomposition #10

- Dataset: `hotpotqa`
- Question: The 2000 ICC KnockOut Trophy was the debut of a Jamaican cricketer who is a right-handed what?
- Gold answer: middle order batsman

## 1. Semantic-Normalized Question
The 2000 ICC KnockOut Trophy was the debut of a Jamaican cricketer who is a right-handed batsman?

## 2. Mask Spans
- ICC KnockOut Trophy (entity, Person)

## 3. Selective Masked Question
The 2000 PersonA was the debut of a Jamaican cricketer who is a right-handed batsman?

## 4. CoreNLP Dependency Parse
- PersonA[3] --det--> The[1]
- PersonA[3] --nummod--> 2000[2]
- debut[6] --nsubj--> PersonA[3]
- debut[6] --cop--> was[4]
- debut[6] --det--> the[5]
- cricketer[10] --case--> of[7]
- cricketer[10] --det--> a[8]
- cricketer[10] --amod--> Jamaican[9]
- debut[6] --nmod:of--> cricketer[10]
- cricketer[10] --acl--> who[11]
- who[11] --cop--> is[12]
- batsman[17] --det--> a[13]
- handed[16] --amod--> right[14]
- handed[16] --punct--> -[15]
- batsman[17] --amod--> handed[16]
- who[11] --nsubj--> batsman[17]
- debut[6] --punct--> ?[18]

## 5. Undirected Dependency Graph
- The[1] --det-- ICC KnockOut Trophy[3]
- 2000[2] --nummod-- ICC KnockOut Trophy[3]
- ICC KnockOut Trophy[3] --nsubj-- debut[6]
- was[4] --cop-- debut[6]
- the[5] --det-- debut[6]
- debut[6] --nmod:of-- cricketer[10]
- debut[6] --punct-- ?[18]
- of[7] --case-- cricketer[10]
- a[8] --det-- cricketer[10]
- Jamaican[9] --amod-- cricketer[10]
- cricketer[10] --acl-- who[11]
- who[11] --cop-- is[12]
- who[11] --nsubj-- batsman[17]
- a[13] --det-- batsman[17]
- right[14] --amod-- handed[16]
- -[15] --punct-- handed[16]
- handed[16] --amod-- batsman[17]

## 6. Entity Start Nodes
- e1: ICC KnockOut Trophy graph_node_ids=['3']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who -- batsman -- handed -- right
- e1_p2 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who -- batsman -- handed
- e1_p3 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who -- batsman -- handed -- -
- e1_p4 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who -- batsman
- e1_p5 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who -- batsman -- a
- e1_p6 (e1): ICC KnockOut Trophy -- debut -- cricketer -- Jamaican
- e1_p7 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who
- e1_p8 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who -- is
- e1_p9 (e1): ICC KnockOut Trophy -- debut -- cricketer
- e1_p10 (e1): ICC KnockOut Trophy -- debut -- cricketer -- of
- e1_p11 (e1): ICC KnockOut Trophy -- debut -- cricketer -- a
- e1_p12 (e1): ICC KnockOut Trophy -- 2000
- e1_p13 (e1): ICC KnockOut Trophy -- debut
- e1_p14 (e1): ICC KnockOut Trophy -- debut -- was
- e1_p15 (e1): ICC KnockOut Trophy -- debut -- the
- e1_p16 (e1): ICC KnockOut Trophy -- debut -- ?
- e1_p17 (e1): ICC KnockOut Trophy -- The

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, includes the who cue, and covers the answer intent of identifying the batsman.
- e1: e1_p2 score=85.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, includes the who cue, and covers the answer intent of identifying the batsman, but misses the right-handed modifier.
- e1: e1_p3 score=70.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, includes the who cue, but does not cover the right-handed modifier, making it less effective.
- e1: e1_p4 score=80.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, includes the who cue, and covers the answer intent of identifying the batsman, but misses the right-handed modifier.
- e1: e1_p5 score=65.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, includes the who cue, but does not cover the right-handed modifier and ends with a determiner.
- e1: e1_p6 score=60.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, but does not cover the answer intent of identifying the batsman and misses the right-handed modifier.
- e1: e1_p7 score=55.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, but does not cover the answer intent of identifying the batsman and misses the right-handed modifier.
- e1: e1_p8 score=50.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, but does not cover the answer intent of identifying the batsman and misses the right-handed modifier.
- e1: e1_p9 score=40.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy and reaches cricketer, but does not cover the answer intent of identifying the batsman.
- e1: e1_p10 score=30.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy and reaches cricketer, but does not cover the answer intent of identifying the batsman.
- e1: e1_p11 score=25.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy and reaches cricketer, but does not cover the answer intent of identifying the batsman.
- e1: e1_p12 score=20.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy but does not cover the answer intent of identifying the batsman.
- e1: e1_p13 score=15.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy but does not cover the answer intent of identifying the batsman.
- e1: e1_p14 score=10.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy but does not cover the answer intent of identifying the batsman.
- e1: e1_p15 score=5.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy but does not cover the answer intent of identifying the batsman.
- e1: e1_p16 score=0.0 valid=False terminal=batsman
  Reason: The path ends with punctuation and does not contribute to identifying the batsman.
- e1: e1_p17 score=0.0 valid=False terminal=batsman
  Reason: The path does not contribute to identifying the batsman.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p2'} mean_path_score=85.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- icc_knockout_trophy -> cricketer (debut of cricketer in ICC KnockOut Trophy)
- cricketer -> batsman (role of cricketer as batsman)
### ast_ps2 (ps2)
- icc_knockout_trophy -> cricketer (debut of cricketer in ICC KnockOut Trophy)
- cricketer -> batsman (role of cricketer as batsman)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively captures the necessary branches for identifying the cricketer and their role as a batsman, covering all aspects of the original question.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- icc_knockout_trophy: ICC KnockOut Trophy (entity)
- cricketer: cricketer (type_variable)
- batsman: batsman (value_slot)

Edges:
- icc_knockout_trophy -> cricketer (debut of cricketer in ICC KnockOut Trophy)
- cricketer -> batsman (role of cricketer as batsman)

## 11. Atomic Subquestion DAG
- None: Who is the cricketer that made their debut in the ICC KnockOut Trophy?
- None: What is the role of the cricketer of ICC KnockOut Trophy as a batsman?
