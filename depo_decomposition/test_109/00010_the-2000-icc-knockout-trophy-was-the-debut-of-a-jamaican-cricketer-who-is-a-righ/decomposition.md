# DEPO Decomposition #10

- Dataset: `hotpotqa`
- Question: The 2000 ICC KnockOut Trophy was the debut of a Jamaican cricketer who is a right-handed what?
- Gold answer: middle order batsman

## 1. Semantic-Normalized Question
The 2000 ICC KnockOut Trophy was the debut of a Jamaican cricketer who is a right-handed batsman?

## 2. Explicit Entities
- ICC KnockOut Trophy (Person) span=(9, 28)
- Jamaican (Country) span=(48, 56)

## 3. Entity Masking
- PersonA -> ICC KnockOut Trophy
- CountryA -> Jamaican

The 2000 PersonA was the debut of a CountryA cricketer who is a right-handed batsman?

## 4. CoreNLP Dependency Parse
- PersonA[3] --det--> The[1]
- PersonA[3] --nummod--> 2000[2]
- debut[6] --nsubj--> PersonA[3]
- debut[6] --cop--> was[4]
- debut[6] --det--> the[5]
- cricketer[10] --case--> of[7]
- cricketer[10] --det--> a[8]
- cricketer[10] --compound--> CountryA[9]
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
- Jamaican[9] --compound-- cricketer[10]
- cricketer[10] --acl-- who[11]
- who[11] --cop-- is[12]
- who[11] --nsubj-- batsman[17]
- a[13] --det-- batsman[17]
- right[14] --amod-- handed[16]
- -[15] --punct-- handed[16]
- handed[16] --amod-- batsman[17]

## 6. Entity Start Nodes from Explicit Entities
- e1: ICC KnockOut Trophy graph_node_ids=['3']
- e2: Jamaican graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who -- batsman -- handed -- right
- e1_p2 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who -- batsman -- handed
- e1_p3 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who -- batsman -- handed -- -
- e1_p4 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who -- batsman
- e1_p5 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who -- batsman -- a
- e1_p6 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who
- e1_p7 (e1): ICC KnockOut Trophy -- debut -- cricketer -- who -- is
- e1_p8 (e1): ICC KnockOut Trophy -- debut -- cricketer
- e1_p9 (e1): ICC KnockOut Trophy -- debut -- cricketer -- of
- e1_p10 (e1): ICC KnockOut Trophy -- debut -- cricketer -- a
- e1_p11 (e1): ICC KnockOut Trophy -- 2000
- e1_p12 (e1): ICC KnockOut Trophy -- debut
- e1_p13 (e1): ICC KnockOut Trophy -- debut -- was
- e1_p14 (e1): ICC KnockOut Trophy -- debut -- the
- e1_p15 (e1): ICC KnockOut Trophy -- debut -- ?
- e1_p16 (e1): ICC KnockOut Trophy -- The
- e1_p17 (e1): ICC KnockOut Trophy -- debut -- cricketer -- Jamaican
- e2_p1 (e2): Jamaican -- cricketer -- who -- batsman -- handed -- right
- e2_p2 (e2): Jamaican -- cricketer -- who -- batsman -- handed
- e2_p3 (e2): Jamaican -- cricketer -- who -- batsman -- handed -- -
- e2_p4 (e2): Jamaican -- cricketer -- who -- batsman
- e2_p5 (e2): Jamaican -- cricketer -- who -- batsman -- a
- e2_p6 (e2): Jamaican -- cricketer -- debut
- e2_p7 (e2): Jamaican -- cricketer -- debut -- was
- e2_p8 (e2): Jamaican -- cricketer -- debut -- the
- e2_p9 (e2): Jamaican -- cricketer -- debut -- ?
- e2_p10 (e2): Jamaican -- cricketer -- who
- e2_p11 (e2): Jamaican -- cricketer -- who -- is
- e2_p12 (e2): Jamaican -- cricketer
- e2_p13 (e2): Jamaican -- cricketer -- of
- e2_p14 (e2): Jamaican -- cricketer -- a
- e2_p15 (e2): Jamaican -- cricketer -- debut -- ICC KnockOut Trophy
- e2_p16 (e2): Jamaican -- cricketer -- debut -- ICC KnockOut Trophy -- 2000
- e2_p17 (e2): Jamaican -- cricketer -- debut -- ICC KnockOut Trophy -- The

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, covers the who predicate, and includes the batsman cue.
- e1: e1_p2 score=85.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, covers the who predicate, and includes the batsman cue.
- e1: e1_p3 score=30.0 valid=False terminal=batsman
  Reason: The path stops too early and does not reach the necessary answer slot.
- e1: e1_p4 score=95.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, covers the who predicate, and includes the batsman cue.
- e1: e1_p5 score=75.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, covers the who predicate, and includes the batsman cue.
- e1: e1_p6 score=40.0 valid=False terminal=batsman
  Reason: The path stops too early and does not reach the necessary answer slot.
- e1: e1_p7 score=50.0 valid=False terminal=batsman
  Reason: The path stops too early and does not reach the necessary answer slot.
- e1: e1_p8 score=20.0 valid=False terminal=batsman
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p9 score=30.0 valid=False terminal=batsman
  Reason: The path does not reach the necessary answer slot.
- e1: e1_p10 score=20.0 valid=False terminal=batsman
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p11 score=10.0 valid=False terminal=batsman
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p12 score=20.0 valid=False terminal=batsman
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p13 score=30.0 valid=False terminal=batsman
  Reason: The path does not reach the necessary answer slot.
- e1: e1_p14 score=20.0 valid=False terminal=batsman
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p15 score=10.0 valid=False terminal=batsman
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p16 score=20.0 valid=False terminal=batsman
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p17 score=90.0 valid=True terminal=batsman
  Reason: The path starts from ICC KnockOut Trophy, reaches cricketer, covers the who predicate, and includes the batsman cue.
- e2: e2_p1 score=90.0 valid=True terminal=batsman
  Reason: The path starts from Jamaican, reaches cricketer, covers the who predicate, and includes the batsman cue.
- e2: e2_p2 score=85.0 valid=True terminal=batsman
  Reason: The path starts from Jamaican, reaches cricketer, covers the who predicate, and includes the batsman cue.
- e2: e2_p3 score=30.0 valid=False terminal=batsman
  Reason: The path stops too early and does not reach the necessary answer slot.
- e2: e2_p4 score=90.0 valid=True terminal=batsman
  Reason: The path starts from Jamaican, reaches cricketer, covers the who predicate, and includes the batsman cue.
- e2: e2_p5 score=75.0 valid=True terminal=batsman
  Reason: The path starts from Jamaican, reaches cricketer, covers the who predicate, and includes the batsman cue.
- e2: e2_p6 score=40.0 valid=False terminal=batsman
  Reason: The path stops too early and does not reach the necessary answer slot.
- e2: e2_p7 score=50.0 valid=False terminal=batsman
  Reason: The path stops too early and does not reach the necessary answer slot.
- e2: e2_p8 score=20.0 valid=False terminal=batsman
  Reason: The path is too short and does not provide any useful information.
- e2: e2_p9 score=30.0 valid=False terminal=batsman
  Reason: The path does not reach the necessary answer slot.
- e2: e2_p10 score=20.0 valid=False terminal=batsman
  Reason: The path is too short and does not provide any useful information.
- e2: e2_p11 score=10.0 valid=False terminal=batsman
  Reason: The path is too short and does not provide any useful information.
- e2: e2_p12 score=20.0 valid=False terminal=batsman
  Reason: The path is too short and does not provide any useful information.
- e2: e2_p13 score=30.0 valid=False terminal=batsman
  Reason: The path does not reach the necessary answer slot.
- e2: e2_p14 score=20.0 valid=False terminal=batsman
  Reason: The path is too short and does not provide any useful information.
- e2: e2_p15 score=90.0 valid=True terminal=batsman
  Reason: The path starts from Jamaican, reaches cricketer, covers the who predicate, and includes the batsman cue.
- e2: e2_p16 score=90.0 valid=True terminal=batsman
  Reason: The path starts from Jamaican, reaches cricketer, covers the who predicate, and includes the batsman cue.
- e2: e2_p17 score=90.0 valid=True terminal=batsman
  Reason: The path starts from Jamaican, reaches cricketer, covers the who predicate, and includes the batsman cue.

## 8.1 Top-2 Paths per Entity
- e1: e1_p4, e1_p1
- e2: e2_p1, e2_p15

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p4', 'e2': 'e2_p1'} mean_path_score=92.5
- ps2: {'e1': 'e1_p4', 'e2': 'e2_p15'} mean_path_score=92.5
- ps3: {'e1': 'e1_p1', 'e2': 'e2_p1'} mean_path_score=90.0
- ps4: {'e1': 'e1_p1', 'e2': 'e2_p15'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: The 2000 ICC KnockOut Trophy was the debut of a Jamaican cricketer who is a right-handed what?
- ps1
  - e1_p4: ICC KnockOut Trophy -> debut -> cricketer -> who -> batsman
  - e2_p1: Jamaican -> cricketer -> who -> batsman -> handed -> right
- ps2
  - e1_p4: ICC KnockOut Trophy -> debut -> cricketer -> who -> batsman
  - e2_p15: Jamaican -> cricketer -> debut -> ICC KnockOut Trophy
- ps3
  - e1_p1: ICC KnockOut Trophy -> debut -> cricketer -> who -> batsman -> handed -> right
  - e2_p1: Jamaican -> cricketer -> who -> batsman -> handed -> right
- ps4
  - e1_p1: ICC KnockOut Trophy -> debut -> cricketer -> who -> batsman -> handed -> right
  - e2_p15: Jamaican -> cricketer -> debut -> ICC KnockOut Trophy

Output:
- selected_path_set_ids: ['ps1', 'ps3']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who is the cricketer that debuted in the ICC KnockOut Trophy? depends_on=[] support=['e1_p4']
- q2: What is the handedness of the cricketer from q1's answer? depends_on=['q1'] support=['e1_p1']

## 10. Atomic Subquestion DAG
- None: Who is the cricketer that debuted in the ICC KnockOut Trophy?
- None: What is the handedness of the cricketer from q1's answer?
