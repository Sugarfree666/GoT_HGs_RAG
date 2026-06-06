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

## 8. LLM Selected Entity Paths
- e1: e1_p1 ICC KnockOut Trophy -- debut -- cricketer -- who -- batsman -- handed -- right
  Reason: This path provides a comprehensive reasoning chain from the ICC KnockOut Trophy to the cricketer's debut and their batting style, which is essential for answering the question.

## 9. Selected Path Semantic Transduction
Nodes:
- icc_knockout_trophy: ICC KnockOut Trophy (entity)
- debut: debut (type_variable)
- cricketer: cricketer (type_variable)
- batsman: batsman (type_variable)
- handed: handed (type_variable)
- right: right (value_slot)

Edges:
- icc_knockout_trophy -> debut (debut of ICC KnockOut Trophy)
- debut -> cricketer (cricketer who debuted)
- cricketer -> batsman (batsman role of cricketer)
- batsman -> handed (handedness of batsman)
- handed -> right (right-handedness)

## 10. Atomic Subquestion DAG
- None: When did the ICC KnockOut Trophy debut?
- None: Who is the cricketer who debuted in the ICC KnockOut Trophy?
- None: What is the batting style of the cricketer of the debut of ICC KnockOut Trophy?
- None: What is the handedness of the batsman of the cricketer of the debut of ICC KnockOut Trophy?
- None: Is the batsman of the cricketer of the debut of ICC KnockOut Trophy right-handed?
