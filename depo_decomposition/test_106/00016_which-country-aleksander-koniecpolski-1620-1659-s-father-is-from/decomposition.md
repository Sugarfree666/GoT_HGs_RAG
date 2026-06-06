# DEPO Decomposition #16

- Dataset: `2wikimultihopqa`
- Question: Which country Aleksander Koniecpolski (1620–1659)'s father is from?
- Gold answer: Polish-Lithuanian Commonwealth

## 1. Semantic-Normalized Question
From which country is Aleksander Koniecpolski (1620–1659)'s father?

## 2. Explicit Entities
- Aleksander Koniecpolski (Person) span=(22, 45)

## 3. Entity Masking
- PersonA -> Aleksander Koniecpolski

From which country is PersonA (1620–1659)'s father?

## 4. CoreNLP Dependency Parse
- country[3] --case--> From[1]
- country[3] --det--> which[2]
- PersonA[5] --obl:from--> country[3]
- PersonA[5] --cop--> is[4]
- father[12] --nmod:poss--> PersonA[5]
- 1620[7] --punct--> ([6]
- PersonA[5] --dep--> 1620[7]
- 1659[9] --dep--> –[8]
- 1620[7] --nmod--> 1659[9]
- 1620[7] --punct--> )[10]
- PersonA[5] --case--> 's[11]
- father[12] --punct--> ?[13]

## 5. Undirected Dependency Graph
- From[1] --case-- country[3]
- which[2] --det-- country[3]
- country[3] --obl:from-- Aleksander Koniecpolski[5]
- is[4] --cop-- Aleksander Koniecpolski[5]
- Aleksander Koniecpolski[5] --nmod:poss-- father[12]
- Aleksander Koniecpolski[5] --dep-- 1620[7]
- Aleksander Koniecpolski[5] --case-- 's[11]
- ([6] --punct-- 1620[7]
- 1620[7] --nmod-- 1659[9]
- 1620[7] --punct-- )[10]
- –[8] --dep-- 1659[9]
- father[12] --punct-- ?[13]

## 6. Entity Start Nodes from Explicit Entities
- e1: Aleksander Koniecpolski graph_node_ids=['5']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Aleksander Koniecpolski -- 1620 -- 1659
- e1_p2 (e1): Aleksander Koniecpolski -- 1620 -- 1659 -- –
- e1_p3 (e1): Aleksander Koniecpolski -- country -- which
- e1_p4 (e1): Aleksander Koniecpolski -- country
- e1_p5 (e1): Aleksander Koniecpolski -- country -- From
- e1_p6 (e1): Aleksander Koniecpolski -- 1620
- e1_p7 (e1): Aleksander Koniecpolski -- 's
- e1_p8 (e1): Aleksander Koniecpolski -- father
- e1_p9 (e1): Aleksander Koniecpolski -- 1620 -- (
- e1_p10 (e1): Aleksander Koniecpolski -- 1620 -- )
- e1_p11 (e1): Aleksander Koniecpolski -- father -- ?
- e1_p12 (e1): Aleksander Koniecpolski -- is

## 8. LLM Path Scores
- e1: e1_p1 score=30.0 valid=False
  Reason: The path does not include the necessary intermediate role of 'father' or the answer target 'country'.
- e1: e1_p2 score=30.0 valid=False
  Reason: The path does not include the necessary intermediate role of 'father' or the answer target 'country'.
- e1: e1_p3 score=90.0 valid=True terminal=country
  Reason: The path starts from Aleksander Koniecpolski, reaches 'country', and includes the 'which' cue, effectively supporting the answer intent.
- e1: e1_p4 score=75.0 valid=True terminal=country
  Reason: The path starts from Aleksander Koniecpolski and reaches 'country', but it misses the 'which' cue.
- e1: e1_p5 score=90.0 valid=True terminal=country
  Reason: The path starts from Aleksander Koniecpolski, reaches 'country', and includes the 'From' cue, effectively supporting the answer intent.
- e1: e1_p6 score=30.0 valid=False
  Reason: The path does not include the necessary intermediate role of 'father' or the answer target 'country'.
- e1: e1_p7 score=0.0 valid=False
  Reason: The path ends at a possessive marker and does not provide any useful information.
- e1: e1_p8 score=90.0 valid=True terminal=country
  Reason: The path starts from Aleksander Koniecpolski, reaches 'father', and effectively supports the answer intent regarding the country.
- e1: e1_p9 score=0.0 valid=False
  Reason: The path ends at punctuation and does not provide any useful information.
- e1: e1_p10 score=0.0 valid=False
  Reason: The path ends at punctuation and does not provide any useful information.
- e1: e1_p11 score=30.0 valid=False
  Reason: The path includes 'father' but does not reach the answer target 'country'.
- e1: e1_p12 score=0.0 valid=False
  Reason: The path ends at an auxiliary and does not provide any useful information.

## 8.1 Top-2 Paths per Entity
- e1: e1_p3, e1_p5

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p3'} mean_path_score=90.0
- ps2: {'e1': 'e1_p5'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- aleksander_koniecpolski -> father (father of Aleksander Koniecpolski)
- father -> country (country of the father)
### ast_ps2 (ps2)
- aleksander_koniecpolski -> father (father of Aleksander Koniecpolski)
- father -> country (country of the father)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively captures the relationship between Aleksander Koniecpolski and his father, leading to the country of origin, thus supporting the decomposition into atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- aleksander_koniecpolski: Aleksander Koniecpolski (entity)
- father: father (type_variable)
- country: country (value_slot)

Edges:
- aleksander_koniecpolski -> father (father of Aleksander Koniecpolski)
- father -> country (country of the father)

## 11. Atomic Subquestion DAG
- None: Who is the father of Aleksander Koniecpolski?
- None: From which country is the father of Aleksander Koniecpolski?
