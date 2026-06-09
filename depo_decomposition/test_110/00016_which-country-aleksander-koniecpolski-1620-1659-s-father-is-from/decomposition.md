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

## 7.5 Terminal Glue Path Pruning
Total raw paths: 12
Total kept paths: 6
Total pruned paths: 6
Total pruned ratio: 50.00%

### By Entity
- e1 / Aleksander Koniecpolski
  - raw: 12
  - kept: 6
  - pruned: 6
  - fallback_used: False
  - examples:
    - e1_p5: Aleksander Koniecpolski -> country -> From [terminal=From, reason=terminal_glue_token]
    - e1_p7: Aleksander Koniecpolski -> 's [terminal='s, reason=terminal_glue_dependency_label]
    - e1_p9: Aleksander Koniecpolski -> 1620 -> ( [terminal=(, reason=terminal_glue_dependency_label]
    - e1_p10: Aleksander Koniecpolski -> 1620 -> ) [terminal=), reason=terminal_glue_dependency_label]
    - e1_p11: Aleksander Koniecpolski -> father -> ? [terminal=?, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=30.0 valid=True terminal=father_country
  Reason: The path does not reach the necessary intermediate role of 'father' or the answer target 'country'.
- e1: e1_p2 score=30.0 valid=True terminal=father_country
  Reason: The path does not reach the necessary intermediate role of 'father' or the answer target 'country'.
- e1: e1_p3 score=90.0 valid=True terminal=father_country
  Reason: The path starts from Aleksander Koniecpolski, reaches 'country', and includes the 'which' cue, supporting the answer intent.
- e1: e1_p4 score=75.0 valid=True terminal=father_country
  Reason: The path reaches 'country' but misses the necessary intermediate role of 'father'.
- e1: e1_p6 score=30.0 valid=True terminal=father_country
  Reason: The path does not reach the necessary intermediate role of 'father' or the answer target 'country'.
- e1: e1_p8 score=55.0 valid=True terminal=father_country
  Reason: The path reaches 'father' but does not connect to the answer target 'country'.

## 8.1 Top-2 Paths per Entity
- e1: e1_p3, e1_p4

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p3'} mean_path_score=90.0
- ps2: {'e1': 'e1_p4'} mean_path_score=75.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Which country Aleksander Koniecpolski (1620–1659)'s father is from?
- ps1
  - e1_p3: Aleksander Koniecpolski -> country -> which
- ps2
  - e1_p4: Aleksander Koniecpolski -> country

Output:
- selected_path_set_ids: ['ps1', 'ps2']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: What country is Aleksander Koniecpolski from? depends_on=[] support=['e1_p4']
- q2: What country is the father of Aleksander Koniecpolski from? depends_on=['q1'] support=['e1_p3']

## 10. Atomic Subquestion DAG
- None: What country is Aleksander Koniecpolski from?
- None: What country is the father of Aleksander Koniecpolski from?
