# DEPO Decomposition #16

- Dataset: `2wikimultihopqa`
- Question: Which country Aleksander Koniecpolski (1620–1659)'s father is from?
- Gold answer: Polish-Lithuanian Commonwealth

## 1. Semantic-Normalized Question
Which country is Aleksander Koniecpolski (1620–1659)'s father from?

## 2. Explicit Entities
- Aleksander Koniecpolski (1620–1659) (Person) span=(17, 52)

## 3. Entity Masking
- PersonA -> Aleksander Koniecpolski (1620–1659)

Which country is PersonA's father from?

## 4. CoreNLP Dependency Parse
- country[2] --det--> Which[1]
- father[6] --nsubj--> country[2]
- father[6] --cop--> is[3]
- father[6] --nmod:poss--> PersonA[4]
- PersonA[4] --case--> 's[5]
- father[6] --dep--> from[7]
- father[6] --punct--> ?[8]

## 5. Undirected Dependency Graph
- Which[1] --det-- country[2]
- country[2] --nsubj-- father[6]
- is[3] --cop-- father[6]
- Aleksander Koniecpolski (1620–1659)[4] --nmod:poss-- father[6]
- Aleksander Koniecpolski (1620–1659)[4] --case-- 's[5]
- father[6] --dep-- from[7]
- father[6] --punct-- ?[8]

## 6. Entity Start Nodes from Explicit Entities
- e1: Aleksander Koniecpolski (1620–1659) graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Aleksander Koniecpolski (1620–1659) -- father -- country -- Which
- e1_p2 (e1): Aleksander Koniecpolski (1620–1659) -- father -- country
- e1_p3 (e1): Aleksander Koniecpolski (1620–1659) -- 's
- e1_p4 (e1): Aleksander Koniecpolski (1620–1659) -- father
- e1_p5 (e1): Aleksander Koniecpolski (1620–1659) -- father -- is
- e1_p6 (e1): Aleksander Koniecpolski (1620–1659) -- father -- from
- e1_p7 (e1): Aleksander Koniecpolski (1620–1659) -- father -- ?

## 7.5 Terminal Glue Path Pruning
Total raw paths: 7
Total kept paths: 3
Total pruned paths: 4
Total pruned ratio: 57.14%

### By Entity
- e1 / Aleksander Koniecpolski (1620–1659)
  - raw: 7
  - kept: 3
  - pruned: 4
  - fallback_used: False
  - examples:
    - e1_p3: Aleksander Koniecpolski (1620–1659) -> 's [terminal='s, reason=terminal_glue_dependency_label]
    - e1_p5: Aleksander Koniecpolski (1620–1659) -> father -> is [terminal=is, reason=terminal_glue_token]
    - e1_p6: Aleksander Koniecpolski (1620–1659) -> father -> from [terminal=from, reason=terminal_glue_token]
    - e1_p7: Aleksander Koniecpolski (1620–1659) -> father -> ? [terminal=?, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=country
  Reason: The path starts from Aleksander Koniecpolski, connects to father, and then to country, effectively covering the answer intent.
- e1: e1_p2 score=85.0 valid=True terminal=country
  Reason: The path starts from Aleksander Koniecpolski, connects to father, and then to country, but misses the wh cue 'which' at the end.
- e1: e1_p4 score=50.0 valid=True terminal=father
  Reason: The path only connects Aleksander Koniecpolski to father, missing the necessary connection to country and the wh cue.

## 8.1 Highest-Scored Path per Entity
- e1: e1_p1 score=90.0

## 8.2 Selected Path Set
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Which country Aleksander Koniecpolski (1620–1659)'s father is from?
- ps1
  - e1_p1: Aleksander Koniecpolski (1620–1659) -> father -> country -> Which

Output:
- reason: The DAG compiles each semantic reasoning edge into one atomic lookup.
- q1: Who is Aleksander Koniecpolski (1620–1659)'s father? depends_on=[] support=[]
- q2: Which country is Aleksander Koniecpolski (1620–1659)'s father's answer from? depends_on=['q1'] support=[]

## 10. Atomic Subquestion DAG
- None: Who is Aleksander Koniecpolski (1620–1659)'s father?
- None: Which country is Aleksander Koniecpolski (1620–1659)'s father's answer from?
