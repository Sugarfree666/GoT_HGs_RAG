# DEPO Decomposition #10

- Dataset: `2wikimultihopqa`
- Question: What nationality is the director of film Blood Street?
- Gold answer: Chinese

## 1. Semantic-Normalized Question
What nationality is the director of the film Blood Street?

## 2. Explicit Entities
- Blood Street (Film) span=(45, 57)

## 3. Entity Masking
- FilmA -> Blood Street

What nationality is the director of the film FilmA?

## 4. CoreNLP Dependency Parse
- nationality[2] --det--> What[1]
- is[3] --obj--> nationality[2]
- director[5] --det--> the[4]
- is[3] --nsubj--> director[5]
- FilmA[9] --case--> of[6]
- FilmA[9] --det--> the[7]
- FilmA[9] --compound--> film[8]
- director[5] --nmod:of--> FilmA[9]
- is[3] --punct--> ?[10]

## 5. Undirected Dependency Graph
- What[1] --det-- nationality[2]
- nationality[2] --obj-- is[3]
- is[3] --nsubj-- director[5]
- is[3] --punct-- ?[10]
- the[4] --det-- director[5]
- director[5] --nmod:of-- Blood Street[9]
- of[6] --case-- Blood Street[9]
- the[7] --det-- Blood Street[9]
- film[8] --compound-- Blood Street[9]

## 6. Entity Start Nodes from Explicit Entities
- e1: Blood Street graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Blood Street -- director -- is -- nationality -- What
- e1_p2 (e1): Blood Street -- director -- is -- nationality
- e1_p3 (e1): Blood Street -- director
- e1_p4 (e1): Blood Street -- director -- is
- e1_p5 (e1): Blood Street -- director -- the
- e1_p6 (e1): Blood Street -- director -- is -- ?
- e1_p7 (e1): Blood Street -- film
- e1_p8 (e1): Blood Street -- of
- e1_p9 (e1): Blood Street -- the

## 7.5 Terminal Glue Path Pruning
Total raw paths: 9
Total kept paths: 4
Total pruned paths: 5
Total pruned ratio: 55.56%

### By Entity
- e1 / Blood Street
  - raw: 9
  - kept: 4
  - pruned: 5
  - fallback_used: False
  - examples:
    - e1_p4: Blood Street -> director -> is [terminal=is, reason=terminal_glue_token]
    - e1_p5: Blood Street -> director -> the [terminal=the, reason=terminal_glue_token]
    - e1_p6: Blood Street -> director -> is -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p8: Blood Street -> of [terminal=of, reason=terminal_glue_token]
    - e1_p9: Blood Street -> the [terminal=the, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Blood Street, reaches director, and covers the nationality attribute, effectively supporting the question intent.
- e1: e1_p2 score=85.0 valid=True terminal=nationality
  Reason: The path starts from Blood Street, reaches director, and covers the nationality attribute, but lacks the wh cue for full intent coverage.
- e1: e1_p3 score=30.0 valid=False terminal=none
  Reason: The path only connects Blood Street to director, failing to reach the necessary attributes for the question.
- e1: e1_p7 score=30.0 valid=False terminal=none
  Reason: The path connects Blood Street to film but does not include the director or nationality, making it irrelevant.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p2'} mean_path_score=85.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: What nationality is the director of film Blood Street?
- ps1
  - e1_p1: Blood Street -> director -> is -> nationality -> What
- ps2
  - e1_p2: Blood Street -> director -> is -> nationality

Output:
- selected_path_set_ids: ['ps1', 'ps2']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who is the director of Blood Street? depends_on=[] support=['e1_p1']
- q2: What is the nationality of q1's answer? depends_on=['q1'] support=['e1_p2']

## 10. Atomic Subquestion DAG
- None: Who is the director of Blood Street?
- None: What is the nationality of q1's answer?
