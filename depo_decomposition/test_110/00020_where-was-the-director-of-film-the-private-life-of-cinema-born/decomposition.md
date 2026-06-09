# DEPO Decomposition #20

- Dataset: `2wikimultihopqa`
- Question: Where was the director of film The Private Life Of Cinema born?
- Gold answer: Montreal, Quebec

## 1. Semantic-Normalized Question
Where was the director of the film The Private Life Of Cinema born?

## 2. Explicit Entities
- The Private Life Of Cinema (Film) span=(35, 61)

## 3. Entity Masking
- FilmA -> The Private Life Of Cinema

Where was the director of the film FilmA born?

## 4. CoreNLP Dependency Parse
- was[2] --advmod--> Where[1]
- born[9] --aux:pass--> was[2]
- director[4] --det--> the[3]
- born[9] --nsubj:pass--> director[4]
- FilmA[8] --case--> of[5]
- FilmA[8] --det--> the[6]
- FilmA[8] --compound--> film[7]
- director[4] --nmod:of--> FilmA[8]
- born[9] --punct--> ?[10]

## 5. Undirected Dependency Graph
- Where[1] --advmod-- was[2]
- was[2] --aux:pass-- born[9]
- the[3] --det-- director[4]
- director[4] --nsubj:pass-- born[9]
- director[4] --nmod:of-- The Private Life Of Cinema[8]
- of[5] --case-- The Private Life Of Cinema[8]
- the[6] --det-- The Private Life Of Cinema[8]
- film[7] --compound-- The Private Life Of Cinema[8]
- born[9] --punct-- ?[10]

## 6. Entity Start Nodes from Explicit Entities
- e1: The Private Life Of Cinema graph_node_ids=['8']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): The Private Life Of Cinema -- director -- born -- was -- Where
- e1_p2 (e1): The Private Life Of Cinema -- director -- born
- e1_p3 (e1): The Private Life Of Cinema -- director -- born -- was
- e1_p4 (e1): The Private Life Of Cinema -- director -- born -- ?
- e1_p5 (e1): The Private Life Of Cinema -- director
- e1_p6 (e1): The Private Life Of Cinema -- director -- the
- e1_p7 (e1): The Private Life Of Cinema -- film
- e1_p8 (e1): The Private Life Of Cinema -- of
- e1_p9 (e1): The Private Life Of Cinema -- the

## 7.5 Terminal Glue Path Pruning
Total raw paths: 9
Total kept paths: 4
Total pruned paths: 5
Total pruned ratio: 55.56%

### By Entity
- e1 / The Private Life Of Cinema
  - raw: 9
  - kept: 4
  - pruned: 5
  - fallback_used: False
  - examples:
    - e1_p3: The Private Life Of Cinema -> director -> born -> was [terminal=was, reason=terminal_glue_token]
    - e1_p4: The Private Life Of Cinema -> director -> born -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p6: The Private Life Of Cinema -> director -> the [terminal=the, reason=terminal_glue_token]
    - e1_p8: The Private Life Of Cinema -> of [terminal=of, reason=terminal_glue_token]
    - e1_p9: The Private Life Of Cinema -> the [terminal=the, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=85.0 valid=True terminal=birth_location
  Reason: The path starts from The Private Life Of Cinema, reaches director, and includes the born predicate, but it ends with an auxiliary which slightly reduces its score.
- e1: e1_p2 score=90.0 valid=True terminal=birth_location
  Reason: The path starts from The Private Life Of Cinema, reaches director, and includes the born predicate, effectively supporting the question intent.
- e1: e1_p5 score=30.0 valid=False terminal=birth_location
  Reason: The path stops too early at director and does not include the necessary predicates or cues to support the question.
- e1: e1_p7 score=20.0 valid=False terminal=birth_location
  Reason: The path only connects the film entity to the term 'film', lacking any relevant predicates or cues for the question.

## 8.1 Top-2 Paths per Entity
- e1: e1_p2, e1_p1

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p2'} mean_path_score=90.0
- ps2: {'e1': 'e1_p1'} mean_path_score=85.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Where was the director of film The Private Life Of Cinema born?
- ps1
  - e1_p2: The Private Life Of Cinema -> director -> born
- ps2
  - e1_p1: The Private Life Of Cinema -> director -> born -> was -> Where

Output:
- selected_path_set_ids: ['ps1']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who is the director of The Private Life Of Cinema? depends_on=[] support=['e1_p2']
- q2: Where was q1's answer born? depends_on=['q1'] support=['e1_p2']

## 10. Atomic Subquestion DAG
- None: Who is the director of The Private Life Of Cinema?
- None: Where was q1's answer born?
