# DEPO Decomposition #15

- Dataset: `2wikimultihopqa`
- Question: Where was the place of death of Maurice, Prince Of Orange's father?
- Gold answer: Delft

## 1. Semantic-Normalized Question
Where was the place of death of the father of Maurice, Prince Of Orange?

## 2. Explicit Entities
- Prince Of Orange (Location) span=(55, 71)

## 3. Entity Masking
- LocationA -> Prince Of Orange

Where was the place of death of the father of Maurice, LocationA?

## 4. CoreNLP Dependency Parse
- was[2] --advmod--> Where[1]
- place[4] --det--> the[3]
- was[2] --nsubj--> place[4]
- death[6] --case--> of[5]
- place[4] --nmod:of--> death[6]
- father[9] --case--> of[7]
- father[9] --det--> the[8]
- death[6] --nmod:of--> father[9]
- Maurice[11] --case--> of[10]
- father[9] --nmod:of--> Maurice[11]
- Maurice[11] --punct--> ,[12]
- Maurice[11] --appos--> LocationA[13]
- was[2] --punct--> ?[14]

## 5. Undirected Dependency Graph
- Where[1] --advmod-- was[2]
- was[2] --nsubj-- place[4]
- was[2] --punct-- ?[14]
- the[3] --det-- place[4]
- place[4] --nmod:of-- death[6]
- of[5] --case-- death[6]
- death[6] --nmod:of-- father[9]
- of[7] --case-- father[9]
- the[8] --det-- father[9]
- father[9] --nmod:of-- Maurice[11]
- of[10] --case-- Maurice[11]
- Maurice[11] --punct-- ,[12]
- Maurice[11] --appos-- Prince Of Orange[13]

## 6. Entity Start Nodes from Explicit Entities
- e1: Prince Of Orange graph_node_ids=['13']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Prince Of Orange -- Maurice -- father -- death -- place -- was -- Where
- e1_p2 (e1): Prince Of Orange -- Maurice -- father -- death -- place
- e1_p3 (e1): Prince Of Orange -- Maurice -- father -- death -- place -- was
- e1_p4 (e1): Prince Of Orange -- Maurice -- father -- death -- place -- the
- e1_p5 (e1): Prince Of Orange -- Maurice -- father -- death -- place -- was -- ?
- e1_p6 (e1): Prince Of Orange -- Maurice -- father -- death
- e1_p7 (e1): Prince Of Orange -- Maurice -- father -- death -- of
- e1_p8 (e1): Prince Of Orange -- Maurice -- father
- e1_p9 (e1): Prince Of Orange -- Maurice -- father -- of
- e1_p10 (e1): Prince Of Orange -- Maurice -- father -- the
- e1_p11 (e1): Prince Of Orange -- Maurice
- e1_p12 (e1): Prince Of Orange -- Maurice -- of
- e1_p13 (e1): Prince Of Orange -- Maurice -- ,

## 7.5 Terminal Glue Path Pruning
Total raw paths: 13
Total kept paths: 5
Total pruned paths: 8
Total pruned ratio: 61.54%

### By Entity
- e1 / Prince Of Orange
  - raw: 13
  - kept: 5
  - pruned: 8
  - fallback_used: False
  - examples:
    - e1_p3: Prince Of Orange -> Maurice -> father -> death -> place -> was [terminal=was, reason=terminal_glue_token]
    - e1_p4: Prince Of Orange -> Maurice -> father -> death -> place -> the [terminal=the, reason=terminal_glue_token]
    - e1_p5: Prince Of Orange -> Maurice -> father -> death -> place -> was -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p7: Prince Of Orange -> Maurice -> father -> death -> of [terminal=of, reason=terminal_glue_token]
    - e1_p9: Prince Of Orange -> Maurice -> father -> of [terminal=of, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=place_of_death
  Reason: The path starts from Prince Of Orange, covers the necessary roles leading to the place of death, and includes the where cue.
- e1: e1_p2 score=85.0 valid=True terminal=place_of_death
  Reason: The path starts from Prince Of Orange and effectively leads to the place of death, covering the necessary roles.
- e1: e1_p6 score=75.0 valid=True terminal=place_of_death
  Reason: The path covers the necessary roles but stops before reaching the final answer target, missing the place.
- e1: e1_p8 score=60.0 valid=True terminal=father
  Reason: The path stops too early, only reaching the father without covering the death or place.
- e1: e1_p11 score=30.0 valid=True terminal=father
  Reason: The path is too short and does not reach any relevant answer targets or cues.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p2'} mean_path_score=85.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Where was the place of death of Maurice, Prince Of Orange's father?
- ps1
  - e1_p1: Prince Of Orange -> Maurice -> father -> death -> place -> was -> Where
- ps2
  - e1_p2: Prince Of Orange -> Maurice -> father -> death -> place

Output:
- selected_path_set_ids: ['ps1', 'ps2']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who is the father of Prince Of Orange? depends_on=[] support=['e1_p1']
- q2: Where did q1's answer die? depends_on=['q1'] support=['e1_p2']

## 10. Atomic Subquestion DAG
- None: Who is the father of Prince Of Orange?
- None: Where did q1's answer die?
