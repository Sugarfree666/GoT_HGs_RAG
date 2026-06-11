# DEPO Decomposition #15

- Dataset: `2wikimultihopqa`
- Question: Where was the place of death of Maurice, Prince Of Orange's father?
- Gold answer: Delft

## 1. Semantic-Normalized Question
Where was the place of death of Maurice, Prince Of Orange's father?

## 2. Explicit Entities
- Maurice (Person) span=(32, 39)
- Prince Of Orange (Person) span=(41, 57)

## 3. Entity Masking
- PersonA -> Maurice
- PersonB -> Prince Of Orange

Where was the place of death of PersonA, PersonB's father?

## 4. CoreNLP Dependency Parse
- was[2] --advmod--> Where[1]
- father[12] --dep--> was[2]
- place[4] --det--> the[3]
- was[2] --nsubj--> place[4]
- death[6] --case--> of[5]
- place[4] --nmod:of--> death[6]
- PersonA[8] --case--> of[7]
- death[6] --nmod:of--> PersonA[8]
- father[12] --punct--> ,[9]
- father[12] --nmod:poss--> PersonB[10]
- PersonB[10] --case--> 's[11]
- father[12] --punct--> ?[13]

## 5. Undirected Dependency Graph
- Where[1] --advmod-- was[2]
- was[2] --dep-- father[12]
- was[2] --nsubj-- place[4]
- the[3] --det-- place[4]
- place[4] --nmod:of-- death[6]
- of[5] --case-- death[6]
- death[6] --nmod:of-- Maurice[8]
- of[7] --case-- Maurice[8]
- ,[9] --punct-- father[12]
- Prince Of Orange[10] --nmod:poss-- father[12]
- Prince Of Orange[10] --case-- 's[11]
- father[12] --punct-- ?[13]

## 6. Entity Start Nodes from Explicit Entities
- e1: Maurice graph_node_ids=['8']
- e2: Prince Of Orange graph_node_ids=['10']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Maurice -- death -- place -- was -- father
- e1_p2 (e1): Maurice -- death -- place -- was -- father -- ,
- e1_p3 (e1): Maurice -- death -- place -- was -- father -- ?
- e1_p4 (e1): Maurice -- death -- place -- was -- Where
- e1_p5 (e1): Maurice -- death -- place
- e1_p6 (e1): Maurice -- death -- place -- was
- e1_p7 (e1): Maurice -- death -- place -- the
- e1_p8 (e1): Maurice -- death
- e1_p9 (e1): Maurice -- death -- of
- e1_p10 (e1): Maurice -- of
- e1_p11 (e1): Maurice -- death -- place -- was -- father -- Prince Of Orange
- e1_p12 (e1): Maurice -- death -- place -- was -- father -- Prince Of Orange -- 's
- e2_p1 (e2): Prince Of Orange -- father -- was -- place -- death
- e2_p2 (e2): Prince Of Orange -- father -- was -- place -- death -- of
- e2_p3 (e2): Prince Of Orange -- father -- was -- place
- e2_p4 (e2): Prince Of Orange -- father -- was -- place -- the
- e2_p5 (e2): Prince Of Orange -- father -- was -- Where
- e2_p6 (e2): Prince Of Orange -- 's
- e2_p7 (e2): Prince Of Orange -- father
- e2_p8 (e2): Prince Of Orange -- father -- was
- e2_p9 (e2): Prince Of Orange -- father -- ,
- e2_p10 (e2): Prince Of Orange -- father -- ?
- e2_p11 (e2): Prince Of Orange -- father -- was -- place -- death -- Maurice
- e2_p12 (e2): Prince Of Orange -- father -- was -- place -- death -- Maurice -- of

## 7.5 Terminal Glue Path Pruning
Total raw paths: 24
Total kept paths: 10
Total pruned paths: 14
Total pruned ratio: 58.33%

### By Entity
- e1 / Maurice
  - raw: 12
  - kept: 5
  - pruned: 7
  - fallback_used: False
  - examples:
    - e1_p2: Maurice -> death -> place -> was -> father -> , [terminal=,, reason=terminal_glue_token]
    - e1_p3: Maurice -> death -> place -> was -> father -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p6: Maurice -> death -> place -> was [terminal=was, reason=terminal_glue_token]
    - e1_p7: Maurice -> death -> place -> the [terminal=the, reason=terminal_glue_token]
    - e1_p9: Maurice -> death -> of [terminal=of, reason=terminal_glue_token]
- e2 / Prince Of Orange
  - raw: 12
  - kept: 5
  - pruned: 7
  - fallback_used: False
  - examples:
    - e2_p2: Prince Of Orange -> father -> was -> place -> death -> of [terminal=of, reason=terminal_glue_token]
    - e2_p4: Prince Of Orange -> father -> was -> place -> the [terminal=the, reason=terminal_glue_token]
    - e2_p6: Prince Of Orange -> 's [terminal='s, reason=terminal_glue_dependency_label]
    - e2_p8: Prince Of Orange -> father -> was [terminal=was, reason=terminal_glue_token]
    - e2_p9: Prince Of Orange -> father -> , [terminal=,, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=place_of_death
  Reason: The path starts from Maurice, covers the necessary roles leading to the place of death, and includes the answer intent cue 'where'.
- e1: e1_p4 score=90.0 valid=True terminal=place_of_death
  Reason: The path starts from Maurice, includes the 'where' cue, and effectively leads to the place of death.
- e1: e1_p5 score=75.0 valid=True terminal=place_of_death
  Reason: The path starts from Maurice and covers the death and place, but it misses the 'where' cue.
- e1: e1_p8 score=30.0 valid=True terminal=place_of_death
  Reason: The path only covers 'Maurice' and 'death', missing critical elements like 'place' and 'where'.
- e1: e1_p11 score=95.0 valid=True terminal=place_of_death
  Reason: The path starts from Maurice, includes the necessary roles, and effectively leads to the place of death with the 'where' cue.
- e2: e2_p1 score=90.0 valid=True terminal=place_of_death
  Reason: The path starts from Prince Of Orange, covers the necessary roles leading to the place of death, and includes the answer intent cue 'where'.
- e2: e2_p3 score=75.0 valid=True terminal=place_of_death
  Reason: The path starts from Prince Of Orange and covers the father and place, but it misses the 'where' cue.
- e2: e2_p5 score=75.0 valid=True terminal=place_of_death
  Reason: The path starts from Prince Of Orange, includes the 'where' cue, and effectively leads to the place of death.
- e2: e2_p7 score=30.0 valid=True terminal=place_of_death
  Reason: The path only covers 'Prince Of Orange' and 'father', missing critical elements like 'place' and 'where'.
- e2: e2_p11 score=95.0 valid=True terminal=place_of_death
  Reason: The path starts from Prince Of Orange, includes the necessary roles, and effectively leads to the place of death with the 'where' cue.

## 8.1 Highest-Scored Path per Entity
- e1: e1_p11 score=95.0
- e2: e2_p11 score=95.0

## 8.2 Selected Path Set
- ps1: {'e1': 'e1_p11', 'e2': 'e2_p11'} mean_path_score=95.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Where was the place of death of Maurice, Prince Of Orange's father?
- ps1
  - e1_p11: Maurice -> death -> place -> was -> father -> Prince Of Orange
  - e2_p11: Prince Of Orange -> father -> was -> place -> death -> Maurice

Output:
- reason: The DAG compiles each semantic reasoning edge into one atomic lookup.
- q1: Where did Maurice die? depends_on=[] support=[]
- q2: Where did Prince Of Orange's father die? depends_on=[] support=[]

## 10. Atomic Subquestion DAG
- None: Where did Maurice die?
- None: Where did Prince Of Orange's father die?
