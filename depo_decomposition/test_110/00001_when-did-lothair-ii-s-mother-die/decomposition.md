# DEPO Decomposition #1

- Dataset: `2wikimultihopqa`
- Question: When did Lothair Ii's mother die?
- Gold answer: 20 March 851

## 1. Semantic-Normalized Question
When did Lothair II's mother die?

## 2. Explicit Entities
- Lothair II (Person) span=(9, 19)

## 3. Entity Masking
- PersonA -> Lothair II

When did PersonA's mother die?

## 4. CoreNLP Dependency Parse
- die[6] --advmod--> When[1]
- die[6] --aux--> did[2]
- mother[5] --nmod:poss--> PersonA[3]
- PersonA[3] --case--> 's[4]
- die[6] --nsubj--> mother[5]
- die[6] --punct--> ?[7]

## 5. Undirected Dependency Graph
- When[1] --advmod-- die[6]
- did[2] --aux-- die[6]
- Lothair II[3] --nmod:poss-- mother[5]
- Lothair II[3] --case-- 's[4]
- mother[5] --nsubj-- die[6]
- die[6] --punct-- ?[7]

## 6. Entity Start Nodes from Explicit Entities
- e1: Lothair II graph_node_ids=['3']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Lothair II -- mother -- die -- When
- e1_p2 (e1): Lothair II -- mother -- die
- e1_p3 (e1): Lothair II -- mother -- die -- did
- e1_p4 (e1): Lothair II -- mother -- die -- ?
- e1_p5 (e1): Lothair II -- mother
- e1_p6 (e1): Lothair II -- 's

## 7.5 Terminal Glue Path Pruning
Total raw paths: 6
Total kept paths: 3
Total pruned paths: 3
Total pruned ratio: 50.00%

### By Entity
- e1 / Lothair II
  - raw: 6
  - kept: 3
  - pruned: 3
  - fallback_used: False
  - examples:
    - e1_p3: Lothair II -> mother -> die -> did [terminal=did, reason=terminal_glue_token]
    - e1_p4: Lothair II -> mother -> die -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p6: Lothair II -> 's [terminal='s, reason=terminal_glue_dependency_label]

## 8. LLM Path Scores
- e1: e1_p1 score=95.0 valid=True terminal=death_time
  Reason: The path starts from Lothair II, reaches mother, covers the die predicate, and includes the when cue.
- e1: e1_p2 score=85.0 valid=True terminal=death_time
  Reason: The path starts from Lothair II, reaches mother, and covers the die predicate, but it misses the when cue.
- e1: e1_p5 score=50.0 valid=True terminal=death_time
  Reason: The path starts from Lothair II and reaches mother, but it does not cover the die predicate or the when cue.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=95.0
- ps2: {'e1': 'e1_p2'} mean_path_score=85.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: When did Lothair Ii's mother die?
- ps1
  - e1_p1: Lothair II -> mother -> die -> When
- ps2
  - e1_p2: Lothair II -> mother -> die

Output:
- selected_path_set_ids: ['ps1']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who is Lothair II's mother? depends_on=[] support=['e1_p1']
- q2: When did q1's answer die? depends_on=['q1'] support=['e1_p1']

## 10. Atomic Subquestion DAG
- None: Who is Lothair II's mother?
- None: When did q1's answer die?
