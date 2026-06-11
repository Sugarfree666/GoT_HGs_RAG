# DEPO Decomposition #7

- Dataset: `2wikimultihopqa`
- Question: Who was born first out of Aivar Kuusmaa and Andy Summers?
- Gold answer: Andy Summers

## 1. Semantic-Normalized Question
Who was born first out of Aivar Kuusmaa and Andy Summers?

## 2. Explicit Entities
- Aivar Kuusmaa (Person) span=(26, 39)
- Andy Summers (Person) span=(44, 56)

## 3. Entity Masking
- PersonA -> Aivar Kuusmaa
- PersonB -> Andy Summers

Who was born first out of PersonA and PersonB?

## 4. CoreNLP Dependency Parse
- born[3] --nsubj:pass--> Who[1]
- born[3] --aux:pass--> was[2]
- born[3] --advmod--> first[4]
- PersonA[7] --case--> out[5]
- out[5] --fixed--> of[6]
- born[3] --obl:out_of--> PersonA[7]
- PersonB[9] --cc--> and[8]
- born[3] --obl:out_of--> PersonB[9]
- PersonA[7] --conj:and--> PersonB[9]
- born[3] --punct--> ?[10]

## 5. Undirected Dependency Graph
- Who[1] --nsubj:pass-- born[3]
- was[2] --aux:pass-- born[3]
- born[3] --advmod-- first[4]
- born[3] --obl:out_of-- Aivar Kuusmaa[7]
- born[3] --obl:out_of-- Andy Summers[9]
- born[3] --punct-- ?[10]
- out[5] --case-- Aivar Kuusmaa[7]
- out[5] --fixed-- of[6]
- Aivar Kuusmaa[7] --conj:and-- Andy Summers[9]
- and[8] --cc-- Andy Summers[9]

## 6. Entity Start Nodes from Explicit Entities
- e1: Aivar Kuusmaa graph_node_ids=['7']
- e2: Andy Summers graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Aivar Kuusmaa -- born -- first
- e1_p2 (e1): Aivar Kuusmaa -- born -- Who
- e1_p3 (e1): Aivar Kuusmaa -- born
- e1_p4 (e1): Aivar Kuusmaa -- out
- e1_p5 (e1): Aivar Kuusmaa -- born -- was
- e1_p6 (e1): Aivar Kuusmaa -- born -- ?
- e1_p7 (e1): Aivar Kuusmaa -- out -- of
- e1_p8 (e1): Aivar Kuusmaa -- born -- Andy Summers
- e1_p9 (e1): Aivar Kuusmaa -- Andy Summers
- e1_p10 (e1): Aivar Kuusmaa -- Andy Summers -- born -- first
- e1_p11 (e1): Aivar Kuusmaa -- Andy Summers -- born -- Who
- e1_p12 (e1): Aivar Kuusmaa -- Andy Summers -- born
- e1_p13 (e1): Aivar Kuusmaa -- born -- Andy Summers -- and
- e1_p14 (e1): Aivar Kuusmaa -- Andy Summers -- born -- was
- e1_p15 (e1): Aivar Kuusmaa -- Andy Summers -- born -- ?
- e1_p16 (e1): Aivar Kuusmaa -- Andy Summers -- and
- e2_p1 (e2): Andy Summers -- born -- first
- e2_p2 (e2): Andy Summers -- born -- Who
- e2_p3 (e2): Andy Summers -- born
- e2_p4 (e2): Andy Summers -- born -- was
- e2_p5 (e2): Andy Summers -- born -- ?
- e2_p6 (e2): Andy Summers -- and
- e2_p7 (e2): Andy Summers -- born -- Aivar Kuusmaa
- e2_p8 (e2): Andy Summers -- Aivar Kuusmaa
- e2_p9 (e2): Andy Summers -- born -- Aivar Kuusmaa -- out
- e2_p10 (e2): Andy Summers -- Aivar Kuusmaa -- born -- first
- e2_p11 (e2): Andy Summers -- born -- Aivar Kuusmaa -- out -- of
- e2_p12 (e2): Andy Summers -- Aivar Kuusmaa -- born -- Who
- e2_p13 (e2): Andy Summers -- Aivar Kuusmaa -- born
- e2_p14 (e2): Andy Summers -- Aivar Kuusmaa -- out
- e2_p15 (e2): Andy Summers -- Aivar Kuusmaa -- born -- was
- e2_p16 (e2): Andy Summers -- Aivar Kuusmaa -- born -- ?
- e2_p17 (e2): Andy Summers -- Aivar Kuusmaa -- out -- of

## 7.5 Terminal Glue Path Pruning
Total raw paths: 33
Total kept paths: 16
Total pruned paths: 17
Total pruned ratio: 51.52%

### By Entity
- e1 / Aivar Kuusmaa
  - raw: 16
  - kept: 8
  - pruned: 8
  - fallback_used: False
  - examples:
    - e1_p4: Aivar Kuusmaa -> out [terminal=out, reason=terminal_glue_dependency_label]
    - e1_p5: Aivar Kuusmaa -> born -> was [terminal=was, reason=terminal_glue_token]
    - e1_p6: Aivar Kuusmaa -> born -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p7: Aivar Kuusmaa -> out -> of [terminal=of, reason=terminal_glue_token]
    - e1_p13: Aivar Kuusmaa -> born -> Andy Summers -> and [terminal=and, reason=terminal_glue_token]
- e2 / Andy Summers
  - raw: 17
  - kept: 8
  - pruned: 9
  - fallback_used: False
  - examples:
    - e2_p4: Andy Summers -> born -> was [terminal=was, reason=terminal_glue_token]
    - e2_p5: Andy Summers -> born -> ? [terminal=?, reason=terminal_glue_token]
    - e2_p6: Andy Summers -> and [terminal=and, reason=terminal_glue_token]
    - e2_p9: Andy Summers -> born -> Aivar Kuusmaa -> out [terminal=out, reason=terminal_glue_dependency_label]
    - e2_p11: Andy Summers -> born -> Aivar Kuusmaa -> out -> of [terminal=of, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=85.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, and includes the first cue, but lacks a direct connection to the other entity.
- e1: e1_p2 score=80.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, and includes the Who cue, but does not connect to the other entity.
- e1: e1_p3 score=55.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa and reaches the born predicate but does not cover the first cue or connect to the other entity.
- e1: e1_p8 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, and connects to Andy Summers, but does not include the first cue.
- e1: e1_p9 score=30.0 valid=True terminal=birth_order
  Reason: The path connects Aivar Kuusmaa and Andy Summers but does not include any relevant predicates or cues.
- e1: e1_p10 score=95.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, connects to Andy Summers, reaches the born predicate, and includes the first cue.
- e1: e1_p11 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, connects to Andy Summers, reaches the born predicate, and includes the Who cue.
- e1: e1_p12 score=85.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, connects to Andy Summers, and reaches the born predicate but does not include the first cue.
- e2: e2_p1 score=85.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, and includes the first cue, but lacks a direct connection to the other entity.
- e2: e2_p2 score=80.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, and includes the Who cue, but does not connect to the other entity.
- e2: e2_p3 score=55.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers and reaches the born predicate but does not cover the first cue or connect to the other entity.
- e2: e2_p7 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, and connects to Aivar Kuusmaa, but does not include the first cue.
- e2: e2_p8 score=30.0 valid=True terminal=birth_order
  Reason: The path connects Andy Summers and Aivar Kuusmaa but does not include any relevant predicates or cues.
- e2: e2_p10 score=95.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, connects to Aivar Kuusmaa, reaches the born predicate, and includes the first cue.
- e2: e2_p12 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, connects to Aivar Kuusmaa, reaches the born predicate, and includes the Who cue.
- e2: e2_p13 score=85.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, connects to Aivar Kuusmaa, and reaches the born predicate but does not include the first cue.

## 8.1 Highest-Scored Path per Entity
- e1: e1_p10 score=95.0
- e2: e2_p10 score=95.0

## 8.2 Selected Path Set
- ps1: {'e1': 'e1_p10', 'e2': 'e2_p10'} mean_path_score=95.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Who was born first out of Aivar Kuusmaa and Andy Summers?
- ps1
  - e1_p10: Aivar Kuusmaa -> Andy Summers -> born -> first
  - e2_p10: Andy Summers -> Aivar Kuusmaa -> born -> first

Output:
- reason: The DAG compiles each semantic reasoning edge into one atomic lookup.
- q1: When was Aivar Kuusmaa born? depends_on=[] support=[]
- q2: When was Andy Summers born? depends_on=[] support=[]

## 10. Atomic Subquestion DAG
- None: When was Aivar Kuusmaa born?
- None: When was Andy Summers born?
