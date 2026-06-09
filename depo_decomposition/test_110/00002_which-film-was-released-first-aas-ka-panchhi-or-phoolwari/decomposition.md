# DEPO Decomposition #2

- Dataset: `2wikimultihopqa`
- Question: Which film was released first, Aas Ka Panchhi or Phoolwari?
- Gold answer: Phoolwari

## 1. Semantic-Normalized Question
Which film was released first, Aas Ka Panchhi or Phoolwari?

## 2. Explicit Entities
- Aas Ka Panchhi (Film) span=(31, 45)
- Phoolwari (Film) span=(49, 58)

## 3. Entity Masking
- FilmA -> Aas Ka Panchhi
- FilmB -> Phoolwari

Which film was released first, FilmA or FilmB?

## 4. CoreNLP Dependency Parse
- film[2] --det--> Which[1]
- released[4] --nsubj:pass--> film[2]
- released[4] --aux:pass--> was[3]
- released[4] --advmod--> first[5]
- released[4] --punct--> ,[6]
- released[4] --obj--> FilmA[7]
- FilmB[9] --cc--> or[8]
- released[4] --obj--> FilmB[9]
- FilmA[7] --conj:or--> FilmB[9]
- released[4] --punct--> ?[10]

## 5. Undirected Dependency Graph
- Which[1] --det-- film[2]
- film[2] --nsubj:pass-- released[4]
- was[3] --aux:pass-- released[4]
- released[4] --advmod-- first[5]
- released[4] --punct-- ,[6]
- released[4] --obj-- Aas Ka Panchhi[7]
- released[4] --obj-- Phoolwari[9]
- released[4] --punct-- ?[10]
- Aas Ka Panchhi[7] --conj:or-- Phoolwari[9]
- or[8] --cc-- Phoolwari[9]

## 6. Entity Start Nodes from Explicit Entities
- e1: Aas Ka Panchhi graph_node_ids=['7']
- e2: Phoolwari graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Aas Ka Panchhi -- released -- film -- Which
- e1_p2 (e1): Aas Ka Panchhi -- released -- film
- e1_p3 (e1): Aas Ka Panchhi -- released -- first
- e1_p4 (e1): Aas Ka Panchhi -- released
- e1_p5 (e1): Aas Ka Panchhi -- released -- was
- e1_p6 (e1): Aas Ka Panchhi -- released -- ,
- e1_p7 (e1): Aas Ka Panchhi -- released -- ?
- e1_p8 (e1): Aas Ka Panchhi -- released -- Phoolwari
- e1_p9 (e1): Aas Ka Panchhi -- Phoolwari
- e1_p10 (e1): Aas Ka Panchhi -- Phoolwari -- released -- film -- Which
- e1_p11 (e1): Aas Ka Panchhi -- Phoolwari -- released -- film
- e1_p12 (e1): Aas Ka Panchhi -- Phoolwari -- released -- first
- e1_p13 (e1): Aas Ka Panchhi -- Phoolwari -- released
- e1_p14 (e1): Aas Ka Panchhi -- released -- Phoolwari -- or
- e1_p15 (e1): Aas Ka Panchhi -- Phoolwari -- released -- was
- e1_p16 (e1): Aas Ka Panchhi -- Phoolwari -- released -- ,
- e1_p17 (e1): Aas Ka Panchhi -- Phoolwari -- released -- ?
- e1_p18 (e1): Aas Ka Panchhi -- Phoolwari -- or
- e2_p1 (e2): Phoolwari -- released -- film -- Which
- e2_p2 (e2): Phoolwari -- released -- film
- e2_p3 (e2): Phoolwari -- released -- first
- e2_p4 (e2): Phoolwari -- released
- e2_p5 (e2): Phoolwari -- released -- was
- e2_p6 (e2): Phoolwari -- released -- ,
- e2_p7 (e2): Phoolwari -- released -- ?
- e2_p8 (e2): Phoolwari -- or
- e2_p9 (e2): Phoolwari -- released -- Aas Ka Panchhi
- e2_p10 (e2): Phoolwari -- Aas Ka Panchhi
- e2_p11 (e2): Phoolwari -- Aas Ka Panchhi -- released -- film -- Which
- e2_p12 (e2): Phoolwari -- Aas Ka Panchhi -- released -- film
- e2_p13 (e2): Phoolwari -- Aas Ka Panchhi -- released -- first
- e2_p14 (e2): Phoolwari -- Aas Ka Panchhi -- released
- e2_p15 (e2): Phoolwari -- Aas Ka Panchhi -- released -- was
- e2_p16 (e2): Phoolwari -- Aas Ka Panchhi -- released -- ,
- e2_p17 (e2): Phoolwari -- Aas Ka Panchhi -- released -- ?

## 7.5 Terminal Glue Path Pruning
Total raw paths: 35
Total kept paths: 20
Total pruned paths: 15
Total pruned ratio: 42.86%

### By Entity
- e1 / Aas Ka Panchhi
  - raw: 18
  - kept: 10
  - pruned: 8
  - fallback_used: False
  - examples:
    - e1_p5: Aas Ka Panchhi -> released -> was [terminal=was, reason=terminal_glue_token]
    - e1_p6: Aas Ka Panchhi -> released -> , [terminal=,, reason=terminal_glue_token]
    - e1_p7: Aas Ka Panchhi -> released -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p14: Aas Ka Panchhi -> released -> Phoolwari -> or [terminal=or, reason=terminal_glue_token]
    - e1_p15: Aas Ka Panchhi -> Phoolwari -> released -> was [terminal=was, reason=terminal_glue_token]
- e2 / Phoolwari
  - raw: 17
  - kept: 10
  - pruned: 7
  - fallback_used: False
  - examples:
    - e2_p5: Phoolwari -> released -> was [terminal=was, reason=terminal_glue_token]
    - e2_p6: Phoolwari -> released -> , [terminal=,, reason=terminal_glue_token]
    - e2_p7: Phoolwari -> released -> ? [terminal=?, reason=terminal_glue_token]
    - e2_p8: Phoolwari -> or [terminal=or, reason=terminal_glue_token]
    - e2_p15: Phoolwari -> Aas Ka Panchhi -> released -> was [terminal=was, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, reaches the released predicate, and includes the which cue.
- e1: e1_p2 score=85.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi and reaches the released predicate but misses the which cue.
- e1: e1_p3 score=80.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, reaches the released predicate, and includes the first cue but misses the which cue.
- e1: e1_p4 score=60.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi and reaches the released predicate but lacks coverage for the answer intent.
- e1: e1_p8 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, reaches the released predicate, and includes the necessary cues.
- e1: e1_p9 score=50.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi but only connects to Phoolwari without reaching the necessary predicates.
- e1: e1_p10 score=95.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, connects to Phoolwari, and includes the released predicate and the which cue.
- e1: e1_p11 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, connects to Phoolwari, and includes the released predicate but misses the which cue.
- e1: e1_p12 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, connects to Phoolwari, and includes the released predicate and the first cue but misses the which cue.
- e1: e1_p13 score=70.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, connects to Phoolwari, but lacks coverage for the answer intent.
- e2: e2_p1 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, reaches the released predicate, and includes the which cue.
- e2: e2_p2 score=85.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari and reaches the released predicate but misses the which cue.
- e2: e2_p3 score=80.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, reaches the released predicate, and includes the first cue but misses the which cue.
- e2: e2_p4 score=60.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari and reaches the released predicate but lacks coverage for the answer intent.
- e2: e2_p9 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, reaches the released predicate, and includes the necessary cues.
- e2: e2_p10 score=50.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari but only connects to Aas Ka Panchhi without reaching the necessary predicates.
- e2: e2_p11 score=95.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, connects to Aas Ka Panchhi, and includes the released predicate and the which cue.
- e2: e2_p12 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, connects to Aas Ka Panchhi, and includes the released predicate and the first cue but misses the which cue.
- e2: e2_p13 score=70.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, connects to Aas Ka Panchhi, but lacks coverage for the answer intent.
- e2: e2_p14 score=70.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, connects to Aas Ka Panchhi, but lacks coverage for the answer intent.

## 8.1 Top-2 Paths per Entity
- e1: e1_p10, e1_p1
- e2: e2_p11, e2_p1

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p10', 'e2': 'e2_p11'} mean_path_score=95.0
- ps2: {'e1': 'e1_p10', 'e2': 'e2_p1'} mean_path_score=92.5
- ps3: {'e1': 'e1_p1', 'e2': 'e2_p11'} mean_path_score=92.5
- ps4: {'e1': 'e1_p1', 'e2': 'e2_p1'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Which film was released first, Aas Ka Panchhi or Phoolwari?
- ps1
  - e1_p10: Aas Ka Panchhi -> Phoolwari -> released -> film -> Which
  - e2_p11: Phoolwari -> Aas Ka Panchhi -> released -> film -> Which
- ps2
  - e1_p10: Aas Ka Panchhi -> Phoolwari -> released -> film -> Which
  - e2_p1: Phoolwari -> released -> film -> Which
- ps3
  - e1_p1: Aas Ka Panchhi -> released -> film -> Which
  - e2_p11: Phoolwari -> Aas Ka Panchhi -> released -> film -> Which
- ps4
  - e1_p1: Aas Ka Panchhi -> released -> film -> Which
  - e2_p1: Phoolwari -> released -> film -> Which

Output:
- selected_path_set_ids: ['ps3', 'ps4']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: When was Aas Ka Panchhi released? depends_on=[] support=['e1_p1']
- q2: When was Phoolwari released? depends_on=[] support=['e2_p1']

## 10. Atomic Subquestion DAG
- None: When was Aas Ka Panchhi released?
- None: When was Phoolwari released?
