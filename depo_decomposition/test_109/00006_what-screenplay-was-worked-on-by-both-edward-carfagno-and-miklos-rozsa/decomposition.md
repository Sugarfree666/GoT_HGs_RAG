# DEPO Decomposition #6

- Dataset: `hotpotqa`
- Question: What screenplay was worked on by both Edward Carfagno and Miklos Rozsa?
- Gold answer: Julius Caesar

## 1. Semantic-Normalized Question
What screenplay was worked on by both Edward Carfagno and Miklos Rozsa?

## 2. Explicit Entities
- Edward Carfagno (Person) span=(38, 53)
- Miklos Rozsa (Person) span=(58, 70)

## 3. Entity Masking
- PersonA -> Edward Carfagno
- PersonB -> Miklos Rozsa

What screenplay was worked on by both PersonA and PersonB?

## 4. CoreNLP Dependency Parse
- screenplay[2] --det--> What[1]
- worked[4] --nsubj:pass--> screenplay[2]
- worked[4] --aux:pass--> was[3]
- worked[4] --compound:prt--> on[5]
- PersonA[8] --case--> by[6]
- PersonA[8] --cc:preconj--> both[7]
- worked[4] --obl:agent--> PersonA[8]
- PersonB[10] --cc--> and[9]
- worked[4] --obl:agent--> PersonB[10]
- PersonA[8] --conj:and--> PersonB[10]
- worked[4] --punct--> ?[11]

## 5. Undirected Dependency Graph
- What[1] --det-- screenplay[2]
- screenplay[2] --nsubj:pass-- worked[4]
- was[3] --aux:pass-- worked[4]
- worked[4] --compound:prt-- on[5]
- worked[4] --obl:agent-- Edward Carfagno[8]
- worked[4] --obl:agent-- Miklos Rozsa[10]
- worked[4] --punct-- ?[11]
- by[6] --case-- Edward Carfagno[8]
- both[7] --cc:preconj-- Edward Carfagno[8]
- Edward Carfagno[8] --conj:and-- Miklos Rozsa[10]
- and[9] --cc-- Miklos Rozsa[10]

## 6. Entity Start Nodes from Explicit Entities
- e1: Edward Carfagno graph_node_ids=['8']
- e2: Miklos Rozsa graph_node_ids=['10']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Edward Carfagno -- worked -- screenplay -- What
- e1_p2 (e1): Edward Carfagno -- worked -- screenplay
- e1_p3 (e1): Edward Carfagno -- worked -- on
- e1_p4 (e1): Edward Carfagno -- worked
- e1_p5 (e1): Edward Carfagno -- both
- e1_p6 (e1): Edward Carfagno -- worked -- was
- e1_p7 (e1): Edward Carfagno -- worked -- ?
- e1_p8 (e1): Edward Carfagno -- by
- e1_p9 (e1): Edward Carfagno -- worked -- Miklos Rozsa
- e1_p10 (e1): Edward Carfagno -- Miklos Rozsa
- e1_p11 (e1): Edward Carfagno -- Miklos Rozsa -- worked -- screenplay -- What
- e1_p12 (e1): Edward Carfagno -- Miklos Rozsa -- worked -- screenplay
- e1_p13 (e1): Edward Carfagno -- Miklos Rozsa -- worked -- on
- e1_p14 (e1): Edward Carfagno -- Miklos Rozsa -- worked
- e1_p15 (e1): Edward Carfagno -- worked -- Miklos Rozsa -- and
- e1_p16 (e1): Edward Carfagno -- Miklos Rozsa -- worked -- was
- e1_p17 (e1): Edward Carfagno -- Miklos Rozsa -- worked -- ?
- e1_p18 (e1): Edward Carfagno -- Miklos Rozsa -- and
- e2_p1 (e2): Miklos Rozsa -- worked -- screenplay -- What
- e2_p2 (e2): Miklos Rozsa -- worked -- screenplay
- e2_p3 (e2): Miklos Rozsa -- worked -- on
- e2_p4 (e2): Miklos Rozsa -- worked
- e2_p5 (e2): Miklos Rozsa -- worked -- was
- e2_p6 (e2): Miklos Rozsa -- worked -- ?
- e2_p7 (e2): Miklos Rozsa -- and
- e2_p8 (e2): Miklos Rozsa -- worked -- Edward Carfagno
- e2_p9 (e2): Miklos Rozsa -- Edward Carfagno
- e2_p10 (e2): Miklos Rozsa -- Edward Carfagno -- worked -- screenplay -- What
- e2_p11 (e2): Miklos Rozsa -- worked -- Edward Carfagno -- both
- e2_p12 (e2): Miklos Rozsa -- Edward Carfagno -- worked -- screenplay
- e2_p13 (e2): Miklos Rozsa -- Edward Carfagno -- worked -- on
- e2_p14 (e2): Miklos Rozsa -- Edward Carfagno -- worked
- e2_p15 (e2): Miklos Rozsa -- Edward Carfagno -- both
- e2_p16 (e2): Miklos Rozsa -- worked -- Edward Carfagno -- by
- e2_p17 (e2): Miklos Rozsa -- Edward Carfagno -- worked -- was
- e2_p18 (e2): Miklos Rozsa -- Edward Carfagno -- worked -- ?
- e2_p19 (e2): Miklos Rozsa -- Edward Carfagno -- by

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno, reaches the focus predicate 'worked', and includes the answer slot cue 'what'.
- e1: e1_p2 score=85.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno, reaches the focus predicate 'worked', and includes the answer slot cue 'what', but lacks a direct connection to screenplay.
- e1: e1_p3 score=70.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno and reaches the predicate 'worked', but does not connect to the answer slot cue 'what'.
- e1: e1_p4 score=50.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno and reaches the predicate 'worked', but does not connect to screenplay or the answer slot cue 'what'.
- e1: e1_p5 score=30.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno but only connects to 'both', missing the necessary predicates and answer cues.
- e1: e1_p6 score=40.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno and reaches the predicate 'worked', but does not connect to screenplay or the answer slot cue 'what'.
- e1: e1_p7 score=20.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno but ends with a question mark, failing to connect to any relevant predicates or answer cues.
- e1: e1_p8 score=25.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno but only connects to 'by', missing the necessary predicates and answer cues.
- e1: e1_p9 score=95.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno, reaches the focus predicate 'worked', and includes the answer slot cue 'what', while also connecting to Miklos Rozsa.
- e1: e1_p10 score=50.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno and connects to Miklos Rozsa, but does not reach the necessary predicates or answer cues.
- e1: e1_p11 score=100.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno, connects to Miklos Rozsa, reaches the focus predicate 'worked', and includes the answer slot cue 'what'.
- e1: e1_p12 score=90.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno, connects to Miklos Rozsa, reaches the focus predicate 'worked', and includes the answer slot cue 'what'.
- e1: e1_p13 score=85.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno, connects to Miklos Rozsa, reaches the focus predicate 'worked', and includes the answer slot cue 'what'.
- e1: e1_p14 score=70.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno, connects to Miklos Rozsa, and reaches the predicate 'worked', but does not connect to the answer slot cue 'what'.
- e1: e1_p15 score=80.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno, connects to Miklos Rozsa, reaches the predicate 'worked', but does not connect to the answer slot cue 'what'.
- e1: e1_p16 score=60.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno, connects to Miklos Rozsa, reaches the predicate 'worked', but does not connect to the answer slot cue 'what'.
- e1: e1_p17 score=50.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno, connects to Miklos Rozsa, but ends with a question mark, failing to connect to any relevant predicates or answer cues.
- e1: e1_p18 score=40.0 valid=True terminal=screenplay
  Reason: The path starts from Edward Carfagno, connects to Miklos Rozsa, but does not reach the necessary predicates or answer cues.
- e2: e2_p1 score=90.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa, reaches the focus predicate 'worked', and includes the answer slot cue 'what'.
- e2: e2_p2 score=85.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa, reaches the focus predicate 'worked', and includes the answer slot cue 'what', but lacks a direct connection to screenplay.
- e2: e2_p3 score=70.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa and reaches the predicate 'worked', but does not connect to the answer slot cue 'what'.
- e2: e2_p4 score=50.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa and reaches the predicate 'worked', but does not connect to screenplay or the answer slot cue 'what'.
- e2: e2_p5 score=40.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa and reaches the predicate 'worked', but does not connect to screenplay or the answer slot cue 'what'.
- e2: e2_p6 score=20.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa but ends with a question mark, failing to connect to any relevant predicates or answer cues.
- e2: e2_p7 score=25.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa but only connects to 'and', missing the necessary predicates and answer cues.
- e2: e2_p8 score=95.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa, connects to Edward Carfagno, reaches the focus predicate 'worked', and includes the answer slot cue 'what'.
- e2: e2_p9 score=50.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa and connects to Edward Carfagno, but does not reach the necessary predicates or answer cues.
- e2: e2_p10 score=100.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa, connects to Edward Carfagno, reaches the focus predicate 'worked', and includes the answer slot cue 'what'.
- e2: e2_p11 score=90.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa, connects to Edward Carfagno, reaches the focus predicate 'worked', and includes the answer slot cue 'what'.
- e2: e2_p12 score=90.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa, connects to Edward Carfagno, reaches the focus predicate 'worked', and includes the answer slot cue 'what'.
- e2: e2_p13 score=85.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa, connects to Edward Carfagno, reaches the focus predicate 'worked', and includes the answer slot cue 'what'.
- e2: e2_p14 score=70.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa, connects to Edward Carfagno, and reaches the predicate 'worked', but does not connect to the answer slot cue 'what'.
- e2: e2_p15 score=80.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa, connects to Edward Carfagno, reaches the predicate 'worked', but does not connect to the answer slot cue 'what'.
- e2: e2_p16 score=60.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa, connects to Edward Carfagno, reaches the predicate 'worked', but does not connect to the answer slot cue 'what'.
- e2: e2_p17 score=50.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa, connects to Edward Carfagno, but ends with a question mark, failing to connect to any relevant predicates or answer cues.
- e2: e2_p18 score=40.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa, connects to Edward Carfagno, but does not reach the necessary predicates or answer cues.
- e2: e2_p19 score=30.0 valid=True terminal=screenplay
  Reason: The path starts from Miklos Rozsa but only connects to 'by', missing the necessary predicates and answer cues.

## 8.1 Top-2 Paths per Entity
- e1: e1_p11, e1_p9
- e2: e2_p10, e2_p8

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p11', 'e2': 'e2_p10'} mean_path_score=100.0
- ps2: {'e1': 'e1_p11', 'e2': 'e2_p8'} mean_path_score=97.5
- ps3: {'e1': 'e1_p9', 'e2': 'e2_p10'} mean_path_score=97.5
- ps4: {'e1': 'e1_p9', 'e2': 'e2_p8'} mean_path_score=95.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: What screenplay was worked on by both Edward Carfagno and Miklos Rozsa?
- ps1
  - e1_p11: Edward Carfagno -> Miklos Rozsa -> worked -> screenplay -> What
  - e2_p10: Miklos Rozsa -> Edward Carfagno -> worked -> screenplay -> What
- ps2
  - e1_p11: Edward Carfagno -> Miklos Rozsa -> worked -> screenplay -> What
  - e2_p8: Miklos Rozsa -> worked -> Edward Carfagno
- ps3
  - e1_p9: Edward Carfagno -> worked -> Miklos Rozsa
  - e2_p10: Miklos Rozsa -> Edward Carfagno -> worked -> screenplay -> What
- ps4
  - e1_p9: Edward Carfagno -> worked -> Miklos Rozsa
  - e2_p8: Miklos Rozsa -> worked -> Edward Carfagno

Output:
- selected_path_set_ids: ['ps1']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: What screenplay was worked on by Edward Carfagno? depends_on=[] support=['e1_p11']
- q2: What screenplay was worked on by Miklos Rozsa? depends_on=[] support=['e2_p10']

## 10. Atomic Subquestion DAG
- None: What screenplay was worked on by Edward Carfagno?
- None: What screenplay was worked on by Miklos Rozsa?
