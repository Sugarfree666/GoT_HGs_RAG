# DEPO Decomposition #1

- Dataset: `hotpotqa`
- Question: what is one of the stars of  The Newcomers known for
- Gold answer: superhero roles as the Marvel Comics

## 1. Semantic-Normalized Question
what is one of the stars of The Newcomers known for?

## 2. Explicit Entities
- The Newcomers (Work) span=(28, 41)

## 3. Entity Masking
- WorkA -> The Newcomers

what is one of the stars of WorkA known for?

## 4. CoreNLP Dependency Parse
- what[1] --cop--> is[2]
- stars[6] --det:qmod--> one[3]
- one[3] --fixed--> of[4]
- stars[6] --det--> the[5]
- known[9] --nsubj--> stars[6]
- WorkA[8] --case--> of[7]
- stars[6] --nmod:of--> WorkA[8]
- what[1] --dep--> known[9]
- known[9] --dep--> for[10]
- what[1] --punct--> ?[11]

## 5. Undirected Dependency Graph
- what[1] --cop-- is[2]
- what[1] --dep-- known[9]
- what[1] --punct-- ?[11]
- one[3] --det:qmod-- stars[6]
- one[3] --fixed-- of[4]
- the[5] --det-- stars[6]
- stars[6] --nsubj-- known[9]
- stars[6] --nmod:of-- The Newcomers[8]
- of[7] --case-- The Newcomers[8]
- known[9] --dep-- for[10]

## 6. Entity Start Nodes from Explicit Entities
- e1: The Newcomers graph_node_ids=['8']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): The Newcomers -- stars -- known -- what
- e1_p2 (e1): The Newcomers -- stars -- known -- what -- is
- e1_p3 (e1): The Newcomers -- stars -- known -- what -- ?
- e1_p4 (e1): The Newcomers -- stars -- one
- e1_p5 (e1): The Newcomers -- stars -- known
- e1_p6 (e1): The Newcomers -- stars -- one -- of
- e1_p7 (e1): The Newcomers -- stars -- known -- for
- e1_p8 (e1): The Newcomers -- stars
- e1_p9 (e1): The Newcomers -- stars -- the
- e1_p10 (e1): The Newcomers -- of

## 8. LLM Path Scores
- e1: e1_p1 score=85.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers, reaches stars, and includes the known predicate, covering the answer intent well.
- e1: e1_p2 score=90.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers, reaches stars, includes the known predicate, and has the is copula, supporting the answer intent effectively.
- e1: e1_p3 score=80.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers, reaches stars, includes the known predicate, but ends with a question mark, which slightly reduces its executability.
- e1: e1_p4 score=70.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers and reaches stars, but it lacks the known predicate and the what cue, making it less effective.
- e1: e1_p5 score=75.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers, reaches stars, and includes the known predicate, but it misses the what cue, which is important for the answer intent.
- e1: e1_p6 score=65.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers and reaches stars, but it includes 'one' and 'of' without covering the known predicate or the what cue, making it less relevant.
- e1: e1_p7 score=95.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers, reaches stars, includes the known predicate, and has the for preposition, effectively supporting the answer intent.
- e1: e1_p8 score=50.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers and reaches stars, but it lacks the known predicate and any cues related to the answer intent, making it weakly related.
- e1: e1_p9 score=60.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers and reaches stars, but it includes 'the' without covering the known predicate or the what cue, making it less effective.
- e1: e1_p10 score=40.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers and only reaches 'of', which does not contribute to the answer intent or provide relevant information.

## 8.1 Top-2 Paths per Entity
- e1: e1_p7, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p7'} mean_path_score=95.0
- ps2: {'e1': 'e1_p2'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: what is one of the stars of  The Newcomers known for
- ps1
  - e1_p7: The Newcomers -> stars -> known -> for
- ps2
  - e1_p2: The Newcomers -> stars -> known -> what -> is

Output:
- selected_path_set_ids: ['ps1', 'ps2']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who are the stars of The Newcomers? depends_on=[] support=['e1_p7']
- q2: What is q1's answer known for? depends_on=['q1'] support=['e1_p2']

## 10. Atomic Subquestion DAG
- None: Who are the stars of The Newcomers?
- None: What is q1's answer known for?
