# DEPO Decomposition #1

- Dataset: `hotpotqa`
- Question: what is one of the stars of  The Newcomers known for
- Gold answer: superhero roles as the Marvel Comics

## 1. Semantic-Normalized Question
what is one of the stars of The Newcomers known for?

## 2. Mask Spans
- The Newcomers (entity, Film)

## 3. Selective Masked Question
what is one of the stars of MovieA known for?

## 4. CoreNLP Dependency Parse
- what[1] --cop--> is[2]
- stars[6] --det:qmod--> one[3]
- one[3] --fixed--> of[4]
- stars[6] --det--> the[5]
- known[9] --nsubj--> stars[6]
- MovieA[8] --case--> of[7]
- stars[6] --nmod:of--> MovieA[8]
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

## 6. Entity Start Nodes
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
  Reason: The path starts from The Newcomers, reaches stars, and includes the known predicate, but lacks a direct connection to the answer intent.
- e1: e1_p2 score=90.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers, reaches stars, includes the known predicate, and connects to the is copula, supporting the answer intent.
- e1: e1_p3 score=80.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers, reaches stars, includes the known predicate, but ends with punctuation, which weakens its executability.
- e1: e1_p4 score=70.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers and reaches stars, but it does not include the known predicate, which is crucial for the answer intent.
- e1: e1_p5 score=75.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers, reaches stars, and includes the known predicate, but lacks a direct connection to the what cue.
- e1: e1_p6 score=60.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers and reaches stars, but it does not include the known predicate or the what cue, which are essential for the answer intent.
- e1: e1_p7 score=95.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers, reaches stars, includes the known predicate, and connects to for, fully supporting the answer intent.
- e1: e1_p8 score=50.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers and reaches stars, but it lacks the known predicate and does not connect to the answer intent.
- e1: e1_p9 score=55.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers and reaches stars, but it lacks the known predicate and does not connect to the answer intent.
- e1: e1_p10 score=40.0 valid=True terminal=known_for
  Reason: The path starts from The Newcomers and reaches of, but it does not connect to any relevant predicates or answer intent.

## 8.1 Top-2 Paths per Entity
- e1: e1_p7, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p7'} mean_path_score=95.0
- ps2: {'e1': 'e1_p2'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- the_newcomers -> stars (stars of The Newcomers)
- stars -> known_for (known for)
### ast_ps2 (ps2)
- the_newcomers -> stars (stars of The Newcomers)
- stars -> known_for (known for)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST fully supports the original question by connecting 'The Newcomers' to its 'stars' and their 'known for' attributes, allowing for one-hop executable atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- the_newcomers: The Newcomers (entity)
- stars: stars (type_variable)
- known_for: known_for (value_slot)

Edges:
- the_newcomers -> stars (stars of The Newcomers)
- stars -> known_for (known for)

## 11. Atomic Subquestion DAG
- None: Who are the stars of The Newcomers?
- None: What is one of the stars of The Newcomers known for?
