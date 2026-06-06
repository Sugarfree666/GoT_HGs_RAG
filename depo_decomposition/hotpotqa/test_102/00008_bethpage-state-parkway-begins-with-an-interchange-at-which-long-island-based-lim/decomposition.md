# DEPO Decomposition #8

- Dataset: `hotpotqa`
- Question: Bethpage State Parkway begins with an interchange at which Long Island-based limited access highway?
- Gold answer: Southern State Parkway

## 1. Semantic-Normalized Question
At which interchange does the Bethpage State Parkway begin with a Long Island-based limited access highway?

## 2. Mask Spans
- Bethpage State Parkway (entity, BethpageStateParkway)
- Long Island-based (entity, Region)

## 3. Selective Masked Question
At which interchange does the SomeEntityA begin with a RegionA limited access highway?

## 4. CoreNLP Dependency Parse
- which[2] --case--> At[1]
- does[4] --obl:at--> which[2]
- does[4] --nsubj--> interchange[3]
- SomeEntityA[6] --det--> the[5]
- begin[7] --nsubj--> SomeEntityA[6]
- does[4] --ccomp--> begin[7]
- highway[13] --case--> with[8]
- highway[13] --det--> a[9]
- highway[13] --compound--> RegionA[10]
- access[12] --amod--> limited[11]
- highway[13] --compound--> access[12]
- begin[7] --obl:with--> highway[13]
- does[4] --punct--> ?[14]

## 5. Undirected Dependency Graph
- At[1] --case-- which[2]
- which[2] --obl:at-- does[4]
- interchange[3] --nsubj-- does[4]
- does[4] --ccomp-- begin[7]
- does[4] --punct-- ?[14]
- the[5] --det-- Bethpage State Parkway[6]
- Bethpage State Parkway[6] --nsubj-- begin[7]
- begin[7] --obl:with-- highway[13]
- with[8] --case-- highway[13]
- a[9] --det-- highway[13]
- Long Island-based[10] --compound-- highway[13]
- limited[11] --amod-- access[12]
- access[12] --compound-- highway[13]

## 6. Entity Start Nodes
- e1: Bethpage State Parkway graph_node_ids=['6']
- e2: Long Island-based graph_node_ids=['10']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Bethpage State Parkway -- begin -- highway -- access -- limited
- e1_p2 (e1): Bethpage State Parkway -- begin -- highway -- access
- e1_p3 (e1): Bethpage State Parkway -- begin -- does -- which -- At
- e1_p4 (e1): Bethpage State Parkway -- begin -- highway
- e1_p5 (e1): Bethpage State Parkway -- begin -- does -- interchange
- e1_p6 (e1): Bethpage State Parkway -- begin -- highway -- with
- e1_p7 (e1): Bethpage State Parkway -- begin -- highway -- a
- e1_p8 (e1): Bethpage State Parkway -- begin -- does -- which
- e1_p9 (e1): Bethpage State Parkway -- begin
- e1_p10 (e1): Bethpage State Parkway -- begin -- does
- e1_p11 (e1): Bethpage State Parkway -- begin -- does -- ?
- e1_p12 (e1): Bethpage State Parkway -- the
- e1_p13 (e1): Bethpage State Parkway -- begin -- highway -- Long Island-based
- e2_p1 (e2): Long Island-based -- highway -- begin -- does -- which -- At
- e2_p2 (e2): Long Island-based -- highway -- access -- limited
- e2_p3 (e2): Long Island-based -- highway -- begin -- does -- interchange
- e2_p4 (e2): Long Island-based -- highway -- begin -- does -- which
- e2_p5 (e2): Long Island-based -- highway -- begin
- e2_p6 (e2): Long Island-based -- highway -- access
- e2_p7 (e2): Long Island-based -- highway -- begin -- does
- e2_p8 (e2): Long Island-based -- highway -- begin -- does -- ?
- e2_p9 (e2): Long Island-based -- highway
- e2_p10 (e2): Long Island-based -- highway -- with
- e2_p11 (e2): Long Island-based -- highway -- a
- e2_p12 (e2): Long Island-based -- highway -- begin -- Bethpage State Parkway
- e2_p13 (e2): Long Island-based -- highway -- begin -- Bethpage State Parkway -- the

## 8. LLM Path Scores
- e1: e1_p1 score=85.0 valid=True terminal=interchange
  Reason: The path starts from Bethpage State Parkway, covers the begin predicate, and includes the necessary intermediate roles leading to the interchange.
- e1: e1_p2 score=80.0 valid=True terminal=interchange
  Reason: The path starts from Bethpage State Parkway and covers the begin predicate, leading to the highway, but lacks the final interchange cue.
- e1: e1_p3 score=30.0 valid=False terminal=interchange
  Reason: The path stops too early and does not reach the necessary interchange.
- e1: e1_p4 score=70.0 valid=True terminal=interchange
  Reason: The path starts from Bethpage State Parkway and covers the begin predicate, but does not reach the interchange.
- e1: e1_p5 score=75.0 valid=True terminal=interchange
  Reason: The path starts from Bethpage State Parkway and covers the begin predicate, leading to interchange, but lacks a direct connection.
- e1: e1_p6 score=50.0 valid=False terminal=interchange
  Reason: The path does not reach the necessary interchange and stops too early.
- e1: e1_p7 score=40.0 valid=False terminal=interchange
  Reason: The path does not reach the necessary interchange and is too minimal.
- e1: e1_p8 score=60.0 valid=False terminal=interchange
  Reason: The path does not reach the necessary interchange and is incomplete.
- e1: e1_p9 score=20.0 valid=False terminal=interchange
  Reason: The path is too minimal and does not provide any useful information.
- e1: e1_p10 score=50.0 valid=False terminal=interchange
  Reason: The path does not reach the necessary interchange and is incomplete.
- e1: e1_p11 score=10.0 valid=False terminal=interchange
  Reason: The path is too minimal and does not provide any useful information.
- e1: e1_p12 score=75.0 valid=True terminal=interchange
  Reason: The path starts from Bethpage State Parkway and covers the begin predicate, leading to the highway, but lacks a direct connection to interchange.
- e1: e1_p13 score=90.0 valid=True terminal=interchange
  Reason: The path starts from Bethpage State Parkway, covers the begin predicate, and includes the necessary intermediate roles leading to the Long Island-based highway.
- e2: e2_p1 score=85.0 valid=True terminal=interchange
  Reason: The path starts from Long Island-based, covers the begin predicate, and includes the necessary intermediate roles leading to the interchange.
- e2: e2_p2 score=80.0 valid=True terminal=interchange
  Reason: The path starts from Long Island-based and covers the highway and access, but does not reach the interchange.
- e2: e2_p3 score=75.0 valid=True terminal=interchange
  Reason: The path starts from Long Island-based and covers the begin predicate, leading to interchange, but lacks a direct connection.
- e2: e2_p4 score=70.0 valid=True terminal=interchange
  Reason: The path starts from Long Island-based and covers the begin predicate, but does not reach the interchange.
- e2: e2_p5 score=60.0 valid=False terminal=interchange
  Reason: The path does not reach the necessary interchange and stops too early.
- e2: e2_p6 score=50.0 valid=False terminal=interchange
  Reason: The path does not reach the necessary interchange and is incomplete.
- e2: e2_p7 score=40.0 valid=False terminal=interchange
  Reason: The path does not reach the necessary interchange and is too minimal.
- e2: e2_p8 score=30.0 valid=False terminal=interchange
  Reason: The path is too minimal and does not provide any useful information.
- e2: e2_p9 score=20.0 valid=False terminal=interchange
  Reason: The path is too minimal and does not provide any useful information.
- e2: e2_p10 score=50.0 valid=False terminal=interchange
  Reason: The path does not reach the necessary interchange and is incomplete.
- e2: e2_p11 score=10.0 valid=False terminal=interchange
  Reason: The path is too minimal and does not provide any useful information.
- e2: e2_p12 score=90.0 valid=True terminal=interchange
  Reason: The path starts from Long Island-based, covers the begin predicate, and includes the necessary intermediate roles leading to the Bethpage State Parkway.
- e2: e2_p13 score=95.0 valid=True terminal=interchange
  Reason: The path starts from Long Island-based, covers the begin predicate, and includes the necessary intermediate roles leading to the Bethpage State Parkway.

## 8.1 Top-2 Paths per Entity
- e1: e1_p13, e1_p1
- e2: e2_p13, e2_p12

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p13', 'e2': 'e2_p13'} mean_path_score=92.5
- ps2: {'e1': 'e1_p13', 'e2': 'e2_p12'} mean_path_score=90.0
- ps3: {'e1': 'e1_p1', 'e2': 'e2_p13'} mean_path_score=90.0
- ps4: {'e1': 'e1_p1', 'e2': 'e2_p12'} mean_path_score=87.5

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- bethpage_state_parkway -> highway (begin with highway)
- highway -> interchange (interchange of highway)
- long_island_based -> long_island_based_highway (compound of highway)
- long_island_based_highway -> interchange (interchange of Long Island-based highway)
### ast_ps2 (ps2)
- bethpage_state_parkway -> highway (begin with highway)
- highway -> interchange (interchange of highway)
- long_island_based -> highway (descriptor of highway)
### ast_ps3 (ps3)
- bethpage_state_parkway -> highway_e1 (begin with highway)
- highway_e1 -> interchange_e1 (interchange of highway)
- highway_e2 -> interchange_e2 (interchange of highway)
- long_island_based -> highway_e2 (compound of highway)
### ast_ps4 (ps4)
- bethpage_state_parkway -> highway_e1 (highway associated with Bethpage State Parkway)
- bethpage_state_parkway -> highway_e2 (highway associated with Bethpage State Parkway)
- highway_e1 -> interchange_e1 (interchange for highway)
- highway_e2 -> interchange_e2 (interchange for highway)
- long_island_based -> highway_e2 (highway associated with Long Island-based)
- highway_e1 -> bethpage_state_parkway (highway begins with Bethpage State Parkway)
- highway_e2 -> bethpage_state_parkway (highway begins with Bethpage State Parkway)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively captures the relationship between the Bethpage State Parkway and the Long Island-based highway, allowing for clear decomposition into atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- bethpage_state_parkway: Bethpage State Parkway (entity)
- highway: highway (type_variable)
- interchange: interchange (value_slot)
- long_island_based: Long Island-based (type_variable)
- long_island_based_highway: Long Island-based highway (value_slot)

Edges:
- bethpage_state_parkway -> highway (begin with highway)
- highway -> interchange (interchange of highway)
- long_island_based -> long_island_based_highway (compound of highway)
- long_island_based_highway -> interchange (interchange of Long Island-based highway)

## 11. Atomic Subquestion DAG
- None: What highway does the Bethpage State Parkway begin with?
- None: At which interchange does the highway of Bethpage State Parkway begin?
- None: What is the Long Island-based highway?
- None: What is the interchange of the Long Island-based highway?
