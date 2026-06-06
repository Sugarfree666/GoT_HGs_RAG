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

## 8. LLM Selected Entity Paths
- e1: e1_p13 Bethpage State Parkway -- begin -- highway -- Long Island-based
  Reason: This path effectively connects the Bethpage State Parkway to the Long Island-based highway, providing a clear reasoning chain to the interchange.
- e2: e2_p12 Long Island-based -- highway -- begin -- Bethpage State Parkway
  Reason: This path connects Long Island-based directly to the Bethpage State Parkway, establishing a clear relationship to the interchange.

## 9. Selected Path Semantic Transduction
Nodes:
- bethpage_state_parkway: Bethpage State Parkway (entity)
- highway: highway (type_variable)
- long_island_based: Long Island-based (type_variable)
- interchange: interchange (value_slot)

Edges:
- bethpage_state_parkway -> highway (begin with highway)
- highway -> long_island_based (is a)
- highway -> interchange (located at)

## 10. Atomic Subquestion DAG
- None: What highway does the Bethpage State Parkway begin with?
- None: What is a Long Island-based highway?
- None: At which interchange is the highway of Bethpage State Parkway located?
