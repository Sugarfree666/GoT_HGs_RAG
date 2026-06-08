# DEPO Decomposition #7

- Dataset: `hotpotqa`
- Question: Which Walt Disney film was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?
- Gold answer: The Apple Dumpling Gang

## 1. Semantic-Normalized Question
Which Walt Disney film was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?

## 2. Explicit Entities
- Walt Disney (Organization) span=(6, 17)
- The Apple Dumpling Gang (Film) span=(43, 66)
- Something Wicked This Way Comes (Film) span=(70, 101)

## 3. Entity Masking
- OrganizationA -> Walt Disney
- FilmA -> The Apple Dumpling Gang
- FilmB -> Something Wicked This Way Comes

Which OrganizationA film was produced first, FilmA or FilmB?

## 4. CoreNLP Dependency Parse
- film[3] --det--> Which[1]
- film[3] --compound--> OrganizationA[2]
- produced[5] --nsubj:pass--> film[3]
- produced[5] --aux:pass--> was[4]
- produced[5] --advmod--> first[6]
- produced[5] --punct--> ,[7]
- produced[5] --obj--> FilmA[8]
- FilmB[10] --cc--> or[9]
- produced[5] --obj--> FilmB[10]
- FilmA[8] --conj:or--> FilmB[10]
- produced[5] --punct--> ?[11]

## 5. Undirected Dependency Graph
- Which[1] --det-- film[3]
- Walt Disney[2] --compound-- film[3]
- film[3] --nsubj:pass-- produced[5]
- was[4] --aux:pass-- produced[5]
- produced[5] --advmod-- first[6]
- produced[5] --punct-- ,[7]
- produced[5] --obj-- The Apple Dumpling Gang[8]
- produced[5] --obj-- Something Wicked This Way Comes[10]
- produced[5] --punct-- ?[11]
- The Apple Dumpling Gang[8] --conj:or-- Something Wicked This Way Comes[10]
- or[9] --cc-- Something Wicked This Way Comes[10]

## 6. Entity Start Nodes from Explicit Entities
- e1: Walt Disney graph_node_ids=['2']
- e2: The Apple Dumpling Gang graph_node_ids=['8']
- e3: Something Wicked This Way Comes graph_node_ids=['10']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Walt Disney -- film -- produced -- first
- e1_p2 (e1): Walt Disney -- film -- produced
- e1_p3 (e1): Walt Disney -- film -- produced -- was
- e1_p4 (e1): Walt Disney -- film -- produced -- ,
- e1_p5 (e1): Walt Disney -- film -- produced -- ?
- e1_p6 (e1): Walt Disney -- film -- Which
- e1_p7 (e1): Walt Disney -- film
- e1_p8 (e1): Walt Disney -- film -- produced -- The Apple Dumpling Gang
- e1_p9 (e1): Walt Disney -- film -- produced -- Something Wicked This Way Comes
- e1_p10 (e1): Walt Disney -- film -- produced -- Something Wicked This Way Comes -- or
- e1_p11 (e1): Walt Disney -- film -- produced -- The Apple Dumpling Gang -- Something Wicked This Way Comes
- e1_p12 (e1): Walt Disney -- film -- produced -- Something Wicked This Way Comes -- The Apple Dumpling Gang
- e1_p13 (e1): Walt Disney -- film -- produced -- The Apple Dumpling Gang -- Something Wicked This Way Comes -- or
- e2_p1 (e2): The Apple Dumpling Gang -- produced -- film -- Which
- e2_p2 (e2): The Apple Dumpling Gang -- produced -- film
- e2_p3 (e2): The Apple Dumpling Gang -- produced -- first
- e2_p4 (e2): The Apple Dumpling Gang -- produced
- e2_p5 (e2): The Apple Dumpling Gang -- produced -- was
- e2_p6 (e2): The Apple Dumpling Gang -- produced -- ,
- e2_p7 (e2): The Apple Dumpling Gang -- produced -- ?
- e2_p8 (e2): The Apple Dumpling Gang -- produced -- film -- Walt Disney
- e2_p9 (e2): The Apple Dumpling Gang -- produced -- Something Wicked This Way Comes
- e2_p10 (e2): The Apple Dumpling Gang -- Something Wicked This Way Comes
- e2_p11 (e2): The Apple Dumpling Gang -- Something Wicked This Way Comes -- produced -- film -- Which
- e2_p12 (e2): The Apple Dumpling Gang -- Something Wicked This Way Comes -- produced -- film
- e2_p13 (e2): The Apple Dumpling Gang -- Something Wicked This Way Comes -- produced -- first
- e2_p14 (e2): The Apple Dumpling Gang -- Something Wicked This Way Comes -- produced
- e2_p15 (e2): The Apple Dumpling Gang -- produced -- Something Wicked This Way Comes -- or
- e2_p16 (e2): The Apple Dumpling Gang -- Something Wicked This Way Comes -- produced -- was
- e2_p17 (e2): The Apple Dumpling Gang -- Something Wicked This Way Comes -- produced -- ,
- e2_p18 (e2): The Apple Dumpling Gang -- Something Wicked This Way Comes -- produced -- ?
- e2_p19 (e2): The Apple Dumpling Gang -- Something Wicked This Way Comes -- or
- e2_p20 (e2): The Apple Dumpling Gang -- Something Wicked This Way Comes -- produced -- film -- Walt Disney
- e3_p1 (e3): Something Wicked This Way Comes -- produced -- film -- Which
- e3_p2 (e3): Something Wicked This Way Comes -- produced -- film
- e3_p3 (e3): Something Wicked This Way Comes -- produced -- first
- e3_p4 (e3): Something Wicked This Way Comes -- produced
- e3_p5 (e3): Something Wicked This Way Comes -- produced -- was
- e3_p6 (e3): Something Wicked This Way Comes -- produced -- ,
- e3_p7 (e3): Something Wicked This Way Comes -- produced -- ?
- e3_p8 (e3): Something Wicked This Way Comes -- or
- e3_p9 (e3): Something Wicked This Way Comes -- produced -- film -- Walt Disney
- e3_p10 (e3): Something Wicked This Way Comes -- produced -- The Apple Dumpling Gang
- e3_p11 (e3): Something Wicked This Way Comes -- The Apple Dumpling Gang
- e3_p12 (e3): Something Wicked This Way Comes -- The Apple Dumpling Gang -- produced -- film -- Which
- e3_p13 (e3): Something Wicked This Way Comes -- The Apple Dumpling Gang -- produced -- film
- e3_p14 (e3): Something Wicked This Way Comes -- The Apple Dumpling Gang -- produced -- first
- e3_p15 (e3): Something Wicked This Way Comes -- The Apple Dumpling Gang -- produced
- e3_p16 (e3): Something Wicked This Way Comes -- The Apple Dumpling Gang -- produced -- was
- e3_p17 (e3): Something Wicked This Way Comes -- The Apple Dumpling Gang -- produced -- ,
- e3_p18 (e3): Something Wicked This Way Comes -- The Apple Dumpling Gang -- produced -- ?
- e3_p19 (e3): Something Wicked This Way Comes -- The Apple Dumpling Gang -- produced -- film -- Walt Disney

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Walt Disney, covers the film produced predicate, and includes the first cue, directly supporting the question intent.
- e1: e1_p2 score=85.0 valid=True terminal=film_production_order
  Reason: The path starts from Walt Disney and covers the produced predicate, but it misses the first cue, which is important for the question intent.
- e1: e1_p3 score=75.0 valid=True terminal=film_production_order
  Reason: The path starts from Walt Disney and covers the produced predicate, but it includes an auxiliary that does not contribute to the question intent.
- e1: e1_p4 score=30.0 valid=False terminal=film_production_order
  Reason: The path stops at a punctuation mark, failing to reach the necessary predicates for the question intent.
- e1: e1_p5 score=30.0 valid=False terminal=film_production_order
  Reason: The path stops at a punctuation mark, failing to reach the necessary predicates for the question intent.
- e1: e1_p6 score=55.0 valid=True terminal=film_production_order
  Reason: The path starts from Walt Disney and covers the produced predicate, but it misses the first cue, which is important for the question intent.
- e1: e1_p7 score=20.0 valid=False terminal=film_production_order
  Reason: The path is too short and does not cover the necessary predicates for the question intent.
- e1: e1_p8 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Walt Disney, covers the produced predicate, and includes the film title, directly supporting the question intent.
- e1: e1_p9 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Walt Disney, covers the produced predicate, and includes the film title, directly supporting the question intent.
- e1: e1_p10 score=75.0 valid=True terminal=film_production_order
  Reason: The path starts from Walt Disney, covers the produced predicate, and includes the film title, but it misses the first cue, which is important for the question intent.
- e1: e1_p11 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Walt Disney, covers the produced predicate, and includes both film titles, directly supporting the question intent.
- e1: e1_p12 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Walt Disney, covers the produced predicate, and includes both film titles, directly supporting the question intent.
- e1: e1_p13 score=75.0 valid=True terminal=film_production_order
  Reason: The path starts from Walt Disney, covers the produced predicate, and includes both film titles, but it misses the first cue, which is important for the question intent.
- e2: e2_p1 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from The Apple Dumpling Gang, covers the produced predicate, and includes the film title, directly supporting the question intent.
- e2: e2_p2 score=85.0 valid=True terminal=film_production_order
  Reason: The path starts from The Apple Dumpling Gang and covers the produced predicate, but it misses the first cue, which is important for the question intent.
- e2: e2_p3 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from The Apple Dumpling Gang, covers the produced predicate, and includes the first cue, directly supporting the question intent.
- e2: e2_p4 score=30.0 valid=False terminal=film_production_order
  Reason: The path is too short and does not cover the necessary predicates for the question intent.
- e2: e2_p5 score=30.0 valid=False terminal=film_production_order
  Reason: The path is too short and does not cover the necessary predicates for the question intent.
- e2: e2_p6 score=30.0 valid=False terminal=film_production_order
  Reason: The path is too short and does not cover the necessary predicates for the question intent.
- e2: e2_p7 score=30.0 valid=False terminal=film_production_order
  Reason: The path is too short and does not cover the necessary predicates for the question intent.
- e2: e2_p8 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from The Apple Dumpling Gang, covers the produced predicate, and includes the organization title, directly supporting the question intent.
- e2: e2_p9 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from The Apple Dumpling Gang, covers the produced predicate, and includes the film title, directly supporting the question intent.
- e2: e2_p10 score=75.0 valid=True terminal=film_production_order
  Reason: The path starts from The Apple Dumpling Gang, covers the produced predicate, and includes the film title, but it misses the first cue, which is important for the question intent.
- e2: e2_p11 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from The Apple Dumpling Gang, covers the produced predicate, and includes both film titles, directly supporting the question intent.
- e2: e2_p12 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from The Apple Dumpling Gang, covers the produced predicate, and includes both film titles, directly supporting the question intent.
- e2: e2_p13 score=75.0 valid=True terminal=film_production_order
  Reason: The path starts from The Apple Dumpling Gang, covers the produced predicate, and includes both film titles, but it misses the first cue, which is important for the question intent.
- e2: e2_p14 score=0.0 valid=False
  Reason: missing from LLM output
- e2: e2_p15 score=0.0 valid=False
  Reason: missing from LLM output
- e2: e2_p16 score=0.0 valid=False
  Reason: missing from LLM output
- e2: e2_p17 score=0.0 valid=False
  Reason: missing from LLM output
- e2: e2_p18 score=0.0 valid=False
  Reason: missing from LLM output
- e2: e2_p19 score=0.0 valid=False
  Reason: missing from LLM output
- e2: e2_p20 score=0.0 valid=False
  Reason: missing from LLM output
- e3: e3_p1 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes, covers the produced predicate, and includes the film title, directly supporting the question intent.
- e3: e3_p2 score=85.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes and covers the produced predicate, but it misses the first cue, which is important for the question intent.
- e3: e3_p3 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes, covers the produced predicate, and includes the first cue, directly supporting the question intent.
- e3: e3_p4 score=30.0 valid=False terminal=film_production_order
  Reason: The path is too short and does not cover the necessary predicates for the question intent.
- e3: e3_p5 score=30.0 valid=False terminal=film_production_order
  Reason: The path is too short and does not cover the necessary predicates for the question intent.
- e3: e3_p6 score=30.0 valid=False terminal=film_production_order
  Reason: The path is too short and does not cover the necessary predicates for the question intent.
- e3: e3_p7 score=30.0 valid=False terminal=film_production_order
  Reason: The path is too short and does not cover the necessary predicates for the question intent.
- e3: e3_p8 score=20.0 valid=False terminal=film_production_order
  Reason: The path is too short and does not cover the necessary predicates for the question intent.
- e3: e3_p9 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes, covers the produced predicate, and includes the film title, directly supporting the question intent.
- e3: e3_p10 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes, covers the produced predicate, and includes the film title, directly supporting the question intent.
- e3: e3_p11 score=30.0 valid=False terminal=film_production_order
  Reason: The path is too short and does not cover the necessary predicates for the question intent.
- e3: e3_p12 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes, covers the produced predicate, and includes both film titles, directly supporting the question intent.
- e3: e3_p13 score=75.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes, covers the produced predicate, and includes both film titles, but it misses the first cue, which is important for the question intent.
- e3: e3_p14 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes, covers the produced predicate, and includes the first cue, directly supporting the question intent.
- e3: e3_p15 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes, covers the produced predicate, and includes the film title, directly supporting the question intent.
- e3: e3_p16 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes, covers the produced predicate, and includes the first cue, directly supporting the question intent.
- e3: e3_p17 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes, covers the produced predicate, and includes the film title, directly supporting the question intent.
- e3: e3_p18 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes, covers the produced predicate, and includes the film title, directly supporting the question intent.
- e3: e3_p19 score=90.0 valid=True terminal=film_production_order
  Reason: The path starts from Something Wicked This Way Comes, covers the produced predicate, and includes the film title, directly supporting the question intent.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p11
- e2: e2_p1, e2_p11
- e3: e3_p1, e3_p10

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1', 'e2': 'e2_p1', 'e3': 'e3_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p1', 'e2': 'e2_p1', 'e3': 'e3_p10'} mean_path_score=90.0
- ps3: {'e1': 'e1_p1', 'e2': 'e2_p11', 'e3': 'e3_p1'} mean_path_score=90.0
- ps4: {'e1': 'e1_p1', 'e2': 'e2_p11', 'e3': 'e3_p10'} mean_path_score=90.0
- ps5: {'e1': 'e1_p11', 'e2': 'e2_p1', 'e3': 'e3_p1'} mean_path_score=90.0
- ps6: {'e1': 'e1_p11', 'e2': 'e2_p1', 'e3': 'e3_p10'} mean_path_score=90.0
- ps7: {'e1': 'e1_p11', 'e2': 'e2_p11', 'e3': 'e3_p1'} mean_path_score=90.0
- ps8: {'e1': 'e1_p11', 'e2': 'e2_p11', 'e3': 'e3_p10'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Which Walt Disney film was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?
- ps1
  - e1_p1: Walt Disney -> film -> produced -> first
  - e2_p1: The Apple Dumpling Gang -> produced -> film -> Which
  - e3_p1: Something Wicked This Way Comes -> produced -> film -> Which
- ps2
  - e1_p1: Walt Disney -> film -> produced -> first
  - e2_p1: The Apple Dumpling Gang -> produced -> film -> Which
  - e3_p10: Something Wicked This Way Comes -> produced -> The Apple Dumpling Gang
- ps3
  - e1_p1: Walt Disney -> film -> produced -> first
  - e2_p11: The Apple Dumpling Gang -> Something Wicked This Way Comes -> produced -> film -> Which
  - e3_p1: Something Wicked This Way Comes -> produced -> film -> Which
- ps4
  - e1_p1: Walt Disney -> film -> produced -> first
  - e2_p11: The Apple Dumpling Gang -> Something Wicked This Way Comes -> produced -> film -> Which
  - e3_p10: Something Wicked This Way Comes -> produced -> The Apple Dumpling Gang

Output:
- selected_path_set_ids: ['ps1']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: When was The Apple Dumpling Gang produced? depends_on=[] support=['e2_p1']
- q2: When was Something Wicked This Way Comes produced? depends_on=[] support=['e3_p1']

## 10. Atomic Subquestion DAG
- None: When was The Apple Dumpling Gang produced?
- None: When was Something Wicked This Way Comes produced?
