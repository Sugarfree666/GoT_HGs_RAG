# DEPO Decomposition #6

- Dataset: `2wikimultihopqa`
- Question: Which film whose director was born first, El Tonto or The Heart Of Doreon?
- Gold answer: The Heart Of Doreon

## 1. Semantic-Normalized Question
Which film, whose director was born first, El Tonto or The Heart Of Doreon?

## 2. Explicit Entities
- El Tonto (Film) span=(43, 51)
- The Heart Of Doreon (Film) span=(55, 74)

## 3. Entity Masking
- FilmA -> El Tonto
- FilmB -> The Heart Of Doreon

Which film, whose director was born first, FilmA or FilmB?

## 4. CoreNLP Dependency Parse
- film[2] --det--> Which[1]
- film[2] --punct--> ,[3]
- director[5] --nmod:poss--> whose[4]
- born[7] --nsubj:pass--> director[5]
- born[7] --aux:pass--> was[6]
- film[2] --dep--> born[7]
- born[7] --advmod--> first[8]
- born[7] --punct--> ,[9]
- born[7] --obj--> FilmA[10]
- FilmB[12] --cc--> or[11]
- born[7] --obj--> FilmB[12]
- FilmA[10] --conj:or--> FilmB[12]
- film[2] --punct--> ?[13]

## 5. Undirected Dependency Graph
- Which[1] --det-- film[2]
- film[2] --punct-- ,[3]
- film[2] --dep-- born[7]
- film[2] --punct-- ?[13]
- whose[4] --nmod:poss-- director[5]
- director[5] --nsubj:pass-- born[7]
- was[6] --aux:pass-- born[7]
- born[7] --advmod-- first[8]
- born[7] --punct-- ,[9]
- born[7] --obj-- El Tonto[10]
- born[7] --obj-- The Heart Of Doreon[12]
- El Tonto[10] --conj:or-- The Heart Of Doreon[12]
- or[11] --cc-- The Heart Of Doreon[12]

## 6. Entity Start Nodes from Explicit Entities
- e1: El Tonto graph_node_ids=['10']
- e2: The Heart Of Doreon graph_node_ids=['12']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): El Tonto -- born -- film -- Which
- e1_p2 (e1): El Tonto -- born -- director
- e1_p3 (e1): El Tonto -- born -- director -- whose
- e1_p4 (e1): El Tonto -- born -- film
- e1_p5 (e1): El Tonto -- born -- first
- e1_p6 (e1): El Tonto -- born -- film -- ,
- e1_p7 (e1): El Tonto -- born -- film -- ?
- e1_p8 (e1): El Tonto -- born
- e1_p9 (e1): El Tonto -- born -- was
- e1_p10 (e1): El Tonto -- born -- ,
- e1_p11 (e1): El Tonto -- born -- The Heart Of Doreon
- e1_p12 (e1): El Tonto -- The Heart Of Doreon
- e1_p13 (e1): El Tonto -- The Heart Of Doreon -- born -- film -- Which
- e1_p14 (e1): El Tonto -- The Heart Of Doreon -- born -- director
- e1_p15 (e1): El Tonto -- The Heart Of Doreon -- born -- director -- whose
- e1_p16 (e1): El Tonto -- The Heart Of Doreon -- born -- film
- e1_p17 (e1): El Tonto -- The Heart Of Doreon -- born -- first
- e1_p18 (e1): El Tonto -- The Heart Of Doreon -- born -- film -- ,
- e1_p19 (e1): El Tonto -- The Heart Of Doreon -- born -- film -- ?
- e1_p20 (e1): El Tonto -- The Heart Of Doreon -- born
- e1_p21 (e1): El Tonto -- born -- The Heart Of Doreon -- or
- e1_p22 (e1): El Tonto -- The Heart Of Doreon -- born -- was
- e1_p23 (e1): El Tonto -- The Heart Of Doreon -- born -- ,
- e1_p24 (e1): El Tonto -- The Heart Of Doreon -- or
- e2_p1 (e2): The Heart Of Doreon -- born -- film -- Which
- e2_p2 (e2): The Heart Of Doreon -- born -- director
- e2_p3 (e2): The Heart Of Doreon -- born -- director -- whose
- e2_p4 (e2): The Heart Of Doreon -- born -- film
- e2_p5 (e2): The Heart Of Doreon -- born -- first
- e2_p6 (e2): The Heart Of Doreon -- born -- film -- ,
- e2_p7 (e2): The Heart Of Doreon -- born -- film -- ?
- e2_p8 (e2): The Heart Of Doreon -- born
- e2_p9 (e2): The Heart Of Doreon -- born -- was
- e2_p10 (e2): The Heart Of Doreon -- born -- ,
- e2_p11 (e2): The Heart Of Doreon -- or
- e2_p12 (e2): The Heart Of Doreon -- born -- El Tonto
- e2_p13 (e2): The Heart Of Doreon -- El Tonto
- e2_p14 (e2): The Heart Of Doreon -- El Tonto -- born -- film -- Which
- e2_p15 (e2): The Heart Of Doreon -- El Tonto -- born -- director
- e2_p16 (e2): The Heart Of Doreon -- El Tonto -- born -- director -- whose
- e2_p17 (e2): The Heart Of Doreon -- El Tonto -- born -- film
- e2_p18 (e2): The Heart Of Doreon -- El Tonto -- born -- first
- e2_p19 (e2): The Heart Of Doreon -- El Tonto -- born -- film -- ,
- e2_p20 (e2): The Heart Of Doreon -- El Tonto -- born -- film -- ?
- e2_p21 (e2): The Heart Of Doreon -- El Tonto -- born
- e2_p22 (e2): The Heart Of Doreon -- El Tonto -- born -- was
- e2_p23 (e2): The Heart Of Doreon -- El Tonto -- born -- ,

## 7.5 Terminal Glue Path Pruning
Total raw paths: 47
Total kept paths: 28
Total pruned paths: 19
Total pruned ratio: 40.43%

### By Entity
- e1 / El Tonto
  - raw: 24
  - kept: 14
  - pruned: 10
  - fallback_used: False
  - examples:
    - e1_p6: El Tonto -> born -> film -> , [terminal=,, reason=terminal_glue_token]
    - e1_p7: El Tonto -> born -> film -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p9: El Tonto -> born -> was [terminal=was, reason=terminal_glue_token]
    - e1_p10: El Tonto -> born -> , [terminal=,, reason=terminal_glue_token]
    - e1_p18: El Tonto -> The Heart Of Doreon -> born -> film -> , [terminal=,, reason=terminal_glue_token]
- e2 / The Heart Of Doreon
  - raw: 23
  - kept: 14
  - pruned: 9
  - fallback_used: False
  - examples:
    - e2_p6: The Heart Of Doreon -> born -> film -> , [terminal=,, reason=terminal_glue_token]
    - e2_p7: The Heart Of Doreon -> born -> film -> ? [terminal=?, reason=terminal_glue_token]
    - e2_p9: The Heart Of Doreon -> born -> was [terminal=was, reason=terminal_glue_token]
    - e2_p10: The Heart Of Doreon -> born -> , [terminal=,, reason=terminal_glue_token]
    - e2_p11: The Heart Of Doreon -> or [terminal=or, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=55.0 valid=True terminal=film
  Reason: The path starts from El Tonto and reaches the film node but lacks coverage of the director and the first cue.
- e1: e1_p2 score=75.0 valid=True terminal=director
  Reason: The path starts from El Tonto and covers the born predicate and director but misses the first cue.
- e1: e1_p3 score=90.0 valid=True terminal=director
  Reason: The path starts from El Tonto, covers the born predicate, includes the director and whose cues, but misses the first cue.
- e1: e1_p4 score=70.0 valid=True terminal=film
  Reason: The path starts from El Tonto and reaches the film node but lacks coverage of the director and the first cue.
- e1: e1_p5 score=80.0 valid=True terminal=first
  Reason: The path starts from El Tonto, covers the born predicate, includes the first cue, but misses the director.
- e1: e1_p8 score=30.0 valid=True terminal=born
  Reason: The path starts from El Tonto and only covers the born predicate, missing all other necessary cues.
- e1: e1_p11 score=85.0 valid=True terminal=film
  Reason: The path starts from El Tonto, covers the born predicate, but misses the director and first cues.
- e1: e1_p12 score=40.0 valid=True terminal=film
  Reason: The path starts from El Tonto and only connects to The Heart Of Doreon, missing all necessary cues.
- e1: e1_p13 score=90.0 valid=True terminal=film
  Reason: The path starts from El Tonto, connects to The Heart Of Doreon, covers the born predicate, and includes the which cue but misses the first cue.
- e1: e1_p14 score=85.0 valid=True terminal=director
  Reason: The path starts from El Tonto, connects to The Heart Of Doreon, covers the born predicate, but misses the first cue.
- e1: e1_p15 score=90.0 valid=True terminal=director
  Reason: The path starts from El Tonto, connects to The Heart Of Doreon, covers the born predicate, includes the whose cue but misses the first cue.
- e1: e1_p16 score=85.0 valid=True terminal=film
  Reason: The path starts from El Tonto, connects to The Heart Of Doreon, covers the born predicate, but misses the first and director cues.
- e1: e1_p17 score=90.0 valid=True terminal=film
  Reason: The path starts from El Tonto, connects to The Heart Of Doreon, covers the born predicate, but misses the first and director cues.
- e1: e1_p20 score=70.0 valid=True terminal=born
  Reason: The path starts from El Tonto and only connects to The Heart Of Doreon, missing all necessary cues.
- e2: e2_p1 score=55.0 valid=True terminal=film
  Reason: The path starts from The Heart Of Doreon and reaches the film node but lacks coverage of the director and the first cue.
- e2: e2_p2 score=75.0 valid=True terminal=director
  Reason: The path starts from The Heart Of Doreon and covers the born predicate and director but misses the first cue.
- e2: e2_p3 score=90.0 valid=True terminal=director
  Reason: The path starts from The Heart Of Doreon, covers the born predicate, includes the director and whose cues, but misses the first cue.
- e2: e2_p4 score=70.0 valid=True terminal=film
  Reason: The path starts from The Heart Of Doreon and reaches the film node but lacks coverage of the director and the first cue.
- e2: e2_p5 score=80.0 valid=True terminal=first
  Reason: The path starts from The Heart Of Doreon, covers the born predicate, includes the first cue, but misses the director.
- e2: e2_p8 score=30.0 valid=True terminal=born
  Reason: The path starts from The Heart Of Doreon and only covers the born predicate, missing all other necessary cues.
- e2: e2_p12 score=85.0 valid=True terminal=film
  Reason: The path starts from The Heart Of Doreon, covers the born predicate, but misses the director and first cues.
- e2: e2_p13 score=40.0 valid=True terminal=film
  Reason: The path starts from The Heart Of Doreon and only connects to El Tonto, missing all necessary cues.
- e2: e2_p14 score=90.0 valid=True terminal=film
  Reason: The path starts from The Heart Of Doreon, connects to El Tonto, covers the born predicate, and includes the which cue but misses the first cue.
- e2: e2_p15 score=90.0 valid=True terminal=director
  Reason: The path starts from The Heart Of Doreon, connects to El Tonto, covers the born predicate, includes the whose cue but misses the first cue.
- e2: e2_p16 score=85.0 valid=True terminal=film
  Reason: The path starts from The Heart Of Doreon, connects to El Tonto, covers the born predicate, but misses the first and director cues.
- e2: e2_p17 score=90.0 valid=True terminal=film
  Reason: The path starts from The Heart Of Doreon, connects to El Tonto, covers the born predicate, but misses the first and director cues.
- e2: e2_p18 score=0.0 valid=False
  Reason: missing from LLM output
- e2: e2_p21 score=70.0 valid=True terminal=born
  Reason: The path starts from The Heart Of Doreon and only connects to El Tonto, missing all necessary cues.

## 8.1 Highest-Scored Path per Entity
- e1: e1_p13 score=90.0
- e2: e2_p14 score=90.0

## 8.2 Selected Path Set
- ps1: {'e1': 'e1_p13', 'e2': 'e2_p14'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Which film whose director was born first, El Tonto or The Heart Of Doreon?
- ps1
  - e1_p13: El Tonto -> The Heart Of Doreon -> born -> film -> Which
  - e2_p14: The Heart Of Doreon -> El Tonto -> born -> film -> Which

Output:
- reason: The DAG compiles each semantic reasoning edge into one atomic lookup.
- q1: Who is the director of El Tonto? depends_on=[] support=[]
- q2: When was the director of El Tonto born? depends_on=['q1'] support=[]
- q3: Who is the director of The Heart Of Doreon? depends_on=[] support=[]
- q4: When was the director of The Heart Of Doreon born? depends_on=['q3'] support=[]

## 10. Atomic Subquestion DAG
- None: Who is the director of El Tonto?
- None: When was the director of El Tonto born?
- None: Who is the director of The Heart Of Doreon?
- None: When was the director of The Heart Of Doreon born?
