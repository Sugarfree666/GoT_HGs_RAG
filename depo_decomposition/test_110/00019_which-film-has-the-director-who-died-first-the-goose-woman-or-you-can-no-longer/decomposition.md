# DEPO Decomposition #19

- Dataset: `2wikimultihopqa`
- Question: Which film has the director who died first, The Goose Woman or You Can No Longer Remain Silent?
- Gold answer: You Can No Longer Remain Silent

## 1. Semantic-Normalized Question
Which film has the director who died first, The Goose Woman or You Can No Longer Remain Silent?

## 2. Explicit Entities
- The Goose Woman (Film) span=(44, 59)
- You Can No Longer Remain Silent (Film) span=(63, 94)

## 3. Entity Masking
- FilmA -> The Goose Woman
- FilmB -> You Can No Longer Remain Silent

Which film has the director who died first, FilmA or FilmB?

## 4. CoreNLP Dependency Parse
- film[2] --det--> Which[1]
- has[3] --nsubj--> film[2]
- director[5] --det--> the[4]
- has[3] --obj--> director[5]
- died[7] --nsubj--> director[5]
- director[5] --ref--> who[6]
- director[5] --acl:relcl--> died[7]
- died[7] --advmod--> first[8]
- died[7] --punct--> ,[9]
- died[7] --obj--> FilmA[10]
- FilmB[12] --cc--> or[11]
- died[7] --obj--> FilmB[12]
- FilmA[10] --conj:or--> FilmB[12]
- has[3] --punct--> ?[13]

## 5. Undirected Dependency Graph
- Which[1] --det-- film[2]
- film[2] --nsubj-- has[3]
- has[3] --obj-- director[5]
- has[3] --punct-- ?[13]
- the[4] --det-- director[5]
- director[5] --nsubj/acl:relcl-- died[7]
- director[5] --ref-- who[6]
- died[7] --advmod-- first[8]
- died[7] --punct-- ,[9]
- died[7] --obj-- The Goose Woman[10]
- died[7] --obj-- You Can No Longer Remain Silent[12]
- The Goose Woman[10] --conj:or-- You Can No Longer Remain Silent[12]
- or[11] --cc-- You Can No Longer Remain Silent[12]

## 6. Entity Start Nodes from Explicit Entities
- e1: The Goose Woman graph_node_ids=['10']
- e2: You Can No Longer Remain Silent graph_node_ids=['12']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): The Goose Woman -- died -- director -- has -- film -- Which
- e1_p2 (e1): The Goose Woman -- died -- director -- has -- film
- e1_p3 (e1): The Goose Woman -- died -- director -- has
- e1_p4 (e1): The Goose Woman -- died -- director -- has -- ?
- e1_p5 (e1): The Goose Woman -- died -- director -- who
- e1_p6 (e1): The Goose Woman -- died -- director
- e1_p7 (e1): The Goose Woman -- died -- director -- the
- e1_p8 (e1): The Goose Woman -- died -- first
- e1_p9 (e1): The Goose Woman -- died
- e1_p10 (e1): The Goose Woman -- died -- ,
- e1_p11 (e1): The Goose Woman -- died -- You Can No Longer Remain Silent
- e1_p12 (e1): The Goose Woman -- You Can No Longer Remain Silent
- e1_p13 (e1): The Goose Woman -- You Can No Longer Remain Silent -- died -- director -- has -- film -- Which
- e1_p14 (e1): The Goose Woman -- You Can No Longer Remain Silent -- died -- director -- has -- film
- e1_p15 (e1): The Goose Woman -- You Can No Longer Remain Silent -- died -- director -- has
- e1_p16 (e1): The Goose Woman -- You Can No Longer Remain Silent -- died -- director -- has -- ?
- e1_p17 (e1): The Goose Woman -- You Can No Longer Remain Silent -- died -- director -- who
- e1_p18 (e1): The Goose Woman -- You Can No Longer Remain Silent -- died -- director
- e1_p19 (e1): The Goose Woman -- You Can No Longer Remain Silent -- died -- director -- the
- e1_p20 (e1): The Goose Woman -- You Can No Longer Remain Silent -- died -- first
- e1_p21 (e1): The Goose Woman -- You Can No Longer Remain Silent -- died
- e1_p22 (e1): The Goose Woman -- died -- You Can No Longer Remain Silent -- or
- e1_p23 (e1): The Goose Woman -- You Can No Longer Remain Silent -- died -- ,
- e1_p24 (e1): The Goose Woman -- You Can No Longer Remain Silent -- or
- e2_p1 (e2): You Can No Longer Remain Silent -- died -- director -- has -- film -- Which
- e2_p2 (e2): You Can No Longer Remain Silent -- died -- director -- has -- film
- e2_p3 (e2): You Can No Longer Remain Silent -- died -- director -- has
- e2_p4 (e2): You Can No Longer Remain Silent -- died -- director -- has -- ?
- e2_p5 (e2): You Can No Longer Remain Silent -- died -- director -- who
- e2_p6 (e2): You Can No Longer Remain Silent -- died -- director
- e2_p7 (e2): You Can No Longer Remain Silent -- died -- director -- the
- e2_p8 (e2): You Can No Longer Remain Silent -- died -- first
- e2_p9 (e2): You Can No Longer Remain Silent -- died
- e2_p10 (e2): You Can No Longer Remain Silent -- died -- ,
- e2_p11 (e2): You Can No Longer Remain Silent -- or
- e2_p12 (e2): You Can No Longer Remain Silent -- died -- The Goose Woman
- e2_p13 (e2): You Can No Longer Remain Silent -- The Goose Woman
- e2_p14 (e2): You Can No Longer Remain Silent -- The Goose Woman -- died -- director -- has -- film -- Which
- e2_p15 (e2): You Can No Longer Remain Silent -- The Goose Woman -- died -- director -- has -- film
- e2_p16 (e2): You Can No Longer Remain Silent -- The Goose Woman -- died -- director -- has
- e2_p17 (e2): You Can No Longer Remain Silent -- The Goose Woman -- died -- director -- has -- ?
- e2_p18 (e2): You Can No Longer Remain Silent -- The Goose Woman -- died -- director -- who
- e2_p19 (e2): You Can No Longer Remain Silent -- The Goose Woman -- died -- director
- e2_p20 (e2): You Can No Longer Remain Silent -- The Goose Woman -- died -- director -- the
- e2_p21 (e2): You Can No Longer Remain Silent -- The Goose Woman -- died -- first
- e2_p22 (e2): You Can No Longer Remain Silent -- The Goose Woman -- died
- e2_p23 (e2): You Can No Longer Remain Silent -- The Goose Woman -- died -- ,

## 7.5 Terminal Glue Path Pruning
Total raw paths: 47
Total kept paths: 28
Total pruned paths: 19
Total pruned ratio: 40.43%

### By Entity
- e1 / The Goose Woman
  - raw: 24
  - kept: 14
  - pruned: 10
  - fallback_used: False
  - examples:
    - e1_p3: The Goose Woman -> died -> director -> has [terminal=has, reason=terminal_glue_token]
    - e1_p4: The Goose Woman -> died -> director -> has -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p7: The Goose Woman -> died -> director -> the [terminal=the, reason=terminal_glue_token]
    - e1_p10: The Goose Woman -> died -> , [terminal=,, reason=terminal_glue_token]
    - e1_p15: The Goose Woman -> You Can No Longer Remain Silent -> died -> director -> has [terminal=has, reason=terminal_glue_token]
- e2 / You Can No Longer Remain Silent
  - raw: 23
  - kept: 14
  - pruned: 9
  - fallback_used: False
  - examples:
    - e2_p3: You Can No Longer Remain Silent -> died -> director -> has [terminal=has, reason=terminal_glue_token]
    - e2_p4: You Can No Longer Remain Silent -> died -> director -> has -> ? [terminal=?, reason=terminal_glue_token]
    - e2_p7: You Can No Longer Remain Silent -> died -> director -> the [terminal=the, reason=terminal_glue_token]
    - e2_p10: You Can No Longer Remain Silent -> died -> , [terminal=,, reason=terminal_glue_token]
    - e2_p11: You Can No Longer Remain Silent -> or [terminal=or, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman, reaches the director, covers the died predicate, and includes the Which cue.
- e1: e1_p2 score=85.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman, reaches the director, covers the died predicate, and includes the Which cue.
- e1: e1_p5 score=80.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman, reaches the director, covers the died predicate, and includes the who cue but misses the Which cue.
- e1: e1_p6 score=70.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman and reaches the died predicate but does not cover the director or the Which cue.
- e1: e1_p8 score=75.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman, reaches the died predicate, and includes the first cue but misses the director and the Which cue.
- e1: e1_p9 score=50.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman and reaches the died predicate but does not cover the director or the Which cue.
- e1: e1_p11 score=85.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman, reaches the died predicate, includes the other entity, and covers the Which cue.
- e1: e1_p12 score=40.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman and directly connects to the other entity without covering any relevant predicates or cues.
- e1: e1_p13 score=90.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman, connects to the other entity, reaches the died predicate, and covers the Which cue.
- e1: e1_p14 score=85.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman, connects to the other entity, reaches the died predicate, and covers the Which cue.
- e1: e1_p17 score=80.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman, connects to the other entity, reaches the died predicate, and includes the who cue but misses the Which cue.
- e1: e1_p18 score=70.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman, connects to the other entity, reaches the died predicate but does not cover the director or the Which cue.
- e1: e1_p20 score=75.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman, connects to the other entity, reaches the died predicate, and includes the first cue but misses the director and the Which cue.
- e1: e1_p21 score=50.0 valid=True terminal=film
  Reason: The path starts from The Goose Woman, connects to the other entity, reaches the died predicate but does not cover the director or the Which cue.
- e2: e2_p1 score=90.0 valid=True terminal=film
  Reason: The path starts from You Can No Longer Remain Silent, reaches the director, covers the died predicate, and includes the Which cue.
- e2: e2_p2 score=85.0 valid=True terminal=film
  Reason: The path starts from You Can No Longer Remain Silent, reaches the director, covers the died predicate, and includes the Which cue.
- e2: e2_p5 score=80.0 valid=True terminal=film
  Reason: The path starts from You Can No Longer Remain Silent, reaches the director, covers the died predicate, and includes the who cue but misses the Which cue.
- e2: e2_p6 score=70.0 valid=True terminal=film
  Reason: The path starts from You Can No Longer Remain Silent and reaches the died predicate but does not cover the director or the Which cue.
- e2: e2_p8 score=75.0 valid=True terminal=film
  Reason: The path starts from You Can No Longer Remain Silent, reaches the died predicate, and includes the first cue but misses the director and the Which cue.
- e2: e2_p9 score=50.0 valid=True terminal=film
  Reason: The path starts from You Can No Longer Remain Silent and reaches the died predicate but does not cover the director or the Which cue.
- e2: e2_p12 score=40.0 valid=True terminal=film
  Reason: The path starts from You Can No Longer Remain Silent and directly connects to the other entity without covering any relevant predicates or cues.
- e2: e2_p13 score=0.0 valid=False
  Reason: missing from LLM output
- e2: e2_p14 score=90.0 valid=True terminal=film
  Reason: The path starts from You Can No Longer Remain Silent, connects to the other entity, reaches the died predicate, and covers the Which cue.
- e2: e2_p15 score=85.0 valid=True terminal=film
  Reason: The path starts from You Can No Longer Remain Silent, connects to the other entity, reaches the died predicate, and covers the Which cue.
- e2: e2_p18 score=80.0 valid=True terminal=film
  Reason: The path starts from You Can No Longer Remain Silent, connects to the other entity, reaches the died predicate, and includes the who cue but misses the Which cue.
- e2: e2_p19 score=70.0 valid=True terminal=film
  Reason: The path starts from You Can No Longer Remain Silent, connects to the other entity, reaches the died predicate but does not cover the director or the Which cue.
- e2: e2_p21 score=50.0 valid=True terminal=film
  Reason: The path starts from You Can No Longer Remain Silent, connects to the other entity, reaches the died predicate but does not cover the director or the Which cue.
- e2: e2_p22 score=0.0 valid=False
  Reason: missing from LLM output

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p13
- e2: e2_p1, e2_p14

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1', 'e2': 'e2_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p1', 'e2': 'e2_p14'} mean_path_score=90.0
- ps3: {'e1': 'e1_p13', 'e2': 'e2_p1'} mean_path_score=90.0
- ps4: {'e1': 'e1_p13', 'e2': 'e2_p14'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Which film has the director who died first, The Goose Woman or You Can No Longer Remain Silent?
- ps1
  - e1_p1: The Goose Woman -> died -> director -> has -> film -> Which
  - e2_p1: You Can No Longer Remain Silent -> died -> director -> has -> film -> Which
- ps2
  - e1_p1: The Goose Woman -> died -> director -> has -> film -> Which
  - e2_p14: You Can No Longer Remain Silent -> The Goose Woman -> died -> director -> has -> film -> Which
- ps3
  - e1_p13: The Goose Woman -> You Can No Longer Remain Silent -> died -> director -> has -> film -> Which
  - e2_p1: You Can No Longer Remain Silent -> died -> director -> has -> film -> Which
- ps4
  - e1_p13: The Goose Woman -> You Can No Longer Remain Silent -> died -> director -> has -> film -> Which
  - e2_p14: You Can No Longer Remain Silent -> The Goose Woman -> died -> director -> has -> film -> Which

Output:
- selected_path_set_ids: ['ps1']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who is the director of The Goose Woman? depends_on=[] support=['e1_p1']
- q2: Who is the director of You Can No Longer Remain Silent? depends_on=[] support=['e2_p1']
- q3: When did the director of q1 die? depends_on=['q1'] support=['e1_p1']
- q4: When did the director of q2 die? depends_on=['q2'] support=['e2_p1']

## 10. Atomic Subquestion DAG
- None: Who is the director of The Goose Woman?
- None: Who is the director of You Can No Longer Remain Silent?
- None: When did the director of q1 die?
- None: When did the director of q2 die?
