# DEPO Decomposition #15

- Dataset: `hotpotqa`
- Question: Which 2009 animated film is from Japan, Summer Wars or The Secret of Kells?
- Gold answer: Summer Wars

## 1. Semantic-Normalized Question
Which 2009 animated film is from Japan, Summer Wars or The Secret of Kells?

## 2. Explicit Entities
- Summer Wars (Film) span=(40, 51)
- The Secret of Kells (Film) span=(55, 74)

## 3. Entity Masking
- FilmA -> Summer Wars
- FilmB -> The Secret of Kells

Which 2009 animated film is from Japan, FilmA or FilmB?

## 4. CoreNLP Dependency Parse
- film[4] --det--> Which[1]
- film[4] --nummod--> 2009[2]
- film[4] --amod--> animated[3]
- Japan[7] --nsubj--> film[4]
- Japan[7] --cop--> is[5]
- Japan[7] --case--> from[6]
- Japan[7] --punct--> ,[8]
- Japan[7] --conj:or--> FilmA[9]
- FilmB[11] --cc--> or[10]
- Japan[7] --conj:or--> FilmB[11]
- Japan[7] --punct--> ?[12]

## 5. Undirected Dependency Graph
- Which[1] --det-- film[4]
- 2009[2] --nummod-- film[4]
- animated[3] --amod-- film[4]
- film[4] --nsubj-- Japan[7]
- is[5] --cop-- Japan[7]
- from[6] --case-- Japan[7]
- Japan[7] --punct-- ,[8]
- Japan[7] --conj:or-- Summer Wars[9]
- Japan[7] --conj:or-- The Secret of Kells[11]
- Japan[7] --punct-- ?[12]
- or[10] --cc-- The Secret of Kells[11]

## 6. Entity Start Nodes from Explicit Entities
- e1: Summer Wars graph_node_ids=['9']
- e2: The Secret of Kells graph_node_ids=['11']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Summer Wars -- Japan -- film -- 2009
- e1_p2 (e1): Summer Wars -- Japan -- film -- animated
- e1_p3 (e1): Summer Wars -- Japan -- film -- Which
- e1_p4 (e1): Summer Wars -- Japan -- film
- e1_p5 (e1): Summer Wars -- Japan
- e1_p6 (e1): Summer Wars -- Japan -- is
- e1_p7 (e1): Summer Wars -- Japan -- from
- e1_p8 (e1): Summer Wars -- Japan -- ,
- e1_p9 (e1): Summer Wars -- Japan -- ?
- e1_p10 (e1): Summer Wars -- Japan -- The Secret of Kells
- e1_p11 (e1): Summer Wars -- Japan -- The Secret of Kells -- or
- e2_p1 (e2): The Secret of Kells -- Japan -- film -- 2009
- e2_p2 (e2): The Secret of Kells -- Japan -- film -- animated
- e2_p3 (e2): The Secret of Kells -- Japan -- film -- Which
- e2_p4 (e2): The Secret of Kells -- Japan -- film
- e2_p5 (e2): The Secret of Kells -- Japan
- e2_p6 (e2): The Secret of Kells -- Japan -- is
- e2_p7 (e2): The Secret of Kells -- Japan -- from
- e2_p8 (e2): The Secret of Kells -- Japan -- ,
- e2_p9 (e2): The Secret of Kells -- Japan -- ?
- e2_p10 (e2): The Secret of Kells -- or
- e2_p11 (e2): The Secret of Kells -- Japan -- Summer Wars

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=film
  Reason: The path starts from Summer Wars, reaches Japan, and includes the film and year, effectively supporting the question intent.
- e1: e1_p2 score=90.0 valid=True terminal=film
  Reason: The path starts from Summer Wars, reaches Japan, and includes the film type, effectively supporting the question intent.
- e1: e1_p3 score=75.0 valid=True terminal=film
  Reason: The path starts from Summer Wars and includes the question cue 'which', but lacks a direct connection to the year.
- e1: e1_p4 score=70.0 valid=True terminal=film
  Reason: The path starts from Summer Wars and reaches Japan and film, but it does not include the year or the question cue.
- e1: e1_p5 score=30.0 valid=True terminal=film
  Reason: The path only connects Summer Wars and Japan, lacking necessary information to support the question.
- e1: e1_p6 score=40.0 valid=True terminal=film
  Reason: The path includes 'is' but does not provide sufficient information to support the question intent.
- e1: e1_p7 score=30.0 valid=True terminal=film
  Reason: The path only connects Summer Wars and Japan, lacking necessary information to support the question.
- e1: e1_p8 score=0.0 valid=False terminal=film
  Reason: The path ends with punctuation, providing no useful information.
- e1: e1_p9 score=0.0 valid=False terminal=film
  Reason: The path ends with punctuation, providing no useful information.
- e1: e1_p10 score=60.0 valid=True terminal=film
  Reason: The path connects Summer Wars, Japan, and The Secret of Kells, but lacks the year and the question cue.
- e1: e1_p11 score=50.0 valid=True terminal=film
  Reason: The path connects Summer Wars, Japan, and The Secret of Kells, but does not effectively support the question intent.
- e2: e2_p1 score=90.0 valid=True terminal=film
  Reason: The path starts from The Secret of Kells, reaches Japan, and includes the film and year, effectively supporting the question intent.
- e2: e2_p2 score=90.0 valid=True terminal=film
  Reason: The path starts from The Secret of Kells, reaches Japan, and includes the film type, effectively supporting the question intent.
- e2: e2_p3 score=75.0 valid=True terminal=film
  Reason: The path starts from The Secret of Kells and includes the question cue 'which', but lacks a direct connection to the year.
- e2: e2_p4 score=70.0 valid=True terminal=film
  Reason: The path starts from The Secret of Kells and reaches Japan and film, but it does not include the year or the question cue.
- e2: e2_p5 score=30.0 valid=True terminal=film
  Reason: The path only connects The Secret of Kells and Japan, lacking necessary information to support the question.
- e2: e2_p6 score=40.0 valid=True terminal=film
  Reason: The path includes 'is' but does not provide sufficient information to support the question intent.
- e2: e2_p7 score=30.0 valid=True terminal=film
  Reason: The path only connects The Secret of Kells and Japan, lacking necessary information to support the question.
- e2: e2_p8 score=0.0 valid=False terminal=film
  Reason: The path ends with punctuation, providing no useful information.
- e2: e2_p9 score=0.0 valid=False terminal=film
  Reason: The path ends with punctuation, providing no useful information.
- e2: e2_p10 score=60.0 valid=True terminal=film
  Reason: The path connects The Secret of Kells and the conjunction 'or', but lacks the year and the question cue.
- e2: e2_p11 score=50.0 valid=True terminal=film
  Reason: The path connects The Secret of Kells, Japan, and Summer Wars, but does not effectively support the question intent.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2
- e2: e2_p1, e2_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1', 'e2': 'e2_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p1', 'e2': 'e2_p2'} mean_path_score=90.0
- ps3: {'e1': 'e1_p2', 'e2': 'e2_p1'} mean_path_score=90.0
- ps4: {'e1': 'e1_p2', 'e2': 'e2_p2'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Which 2009 animated film is from Japan, Summer Wars or The Secret of Kells?
- ps1
  - e1_p1: Summer Wars -> Japan -> film -> 2009
  - e2_p1: The Secret of Kells -> Japan -> film -> 2009
- ps2
  - e1_p1: Summer Wars -> Japan -> film -> 2009
  - e2_p2: The Secret of Kells -> Japan -> film -> animated
- ps3
  - e1_p2: Summer Wars -> Japan -> film -> animated
  - e2_p1: The Secret of Kells -> Japan -> film -> 2009
- ps4
  - e1_p2: Summer Wars -> Japan -> film -> animated
  - e2_p2: The Secret of Kells -> Japan -> film -> animated

Output:
- selected_path_set_ids: ['ps1']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Is Summer Wars from Japan? depends_on=[] support=['e1_p1']
- q2: Is The Secret of Kells from Japan? depends_on=[] support=['e2_p1']
- q3: Was Summer Wars released in 2009? depends_on=[] support=['e1_p1']
- q4: Was The Secret of Kells released in 2009? depends_on=[] support=['e2_p1']

## 10. Atomic Subquestion DAG
- None: Is Summer Wars from Japan?
- None: Is The Secret of Kells from Japan?
- None: Was Summer Wars released in 2009?
- None: Was The Secret of Kells released in 2009?
