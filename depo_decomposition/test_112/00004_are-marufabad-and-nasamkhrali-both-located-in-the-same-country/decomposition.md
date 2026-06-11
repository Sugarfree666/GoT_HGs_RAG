# DEPO Decomposition #4

- Dataset: `2wikimultihopqa`
- Question: Are Marufabad and Nasamkhrali both located in the same country?
- Gold answer: no

## 1. Semantic-Normalized Question
Are Marufabad and Nasamkhrali both located in the same country?

## 2. Explicit Entities
- Marufabad (Location) span=(4, 13)
- Nasamkhrali (Location) span=(18, 29)

## 3. Entity Masking
- LocationA -> Marufabad
- LocationB -> Nasamkhrali

Are LocationA and LocationB both located in the same country?

## 4. CoreNLP Dependency Parse
- located[6] --cop--> Are[1]
- located[6] --nsubj--> LocationA[2]
- LocationB[4] --cc--> and[3]
- LocationA[2] --conj:and--> LocationB[4]
- located[6] --nsubj--> LocationB[4]
- located[6] --cc:preconj--> both[5]
- country[10] --case--> in[7]
- country[10] --det--> the[8]
- country[10] --amod--> same[9]
- located[6] --obl:in--> country[10]
- located[6] --punct--> ?[11]

## 5. Undirected Dependency Graph
- Are[1] --cop-- located[6]
- Marufabad[2] --nsubj-- located[6]
- Marufabad[2] --conj:and-- Nasamkhrali[4]
- and[3] --cc-- Nasamkhrali[4]
- Nasamkhrali[4] --nsubj-- located[6]
- both[5] --cc:preconj-- located[6]
- located[6] --obl:in-- country[10]
- located[6] --punct-- ?[11]
- in[7] --case-- country[10]
- the[8] --det-- country[10]
- same[9] --amod-- country[10]

## 6. Entity Start Nodes from Explicit Entities
- e1: Marufabad graph_node_ids=['2']
- e2: Nasamkhrali graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Marufabad -- located -- country
- e1_p2 (e1): Marufabad -- located -- country -- in
- e1_p3 (e1): Marufabad -- located -- country -- the
- e1_p4 (e1): Marufabad -- located -- country -- same
- e1_p5 (e1): Marufabad -- located -- both
- e1_p6 (e1): Marufabad -- located
- e1_p7 (e1): Marufabad -- located -- Are
- e1_p8 (e1): Marufabad -- located -- ?
- e1_p9 (e1): Marufabad -- located -- Nasamkhrali
- e1_p10 (e1): Marufabad -- Nasamkhrali
- e1_p11 (e1): Marufabad -- Nasamkhrali -- located -- country
- e1_p12 (e1): Marufabad -- Nasamkhrali -- located -- country -- in
- e1_p13 (e1): Marufabad -- Nasamkhrali -- located -- country -- the
- e1_p14 (e1): Marufabad -- Nasamkhrali -- located -- country -- same
- e1_p15 (e1): Marufabad -- Nasamkhrali -- located -- both
- e1_p16 (e1): Marufabad -- Nasamkhrali -- located
- e1_p17 (e1): Marufabad -- Nasamkhrali -- located -- Are
- e1_p18 (e1): Marufabad -- Nasamkhrali -- located -- ?
- e1_p19 (e1): Marufabad -- located -- Nasamkhrali -- and
- e1_p20 (e1): Marufabad -- Nasamkhrali -- and
- e2_p1 (e2): Nasamkhrali -- located -- country
- e2_p2 (e2): Nasamkhrali -- located -- country -- in
- e2_p3 (e2): Nasamkhrali -- located -- country -- the
- e2_p4 (e2): Nasamkhrali -- located -- country -- same
- e2_p5 (e2): Nasamkhrali -- located -- both
- e2_p6 (e2): Nasamkhrali -- located
- e2_p7 (e2): Nasamkhrali -- located -- Are
- e2_p8 (e2): Nasamkhrali -- located -- ?
- e2_p9 (e2): Nasamkhrali -- and
- e2_p10 (e2): Nasamkhrali -- located -- Marufabad
- e2_p11 (e2): Nasamkhrali -- Marufabad
- e2_p12 (e2): Nasamkhrali -- Marufabad -- located -- country
- e2_p13 (e2): Nasamkhrali -- Marufabad -- located -- country -- in
- e2_p14 (e2): Nasamkhrali -- Marufabad -- located -- country -- the
- e2_p15 (e2): Nasamkhrali -- Marufabad -- located -- country -- same
- e2_p16 (e2): Nasamkhrali -- Marufabad -- located -- both
- e2_p17 (e2): Nasamkhrali -- Marufabad -- located
- e2_p18 (e2): Nasamkhrali -- Marufabad -- located -- Are
- e2_p19 (e2): Nasamkhrali -- Marufabad -- located -- ?

## 7.5 Terminal Glue Path Pruning
Total raw paths: 39
Total kept paths: 16
Total pruned paths: 23
Total pruned ratio: 58.97%

### By Entity
- e1 / Marufabad
  - raw: 20
  - kept: 8
  - pruned: 12
  - fallback_used: False
  - examples:
    - e1_p2: Marufabad -> located -> country -> in [terminal=in, reason=terminal_glue_token]
    - e1_p3: Marufabad -> located -> country -> the [terminal=the, reason=terminal_glue_token]
    - e1_p5: Marufabad -> located -> both [terminal=both, reason=terminal_glue_dependency_label]
    - e1_p7: Marufabad -> located -> Are [terminal=Are, reason=terminal_glue_token]
    - e1_p8: Marufabad -> located -> ? [terminal=?, reason=terminal_glue_token]
- e2 / Nasamkhrali
  - raw: 19
  - kept: 8
  - pruned: 11
  - fallback_used: False
  - examples:
    - e2_p2: Nasamkhrali -> located -> country -> in [terminal=in, reason=terminal_glue_token]
    - e2_p3: Nasamkhrali -> located -> country -> the [terminal=the, reason=terminal_glue_token]
    - e2_p5: Nasamkhrali -> located -> both [terminal=both, reason=terminal_glue_dependency_label]
    - e2_p7: Nasamkhrali -> located -> Are [terminal=Are, reason=terminal_glue_token]
    - e2_p8: Nasamkhrali -> located -> ? [terminal=?, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=country
  Reason: The path starts from Marufabad, reaches located, and covers the country, directly supporting the question's intent.
- e1: e1_p4 score=85.0 valid=True terminal=country
  Reason: The path includes the same cue, indicating a comparison, while still covering the necessary elements.
- e1: e1_p6 score=70.0 valid=True terminal=country
  Reason: The path covers the located predicate but does not reach the answer target of country.
- e1: e1_p9 score=80.0 valid=True terminal=country
  Reason: The path connects Marufabad to located and Nasamkhrali, but does not reach the country.
- e1: e1_p10 score=60.0 valid=True terminal=country
  Reason: The path connects Marufabad and Nasamkhrali but lacks the necessary predicates to support the question.
- e1: e1_p11 score=95.0 valid=True terminal=country
  Reason: The path effectively connects both entities to the located predicate and the country, fully supporting the question.
- e1: e1_p14 score=90.0 valid=True terminal=country
  Reason: The path includes the same cue and connects both entities to the located predicate and the country.
- e1: e1_p16 score=75.0 valid=True terminal=country
  Reason: The path connects both entities but does not reach the country, limiting its effectiveness.
- e2: e2_p1 score=90.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, reaches located, and covers the country, directly supporting the question's intent.
- e2: e2_p4 score=85.0 valid=True terminal=country
  Reason: The path includes the same cue, indicating a comparison, while still covering the necessary elements.
- e2: e2_p6 score=70.0 valid=True terminal=country
  Reason: The path covers the located predicate but does not reach the answer target of country.
- e2: e2_p10 score=60.0 valid=True terminal=country
  Reason: The path connects Nasamkhrali and Marufabad but lacks the necessary predicates to support the question.
- e2: e2_p11 score=60.0 valid=True terminal=country
  Reason: The path connects Nasamkhrali and Marufabad but lacks the necessary predicates to support the question.
- e2: e2_p12 score=95.0 valid=True terminal=country
  Reason: The path effectively connects both entities to the located predicate and the country, fully supporting the question.
- e2: e2_p15 score=90.0 valid=True terminal=country
  Reason: The path includes the same cue and connects both entities to the located predicate and the country.
- e2: e2_p17 score=75.0 valid=True terminal=country
  Reason: The path connects both entities but does not reach the country, limiting its effectiveness.

## 8.1 Highest-Scored Path per Entity
- e1: e1_p11 score=95.0
- e2: e2_p12 score=95.0

## 8.2 Selected Path Set
- ps1: {'e1': 'e1_p11', 'e2': 'e2_p12'} mean_path_score=95.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Are Marufabad and Nasamkhrali both located in the same country?
- ps1
  - e1_p11: Marufabad -> Nasamkhrali -> located -> country
  - e2_p12: Nasamkhrali -> Marufabad -> located -> country

Output:
- reason: The DAG compiles each semantic reasoning edge into one atomic lookup.
- q1: In which country is Marufabad located? depends_on=[] support=[]
- q2: In which country is Nasamkhrali located? depends_on=[] support=[]

## 10. Atomic Subquestion DAG
- None: In which country is Marufabad located?
- None: In which country is Nasamkhrali located?
