# DEPO Decomposition #12

- Dataset: `2wikimultihopqa`
- Question: Are Vasilyevsky Island and Preobrazheniya Island located in the same country?
- Gold answer: yes

## 1. Semantic-Normalized Question
Are Vasilyevsky Island and Preobrazheniya Island located in the same country?

## 2. Explicit Entities
- Vasilyevsky Island (Location) span=(4, 22)
- Preobrazheniya Island (Location) span=(27, 48)

## 3. Entity Masking
- LocationA -> Vasilyevsky Island
- LocationB -> Preobrazheniya Island

Are LocationA and LocationB located in the same country?

## 4. CoreNLP Dependency Parse
- located[5] --cop--> Are[1]
- located[5] --nsubj--> LocationA[2]
- LocationB[4] --cc--> and[3]
- LocationA[2] --conj:and--> LocationB[4]
- located[5] --nsubj--> LocationB[4]
- country[9] --case--> in[6]
- country[9] --det--> the[7]
- country[9] --amod--> same[8]
- located[5] --obl:in--> country[9]
- located[5] --punct--> ?[10]

## 5. Undirected Dependency Graph
- Are[1] --cop-- located[5]
- Vasilyevsky Island[2] --nsubj-- located[5]
- Vasilyevsky Island[2] --conj:and-- Preobrazheniya Island[4]
- and[3] --cc-- Preobrazheniya Island[4]
- Preobrazheniya Island[4] --nsubj-- located[5]
- located[5] --obl:in-- country[9]
- located[5] --punct-- ?[10]
- in[6] --case-- country[9]
- the[7] --det-- country[9]
- same[8] --amod-- country[9]

## 6. Entity Start Nodes from Explicit Entities
- e1: Vasilyevsky Island graph_node_ids=['2']
- e2: Preobrazheniya Island graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Vasilyevsky Island -- located -- country
- e1_p2 (e1): Vasilyevsky Island -- located -- country -- in
- e1_p3 (e1): Vasilyevsky Island -- located -- country -- the
- e1_p4 (e1): Vasilyevsky Island -- located -- country -- same
- e1_p5 (e1): Vasilyevsky Island -- located
- e1_p6 (e1): Vasilyevsky Island -- located -- Are
- e1_p7 (e1): Vasilyevsky Island -- located -- ?
- e1_p8 (e1): Vasilyevsky Island -- located -- Preobrazheniya Island
- e1_p9 (e1): Vasilyevsky Island -- Preobrazheniya Island
- e1_p10 (e1): Vasilyevsky Island -- Preobrazheniya Island -- located -- country
- e1_p11 (e1): Vasilyevsky Island -- Preobrazheniya Island -- located -- country -- in
- e1_p12 (e1): Vasilyevsky Island -- Preobrazheniya Island -- located -- country -- the
- e1_p13 (e1): Vasilyevsky Island -- Preobrazheniya Island -- located -- country -- same
- e1_p14 (e1): Vasilyevsky Island -- Preobrazheniya Island -- located
- e1_p15 (e1): Vasilyevsky Island -- Preobrazheniya Island -- located -- Are
- e1_p16 (e1): Vasilyevsky Island -- Preobrazheniya Island -- located -- ?
- e1_p17 (e1): Vasilyevsky Island -- located -- Preobrazheniya Island -- and
- e1_p18 (e1): Vasilyevsky Island -- Preobrazheniya Island -- and
- e2_p1 (e2): Preobrazheniya Island -- located -- country
- e2_p2 (e2): Preobrazheniya Island -- located -- country -- in
- e2_p3 (e2): Preobrazheniya Island -- located -- country -- the
- e2_p4 (e2): Preobrazheniya Island -- located -- country -- same
- e2_p5 (e2): Preobrazheniya Island -- located
- e2_p6 (e2): Preobrazheniya Island -- located -- Are
- e2_p7 (e2): Preobrazheniya Island -- located -- ?
- e2_p8 (e2): Preobrazheniya Island -- and
- e2_p9 (e2): Preobrazheniya Island -- located -- Vasilyevsky Island
- e2_p10 (e2): Preobrazheniya Island -- Vasilyevsky Island
- e2_p11 (e2): Preobrazheniya Island -- Vasilyevsky Island -- located -- country
- e2_p12 (e2): Preobrazheniya Island -- Vasilyevsky Island -- located -- country -- in
- e2_p13 (e2): Preobrazheniya Island -- Vasilyevsky Island -- located -- country -- the
- e2_p14 (e2): Preobrazheniya Island -- Vasilyevsky Island -- located -- country -- same
- e2_p15 (e2): Preobrazheniya Island -- Vasilyevsky Island -- located
- e2_p16 (e2): Preobrazheniya Island -- Vasilyevsky Island -- located -- Are
- e2_p17 (e2): Preobrazheniya Island -- Vasilyevsky Island -- located -- ?

## 7.5 Terminal Glue Path Pruning
Total raw paths: 35
Total kept paths: 16
Total pruned paths: 19
Total pruned ratio: 54.29%

### By Entity
- e1 / Vasilyevsky Island
  - raw: 18
  - kept: 8
  - pruned: 10
  - fallback_used: False
  - examples:
    - e1_p2: Vasilyevsky Island -> located -> country -> in [terminal=in, reason=terminal_glue_token]
    - e1_p3: Vasilyevsky Island -> located -> country -> the [terminal=the, reason=terminal_glue_token]
    - e1_p6: Vasilyevsky Island -> located -> Are [terminal=Are, reason=terminal_glue_token]
    - e1_p7: Vasilyevsky Island -> located -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p11: Vasilyevsky Island -> Preobrazheniya Island -> located -> country -> in [terminal=in, reason=terminal_glue_token]
- e2 / Preobrazheniya Island
  - raw: 17
  - kept: 8
  - pruned: 9
  - fallback_used: False
  - examples:
    - e2_p2: Preobrazheniya Island -> located -> country -> in [terminal=in, reason=terminal_glue_token]
    - e2_p3: Preobrazheniya Island -> located -> country -> the [terminal=the, reason=terminal_glue_token]
    - e2_p6: Preobrazheniya Island -> located -> Are [terminal=Are, reason=terminal_glue_token]
    - e2_p7: Preobrazheniya Island -> located -> ? [terminal=?, reason=terminal_glue_token]
    - e2_p8: Preobrazheniya Island -> and [terminal=and, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=country
  Reason: The path starts from Vasilyevsky Island, reaches the predicate 'located', and connects to 'country', effectively supporting the question's intent.
- e1: e1_p4 score=95.0 valid=True terminal=country
  Reason: The path effectively connects Vasilyevsky Island to 'located', 'country', and 'same', fully supporting the question's intent.
- e1: e1_p5 score=55.0 valid=True terminal=country
  Reason: The path only connects Vasilyevsky Island to 'located', missing the necessary connection to 'country'.
- e1: e1_p8 score=75.0 valid=True terminal=country
  Reason: The path connects Vasilyevsky Island to 'located' and then to Preobrazheniya Island, but it does not reach 'country'.
- e1: e1_p9 score=30.0 valid=True terminal=country
  Reason: The path only connects Vasilyevsky Island to Preobrazheniya Island, missing the necessary predicates and connections.
- e1: e1_p10 score=90.0 valid=True terminal=country
  Reason: The path connects both islands through 'located' to 'country', effectively supporting the question's intent.
- e1: e1_p13 score=95.0 valid=True terminal=country
  Reason: The path effectively connects both islands to 'located', 'country', and 'same', fully supporting the question's intent.
- e1: e1_p14 score=75.0 valid=True terminal=country
  Reason: The path connects both islands but does not reach 'country', missing a key element of the question.
- e2: e2_p1 score=90.0 valid=True terminal=country
  Reason: The path starts from Preobrazheniya Island, reaches 'located', and connects to 'country', effectively supporting the question's intent.
- e2: e2_p4 score=95.0 valid=True terminal=country
  Reason: The path effectively connects Preobrazheniya Island to 'located', 'country', and 'same', fully supporting the question's intent.
- e2: e2_p5 score=55.0 valid=True terminal=country
  Reason: The path only connects Preobrazheniya Island to 'located', missing the necessary connection to 'country'.
- e2: e2_p9 score=75.0 valid=True terminal=country
  Reason: The path connects Preobrazheniya Island to 'located' and then to Vasilyevsky Island, but it does not reach 'country'.
- e2: e2_p10 score=30.0 valid=True terminal=country
  Reason: The path only connects Preobrazheniya Island to Vasilyevsky Island, missing the necessary predicates and connections.
- e2: e2_p11 score=90.0 valid=True terminal=country
  Reason: The path connects both islands through 'located' to 'country', effectively supporting the question's intent.
- e2: e2_p14 score=95.0 valid=True terminal=country
  Reason: The path effectively connects both islands to 'located', 'country', and 'same', fully supporting the question's intent.
- e2: e2_p15 score=75.0 valid=True terminal=country
  Reason: The path connects both islands but does not reach 'country', missing a key element of the question.

## 8.1 Highest-Scored Path per Entity
- e1: e1_p13 score=95.0
- e2: e2_p14 score=95.0

## 8.2 Selected Path Set
- ps1: {'e1': 'e1_p13', 'e2': 'e2_p14'} mean_path_score=95.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Are Vasilyevsky Island and Preobrazheniya Island located in the same country?
- ps1
  - e1_p13: Vasilyevsky Island -> Preobrazheniya Island -> located -> country -> same
  - e2_p14: Preobrazheniya Island -> Vasilyevsky Island -> located -> country -> same

Output:
- reason: The DAG compiles each semantic reasoning edge into one atomic lookup.
- q1: In which country is Vasilyevsky Island located? depends_on=[] support=[]
- q2: In which country is Preobrazheniya Island located? depends_on=[] support=[]

## 10. Atomic Subquestion DAG
- None: In which country is Vasilyevsky Island located?
- None: In which country is Preobrazheniya Island located?
