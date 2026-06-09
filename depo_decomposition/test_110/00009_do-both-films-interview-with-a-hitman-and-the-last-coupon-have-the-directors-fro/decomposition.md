# DEPO Decomposition #9

- Dataset: `2wikimultihopqa`
- Question: Do both films Interview With A Hitman and The Last Coupon have the directors from the same country?
- Gold answer: yes

## 1. Semantic-Normalized Question
Do both films Interview With A Hitman and The Last Coupon have the directors from the same country?

## 2. Explicit Entities
- Interview With A Hitman (Film) span=(14, 37)
- The Last Coupon (Film) span=(42, 57)

## 3. Entity Masking
- FilmA -> Interview With A Hitman
- FilmB -> The Last Coupon

Do both films FilmA and FilmB have the directors from the same country?

## 4. CoreNLP Dependency Parse
- have[7] --aux--> Do[1]
- FilmA[4] --cc:preconj--> both[2]
- FilmA[4] --compound--> films[3]
- have[7] --nsubj--> FilmA[4]
- FilmB[6] --cc--> and[5]
- FilmA[4] --conj:and--> FilmB[6]
- have[7] --nsubj--> FilmB[6]
- directors[9] --det--> the[8]
- have[7] --obj--> directors[9]
- country[13] --case--> from[10]
- country[13] --det--> the[11]
- country[13] --amod--> same[12]
- directors[9] --nmod:from--> country[13]
- have[7] --punct--> ?[14]

## 5. Undirected Dependency Graph
- Do[1] --aux-- have[7]
- both[2] --cc:preconj-- Interview With A Hitman[4]
- films[3] --compound-- Interview With A Hitman[4]
- Interview With A Hitman[4] --nsubj-- have[7]
- Interview With A Hitman[4] --conj:and-- The Last Coupon[6]
- and[5] --cc-- The Last Coupon[6]
- The Last Coupon[6] --nsubj-- have[7]
- have[7] --obj-- directors[9]
- have[7] --punct-- ?[14]
- the[8] --det-- directors[9]
- directors[9] --nmod:from-- country[13]
- from[10] --case-- country[13]
- the[11] --det-- country[13]
- same[12] --amod-- country[13]

## 6. Entity Start Nodes from Explicit Entities
- e1: Interview With A Hitman graph_node_ids=['4']
- e2: The Last Coupon graph_node_ids=['6']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Interview With A Hitman -- have -- directors -- country
- e1_p2 (e1): Interview With A Hitman -- have -- directors -- country -- from
- e1_p3 (e1): Interview With A Hitman -- have -- directors -- country -- the
- e1_p4 (e1): Interview With A Hitman -- have -- directors -- country -- same
- e1_p5 (e1): Interview With A Hitman -- have -- directors
- e1_p6 (e1): Interview With A Hitman -- have -- directors -- the
- e1_p7 (e1): Interview With A Hitman -- both
- e1_p8 (e1): Interview With A Hitman -- films
- e1_p9 (e1): Interview With A Hitman -- have
- e1_p10 (e1): Interview With A Hitman -- have -- Do
- e1_p11 (e1): Interview With A Hitman -- have -- ?
- e1_p12 (e1): Interview With A Hitman -- have -- The Last Coupon
- e1_p13 (e1): Interview With A Hitman -- The Last Coupon
- e1_p14 (e1): Interview With A Hitman -- The Last Coupon -- have -- directors -- country
- e1_p15 (e1): Interview With A Hitman -- The Last Coupon -- have -- directors -- country -- from
- e1_p16 (e1): Interview With A Hitman -- The Last Coupon -- have -- directors -- country -- the
- e1_p17 (e1): Interview With A Hitman -- The Last Coupon -- have -- directors -- country -- same
- e1_p18 (e1): Interview With A Hitman -- The Last Coupon -- have -- directors
- e1_p19 (e1): Interview With A Hitman -- The Last Coupon -- have -- directors -- the
- e1_p20 (e1): Interview With A Hitman -- The Last Coupon -- have
- e1_p21 (e1): Interview With A Hitman -- The Last Coupon -- have -- Do
- e1_p22 (e1): Interview With A Hitman -- The Last Coupon -- have -- ?
- e1_p23 (e1): Interview With A Hitman -- have -- The Last Coupon -- and
- e1_p24 (e1): Interview With A Hitman -- The Last Coupon -- and
- e2_p1 (e2): The Last Coupon -- have -- directors -- country
- e2_p2 (e2): The Last Coupon -- have -- directors -- country -- from
- e2_p3 (e2): The Last Coupon -- have -- directors -- country -- the
- e2_p4 (e2): The Last Coupon -- have -- directors -- country -- same
- e2_p5 (e2): The Last Coupon -- have -- directors
- e2_p6 (e2): The Last Coupon -- have -- directors -- the
- e2_p7 (e2): The Last Coupon -- have
- e2_p8 (e2): The Last Coupon -- have -- Do
- e2_p9 (e2): The Last Coupon -- have -- ?
- e2_p10 (e2): The Last Coupon -- and
- e2_p11 (e2): The Last Coupon -- have -- Interview With A Hitman
- e2_p12 (e2): The Last Coupon -- Interview With A Hitman
- e2_p13 (e2): The Last Coupon -- Interview With A Hitman -- have -- directors -- country
- e2_p14 (e2): The Last Coupon -- Interview With A Hitman -- have -- directors -- country -- from
- e2_p15 (e2): The Last Coupon -- Interview With A Hitman -- have -- directors -- country -- the
- e2_p16 (e2): The Last Coupon -- Interview With A Hitman -- have -- directors -- country -- same
- e2_p17 (e2): The Last Coupon -- Interview With A Hitman -- have -- directors
- e2_p18 (e2): The Last Coupon -- have -- Interview With A Hitman -- both
- e2_p19 (e2): The Last Coupon -- have -- Interview With A Hitman -- films
- e2_p20 (e2): The Last Coupon -- Interview With A Hitman -- have -- directors -- the
- e2_p21 (e2): The Last Coupon -- Interview With A Hitman -- both
- e2_p22 (e2): The Last Coupon -- Interview With A Hitman -- films
- e2_p23 (e2): The Last Coupon -- Interview With A Hitman -- have
- e2_p24 (e2): The Last Coupon -- Interview With A Hitman -- have -- Do
- e2_p25 (e2): The Last Coupon -- Interview With A Hitman -- have -- ?

## 7.5 Terminal Glue Path Pruning
Total raw paths: 49
Total kept paths: 19
Total pruned paths: 30
Total pruned ratio: 61.22%

### By Entity
- e1 / Interview With A Hitman
  - raw: 24
  - kept: 9
  - pruned: 15
  - fallback_used: False
  - examples:
    - e1_p2: Interview With A Hitman -> have -> directors -> country -> from [terminal=from, reason=terminal_glue_token]
    - e1_p3: Interview With A Hitman -> have -> directors -> country -> the [terminal=the, reason=terminal_glue_token]
    - e1_p6: Interview With A Hitman -> have -> directors -> the [terminal=the, reason=terminal_glue_token]
    - e1_p7: Interview With A Hitman -> both [terminal=both, reason=terminal_glue_dependency_label]
    - e1_p9: Interview With A Hitman -> have [terminal=have, reason=terminal_glue_token]
- e2 / The Last Coupon
  - raw: 25
  - kept: 10
  - pruned: 15
  - fallback_used: False
  - examples:
    - e2_p2: The Last Coupon -> have -> directors -> country -> from [terminal=from, reason=terminal_glue_token]
    - e2_p3: The Last Coupon -> have -> directors -> country -> the [terminal=the, reason=terminal_glue_token]
    - e2_p6: The Last Coupon -> have -> directors -> the [terminal=the, reason=terminal_glue_token]
    - e2_p7: The Last Coupon -> have [terminal=have, reason=terminal_glue_token]
    - e2_p8: The Last Coupon -> have -> Do [terminal=Do, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman, reaches directors, and covers the country aspect, directly supporting the question's intent.
- e1: e1_p4 score=95.0 valid=True terminal=country
  Reason: The path effectively covers the directors and country, while also including the 'same' cue, which is crucial for the question's intent.
- e1: e1_p5 score=75.0 valid=True terminal=directors
  Reason: The path covers the directors but misses the country aspect, which is essential for answering the question.
- e1: e1_p8 score=30.0 valid=False terminal=none
  Reason: The path only connects to 'films' and does not address the necessary elements of directors or country.
- e1: e1_p12 score=70.0 valid=True terminal=directors
  Reason: The path connects Interview With A Hitman to The Last Coupon and to directors, but it lacks the country aspect.
- e1: e1_p13 score=30.0 valid=False terminal=none
  Reason: The path only connects the two films without addressing the necessary elements of directors or country.
- e1: e1_p14 score=90.0 valid=True terminal=country
  Reason: The path effectively connects both films to their directors and the country, supporting the question's intent.
- e1: e1_p17 score=95.0 valid=True terminal=country
  Reason: The path connects both films to their directors and includes the 'same' cue, which is crucial for the question's intent.
- e1: e1_p18 score=70.0 valid=True terminal=directors
  Reason: The path connects both films to their directors but misses the country aspect, which is essential for answering the question.
- e2: e2_p1 score=90.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, reaches directors, and covers the country aspect, directly supporting the question's intent.
- e2: e2_p4 score=95.0 valid=True terminal=country
  Reason: The path effectively covers the directors and country, while also including the 'same' cue, which is crucial for the question's intent.
- e2: e2_p5 score=75.0 valid=True terminal=directors
  Reason: The path covers the directors but misses the country aspect, which is essential for answering the question.
- e2: e2_p11 score=70.0 valid=True terminal=directors
  Reason: The path connects The Last Coupon to Interview With A Hitman but lacks the country aspect.
- e2: e2_p12 score=30.0 valid=False terminal=none
  Reason: The path only connects the two films without addressing the necessary elements of directors or country.
- e2: e2_p13 score=90.0 valid=True terminal=country
  Reason: The path effectively connects both films to their directors and the country, supporting the question's intent.
- e2: e2_p16 score=95.0 valid=True terminal=country
  Reason: The path connects both films to their directors and includes the 'same' cue, which is crucial for the question's intent.
- e2: e2_p17 score=70.0 valid=True terminal=directors
  Reason: The path connects both films to their directors but misses the country aspect, which is essential for answering the question.
- e2: e2_p19 score=70.0 valid=True terminal=directors
  Reason: The path connects The Last Coupon to Interview With A Hitman but lacks the country aspect.
- e2: e2_p22 score=30.0 valid=False terminal=none
  Reason: The path only connects the two films without addressing the necessary elements of directors or country.

## 8.1 Top-2 Paths per Entity
- e1: e1_p17, e1_p4
- e2: e2_p16, e2_p4

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p17', 'e2': 'e2_p16'} mean_path_score=95.0
- ps2: {'e1': 'e1_p17', 'e2': 'e2_p4'} mean_path_score=95.0
- ps3: {'e1': 'e1_p4', 'e2': 'e2_p16'} mean_path_score=95.0
- ps4: {'e1': 'e1_p4', 'e2': 'e2_p4'} mean_path_score=95.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Do both films Interview With A Hitman and The Last Coupon have the directors from the same country?
- ps1
  - e1_p17: Interview With A Hitman -> The Last Coupon -> have -> directors -> country -> same
  - e2_p16: The Last Coupon -> Interview With A Hitman -> have -> directors -> country -> same
- ps2
  - e1_p17: Interview With A Hitman -> The Last Coupon -> have -> directors -> country -> same
  - e2_p4: The Last Coupon -> have -> directors -> country -> same
- ps3
  - e1_p4: Interview With A Hitman -> have -> directors -> country -> same
  - e2_p16: The Last Coupon -> Interview With A Hitman -> have -> directors -> country -> same
- ps4
  - e1_p4: Interview With A Hitman -> have -> directors -> country -> same
  - e2_p4: The Last Coupon -> have -> directors -> country -> same

Output:
- selected_path_set_ids: ['ps1', 'ps2', 'ps3', 'ps4']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who is the director of Interview With A Hitman? depends_on=[] support=['e1_p17']
- q2: Who is the director of The Last Coupon? depends_on=[] support=['e2_p16']
- q3: What country is the director of Interview With A Hitman from? depends_on=['q1'] support=['e1_p4']
- q4: What country is the director of The Last Coupon from? depends_on=['q2'] support=['e2_p4']

## 10. Atomic Subquestion DAG
- None: Who is the director of Interview With A Hitman?
- None: Who is the director of The Last Coupon?
- None: What country is the director of Interview With A Hitman from?
- None: What country is the director of The Last Coupon from?
