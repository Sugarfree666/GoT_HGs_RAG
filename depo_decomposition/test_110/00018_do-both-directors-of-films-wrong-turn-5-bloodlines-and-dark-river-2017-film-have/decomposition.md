# DEPO Decomposition #18

- Dataset: `2wikimultihopqa`
- Question: Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?
- Gold answer: no

## 1. Semantic-Normalized Question
Do both directors of the films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?

## 2. Explicit Entities
- Wrong Turn (Entity) span=(31, 41)
- Dark River (2017 Film) (Film) span=(60, 82)

## 3. Entity Masking
- EntityA -> Wrong Turn
- FilmA -> Dark River (2017 Film)

Do both directors of the films EntityA 5: Bloodlines and FilmA have the same nationality?

## 4. CoreNLP Dependency Parse
- directors[3] --det--> both[2]
- Do[1] --obj--> directors[3]
- EntityA[7] --case--> of[4]
- EntityA[7] --det--> the[5]
- EntityA[7] --compound--> films[6]
- directors[3] --nmod:of--> EntityA[7]
- EntityA[7] --nummod--> 5[8]
- directors[3] --punct--> :[9]
- have[13] --nsubj--> Bloodlines[10]
- FilmA[12] --cc--> and[11]
- Bloodlines[10] --conj:and--> FilmA[12]
- have[13] --nsubj--> FilmA[12]
- directors[3] --dep--> have[13]
- nationality[16] --det--> the[14]
- nationality[16] --amod--> same[15]
- have[13] --obj--> nationality[16]
- Do[1] --punct--> ?[17]

## 5. Undirected Dependency Graph
- Do[1] --obj-- directors[3]
- Do[1] --punct-- ?[17]
- both[2] --det-- directors[3]
- directors[3] --nmod:of-- Wrong Turn[7]
- directors[3] --punct-- :[9]
- directors[3] --dep-- have[13]
- of[4] --case-- Wrong Turn[7]
- the[5] --det-- Wrong Turn[7]
- films[6] --compound-- Wrong Turn[7]
- Wrong Turn[7] --nummod-- 5[8]
- Bloodlines[10] --nsubj-- have[13]
- Bloodlines[10] --conj:and-- Dark River (2017 Film)[12]
- and[11] --cc-- Dark River (2017 Film)[12]
- Dark River (2017 Film)[12] --nsubj-- have[13]
- have[13] --obj-- nationality[16]
- the[14] --det-- nationality[16]
- same[15] --amod-- nationality[16]

## 6. Entity Start Nodes from Explicit Entities
- e1: Wrong Turn graph_node_ids=['7']
- e2: Dark River (2017 Film) graph_node_ids=['12']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Wrong Turn -- directors -- have -- nationality
- e1_p2 (e1): Wrong Turn -- directors -- have -- nationality -- the
- e1_p3 (e1): Wrong Turn -- directors -- have -- nationality -- same
- e1_p4 (e1): Wrong Turn -- directors -- have -- Bloodlines
- e1_p5 (e1): Wrong Turn -- directors -- both
- e1_p6 (e1): Wrong Turn -- directors -- have
- e1_p7 (e1): Wrong Turn -- directors
- e1_p8 (e1): Wrong Turn -- films
- e1_p9 (e1): Wrong Turn -- 5
- e1_p10 (e1): Wrong Turn -- directors -- Do
- e1_p11 (e1): Wrong Turn -- directors -- :
- e1_p12 (e1): Wrong Turn -- directors -- Do -- ?
- e1_p13 (e1): Wrong Turn -- of
- e1_p14 (e1): Wrong Turn -- the
- e1_p15 (e1): Wrong Turn -- directors -- have -- Bloodlines -- Dark River (2017 Film)
- e1_p16 (e1): Wrong Turn -- directors -- have -- Dark River (2017 Film)
- e1_p17 (e1): Wrong Turn -- directors -- have -- Dark River (2017 Film) -- Bloodlines
- e1_p18 (e1): Wrong Turn -- directors -- have -- Bloodlines -- Dark River (2017 Film) -- and
- e1_p19 (e1): Wrong Turn -- directors -- have -- Dark River (2017 Film) -- and
- e2_p1 (e2): Dark River (2017 Film) -- Bloodlines -- have -- directors -- both
- e2_p2 (e2): Dark River (2017 Film) -- Bloodlines -- have -- nationality
- e2_p3 (e2): Dark River (2017 Film) -- Bloodlines -- have -- nationality -- the
- e2_p4 (e2): Dark River (2017 Film) -- Bloodlines -- have -- nationality -- same
- e2_p5 (e2): Dark River (2017 Film) -- Bloodlines -- have -- directors
- e2_p6 (e2): Dark River (2017 Film) -- have -- directors -- both
- e2_p7 (e2): Dark River (2017 Film) -- Bloodlines -- have -- directors -- Do
- e2_p8 (e2): Dark River (2017 Film) -- Bloodlines -- have -- directors -- :
- e2_p9 (e2): Dark River (2017 Film) -- Bloodlines -- have -- directors -- Do -- ?
- e2_p10 (e2): Dark River (2017 Film) -- have -- nationality
- e2_p11 (e2): Dark River (2017 Film) -- have -- nationality -- the
- e2_p12 (e2): Dark River (2017 Film) -- have -- nationality -- same
- e2_p13 (e2): Dark River (2017 Film) -- Bloodlines -- have
- e2_p14 (e2): Dark River (2017 Film) -- have -- directors
- e2_p15 (e2): Dark River (2017 Film) -- have -- Bloodlines
- e2_p16 (e2): Dark River (2017 Film) -- have -- directors -- Do
- e2_p17 (e2): Dark River (2017 Film) -- have -- directors -- :
- e2_p18 (e2): Dark River (2017 Film) -- have -- directors -- Do -- ?
- e2_p19 (e2): Dark River (2017 Film) -- Bloodlines
- e2_p20 (e2): Dark River (2017 Film) -- have
- e2_p21 (e2): Dark River (2017 Film) -- and
- e2_p22 (e2): Dark River (2017 Film) -- Bloodlines -- have -- directors -- Wrong Turn
- e2_p23 (e2): Dark River (2017 Film) -- have -- directors -- Wrong Turn
- e2_p24 (e2): Dark River (2017 Film) -- Bloodlines -- have -- directors -- Wrong Turn -- films
- e2_p25 (e2): Dark River (2017 Film) -- Bloodlines -- have -- directors -- Wrong Turn -- 5
- e2_p26 (e2): Dark River (2017 Film) -- have -- directors -- Wrong Turn -- films
- e2_p27 (e2): Dark River (2017 Film) -- have -- directors -- Wrong Turn -- 5
- e2_p28 (e2): Dark River (2017 Film) -- Bloodlines -- have -- directors -- Wrong Turn -- of
- e2_p29 (e2): Dark River (2017 Film) -- Bloodlines -- have -- directors -- Wrong Turn -- the
- e2_p30 (e2): Dark River (2017 Film) -- have -- directors -- Wrong Turn -- of
- e2_p31 (e2): Dark River (2017 Film) -- have -- directors -- Wrong Turn -- the

## 7.5 Terminal Glue Path Pruning
Total raw paths: 50
Total kept paths: 23
Total pruned paths: 27
Total pruned ratio: 54.00%

### By Entity
- e1 / Wrong Turn
  - raw: 19
  - kept: 9
  - pruned: 10
  - fallback_used: False
  - examples:
    - e1_p2: Wrong Turn -> directors -> have -> nationality -> the [terminal=the, reason=terminal_glue_token]
    - e1_p5: Wrong Turn -> directors -> both [terminal=both, reason=terminal_glue_dependency_label]
    - e1_p6: Wrong Turn -> directors -> have [terminal=have, reason=terminal_glue_token]
    - e1_p10: Wrong Turn -> directors -> Do [terminal=Do, reason=terminal_glue_token]
    - e1_p11: Wrong Turn -> directors -> : [terminal=:, reason=terminal_glue_token]
- e2 / Dark River (2017 Film)
  - raw: 31
  - kept: 14
  - pruned: 17
  - fallback_used: False
  - examples:
    - e2_p1: Dark River (2017 Film) -> Bloodlines -> have -> directors -> both [terminal=both, reason=terminal_glue_dependency_label]
    - e2_p3: Dark River (2017 Film) -> Bloodlines -> have -> nationality -> the [terminal=the, reason=terminal_glue_token]
    - e2_p6: Dark River (2017 Film) -> have -> directors -> both [terminal=both, reason=terminal_glue_dependency_label]
    - e2_p7: Dark River (2017 Film) -> Bloodlines -> have -> directors -> Do [terminal=Do, reason=terminal_glue_token]
    - e2_p8: Dark River (2017 Film) -> Bloodlines -> have -> directors -> : [terminal=:, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn, reaches directors, and covers the nationality aspect directly.
- e1: e1_p3 score=95.0 valid=True terminal=nationality
  Reason: The path effectively covers the same nationality aspect, starting from Wrong Turn and reaching the necessary predicates.
- e1: e1_p4 score=70.0 valid=True terminal=nationality
  Reason: The path reaches Bloodlines but does not cover the nationality aspect, making it less effective.
- e1: e1_p7 score=30.0 valid=True terminal=directors
  Reason: The path stops too early at directors without reaching the necessary predicates.
- e1: e1_p8 score=30.0 valid=True terminal=films
  Reason: The path only connects to films and does not address the nationality aspect.
- e1: e1_p9 score=30.0 valid=True terminal=5
  Reason: The path only connects to the number 5 and does not address the nationality aspect.
- e1: e1_p15 score=95.0 valid=True terminal=nationality
  Reason: The path effectively connects both films and covers the nationality aspect.
- e1: e1_p16 score=90.0 valid=True terminal=nationality
  Reason: The path connects to Dark River (2017 Film) and covers the nationality aspect.
- e1: e1_p17 score=85.0 valid=True terminal=nationality
  Reason: The path connects both films and covers the nationality aspect, but passes through another entity.
- e2: e2_p2 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, and covers the nationality aspect directly.
- e2: e2_p4 score=95.0 valid=True terminal=nationality
  Reason: The path effectively covers the same nationality aspect, starting from Dark River (2017 Film) and reaching the necessary predicates.
- e2: e2_p5 score=70.0 valid=True terminal=directors
  Reason: The path reaches Bloodlines but does not cover the nationality aspect, making it less effective.
- e2: e2_p10 score=60.0 valid=True terminal=nationality
  Reason: The path connects to Dark River (2017 Film) and has a weak connection to nationality.
- e2: e2_p12 score=90.0 valid=True terminal=nationality
  Reason: The path effectively covers the same nationality aspect, starting from Dark River (2017 Film) and reaching the necessary predicates.
- e2: e2_p14 score=70.0 valid=True terminal=directors
  Reason: The path reaches directors but does not cover the nationality aspect, making it less effective.
- e2: e2_p15 score=60.0 valid=True terminal=Bloodlines
  Reason: The path connects to Bloodlines but does not address the nationality aspect.
- e2: e2_p19 score=30.0 valid=True terminal=Bloodlines
  Reason: The path only connects to Bloodlines and does not address the nationality aspect.
- e2: e2_p22 score=95.0 valid=True terminal=nationality
  Reason: The path effectively connects both films and covers the nationality aspect.
- e2: e2_p23 score=70.0 valid=True terminal=directors
  Reason: The path reaches directors but does not cover the nationality aspect, making it less effective.
- e2: e2_p24 score=95.0 valid=True terminal=nationality
  Reason: The path effectively connects both films and covers the nationality aspect.
- e2: e2_p25 score=95.0 valid=True terminal=nationality
  Reason: The path effectively connects both films and covers the nationality aspect.
- e2: e2_p26 score=90.0 valid=True terminal=nationality
  Reason: The path effectively connects both films and covers the nationality aspect.
- e2: e2_p27 score=90.0 valid=True terminal=nationality
  Reason: The path effectively connects both films and covers the nationality aspect.

## 8.1 Top-2 Paths per Entity
- e1: e1_p15, e1_p3
- e2: e2_p22, e2_p24

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p15', 'e2': 'e2_p22'} mean_path_score=95.0
- ps2: {'e1': 'e1_p15', 'e2': 'e2_p24'} mean_path_score=95.0
- ps3: {'e1': 'e1_p3', 'e2': 'e2_p22'} mean_path_score=95.0
- ps4: {'e1': 'e1_p3', 'e2': 'e2_p24'} mean_path_score=95.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?
- ps1
  - e1_p15: Wrong Turn -> directors -> have -> Bloodlines -> Dark River (2017 Film)
  - e2_p22: Dark River (2017 Film) -> Bloodlines -> have -> directors -> Wrong Turn
- ps2
  - e1_p15: Wrong Turn -> directors -> have -> Bloodlines -> Dark River (2017 Film)
  - e2_p24: Dark River (2017 Film) -> Bloodlines -> have -> directors -> Wrong Turn -> films
- ps3
  - e1_p3: Wrong Turn -> directors -> have -> nationality -> same
  - e2_p22: Dark River (2017 Film) -> Bloodlines -> have -> directors -> Wrong Turn
- ps4
  - e1_p3: Wrong Turn -> directors -> have -> nationality -> same
  - e2_p24: Dark River (2017 Film) -> Bloodlines -> have -> directors -> Wrong Turn -> films

Output:
- selected_path_set_ids: ['ps1', 'ps2', 'ps3', 'ps4']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who are the directors of Wrong Turn 5: Bloodlines? depends_on=[] support=['e1_p15']
- q2: Who are the directors of Dark River (2017 Film)? depends_on=[] support=['e2_p22']
- q3: What is the nationality of the directors of Wrong Turn 5: Bloodlines? depends_on=['q1'] support=['e1_p3']
- q4: What is the nationality of the directors of Dark River (2017 Film)? depends_on=['q2'] support=['e2_p24']
- warning: Node q4 support #1 cites node_texts not present in ps4/e2_p24: ['nationality'].
- warning: Accepted support for q4 with node_texts not present in selected path ps4/e2_p24: ['nationality'].

## 10. Atomic Subquestion DAG
- None: Who are the directors of Wrong Turn 5: Bloodlines?
- None: Who are the directors of Dark River (2017 Film)?
- None: What is the nationality of the directors of Wrong Turn 5: Bloodlines?
- None: What is the nationality of the directors of Dark River (2017 Film)?
