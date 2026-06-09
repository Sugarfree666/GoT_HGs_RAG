# DEPO Decomposition #5

- Dataset: `2wikimultihopqa`
- Question: Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?
- Gold answer: God'S Gift To Women

## 1. Semantic-Normalized Question
Which film has the director who is older than God'S Gift To Women or Aldri Annet Enn Bråk?

## 2. Explicit Entities
- God'S Gift To Women (Work) span=(46, 65)
- Aldri Annet Enn Br (Person) span=(69, 87)

## 3. Entity Masking
- WorkA -> God'S Gift To Women
- PersonA -> Aldri Annet Enn Br

Which film has the director who is older than WorkA or PersonAåk?

## 4. CoreNLP Dependency Parse
- film[2] --det--> Which[1]
- has[3] --nsubj--> film[2]
- director[5] --det--> the[4]
- has[3] --obj--> director[5]
- older[8] --nsubj--> director[5]
- director[5] --ref--> who[6]
- older[8] --cop--> is[7]
- director[5] --acl:relcl--> older[8]
- WorkA[10] --case--> than[9]
- older[8] --obl:than--> WorkA[10]
- PersonAåk[12] --cc--> or[11]
- older[8] --obl:than--> PersonAåk[12]
- WorkA[10] --conj:or--> PersonAåk[12]
- has[3] --punct--> ?[13]

## 5. Undirected Dependency Graph
- Which[1] --det-- film[2]
- film[2] --nsubj-- has[3]
- has[3] --obj-- director[5]
- has[3] --punct-- ?[13]
- the[4] --det-- director[5]
- director[5] --nsubj/acl:relcl-- older[8]
- director[5] --ref-- who[6]
- is[7] --cop-- older[8]
- older[8] --obl:than-- God'S Gift To Women[10]
- older[8] --obl:than-- Aldri Annet Enn Br[12]
- than[9] --case-- God'S Gift To Women[10]
- God'S Gift To Women[10] --conj:or-- Aldri Annet Enn Br[12]
- or[11] --cc-- Aldri Annet Enn Br[12]

## 6. Entity Start Nodes from Explicit Entities
- e1: God'S Gift To Women graph_node_ids=['10']
- e2: Aldri Annet Enn Br graph_node_ids=['12']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): God'S Gift To Women -- older -- director -- has -- film -- Which
- e1_p2 (e1): God'S Gift To Women -- older -- director -- has -- film
- e1_p3 (e1): God'S Gift To Women -- older -- director -- has
- e1_p4 (e1): God'S Gift To Women -- older -- director -- has -- ?
- e1_p5 (e1): God'S Gift To Women -- older -- director -- who
- e1_p6 (e1): God'S Gift To Women -- older -- director
- e1_p7 (e1): God'S Gift To Women -- older -- director -- the
- e1_p8 (e1): God'S Gift To Women -- older
- e1_p9 (e1): God'S Gift To Women -- than
- e1_p10 (e1): God'S Gift To Women -- older -- is
- e1_p11 (e1): God'S Gift To Women -- older -- Aldri Annet Enn Br
- e1_p12 (e1): God'S Gift To Women -- Aldri Annet Enn Br
- e1_p13 (e1): God'S Gift To Women -- Aldri Annet Enn Br -- older -- director -- has -- film -- Which
- e1_p14 (e1): God'S Gift To Women -- Aldri Annet Enn Br -- older -- director -- has -- film
- e1_p15 (e1): God'S Gift To Women -- Aldri Annet Enn Br -- older -- director -- has
- e1_p16 (e1): God'S Gift To Women -- Aldri Annet Enn Br -- older -- director -- has -- ?
- e1_p17 (e1): God'S Gift To Women -- Aldri Annet Enn Br -- older -- director -- who
- e1_p18 (e1): God'S Gift To Women -- Aldri Annet Enn Br -- older -- director
- e1_p19 (e1): God'S Gift To Women -- Aldri Annet Enn Br -- older -- director -- the
- e1_p20 (e1): God'S Gift To Women -- Aldri Annet Enn Br -- older
- e1_p21 (e1): God'S Gift To Women -- older -- Aldri Annet Enn Br -- or
- e1_p22 (e1): God'S Gift To Women -- Aldri Annet Enn Br -- older -- is
- e1_p23 (e1): God'S Gift To Women -- Aldri Annet Enn Br -- or
- e2_p1 (e2): Aldri Annet Enn Br -- older -- director -- has -- film -- Which
- e2_p2 (e2): Aldri Annet Enn Br -- older -- director -- has -- film
- e2_p3 (e2): Aldri Annet Enn Br -- older -- director -- has
- e2_p4 (e2): Aldri Annet Enn Br -- older -- director -- has -- ?
- e2_p5 (e2): Aldri Annet Enn Br -- older -- director -- who
- e2_p6 (e2): Aldri Annet Enn Br -- older -- director
- e2_p7 (e2): Aldri Annet Enn Br -- older -- director -- the
- e2_p8 (e2): Aldri Annet Enn Br -- older
- e2_p9 (e2): Aldri Annet Enn Br -- older -- is
- e2_p10 (e2): Aldri Annet Enn Br -- or
- e2_p11 (e2): Aldri Annet Enn Br -- older -- God'S Gift To Women
- e2_p12 (e2): Aldri Annet Enn Br -- God'S Gift To Women
- e2_p13 (e2): Aldri Annet Enn Br -- God'S Gift To Women -- older -- director -- has -- film -- Which
- e2_p14 (e2): Aldri Annet Enn Br -- God'S Gift To Women -- older -- director -- has -- film
- e2_p15 (e2): Aldri Annet Enn Br -- God'S Gift To Women -- older -- director -- has
- e2_p16 (e2): Aldri Annet Enn Br -- God'S Gift To Women -- older -- director -- has -- ?
- e2_p17 (e2): Aldri Annet Enn Br -- God'S Gift To Women -- older -- director -- who
- e2_p18 (e2): Aldri Annet Enn Br -- God'S Gift To Women -- older -- director
- e2_p19 (e2): Aldri Annet Enn Br -- God'S Gift To Women -- older -- director -- the
- e2_p20 (e2): Aldri Annet Enn Br -- older -- God'S Gift To Women -- than
- e2_p21 (e2): Aldri Annet Enn Br -- God'S Gift To Women -- older
- e2_p22 (e2): Aldri Annet Enn Br -- God'S Gift To Women -- than
- e2_p23 (e2): Aldri Annet Enn Br -- God'S Gift To Women -- older -- is

## 7.5 Terminal Glue Path Pruning
Total raw paths: 46
Total kept paths: 24
Total pruned paths: 22
Total pruned ratio: 47.83%

### By Entity
- e1 / God'S Gift To Women
  - raw: 23
  - kept: 12
  - pruned: 11
  - fallback_used: False
  - examples:
    - e1_p3: God'S Gift To Women -> older -> director -> has [terminal=has, reason=terminal_glue_token]
    - e1_p4: God'S Gift To Women -> older -> director -> has -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p7: God'S Gift To Women -> older -> director -> the [terminal=the, reason=terminal_glue_token]
    - e1_p9: God'S Gift To Women -> than [terminal=than, reason=terminal_glue_dependency_label]
    - e1_p10: God'S Gift To Women -> older -> is [terminal=is, reason=terminal_glue_token]
- e2 / Aldri Annet Enn Br
  - raw: 23
  - kept: 12
  - pruned: 11
  - fallback_used: False
  - examples:
    - e2_p3: Aldri Annet Enn Br -> older -> director -> has [terminal=has, reason=terminal_glue_token]
    - e2_p4: Aldri Annet Enn Br -> older -> director -> has -> ? [terminal=?, reason=terminal_glue_token]
    - e2_p7: Aldri Annet Enn Br -> older -> director -> the [terminal=the, reason=terminal_glue_token]
    - e2_p9: Aldri Annet Enn Br -> older -> is [terminal=is, reason=terminal_glue_token]
    - e2_p10: Aldri Annet Enn Br -> or [terminal=or, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=film
  Reason: The path starts from God'S Gift To Women, reaches the director, and includes the older predicate, effectively supporting the question's intent.
- e1: e1_p2 score=85.0 valid=True terminal=film
  Reason: The path starts from God'S Gift To Women and covers the necessary roles, but lacks the wh cue for full intent coverage.
- e1: e1_p5 score=90.0 valid=True terminal=film
  Reason: The path effectively connects God'S Gift To Women to the director and includes the wh cue, supporting the question's intent.
- e1: e1_p6 score=75.0 valid=True terminal=film
  Reason: The path covers the necessary roles but lacks the wh cue, which is important for the answer intent.
- e1: e1_p8 score=30.0 valid=False terminal=film
  Reason: The path stops too early and does not cover the necessary roles or answer intent.
- e1: e1_p11 score=80.0 valid=True terminal=film
  Reason: The path connects God'S Gift To Women to Aldri Annet Enn Br and includes the older predicate, supporting the question's intent.
- e1: e1_p12 score=30.0 valid=False terminal=film
  Reason: The path does not cover the necessary roles or answer intent.
- e1: e1_p13 score=70.0 valid=True terminal=film
  Reason: The path connects both entities but lacks the wh cue, which is important for the answer intent.
- e1: e1_p14 score=75.0 valid=True terminal=film
  Reason: The path connects both entities and includes the older predicate, supporting the question's intent.
- e1: e1_p17 score=80.0 valid=True terminal=film
  Reason: The path connects both entities and includes the older predicate, supporting the question's intent.
- e1: e1_p18 score=70.0 valid=True terminal=film
  Reason: The path connects both entities but lacks the wh cue, which is important for the answer intent.
- e1: e1_p20 score=30.0 valid=False terminal=film
  Reason: The path does not cover the necessary roles or answer intent.
- e2: e2_p1 score=90.0 valid=True terminal=film
  Reason: The path starts from Aldri Annet Enn Br, reaches the director, and includes the older predicate, effectively supporting the question's intent.
- e2: e2_p2 score=85.0 valid=True terminal=film
  Reason: The path starts from Aldri Annet Enn Br and covers the necessary roles, but lacks the wh cue for full intent coverage.
- e2: e2_p5 score=90.0 valid=True terminal=film
  Reason: The path effectively connects Aldri Annet Enn Br to the director and includes the wh cue, supporting the question's intent.
- e2: e2_p6 score=75.0 valid=True terminal=film
  Reason: The path covers the necessary roles but lacks the wh cue, which is important for the answer intent.
- e2: e2_p8 score=30.0 valid=False terminal=film
  Reason: The path stops too early and does not cover the necessary roles or answer intent.
- e2: e2_p11 score=80.0 valid=True terminal=film
  Reason: The path connects Aldri Annet Enn Br to God'S Gift To Women and includes the older predicate, supporting the question's intent.
- e2: e2_p12 score=30.0 valid=False terminal=film
  Reason: The path does not cover the necessary roles or answer intent.
- e2: e2_p13 score=70.0 valid=True terminal=film
  Reason: The path connects both entities but lacks the wh cue, which is important for the answer intent.
- e2: e2_p14 score=75.0 valid=True terminal=film
  Reason: The path connects both entities and includes the older predicate, supporting the question's intent.
- e2: e2_p17 score=80.0 valid=True terminal=film
  Reason: The path connects both entities and includes the older predicate, supporting the question's intent.
- e2: e2_p18 score=70.0 valid=True terminal=film
  Reason: The path connects both entities but lacks the wh cue, which is important for the answer intent.
- e2: e2_p21 score=30.0 valid=False terminal=film
  Reason: The path does not cover the necessary roles or answer intent.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p5
- e2: e2_p1, e2_p5

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1', 'e2': 'e2_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p1', 'e2': 'e2_p5'} mean_path_score=90.0
- ps3: {'e1': 'e1_p5', 'e2': 'e2_p1'} mean_path_score=90.0
- ps4: {'e1': 'e1_p5', 'e2': 'e2_p5'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?
- ps1
  - e1_p1: God'S Gift To Women -> older -> director -> has -> film -> Which
  - e2_p1: Aldri Annet Enn Br -> older -> director -> has -> film -> Which
- ps2
  - e1_p1: God'S Gift To Women -> older -> director -> has -> film -> Which
  - e2_p5: Aldri Annet Enn Br -> older -> director -> who
- ps3
  - e1_p5: God'S Gift To Women -> older -> director -> who
  - e2_p1: Aldri Annet Enn Br -> older -> director -> has -> film -> Which
- ps4
  - e1_p5: God'S Gift To Women -> older -> director -> who
  - e2_p5: Aldri Annet Enn Br -> older -> director -> who

Output:
- selected_path_set_ids: ['ps1', 'ps2', 'ps3']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who is the director of God'S Gift To Women? depends_on=[] support=['e1_p1']
- q2: Who is the director of Aldri Annet Enn Bråk? depends_on=[] support=['e2_p1']
- q3: What is the birth year of the director of God'S Gift To Women? depends_on=['q1'] support=['e1_p5']
- q4: What is the birth year of the director of Aldri Annet Enn Bråk? depends_on=['q2'] support=['e2_p5']

## 10. Atomic Subquestion DAG
- None: Who is the director of God'S Gift To Women?
- None: Who is the director of Aldri Annet Enn Bråk?
- None: What is the birth year of the director of God'S Gift To Women?
- None: What is the birth year of the director of Aldri Annet Enn Bråk?
