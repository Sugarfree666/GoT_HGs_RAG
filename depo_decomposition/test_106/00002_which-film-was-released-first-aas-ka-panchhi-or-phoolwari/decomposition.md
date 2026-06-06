# DEPO Decomposition #2

- Dataset: `2wikimultihopqa`
- Question: Which film was released first, Aas Ka Panchhi or Phoolwari?
- Gold answer: Phoolwari

## 1. Semantic-Normalized Question
Which film was released first, Aas Ka Panchhi or Phoolwari?

## 2. Explicit Entities
- Aas Ka Panchhi (Film) span=(31, 45)
- Phoolwari (Film) span=(49, 58)

## 3. Entity Masking
- FilmA -> Aas Ka Panchhi
- FilmB -> Phoolwari

Which film was released first, FilmA or FilmB?

## 4. CoreNLP Dependency Parse
- film[2] --det--> Which[1]
- released[4] --nsubj:pass--> film[2]
- released[4] --aux:pass--> was[3]
- released[4] --advmod--> first[5]
- released[4] --punct--> ,[6]
- released[4] --obj--> FilmA[7]
- FilmB[9] --cc--> or[8]
- released[4] --obj--> FilmB[9]
- FilmA[7] --conj:or--> FilmB[9]
- released[4] --punct--> ?[10]

## 5. Undirected Dependency Graph
- Which[1] --det-- film[2]
- film[2] --nsubj:pass-- released[4]
- was[3] --aux:pass-- released[4]
- released[4] --advmod-- first[5]
- released[4] --punct-- ,[6]
- released[4] --obj-- Aas Ka Panchhi[7]
- released[4] --obj-- Phoolwari[9]
- released[4] --punct-- ?[10]
- Aas Ka Panchhi[7] --conj:or-- Phoolwari[9]
- or[8] --cc-- Phoolwari[9]

## 6. Entity Start Nodes from Explicit Entities
- e1: Aas Ka Panchhi graph_node_ids=['7']
- e2: Phoolwari graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Aas Ka Panchhi -- released -- film -- Which
- e1_p2 (e1): Aas Ka Panchhi -- released -- film
- e1_p3 (e1): Aas Ka Panchhi -- released -- first
- e1_p4 (e1): Aas Ka Panchhi -- released
- e1_p5 (e1): Aas Ka Panchhi -- released -- was
- e1_p6 (e1): Aas Ka Panchhi -- released -- ,
- e1_p7 (e1): Aas Ka Panchhi -- released -- ?
- e1_p8 (e1): Aas Ka Panchhi -- released -- Phoolwari
- e1_p9 (e1): Aas Ka Panchhi -- Phoolwari
- e1_p10 (e1): Aas Ka Panchhi -- Phoolwari -- released -- film -- Which
- e1_p11 (e1): Aas Ka Panchhi -- Phoolwari -- released -- film
- e1_p12 (e1): Aas Ka Panchhi -- Phoolwari -- released -- first
- e1_p13 (e1): Aas Ka Panchhi -- Phoolwari -- released
- e1_p14 (e1): Aas Ka Panchhi -- released -- Phoolwari -- or
- e1_p15 (e1): Aas Ka Panchhi -- Phoolwari -- released -- was
- e1_p16 (e1): Aas Ka Panchhi -- Phoolwari -- released -- ,
- e1_p17 (e1): Aas Ka Panchhi -- Phoolwari -- released -- ?
- e1_p18 (e1): Aas Ka Panchhi -- Phoolwari -- or
- e2_p1 (e2): Phoolwari -- released -- film -- Which
- e2_p2 (e2): Phoolwari -- released -- film
- e2_p3 (e2): Phoolwari -- released -- first
- e2_p4 (e2): Phoolwari -- released
- e2_p5 (e2): Phoolwari -- released -- was
- e2_p6 (e2): Phoolwari -- released -- ,
- e2_p7 (e2): Phoolwari -- released -- ?
- e2_p8 (e2): Phoolwari -- or
- e2_p9 (e2): Phoolwari -- released -- Aas Ka Panchhi
- e2_p10 (e2): Phoolwari -- Aas Ka Panchhi
- e2_p11 (e2): Phoolwari -- Aas Ka Panchhi -- released -- film -- Which
- e2_p12 (e2): Phoolwari -- Aas Ka Panchhi -- released -- film
- e2_p13 (e2): Phoolwari -- Aas Ka Panchhi -- released -- first
- e2_p14 (e2): Phoolwari -- Aas Ka Panchhi -- released
- e2_p15 (e2): Phoolwari -- Aas Ka Panchhi -- released -- was
- e2_p16 (e2): Phoolwari -- Aas Ka Panchhi -- released -- ,
- e2_p17 (e2): Phoolwari -- Aas Ka Panchhi -- released -- ?

## 8. LLM Path Scores
- e1: e1_p1 score=85.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, reaches the release predicate, and includes the 'which' cue for the question intent.
- e1: e1_p2 score=80.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, reaches the release predicate, and includes the 'which' cue for the question intent.
- e1: e1_p3 score=85.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, reaches the release predicate, and includes the 'first' cue for the question intent.
- e1: e1_p4 score=60.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi and reaches the release predicate but misses the 'which' cue for the question intent.
- e1: e1_p5 score=55.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi and reaches the release predicate but misses the 'which' cue for the question intent.
- e1: e1_p6 score=30.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi but ends at punctuation, missing key cues for the question intent.
- e1: e1_p7 score=30.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi but ends at punctuation, missing key cues for the question intent.
- e1: e1_p8 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, reaches the release predicate, and includes the other film, Phoolwari, which is relevant to the question intent.
- e1: e1_p9 score=40.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi and reaches Phoolwari but lacks the release predicate and key cues for the question intent.
- e1: e1_p10 score=95.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, includes Phoolwari, reaches the release predicate, and covers the 'which' cue for the question intent.
- e1: e1_p11 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, includes Phoolwari, reaches the release predicate, but misses the 'which' cue for the question intent.
- e1: e1_p12 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, includes Phoolwari, reaches the release predicate, but misses the 'which' cue for the question intent.
- e1: e1_p13 score=70.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, includes Phoolwari, but ends prematurely without reaching the release predicate.
- e1: e1_p14 score=80.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, includes Phoolwari, reaches the release predicate, but misses the 'which' cue for the question intent.
- e1: e1_p15 score=85.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi, includes Phoolwari, reaches the release predicate, but misses the 'which' cue for the question intent.
- e1: e1_p16 score=30.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi but ends at punctuation, missing key cues for the question intent.
- e1: e1_p17 score=30.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi but ends at punctuation, missing key cues for the question intent.
- e1: e1_p18 score=40.0 valid=True terminal=release_date
  Reason: The path starts from Aas Ka Panchhi and reaches Phoolwari but lacks the release predicate and key cues for the question intent.
- e2: e2_p1 score=85.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, reaches the release predicate, and includes the 'which' cue for the question intent.
- e2: e2_p2 score=80.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, reaches the release predicate, and includes the 'which' cue for the question intent.
- e2: e2_p3 score=85.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, reaches the release predicate, and includes the 'first' cue for the question intent.
- e2: e2_p4 score=60.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari and reaches the release predicate but misses the 'which' cue for the question intent.
- e2: e2_p5 score=55.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari and reaches the release predicate but misses the 'which' cue for the question intent.
- e2: e2_p6 score=30.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari but ends at punctuation, missing key cues for the question intent.
- e2: e2_p7 score=30.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari but ends at punctuation, missing key cues for the question intent.
- e2: e2_p8 score=40.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari but lacks the release predicate and key cues for the question intent.
- e2: e2_p9 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, reaches the release predicate, and includes Aas Ka Panchhi, which is relevant to the question intent.
- e2: e2_p10 score=40.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari and reaches Aas Ka Panchhi but lacks the release predicate and key cues for the question intent.
- e2: e2_p11 score=95.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, includes Aas Ka Panchhi, reaches the release predicate, and covers the 'which' cue for the question intent.
- e2: e2_p12 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, includes Aas Ka Panchhi, reaches the release predicate, but misses the 'which' cue for the question intent.
- e2: e2_p13 score=90.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, includes Aas Ka Panchhi, reaches the release predicate, but misses the 'which' cue for the question intent.
- e2: e2_p14 score=70.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, includes Aas Ka Panchhi, but ends prematurely without reaching the release predicate.
- e2: e2_p15 score=85.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari, includes Aas Ka Panchhi, reaches the release predicate, but misses the 'which' cue for the question intent.
- e2: e2_p16 score=30.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari but ends at punctuation, missing key cues for the question intent.
- e2: e2_p17 score=30.0 valid=True terminal=release_date
  Reason: The path starts from Phoolwari but ends at punctuation, missing key cues for the question intent.

## 8.1 Top-2 Paths per Entity
- e1: e1_p10, e1_p11
- e2: e2_p11, e2_p12

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p10', 'e2': 'e2_p11'} mean_path_score=95.0
- ps2: {'e1': 'e1_p10', 'e2': 'e2_p12'} mean_path_score=92.5
- ps3: {'e1': 'e1_p11', 'e2': 'e2_p11'} mean_path_score=92.5
- ps4: {'e1': 'e1_p11', 'e2': 'e2_p12'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- aas_ka_panchhi -> release_date_e1 (release date of Aas Ka Panchhi)
- phoolwari -> release_date_e2 (release date of Phoolwari)
### ast_ps2 (ps2)
- aas_ka_panchhi -> release_date (release date of Aas Ka Panchhi)
- phoolwari -> phoolwari_release_date (release date of Phoolwari)
### ast_ps3 (ps3)
- aas_ka_panchhi -> release_date_e1 (release date of Aas Ka Panchhi)
- phoolwari -> release_date_e2 (release date of Phoolwari)
### ast_ps4 (ps4)
- aas_ka_panchhi -> release_date_e1 (release date of Aas Ka Panchhi)
- phoolwari -> release_date_e2 (release date of Phoolwari)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively captures the release dates of both films, allowing for direct comparison without generating a final comparison question.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- aas_ka_panchhi: Aas Ka Panchhi (entity)
- phoolwari: Phoolwari (entity)
- release_date_e1: release_date (value_slot)
- release_date_e2: release_date (value_slot)

Edges:
- aas_ka_panchhi -> release_date_e1 (release date of Aas Ka Panchhi)
- phoolwari -> release_date_e2 (release date of Phoolwari)

## 11. Atomic Subquestion DAG
- None: What is the release date of Aas Ka Panchhi?
- None: What is the release date of Phoolwari?
