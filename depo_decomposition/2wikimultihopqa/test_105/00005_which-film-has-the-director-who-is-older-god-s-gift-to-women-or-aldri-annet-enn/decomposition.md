# DEPO Decomposition #5

- Dataset: `2wikimultihopqa`
- Question: Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?
- Gold answer: God'S Gift To Women

## 1. Semantic-Normalized Question
Which film has the director who is older than God'S Gift To Women or Aldri Annet Enn Bråk?

## 2. Mask Spans
- has the director (entity, Film)
- God'S Gift To Women (entity, Person)
- Aldri Annet Enn Bråk (entity, Person)

## 3. Selective Masked Question
Which film PersonA who is older than PersonB or PersonC?

## 4. CoreNLP Dependency Parse
- PersonA[3] --det--> Which[1]
- PersonA[3] --compound--> film[2]
- older[6] --dep--> PersonA[3]
- older[6] --nsubj--> who[4]
- older[6] --cop--> is[5]
- PersonB[8] --case--> than[7]
- older[6] --obl:than--> PersonB[8]
- PersonC[10] --cc--> or[9]
- older[6] --obl:than--> PersonC[10]
- PersonB[8] --conj:or--> PersonC[10]
- older[6] --punct--> ?[11]

## 5. Undirected Dependency Graph
- Which[1] --det-- has the director[3]
- film[2] --compound-- has the director[3]
- has the director[3] --dep-- older[6]
- who[4] --nsubj-- older[6]
- is[5] --cop-- older[6]
- older[6] --obl:than-- God'S Gift To Women[8]
- older[6] --obl:than-- Aldri Annet Enn Bråk[10]
- older[6] --punct-- ?[11]
- than[7] --case-- God'S Gift To Women[8]
- God'S Gift To Women[8] --conj:or-- Aldri Annet Enn Bråk[10]
- or[9] --cc-- Aldri Annet Enn Bråk[10]

## 6. Entity Start Nodes
- e1: has the director graph_node_ids=['3']
- e2: God'S Gift To Women graph_node_ids=['8']
- e3: Aldri Annet Enn Bråk graph_node_ids=['10']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): has the director -- older -- who
- e1_p2 (e1): has the director -- film
- e1_p3 (e1): has the director -- older
- e1_p4 (e1): has the director -- older -- is
- e1_p5 (e1): has the director -- older -- ?
- e1_p6 (e1): has the director -- Which
- e1_p7 (e1): has the director -- older -- God'S Gift To Women
- e1_p8 (e1): has the director -- older -- Aldri Annet Enn Bråk
- e1_p9 (e1): has the director -- older -- God'S Gift To Women -- than
- e1_p10 (e1): has the director -- older -- Aldri Annet Enn Bråk -- or
- e1_p11 (e1): has the director -- older -- God'S Gift To Women -- Aldri Annet Enn Bråk
- e1_p12 (e1): has the director -- older -- Aldri Annet Enn Bråk -- God'S Gift To Women
- e1_p13 (e1): has the director -- older -- Aldri Annet Enn Bråk -- God'S Gift To Women -- than
- e1_p14 (e1): has the director -- older -- God'S Gift To Women -- Aldri Annet Enn Bråk -- or
- e2_p1 (e2): God'S Gift To Women -- older -- who
- e2_p2 (e2): God'S Gift To Women -- older
- e2_p3 (e2): God'S Gift To Women -- than
- e2_p4 (e2): God'S Gift To Women -- older -- is
- e2_p5 (e2): God'S Gift To Women -- older -- ?
- e2_p6 (e2): God'S Gift To Women -- older -- has the director
- e2_p7 (e2): God'S Gift To Women -- older -- Aldri Annet Enn Bråk
- e2_p8 (e2): God'S Gift To Women -- Aldri Annet Enn Bråk
- e2_p9 (e2): God'S Gift To Women -- older -- has the director -- film
- e2_p10 (e2): God'S Gift To Women -- older -- has the director -- Which
- e2_p11 (e2): God'S Gift To Women -- Aldri Annet Enn Bråk -- older -- who
- e2_p12 (e2): God'S Gift To Women -- Aldri Annet Enn Bråk -- older
- e2_p13 (e2): God'S Gift To Women -- older -- Aldri Annet Enn Bråk -- or
- e2_p14 (e2): God'S Gift To Women -- Aldri Annet Enn Bråk -- older -- is
- e2_p15 (e2): God'S Gift To Women -- Aldri Annet Enn Bråk -- older -- ?
- e2_p16 (e2): God'S Gift To Women -- Aldri Annet Enn Bråk -- or
- e2_p17 (e2): God'S Gift To Women -- Aldri Annet Enn Bråk -- older -- has the director
- e2_p18 (e2): God'S Gift To Women -- Aldri Annet Enn Bråk -- older -- has the director -- film
- e2_p19 (e2): God'S Gift To Women -- Aldri Annet Enn Bråk -- older -- has the director -- Which
- e3_p1 (e3): Aldri Annet Enn Bråk -- older -- who
- e3_p2 (e3): Aldri Annet Enn Bråk -- older
- e3_p3 (e3): Aldri Annet Enn Bråk -- older -- is
- e3_p4 (e3): Aldri Annet Enn Bråk -- older -- ?
- e3_p5 (e3): Aldri Annet Enn Bråk -- or
- e3_p6 (e3): Aldri Annet Enn Bråk -- older -- has the director
- e3_p7 (e3): Aldri Annet Enn Bråk -- older -- God'S Gift To Women
- e3_p8 (e3): Aldri Annet Enn Bråk -- God'S Gift To Women
- e3_p9 (e3): Aldri Annet Enn Bråk -- older -- has the director -- film
- e3_p10 (e3): Aldri Annet Enn Bråk -- older -- God'S Gift To Women -- than
- e3_p11 (e3): Aldri Annet Enn Bråk -- older -- has the director -- Which
- e3_p12 (e3): Aldri Annet Enn Bråk -- God'S Gift To Women -- older -- who
- e3_p13 (e3): Aldri Annet Enn Bråk -- God'S Gift To Women -- older
- e3_p14 (e3): Aldri Annet Enn Bråk -- God'S Gift To Women -- than
- e3_p15 (e3): Aldri Annet Enn Bråk -- God'S Gift To Women -- older -- is
- e3_p16 (e3): Aldri Annet Enn Bråk -- God'S Gift To Women -- older -- ?
- e3_p17 (e3): Aldri Annet Enn Bråk -- God'S Gift To Women -- older -- has the director
- e3_p18 (e3): Aldri Annet Enn Bråk -- God'S Gift To Women -- older -- has the director -- film
- e3_p19 (e3): Aldri Annet Enn Bråk -- God'S Gift To Women -- older -- has the director -- Which

## 8. LLM Path Scores
- e1: e1_p1 score=75.0 valid=True terminal=director_age_comparison
  Reason: The path covers the necessary cues but lacks the film node, which is essential for a complete semantic chain.
- e1: e1_p2 score=55.0 valid=True terminal=film_director
  Reason: The path identifies the film but does not connect to the age comparison or the director.
- e1: e1_p3 score=70.0 valid=True terminal=director_age_comparison
  Reason: The path includes the age predicate but lacks the film and who cues necessary for a complete chain.
- e1: e1_p4 score=75.0 valid=True terminal=director_age_comparison
  Reason: The path includes the age predicate and the is copula but misses the film and who cues.
- e1: e1_p5 score=30.0 valid=True terminal=director_age_comparison
  Reason: The path ends with a punctuation mark and does not contribute to the semantic chain.
- e1: e1_p6 score=30.0 valid=True terminal=director_age_comparison
  Reason: The path ends with a determiner and does not contribute to the semantic chain.
- e1: e1_p7 score=90.0 valid=True terminal=director_age_comparison
  Reason: The path effectively connects the age comparison to one of the entities but misses the film and who cues.
- e1: e1_p8 score=90.0 valid=True terminal=director_age_comparison
  Reason: The path effectively connects the age comparison to the other entity but misses the film and who cues.
- e1: e1_p9 score=85.0 valid=True terminal=director_age_comparison
  Reason: The path connects the age comparison and includes the than cue but misses the film and who cues.
- e1: e1_p10 score=85.0 valid=True terminal=director_age_comparison
  Reason: The path connects the age comparison and includes the or cue but misses the film and who cues.
- e1: e1_p11 score=90.0 valid=True terminal=director_age_comparison
  Reason: The path effectively connects both entities in the age comparison but misses the film and who cues.
- e1: e1_p12 score=90.0 valid=True terminal=director_age_comparison
  Reason: The path effectively connects both entities in the age comparison but misses the film and who cues.
- e1: e1_p13 score=80.0 valid=True terminal=director_age_comparison
  Reason: The path connects both entities in the age comparison and includes the than cue but misses the film and who cues.
- e1: e1_p14 score=80.0 valid=True terminal=director_age_comparison
  Reason: The path connects both entities in the age comparison and includes the or cue but misses the film and who cues.
- e2: e2_p1 score=75.0 valid=True terminal=director_age_comparison
  Reason: The path covers the necessary cues but lacks the film node, which is essential for a complete semantic chain.
- e2: e2_p2 score=55.0 valid=True terminal=film_director
  Reason: The path identifies the film but does not connect to the age comparison or the director.
- e2: e2_p3 score=30.0 valid=True terminal=director_age_comparison
  Reason: The path ends with a case marker and does not contribute to the semantic chain.
- e2: e2_p4 score=70.0 valid=True terminal=director_age_comparison
  Reason: The path includes the age predicate and the is copula but misses the film and who cues.
- e2: e2_p5 score=30.0 valid=True terminal=director_age_comparison
  Reason: The path ends with a punctuation mark and does not contribute to the semantic chain.
- e2: e2_p6 score=90.0 valid=True terminal=director_age_comparison
  Reason: The path effectively connects the age comparison to the director but misses the film and who cues.
- e2: e2_p7 score=90.0 valid=True terminal=director_age_comparison
  Reason: The path effectively connects the age comparison to the other entity but misses the film and who cues.
- e2: e2_p8 score=30.0 valid=True terminal=director_age_comparison
  Reason: The path ends with a conjunction and does not contribute to the semantic chain.
- e2: e2_p9 score=85.0 valid=True terminal=director_age_comparison
  Reason: The path connects the age comparison and includes the film cue but misses the who cue.
- e2: e2_p10 score=85.0 valid=True terminal=director_age_comparison
  Reason: The path connects the age comparison and includes the Which cue but misses the film and who cues.
- e2: e2_p11 score=90.0 valid=True terminal=director_age_comparison
  Reason: The path effectively connects the age comparison to one of the entities but misses the film cue.
- e2: e2_p12 score=90.0 valid=True terminal=director_age_comparison
  Reason: The path effectively connects both entities in the age comparison but misses the film cue.
- e2: e2_p13 score=80.0 valid=True terminal=director_age_comparison
  Reason: The path connects both entities in the age comparison and includes the than cue but misses the film cue.
- e2: e2_p14 score=80.0 valid=True terminal=director_age_comparison
  Reason: The path connects both entities in the age comparison and includes the or cue but misses the film cue.
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
- e3: e3_p1 score=75.0 valid=True terminal=director_age_comparison
  Reason: The path covers the necessary cues but lacks the film node, which is essential for a complete semantic chain.
- e3: e3_p2 score=55.0 valid=True terminal=film_director
  Reason: The path identifies the film but does not connect to the age comparison or the director.
- e3: e3_p3 score=70.0 valid=True terminal=director_age_comparison
  Reason: The path includes the age predicate and the is copula but misses the film and who cues.
- e3: e3_p4 score=30.0 valid=True terminal=director_age_comparison
  Reason: The path ends with a punctuation mark and does not contribute to the semantic chain.
- e3: e3_p5 score=30.0 valid=True terminal=director_age_comparison
  Reason: The path ends with a conjunction and does not contribute to the semantic chain.
- e3: e3_p6 score=90.0 valid=True terminal=director_age_comparison
  Reason: The path effectively connects the age comparison to the director but misses the film and who cues.
- e3: e3_p7 score=90.0 valid=True terminal=director_age_comparison
  Reason: The path effectively connects the age comparison to the other entity but misses the film and who cues.
- e3: e3_p8 score=30.0 valid=True terminal=director_age_comparison
  Reason: The path ends with a conjunction and does not contribute to the semantic chain.
- e3: e3_p9 score=85.0 valid=True terminal=director_age_comparison
  Reason: The path connects the age comparison and includes the film cue but misses the who cue.
- e3: e3_p10 score=85.0 valid=True terminal=director_age_comparison
  Reason: The path connects the age comparison and includes the God'S Gift To Women cue but misses the film and who cues.
- e3: e3_p11 score=90.0 valid=True terminal=director_age_comparison
  Reason: The path effectively connects the age comparison to one of the entities but misses the film cue.
- e3: e3_p12 score=90.0 valid=True terminal=director_age_comparison
  Reason: The path effectively connects both entities in the age comparison but misses the film cue.
- e3: e3_p13 score=80.0 valid=True terminal=director_age_comparison
  Reason: The path connects both entities in the age comparison and includes the than cue but misses the film cue.
- e3: e3_p14 score=80.0 valid=True terminal=director_age_comparison
  Reason: The path connects both entities in the age comparison and includes the or cue but misses the film cue.
- e3: e3_p15 score=0.0 valid=False
  Reason: missing from LLM output
- e3: e3_p16 score=0.0 valid=False
  Reason: missing from LLM output
- e3: e3_p17 score=0.0 valid=False
  Reason: missing from LLM output
- e3: e3_p18 score=0.0 valid=False
  Reason: missing from LLM output
- e3: e3_p19 score=0.0 valid=False
  Reason: missing from LLM output

## 8.1 Top-2 Paths per Entity
- e1: e1_p11, e1_p12
- e2: e2_p11, e2_p12
- e3: e3_p11, e3_p12

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p11', 'e2': 'e2_p11', 'e3': 'e3_p11'} mean_path_score=90.0
- ps2: {'e1': 'e1_p11', 'e2': 'e2_p11', 'e3': 'e3_p12'} mean_path_score=90.0
- ps3: {'e1': 'e1_p11', 'e2': 'e2_p12', 'e3': 'e3_p11'} mean_path_score=90.0
- ps4: {'e1': 'e1_p11', 'e2': 'e2_p12', 'e3': 'e3_p12'} mean_path_score=90.0
- ps5: {'e1': 'e1_p12', 'e2': 'e2_p11', 'e3': 'e3_p11'} mean_path_score=90.0
- ps6: {'e1': 'e1_p12', 'e2': 'e2_p11', 'e3': 'e3_p12'} mean_path_score=90.0
- ps7: {'e1': 'e1_p12', 'e2': 'e2_p12', 'e3': 'e3_p11'} mean_path_score=90.0
- ps8: {'e1': 'e1_p12', 'e2': 'e2_p12', 'e3': 'e3_p12'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- god_s_gift_to_women -> director_r1 (director of God's Gift To Women)
- aldrin_annet_enn_br_k -> director_r2 (director of Aldri Annet Enn Bråk)
- director_r1 -> age_r1 (age of the director)
- director_r2 -> age_r2 (age of the director)
### ast_ps2 (ps2)
- god_s_gift_to_women -> director_r1 (director of God'S Gift To Women)
- aldrin_annet_enn_brak -> director_r2 (director of Aldri Annet Enn Bråk)
- director_r1 -> age_r1 (age of the director)
- director_r2 -> age_r2 (age of the director)
### ast_ps3 (ps3)
- god_s_gift_to_women -> director_r1 (director of God'S Gift To Women)
- aldrin_annet_enn_brak -> director_r2 (director of Aldri Annet Enn Bråk)
- director_r1 -> age_r1 (age of the director)
- director_r2 -> age_r2 (age of the director)
### ast_ps4 (ps4)
- god_s_gift_to_women -> director_r1 (director of God'S Gift To Women)
- aldrin_annet_enn_brak -> director_r2 (director of Aldri Annet Enn Bråk)
- aldrin_annet_enn_brak -> director_r3 (director of Aldri Annet Enn Bråk)
- director_r1 -> age_r1 (age of the director)
- director_r2 -> age_r2 (age of the director)
- director_r3 -> age_r3 (age of the director)
### ast_ps5 (ps5)
- god_s_gift_to_women -> director_r1 (director of God'S Gift To Women)
- aldria_annet_enn_br_k -> director_r2 (director of Aldri Annet Enn Bråk)
- director_r1 -> age_r1 (age of the director)
- director_r2 -> age_r2 (age of the director)
### ast_ps6 (ps6)
- god_s_gift_to_women -> director_r1 (director of God'S Gift To Women)
- aldrin_annet_enn_brak -> director_r2 (director of Aldri Annet Enn Bråk)
- director_r1 -> age_r1 (age of the director)
- director_r2 -> age_r2 (age of the director)
### ast_ps7 (ps7)
- god_s_gift_to_women -> director_r1 (director of God'S Gift To Women)
- aldrin_annet_enn_brak -> director_r2 (director of Aldri Annet Enn Bråk)
- director_r1 -> age_r1 (age of the director)
- director_r2 -> age_r2 (age of the director)
### ast_ps8 (ps8)
- god_s_gift_to_women -> director_r1 (director of God'S Gift To Women)
- aldr_i_annet_enn_brak -> director_r2 (director of Aldri Annet Enn Bråk)
- director_r1 -> age_r1 (age of director)
- director_r2 -> age_r2 (age of director)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST decomposes each film into director and age branches without generating a final comparison question.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- god_s_gift_to_women: God's Gift To Women (entity)
- aldrin_annet_enn_br_k: Aldri Annet Enn Bråk (entity)
- director_r1: director (type_variable)
- director_r2: director (type_variable)
- age_r1: age (value_slot)
- age_r2: age (value_slot)

Edges:
- god_s_gift_to_women -> director_r1 (director of God's Gift To Women)
- aldrin_annet_enn_br_k -> director_r2 (director of Aldri Annet Enn Bråk)
- director_r1 -> age_r1 (age of the director)
- director_r2 -> age_r2 (age of the director)

## 11. Atomic Subquestion DAG
- None: Who is the director of God's Gift To Women?
- None: What is the age of the director of God's Gift To Women?
- None: Who is the director of Aldri Annet Enn Bråk?
- None: What is the age of the director of Aldri Annet Enn Bråk?
