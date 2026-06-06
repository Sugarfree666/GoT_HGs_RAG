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

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn, reaches directors, covers the have predicate, and includes the nationality cue.
- e1: e1_p2 score=75.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn, reaches directors, covers the have predicate, and includes the nationality cue, but the presence of 'the' adds noise.
- e1: e1_p3 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn, reaches directors, covers the have predicate, and includes the same cue.
- e1: e1_p4 score=55.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn, reaches directors, and covers the have predicate, but it does not address the nationality cue.
- e1: e1_p5 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn and reaches directors, but it only covers 'both' and misses the necessary cues.
- e1: e1_p6 score=55.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn, reaches directors, and covers the have predicate, but it does not address the nationality cue.
- e1: e1_p7 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn and reaches directors, but it does not cover the necessary cues.
- e1: e1_p8 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn and reaches films, but it does not cover the necessary cues.
- e1: e1_p9 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn and reaches 5, but it does not cover the necessary cues.
- e1: e1_p10 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn and reaches Do, but it does not cover the necessary cues.
- e1: e1_p11 score=0.0 valid=False terminal=nationality
  Reason: The path starts from Wrong Turn but ends at punctuation, failing to cover necessary cues.
- e1: e1_p12 score=0.0 valid=False terminal=nationality
  Reason: The path starts from Wrong Turn but ends at punctuation, failing to cover necessary cues.
- e1: e1_p13 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn and reaches of, but it does not cover the necessary cues.
- e1: e1_p14 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn and reaches the, but it does not cover the necessary cues.
- e1: e1_p15 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn, reaches directors, covers the have predicate, and includes both films.
- e1: e1_p16 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn, reaches directors, covers the have predicate, and includes Dark River (2017 Film).
- e1: e1_p17 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn, reaches directors, covers the have predicate, and includes both films.
- e1: e1_p18 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn, reaches directors, covers the have predicate, and includes both films.
- e1: e1_p19 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Wrong Turn, reaches directors, covers the have predicate, and includes Dark River (2017 Film).
- e2: e2_p1 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes both directors.
- e2: e2_p2 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes the nationality cue.
- e2: e2_p3 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes the nationality cue.
- e2: e2_p4 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes the same cue.
- e2: e2_p5 score=55.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, but does not address the nationality cue.
- e2: e2_p6 score=55.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, but does not address the nationality cue.
- e2: e2_p7 score=55.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, but does not address the nationality cue.
- e2: e2_p8 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film) and reaches directors, but it ends at punctuation and does not cover necessary cues.
- e2: e2_p9 score=0.0 valid=False terminal=nationality
  Reason: The path starts from Dark River (2017 Film) but ends at punctuation, failing to cover necessary cues.
- e2: e2_p10 score=55.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes the nationality cue.
- e2: e2_p11 score=55.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes the nationality cue.
- e2: e2_p12 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes the same cue.
- e2: e2_p13 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film) and reaches Bloodlines, but it does not cover the necessary cues.
- e2: e2_p14 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film) and reaches Bloodlines, but it does not cover the necessary cues.
- e2: e2_p15 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film) and reaches Bloodlines, but it does not cover the necessary cues.
- e2: e2_p16 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes the Do cue.
- e2: e2_p17 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes the directors cue.
- e2: e2_p18 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes the Do cue.
- e2: e2_p19 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film) and reaches Bloodlines, but it does not cover the necessary cues.
- e2: e2_p20 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film) and reaches Bloodlines, but it does not cover the necessary cues.
- e2: e2_p21 score=30.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film) but does not cover the necessary cues.
- e2: e2_p22 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes directors from Wrong Turn.
- e2: e2_p23 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes directors from Wrong Turn.
- e2: e2_p24 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes directors from Wrong Turn.
- e2: e2_p25 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes directors from Wrong Turn.
- e2: e2_p26 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes directors from Wrong Turn.
- e2: e2_p27 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes directors from Wrong Turn.
- e2: e2_p28 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes directors from Wrong Turn.
- e2: e2_p29 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes directors from Wrong Turn.
- e2: e2_p30 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes directors from Wrong Turn.
- e2: e2_p31 score=90.0 valid=True terminal=nationality
  Reason: The path starts from Dark River (2017 Film), reaches Bloodlines, covers the have predicate, and includes directors from Wrong Turn.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p15
- e2: e2_p1, e2_p12

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1', 'e2': 'e2_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p1', 'e2': 'e2_p12'} mean_path_score=90.0
- ps3: {'e1': 'e1_p15', 'e2': 'e2_p1'} mean_path_score=90.0
- ps4: {'e1': 'e1_p15', 'e2': 'e2_p12'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- wrong_turn -> director_r1 (director of Wrong Turn 5: Bloodlines)
- director_r1 -> nationality_r1 (nationality of the director)
- dark_river -> director_r2 (director of Dark River (2017 Film))
- director_r2 -> nationality_r2 (nationality of the director)
### ast_ps2 (ps2)
- wrong_turn -> director_r1 (director of Wrong Turn 5: Bloodlines)
- director_r1 -> nationality_r1 (nationality of the director)
- dark_river -> director_r2 (director of Dark River (2017 Film))
- director_r2 -> nationality_r2 (nationality of the director)
### ast_ps3 (ps3)
- wrong_turn_5_bloodlines -> director_r1 (director of Wrong Turn 5: Bloodlines)
- dark_river_2017_film -> director_r2 (director of Dark River (2017 Film))
- director_r1 -> nationality_r1 (nationality of the director)
- director_r2 -> nationality_r2 (nationality of the director)
### ast_ps4 (ps4)
- wrong_turn_5_bloodlines -> director_r1 (director of Wrong Turn 5: Bloodlines)
- director_r1 -> nationality_r1 (nationality of the director)
- dark_river_2017_film -> director_r2 (director of Dark River (2017 Film))
- director_r2 -> nationality_r2 (nationality of the director)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST decomposes each film into director and nationality branches without generating a final comparison question.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- wrong_turn: Wrong Turn 5: Bloodlines (entity)
- director_r1: director (type_variable)
- nationality_r1: nationality (value_slot)
- dark_river: Dark River (2017 Film) (entity)
- director_r2: director (type_variable)
- nationality_r2: nationality (value_slot)

Edges:
- wrong_turn -> director_r1 (director of Wrong Turn 5: Bloodlines)
- director_r1 -> nationality_r1 (nationality of the director)
- dark_river -> director_r2 (director of Dark River (2017 Film))
- director_r2 -> nationality_r2 (nationality of the director)

## 11. Atomic Subquestion DAG
- None: Who is the director of Wrong Turn 5: Bloodlines?
- None: What is the nationality of the director of Wrong Turn 5: Bloodlines?
- None: Who is the director of Dark River (2017 Film)?
- None: What is the nationality of the director of Dark River (2017 Film)?
