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

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman, covers the necessary roles, and includes the focus predicate 'have' and the answer slot 'country'.
- e1: e1_p2 score=90.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', and the preposition 'from'.
- e1: e1_p3 score=75.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman, covers the necessary roles, and includes the focus predicate 'have' and the answer slot 'country', but the determiner 'the' adds noise.
- e1: e1_p4 score=90.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', and the comparative 'same'.
- e1: e1_p5 score=55.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman and covers the focus predicate 'have' and the role 'directors', but it does not reach the answer slot 'country'.
- e1: e1_p6 score=75.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', but the determiner 'the' adds noise.
- e1: e1_p7 score=30.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman but does not cover any necessary roles or the answer slot.
- e1: e1_p8 score=30.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman but does not cover any necessary roles or the answer slot.
- e1: e1_p9 score=30.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman but does not cover any necessary roles or the answer slot.
- e1: e1_p10 score=30.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman but does not cover any necessary roles or the answer slot.
- e1: e1_p11 score=30.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman but does not cover any necessary roles or the answer slot.
- e1: e1_p12 score=55.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman but does not cover any necessary roles or the answer slot.
- e1: e1_p13 score=30.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman but does not cover any necessary roles or the answer slot.
- e1: e1_p14 score=90.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman, passes through The Last Coupon, covers the necessary roles, and includes the focus predicate 'have' and the answer slot 'country'.
- e1: e1_p15 score=90.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman, passes through The Last Coupon, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', and the preposition 'from'.
- e1: e1_p16 score=90.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman, passes through The Last Coupon, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', and the determiner 'the'.
- e1: e1_p17 score=90.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman, passes through The Last Coupon, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', and the comparative 'same'.
- e1: e1_p18 score=75.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman, passes through The Last Coupon, covers the necessary roles, and includes the focus predicate 'have' but does not reach the answer slot 'country'.
- e1: e1_p19 score=75.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman, passes through The Last Coupon, covers the necessary roles, and includes the focus predicate 'have' but does not reach the answer slot 'country'.
- e1: e1_p20 score=55.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman but does not cover any necessary roles or the answer slot.
- e1: e1_p21 score=55.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman but does not cover any necessary roles or the answer slot.
- e1: e1_p22 score=30.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman but does not cover any necessary roles or the answer slot.
- e1: e1_p23 score=30.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman but does not cover any necessary roles or the answer slot.
- e1: e1_p24 score=30.0 valid=True terminal=country
  Reason: The path starts from Interview With A Hitman but does not cover any necessary roles or the answer slot.
- e2: e2_p1 score=90.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, covers the necessary roles, and includes the focus predicate 'have' and the answer slot 'country'.
- e2: e2_p2 score=90.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', and the preposition 'from'.
- e2: e2_p3 score=75.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', but the determiner 'the' adds noise.
- e2: e2_p4 score=90.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', and the comparative 'same'.
- e2: e2_p5 score=55.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon and covers the focus predicate 'have' and the role 'directors', but it does not reach the answer slot 'country'.
- e2: e2_p6 score=75.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', but the determiner 'the' adds noise.
- e2: e2_p7 score=30.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon but does not cover any necessary roles or the answer slot.
- e2: e2_p8 score=30.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon but does not cover any necessary roles or the answer slot.
- e2: e2_p9 score=30.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon but does not cover any necessary roles or the answer slot.
- e2: e2_p10 score=30.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon but does not cover any necessary roles or the answer slot.
- e2: e2_p11 score=55.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon but does not cover any necessary roles or the answer slot.
- e2: e2_p12 score=30.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon but does not cover any necessary roles or the answer slot.
- e2: e2_p13 score=90.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, passes through Interview With A Hitman, covers the necessary roles, and includes the focus predicate 'have' and the answer slot 'country'.
- e2: e2_p14 score=90.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, passes through Interview With A Hitman, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', and the preposition 'from'.
- e2: e2_p15 score=90.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, passes through Interview With A Hitman, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', and the determiner 'the'.
- e2: e2_p16 score=90.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, passes through Interview With A Hitman, covers the necessary roles, and includes the focus predicate 'have', the answer slot 'country', and the comparative 'same'.
- e2: e2_p17 score=75.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, passes through Interview With A Hitman, covers the necessary roles, and includes the focus predicate 'have' but does not reach the answer slot 'country'.
- e2: e2_p18 score=75.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, passes through Interview With A Hitman, covers the necessary roles, and includes the focus predicate 'have' but does not reach the answer slot 'country'.
- e2: e2_p19 score=75.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon, passes through Interview With A Hitman, covers the necessary roles, and includes the focus predicate 'have' but does not reach the answer slot 'country'.
- e2: e2_p20 score=55.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon but does not cover any necessary roles or the answer slot.
- e2: e2_p21 score=30.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon but does not cover any necessary roles or the answer slot.
- e2: e2_p22 score=30.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon but does not cover any necessary roles or the answer slot.
- e2: e2_p23 score=30.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon but does not cover any necessary roles or the answer slot.
- e2: e2_p24 score=30.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon but does not cover any necessary roles or the answer slot.
- e2: e2_p25 score=30.0 valid=True terminal=country
  Reason: The path starts from The Last Coupon but does not cover any necessary roles or the answer slot.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p14
- e2: e2_p1, e2_p13

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1', 'e2': 'e2_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p1', 'e2': 'e2_p13'} mean_path_score=90.0
- ps3: {'e1': 'e1_p14', 'e2': 'e2_p1'} mean_path_score=90.0
- ps4: {'e1': 'e1_p14', 'e2': 'e2_p13'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- interview_with_a_hitman -> director_r1 (director of Interview With A Hitman)
- director_r1 -> nationality_r1 (nationality of the director)
- the_last_coupon -> director_r2 (director of The Last Coupon)
- director_r2 -> nationality_r2 (nationality of the director)
### ast_ps2 (ps2)
- interview_with_a_hitman -> director_r1 (director of Interview With A Hitman)
- director_r1 -> nationality_r1 (country of the director)
- the_last_coupon -> director_r2 (director of The Last Coupon)
- director_r2 -> nationality_r2 (country of the director)
### ast_ps3 (ps3)
- interview_with_a_hitman -> director_r1 (director of Interview With A Hitman)
- director_r1 -> country_r1 (country of the director)
- the_last_coupon -> director_r2 (director of The Last Coupon)
- director_r2 -> country_r2 (country of the director)
### ast_ps4 (ps4)
- interview_with_a_hitman -> director_r1 (director of Interview With A Hitman)
- director_r1 -> country_r1 (country of the director)
- the_last_coupon -> director_r2 (director of The Last Coupon)
- director_r2 -> country_r2 (country of the director)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST decomposes each film into director and nationality branches without generating a final comparison question.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- interview_with_a_hitman: Interview With A Hitman (entity)
- director_r1: director (type_variable)
- nationality_r1: nationality (value_slot)
- the_last_coupon: The Last Coupon (entity)
- director_r2: director (type_variable)
- nationality_r2: nationality (value_slot)

Edges:
- interview_with_a_hitman -> director_r1 (director of Interview With A Hitman)
- director_r1 -> nationality_r1 (nationality of the director)
- the_last_coupon -> director_r2 (director of The Last Coupon)
- director_r2 -> nationality_r2 (nationality of the director)

## 11. Atomic Subquestion DAG
- None: Who is the director of Interview With A Hitman?
- None: What is the nationality of the director of Interview With A Hitman?
- None: Who is the director of The Last Coupon?
- None: What is the nationality of the director of The Last Coupon?
