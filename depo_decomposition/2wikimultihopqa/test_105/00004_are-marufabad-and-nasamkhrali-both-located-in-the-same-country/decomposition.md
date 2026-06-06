# DEPO Decomposition #4

- Dataset: `2wikimultihopqa`
- Question: Are Marufabad and Nasamkhrali both located in the same country?
- Gold answer: no

## 1. Semantic-Normalized Question
Are Marufabad and Nasamkhrali both located in the same country?

## 2. Mask Spans
(none)

## 3. Selective Masked Question
Are Marufabad and Nasamkhrali both located in the same country?

## 4. CoreNLP Dependency Parse
- located[6] --cop--> Are[1]
- located[6] --nsubj--> Marufabad[2]
- Nasamkhrali[4] --cc--> and[3]
- Marufabad[2] --conj:and--> Nasamkhrali[4]
- located[6] --nsubj--> Nasamkhrali[4]
- Marufabad[2] --dep--> both[5]
- country[10] --case--> in[7]
- country[10] --det--> the[8]
- country[10] --amod--> same[9]
- located[6] --obl:in--> country[10]
- located[6] --punct--> ?[11]

## 5. Undirected Dependency Graph
- Are[1] --cop-- located[6]
- Marufabad[2] --nsubj-- located[6]
- Marufabad[2] --conj:and-- Nasamkhrali[4]
- Marufabad[2] --dep-- both[5]
- and[3] --cc-- Nasamkhrali[4]
- Nasamkhrali[4] --nsubj-- located[6]
- located[6] --obl:in-- country[10]
- located[6] --punct-- ?[11]
- in[7] --case-- country[10]
- the[8] --det-- country[10]
- same[9] --amod-- country[10]

## 6. Entity Start Nodes
- e1: Marufabad graph_node_ids=['2']
- e2: Nasamkhrali graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Marufabad -- located -- country
- e1_p2 (e1): Marufabad -- located -- country -- in
- e1_p3 (e1): Marufabad -- located -- country -- the
- e1_p4 (e1): Marufabad -- located -- country -- same
- e1_p5 (e1): Marufabad -- both
- e1_p6 (e1): Marufabad -- located
- e1_p7 (e1): Marufabad -- located -- Are
- e1_p8 (e1): Marufabad -- located -- ?
- e1_p9 (e1): Marufabad -- located -- Nasamkhrali
- e1_p10 (e1): Marufabad -- Nasamkhrali
- e1_p11 (e1): Marufabad -- Nasamkhrali -- located -- country
- e1_p12 (e1): Marufabad -- Nasamkhrali -- located -- country -- in
- e1_p13 (e1): Marufabad -- Nasamkhrali -- located -- country -- the
- e1_p14 (e1): Marufabad -- Nasamkhrali -- located -- country -- same
- e1_p15 (e1): Marufabad -- Nasamkhrali -- located
- e1_p16 (e1): Marufabad -- Nasamkhrali -- located -- Are
- e1_p17 (e1): Marufabad -- Nasamkhrali -- located -- ?
- e1_p18 (e1): Marufabad -- located -- Nasamkhrali -- and
- e1_p19 (e1): Marufabad -- Nasamkhrali -- and
- e2_p1 (e2): Nasamkhrali -- located -- country
- e2_p2 (e2): Nasamkhrali -- located -- country -- in
- e2_p3 (e2): Nasamkhrali -- located -- country -- the
- e2_p4 (e2): Nasamkhrali -- located -- country -- same
- e2_p5 (e2): Nasamkhrali -- located
- e2_p6 (e2): Nasamkhrali -- located -- Are
- e2_p7 (e2): Nasamkhrali -- located -- ?
- e2_p8 (e2): Nasamkhrali -- and
- e2_p9 (e2): Nasamkhrali -- located -- Marufabad
- e2_p10 (e2): Nasamkhrali -- Marufabad
- e2_p11 (e2): Nasamkhrali -- Marufabad -- located -- country
- e2_p12 (e2): Nasamkhrali -- Marufabad -- located -- country -- in
- e2_p13 (e2): Nasamkhrali -- Marufabad -- located -- country -- the
- e2_p14 (e2): Nasamkhrali -- Marufabad -- located -- country -- same
- e2_p15 (e2): Nasamkhrali -- located -- Marufabad -- both
- e2_p16 (e2): Nasamkhrali -- Marufabad -- both
- e2_p17 (e2): Nasamkhrali -- Marufabad -- located
- e2_p18 (e2): Nasamkhrali -- Marufabad -- located -- Are
- e2_p19 (e2): Nasamkhrali -- Marufabad -- located -- ?

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=country
  Reason: The path starts from Marufabad, reaches located, and covers the country, directly supporting the question's intent.
- e1: e1_p2 score=85.0 valid=True terminal=country
  Reason: The path starts from Marufabad, reaches located, and covers the country, supporting the question's intent but includes an unnecessary preposition.
- e1: e1_p3 score=75.0 valid=True terminal=country
  Reason: The path starts from Marufabad and reaches located and country, but includes a determiner that adds noise.
- e1: e1_p4 score=80.0 valid=True terminal=country
  Reason: The path starts from Marufabad, reaches located and country, and includes the same cue, supporting the question's intent.
- e1: e1_p5 score=30.0 valid=False terminal=country
  Reason: The path stops too early and does not reach the necessary answer target.
- e1: e1_p6 score=55.0 valid=True terminal=country
  Reason: The path starts from Marufabad and reaches located but does not cover the country, missing the answer target.
- e1: e1_p7 score=30.0 valid=False terminal=country
  Reason: The path includes an auxiliary that does not contribute to the answer intent.
- e1: e1_p8 score=30.0 valid=False terminal=country
  Reason: The path includes punctuation that does not contribute to the answer intent.
- e1: e1_p9 score=90.0 valid=True terminal=country
  Reason: The path starts from Marufabad, reaches located, and includes Nasamkhrali, directly supporting the question's intent.
- e1: e1_p10 score=30.0 valid=False terminal=country
  Reason: The path stops too early and does not reach the necessary answer target.
- e1: e1_p11 score=95.0 valid=True terminal=country
  Reason: The path starts from Marufabad, includes Nasamkhrali, reaches located, and covers the country, directly supporting the question's intent.
- e1: e1_p12 score=90.0 valid=True terminal=country
  Reason: The path starts from Marufabad, includes Nasamkhrali, reaches located, and covers the country, directly supporting the question's intent.
- e1: e1_p13 score=90.0 valid=True terminal=country
  Reason: The path starts from Marufabad, includes Nasamkhrali, reaches located, and covers the country, directly supporting the question's intent.
- e1: e1_p14 score=90.0 valid=True terminal=country
  Reason: The path starts from Marufabad, includes Nasamkhrali, reaches located, and covers the country, directly supporting the question's intent.
- e1: e1_p15 score=75.0 valid=True terminal=country
  Reason: The path starts from Marufabad, includes Nasamkhrali, and reaches located but does not cover the country, missing the answer target.
- e1: e1_p16 score=80.0 valid=True terminal=country
  Reason: The path starts from Marufabad, includes Nasamkhrali, reaches located, and covers the country, directly supporting the question's intent.
- e1: e1_p17 score=80.0 valid=True terminal=country
  Reason: The path starts from Marufabad, includes Nasamkhrali, reaches located, and covers the country, directly supporting the question's intent.
- e1: e1_p18 score=80.0 valid=True terminal=country
  Reason: The path starts from Marufabad, includes Nasamkhrali, reaches located, and covers the country, directly supporting the question's intent.
- e1: e1_p19 score=80.0 valid=True terminal=country
  Reason: The path starts from Marufabad, includes Nasamkhrali, reaches located, and covers the country, directly supporting the question's intent.
- e2: e2_p1 score=90.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, reaches located, and covers the country, directly supporting the question's intent.
- e2: e2_p2 score=85.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, reaches located, and covers the country, directly supporting the question's intent but includes an unnecessary preposition.
- e2: e2_p3 score=75.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali and reaches located and country, but includes a determiner that adds noise.
- e2: e2_p4 score=80.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, reaches located and country, and includes the same cue, supporting the question's intent.
- e2: e2_p5 score=55.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali and reaches located but does not cover the country, missing the answer target.
- e2: e2_p6 score=30.0 valid=False terminal=country
  Reason: The path includes an auxiliary that does not contribute to the answer intent.
- e2: e2_p7 score=30.0 valid=False terminal=country
  Reason: The path includes punctuation that does not contribute to the answer intent.
- e2: e2_p8 score=30.0 valid=False terminal=country
  Reason: The path stops too early and does not reach the necessary answer target.
- e2: e2_p9 score=90.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, reaches located, and covers the country, directly supporting the question's intent.
- e2: e2_p10 score=30.0 valid=False terminal=country
  Reason: The path stops too early and does not reach the necessary answer target.
- e2: e2_p11 score=95.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, includes Marufabad, reaches located, and covers the country, directly supporting the question's intent.
- e2: e2_p12 score=90.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, includes Marufabad, reaches located, and covers the country, directly supporting the question's intent.
- e2: e2_p13 score=90.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, includes Marufabad, reaches located, and covers the country, directly supporting the question's intent.
- e2: e2_p14 score=90.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, includes Marufabad, reaches located, and covers the country, directly supporting the question's intent.
- e2: e2_p15 score=95.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, includes Marufabad, reaches located, and covers the country, directly supporting the question's intent.
- e2: e2_p16 score=80.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, includes Marufabad, reaches located, and covers the country, directly supporting the question's intent.
- e2: e2_p17 score=80.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, includes Marufabad, reaches located, and covers the country, directly supporting the question's intent.
- e2: e2_p18 score=80.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, includes Marufabad, reaches located, and covers the country, directly supporting the question's intent.
- e2: e2_p19 score=80.0 valid=True terminal=country
  Reason: The path starts from Nasamkhrali, includes Marufabad, reaches located, and covers the country, directly supporting the question's intent.

## 8.1 Top-2 Paths per Entity
- e1: e1_p11, e1_p1
- e2: e2_p11, e2_p15

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p11', 'e2': 'e2_p11'} mean_path_score=95.0
- ps2: {'e1': 'e1_p11', 'e2': 'e2_p15'} mean_path_score=95.0
- ps3: {'e1': 'e1_p1', 'e2': 'e2_p11'} mean_path_score=92.5
- ps4: {'e1': 'e1_p1', 'e2': 'e2_p15'} mean_path_score=92.5

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- marufabad -> located_e1 (located in)
- nasamkhrali -> located_e2 (located in)
- located_e1 -> country_e1 (in)
- located_e2 -> country_e2 (in)
### ast_ps2 (ps2)
- marufabad -> located_e1 (located in)
- nasamkhrali -> located_e2 (located in)
- located_e1 -> country_e1 (in)
- located_e2 -> country_e2 (in)
### ast_ps3 (ps3)
- marufabad -> country_r1 (located in)
- nasamkhrali -> country_r2 (located in)
### ast_ps4 (ps4)
- marufabad -> country_r1 (located in)
- nasamkhrali -> country_r2 (located in)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively covers the required branches for both entities, providing a clear path to determine their locations and ensuring compatibility with the question's intent.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- marufabad: Marufabad (entity)
- nasamkhrali: Nasamkhrali (entity)
- located_e1: located (type_variable)
- located_e2: located (type_variable)
- country_e1: country (value_slot)
- country_e2: country (value_slot)

Edges:
- marufabad -> located_e1 (located in)
- nasamkhrali -> located_e2 (located in)
- located_e1 -> country_e1 (in)
- located_e2 -> country_e2 (in)

## 11. Atomic Subquestion DAG
- None: Where is Marufabad located?
- None: In which country is the located of Marufabad?
- None: In which country is Nasamkhrali located?
- None: In which country is Nasamkhrali located?
