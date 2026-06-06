# DEPO Decomposition #15

- Dataset: `hotpotqa`
- Question: Which 2009 animated film is from Japan, Summer Wars or The Secret of Kells?
- Gold answer: Summer Wars

## 1. Semantic-Normalized Question
Which animated film from 2009 is from Japan, Summer Wars or The Secret of Kells?

## 2. Mask Spans
- from 2009 (entity, Film)
- Summer Wars (entity, Film)
- The Secret of Kells (entity, Film)

## 3. Selective Masked Question
Which animated film MovieA is from Japan, MovieB or MovieC?

## 4. CoreNLP Dependency Parse
- MovieA[4] --det--> Which[1]
- MovieA[4] --amod--> animated[2]
- MovieA[4] --compound--> film[3]
- Japan[7] --nsubj--> MovieA[4]
- Japan[7] --cop--> is[5]
- Japan[7] --case--> from[6]
- Japan[7] --punct--> ,[8]
- Japan[7] --conj:or--> MovieB[9]
- MovieC[11] --cc--> or[10]
- Japan[7] --conj:or--> MovieC[11]
- Japan[7] --punct--> ?[12]

## 5. Undirected Dependency Graph
- Which[1] --det-- from 2009[4]
- animated[2] --amod-- from 2009[4]
- film[3] --compound-- from 2009[4]
- from 2009[4] --nsubj-- Japan[7]
- is[5] --cop-- Japan[7]
- from[6] --case-- Japan[7]
- Japan[7] --punct-- ,[8]
- Japan[7] --conj:or-- Summer Wars[9]
- Japan[7] --conj:or-- The Secret of Kells[11]
- Japan[7] --punct-- ?[12]
- or[10] --cc-- The Secret of Kells[11]

## 6. Entity Start Nodes
- e1: from 2009 graph_node_ids=['4']
- e2: Summer Wars graph_node_ids=['9']
- e3: The Secret of Kells graph_node_ids=['11']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): from 2009 -- animated
- e1_p2 (e1): from 2009 -- film
- e1_p3 (e1): from 2009 -- Japan
- e1_p4 (e1): from 2009 -- Japan -- is
- e1_p5 (e1): from 2009 -- Japan -- from
- e1_p6 (e1): from 2009 -- Japan -- ,
- e1_p7 (e1): from 2009 -- Japan -- ?
- e1_p8 (e1): from 2009 -- Which
- e1_p9 (e1): from 2009 -- Japan -- Summer Wars
- e1_p10 (e1): from 2009 -- Japan -- The Secret of Kells
- e1_p11 (e1): from 2009 -- Japan -- The Secret of Kells -- or
- e2_p1 (e2): Summer Wars -- Japan
- e2_p2 (e2): Summer Wars -- Japan -- is
- e2_p3 (e2): Summer Wars -- Japan -- from
- e2_p4 (e2): Summer Wars -- Japan -- ,
- e2_p5 (e2): Summer Wars -- Japan -- ?
- e2_p6 (e2): Summer Wars -- Japan -- from 2009
- e2_p7 (e2): Summer Wars -- Japan -- The Secret of Kells
- e2_p8 (e2): Summer Wars -- Japan -- from 2009 -- animated
- e2_p9 (e2): Summer Wars -- Japan -- from 2009 -- film
- e2_p10 (e2): Summer Wars -- Japan -- from 2009 -- Which
- e2_p11 (e2): Summer Wars -- Japan -- The Secret of Kells -- or
- e3_p1 (e3): The Secret of Kells -- Japan
- e3_p2 (e3): The Secret of Kells -- Japan -- is
- e3_p3 (e3): The Secret of Kells -- Japan -- from
- e3_p4 (e3): The Secret of Kells -- Japan -- ,
- e3_p5 (e3): The Secret of Kells -- Japan -- ?
- e3_p6 (e3): The Secret of Kells -- or
- e3_p7 (e3): The Secret of Kells -- Japan -- from 2009
- e3_p8 (e3): The Secret of Kells -- Japan -- Summer Wars
- e3_p9 (e3): The Secret of Kells -- Japan -- from 2009 -- animated
- e3_p10 (e3): The Secret of Kells -- Japan -- from 2009 -- film
- e3_p11 (e3): The Secret of Kells -- Japan -- from 2009 -- Which

## 8. LLM Path Scores
- e1: e1_p1 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e1: e1_p2 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e1: e1_p3 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e1: e1_p4 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e1: e1_p5 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e1: e1_p6 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e1: e1_p7 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e1: e1_p8 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e1: e1_p9 score=90.0 valid=True terminal=film
  Reason: The path connects 'from 2009' to 'Japan' and then to 'Summer Wars', covering the necessary roles and answer target.
- e1: e1_p10 score=90.0 valid=True terminal=film
  Reason: The path connects 'from 2009' to 'Japan' and then to 'The Secret of Kells', covering the necessary roles and answer target.
- e1: e1_p11 score=75.0 valid=True terminal=film
  Reason: The path connects 'from 2009' to 'Japan' and then to 'The Secret of Kells', but ends with a conjunction which slightly reduces its effectiveness.
- e2: e2_p1 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e2: e2_p2 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e2: e2_p3 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e2: e2_p4 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e2: e2_p5 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e2: e2_p6 score=90.0 valid=True terminal=film
  Reason: The path connects 'Summer Wars' to 'Japan' and then to 'from 2009', covering the necessary roles and answer target.
- e2: e2_p7 score=90.0 valid=True terminal=film
  Reason: The path connects 'Summer Wars' to 'Japan' and then to 'The Secret of Kells', covering the necessary roles and answer target.
- e2: e2_p8 score=75.0 valid=True terminal=film
  Reason: The path connects 'Summer Wars' to 'Japan' and then to 'from 2009', but ends with a conjunction which slightly reduces its effectiveness.
- e2: e2_p9 score=90.0 valid=True terminal=film
  Reason: The path connects 'Summer Wars' to 'Japan' and then to 'film', covering the necessary roles and answer target.
- e2: e2_p10 score=90.0 valid=True terminal=film
  Reason: The path connects 'Summer Wars' to 'Japan' and then to 'from 2009', covering the necessary roles and answer target.
- e2: e2_p11 score=75.0 valid=True terminal=film
  Reason: The path connects 'Summer Wars' to 'Japan' and then to 'The Secret of Kells', but ends with a conjunction which slightly reduces its effectiveness.
- e3: e3_p1 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e3: e3_p2 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e3: e3_p3 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e3: e3_p4 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e3: e3_p5 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e3: e3_p6 score=30.0 valid=True
  Reason: The path does not reach the necessary intermediate roles or answer target.
- e3: e3_p7 score=90.0 valid=True terminal=film
  Reason: The path connects 'The Secret of Kells' to 'Japan' and then to 'from 2009', covering the necessary roles and answer target.
- e3: e3_p8 score=90.0 valid=True terminal=film
  Reason: The path connects 'The Secret of Kells' to 'Japan' and then to 'Summer Wars', covering the necessary roles and answer target.
- e3: e3_p9 score=90.0 valid=True terminal=film
  Reason: The path connects 'The Secret of Kells' to 'Japan' and then to 'from 2009', covering the necessary roles and answer target.
- e3: e3_p10 score=90.0 valid=True terminal=film
  Reason: The path connects 'The Secret of Kells' to 'Japan' and then to 'film', covering the necessary roles and answer target.
- e3: e3_p11 score=75.0 valid=True terminal=film
  Reason: The path connects 'The Secret of Kells' to 'Japan' and then to 'Summer Wars', but ends with a conjunction which slightly reduces its effectiveness.

## 8.1 Top-2 Paths per Entity
- e1: e1_p10, e1_p9
- e2: e2_p10, e2_p6
- e3: e3_p10, e3_p7

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p10', 'e2': 'e2_p10', 'e3': 'e3_p10'} mean_path_score=90.0
- ps2: {'e1': 'e1_p10', 'e2': 'e2_p10', 'e3': 'e3_p7'} mean_path_score=90.0
- ps3: {'e1': 'e1_p10', 'e2': 'e2_p6', 'e3': 'e3_p10'} mean_path_score=90.0
- ps4: {'e1': 'e1_p10', 'e2': 'e2_p6', 'e3': 'e3_p7'} mean_path_score=90.0
- ps5: {'e1': 'e1_p9', 'e2': 'e2_p10', 'e3': 'e3_p10'} mean_path_score=90.0
- ps6: {'e1': 'e1_p9', 'e2': 'e2_p10', 'e3': 'e3_p7'} mean_path_score=90.0
- ps7: {'e1': 'e1_p9', 'e2': 'e2_p6', 'e3': 'e3_p10'} mean_path_score=90.0
- ps8: {'e1': 'e1_p9', 'e2': 'e2_p6', 'e3': 'e3_p7'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- summer_wars -> japan_e2 (set in)
- japan_e2 -> from_2009_e2 (released in)
- japan_e3 -> from_2009_e3 (released in)
- the_secret_of_kells -> japan_e3 (set in)
- japan_e2 -> film_e2 (film from)
- japan_e3 -> film_e3 (film from)
### ast_ps2 (ps2)
- summer_wars -> japan_e2 (is from)
- japan_e2 -> from_2009_e2 (release_date of)
- japan_e3 -> from_2009_e3 (release_date of)
- the_secret_of_kells -> japan_e3 (is from)
### ast_ps3 (ps3)
- the_secret_of_kells -> japan_e3 (is from)
- japan_e2 -> film_e2 (is a)
- japan_e3 -> film_e3 (is a)
- summer_wars -> japan_e2 (is from)
### ast_ps4 (ps4)
- summer_wars -> japan_e2 (set in)
- japan_e2 -> from_2009_e2 (released in)
- japan_e3 -> from_2009_e3 (released in)
- the_secret_of_kells -> japan_e3 (set in)
- japan_e2 -> from_2009_kells_e2 (released in)
- japan_e3 -> from_2009_kells_e3 (released in)
### ast_ps5 (ps5)
- summer_wars -> japan_e2 (set in)
- the_secret_of_kells -> japan_e3 (set in)
- the_secret_of_kells -> film_e3 (is a)
- summer_wars -> film_e2 (is a)
### ast_ps6 (ps6)
- summer_wars -> japan_e2 (set in)
- japan_e2 -> from_2009_e2 (released in)
- japan_e3 -> from_2009_e3 (released in)
- the_secret_of_kells -> japan_e3 (set in)
### ast_ps7 (ps7)
- summer_wars -> japan_e2 (set in)
- the_secret_of_kells -> japan_e3 (set in)
- the_secret_of_kells -> film (is a)
### ast_ps8 (ps8)
- summer_wars -> japan_e2 (set in)
- japan_e2 -> from_2009_e2 (released in)
- japan_e3 -> from_2009_e3 (released in)
- the_secret_of_kells -> japan_e3 (set in)
- japan_e2 -> from_2009_kells_e2 (released in)
- japan_e3 -> from_2009_kells_e3 (released in)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST decomposes each film into director and nationality branches without generating a final comparison question.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- summer_wars: Summer Wars (entity)
- japan_e2: Japan (type_variable)
- japan_e3: Japan (type_variable)
- from_2009_e2: from 2009 (type_variable)
- from_2009_e3: from 2009 (type_variable)
- the_secret_of_kells: The Secret of Kells (entity)
- film_e2: film (value_slot)
- film_e3: film (value_slot)

Edges:
- summer_wars -> japan_e2 (set in)
- japan_e2 -> from_2009_e2 (released in)
- japan_e3 -> from_2009_e3 (released in)
- the_secret_of_kells -> japan_e3 (set in)
- japan_e2 -> film_e2 (film from)
- japan_e3 -> film_e3 (film from)

## 11. Atomic Subquestion DAG
- None: Is Summer Wars set in Japan?
- None: Which animated film from Japan was released in 2009?
- None: What film is from Japan?
- None: Is The Secret of Kells set in Japan?
- None: Which animated film from Japan was released in 2009?
- None: What film is from Japan?
