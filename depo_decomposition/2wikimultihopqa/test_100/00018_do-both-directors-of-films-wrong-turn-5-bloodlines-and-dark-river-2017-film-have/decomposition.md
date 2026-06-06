# DEPO Decomposition #18

- Dataset: `2wikimultihopqa`
- Question: Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?
- Gold answer: no

## 1. Semantic-Normalized Question
Do both directors of the films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?

## 2. Mask Spans
- Wrong Turn 5: Bloodlines (entity, Film)
- Dark River (2017 Film) (entity, Film)

## 3. Selective Masked Question
Do both directors of the films MovieA and MovieB have the same nationality?

## 4. CoreNLP Dependency Parse
- have[10] --aux--> Do[1]
- directors[3] --det--> both[2]
- have[10] --nsubj--> directors[3]
- MovieA[7] --case--> of[4]
- MovieA[7] --det--> the[5]
- MovieA[7] --compound--> films[6]
- directors[3] --nmod:of--> MovieA[7]
- MovieB[9] --cc--> and[8]
- directors[3] --conj:and--> MovieB[9]
- have[10] --nsubj--> MovieB[9]
- nationality[13] --det--> the[11]
- nationality[13] --amod--> same[12]
- have[10] --obj--> nationality[13]
- have[10] --punct--> ?[14]

## 5. Undirected Dependency Graph
- Do[1] --aux-- have[10]
- both[2] --det-- directors[3]
- directors[3] --nsubj-- have[10]
- directors[3] --nmod:of-- Wrong Turn 5: Bloodlines[7]
- directors[3] --conj:and-- Dark River (2017 Film)[9]
- of[4] --case-- Wrong Turn 5: Bloodlines[7]
- the[5] --det-- Wrong Turn 5: Bloodlines[7]
- films[6] --compound-- Wrong Turn 5: Bloodlines[7]
- and[8] --cc-- Dark River (2017 Film)[9]
- Dark River (2017 Film)[9] --nsubj-- have[10]
- have[10] --obj-- nationality[13]
- have[10] --punct-- ?[14]
- the[11] --det-- nationality[13]
- same[12] --amod-- nationality[13]

## 6. Entity Start Nodes
- e1: Wrong Turn 5: Bloodlines graph_node_ids=['7']
- e2: Dark River (2017 Film) graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Wrong Turn 5: Bloodlines -- directors -- have -- nationality
- e1_p2 (e1): Wrong Turn 5: Bloodlines -- directors -- have -- nationality -- the
- e1_p3 (e1): Wrong Turn 5: Bloodlines -- directors -- have -- nationality -- same
- e1_p4 (e1): Wrong Turn 5: Bloodlines -- directors -- both
- e1_p5 (e1): Wrong Turn 5: Bloodlines -- directors -- have
- e1_p6 (e1): Wrong Turn 5: Bloodlines -- directors -- have -- Do
- e1_p7 (e1): Wrong Turn 5: Bloodlines -- directors -- have -- ?
- e1_p8 (e1): Wrong Turn 5: Bloodlines -- directors
- e1_p9 (e1): Wrong Turn 5: Bloodlines -- films
- e1_p10 (e1): Wrong Turn 5: Bloodlines -- of
- e1_p11 (e1): Wrong Turn 5: Bloodlines -- the
- e1_p12 (e1): Wrong Turn 5: Bloodlines -- directors -- have -- Dark River (2017 Film)
- e1_p13 (e1): Wrong Turn 5: Bloodlines -- directors -- Dark River (2017 Film)
- e1_p14 (e1): Wrong Turn 5: Bloodlines -- directors -- Dark River (2017 Film) -- have -- nationality
- e1_p15 (e1): Wrong Turn 5: Bloodlines -- directors -- Dark River (2017 Film) -- have -- nationality -- the
- e1_p16 (e1): Wrong Turn 5: Bloodlines -- directors -- Dark River (2017 Film) -- have -- nationality -- same
- e1_p17 (e1): Wrong Turn 5: Bloodlines -- directors -- Dark River (2017 Film) -- have
- e1_p18 (e1): Wrong Turn 5: Bloodlines -- directors -- Dark River (2017 Film) -- have -- Do
- e1_p19 (e1): Wrong Turn 5: Bloodlines -- directors -- Dark River (2017 Film) -- have -- ?
- e1_p20 (e1): Wrong Turn 5: Bloodlines -- directors -- have -- Dark River (2017 Film) -- and
- e1_p21 (e1): Wrong Turn 5: Bloodlines -- directors -- Dark River (2017 Film) -- and
- e2_p1 (e2): Dark River (2017 Film) -- directors -- have -- nationality
- e2_p2 (e2): Dark River (2017 Film) -- directors -- have -- nationality -- the
- e2_p3 (e2): Dark River (2017 Film) -- directors -- have -- nationality -- same
- e2_p4 (e2): Dark River (2017 Film) -- have -- directors -- both
- e2_p5 (e2): Dark River (2017 Film) -- have -- nationality
- e2_p6 (e2): Dark River (2017 Film) -- have -- nationality -- the
- e2_p7 (e2): Dark River (2017 Film) -- have -- nationality -- same
- e2_p8 (e2): Dark River (2017 Film) -- directors -- both
- e2_p9 (e2): Dark River (2017 Film) -- directors -- have
- e2_p10 (e2): Dark River (2017 Film) -- have -- directors
- e2_p11 (e2): Dark River (2017 Film) -- directors -- have -- Do
- e2_p12 (e2): Dark River (2017 Film) -- directors -- have -- ?
- e2_p13 (e2): Dark River (2017 Film) -- directors
- e2_p14 (e2): Dark River (2017 Film) -- have
- e2_p15 (e2): Dark River (2017 Film) -- have -- Do
- e2_p16 (e2): Dark River (2017 Film) -- have -- ?
- e2_p17 (e2): Dark River (2017 Film) -- and
- e2_p18 (e2): Dark River (2017 Film) -- have -- directors -- Wrong Turn 5: Bloodlines
- e2_p19 (e2): Dark River (2017 Film) -- directors -- Wrong Turn 5: Bloodlines
- e2_p20 (e2): Dark River (2017 Film) -- have -- directors -- Wrong Turn 5: Bloodlines -- films
- e2_p21 (e2): Dark River (2017 Film) -- directors -- Wrong Turn 5: Bloodlines -- films
- e2_p22 (e2): Dark River (2017 Film) -- have -- directors -- Wrong Turn 5: Bloodlines -- of
- e2_p23 (e2): Dark River (2017 Film) -- have -- directors -- Wrong Turn 5: Bloodlines -- the
- e2_p24 (e2): Dark River (2017 Film) -- directors -- Wrong Turn 5: Bloodlines -- of
- e2_p25 (e2): Dark River (2017 Film) -- directors -- Wrong Turn 5: Bloodlines -- the

## 8. LLM Selected Entity Paths
- e1: e1_p3 Wrong Turn 5: Bloodlines -- directors -- have -- nationality -- same
  Reason: This path connects the film 'Wrong Turn 5: Bloodlines' to its directors and their nationality, which is essential for comparing with the other film.
- e2: e2_p3 Dark River (2017 Film) -- directors -- have -- nationality -- same
  Reason: This path connects the film 'Dark River (2017 Film)' to its directors and their nationality, which is essential for comparing with the other film.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": null,
  "answer_kind": "comparison_or_boolean",
  "answer_slot_hint": "nationality",
  "focus_predicate": null,
  "focus_noun": null,
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- wrong_turn_5_bloodlines: Wrong Turn 5: Bloodlines (entity)
- director_r1: directors (type_variable)
- nationality_r1: nationality (value_slot)
- dark_river: Dark River (2017 Film) (entity)
- director_r2: directors (type_variable)
- nationality_r2: nationality (value_slot)

Edges:
- wrong_turn_5_bloodlines -> director_r1 (directors of Wrong Turn 5: Bloodlines)
- director_r1 -> nationality_r1 (nationality of the directors)
- dark_river -> director_r2 (directors of Dark River (2017 Film))
- director_r2 -> nationality_r2 (nationality of the directors)

## 10. Atomic Subquestion DAG
- None: Who is the director of Wrong Turn 5: Bloodlines?
- None: What is the nationality of the director of Wrong Turn 5: Bloodlines?
- None: Who is the director of Dark River (2017 Film)?
- None: What is the nationality of the director of Dark River (2017 Film)?
