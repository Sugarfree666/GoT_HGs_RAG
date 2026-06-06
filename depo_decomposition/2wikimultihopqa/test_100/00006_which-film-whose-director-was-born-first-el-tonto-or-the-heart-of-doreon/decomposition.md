# DEPO Decomposition #6

- Dataset: `2wikimultihopqa`
- Question: Which film whose director was born first, El Tonto or The Heart Of Doreon?
- Gold answer: The Heart Of Doreon

## 1. Semantic-Normalized Question
Which film, whose director was born first, El Tonto or The Heart Of Doreon?

## 2. Mask Spans
- El Tonto (entity, Film)
- The Heart Of Doreon (entity, Film)

## 3. Selective Masked Question
Which film, whose director was born first, MovieA or MovieB?

## 4. CoreNLP Dependency Parse
- film[2] --det--> Which[1]
- film[2] --punct--> ,[3]
- director[5] --nmod:poss--> whose[4]
- born[7] --nsubj:pass--> director[5]
- born[7] --aux:pass--> was[6]
- film[2] --dep--> born[7]
- born[7] --advmod--> first[8]
- born[7] --punct--> ,[9]
- born[7] --obj--> MovieA[10]
- MovieB[12] --cc--> or[11]
- born[7] --obj--> MovieB[12]
- MovieA[10] --conj:or--> MovieB[12]
- film[2] --punct--> ?[13]

## 5. Undirected Dependency Graph
- Which[1] --det-- film[2]
- film[2] --punct-- ,[3]
- film[2] --dep-- born[7]
- film[2] --punct-- ?[13]
- whose[4] --nmod:poss-- director[5]
- director[5] --nsubj:pass-- born[7]
- was[6] --aux:pass-- born[7]
- born[7] --advmod-- first[8]
- born[7] --punct-- ,[9]
- born[7] --obj-- El Tonto[10]
- born[7] --obj-- The Heart Of Doreon[12]
- El Tonto[10] --conj:or-- The Heart Of Doreon[12]
- or[11] --cc-- The Heart Of Doreon[12]

## 6. Entity Start Nodes
- e1: El Tonto graph_node_ids=['10']
- e2: The Heart Of Doreon graph_node_ids=['12']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): El Tonto -- born -- film -- Which
- e1_p2 (e1): El Tonto -- born -- director
- e1_p3 (e1): El Tonto -- born -- director -- whose
- e1_p4 (e1): El Tonto -- born -- film
- e1_p5 (e1): El Tonto -- born -- first
- e1_p6 (e1): El Tonto -- born -- film -- ,
- e1_p7 (e1): El Tonto -- born -- film -- ?
- e1_p8 (e1): El Tonto -- born
- e1_p9 (e1): El Tonto -- born -- was
- e1_p10 (e1): El Tonto -- born -- ,
- e1_p11 (e1): El Tonto -- born -- The Heart Of Doreon
- e1_p12 (e1): El Tonto -- The Heart Of Doreon
- e1_p13 (e1): El Tonto -- The Heart Of Doreon -- born -- film -- Which
- e1_p14 (e1): El Tonto -- The Heart Of Doreon -- born -- director
- e1_p15 (e1): El Tonto -- The Heart Of Doreon -- born -- director -- whose
- e1_p16 (e1): El Tonto -- The Heart Of Doreon -- born -- film
- e1_p17 (e1): El Tonto -- The Heart Of Doreon -- born -- first
- e1_p18 (e1): El Tonto -- The Heart Of Doreon -- born -- film -- ,
- e1_p19 (e1): El Tonto -- The Heart Of Doreon -- born -- film -- ?
- e1_p20 (e1): El Tonto -- The Heart Of Doreon -- born
- e1_p21 (e1): El Tonto -- born -- The Heart Of Doreon -- or
- e1_p22 (e1): El Tonto -- The Heart Of Doreon -- born -- was
- e1_p23 (e1): El Tonto -- The Heart Of Doreon -- born -- ,
- e1_p24 (e1): El Tonto -- The Heart Of Doreon -- or
- e2_p1 (e2): The Heart Of Doreon -- born -- film -- Which
- e2_p2 (e2): The Heart Of Doreon -- born -- director
- e2_p3 (e2): The Heart Of Doreon -- born -- director -- whose
- e2_p4 (e2): The Heart Of Doreon -- born -- film
- e2_p5 (e2): The Heart Of Doreon -- born -- first
- e2_p6 (e2): The Heart Of Doreon -- born -- film -- ,
- e2_p7 (e2): The Heart Of Doreon -- born -- film -- ?
- e2_p8 (e2): The Heart Of Doreon -- born
- e2_p9 (e2): The Heart Of Doreon -- born -- was
- e2_p10 (e2): The Heart Of Doreon -- born -- ,
- e2_p11 (e2): The Heart Of Doreon -- or
- e2_p12 (e2): The Heart Of Doreon -- born -- El Tonto
- e2_p13 (e2): The Heart Of Doreon -- El Tonto
- e2_p14 (e2): The Heart Of Doreon -- El Tonto -- born -- film -- Which
- e2_p15 (e2): The Heart Of Doreon -- El Tonto -- born -- director
- e2_p16 (e2): The Heart Of Doreon -- El Tonto -- born -- director -- whose
- e2_p17 (e2): The Heart Of Doreon -- El Tonto -- born -- film
- e2_p18 (e2): The Heart Of Doreon -- El Tonto -- born -- first
- e2_p19 (e2): The Heart Of Doreon -- El Tonto -- born -- film -- ,
- e2_p20 (e2): The Heart Of Doreon -- El Tonto -- born -- film -- ?
- e2_p21 (e2): The Heart Of Doreon -- El Tonto -- born
- e2_p22 (e2): The Heart Of Doreon -- El Tonto -- born -- was
- e2_p23 (e2): The Heart Of Doreon -- El Tonto -- born -- ,

## 8. LLM Selected Entity Paths
- e1: e1_p5 El Tonto -- born -- first
  Reason: This path connects 'El Tonto' to the attribute 'first' through the 'born' relationship, which is essential for comparing the birth order of directors.
- e2: e2_p5 The Heart Of Doreon -- born -- first
  Reason: This path connects 'The Heart Of Doreon' to the attribute 'first' through the 'born' relationship, which is essential for comparing the birth order of directors.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "who",
  "answer_kind": "comparison_or_boolean",
  "answer_slot_hint": "date",
  "focus_predicate": "born",
  "focus_noun": "film",
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- el_tonto: El Tonto (entity)
- director_r1: director (type_variable)
- birth_date_r1: birth_date (value_slot)

Edges:
- el_tonto -> director_r1 (director of El Tonto)
- director_r1 -> birth_date_r1 (birth date of the director)

## 10. Atomic Subquestion DAG
- None: Who is the director of El Tonto?
- None: When was the birth date of the director of El Tonto?
