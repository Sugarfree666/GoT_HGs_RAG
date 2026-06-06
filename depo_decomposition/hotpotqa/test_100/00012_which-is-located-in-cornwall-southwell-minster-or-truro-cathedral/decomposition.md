# DEPO Decomposition #12

- Dataset: `hotpotqa`
- Question: Which is located in Cornwall, Southwell Minster or Truro Cathedral?
- Gold answer: The Cathedral of the Blessed Virgin Mary, Truro

## 1. Semantic-Normalized Question
Which is located in Cornwall, Southwell Minster or Truro Cathedral?

## 2. Mask Spans
- Southwell Minster (entity, Location)
- Truro Cathedral (entity, Location)

## 3. Selective Masked Question
Which is located in Cornwall, SomeEntityA or SomeEntityB?

## 4. CoreNLP Dependency Parse
- located[3] --nsubj:pass--> Which[1]
- located[3] --aux:pass--> is[2]
- Cornwall[5] --case--> in[4]
- located[3] --obl:in--> Cornwall[5]
- Cornwall[5] --punct--> ,[6]
- located[3] --obl:in--> SomeEntityA[7]
- Cornwall[5] --conj:or--> SomeEntityA[7]
- SomeEntityB[9] --cc--> or[8]
- located[3] --obl:in--> SomeEntityB[9]
- Cornwall[5] --conj:or--> SomeEntityB[9]
- located[3] --punct--> ?[10]

## 5. Undirected Dependency Graph
- Which[1] --nsubj:pass-- located[3]
- is[2] --aux:pass-- located[3]
- located[3] --obl:in-- Cornwall[5]
- located[3] --obl:in-- Southwell Minster[7]
- located[3] --obl:in-- Truro Cathedral[9]
- located[3] --punct-- ?[10]
- in[4] --case-- Cornwall[5]
- Cornwall[5] --punct-- ,[6]
- Cornwall[5] --conj:or-- Southwell Minster[7]
- Cornwall[5] --conj:or-- Truro Cathedral[9]
- or[8] --cc-- Truro Cathedral[9]

## 6. Entity Start Nodes
- e1: Southwell Minster graph_node_ids=['7']
- e2: Truro Cathedral graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Southwell Minster -- Cornwall -- located -- Which
- e1_p2 (e1): Southwell Minster -- located -- Cornwall
- e1_p3 (e1): Southwell Minster -- Cornwall -- located
- e1_p4 (e1): Southwell Minster -- located -- Cornwall -- in
- e1_p5 (e1): Southwell Minster -- located -- Cornwall -- ,
- e1_p6 (e1): Southwell Minster -- Cornwall -- located -- is
- e1_p7 (e1): Southwell Minster -- Cornwall -- located -- ?
- e1_p8 (e1): Southwell Minster -- located -- Which
- e1_p9 (e1): Southwell Minster -- located
- e1_p10 (e1): Southwell Minster -- Cornwall
- e1_p11 (e1): Southwell Minster -- located -- is
- e1_p12 (e1): Southwell Minster -- located -- ?
- e1_p13 (e1): Southwell Minster -- Cornwall -- in
- e1_p14 (e1): Southwell Minster -- Cornwall -- ,
- e1_p15 (e1): Southwell Minster -- located -- Cornwall -- Truro Cathedral
- e1_p16 (e1): Southwell Minster -- Cornwall -- located -- Truro Cathedral
- e1_p17 (e1): Southwell Minster -- located -- Truro Cathedral
- e1_p18 (e1): Southwell Minster -- Cornwall -- Truro Cathedral
- e1_p19 (e1): Southwell Minster -- Cornwall -- Truro Cathedral -- located -- Which
- e1_p20 (e1): Southwell Minster -- located -- Truro Cathedral -- Cornwall
- e1_p21 (e1): Southwell Minster -- Cornwall -- Truro Cathedral -- located
- e1_p22 (e1): Southwell Minster -- located -- Cornwall -- Truro Cathedral -- or
- e1_p23 (e1): Southwell Minster -- located -- Truro Cathedral -- Cornwall -- in
- e1_p24 (e1): Southwell Minster -- located -- Truro Cathedral -- Cornwall -- ,
- e1_p25 (e1): Southwell Minster -- Cornwall -- located -- Truro Cathedral -- or
- e1_p26 (e1): Southwell Minster -- Cornwall -- Truro Cathedral -- located -- is
- e1_p27 (e1): Southwell Minster -- Cornwall -- Truro Cathedral -- located -- ?
- e1_p28 (e1): Southwell Minster -- located -- Truro Cathedral -- or
- e1_p29 (e1): Southwell Minster -- Cornwall -- Truro Cathedral -- or
- e2_p1 (e2): Truro Cathedral -- Cornwall -- located -- Which
- e2_p2 (e2): Truro Cathedral -- located -- Cornwall
- e2_p3 (e2): Truro Cathedral -- Cornwall -- located
- e2_p4 (e2): Truro Cathedral -- located -- Cornwall -- in
- e2_p5 (e2): Truro Cathedral -- located -- Cornwall -- ,
- e2_p6 (e2): Truro Cathedral -- Cornwall -- located -- is
- e2_p7 (e2): Truro Cathedral -- Cornwall -- located -- ?
- e2_p8 (e2): Truro Cathedral -- located -- Which
- e2_p9 (e2): Truro Cathedral -- located
- e2_p10 (e2): Truro Cathedral -- Cornwall
- e2_p11 (e2): Truro Cathedral -- located -- is
- e2_p12 (e2): Truro Cathedral -- located -- ?
- e2_p13 (e2): Truro Cathedral -- Cornwall -- in
- e2_p14 (e2): Truro Cathedral -- Cornwall -- ,
- e2_p15 (e2): Truro Cathedral -- or
- e2_p16 (e2): Truro Cathedral -- located -- Cornwall -- Southwell Minster
- e2_p17 (e2): Truro Cathedral -- Cornwall -- located -- Southwell Minster
- e2_p18 (e2): Truro Cathedral -- located -- Southwell Minster
- e2_p19 (e2): Truro Cathedral -- Cornwall -- Southwell Minster
- e2_p20 (e2): Truro Cathedral -- Cornwall -- Southwell Minster -- located -- Which
- e2_p21 (e2): Truro Cathedral -- located -- Southwell Minster -- Cornwall
- e2_p22 (e2): Truro Cathedral -- Cornwall -- Southwell Minster -- located
- e2_p23 (e2): Truro Cathedral -- located -- Southwell Minster -- Cornwall -- in
- e2_p24 (e2): Truro Cathedral -- located -- Southwell Minster -- Cornwall -- ,
- e2_p25 (e2): Truro Cathedral -- Cornwall -- Southwell Minster -- located -- is
- e2_p26 (e2): Truro Cathedral -- Cornwall -- Southwell Minster -- located -- ?

## 8. LLM Selected Entity Paths
- e1: e1_p1 Southwell Minster -- Cornwall -- located -- Which
  Reason: This path connects Southwell Minster to Cornwall and leads to the question 'Which' effectively.
- e2: e2_p1 Truro Cathedral -- Cornwall -- located -- Which
  Reason: This path connects Truro Cathedral to Cornwall and leads to the question 'Which' effectively.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "which",
  "answer_kind": "entity_or_attribute",
  "answer_slot_hint": null,
  "focus_predicate": "locate",
  "focus_noun": null,
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- southwell_minster: Southwell Minster (entity)
- cornwall: Cornwall (value_slot)

Edges:
- southwell_minster -> cornwall (located in)

## 10. Atomic Subquestion DAG
- None: Is Southwell Minster located in Cornwall?
