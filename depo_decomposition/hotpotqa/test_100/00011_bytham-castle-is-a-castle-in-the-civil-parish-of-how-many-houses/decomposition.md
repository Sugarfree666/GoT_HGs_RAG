# DEPO Decomposition #11

- Dataset: `hotpotqa`
- Question: Bytham Castle is a castle in the civil parish of how many houses?
- Gold answer: 300

## 1. Semantic-Normalized Question
In the civil parish of Bytham Castle, how many houses are there?

## 2. Mask Spans
- Bytham Castle (entity, BythamCastle)

## 3. Selective Masked Question
In the civil parish of SomeEntityA, how many houses are there?

## 4. CoreNLP Dependency Parse
- parish[4] --case--> In[1]
- parish[4] --det--> the[2]
- parish[4] --amod--> civil[3]
- are[11] --obl:in--> parish[4]
- SomeEntityA[6] --case--> of[5]
- parish[4] --nmod:of--> SomeEntityA[6]
- are[11] --punct--> ,[7]
- many[9] --advmod--> how[8]
- houses[10] --amod--> many[9]
- are[11] --nsubj--> houses[10]
- are[11] --expl--> there[12]
- are[11] --punct--> ?[13]

## 5. Undirected Dependency Graph
- In[1] --case-- parish[4]
- the[2] --det-- parish[4]
- civil[3] --amod-- parish[4]
- parish[4] --obl:in-- are[11]
- parish[4] --nmod:of-- Bytham Castle[6]
- of[5] --case-- Bytham Castle[6]
- ,[7] --punct-- are[11]
- how[8] --advmod-- many[9]
- many[9] --amod-- houses[10]
- houses[10] --nsubj-- are[11]
- are[11] --expl-- there[12]
- are[11] --punct-- ?[13]

## 6. Entity Start Nodes
- e1: Bytham Castle graph_node_ids=['6']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Bytham Castle -- parish -- are -- houses -- many -- how
- e1_p2 (e1): Bytham Castle -- parish -- are -- houses -- many
- e1_p3 (e1): Bytham Castle -- parish -- civil
- e1_p4 (e1): Bytham Castle -- parish -- are -- houses
- e1_p5 (e1): Bytham Castle -- parish -- are -- there
- e1_p6 (e1): Bytham Castle -- parish
- e1_p7 (e1): Bytham Castle -- parish -- In
- e1_p8 (e1): Bytham Castle -- parish -- the
- e1_p9 (e1): Bytham Castle -- parish -- are
- e1_p10 (e1): Bytham Castle -- parish -- are -- ,
- e1_p11 (e1): Bytham Castle -- parish -- are -- ?
- e1_p12 (e1): Bytham Castle -- of

## 8. LLM Selected Entity Paths
- e1: e1_p1 Bytham Castle -- parish -- are -- houses -- many -- how
  Reason: This path provides a comprehensive reasoning chain from Bytham Castle to the final answer about the number of houses in the parish.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "how many",
  "answer_kind": "count",
  "answer_slot_hint": "count",
  "focus_predicate": null,
  "focus_noun": null,
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- bytham_castle: Bytham Castle (entity)
- parish: parish (type_variable)
- houses: houses (type_variable)
- count: count (value_slot)

Edges:
- bytham_castle -> parish (parish of Bytham Castle)
- parish -> houses (are in parish)
- houses -> count (number of houses)

## 10. Atomic Subquestion DAG
- None: What is the parish of Bytham Castle?
- None: How many houses are in the parish of Bytham Castle?
- None: How many houses are there in the parish of Bytham Castle?
