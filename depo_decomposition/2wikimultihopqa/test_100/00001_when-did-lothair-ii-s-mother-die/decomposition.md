# DEPO Decomposition #1

- Dataset: `2wikimultihopqa`
- Question: When did Lothair Ii's mother die?
- Gold answer: 20 March 851

## 1. Semantic-Normalized Question
When did Lothair II's mother die?

## 2. Mask Spans
- Lothair II's (entity, LothairIIS)

## 3. Selective Masked Question
When did SomeEntityA mother die?

## 4. CoreNLP Dependency Parse
- die[5] --advmod--> When[1]
- die[5] --aux--> did[2]
- mother[4] --compound--> SomeEntityA[3]
- die[5] --nsubj--> mother[4]
- die[5] --punct--> ?[6]

## 5. Undirected Dependency Graph
- When[1] --advmod-- die[5]
- did[2] --aux-- die[5]
- Lothair II's[3] --compound-- mother[4]
- mother[4] --nsubj-- die[5]
- die[5] --punct-- ?[6]

## 6. Entity Start Nodes
- e1: Lothair II's graph_node_ids=['3']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Lothair II's -- mother -- die -- When
- e1_p2 (e1): Lothair II's -- mother -- die
- e1_p3 (e1): Lothair II's -- mother -- die -- did
- e1_p4 (e1): Lothair II's -- mother -- die -- ?
- e1_p5 (e1): Lothair II's -- mother

## 8. LLM Selected Entity Paths
- e1: e1_p1 Lothair II's -- mother -- die -- When
  Reason: This path follows the useful reasoning chain from Lothair II's mother to the action of dying, and includes the temporal question 'When'.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "when",
  "answer_kind": "temporal",
  "answer_slot_hint": "death_date",
  "focus_predicate": "die"
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- lothair_ii: Lothair II (entity)
- mother: mother (type_variable)
- death_date: death_date (value_slot)

Edges:
- lothair_ii -> mother (mother of Lothair II)
- mother -> death_date (date of death of the mother)

## 10. Atomic Subquestion DAG
- None: Who is the mother of Lothair II?
- None: When did the mother of Lothair II die?
