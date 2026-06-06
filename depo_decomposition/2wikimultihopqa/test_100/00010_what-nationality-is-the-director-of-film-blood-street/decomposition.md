# DEPO Decomposition #10

- Dataset: `2wikimultihopqa`
- Question: What nationality is the director of film Blood Street?
- Gold answer: Chinese

## 1. Semantic-Normalized Question
What nationality is the director of the film Blood Street?

## 2. Mask Spans
- Blood Street? (entity, Film)

## 3. Selective Masked Question
What nationality is the director of the film MovieA

## 4. CoreNLP Dependency Parse
- nationality[2] --det--> What[1]
- is[3] --obj--> nationality[2]
- director[5] --det--> the[4]
- is[3] --nsubj--> director[5]
- MovieA[9] --case--> of[6]
- MovieA[9] --det--> the[7]
- MovieA[9] --compound--> film[8]
- director[5] --nmod:of--> MovieA[9]

## 5. Undirected Dependency Graph
- What[1] --det-- nationality[2]
- nationality[2] --obj-- is[3]
- is[3] --nsubj-- director[5]
- the[4] --det-- director[5]
- director[5] --nmod:of-- Blood Street?[9]
- of[6] --case-- Blood Street?[9]
- the[7] --det-- Blood Street?[9]
- film[8] --compound-- Blood Street?[9]

## 6. Entity Start Nodes
- e1: Blood Street? graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Blood Street? -- director -- is -- nationality -- What
- e1_p2 (e1): Blood Street? -- director -- is -- nationality
- e1_p3 (e1): Blood Street? -- director
- e1_p4 (e1): Blood Street? -- director -- is
- e1_p5 (e1): Blood Street? -- director -- the
- e1_p6 (e1): Blood Street? -- film
- e1_p7 (e1): Blood Street? -- of
- e1_p8 (e1): Blood Street? -- the

## 8. LLM Selected Entity Paths
- e1: e1_p1 Blood Street? -- director -- is -- nationality -- What
  Reason: This path provides a complete reasoning chain from the film 'Blood Street' to its director and then to the nationality, which is the final answer slot.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "what",
  "answer_kind": "entity_or_attribute",
  "answer_slot_hint": "nationality",
  "focus_predicate": null,
  "focus_noun": "nationality",
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- blood_street: Blood Street (entity)
- director: director (type_variable)
- nationality: nationality (value_slot)

Edges:
- blood_street -> director (director of Blood Street)
- director -> nationality (nationality of the director)

## 10. Atomic Subquestion DAG
- None: Who is the director of Blood Street?
- None: What is the nationality of the director of Blood Street?
