# DEPO Decomposition #11

- Dataset: `2wikimultihopqa`
- Question: What is the place of birth of the director of film Gaby: A True Story?
- Gold answer: Mexico City

## 1. Semantic-Normalized Question
What is the place of birth of the director of the film Gaby: A True Story?

## 2. Mask Spans
- Gaby: A True Story? (entity, Film)

## 3. Selective Masked Question
What is the place of birth of the director of the film MovieA

## 4. CoreNLP Dependency Parse
- What[1] --cop--> is[2]
- place[4] --det--> the[3]
- What[1] --nsubj--> place[4]
- birth[6] --case--> of[5]
- place[4] --nmod:of--> birth[6]
- director[9] --case--> of[7]
- director[9] --det--> the[8]
- birth[6] --nmod:of--> director[9]
- MovieA[13] --case--> of[10]
- MovieA[13] --det--> the[11]
- MovieA[13] --compound--> film[12]
- director[9] --nmod:of--> MovieA[13]

## 5. Undirected Dependency Graph
- What[1] --cop-- is[2]
- What[1] --nsubj-- place[4]
- the[3] --det-- place[4]
- place[4] --nmod:of-- birth[6]
- of[5] --case-- birth[6]
- birth[6] --nmod:of-- director[9]
- of[7] --case-- director[9]
- the[8] --det-- director[9]
- director[9] --nmod:of-- Gaby: A True Story?[13]
- of[10] --case-- Gaby: A True Story?[13]
- the[11] --det-- Gaby: A True Story?[13]
- film[12] --compound-- Gaby: A True Story?[13]

## 6. Entity Start Nodes
- e1: Gaby: A True Story? graph_node_ids=['13']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Gaby: A True Story? -- director -- birth -- place -- What
- e1_p2 (e1): Gaby: A True Story? -- director -- birth -- place -- What -- is
- e1_p3 (e1): Gaby: A True Story? -- director -- birth -- place
- e1_p4 (e1): Gaby: A True Story? -- director -- birth -- place -- the
- e1_p5 (e1): Gaby: A True Story? -- director -- birth
- e1_p6 (e1): Gaby: A True Story? -- director -- birth -- of
- e1_p7 (e1): Gaby: A True Story? -- director
- e1_p8 (e1): Gaby: A True Story? -- director -- of
- e1_p9 (e1): Gaby: A True Story? -- director -- the
- e1_p10 (e1): Gaby: A True Story? -- film
- e1_p11 (e1): Gaby: A True Story? -- of
- e1_p12 (e1): Gaby: A True Story? -- the

## 8. LLM Selected Entity Paths
- e1: e1_p1 Gaby: A True Story? -- director -- birth -- place -- What
  Reason: This path provides a complete reasoning chain from the film to its director and then to the place of birth, which is the final answer slot.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "what",
  "answer_kind": "entity_or_attribute",
  "answer_slot_hint": null,
  "focus_predicate": "born",
  "focus_noun": null,
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- gaby_a_true_story: Gaby: A True Story (entity)
- director: director (type_variable)
- birthplace: birthplace (value_slot)

Edges:
- gaby_a_true_story -> director (director of Gaby: A True Story)
- director -> birthplace (place of birth of the director)

## 10. Atomic Subquestion DAG
- None: Who is the director of Gaby: A True Story?
- None: What is the birthplace of the director of Gaby: A True Story?
