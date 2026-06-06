# DEPO Decomposition #17

- Dataset: `2wikimultihopqa`
- Question: What is the date of death of the director of film Madame La Presidente?
- Gold answer: 10 August 1960

## 1. Semantic-Normalized Question
What is the date of death of the director of the film Madame La Presidente?

## 2. Mask Spans
- Madame La Presidente? (entity, Film)

## 3. Selective Masked Question
What is the date of death of the director of the film MovieA

## 4. CoreNLP Dependency Parse
- What[1] --cop--> is[2]
- date[4] --det--> the[3]
- What[1] --nsubj--> date[4]
- death[6] --case--> of[5]
- date[4] --nmod:of--> death[6]
- director[9] --case--> of[7]
- director[9] --det--> the[8]
- death[6] --nmod:of--> director[9]
- MovieA[13] --case--> of[10]
- MovieA[13] --det--> the[11]
- MovieA[13] --compound--> film[12]
- director[9] --nmod:of--> MovieA[13]

## 5. Undirected Dependency Graph
- What[1] --cop-- is[2]
- What[1] --nsubj-- date[4]
- the[3] --det-- date[4]
- date[4] --nmod:of-- death[6]
- of[5] --case-- death[6]
- death[6] --nmod:of-- director[9]
- of[7] --case-- director[9]
- the[8] --det-- director[9]
- director[9] --nmod:of-- Madame La Presidente?[13]
- of[10] --case-- Madame La Presidente?[13]
- the[11] --det-- Madame La Presidente?[13]
- film[12] --compound-- Madame La Presidente?[13]

## 6. Entity Start Nodes
- e1: Madame La Presidente? graph_node_ids=['13']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Madame La Presidente? -- director -- death -- date -- What
- e1_p2 (e1): Madame La Presidente? -- director -- death -- date -- What -- is
- e1_p3 (e1): Madame La Presidente? -- director -- death -- date
- e1_p4 (e1): Madame La Presidente? -- director -- death -- date -- the
- e1_p5 (e1): Madame La Presidente? -- director -- death
- e1_p6 (e1): Madame La Presidente? -- director -- death -- of
- e1_p7 (e1): Madame La Presidente? -- director
- e1_p8 (e1): Madame La Presidente? -- director -- of
- e1_p9 (e1): Madame La Presidente? -- director -- the
- e1_p10 (e1): Madame La Presidente? -- film
- e1_p11 (e1): Madame La Presidente? -- of
- e1_p12 (e1): Madame La Presidente? -- the

## 8. LLM Selected Entity Paths
- e1: e1_p1 Madame La Presidente? -- director -- death -- date -- What
  Reason: This path provides a complete reasoning chain from the film 'Madame La Presidente?' to its director, then to the director's death, and finally to the date of death, which is the information sought in the question.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "what",
  "answer_kind": "entity_or_attribute",
  "answer_slot_hint": "date",
  "focus_predicate": "die"
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- madame_la_presidente: Madame La Presidente (entity)
- director: director (type_variable)
- death_date: death_date (value_slot)

Edges:
- madame_la_presidente -> director (director of Madame La Presidente)
- director -> death_date (date of death of the director)

## 10. Atomic Subquestion DAG
- None: Who is the director of Madame La Presidente?
- None: When did the director of Madame La Presidente die?
