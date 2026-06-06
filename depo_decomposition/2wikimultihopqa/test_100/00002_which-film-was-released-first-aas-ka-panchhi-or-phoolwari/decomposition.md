# DEPO Decomposition #2

- Dataset: `2wikimultihopqa`
- Question: Which film was released first, Aas Ka Panchhi or Phoolwari?
- Gold answer: Phoolwari

## 1. Semantic-Normalized Question
Which film was released first, Aas Ka Panchhi or Phoolwari?

## 2. Mask Spans
- was released first, Aas Ka Panchhi or Phoolwari? (entity, Film)

## 3. Selective Masked Question
Which film MovieA

## 4. CoreNLP Dependency Parse
- MovieA[3] --det--> Which[1]
- MovieA[3] --compound--> film[2]

## 5. Undirected Dependency Graph
- Which[1] --det-- was released first, Aas Ka Panchhi or Phoolwari?[3]
- film[2] --compound-- was released first, Aas Ka Panchhi or Phoolwari?[3]

## 6. Entity Start Nodes
- e1: was released first, Aas Ka Panchhi or Phoolwari? graph_node_ids=['3']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): was released first, Aas Ka Panchhi or Phoolwari? -- film
- e1_p2 (e1): was released first, Aas Ka Panchhi or Phoolwari? -- Which

## 8. LLM Selected Entity Paths
- e1: e1_p1 was released first, Aas Ka Panchhi or Phoolwari? -- film
  Reason: This path connects the entity to the relevant attribute 'film', which is essential for determining which film was released first.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "which",
  "answer_kind": "comparison_or_boolean",
  "answer_slot_hint": "release_date",
  "focus_predicate": "release",
  "focus_noun": "film",
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- aas_ka_panchhi: Aas Ka Panchhi (entity)
- phoolwari: Phoolwari (entity)
- release_date: release_date (value_slot)

Edges:
- aas_ka_panchhi -> release_date (release date of Aas Ka Panchhi)
- phoolwari -> release_date (release date of Phoolwari)

## 10. Atomic Subquestion DAG
- None: What is the release date of Aas Ka Panchhi?
- None: What is the release date of Phoolwari?
