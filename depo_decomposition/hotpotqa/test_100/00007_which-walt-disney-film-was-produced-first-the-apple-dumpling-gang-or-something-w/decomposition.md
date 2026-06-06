# DEPO Decomposition #7

- Dataset: `hotpotqa`
- Question: Which Walt Disney film was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?
- Gold answer: The Apple Dumpling Gang

## 1. Semantic-Normalized Question
Which Walt Disney film was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?

## 2. Mask Spans
- Walt Disney (entity, WaltDisney)
- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? (entity, Film)

## 3. Selective Masked Question
Which SomeEntityA film MovieA

## 4. CoreNLP Dependency Parse
- MovieA[4] --det--> Which[1]
- MovieA[4] --compound--> SomeEntityA[2]
- MovieA[4] --compound--> film[3]

## 5. Undirected Dependency Graph
- Which[1] --det-- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?[4]
- Walt Disney[2] --compound-- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?[4]
- film[3] --compound-- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?[4]

## 6. Entity Start Nodes
- e1: Walt Disney graph_node_ids=['2']
- e2: was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Walt Disney -- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?
- e1_p2 (e1): Walt Disney -- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? -- film
- e1_p3 (e1): Walt Disney -- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? -- Which
- e2_p1 (e2): was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? -- film
- e2_p2 (e2): was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? -- Which
- e2_p3 (e2): was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? -- Walt Disney

## 8. LLM Selected Entity Paths
- e1: e1_p1 Walt Disney -- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?
  Reason: This path directly connects Walt Disney to the question about the films, providing a clear reasoning chain.
- e2: e2_p1 was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? -- film
  Reason: This path connects the question about the films directly to the concept of 'film', which is essential for answering the question.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "which",
  "answer_kind": "comparison_or_boolean",
  "answer_slot_hint": "date",
  "focus_predicate": null,
  "focus_noun": "walt",
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- walt_disney: Walt Disney (entity)
- film_a: The Apple Dumpling Gang (entity)
- film_b: Something Wicked This Way Comes (entity)
- release_date_e1: release_date (value_slot)
- release_date_e2: release_date (value_slot)

Edges:
- walt_disney -> film_a (produced)
- walt_disney -> film_b (produced)
- film_a -> release_date_e1 (release_date of The Apple Dumpling Gang)
- film_a -> release_date_e2 (release_date of The Apple Dumpling Gang)
- film_b -> release_date_e1 (release_date of Something Wicked This Way Comes)

## 10. Atomic Subquestion DAG
- None: What film did Walt Disney produce called The Apple Dumpling Gang?
- None: What is the release date of The Apple Dumpling Gang?
- None: What is the release date of The Apple Dumpling Gang?
- None: What film was produced by Walt Disney titled Something Wicked This Way Comes?
- None: What is the release date of Something Wicked This Way Comes?
