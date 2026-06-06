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
  Reason: This path directly connects Walt Disney to the question about which film was produced first, without passing through another entity.
- e2: e2_p1 was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? -- film
  Reason: This path connects the question about the film directly to the concept of 'film', which is relevant for determining which was produced first.

## 9. Selected Path Semantic Transduction
Nodes:
- walt_disney: Walt Disney (entity)
- film: film (type_variable)
- release_date: release_date (value_slot)

Edges:
- walt_disney -> film (film produced by Walt Disney)
- film -> release_date (release date of the film)

## 10. Atomic Subquestion DAG
- None: What film was produced by Walt Disney?
- None: What is the release date of the film of Walt Disney?
