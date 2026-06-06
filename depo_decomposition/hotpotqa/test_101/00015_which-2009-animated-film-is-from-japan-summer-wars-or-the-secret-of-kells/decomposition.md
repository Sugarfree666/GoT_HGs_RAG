# DEPO Decomposition #15

- Dataset: `hotpotqa`
- Question: Which 2009 animated film is from Japan, Summer Wars or The Secret of Kells?
- Gold answer: Summer Wars

## 1. Semantic-Normalized Question
Which 2009 animated film is from Japan, Summer Wars or The Secret of Kells?

## 2. Mask Spans
- is from Japan, Summer Wars or The Secret of Kells? (entity, Film)

## 3. Selective Masked Question
Which 2009 animated film MovieA

## 4. CoreNLP Dependency Parse
- MovieA[5] --det--> Which[1]
- MovieA[5] --nummod--> 2009[2]
- MovieA[5] --amod--> animated[3]
- MovieA[5] --compound--> film[4]

## 5. Undirected Dependency Graph
- Which[1] --det-- is from Japan, Summer Wars or The Secret of Kells?[5]
- 2009[2] --nummod-- is from Japan, Summer Wars or The Secret of Kells?[5]
- animated[3] --amod-- is from Japan, Summer Wars or The Secret of Kells?[5]
- film[4] --compound-- is from Japan, Summer Wars or The Secret of Kells?[5]

## 6. Entity Start Nodes
- e1: is from Japan, Summer Wars or The Secret of Kells? graph_node_ids=['5']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): is from Japan, Summer Wars or The Secret of Kells? -- 2009
- e1_p2 (e1): is from Japan, Summer Wars or The Secret of Kells? -- animated
- e1_p3 (e1): is from Japan, Summer Wars or The Secret of Kells? -- film
- e1_p4 (e1): is from Japan, Summer Wars or The Secret of Kells? -- Which

## 8. LLM Selected Entity Paths
- e1: e1_p3 is from Japan, Summer Wars or The Secret of Kells? -- film
  Reason: This path connects the entity to the relevant attribute 'film', which is crucial for determining the answer to the question about the animated films.

## 9. Selected Path Semantic Transduction
Nodes:
- summer_wars: Summer Wars (entity)
- film: film (type_variable)
- the_secret_of_kells: The Secret of Kells (entity)
- release_date: release_date (value_slot)

Edges:
- summer_wars -> film (film of Summer Wars)
- the_secret_of_kells -> film (film of The Secret of Kells)

## 10. Atomic Subquestion DAG
- None: What is the film of Summer Wars?
- None: What is the film of The Secret of Kells?
