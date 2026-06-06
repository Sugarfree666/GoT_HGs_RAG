# DEPO Decomposition #1

- Dataset: `hotpotqa`
- Question: what is one of the stars of  The Newcomers known for
- Gold answer: superhero roles as the Marvel Comics

## 1. Semantic-Normalized Question
what is one of the stars of The Newcomers known for?

## 2. Mask Spans
- The Newcomers (entity, Film)

## 3. Selective Masked Question
what is one of the stars of MovieA known for?

## 4. CoreNLP Dependency Parse
- what[1] --cop--> is[2]
- stars[6] --det:qmod--> one[3]
- one[3] --fixed--> of[4]
- stars[6] --det--> the[5]
- known[9] --nsubj--> stars[6]
- MovieA[8] --case--> of[7]
- stars[6] --nmod:of--> MovieA[8]
- what[1] --dep--> known[9]
- known[9] --dep--> for[10]
- what[1] --punct--> ?[11]

## 5. Undirected Dependency Graph
- what[1] --cop-- is[2]
- what[1] --dep-- known[9]
- what[1] --punct-- ?[11]
- one[3] --det:qmod-- stars[6]
- one[3] --fixed-- of[4]
- the[5] --det-- stars[6]
- stars[6] --nsubj-- known[9]
- stars[6] --nmod:of-- The Newcomers[8]
- of[7] --case-- The Newcomers[8]
- known[9] --dep-- for[10]

## 6. Entity Start Nodes
- e1: The Newcomers graph_node_ids=['8']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): The Newcomers -- stars -- known -- what
- e1_p2 (e1): The Newcomers -- stars -- known -- what -- is
- e1_p3 (e1): The Newcomers -- stars -- known -- what -- ?
- e1_p4 (e1): The Newcomers -- stars -- one
- e1_p5 (e1): The Newcomers -- stars -- known
- e1_p6 (e1): The Newcomers -- stars -- one -- of
- e1_p7 (e1): The Newcomers -- stars -- known -- for
- e1_p8 (e1): The Newcomers -- stars
- e1_p9 (e1): The Newcomers -- stars -- the
- e1_p10 (e1): The Newcomers -- of

## 8. LLM Selected Entity Paths
- e1: e1_p7 The Newcomers -- stars -- known -- for
  Reason: This path follows the useful reasoning chain from the entity 'The Newcomers' through 'stars' to 'known for', which directly addresses the question about what one of the stars is known for.

## 9. Selected Path Semantic Transduction
Nodes:
- the_newcomers: The Newcomers (entity)
- stars: stars (type_variable)
- known_for: known_for (value_slot)

Edges:
- the_newcomers -> stars (stars of The Newcomers)
- stars -> known_for (known for)

## 10. Atomic Subquestion DAG
- None: Who are the stars of The Newcomers?
- None: What is one of the stars of The Newcomers known for?
