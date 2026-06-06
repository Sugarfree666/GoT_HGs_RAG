# DEPO Decomposition #16

- Dataset: `hotpotqa`
- Question: Are Steve Perry and Leslie West both singers?
- Gold answer: yes

## 1. Semantic-Normalized Question
Are Steve Perry and Leslie West both singers?

## 2. Mask Spans
- Steve Perry (entity, Person)
- Leslie West (entity, Person)

## 3. Selective Masked Question
Are PersonA and PersonB both singers?

## 4. CoreNLP Dependency Parse
- singers[6] --cop--> Are[1]
- singers[6] --nsubj--> PersonA[2]
- PersonB[4] --cc--> and[3]
- PersonA[2] --conj:and--> PersonB[4]
- singers[6] --nsubj--> PersonB[4]
- singers[6] --dep--> both[5]
- singers[6] --punct--> ?[7]

## 5. Undirected Dependency Graph
- Are[1] --cop-- singers[6]
- Steve Perry[2] --nsubj-- singers[6]
- Steve Perry[2] --conj:and-- Leslie West[4]
- and[3] --cc-- Leslie West[4]
- Leslie West[4] --nsubj-- singers[6]
- both[5] --dep-- singers[6]
- singers[6] --punct-- ?[7]

## 6. Entity Start Nodes
- e1: Steve Perry graph_node_ids=['2']
- e2: Leslie West graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Steve Perry -- singers -- both
- e1_p2 (e1): Steve Perry -- singers
- e1_p3 (e1): Steve Perry -- singers -- Are
- e1_p4 (e1): Steve Perry -- singers -- ?
- e1_p5 (e1): Steve Perry -- singers -- Leslie West
- e1_p6 (e1): Steve Perry -- Leslie West
- e1_p7 (e1): Steve Perry -- Leslie West -- singers -- both
- e1_p8 (e1): Steve Perry -- Leslie West -- singers
- e1_p9 (e1): Steve Perry -- Leslie West -- singers -- Are
- e1_p10 (e1): Steve Perry -- Leslie West -- singers -- ?
- e1_p11 (e1): Steve Perry -- singers -- Leslie West -- and
- e1_p12 (e1): Steve Perry -- Leslie West -- and
- e2_p1 (e2): Leslie West -- singers -- both
- e2_p2 (e2): Leslie West -- singers
- e2_p3 (e2): Leslie West -- singers -- Are
- e2_p4 (e2): Leslie West -- singers -- ?
- e2_p5 (e2): Leslie West -- and
- e2_p6 (e2): Leslie West -- singers -- Steve Perry
- e2_p7 (e2): Leslie West -- Steve Perry
- e2_p8 (e2): Leslie West -- Steve Perry -- singers -- both
- e2_p9 (e2): Leslie West -- Steve Perry -- singers
- e2_p10 (e2): Leslie West -- Steve Perry -- singers -- Are
- e2_p11 (e2): Leslie West -- Steve Perry -- singers -- ?

## 8. LLM Selected Entity Paths
- e1: e1_p5 Steve Perry -- singers -- Leslie West
  Reason: This path connects Steve Perry directly to the shared attribute 'singers' without passing through another entity.
- e2: e2_p6 Leslie West -- singers -- Steve Perry
  Reason: This path connects Leslie West directly to the shared attribute 'singers' without passing through another entity.

## 9. Selected Path Semantic Transduction
Nodes:
- steve_perry: Steve Perry (entity)
- singers: singers (type_variable)
- leslie_west: Leslie West (entity)
- leslie_west_singers: singers (type_variable)

Edges:
- steve_perry -> singers (profession of Steve Perry)
- leslie_west -> leslie_west_singers (profession of Leslie West)

## 10. Atomic Subquestion DAG
- None: What is the profession of Steve Perry?
- None: Is Leslie West a singer?
