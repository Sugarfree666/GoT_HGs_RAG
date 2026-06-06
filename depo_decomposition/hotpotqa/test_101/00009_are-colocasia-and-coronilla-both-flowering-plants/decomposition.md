# DEPO Decomposition #9

- Dataset: `hotpotqa`
- Question: Are Colocasia and Coronilla both flowering plants?
- Gold answer: yes

## 1. Semantic-Normalized Question
Are Colocasia and Coronilla both flowering plants?

## 2. Mask Spans
(none)

## 3. Selective Masked Question
Are Colocasia and Coronilla both flowering plants?

## 4. CoreNLP Dependency Parse
- plants[7] --cop--> Are[1]
- plants[7] --nsubj--> Colocasia[2]
- Coronilla[4] --cc--> and[3]
- Colocasia[2] --conj:and--> Coronilla[4]
- plants[7] --nsubj--> Coronilla[4]
- plants[7] --det--> both[5]
- plants[7] --compound--> flowering[6]
- plants[7] --punct--> ?[8]

## 5. Undirected Dependency Graph
- Are[1] --cop-- plants[7]
- Colocasia[2] --nsubj-- plants[7]
- Colocasia[2] --conj:and-- Coronilla[4]
- and[3] --cc-- Coronilla[4]
- Coronilla[4] --nsubj-- plants[7]
- both[5] --det-- plants[7]
- flowering[6] --compound-- plants[7]
- plants[7] --punct-- ?[8]

## 6. Entity Start Nodes
- e1: Colocasia graph_node_ids=['2']
- e2: Coronilla graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Colocasia -- plants -- both
- e1_p2 (e1): Colocasia -- plants -- flowering
- e1_p3 (e1): Colocasia -- plants
- e1_p4 (e1): Colocasia -- plants -- Are
- e1_p5 (e1): Colocasia -- plants -- ?
- e1_p6 (e1): Colocasia -- plants -- Coronilla
- e1_p7 (e1): Colocasia -- Coronilla
- e1_p8 (e1): Colocasia -- Coronilla -- plants -- both
- e1_p9 (e1): Colocasia -- Coronilla -- plants -- flowering
- e1_p10 (e1): Colocasia -- Coronilla -- plants
- e1_p11 (e1): Colocasia -- Coronilla -- plants -- Are
- e1_p12 (e1): Colocasia -- Coronilla -- plants -- ?
- e1_p13 (e1): Colocasia -- plants -- Coronilla -- and
- e1_p14 (e1): Colocasia -- Coronilla -- and
- e2_p1 (e2): Coronilla -- plants -- both
- e2_p2 (e2): Coronilla -- plants -- flowering
- e2_p3 (e2): Coronilla -- plants
- e2_p4 (e2): Coronilla -- plants -- Are
- e2_p5 (e2): Coronilla -- plants -- ?
- e2_p6 (e2): Coronilla -- and
- e2_p7 (e2): Coronilla -- plants -- Colocasia
- e2_p8 (e2): Coronilla -- Colocasia
- e2_p9 (e2): Coronilla -- Colocasia -- plants -- both
- e2_p10 (e2): Coronilla -- Colocasia -- plants -- flowering
- e2_p11 (e2): Coronilla -- Colocasia -- plants
- e2_p12 (e2): Coronilla -- Colocasia -- plants -- Are
- e2_p13 (e2): Coronilla -- Colocasia -- plants -- ?

## 8. LLM Selected Entity Paths
- e1: e1_p6 Colocasia -- plants -- Coronilla
  Reason: This path connects Colocasia directly to the shared attribute 'plants' without passing through another entity.
- e2: e2_p7 Coronilla -- plants -- Colocasia
  Reason: This path connects Coronilla directly to the shared attribute 'plants' without passing through another entity.

## 9. Selected Path Semantic Transduction
Nodes:
- colocasia: Colocasia (entity)
- plants_e1: plants (type_variable)
- plants_e2: plants (type_variable)
- coronilla: Coronilla (entity)
- flowering_plants: flowering_plants (value_slot)

Edges:
- colocasia -> plants_e1 (is a type of)
- coronilla -> plants_e2 (is a type of)

## 10. Atomic Subquestion DAG
- None: What type of plant is Colocasia?
- None: Is Coronilla a type of plant?
