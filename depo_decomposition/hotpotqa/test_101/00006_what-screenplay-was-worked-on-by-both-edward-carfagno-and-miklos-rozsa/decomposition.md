# DEPO Decomposition #6

- Dataset: `hotpotqa`
- Question: What screenplay was worked on by both Edward Carfagno and Miklos Rozsa?
- Gold answer: Julius Caesar

## 1. Semantic-Normalized Question
What screenplay was worked on by both Edward Carfagno and Miklos Rozsa?

## 2. Mask Spans
- Edward Carfagno (entity, Person)
- Miklos Rozsa (entity, Person)

## 3. Selective Masked Question
What screenplay was worked on by both PersonA and PersonB?

## 4. CoreNLP Dependency Parse
- screenplay[2] --det--> What[1]
- worked[4] --nsubj:pass--> screenplay[2]
- worked[4] --aux:pass--> was[3]
- worked[4] --compound:prt--> on[5]
- PersonA[8] --case--> by[6]
- PersonA[8] --cc:preconj--> both[7]
- worked[4] --obl:agent--> PersonA[8]
- PersonB[10] --cc--> and[9]
- worked[4] --obl:agent--> PersonB[10]
- PersonA[8] --conj:and--> PersonB[10]
- worked[4] --punct--> ?[11]

## 5. Undirected Dependency Graph
- What[1] --det-- screenplay[2]
- screenplay[2] --nsubj:pass-- worked[4]
- was[3] --aux:pass-- worked[4]
- worked[4] --compound:prt-- on[5]
- worked[4] --obl:agent-- Edward Carfagno[8]
- worked[4] --obl:agent-- Miklos Rozsa[10]
- worked[4] --punct-- ?[11]
- by[6] --case-- Edward Carfagno[8]
- both[7] --cc:preconj-- Edward Carfagno[8]
- Edward Carfagno[8] --conj:and-- Miklos Rozsa[10]
- and[9] --cc-- Miklos Rozsa[10]

## 6. Entity Start Nodes
- e1: Edward Carfagno graph_node_ids=['8']
- e2: Miklos Rozsa graph_node_ids=['10']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Edward Carfagno -- worked -- screenplay -- What
- e1_p2 (e1): Edward Carfagno -- worked -- screenplay
- e1_p3 (e1): Edward Carfagno -- worked -- on
- e1_p4 (e1): Edward Carfagno -- worked
- e1_p5 (e1): Edward Carfagno -- both
- e1_p6 (e1): Edward Carfagno -- worked -- was
- e1_p7 (e1): Edward Carfagno -- worked -- ?
- e1_p8 (e1): Edward Carfagno -- by
- e1_p9 (e1): Edward Carfagno -- worked -- Miklos Rozsa
- e1_p10 (e1): Edward Carfagno -- Miklos Rozsa
- e1_p11 (e1): Edward Carfagno -- Miklos Rozsa -- worked -- screenplay -- What
- e1_p12 (e1): Edward Carfagno -- Miklos Rozsa -- worked -- screenplay
- e1_p13 (e1): Edward Carfagno -- Miklos Rozsa -- worked -- on
- e1_p14 (e1): Edward Carfagno -- Miklos Rozsa -- worked
- e1_p15 (e1): Edward Carfagno -- worked -- Miklos Rozsa -- and
- e1_p16 (e1): Edward Carfagno -- Miklos Rozsa -- worked -- was
- e1_p17 (e1): Edward Carfagno -- Miklos Rozsa -- worked -- ?
- e1_p18 (e1): Edward Carfagno -- Miklos Rozsa -- and
- e2_p1 (e2): Miklos Rozsa -- worked -- screenplay -- What
- e2_p2 (e2): Miklos Rozsa -- worked -- screenplay
- e2_p3 (e2): Miklos Rozsa -- worked -- on
- e2_p4 (e2): Miklos Rozsa -- worked
- e2_p5 (e2): Miklos Rozsa -- worked -- was
- e2_p6 (e2): Miklos Rozsa -- worked -- ?
- e2_p7 (e2): Miklos Rozsa -- and
- e2_p8 (e2): Miklos Rozsa -- worked -- Edward Carfagno
- e2_p9 (e2): Miklos Rozsa -- Edward Carfagno
- e2_p10 (e2): Miklos Rozsa -- Edward Carfagno -- worked -- screenplay -- What
- e2_p11 (e2): Miklos Rozsa -- worked -- Edward Carfagno -- both
- e2_p12 (e2): Miklos Rozsa -- Edward Carfagno -- worked -- screenplay
- e2_p13 (e2): Miklos Rozsa -- Edward Carfagno -- worked -- on
- e2_p14 (e2): Miklos Rozsa -- Edward Carfagno -- worked
- e2_p15 (e2): Miklos Rozsa -- Edward Carfagno -- both
- e2_p16 (e2): Miklos Rozsa -- worked -- Edward Carfagno -- by
- e2_p17 (e2): Miklos Rozsa -- Edward Carfagno -- worked -- was
- e2_p18 (e2): Miklos Rozsa -- Edward Carfagno -- worked -- ?
- e2_p19 (e2): Miklos Rozsa -- Edward Carfagno -- by

## 8. LLM Selected Entity Paths
- e1: e1_p9 Edward Carfagno -- worked -- Miklos Rozsa
  Reason: This path directly connects Edward Carfagno to the action of working with Miklos Rozsa on a screenplay.
- e2: e2_p8 Miklos Rozsa -- worked -- Edward Carfagno
  Reason: This path directly connects Miklos Rozsa to the action of working with Edward Carfagno on a screenplay.

## 9. Selected Path Semantic Transduction
Nodes:
- edward_carfagno: Edward Carfagno (entity)
- screenplay_e1: screenplay (value_slot)
- miklos_rozsa: Miklos Rozsa (entity)
- screenplay_e2: screenplay (value_slot)

Edges:
- edward_carfagno -> screenplay_e1 (screenplay worked on by Edward Carfagno)
- miklos_rozsa -> screenplay_e2 (screenplay worked on by Miklos Rozsa)

## 10. Atomic Subquestion DAG
- None: What screenplay was worked on by Edward Carfagno?
- None: What screenplay was worked on by Miklos Rozsa?
