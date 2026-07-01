# DEPO Decomposition #66

- Dataset: `musique`
- Question: What was the 2018 population of the Italian city that's underwater?
- Gold answer: 260,897

## 1. Explicit Entities
- Italian span=(36, 43)

## 2. Entity Masking
- ENTITYA -> Italian

Masked question: What was the population of the ENTITYA city that's underwater in 2018?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: What was the population of the ENTITYA city that ' s underwater in 2018 ?

Mask token checks:
- ENTITYA: OK

Readable SDP edges:
- sdp/dm
  - was[2] --ARG2--> What[1]
  - in[13] --ARG1--> What[1]
  - was[2] --ARG2--> population[4]
  - the[3] --BV--> population[4]
  - population[4] --ARG1--> city[8]
  - of[5] --ARG2--> city[8]
  - the[6] --BV--> city[8]
  - ENTITYA[7] --compound--> city[8]
  - '[10] --ARG1--> city[8]
  - underwater[12] --ARG1--> city[8]
  - s[11] --ARG2--> underwater[12]
  - in[13] --ARG2--> 2018[14]
- sdp/pas
  - was[2] --verb_ARG1--> What[1]
  - ROOT[0] --root--> was[2]
  - was[2] --verb_ARG2--> population[4]
  - the[3] --det_ARG1--> population[4]
  - of[5] --prep_ARG1--> population[4]
  - of[5] --prep_ARG2--> city[8]
  - the[6] --det_ARG1--> city[8]
  - ENTITYA[7] --noun_ARG1--> city[8]
  - that[9] --relative_ARG1--> city[8]
  - '[10] --verb_ARG1--> city[8]
  - s[11] --verb_ARG1--> city[8]
  - underwater[12] --adj_ARG1--> city[8]
  - '[10] --verb_ARG2--> underwater[12]
  - s[11] --verb_ARG2--> underwater[12]
  - in[13] --prep_ARG2--> 2018[14]
- sdp/psd
  - was[2] --PAT-arg--> What[1]
  - city[8] --RSTR--> ENTITYA[7]
  - population[4] --APP--> city[8]
  - s[11] --ACT-arg--> that[9]
  - city[8] --RSTR--> s[11]

## 4. Global Best Path
- Italian ---- city ---- underwater ---- s ---- population

## 5. Step5 Semantic Reasoning Paths
- p1: Italian ---- city ---- underwater ---- s ---- population
  - p1_e1: p1_n1 --retrieve population--> p1_n2
  - p1_e2: p1_n1 --is--> p1_n3

## 6. Step5 Atomic Questions
- q1: What is the 2018 population of the Italian city?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: Which Italian city is underwater?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What is the 2018 population of the Italian city?
  - depends_on: (none)
- q2: Which Italian city is underwater?
  - depends_on: (none)
