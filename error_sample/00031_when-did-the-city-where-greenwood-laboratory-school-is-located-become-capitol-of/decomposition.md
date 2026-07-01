# DEPO Decomposition #31

- Dataset: `musique`
- Question: When did the city where Greenwood Laboratory School is located become capitol of the state where the screenwriter of The Poor Boob was born?
- Gold answer: 1839

## 1. Explicit Entities
- Greenwood Laboratory School span=(24, 51)
- The Poor Boob span=(117, 130)

## 2. Entity Masking
- ENTITYA -> Greenwood Laboratory School
- ENTITYB -> The Poor Boob

Masked question: When did the city where ENTITYA is located become the capitol of the state where the screenwriter of ENTITYB was born?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: When did the city where ENTITYA is located become the capitol of the state where the screenwriter of ENTITYB was born ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - the[3] --BV--> city[4]
  - where[5] --loc--> city[4]
  - become[9] --ARG1--> city[4]
  - located[8] --ARG2--> ENTITYA[6]
  - city[4] --loc--> located[8]
  - When[1] --loc--> become[9]
  - become[9] --ARG2--> capitol[11]
  - the[10] --BV--> capitol[11]
  - of[12] --ARG1--> capitol[11]
  - of[12] --ARG2--> state[14]
  - the[13] --BV--> state[14]
  - the[16] --BV--> screenwriter[17]
  - of[18] --ARG1--> screenwriter[17]
  - born[21] --ARG2--> screenwriter[17]
  - of[18] --ARG2--> ENTITYB[19]
  - state[14] --loc--> born[21]
- sdp/pas
  - did[2] --aux_ARG1--> city[4]
  - the[3] --det_ARG1--> city[4]
  - where[5] --conj_ARG1--> city[4]
  - become[9] --verb_ARG1--> city[4]
  - where[5] --conj_ARG1--> ENTITYA[6]
  - is[7] --aux_ARG1--> ENTITYA[6]
  - located[8] --verb_ARG2--> ENTITYA[6]
  - where[5] --conj_ARG2--> located[8]
  - is[7] --aux_ARG2--> located[8]
  - ROOT[0] --root--> become[9]
  - When[1] --adj_ARG1--> become[9]
  - did[2] --aux_ARG2--> become[9]
  - become[9] --verb_ARG2--> capitol[11]
  - the[10] --det_ARG1--> capitol[11]
  - of[12] --prep_ARG1--> capitol[11]
  - of[12] --prep_ARG2--> state[14]
  - the[13] --det_ARG1--> state[14]
  - where[15] --conj_ARG1--> state[14]
  - the[16] --det_ARG1--> screenwriter[17]
  - of[18] --prep_ARG1--> screenwriter[17]
  - was[20] --aux_ARG1--> screenwriter[17]
  - born[21] --verb_ARG2--> screenwriter[17]
  - of[18] --prep_ARG2--> ENTITYB[19]
  - where[15] --conj_ARG2--> born[21]
  - was[20] --aux_ARG2--> born[21]
- sdp/psd
  - become[9] --ACT-arg--> city[4]
  - located[8] --LOC-arg--> where[5]
  - located[8] --PAT-arg--> ENTITYA[6]
  - city[4] --RSTR--> located[8]
  - ROOT[0] --root--> become[9]
  - become[9] --PAT-arg--> capitol[11]
  - capitol[11] --APP--> state[14]
  - born[21] --LOC--> where[15]
  - born[21] --PAT-arg--> screenwriter[17]
  - screenwriter[17] --PAT-arg--> ENTITYB[19]
  - state[14] --RSTR--> born[21]

## 4. Global Best Path
- The Poor Boob ---- screenwriter ---- state ---- born ---- capitol ---- become ---- city ---- located ---- Greenwood Laboratory School

## 5. Step5 Semantic Reasoning Paths
- p1: The Poor Boob ---- screenwriter ---- state ---- born ---- capitol ---- become ---- city ---- located ---- Greenwood Laboratory School
  - p1_e1: p1_n1 --who is the screenwriter of--> p1_n2
  - p1_e2: p1_n2 --born in--> p1_n3
  - p1_e3: p1_n3 --capitol of--> p1_n5
  - p1_e4: p1_n4 --when did become capitol--> p1_n6

## 6. Step5 Atomic Questions
- q1: Who is the screenwriter of The Poor Boob?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: In which state was the screenwriter of The Poor Boob born?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: What is the capitol of the state where the screenwriter of The Poor Boob was born?
  - depends_on: q2
  - operation: lookup
  - semantic_edge_ids: p1_e3
- q4: When did the city where Greenwood Laboratory School is located become capitol?
  - depends_on: q3
  - operation: lookup
  - semantic_edge_ids: p1_e4

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who is the screenwriter of The Poor Boob?
  - depends_on: (none)
- q2: In which state was the screenwriter of The Poor Boob born?
  - depends_on: q1
- q3: What is the capitol of the state where the screenwriter of The Poor Boob was born?
  - depends_on: q2
- q4: When did the city where Greenwood Laboratory School is located become capitol?
  - depends_on: q3
