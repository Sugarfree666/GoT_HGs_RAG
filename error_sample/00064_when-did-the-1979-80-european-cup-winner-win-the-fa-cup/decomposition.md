# DEPO Decomposition #64

- Dataset: `musique`
- Question: When did the 1979-80 European Cup winner win the FA Cup?
- Gold answer: 1898 and 1959

## 1. Explicit Entities
- European Cup span=(21, 33)
- FA Cup span=(49, 55)

## 2. Entity Masking
- ENTITYA -> European Cup
- ENTITYB -> FA Cup

Masked question: When did the 1979-80 ENTITYA winner win the ENTITYB?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: When did the 1979-80 ENTITYA winner win the ENTITYB ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - the[3] --BV--> winner[6]
  - win[7] --ARG1--> winner[6]
  - When[1] --loc--> win[7]
  - the[8] --BV--> ENTITYB[9]
- sdp/pas
  - did[2] --aux_ARG1--> winner[6]
  - the[3] --det_ARG1--> winner[6]
  - 1979-80[4] --adj_ARG1--> winner[6]
  - ENTITYA[5] --noun_ARG1--> winner[6]
  - win[7] --verb_ARG1--> winner[6]
  - ROOT[0] --root--> win[7]
  - When[1] --adj_ARG1--> win[7]
  - did[2] --aux_ARG2--> win[7]
  - win[7] --verb_ARG2--> ENTITYB[9]
  - the[8] --det_ARG1--> ENTITYB[9]
- sdp/psd
  - win[7] --TWHEN--> When[1]
  - winner[6] --RSTR--> ENTITYA[5]
  - win[7] --ACT-arg--> winner[6]
  - ROOT[0] --root--> win[7]
  - win[7] --PAT-arg--> ENTITYB[9]

## 4. Global Best Path
- FA Cup ---- win ---- European Cup ---- winner ---- 1979-80

## 5. Step5 Semantic Reasoning Paths
- p1: FA Cup ---- win ---- European Cup ---- winner ---- 1979-80
  - p1_e1: p1_n1 --who won--> p1_n2
  - p1_e2: p1_n2 --when did win--> p1_n3

## 6. Step5 Atomic Questions
- q1: Who won the FA Cup?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: When did the 1979-80 European Cup winner win?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who won the FA Cup?
  - depends_on: (none)
- q2: When did the 1979-80 European Cup winner win?
  - depends_on: q1
