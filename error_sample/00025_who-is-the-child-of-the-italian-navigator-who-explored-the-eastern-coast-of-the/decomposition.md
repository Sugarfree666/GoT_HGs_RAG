# DEPO Decomposition #25

- Dataset: `musique`
- Question: Who is the child of the Italian navigator who explored the eastern coast of the continent César Gaytan was born in for the English?
- Gold answer: Sebastian Cabot

## 1. Explicit Entities
- César Gaytan span=(90, 102)

## 2. Entity Masking
- ENTITYA -> César Gaytan

Masked question: Who is the child of the Italian navigator who explored the eastern coast of the continent ENTITYA was born in for the English?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Who is the child of the Italian navigator who explored the eastern coast of the continent ENTITYA was born in for the English ?

Mask token checks:
- ENTITYA: OK

Readable SDP edges:
- sdp/dm
  - is[2] --ARG1--> Who[1]
  - is[2] --ARG2--> child[4]
  - the[3] --BV--> child[4]
  - of[5] --ARG1--> child[4]
  - of[5] --ARG2--> navigator[8]
  - the[6] --BV--> navigator[8]
  - Italian[7] --ARG1--> navigator[8]
  - explored[10] --ARG1--> navigator[8]
  - born[19] --ARG2--> navigator[8]
  - explored[10] --ARG2--> coast[13]
  - the[11] --BV--> coast[13]
  - eastern[12] --ARG1--> coast[13]
  - of[14] --ARG1--> coast[13]
  - coast[13] --ARG1--> continent[16]
  - the[15] --BV--> continent[16]
  - in[20] --ARG1--> born[19]
  - for[21] --ARG1--> born[19]
  - for[21] --ARG2--> English[23]
  - the[22] --BV--> English[23]
- sdp/pas
  - is[2] --verb_ARG1--> Who[1]
  - ROOT[0] --root--> is[2]
  - is[2] --verb_ARG2--> child[4]
  - the[3] --det_ARG1--> child[4]
  - of[5] --prep_ARG1--> child[4]
  - of[5] --prep_ARG2--> navigator[8]
  - the[6] --det_ARG1--> navigator[8]
  - Italian[7] --adj_ARG1--> navigator[8]
  - who[9] --relative_ARG1--> navigator[8]
  - explored[10] --verb_ARG1--> navigator[8]
  - explored[10] --verb_ARG2--> coast[13]
  - the[11] --det_ARG1--> coast[13]
  - eastern[12] --adj_ARG1--> coast[13]
  - of[14] --prep_ARG1--> coast[13]
  - of[14] --prep_ARG2--> continent[16]
  - the[15] --det_ARG1--> continent[16]
  - explored[10] --verb_ARG2--> ENTITYA[17]
  - was[18] --aux_ARG2--> born[19]
  - in[20] --prep_ARG1--> born[19]
  - for[21] --prep_ARG1--> born[19]
  - for[21] --prep_ARG2--> English[23]
  - the[22] --det_ARG1--> English[23]
- sdp/psd
  - is[2] --ACT-arg--> Who[1]
  - is[2] --PAT-arg--> child[4]
  - navigator[8] --RSTR--> Italian[7]
  - child[4] --APP--> navigator[8]
  - explored[10] --ACT-arg--> who[9]
  - navigator[8] --RSTR--> explored[10]
  - coast[13] --RSTR--> eastern[12]
  - explored[10] --PAT-arg--> coast[13]
  - coast[13] --APP--> continent[16]

## 4. Global Best Path
- César Gaytan ---- explored ---- coast ---- navigator ---- Italian ---- born ---- English

## 5. Step5 Semantic Reasoning Paths
- p1: César Gaytan ---- explored ---- coast ---- navigator ---- Italian ---- born ---- English
  - p1_e1: p1_n1 --identified as--> p1_n2
  - p1_e2: p1_n2 --has child--> p1_n3

## 6. Step5 Atomic Questions
- q1: Who is the Italian navigator identified as César Gaytan?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: Who is the child of the Italian navigator from q1's answer?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who is the Italian navigator identified as César Gaytan?
  - depends_on: (none)
- q2: Who is the child of the Italian navigator from q1's answer?
  - depends_on: q1
