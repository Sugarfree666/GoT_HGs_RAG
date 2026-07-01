# DEPO Decomposition #10

- Dataset: `musique`
- Question: What is the Till dom ensamma performer's birth date?
- Gold answer: 11 September 1962

## 1. Explicit Entities
- Till span=(12, 16)

## 2. Entity Masking
- ENTITYA -> Till

Masked question: What is the birth date of the ENTITYA dom ensamma performer?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: What is the birth date of the ENTITYA dom ensamma performer ?

Mask token checks:
- ENTITYA: OK

Readable SDP edges:
- sdp/dm
  - is[2] --ARG1--> What[1]
  - is[2] --ARG2--> date[5]
  - the[3] --BV--> date[5]
  - birth[4] --compound--> date[5]
  - of[6] --ARG1--> date[5]
  - of[6] --ARG2--> performer[11]
  - the[7] --BV--> performer[11]
  - ensamma[10] --compound--> performer[11]
- sdp/pas
  - is[2] --verb_ARG1--> What[1]
  - ROOT[0] --root--> is[2]
  - is[2] --verb_ARG1--> date[5]
  - the[3] --det_ARG1--> date[5]
  - birth[4] --noun_ARG1--> date[5]
  - of[6] --prep_ARG1--> date[5]
  - of[6] --prep_ARG2--> performer[11]
  - the[7] --det_ARG1--> performer[11]
  - ENTITYA[8] --noun_ARG1--> performer[11]
  - dom[9] --noun_ARG1--> performer[11]
  - ensamma[10] --noun_ARG1--> performer[11]
- sdp/psd
  - is[2] --PAT-arg--> What[1]
  - ROOT[0] --root--> is[2]
  - date[5] --RSTR--> birth[4]
  - is[2] --ACT-arg--> date[5]
  - performer[11] --RSTR--> ENTITYA[8]
  - performer[11] --RSTR--> dom[9]
  - performer[11] --RSTR--> ensamma[10]
  - date[5] --APP--> performer[11]

## 4. Global Best Path
- Till ---- performer ---- ensamma ---- dom ---- date ---- birth

## 5. Step5 Semantic Reasoning Paths
- p1: Till ---- performer ---- ensamma ---- dom ---- date ---- birth
  - p1_e1: p1_n1 --is a--> p1_n2
  - p1_e2: p1_n2 --has birth date--> p1_n3

## 6. Step5 Atomic Questions
- q1: What performer is Till?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What is the birth date of q1's answer?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What performer is Till?
  - depends_on: (none)
- q2: What is the birth date of q1's answer?
  - depends_on: q1
