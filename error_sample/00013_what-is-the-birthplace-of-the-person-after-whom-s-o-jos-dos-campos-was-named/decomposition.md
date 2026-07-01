# DEPO Decomposition #13

- Dataset: `musique`
- Question: What is the birthplace of the person after whom São José dos Campos was named?
- Gold answer: Nazareth

## 1. Explicit Entities
- São José dos Campos span=(48, 67)

## 2. Entity Masking
- ENTITYA -> São José dos Campos

Masked question: What is the birthplace of the person after whom ENTITYA was named?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: What is the birthplace of the person after whom ENTITYA was named ?

Mask token checks:
- ENTITYA: OK

Readable SDP edges:
- sdp/dm
  - is[2] --ARG1--> What[1]
  - is[2] --ARG2--> birthplace[4]
  - the[3] --BV--> birthplace[4]
  - of[5] --ARG1--> birthplace[4]
  - the[6] --BV--> person[7]
  - after[8] --ARG2--> person[7]
  - named[12] --ARG2--> ENTITYA[10]
  - after[8] --ARG1--> named[12]
- sdp/pas
  - is[2] --verb_ARG1--> What[1]
  - ROOT[0] --root--> is[2]
  - is[2] --verb_ARG2--> birthplace[4]
  - the[3] --det_ARG1--> birthplace[4]
  - of[5] --prep_ARG1--> birthplace[4]
  - of[5] --prep_ARG2--> person[7]
  - the[6] --det_ARG1--> person[7]
  - after[8] --prep_ARG2--> person[7]
  - whom[9] --relative_ARG1--> person[7]
  - was[11] --aux_ARG1--> ENTITYA[10]
  - named[12] --verb_ARG2--> ENTITYA[10]
  - after[8] --prep_ARG1--> named[12]
  - was[11] --aux_ARG2--> named[12]
- sdp/psd
  - is[2] --PAT-arg--> What[1]
  - is[2] --ACT-arg--> birthplace[4]
  - birthplace[4] --APP--> person[7]
  - named[12] --HER--> whom[9]
  - named[12] --PAT-arg--> ENTITYA[10]
  - person[7] --RSTR--> named[12]

## 4. Global Best Path
- São José dos Campos ---- named ---- person ---- birthplace

## 5. Step5 Semantic Reasoning Paths
- p1: São José dos Campos ---- named ---- person ---- birthplace
  - p1_e1: p1_n1 --identify the person named after--> p1_n2
  - p1_e2: p1_n2 --determine the birthplace of--> p1_n3

## 6. Step5 Atomic Questions
- q1: Who is the person named after São José dos Campos?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What is the birthplace of q1's answer?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who is the person named after São José dos Campos?
  - depends_on: (none)
- q2: What is the birthplace of q1's answer?
  - depends_on: q1
