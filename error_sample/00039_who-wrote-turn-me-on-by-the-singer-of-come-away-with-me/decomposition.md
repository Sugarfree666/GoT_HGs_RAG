# DEPO Decomposition #39

- Dataset: `musique`
- Question: Who wrote Turn Me On by the singer of Come Away with Me?
- Gold answer: John D. Loudermilk

## 1. Explicit Entities
- Turn Me On span=(10, 20)
- Come Away with Me span=(38, 55)

## 2. Entity Masking
- ENTITYA -> Turn Me On
- ENTITYB -> Come Away with Me

Masked question: Who wrote ENTITYA by the singer of ENTITYB?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Who wrote ENTITYA by the singer of ENTITYB ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - wrote[2] --ARG1--> Who[1]
  - by[4] --ARG1--> wrote[2]
  - wrote[2] --ARG2--> ENTITYA[3]
  - by[4] --ARG2--> singer[6]
  - the[5] --BV--> singer[6]
  - of[7] --ARG1--> singer[6]
  - of[7] --ARG2--> ENTITYB[8]
- sdp/pas
  - wrote[2] --verb_ARG1--> Who[1]
  - ROOT[0] --root--> wrote[2]
  - wrote[2] --verb_ARG2--> ENTITYA[3]
  - by[4] --prep_ARG2--> singer[6]
  - the[5] --det_ARG1--> singer[6]
  - of[7] --prep_ARG1--> singer[6]
  - of[7] --prep_ARG2--> ENTITYB[8]
- sdp/psd
  - wrote[2] --ACT-arg--> Who[1]
  - ROOT[0] --root--> wrote[2]
  - wrote[2] --PAT-arg--> ENTITYA[3]
  - wrote[2] --ACT-arg--> singer[6]

## 4. Global Best Path
- Come Away with Me ---- singer ---- wrote ---- Turn Me On

## 5. Step5 Semantic Reasoning Paths
- p1: Come Away with Me ---- singer ---- wrote ---- Turn Me On
  - p1_e1: p1_n1 --singer of--> p1_n2
  - p1_e2: p1_n2 --wrote--> p1_n4

## 6. Step5 Atomic Questions
- q1: Who is the singer of Come Away with Me?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: Who wrote Turn Me On?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who is the singer of Come Away with Me?
  - depends_on: (none)
- q2: Who wrote Turn Me On?
  - depends_on: q1
