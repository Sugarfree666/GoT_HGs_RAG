# DEPO Decomposition #32

- Dataset: `musique`
- Question: Who is played by the director of The Good Shepherd in The Godfather?
- Gold answer: Vito Corleone

## 1. Explicit Entities
- The Good Shepherd span=(33, 50)
- The Godfather span=(54, 67)

## 2. Entity Masking
- ENTITYA -> The Good Shepherd
- ENTITYB -> The Godfather

Masked question: Who is played by the director of ENTITYA in ENTITYB?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Who is played by the director of ENTITYA in ENTITYB ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - played[3] --ARG2--> Who[1]
  - in[9] --ARG1--> played[3]
  - played[3] --ARG1--> director[6]
  - the[5] --BV--> director[6]
  - director[6] --ARG1--> ENTITYA[8]
  - in[9] --ARG2--> ENTITYB[10]
- sdp/pas
  - is[2] --aux_ARG1--> Who[1]
  - played[3] --verb_ARG2--> Who[1]
  - ROOT[0] --root--> played[3]
  - is[2] --aux_ARG2--> played[3]
  - in[9] --prep_ARG1--> played[3]
  - played[3] --verb_ARG1--> director[6]
  - by[4] --lgs_ARG2--> director[6]
  - the[5] --det_ARG1--> director[6]
  - of[7] --prep_ARG1--> director[6]
  - of[7] --prep_ARG2--> ENTITYA[8]
  - in[9] --prep_ARG2--> ENTITYB[10]
- sdp/psd
  - played[3] --PAT-arg--> Who[1]
  - ROOT[0] --root--> played[3]
  - played[3] --ACT-arg--> director[6]
  - director[6] --PAT-arg--> ENTITYA[8]
  - played[3] --LOC--> ENTITYB[10]

## 4. Global Best Path
- The Godfather ---- played ---- director ---- The Good Shepherd

## 5. Step5 Semantic Reasoning Paths
- p1: The Godfather ---- played ---- director ---- The Good Shepherd
  - p1_e1: p1_n2 --who directed--> p1_n3
  - p1_e2: p1_n1 --who is played by--> p1_n4

## 6. Step5 Atomic Questions
- q1: Who directed The Good Shepherd?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: Who is played by the director of The Godfather?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who directed The Good Shepherd?
  - depends_on: (none)
- q2: Who is played by the director of The Godfather?
  - depends_on: q1
