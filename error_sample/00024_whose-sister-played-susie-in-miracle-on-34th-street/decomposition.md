# DEPO Decomposition #24

- Dataset: `musique`
- Question: Whose sister played Susie in miracle on 34th street?
- Gold answer: Lana Wood

## 1. Explicit Entities
- Susie span=(20, 25)
- miracle on 34th street span=(29, 51)

## 2. Entity Masking
- ENTITYA -> Susie
- ENTITYB -> miracle on 34th street

Masked question: Whose sister played ENTITYA in ENTITYB?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Whose sister played ENTITYA in ENTITYB ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - played[3] --ARG1--> sister[2]
  - played[3] --ARG2--> ENTITYA[4]
- sdp/pas
  - Whose[1] --relative_ARG1--> sister[2]
  - played[3] --verb_ARG1--> sister[2]
  - in[5] --prep_ARG1--> played[3]
  - played[3] --verb_ARG2--> ENTITYA[4]
  - in[5] --prep_ARG2--> ENTITYB[6]
- sdp/psd
  - sister[2] --APP--> Whose[1]
  - played[3] --ACT-arg--> sister[2]
  - ROOT[0] --root--> played[3]
  - played[3] --PAT-arg--> ENTITYA[4]
  - played[3] --LOC--> ENTITYB[6]

## 4. Global Best Path
- Whose ---- sister ---- played ---- Susie

## 5. Step5 Semantic Reasoning Paths
- p1: Whose ---- sister ---- played ---- Susie
  - p1_e1: p1_n1 --played by--> p1_n2

## 6. Step5 Atomic Questions
- q1: Who is the sister that played Susie?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who is the sister that played Susie?
  - depends_on: (none)
