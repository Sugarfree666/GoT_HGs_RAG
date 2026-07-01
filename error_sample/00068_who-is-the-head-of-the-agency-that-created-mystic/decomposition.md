# DEPO Decomposition #68

- Dataset: `musique`
- Question: Who is the head of the agency that created MYSTIC?
- Gold answer: ADM Michael S. Rogers

## 1. Explicit Entities
- MYSTIC span=(43, 49)

## 2. Entity Masking
- ENTITYA -> MYSTIC

Masked question: Who is the head of the agency that created ENTITYA?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Who is the head of the agency that created ENTITYA ?

Mask token checks:
- ENTITYA: OK

Readable SDP edges:
- sdp/dm
  - is[2] --ARG1--> Who[1]
  - is[2] --ARG2--> head[4]
  - the[3] --BV--> head[4]
  - head[4] --ARG1--> agency[7]
  - the[6] --BV--> agency[7]
  - created[9] --ARG1--> agency[7]
  - created[9] --ARG2--> ENTITYA[10]
- sdp/pas
  - is[2] --verb_ARG1--> Who[1]
  - ROOT[0] --root--> is[2]
  - is[2] --verb_ARG2--> head[4]
  - the[3] --det_ARG1--> head[4]
  - of[5] --prep_ARG1--> head[4]
  - of[5] --prep_ARG2--> agency[7]
  - the[6] --det_ARG1--> agency[7]
  - that[8] --relative_ARG1--> agency[7]
  - created[9] --verb_ARG1--> agency[7]
  - created[9] --verb_ARG2--> ENTITYA[10]
- sdp/psd
  - is[2] --ACT-arg--> Who[1]
  - ROOT[0] --root--> is[2]
  - is[2] --PAT-arg--> head[4]
  - head[4] --APP--> agency[7]
  - created[9] --ACT-arg--> that[8]
  - agency[7] --RSTR--> created[9]
  - created[9] --PAT-arg--> ENTITYA[10]

## 4. Global Best Path
- MYSTIC ---- created ---- agency ---- head

## 5. Step5 Semantic Reasoning Paths
- p1: MYSTIC ---- created ---- agency ---- head
  - p1_e1: p1_n1 --created by--> p1_n2
  - p1_e2: p1_n2 --head of--> p1_n3

## 6. Step5 Atomic Questions
- q1: What agency was created by MYSTIC?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: Who is the head of the agency from q1's answer?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What agency was created by MYSTIC?
  - depends_on: (none)
- q2: Who is the head of the agency from q1's answer?
  - depends_on: q1
