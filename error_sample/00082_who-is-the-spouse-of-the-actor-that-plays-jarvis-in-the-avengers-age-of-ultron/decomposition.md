# DEPO Decomposition #82

- Dataset: `musique`
- Question: Who is the spouse of the actor that plays Jarvis in the Avengers Age of Ultron?
- Gold answer: Jennifer Connelly

## 1. Explicit Entities
- Jarvis span=(42, 48)
- Avengers Age of Ultron span=(56, 78)

## 2. Entity Masking
- ENTITYA -> Jarvis
- ENTITYB -> Avengers Age of Ultron

Masked question: Who is the spouse of the actor that plays ENTITYA in the ENTITYB?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Who is the spouse of the actor that plays ENTITYA in the ENTITYB ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - is[2] --ARG1--> Who[1]
  - is[2] --ARG2--> spouse[4]
  - the[3] --BV--> spouse[4]
  - of[5] --ARG1--> spouse[4]
  - of[5] --ARG2--> actor[7]
  - the[6] --BV--> actor[7]
  - plays[9] --ARG1--> actor[7]
  - in[11] --ARG1--> plays[9]
  - plays[9] --ARG2--> ENTITYA[10]
- sdp/pas
  - is[2] --verb_ARG1--> Who[1]
  - ROOT[0] --root--> is[2]
  - is[2] --verb_ARG2--> spouse[4]
  - the[3] --det_ARG1--> spouse[4]
  - of[5] --prep_ARG1--> spouse[4]
  - of[5] --prep_ARG2--> actor[7]
  - the[6] --det_ARG1--> actor[7]
  - that[8] --relative_ARG1--> actor[7]
  - plays[9] --verb_ARG1--> actor[7]
  - in[11] --prep_ARG1--> plays[9]
  - plays[9] --verb_ARG2--> ENTITYA[10]
  - the[12] --det_ARG1--> ENTITYB[13]
- sdp/psd
  - is[2] --ACT-arg--> Who[1]
  - ROOT[0] --root--> is[2]
  - is[2] --ACT-arg--> spouse[4]
  - spouse[4] --APP--> actor[7]
  - plays[9] --ACT-arg--> that[8]
  - actor[7] --RSTR--> plays[9]
  - plays[9] --PAT-arg--> ENTITYA[10]

## 4. Global Best Path
- Jarvis ---- plays ---- actor ---- spouse

## 5. Step5 Semantic Reasoning Paths
- p1: Jarvis ---- plays ---- actor ---- spouse
  - p1_e1: p1_n1 --who plays--> p1_n2
  - p1_e2: p1_n2 --spouse of--> p1_n3

## 6. Step5 Atomic Questions
- q1: Who plays Jarvis?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What is the spouse of q1's answer?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who plays Jarvis?
  - depends_on: (none)
- q2: What is the spouse of q1's answer?
  - depends_on: q1
