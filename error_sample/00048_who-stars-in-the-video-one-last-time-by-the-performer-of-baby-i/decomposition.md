# DEPO Decomposition #48

- Dataset: `musique`
- Question: Who stars in the video "One Last Time" by the performer of Baby I?
- Gold answer: Matt Bennett

## 1. Explicit Entities
- One Last Time span=(24, 37)
- Baby I span=(59, 65)

## 2. Entity Masking
- ENTITYA -> One Last Time
- ENTITYB -> Baby I

Masked question: Who stars in the video 'ENTITYA' by the performer of 'ENTITYB'?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Who stars in the video ' ENTITYA ' by the performer of ' ENTITYB ' ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - stars[2] --ARG1--> Who[1]
  - in[3] --ARG1--> stars[2]
  - by[9] --ARG1--> stars[2]
  - in[3] --ARG2--> video[5]
  - the[4] --BV--> video[5]
  - ENTITYA[7] --appos--> video[5]
  - by[9] --ARG2--> performer[11]
  - the[10] --BV--> performer[11]
  - of[12] --ARG1--> performer[11]
  - of[12] --ARG2--> ENTITYB[14]
- sdp/pas
  - stars[2] --verb_ARG1--> Who[1]
  - ROOT[0] --root--> stars[2]
  - in[3] --prep_ARG1--> stars[2]
  - the[4] --det_ARG1--> video[5]
  - video[5] --noun_ARG1--> ENTITYA[7]
  - '[6] --quote_ARG2--> ENTITYA[7]
  - '[6] --quote_ARG3--> '[8]
  - by[9] --prep_ARG2--> performer[11]
  - the[10] --det_ARG1--> performer[11]
  - of[12] --prep_ARG1--> performer[11]
  - '[13] --punct_ARG1--> of[12]
  - of[12] --prep_ARG2--> ENTITYB[14]
  - '[13] --quote_ARG2--> ENTITYB[14]
  - '[13] --quote_ARG3--> '[15]
- sdp/psd
  - stars[2] --ACT-arg--> Who[1]
  - ROOT[0] --root--> stars[2]
  - stars[2] --LOC--> video[5]
  - video[5] --ID--> ENTITYA[7]
  - performer[11] --PAT-arg--> ENTITYB[14]

## 4. Global Best Path
- Baby I ---- performer ---- One Last Time ---- video ---- stars ---- Who

## 5. Step5 Semantic Reasoning Paths
- p1: Baby I ---- performer ---- One Last Time ---- video ---- stars ---- Who
  - p1_e1: p1_n1 --identify performer of--> p1_n2
  - p1_e2: p1_n2 --performer of video--> p1_n3
  - p1_e3: p1_n3 --identify stars in--> p1_n4

## 6. Step5 Atomic Questions
- q1: Who is the performer of Baby I?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What video is the performer of Baby I associated with?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: Who stars in the video One Last Time?
  - depends_on: q2
  - operation: lookup
  - semantic_edge_ids: p1_e3

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who is the performer of Baby I?
  - depends_on: (none)
- q2: What video is the performer of Baby I associated with?
  - depends_on: q1
- q3: Who stars in the video One Last Time?
  - depends_on: q2
