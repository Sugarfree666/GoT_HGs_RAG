# DEPO Decomposition #67

- Dataset: `musique`
- Question: What was the language from which the last name Sylvester originated during the era of the person crowned emperor of the west in 800 CE later known as?
- Gold answer: Medieval Latin

## 1. Explicit Entities
- Sylvester span=(47, 56)
- 800 CE span=(128, 134)

## 2. Entity Masking
- ENTITYA -> Sylvester
- ENTITYB -> 800 CE

Masked question: What was the language from which the last name ENTITYA originated during the era of the person crowned emperor of the west in ENTITYB?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: What was the language from which the last name ENTITYA originated during the era of the person crowned emperor of the west in ENTITYB ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - was[2] --ARG1--> What[1]
  - was[2] --ARG2--> language[4]
  - the[3] --BV--> language[4]
  - from[5] --ARG2--> language[4]
  - the[7] --BV--> name[9]
  - last[8] --ARG1--> name[9]
  - originated[11] --ARG1--> name[9]
  - name[9] --compound--> ENTITYA[10]
  - from[5] --ARG1--> originated[11]
  - during[12] --ARG1--> originated[11]
  - during[12] --ARG2--> era[14]
  - the[13] --BV--> era[14]
  - of[15] --ARG1--> era[14]
  - of[15] --ARG2--> person[17]
  - the[16] --BV--> person[17]
  - crowned[18] --ARG2--> person[17]
  - of[20] --ARG1--> crowned[18]
  - in[23] --ARG1--> crowned[18]
  - crowned[18] --ARG2--> emperor[19]
  - of[20] --ARG2--> west[22]
  - the[21] --BV--> west[22]
  - in[23] --ARG2--> ENTITYB[24]
- sdp/pas
  - was[2] --verb_ARG1--> What[1]
  - ROOT[0] --root--> was[2]
  - was[2] --verb_ARG2--> language[4]
  - the[3] --det_ARG1--> language[4]
  - from[5] --prep_ARG2--> language[4]
  - which[6] --relative_ARG1--> language[4]
  - the[7] --det_ARG1--> ENTITYA[10]
  - last[8] --adj_ARG1--> ENTITYA[10]
  - name[9] --noun_ARG1--> ENTITYA[10]
  - from[5] --prep_ARG1--> originated[11]
  - during[12] --prep_ARG1--> originated[11]
  - during[12] --prep_ARG2--> era[14]
  - the[13] --det_ARG1--> era[14]
  - of[15] --prep_ARG1--> era[14]
  - of[15] --prep_ARG2--> person[17]
  - the[16] --det_ARG1--> person[17]
  - crowned[18] --verb_ARG2--> person[17]
  - in[23] --prep_ARG1--> crowned[18]
  - crowned[18] --verb_ARG3--> emperor[19]
  - of[20] --prep_ARG1--> emperor[19]
  - of[20] --prep_ARG2--> west[22]
  - the[21] --det_ARG1--> west[22]
  - in[23] --prep_ARG2--> ENTITYB[24]
- sdp/psd
  - was[2] --ACT-arg--> What[1]
  - ROOT[0] --root--> was[2]
  - was[2] --PAT-arg--> language[4]
  - originated[11] --ORIG-arg--> which[6]
  - name[9] --RSTR--> last[8]
  - language[4] --RSTR--> originated[11]
  - originated[11] --TPAR--> era[14]
  - era[14] --APP--> person[17]
  - person[17] --RSTR--> crowned[18]
  - crowned[18] --PAT-arg--> emperor[19]
  - emperor[19] --APP--> west[22]
  - crowned[18] --LOC--> ENTITYB[24]

## 4. Global Best Path
- 800 CE ---- person ---- crowned ---- era ---- originated ---- name ---- language

## 5. Step5 Semantic Reasoning Paths
- p1: 800 CE ---- person ---- crowned ---- era ---- originated ---- name ---- language
  - p1_e1: p1_n1 --identify the person crowned emperor of the west in--> p1_n2
  - p1_e2: p1_n2 --determine the last name associated with--> p1_n3
  - p1_e3: p1_n3 --retrieve the language of--> p1_n4

## 6. Step5 Atomic Questions
- q1: Who was the person crowned emperor of the west in 800 CE?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What is the last name associated with q1's answer?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: What is the language of the last name Sylvester?
  - depends_on: q2
  - operation: lookup
  - semantic_edge_ids: p1_e3

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
Invalid DAG
- semantic_reasoning_paths[0].semantic_edges[1].support_tokens contains token not copied from source_token_path: 'last name'.
- semantic_reasoning_paths[0].semantic_edges[2].support_tokens contains token not copied from source_token_path: 'last name'.
