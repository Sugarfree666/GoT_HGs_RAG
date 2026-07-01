# DEPO Decomposition #43

- Dataset: `musique`
- Question: What was the form of the language Auctor is in, used in the era of the Frankish king who created the Holy Roman Empire, later known as?
- Gold answer: Medieval Latin

## 1. Explicit Entities
- Auctor span=(34, 40)
- Frankish span=(71, 79)
- Holy Roman Empire span=(101, 118)

## 2. Entity Masking
- ENTITYA -> Auctor
- ENTITYB -> Frankish
- ENTITYC -> Holy Roman Empire

Masked question: What was the form of the language ENTITYA is in, used in the era of the ENTITYB king who created the ENTITYC?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: What was the form of the language ENTITYA is in , used in the era of the ENTITYB king who created the ENTITYC ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK
- ENTITYC: OK

Readable SDP edges:
- sdp/dm
  - was[2] --ARG1--> form[4]
  - the[3] --BV--> form[4]
  - used[12] --ARG2--> form[4]
  - form[4] --ARG1--> language[7]
  - the[6] --BV--> language[7]
  - in[10] --ARG2--> language[7]
  - in[10] --ARG1--> ENTITYA[8]
  - in[13] --ARG1--> used[12]
  - in[13] --ARG2--> era[15]
  - the[14] --BV--> era[15]
  - of[16] --ARG1--> era[15]
  - of[16] --ARG2--> king[19]
  - the[17] --BV--> king[19]
  - ENTITYB[18] --compound--> king[19]
  - created[21] --ARG1--> king[19]
  - created[21] --ARG2--> ENTITYC[23]
  - the[22] --BV--> ENTITYC[23]
- sdp/pas
  - ROOT[0] --root--> was[2]
  - was[2] --verb_ARG2--> form[4]
  - the[3] --det_ARG1--> form[4]
  - of[5] --prep_ARG1--> form[4]
  - ,[11] --punct_ARG1--> form[4]
  - used[12] --verb_ARG2--> form[4]
  - of[5] --prep_ARG2--> language[7]
  - the[6] --det_ARG1--> language[7]
  - in[10] --prep_ARG2--> language[7]
  - is[9] --verb_ARG1--> ENTITYA[8]
  - in[10] --prep_ARG1--> ENTITYA[8]
  - is[9] --verb_ARG2--> in[10]
  - in[13] --prep_ARG1--> used[12]
  - in[13] --prep_ARG2--> era[15]
  - the[14] --det_ARG1--> era[15]
  - of[16] --prep_ARG1--> era[15]
  - of[16] --prep_ARG2--> king[19]
  - the[17] --det_ARG1--> king[19]
  - ENTITYB[18] --noun_ARG1--> king[19]
  - who[20] --relative_ARG1--> king[19]
  - created[21] --verb_ARG1--> king[19]
  - created[21] --verb_ARG2--> ENTITYC[23]
  - the[22] --det_ARG1--> ENTITYC[23]
- sdp/psd
  - was[2] --PAT-arg--> form[4]
  - form[4] --APP--> language[7]
  - is[9] --ACT-arg--> ENTITYA[8]
  - language[7] --RSTR--> is[9]
  - form[4] --RSTR--> used[12]
  - used[12] --TWHEN--> era[15]
  - king[19] --RSTR--> ENTITYB[18]
  - era[15] --APP--> king[19]
  - created[21] --ACT-arg--> who[20]
  - king[19] --RSTR--> created[21]
  - created[21] --PAT-arg--> ENTITYC[23]

## 4. Global Best Path
- Frankish ---- king ---- Holy Roman Empire ---- created ---- era ---- form ---- used ---- language ---- Auctor

## 5. Step5 Semantic Reasoning Paths
- p1: Frankish ---- king ---- Holy Roman Empire ---- created ---- era ---- form ---- used ---- language ---- Auctor
  - p1_e1: p1_n1 --refers to the era of--> p1_n5
  - p1_e2: p1_n5 --has a form of language used in--> p1_n4
  - p1_e3: p1_n2 --is associated with the form of language of--> p1_n4

## 6. Step5 Atomic Questions
- q1: What is the era of the Frankish king?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What is the form of language used in the era of the Frankish king?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: What is the form of language associated with the Holy Roman Empire?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e3
- q4: What was the form of language used in the era of the Frankish king and associated with the Holy Roman Empire?
  - depends_on: q2, q3
  - operation: select
  - semantic_edge_ids: (none)

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What is the era of the Frankish king?
  - depends_on: (none)
- q2: What is the form of language used in the era of the Frankish king?
  - depends_on: q1
- q3: What is the form of language associated with the Holy Roman Empire?
  - depends_on: (none)
- q4: What was the form of language used in the era of the Frankish king and associated with the Holy Roman Empire?
  - depends_on: q2, q3
