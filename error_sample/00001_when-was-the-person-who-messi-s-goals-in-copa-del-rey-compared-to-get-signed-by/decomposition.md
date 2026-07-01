# DEPO Decomposition #1

- Dataset: `musique`
- Question: When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?
- Gold answer: June 1982

## 1. Explicit Entities
- Messi span=(24, 29)
- Copa del Rey span=(41, 53)
- Barcelona span=(80, 89)

## 2. Entity Masking
- ENTITYA -> Messi
- ENTITYB -> Copa del Rey
- ENTITYC -> Barcelona

Masked question: When was the person who ENTITYA's goals in ENTITYB compared to get signed by ENTITYC?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: When was the person who ENTITYA ' s goals in ENTITYB compared to get signed by ENTITYC ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK
- ENTITYC: OK

Readable SDP edges:
- sdp/dm
  - was[2] --ARG1--> person[4]
  - the[3] --BV--> person[4]
  - get[14] --ARG1--> person[4]
  - signed[15] --ARG2--> person[4]
  - ENTITYA[6] --poss--> goals[9]
  - s[8] --ARG1--> goals[9]
  - in[10] --ARG1--> goals[9]
  - compared[12] --ARG1--> goals[9]
  - in[10] --ARG2--> ENTITYB[11]
  - When[1] --loc--> get[14]
  - When[1] --loc--> signed[15]
  - get[14] --ARG2--> signed[15]
- sdp/pas
  - was[2] --aux_ARG1--> When[1]
  - was[2] --aux_ARG1--> person[4]
  - the[3] --det_ARG1--> person[4]
  - who[5] --relative_ARG1--> person[4]
  - get[14] --aux_ARG1--> person[4]
  - signed[15] --verb_ARG2--> person[4]
  - '[7] --poss_ARG2--> ENTITYA[6]
  - s[8] --poss_ARG2--> ENTITYA[6]
  - '[7] --poss_ARG1--> goals[9]
  - s[8] --poss_ARG1--> goals[9]
  - in[10] --prep_ARG1--> goals[9]
  - compared[12] --verb_ARG1--> goals[9]
  - in[10] --prep_ARG2--> ENTITYB[11]
  - to[13] --prep_ARG1--> compared[12]
  - ROOT[0] --root--> signed[15]
  - When[1] --adj_ARG1--> signed[15]
  - was[2] --aux_ARG2--> signed[15]
  - get[14] --aux_ARG2--> signed[15]
  - signed[15] --verb_ARG1--> ENTITYC[17]
  - by[16] --lgs_ARG2--> ENTITYC[17]
- sdp/psd
  - signed[15] --PAT-arg--> person[4]
  - compared[12] --ACT-arg--> who[5]
  - goals[9] --APP--> ENTITYA[6]
  - compared[12] --PAT-arg--> goals[9]
  - goals[9] --LOC--> ENTITYB[11]
  - person[4] --RSTR--> compared[12]
  - signed[15] --ACT-arg--> ENTITYC[17]

## 4. Global Best Path
- Barcelona ---- signed ---- get ---- person ---- compared ---- goals ---- Messi

## 5. Step5 Semantic Reasoning Paths
- p1: Barcelona ---- signed ---- get ---- person ---- compared ---- goals ---- Messi
  - p1_e1: p1_n1 --signed the person who was compared to Messi's goals in Copa del Rey--> p1_n4
  - p1_e2: p1_n2 --goals in Copa del Rey compared to--> p1_n3
  - p1_e3: p1_n4 --when was the person signed--> p1_n5

## 6. Step5 Atomic Questions
- q1: Who is the person who was compared to Messi's goals in Copa del Rey?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What are Messi's goals in Copa del Rey compared to?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: When was the person signed?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e3

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who is the person who was compared to Messi's goals in Copa del Rey?
  - depends_on: (none)
- q2: What are Messi's goals in Copa del Rey compared to?
  - depends_on: (none)
- q3: When was the person signed?
  - depends_on: q1
