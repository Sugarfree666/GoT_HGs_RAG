# DEPO Decomposition #17

- Dataset: `musique`
- Question: When was the Palau de la Generalitat constructed in the city where Martin from the region where Perdiguera is located died?
- Gold answer: built in the 15th century

## 1. Explicit Entities
- Palau de la Generalitat span=(13, 36)
- Martin span=(67, 73)
- Perdiguera span=(96, 106)

## 2. Entity Masking
- ENTITYA -> Palau de la Generalitat
- ENTITYB -> Martin
- ENTITYC -> Perdiguera

Masked question: When was the ENTITYA constructed in the city where ENTITYB from the region where ENTITYC is located died?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: When was the ENTITYA constructed in the city where ENTITYB from the region where ENTITYC is located died ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK
- ENTITYC: OK

Readable SDP edges:
- sdp/dm
  - the[3] --BV--> ENTITYA[4]
  - constructed[5] --ARG2--> ENTITYA[4]
  - When[1] --loc--> constructed[5]
  - in[6] --ARG1--> constructed[5]
  - in[6] --ARG2--> city[8]
  - the[7] --BV--> city[8]
  - from[11] --ARG1--> ENTITYB[10]
  - died[18] --ARG1--> ENTITYB[10]
  - from[11] --ARG2--> region[13]
  - the[12] --BV--> region[13]
  - where[14] --loc--> region[13]
  - where[14] --loc--> ENTITYC[15]
  - located[17] --ARG2--> ENTITYC[15]
  - region[13] --loc--> located[17]
- sdp/pas
  - was[2] --aux_ARG1--> ENTITYA[4]
  - the[3] --det_ARG1--> ENTITYA[4]
  - constructed[5] --verb_ARG2--> ENTITYA[4]
  - ROOT[0] --root--> constructed[5]
  - When[1] --adj_ARG1--> constructed[5]
  - was[2] --aux_ARG2--> constructed[5]
  - in[6] --prep_ARG1--> constructed[5]
  - in[6] --prep_ARG2--> city[8]
  - the[7] --det_ARG1--> city[8]
  - where[9] --conj_ARG1--> city[8]
  - died[18] --verb_ARG1--> ENTITYB[10]
  - from[11] --prep_ARG2--> region[13]
  - the[12] --det_ARG1--> region[13]
  - where[14] --conj_ARG1--> region[13]
  - is[16] --aux_ARG1--> ENTITYC[15]
  - located[17] --verb_ARG2--> ENTITYC[15]
  - where[14] --conj_ARG2--> located[17]
  - is[16] --aux_ARG2--> located[17]
- sdp/psd
  - constructed[5] --PAT-arg--> ENTITYA[4]
  - constructed[5] --LOC--> city[8]
  - ENTITYB[10] --DIR1--> region[13]
  - located[17] --LOC-arg--> where[14]
  - located[17] --PAT-arg--> ENTITYC[15]
  - region[13] --RSTR--> located[17]
  - city[8] --DESCR--> died[18]

## 4. Global Best Path
- Perdiguera ---- located ---- region ---- Martin ---- died ---- city ---- constructed ---- Palau de la Generalitat

## 5. Step5 Semantic Reasoning Paths
- p1: Perdiguera ---- located ---- region ---- Martin ---- died ---- city ---- constructed ---- Palau de la Generalitat
  - p1_e1: p1_n1 --is located in--> p1_n2
  - p1_e2: p1_n2 --has resident--> p1_n3
  - p1_e3: p1_n3 --died in--> p1_n4
  - p1_e4: p1_n4 --constructed--> p1_n5
  - p1_e5: p1_n5 --was constructed in--> p1_n6

## 6. Step5 Atomic Questions
- q1: What region is Perdiguera located in?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: Who has resident in the region of Perdiguera?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: In which city did Martin die?
  - depends_on: q2
  - operation: lookup
  - semantic_edge_ids: p1_e3
- q4: When was the Palau de la Generalitat constructed in the city where Martin died?
  - depends_on: q3
  - operation: lookup
  - semantic_edge_ids: p1_e5

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What region is Perdiguera located in?
  - depends_on: (none)
- q2: Who has resident in the region of Perdiguera?
  - depends_on: q1
- q3: In which city did Martin die?
  - depends_on: q2
- q4: When was the Palau de la Generalitat constructed in the city where Martin died?
  - depends_on: q3
