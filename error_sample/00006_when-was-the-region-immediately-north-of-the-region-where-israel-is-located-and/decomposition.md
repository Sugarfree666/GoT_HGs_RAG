# DEPO Decomposition #6

- Dataset: `musique`
- Question: When was the region immediately north of the region where Israel is located and the location of the Battle of Qurah and Umm al Maradim created?
- Gold answer: 1930

## 1. Explicit Entities
- Israel span=(58, 64)
- Battle of Qurah and Umm al Maradim span=(100, 134)

## 2. Entity Masking
- ENTITYA -> Israel
- ENTITYB -> Battle of Qurah and Umm al Maradim

Masked question: When was the region immediately north of the region where ENTITYA is located and the location of the ENTITYB created?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: When was the region immediately north of the region where ENTITYA is located and the location of the ENTITYB created ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - the[3] --BV--> region[4]
  - north[6] --loc--> region[4]
  - created[20] --ARG2--> region[4]
  - of[7] --ARG1--> north[6]
  - north[6] --ARG2--> region[9]
  - of[7] --ARG2--> region[9]
  - the[8] --BV--> region[9]
  - located[13] --ARG2--> ENTITYA[11]
  - region[9] --loc--> located[13]
  - the[15] --BV--> location[16]
  - of[17] --ARG1--> location[16]
  - of[17] --ARG2--> ENTITYB[19]
  - the[18] --BV--> ENTITYB[19]
- sdp/pas
  - the[3] --det_ARG1--> region[4]
  - north[6] --adj_ARG1--> region[4]
  - created[20] --verb_ARG2--> region[4]
  - immediately[5] --adj_ARG1--> north[6]
  - of[7] --prep_ARG1--> north[6]
  - of[7] --prep_ARG2--> region[9]
  - the[8] --det_ARG1--> region[9]
  - where[10] --conj_ARG1--> region[9]
  - is[12] --aux_ARG1--> ENTITYA[11]
  - located[13] --verb_ARG2--> ENTITYA[11]
  - where[10] --conj_ARG2--> located[13]
  - is[12] --aux_ARG2--> located[13]
  - was[2] --aux_ARG1--> and[14]
  - created[20] --verb_ARG2--> and[14]
  - and[14] --coord_ARG2--> location[16]
  - the[15] --det_ARG1--> location[16]
  - of[17] --prep_ARG1--> location[16]
  - of[17] --prep_ARG2--> ENTITYB[19]
  - the[18] --det_ARG1--> ENTITYB[19]
  - When[1] --adj_ARG1--> created[20]
  - was[2] --aux_ARG2--> created[20]
- sdp/psd
  - created[20] --TWHEN--> When[1]
  - created[20] --PAT-arg--> region[4]
  - north[6] --EXT--> immediately[5]
  - region[4] --LOC--> north[6]
  - north[6] --DIR1--> region[9]
  - located[13] --LOC-arg--> where[10]
  - located[13] --PAT-arg--> ENTITYA[11]
  - region[9] --RSTR--> located[13]
  - created[20] --PAT-arg--> location[16]
  - location[16] --APP--> ENTITYB[19]
  - ROOT[0] --root--> created[20]

## 4. Global Best Path
- Battle of Qurah and Umm al Maradim ---- location ---- created ---- region ---- Israel ---- region ---- located ---- north ---- immediately

## 5. Step5 Semantic Reasoning Paths
- p1: Battle of Qurah and Umm al Maradim ---- location ---- created ---- region ---- Israel ---- region ---- located ---- north ---- immediately
  - p1_e1: p1_n1 --is located in--> p1_n3
  - p1_e2: p1_n2 --is south of--> p1_n3
  - p1_e3: p1_n3 --was created on--> p1_n4

## 6. Step5 Atomic Questions
- q1: What is the region north of Israel where the Battle of Qurah and Umm al Maradim is located?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1, p1_e2
- q2: When was the region north of Israel created?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e3

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What is the region north of Israel where the Battle of Qurah and Umm al Maradim is located?
  - depends_on: (none)
- q2: When was the region north of Israel created?
  - depends_on: q1
