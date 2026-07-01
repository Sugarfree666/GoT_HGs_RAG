# DEPO Decomposition #62

- Dataset: `musique`
- Question: How long had Pfrang Association's headquarters location been the capitol city of the area where Guangling District is located?
- Gold answer: about 400 years

## 1. Explicit Entities
- Pfrang Association span=(13, 31)
- Guangling District span=(96, 114)

## 2. Entity Masking
- ENTITYA -> Pfrang Association
- ENTITYB -> Guangling District

Masked question: How long had ENTITYA's headquarters location been the capital city of the area where ENTITYB is located?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: How long had ENTITYA ' s headquarters location been the capital city of the area where ENTITYB is located ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - How[1] --measure--> long[2]
  - ENTITYA[4] --poss--> location[8]
  - headquarters[7] --compound--> location[8]
  - been[9] --ARG1--> location[8]
  - long[2] --loc--> been[9]
  - been[9] --ARG2--> city[12]
  - the[10] --BV--> city[12]
  - capital[11] --compound--> city[12]
  - of[13] --ARG1--> city[12]
  - of[13] --ARG2--> area[15]
  - the[14] --BV--> area[15]
  - located[19] --ARG2--> ENTITYB[17]
  - area[15] --loc--> located[19]
- sdp/pas
  - How[1] --adj_ARG1--> long[2]
  - '[5] --poss_ARG2--> ENTITYA[4]
  - s[6] --poss_ARG2--> ENTITYA[4]
  - had[3] --aux_ARG1--> location[8]
  - '[5] --poss_ARG1--> location[8]
  - s[6] --poss_ARG1--> location[8]
  - headquarters[7] --noun_ARG1--> location[8]
  - been[9] --verb_ARG1--> location[8]
  - ROOT[0] --root--> been[9]
  - long[2] --adj_ARG1--> been[9]
  - had[3] --aux_ARG2--> been[9]
  - been[9] --verb_ARG2--> city[12]
  - the[10] --det_ARG1--> city[12]
  - capital[11] --noun_ARG1--> city[12]
  - of[13] --prep_ARG1--> city[12]
  - of[13] --prep_ARG2--> area[15]
  - the[14] --det_ARG1--> area[15]
  - where[16] --conj_ARG1--> area[15]
  - is[18] --aux_ARG1--> ENTITYB[17]
  - located[19] --verb_ARG2--> ENTITYB[17]
  - where[16] --conj_ARG2--> located[19]
  - is[18] --aux_ARG2--> located[19]
- sdp/psd
  - long[2] --EXT--> How[1]
  - been[9] --THL--> long[2]
  - location[8] --APP--> ENTITYA[4]
  - location[8] --RSTR--> headquarters[7]
  - been[9] --ACT-arg--> location[8]
  - ROOT[0] --root--> been[9]
  - city[12] --RSTR--> capital[11]
  - been[9] --PAT-arg--> city[12]
  - city[12] --APP--> area[15]
  - located[19] --LOC-arg--> where[16]
  - located[19] --PAT-arg--> ENTITYB[17]
  - area[15] --RSTR--> located[19]

## 4. Global Best Path
- Guangling District ---- located ---- area ---- capital ---- city ---- long ---- location ---- headquarters ---- Pfrang Association

## 5. Step5 Semantic Reasoning Paths
- p1: Guangling District ---- located ---- area ---- capital ---- city ---- long ---- location ---- headquarters ---- Pfrang Association
  - p1_e1: p1_n1 --is located in--> p1_n2
  - p1_e2: p1_n2 --has capital city--> p1_n3
  - p1_e3: p1_n4 --is the location of--> p1_n3
  - p1_e4: p1_n3 --duration in--> p1_n5

## 6. Step5 Atomic Questions
- q1: What area is Guangling District located in?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What is the capital city of the area where Guangling District is located?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: What is the location of Pfrang Association's headquarters?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e3
- q4: How long has Pfrang Association's headquarters been in the capital city?
  - depends_on: q2, q3
  - operation: lookup
  - semantic_edge_ids: p1_e4

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What area is Guangling District located in?
  - depends_on: (none)
- q2: What is the capital city of the area where Guangling District is located?
  - depends_on: q1
- q3: What is the location of Pfrang Association's headquarters?
  - depends_on: (none)
- q4: How long has Pfrang Association's headquarters been in the capital city?
  - depends_on: q2, q3
