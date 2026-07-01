# DEPO Decomposition #12

- Dataset: `musique`
- Question: How were the people from whom new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the country between Thailand and A Lim's country?
- Gold answer: The dynasty regrouped and defeated the Portuguese

## 1. Explicit Entities
- Somali Muslim Ajuran Empire span=(83, 110)
- Thailand span=(145, 153)
- A Lim span=(158, 163)

## 2. Entity Masking
- ENTITYA -> Somali Muslim Ajuran Empire
- ENTITYB -> Thailand
- ENTITYC -> A Lim

Masked question: How were the people from whom new coins were a proclamation of independence by the ENTITYA expelled from the country between ENTITYB and ENTITYC's country?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: How were the people from whom new coins were a proclamation of independence by the ENTITYA expelled from the country between ENTITYB and ENTITYC ' s country ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK
- ENTITYC: OK

Readable SDP edges:
- sdp/dm
  - the[3] --BV--> people[4]
  - from[5] --ARG2--> people[4]
  - expelled[17] --ARG2--> people[4]
  - new[7] --ARG1--> coins[8]
  - were[9] --ARG1--> coins[8]
  - from[5] --ARG1--> were[9]
  - were[9] --ARG2--> proclamation[11]
  - a[10] --BV--> proclamation[11]
  - of[12] --ARG1--> proclamation[11]
  - by[14] --ARG1--> proclamation[11]
  - of[12] --ARG2--> independence[13]
  - the[15] --BV--> ENTITYA[16]
  - How[1] --manner--> expelled[17]
  - from[18] --ARG1--> expelled[17]
  - from[18] --ARG2--> country[20]
  - the[19] --BV--> country[20]
  - between[21] --ARG1--> country[20]
  - between[21] --ARG2--> ENTITYB[22]
  - ENTITYB[22] --_and_c--> ENTITYC[24]
- sdp/pas
  - ROOT[0] --root--> were[2]
  - How[1] --adj_ARG1--> were[2]
  - were[2] --aux_ARG1--> people[4]
  - the[3] --det_ARG1--> people[4]
  - from[5] --prep_ARG2--> people[4]
  - whom[6] --relative_ARG1--> people[4]
  - expelled[17] --verb_ARG2--> people[4]
  - new[7] --adj_ARG1--> coins[8]
  - were[9] --verb_ARG1--> coins[8]
  - were[9] --verb_ARG2--> proclamation[11]
  - a[10] --det_ARG1--> proclamation[11]
  - of[12] --prep_ARG1--> proclamation[11]
  - by[14] --prep_ARG1--> proclamation[11]
  - of[12] --prep_ARG2--> independence[13]
  - by[14] --prep_ARG2--> ENTITYA[16]
  - the[15] --det_ARG1--> ENTITYA[16]
  - ROOT[0] --root--> expelled[17]
  - How[1] --adj_ARG1--> expelled[17]
  - were[2] --aux_ARG2--> expelled[17]
  - from[18] --prep_ARG1--> expelled[17]
  - from[18] --prep_ARG2--> country[20]
  - the[19] --det_ARG1--> country[20]
  - between[21] --prep_ARG1--> country[20]
  - and[23] --coord_ARG1--> ENTITYB[22]
  - between[21] --prep_ARG2--> and[23]
  - '[25] --poss_ARG2--> and[23]
  - s[26] --poss_ARG2--> and[23]
  - and[23] --coord_ARG2--> ENTITYC[24]
  - between[21] --prep_ARG2--> country[27]
  - s[26] --poss_ARG1--> country[27]
- sdp/psd
  - expelled[17] --PAT-arg--> people[4]
  - were[9] --ORIG-arg--> whom[6]
  - coins[8] --RSTR--> new[7]
  - were[9] --ACT-arg--> coins[8]
  - people[4] --RSTR--> were[9]
  - were[9] --PAT-arg--> proclamation[11]
  - proclamation[11] --PAT-arg--> independence[13]
  - expelled[17] --DIR1-arg--> country[20]
  - and[23] --CONJ.member--> ENTITYB[22]
  - and[23] --CONJ.member--> ENTITYC[24]

## 4. Global Best Path
- Somali Muslim Ajuran Empire ---- proclamation ---- coins ---- people ---- expelled ---- country

## 5. Step5 Semantic Reasoning Paths
- p1: Somali Muslim Ajuran Empire ---- proclamation ---- coins ---- people ---- expelled ---- country
  - p1_e1: p1_n1 --proclamation of independence involves--> p1_n2
  - p1_e2: p1_n2 --new coins were issued by--> p1_n3
  - p1_e3: p1_n2 --were expelled from--> p1_n4

## 6. Step5 Atomic Questions
- q1: Who were the people from the Somali Muslim Ajuran Empire?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What new coins were issued by the people?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: From which country were the people expelled?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e3

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who were the people from the Somali Muslim Ajuran Empire?
  - depends_on: (none)
- q2: What new coins were issued by the people?
  - depends_on: q1
- q3: From which country were the people expelled?
  - depends_on: q1
