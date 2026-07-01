# DEPO Decomposition #76

- Dataset: `musique`
- Question: How were people from whom new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the natural boundary between Thailand and A Don's country?
- Gold answer: The dynasty regrouped and defeated the Portuguese

## 1. Explicit Entities
- Somali Muslim Ajuran Empire span=(79, 106)
- Thailand span=(150, 158)
- A Don span=(163, 168)

## 2. Entity Masking
- ENTITYA -> Somali Muslim Ajuran Empire
- ENTITYB -> Thailand
- ENTITYC -> A Don

Masked question: How were people from whom new coins were a proclamation of independence by the ENTITYA expelled from the natural boundary between ENTITYB and ENTITYC's country?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: How were people from whom new coins were a proclamation of independence by the ENTITYA expelled from the natural boundary between ENTITYB and ENTITYC ' s country ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK
- ENTITYC: OK

Readable SDP edges:
- sdp/dm
  - from[4] --ARG2--> people[3]
  - new[6] --ARG1--> coins[7]
  - were[8] --ARG1--> coins[7]
  - were[8] --ARG2--> proclamation[10]
  - a[9] --BV--> proclamation[10]
  - of[11] --ARG1--> proclamation[10]
  - by[13] --ARG1--> proclamation[10]
  - of[11] --ARG2--> independence[12]
  - the[14] --BV--> ENTITYA[15]
  - How[1] --manner--> expelled[16]
  - from[17] --ARG1--> expelled[16]
  - from[17] --ARG2--> boundary[20]
  - the[18] --BV--> boundary[20]
  - natural[19] --ARG1--> boundary[20]
  - between[21] --ARG1--> boundary[20]
  - ENTITYB[22] --_and_c--> ENTITYC[24]
- sdp/pas
  - How[1] --adj_ARG1--> were[2]
  - were[2] --aux_ARG1--> people[3]
  - from[4] --prep_ARG2--> people[3]
  - whom[5] --relative_ARG1--> people[3]
  - expelled[16] --verb_ARG2--> people[3]
  - from[4] --prep_ARG2--> whom[5]
  - new[6] --adj_ARG1--> coins[7]
  - were[8] --verb_ARG1--> coins[7]
  - were[8] --verb_ARG2--> proclamation[10]
  - a[9] --det_ARG1--> proclamation[10]
  - of[11] --prep_ARG1--> proclamation[10]
  - by[13] --prep_ARG1--> proclamation[10]
  - of[11] --prep_ARG2--> independence[12]
  - by[13] --prep_ARG1--> independence[12]
  - by[13] --prep_ARG2--> ENTITYA[15]
  - the[14] --det_ARG1--> ENTITYA[15]
  - ROOT[0] --root--> expelled[16]
  - How[1] --adj_ARG1--> expelled[16]
  - were[2] --aux_ARG2--> expelled[16]
  - from[17] --prep_ARG1--> expelled[16]
  - from[17] --prep_ARG2--> boundary[20]
  - the[18] --det_ARG1--> boundary[20]
  - natural[19] --adj_ARG1--> boundary[20]
  - between[21] --prep_ARG1--> boundary[20]
  - and[23] --coord_ARG1--> ENTITYB[22]
  - between[21] --prep_ARG2--> and[23]
  - s[26] --poss_ARG2--> and[23]
  - and[23] --coord_ARG2--> ENTITYC[24]
  - from[17] --prep_ARG2--> country[27]
  - between[21] --prep_ARG2--> country[27]
  - s[26] --poss_ARG1--> country[27]
- sdp/psd
  - expelled[16] --PAT-arg--> people[3]
  - were[8] --ORIG-arg--> whom[5]
  - coins[7] --RSTR--> new[6]
  - were[8] --ACT-arg--> coins[7]
  - people[3] --RSTR--> were[8]
  - were[8] --PAT-arg--> proclamation[10]
  - proclamation[10] --PAT-arg--> independence[12]
  - boundary[20] --RSTR--> natural[19]
  - expelled[16] --ORIG-arg--> boundary[20]
  - and[23] --CONJ.member--> ENTITYB[22]
  - and[23] --CONJ.member--> ENTITYC[24]
  - expelled[16] --DIR1-arg--> country[27]

## 4. Global Best Path
- Somali Muslim Ajuran Empire ---- independence ---- proclamation ---- coins ---- people ---- expelled ---- country ---- boundary ---- natural

## 5. Step5 Semantic Reasoning Paths
- p1: Somali Muslim Ajuran Empire ---- independence ---- proclamation ---- coins ---- people ---- expelled ---- country ---- boundary ---- natural
  - p1_e1: p1_n1 --declared through--> p1_n3
  - p1_e2: p1_n3 --issued as--> p1_n4
  - p1_e3: p1_n4 --associated with--> p1_n5
  - p1_e4: p1_n5 --expelled from--> p1_n6
  - p1_e5: p1_n6 --located at--> p1_n7
  - p1_e6: p1_n7 --associated with--> p1_n8

## 6. Step5 Atomic Questions
- q1: What was the proclamation of independence by the Somali Muslim Ajuran Empire?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What were the coins issued as part of the proclamation of independence?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: Who were the people associated with the coins?
  - depends_on: q2
  - operation: lookup
  - semantic_edge_ids: p1_e3
- q4: From which country were the people expelled?
  - depends_on: q3
  - operation: lookup
  - semantic_edge_ids: p1_e4
- q5: What is the natural boundary of the country?
  - depends_on: q4
  - operation: lookup
  - semantic_edge_ids: p1_e5
- q6: How is A Don associated with the natural boundary?
  - depends_on: q5
  - operation: lookup
  - semantic_edge_ids: p1_e6

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
Invalid DAG
- semantic_reasoning_paths[0].semantic_edges[5].support_tokens contains token not copied from source_token_path: 'A Don'.
