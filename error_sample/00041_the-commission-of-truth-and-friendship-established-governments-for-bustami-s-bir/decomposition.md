# DEPO Decomposition #41

- Dataset: `musique`
- Question: The Commission of Truth and Friendship established governments for Bustami's birth country and a second country that had who as president after declaring independence?
- Gold answer: Francisco Guterres

## 1. Explicit Entities
- The Commission of Truth and Friendship span=(0, 38)
- Bustami span=(67, 74)

## 2. Entity Masking
- ENTITYA -> The Commission of Truth and Friendship
- ENTITYB -> Bustami

Masked question: ENTITYA established governments for ENTITYB's birth country and a second country that had who as president after declaring independence?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: ENTITYA established governments for ENTITYB ' s birth country and a second country that had who as president after declaring independence ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - established[2] --ARG1--> ENTITYA[1]
  - established[2] --ARG2--> governments[3]
  - for[4] --ARG1--> governments[3]
  - for[4] --ARG2--> country[9]
  - ENTITYB[5] --poss--> country[9]
  - birth[8] --compound--> country[9]
  - country[9] --_and_c--> country[13]
  - a[11] --BV--> country[13]
  - second[12] --ARG1--> country[13]
  - had[15] --ARG1--> country[13]
  - as[17] --ARG1--> had[15]
  - after[19] --ARG1--> had[15]
  - had[15] --ARG2--> who[16]
  - as[17] --ARG2--> president[18]
  - after[19] --ARG2--> declaring[20]
  - declaring[20] --ARG2--> independence[21]
- sdp/pas
  - ROOT[0] --root--> established[2]
  - established[2] --verb_ARG2--> governments[3]
  - for[4] --prep_ARG1--> governments[3]
  - '[6] --poss_ARG2--> ENTITYB[5]
  - s[7] --poss_ARG2--> ENTITYB[5]
  - s[7] --poss_ARG1--> country[9]
  - birth[8] --noun_ARG1--> country[9]
  - and[10] --coord_ARG1--> country[9]
  - for[4] --prep_ARG2--> and[10]
  - and[10] --coord_ARG2--> country[13]
  - a[11] --det_ARG1--> country[13]
  - second[12] --adj_ARG1--> country[13]
  - that[14] --relative_ARG1--> country[13]
  - had[15] --verb_ARG1--> country[13]
  - as[17] --prep_ARG1--> had[15]
  - after[19] --prep_ARG1--> had[15]
  - had[15] --verb_ARG2--> who[16]
  - as[17] --prep_ARG2--> president[18]
  - after[19] --prep_ARG2--> declaring[20]
  - declaring[20] --verb_ARG2--> independence[21]
- sdp/psd
  - country[9] --APP--> ENTITYB[5]
  - country[9] --RSTR--> birth[8]
  - and[10] --CONJ.member--> country[9]
  - country[13] --RSTR--> second[12]
  - and[10] --CONJ.member--> country[13]
  - had[15] --ACT-arg--> that[14]
  - country[13] --RSTR--> had[15]
  - had[15] --PAT-arg--> who[16]
  - had[15] --COMPL--> president[18]
  - had[15] --TWHEN--> declaring[20]
  - declaring[20] --PAT-arg--> independence[21]

## 4. Global Best Path
- Bustami ---- birth ---- country ---- governments ---- established ---- The Commission of Truth and Friendship

## 5. Step5 Semantic Reasoning Paths
- p1: Bustami ---- birth ---- country ---- governments ---- established ---- The Commission of Truth and Friendship
  - p1_e1: p1_n1 --determines birth country--> p1_n2
  - p1_e2: p1_n2 --establishes governments for--> p1_n3
  - p1_e3: p1_n3 --has president after declaring independence--> p1_n4

## 6. Step5 Atomic Questions
- q1: What is the birth country of Bustami?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What is the second country established by the governments?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: Who was the president of the second country after declaring independence?
  - depends_on: q2
  - operation: lookup
  - semantic_edge_ids: p1_e3

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
Invalid DAG
- semantic_reasoning_paths[0].semantic_edges[2].support_tokens contains token not copied from source_token_path: 'second country'.
- semantic_reasoning_paths[0].semantic_edges[2].support_tokens contains token not copied from source_token_path: 'who as president after declaring independence'.
