# DEPO Decomposition #21

- Dataset: `musique`
- Question: What does seal stand for in the U.S. military branch that operates destroyers, as well as the USS Edsall?
- Gold answer: Sea, Air, and Land

## 1. Explicit Entities
- U.S span=(32, 35)
- USS Edsall span=(94, 104)

## 2. Entity Masking
- ENTITYA -> U.S
- ENTITYB -> USS Edsall

Masked question: What does seal stand for in the ENTITYA. military branch that operates destroyers, as well as the ENTITYB?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: What does seal stand for in the ENTITYA. military branch that operates destroyers , as well as the ENTITYB ?

Mask token checks:
- ENTITYA: FAILED, placeholder was not found as a single HanLP token. This SDP graph may be unreliable because entity masking was split by HanLP tokenization.
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - stand[4] --ARG1--> seal[3]
  - in[6] --ARG1--> stand[4]
  - in[6] --ARG2--> branch[10]
  - the[7] --BV--> branch[10]
  - military[9] --ARG1--> branch[10]
  - operates[12] --ARG1--> branch[10]
  - operates[12] --ARG2--> destroyers[13]
  - the[18] --BV--> ENTITYB[19]
- sdp/pas
  - for[5] --prep_ARG2--> What[1]
  - does[2] --aux_ARG1--> seal[3]
  - stand[4] --verb_ARG1--> seal[3]
  - ROOT[0] --root--> stand[4]
  - does[2] --aux_ARG2--> stand[4]
  - for[5] --prep_ARG1--> stand[4]
  - in[6] --prep_ARG1--> stand[4]
  - in[6] --prep_ARG2--> branch[10]
  - the[7] --det_ARG1--> branch[10]
  - ENTITYA.[8] --noun_ARG1--> branch[10]
  - military[9] --adj_ARG1--> branch[10]
  - that[11] --relative_ARG1--> branch[10]
  - operates[12] --verb_ARG1--> branch[10]
  - ,[14] --punct_ARG1--> branch[10]
  - operates[12] --verb_ARG2--> destroyers[13]
  - as[15] --adj_ARG1--> as[17]
  - well[16] --adj_ARG1--> as[17]
  - as[17] --prep_ARG2--> ENTITYB[19]
  - the[18] --det_ARG1--> ENTITYB[19]
- sdp/psd
  - stand[4] --PAT-arg--> What[1]
  - stand[4] --ACT-arg--> seal[3]
  - ROOT[0] --root--> stand[4]
  - branch[10] --RSTR--> ENTITYA.[8]
  - branch[10] --RSTR--> military[9]
  - stand[4] --LOC--> branch[10]
  - operates[12] --ACT-arg--> that[11]
  - branch[10] --RSTR--> operates[12]
  - operates[12] --PAT-arg--> destroyers[13]

## 4. Global Best Path
- USS Edsall

## 5. Step5 Semantic Reasoning Paths
- p1: USS Edsall
  - p1_e1: p1_n1 --identify military branch associated with--> p1_n2
  - p1_e2: p1_n2 --determine what seal stands for in--> p1_n3

## 6. Step5 Atomic Questions
- q1: What military branch is associated with the USS Edsall?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What does seal stand for in the military branch that operates destroyers?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
Invalid DAG
- semantic_reasoning_paths[0].semantic_edges[1].support_tokens contains token not copied from source_token_path: 'U.S. military branch that operates destroyers'.
