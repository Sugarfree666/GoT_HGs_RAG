# DEPO Decomposition #34

- Dataset: `musique`
- Question: When was the region immediately north of the region that prevailed with the disgrace of Near East and the terrain feature on which shamal is located created?
- Gold answer: 1930

## 1. Explicit Entities
- Near East span=(88, 97)

## 2. Entity Masking
- ENTITYA -> Near East

Masked question: When was the region immediately north of the region that prevailed with the disgrace of ENTITYA and the terrain feature on which shamal is located created?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: When was the region immediately north of the region that prevailed with the disgrace of ENTITYA and the terrain feature on which shamal is located created ?

Mask token checks:
- ENTITYA: OK

Readable SDP edges:
- sdp/dm
  - the[3] --BV--> region[4]
  - north[6] --loc--> region[4]
  - prevailed[11] --ARG1--> region[4]
  - created[26] --ARG2--> region[4]
  - of[7] --mwe--> north[6]
  - north[6] --ARG2--> region[9]
  - of[7] --ARG2--> region[9]
  - the[8] --BV--> region[9]
  - with[12] --ARG2--> disgrace[14]
  - the[13] --BV--> disgrace[14]
  - of[15] --ARG1--> disgrace[14]
  - of[15] --ARG2--> ENTITYA[16]
  - disgrace[14] --_and_c--> feature[20]
  - the[18] --BV--> feature[20]
  - terrain[19] --compound--> feature[20]
  - on[21] --ARG2--> feature[20]
  - located[25] --ARG2--> feature[20]
  - located[25] --ARG2--> shamal[23]
  - on[21] --ARG1--> located[25]
  - When[1] --loc--> created[26]
- sdp/pas
  - was[2] --aux_ARG1--> region[4]
  - the[3] --det_ARG1--> region[4]
  - north[6] --adj_ARG1--> region[4]
  - that[10] --relative_ARG1--> region[4]
  - prevailed[11] --verb_ARG1--> region[4]
  - created[26] --verb_ARG2--> region[4]
  - immediately[5] --adj_ARG1--> north[6]
  - of[7] --prep_ARG1--> north[6]
  - of[7] --prep_ARG2--> region[9]
  - the[8] --det_ARG1--> region[9]
  - with[12] --prep_ARG1--> prevailed[11]
  - with[12] --prep_ARG2--> disgrace[14]
  - the[13] --det_ARG1--> disgrace[14]
  - of[15] --prep_ARG1--> disgrace[14]
  - of[15] --prep_ARG2--> ENTITYA[16]
  - was[2] --aux_ARG1--> and[17]
  - and[17] --coord_ARG2--> feature[20]
  - the[18] --det_ARG1--> feature[20]
  - terrain[19] --noun_ARG1--> feature[20]
  - on[21] --prep_ARG2--> feature[20]
  - which[22] --relative_ARG1--> feature[20]
  - located[25] --verb_ARG2--> feature[20]
  - is[24] --aux_ARG1--> shamal[23]
  - located[25] --verb_ARG2--> shamal[23]
  - on[21] --prep_ARG1--> located[25]
  - is[24] --aux_ARG2--> located[25]
  - ROOT[0] --root--> created[26]
  - When[1] --adj_ARG1--> created[26]
  - was[2] --aux_ARG2--> created[26]
- sdp/psd
  - created[26] --PAT-arg--> region[4]
  - north[6] --EXT--> immediately[5]
  - region[4] --LOC--> north[6]
  - north[6] --DIR1--> region[9]
  - prevailed[11] --ACT-arg--> that[10]
  - region[4] --RSTR--> prevailed[11]
  - prevailed[11] --ACMP--> disgrace[14]
  - disgrace[14] --APP--> ENTITYA[16]
  - feature[20] --RSTR--> terrain[19]
  - and[17] --CONJ.member--> feature[20]
  - located[25] --LOC-arg--> which[22]
  - located[25] --PAT-arg--> shamal[23]
  - feature[20] --RSTR--> located[25]
  - ROOT[0] --root--> created[26]

## 4. Global Best Path
- Near East

## 5. Step5 Semantic Reasoning Paths
- p1: Near East
  - p1_e1: p1_n1 --identify region north of--> p1_n2
  - p1_e2: p1_n2 --retrieve creation date of--> p1_n3

## 6. Step5 Atomic Questions
- q1: What is the region immediately north of the Near East?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: When was the region immediately north of the Near East created?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
Invalid DAG
- semantic_reasoning_paths[0].semantic_edges[1].support_tokens contains token not copied from source_token_path: 'region north of'.
