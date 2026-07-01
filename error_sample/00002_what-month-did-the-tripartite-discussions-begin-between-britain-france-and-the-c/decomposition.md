# DEPO Decomposition #2

- Dataset: `musique`
- Question: What month did the Tripartite discussions begin between Britain, France, and the country where, despite being headquartered in the nation called the nobilities commonwealth, the top-ranking Warsaw Pact operatives originated?
- Gold answer: mid-June

## 1. Explicit Entities
- Tripartite span=(19, 29)
- Britain span=(56, 63)
- France span=(65, 71)
- Warsaw Pact span=(190, 201)

## 2. Entity Masking
- ENTITYA -> Tripartite
- ENTITYB -> Britain
- ENTITYC -> France
- ENTITYD -> Warsaw Pact

Masked question: What month did the ENTITYA discussions begin between ENTITYB, ENTITYC, and the country where, despite being headquartered in the nation called the nobilities commonwealth, the top-ranking ENTITYD operatives originated?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: What month did the ENTITYA discussions begin between ENTITYB , ENTITYC , and the country where , despite being headquartered in the nation called the nobilities commonwealth , the top-ranking ENTITYD operatives originated ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK
- ENTITYC: OK
- ENTITYD: OK

Readable SDP edges:
- sdp/dm
  - What[1] --BV--> month[2]
  - the[4] --BV--> discussions[6]
  - ENTITYA[5] --compound--> discussions[6]
  - begin[7] --ARG1--> discussions[6]
  - ROOT[0] --root--> begin[7]
  - month[2] --loc--> begin[7]
  - between[8] --ARG1--> begin[7]
  - between[8] --ARG2--> ENTITYB[9]
  - ENTITYB[9] --conj--> ENTITYC[11]
  - the[14] --BV--> country[15]
  - despite[18] --ARG2--> headquartered[20]
  - in[21] --ARG1--> headquartered[20]
  - headquartered[20] --ARG2--> nation[23]
  - in[21] --ARG2--> nation[23]
  - the[22] --BV--> nation[23]
  - called[24] --ARG3--> nation[23]
  - called[24] --ARG2--> nobilities[26]
  - commonwealth[27] --appos--> nobilities[26]
  - called[24] --ARG2--> commonwealth[27]
  - the[25] --BV--> commonwealth[27]
  - the[29] --BV--> operatives[32]
  - top-ranking[30] --compound--> operatives[32]
  - ENTITYD[31] --compound--> operatives[32]
  - originated[33] --ARG1--> operatives[32]
  - country[15] --loc--> originated[33]
- sdp/pas
  - What[1] --noun_ARG1--> month[2]
  - did[3] --aux_ARG1--> discussions[6]
  - the[4] --det_ARG1--> discussions[6]
  - ENTITYA[5] --noun_ARG1--> discussions[6]
  - begin[7] --verb_ARG1--> discussions[6]
  - ROOT[0] --root--> begin[7]
  - month[2] --noun_ARG1--> begin[7]
  - did[3] --aux_ARG2--> begin[7]
  - between[8] --prep_ARG1--> begin[7]
  - ,[10] --coord_ARG1--> ENTITYB[9]
  - and[13] --coord_ARG1--> ,[10]
  - ,[10] --coord_ARG2--> ENTITYC[11]
  - between[8] --prep_ARG2--> and[13]
  - ,[12] --punct_ARG1--> and[13]
  - and[13] --coord_ARG2--> country[15]
  - the[14] --det_ARG1--> country[15]
  - where[16] --conj_ARG1--> country[15]
  - despite[18] --prep_ARG2--> being[19]
  - despite[18] --prep_ARG2--> headquartered[20]
  - being[19] --aux_ARG2--> headquartered[20]
  - in[21] --prep_ARG1--> headquartered[20]
  - headquartered[20] --verb_ARG2--> nation[23]
  - in[21] --prep_ARG2--> nation[23]
  - the[22] --det_ARG1--> nation[23]
  - called[24] --verb_ARG2--> nation[23]
  - called[24] --verb_ARG3--> commonwealth[27]
  - the[25] --det_ARG1--> commonwealth[27]
  - nobilities[26] --noun_ARG1--> commonwealth[27]
  - headquartered[20] --verb_ARG2--> operatives[32]
  - the[29] --det_ARG1--> operatives[32]
  - top-ranking[30] --adj_ARG1--> operatives[32]
  - ENTITYD[31] --noun_ARG1--> operatives[32]
  - originated[33] --verb_ARG1--> operatives[32]
  - ,[28] --punct_ARG1--> originated[33]
- sdp/psd
  - month[2] --RSTR--> What[1]
  - begin[7] --TWHEN--> month[2]
  - discussions[6] --RSTR--> ENTITYA[5]
  - begin[7] --ACT-arg--> discussions[6]
  - ROOT[0] --root--> begin[7]
  - discussions[6] --ACT-arg--> ENTITYB[9]
  - and[13] --CONJ.member--> ENTITYB[9]
  - and[13] --CONJ.member--> ENTITYC[11]
  - and[13] --CONJ.member--> country[15]
  - originated[33] --CNCS--> headquartered[20]
  - headquartered[20] --LOC--> nation[23]
  - nation[23] --RSTR--> called[24]
  - commonwealth[27] --RSTR--> nobilities[26]
  - called[24] --EFF-arg--> commonwealth[27]
  - operatives[32] --RSTR--> top-ranking[30]
  - operatives[32] --APP--> ENTITYD[31]
  - originated[33] --ACT-arg--> operatives[32]
  - country[15] --RSTR--> originated[33]

## 4. Global Best Path
- P1: Britain ---- begin
- P2: France ---- begin

## 5. Step5 Semantic Reasoning Paths
- p1: Britain ---- begin
  - p1_e1: p1_n1 --initiated--> p1_n2
  - p1_e2: p1_n2 --occurs in--> p1_n3
- p2: France ---- begin
  - p2_e1: p2_n1 --initiated--> p2_n2
  - p2_e2: p2_n2 --occurs in--> p2_n3

## 6. Step5 Atomic Questions
- q1: When did the Tripartite discussions initiated by Britain begin?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What month did the Tripartite discussions occur in according to q1's answer?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: When did the Tripartite discussions initiated by France begin?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p2_e1
- q4: What month did the Tripartite discussions occur in according to q3's answer?
  - depends_on: q3
  - operation: lookup
  - semantic_edge_ids: p2_e2
- q5: Which month is earlier, q2's answer or q4's answer?
  - depends_on: q2, q4
  - operation: compare
  - semantic_edge_ids: (none)

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: When did the Tripartite discussions initiated by Britain begin?
  - depends_on: (none)
- q2: What month did the Tripartite discussions occur in according to q1's answer?
  - depends_on: q1
- q3: When did the Tripartite discussions initiated by France begin?
  - depends_on: (none)
- q4: What month did the Tripartite discussions occur in according to q3's answer?
  - depends_on: q3
- q5: Which month is earlier, q2's answer or q4's answer?
  - depends_on: q2, q4
