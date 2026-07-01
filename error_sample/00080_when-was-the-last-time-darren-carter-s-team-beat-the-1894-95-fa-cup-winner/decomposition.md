# DEPO Decomposition #80

- Dataset: `musique`
- Question: When was the last time Darren Carter's team beat the 1894-95 FA Cup winner?
- Gold answer: 1 December 2010

## 1. Explicit Entities
- Darren Carter span=(23, 36)
- FA Cup span=(61, 67)

## 2. Entity Masking
- ENTITYA -> Darren Carter
- ENTITYB -> FA Cup

Masked question: When was the last time ENTITYA's team beat the 1894-95 ENTITYB winner?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: When was the last time ENTITYA ' s team beat the 1894-95 ENTITYB winner ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - was[2] --ARG1--> When[1]
  - was[2] --ARG2--> time[5]
  - the[3] --BV--> time[5]
  - last[4] --ARG1--> time[5]
  - ENTITYA[6] --poss--> team[9]
  - beat[10] --ARG1--> team[9]
  - time[5] --loc--> beat[10]
  - beat[10] --ARG2--> winner[14]
  - the[11] --BV--> winner[14]
  - 1894-95[12] --compound--> winner[14]
  - ENTITYB[13] --compound--> winner[14]
- sdp/pas
  - was[2] --verb_ARG1--> When[1]
  - ROOT[0] --root--> was[2]
  - When[1] --adj_ARG1--> was[2]
  - was[2] --verb_ARG2--> time[5]
  - the[3] --det_ARG1--> time[5]
  - last[4] --adj_ARG1--> time[5]
  - '[7] --poss_ARG2--> ENTITYA[6]
  - s[8] --poss_ARG2--> ENTITYA[6]
  - '[7] --poss_ARG1--> team[9]
  - s[8] --poss_ARG1--> team[9]
  - beat[10] --verb_ARG1--> team[9]
  - time[5] --noun_ARG2--> beat[10]
  - beat[10] --verb_ARG2--> winner[14]
  - the[11] --det_ARG1--> winner[14]
  - 1894-95[12] --adj_ARG1--> winner[14]
  - ENTITYB[13] --noun_ARG1--> winner[14]
- sdp/psd
  - was[2] --TWHEN--> When[1]
  - ROOT[0] --root--> was[2]
  - time[5] --RSTR--> last[4]
  - was[2] --ACT-arg--> time[5]
  - team[9] --APP--> ENTITYA[6]
  - beat[10] --ACT-arg--> team[9]
  - time[5] --RSTR--> beat[10]
  - winner[14] --RSTR--> 1894-95[12]
  - winner[14] --RSTR--> ENTITYB[13]
  - beat[10] --PAT-arg--> winner[14]

## 4. Global Best Path
- Darren Carter ---- team ---- beat ---- FA Cup ---- winner ---- time ---- last

## 5. Step5 Semantic Reasoning Paths
- p1: Darren Carter ---- team ---- beat ---- FA Cup ---- winner ---- time ---- last
  - p1_e1: p1_n1 --team of--> p1_n2
  - p1_e2: p1_n2 --beat--> p1_n3

## 6. Step5 Atomic Questions
- q1: What is Darren Carter's team?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: When was the last time q1's answer beat the FA Cup winner?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What is Darren Carter's team?
  - depends_on: (none)
- q2: When was the last time q1's answer beat the FA Cup winner?
  - depends_on: q1
