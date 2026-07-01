# DEPO Decomposition #65

- Dataset: `musique`
- Question: Who was a prominent figure at the radio division of the network that created the version of The Biggest Loser set in the country where Seria is?
- Gold answer: Walter Sabo

## 1. Explicit Entities
- The Biggest Loser span=(92, 109)
- Seria span=(135, 140)

## 2. Entity Masking
- ENTITYA -> The Biggest Loser
- ENTITYB -> Seria

Masked question: Who was a prominent figure at the radio division of the network that created the version of ENTITYA set in the country where ENTITYB is?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Who was a prominent figure at the radio division of the network that created the version of ENTITYA set in the country where ENTITYB is ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - was[2] --ARG1--> Who[1]
  - was[2] --ARG2--> figure[5]
  - a[3] --BV--> figure[5]
  - prominent[4] --ARG1--> figure[5]
  - at[6] --ARG1--> figure[5]
  - at[6] --ARG2--> division[9]
  - the[7] --BV--> division[9]
  - radio[8] --compound--> division[9]
  - created[14] --ARG1--> division[9]
  - division[9] --ARG1--> network[12]
  - the[11] --BV--> network[12]
  - created[14] --ARG2--> version[16]
  - the[15] --BV--> version[16]
  - version[16] --ARG1--> ENTITYA[18]
  - in[20] --ARG1--> set[19]
  - in[20] --ARG2--> country[22]
  - the[21] --BV--> country[22]
  - where[23] --loc--> country[22]
  - country[22] --loc--> is[25]
- sdp/pas
  - was[2] --verb_ARG1--> Who[1]
  - ROOT[0] --root--> was[2]
  - was[2] --verb_ARG2--> figure[5]
  - a[3] --det_ARG1--> figure[5]
  - prominent[4] --adj_ARG1--> figure[5]
  - at[6] --prep_ARG1--> figure[5]
  - at[6] --prep_ARG2--> division[9]
  - the[7] --det_ARG1--> division[9]
  - radio[8] --noun_ARG1--> division[9]
  - of[10] --prep_ARG1--> division[9]
  - that[13] --relative_ARG1--> division[9]
  - created[14] --verb_ARG1--> division[9]
  - of[10] --prep_ARG2--> network[12]
  - the[11] --det_ARG1--> network[12]
  - created[14] --verb_ARG2--> version[16]
  - the[15] --det_ARG1--> version[16]
  - of[17] --prep_ARG1--> version[16]
  - set[19] --verb_ARG2--> version[16]
  - of[17] --prep_ARG2--> ENTITYA[18]
  - in[20] --prep_ARG1--> set[19]
  - in[20] --prep_ARG2--> country[22]
  - the[21] --det_ARG1--> country[22]
  - where[23] --conj_ARG1--> country[22]
  - is[25] --verb_ARG2--> country[22]
  - is[25] --verb_ARG1--> ENTITYB[24]
  - where[23] --conj_ARG2--> is[25]
- sdp/psd
  - was[2] --ACT-arg--> Who[1]
  - ROOT[0] --root--> was[2]
  - figure[5] --RSTR--> prominent[4]
  - was[2] --PAT-arg--> figure[5]
  - division[9] --RSTR--> radio[8]
  - figure[5] --LOC--> division[9]
  - division[9] --APP--> network[12]
  - created[14] --ACT-arg--> that[13]
  - division[9] --RSTR--> created[14]
  - created[14] --PAT-arg--> version[16]
  - version[16] --APP--> ENTITYA[18]
  - version[16] --RSTR--> set[19]
  - set[19] --EFF-arg--> country[22]
  - is[25] --LOC-arg--> where[23]
  - is[25] --ACT-arg--> ENTITYB[24]
  - country[22] --RSTR--> is[25]

## 4. Global Best Path
- Seria ---- country ---- set ---- created ---- figure ---- prominent ---- division ---- version ---- The Biggest Loser

## 5. Step5 Semantic Reasoning Paths
- p1: Seria ---- country ---- set ---- created ---- figure ---- prominent ---- division ---- version ---- The Biggest Loser
  - p1_e1: p1_n1 --is located in--> p1_n2
  - p1_e2: p1_n2 --created version of--> p1_n3
  - p1_e3: p1_n3 --has prominent figure at division--> p1_n4

## 6. Step5 Atomic Questions
- q1: What is the country of Seria?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What version of The Biggest Loser was created in the country of Seria?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: Who is the prominent figure at the division of the version of The Biggest Loser?
  - depends_on: q2
  - operation: lookup
  - semantic_edge_ids: p1_e3

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What is the country of Seria?
  - depends_on: (none)
- q2: What version of The Biggest Loser was created in the country of Seria?
  - depends_on: q1
- q3: Who is the prominent figure at the division of the version of The Biggest Loser?
  - depends_on: q2
