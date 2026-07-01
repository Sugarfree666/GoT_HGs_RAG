# DEPO Decomposition #38

- Dataset: `musique`
- Question: What does dell call the feature letting the interface replacing FireWire in later iterations of the iPod drives to remain powered when the computer is off?
- Gold answer: PowerShare

## 1. Explicit Entities
- FireWire span=(64, 72)
- iPod span=(100, 104)

## 2. Entity Masking
- ENTITYA -> FireWire
- ENTITYB -> iPod

Masked question: What does Dell call the feature letting the interface replace ENTITYA in later iterations of the ENTITYB drives to remain powered when the computer is off?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: What does Dell call the feature letting the interface replace ENTITYA in later iterations of the ENTITYB drives to remain powered when the computer is off ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - call[4] --ARG1--> Dell[3]
  - call[4] --ARG2--> feature[6]
  - the[5] --BV--> feature[6]
  - letting[7] --ARG1--> feature[6]
  - the[8] --BV--> interface[9]
  - replace[10] --ARG1--> interface[9]
  - letting[7] --ARG2--> replace[10]
  - in[12] --ARG1--> replace[10]
  - replace[10] --ARG2--> ENTITYA[11]
  - in[12] --ARG2--> iterations[14]
  - later[13] --ARG1--> iterations[14]
  - the[16] --BV--> drives[18]
  - ENTITYB[17] --compound--> drives[18]
  - when[22] --ARG1--> remain[20]
  - remain[20] --ARG2--> powered[21]
  - the[23] --BV--> computer[24]
  - off[26] --ARG1--> computer[24]
  - when[22] --ARG2--> off[26]
- sdp/pas
  - call[4] --verb_ARG1--> What[1]
  - does[2] --aux_ARG1--> Dell[3]
  - call[4] --verb_ARG1--> Dell[3]
  - ROOT[0] --root--> call[4]
  - does[2] --aux_ARG2--> call[4]
  - call[4] --verb_ARG3--> feature[6]
  - the[5] --det_ARG1--> feature[6]
  - letting[7] --verb_ARG1--> feature[6]
  - letting[7] --verb_ARG2--> interface[9]
  - the[8] --det_ARG1--> interface[9]
  - replace[10] --verb_ARG1--> interface[9]
  - letting[7] --verb_ARG3--> replace[10]
  - in[12] --prep_ARG1--> replace[10]
  - replace[10] --verb_ARG2--> ENTITYA[11]
  - in[12] --prep_ARG2--> iterations[14]
  - later[13] --adj_ARG1--> iterations[14]
  - of[15] --prep_ARG1--> iterations[14]
  - of[15] --prep_ARG2--> drives[18]
  - the[16] --det_ARG1--> drives[18]
  - ENTITYB[17] --noun_ARG1--> drives[18]
  - to[19] --comp_ARG1--> remain[20]
  - when[22] --conj_ARG1--> remain[20]
  - remain[20] --verb_ARG2--> powered[21]
  - the[23] --det_ARG1--> computer[24]
  - is[25] --verb_ARG1--> computer[24]
  - off[26] --adj_ARG1--> computer[24]
  - when[22] --conj_ARG2--> is[25]
  - is[25] --verb_ARG2--> off[26]
- sdp/psd
  - call[4] --EFF-arg--> What[1]
  - call[4] --ACT-arg--> Dell[3]
  - ROOT[0] --root--> call[4]
  - call[4] --EFF-arg--> feature[6]
  - feature[6] --RSTR--> letting[7]
  - letting[7] --ADDR-arg--> interface[9]
  - replace[10] --ACT-arg--> interface[9]
  - letting[7] --PAT-arg--> replace[10]
  - replace[10] --PAT-arg--> ENTITYA[11]
  - iterations[14] --RSTR--> later[13]
  - replace[10] --LOC--> iterations[14]
  - drives[18] --RSTR--> ENTITYB[17]
  - iterations[14] --APP--> drives[18]
  - letting[7] --PAT-arg--> remain[20]
  - remain[20] --PAT-arg--> powered[21]
  - is[25] --ACT-arg--> computer[24]
  - remain[20] --TWHEN--> is[25]
  - is[25] --PAT-arg--> off[26]

## 4. Global Best Path
- iPod ---- drives ---- later ---- iterations ---- replace ---- interface ---- letting ---- feature ---- call ---- What

## 5. Step5 Semantic Reasoning Paths
- p1: iPod ---- drives ---- later ---- iterations ---- replace ---- interface ---- letting ---- feature ---- call ---- What
  - p1_e1: p1_n1 --has drives--> p1_n2
  - p1_e2: p1_n2 --has feature--> p1_n3

## 6. Step5 Atomic Questions
- q1: What drives does the iPod have?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What feature do the drives have?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What drives does the iPod have?
  - depends_on: (none)
- q2: What feature do the drives have?
  - depends_on: q1
