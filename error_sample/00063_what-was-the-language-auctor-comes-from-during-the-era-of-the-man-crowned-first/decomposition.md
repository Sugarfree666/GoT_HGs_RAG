# DEPO Decomposition #63

- Dataset: `musique`
- Question: What was the language Auctor comes from during the era of the man crowned first Holy Roman Emperor later known as?
- Gold answer: Medieval Latin

## 1. Explicit Entities
- Auctor span=(22, 28)
- Holy Roman Emperor span=(80, 98)

## 2. Entity Masking
- ENTITYA -> Auctor
- ENTITYB -> Holy Roman Emperor

Masked question: What was the language ENTITYA comes from during the era of the man crowned the first ENTITYB?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: What was the language ENTITYA comes from during the era of the man crowned the first ENTITYB ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - was[2] --ARG2--> language[4]
  - the[3] --BV--> language[4]
  - comes[6] --ARG2--> language[4]
  - comes[6] --ARG1--> ENTITYA[5]
  - from[7] --ARG1--> comes[6]
  - during[8] --ARG2--> era[10]
  - the[9] --BV--> era[10]
  - of[11] --ARG1--> era[10]
  - of[11] --ARG2--> man[13]
  - the[12] --BV--> man[13]
  - crowned[14] --ARG2--> man[13]
  - crowned[14] --ARG2--> ENTITYB[17]
  - the[15] --BV--> ENTITYB[17]
  - first[16] --ARG1--> ENTITYB[17]
- sdp/pas
  - was[2] --verb_ARG1--> What[1]
  - ROOT[0] --root--> was[2]
  - during[8] --prep_ARG1--> was[2]
  - was[2] --verb_ARG2--> language[4]
  - the[3] --det_ARG1--> language[4]
  - from[7] --prep_ARG2--> language[4]
  - comes[6] --verb_ARG1--> ENTITYA[5]
  - from[7] --prep_ARG1--> comes[6]
  - during[8] --prep_ARG2--> era[10]
  - the[9] --det_ARG1--> era[10]
  - of[11] --prep_ARG1--> era[10]
  - of[11] --prep_ARG2--> man[13]
  - the[12] --det_ARG1--> man[13]
  - crowned[14] --verb_ARG2--> man[13]
  - the[15] --det_ARG1--> ENTITYB[17]
  - first[16] --adj_ARG1--> ENTITYB[17]
- sdp/psd
  - was[2] --PAT-arg--> language[4]
  - language[4] --RSTR--> comes[6]
  - comes[6] --TWHEN--> era[10]
  - era[10] --APP--> man[13]
  - man[13] --RSTR--> crowned[14]
  - ENTITYB[17] --RSTR--> first[16]
  - crowned[14] --PAT-arg--> ENTITYB[17]

## 4. Global Best Path
- Auctor ---- comes ---- language ---- era ---- man ---- crowned ---- Holy Roman Emperor ---- first

## 5. Step5 Semantic Reasoning Paths
- p1: Auctor ---- comes ---- language ---- era ---- man ---- crowned ---- Holy Roman Emperor ---- first
  - p1_e1: p1_n1 --language comes from--> p1_n2
  - p1_e2: p1_n4 --era of--> p1_n3

## 6. Step5 Atomic Questions
- q1: What language does Auctor come from?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: Who was the first Holy Roman Emperor?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: What was the era of the first Holy Roman Emperor?
  - depends_on: q2
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What language does Auctor come from?
  - depends_on: (none)
- q2: Who was the first Holy Roman Emperor?
  - depends_on: (none)
- q3: What was the era of the first Holy Roman Emperor?
  - depends_on: q2
