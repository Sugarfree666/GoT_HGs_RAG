# DEPO Decomposition #74

- Dataset: `musique`
- Question: What is the meaning of the name of the city where the Yongle emperor greeted the person to whom the edict was addressed?
- Gold answer: "Southern Capital"

## 1. Explicit Entities
- Yongle span=(54, 60)

## 2. Entity Masking
- ENTITYA -> Yongle

Masked question: What is the meaning of the name of the city where the ENTITYA emperor greeted the person to whom the edict was addressed?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: What is the meaning of the name of the city where the ENTITYA emperor greeted the person to whom the edict was addressed ?

Mask token checks:
- ENTITYA: OK

Readable SDP edges:
- sdp/dm
  - is[2] --ARG1--> What[1]
  - is[2] --ARG2--> meaning[4]
  - the[3] --BV--> meaning[4]
  - the[6] --BV--> name[7]
  - of[8] --ARG1--> name[7]
  - of[8] --ARG2--> city[10]
  - the[9] --BV--> city[10]
  - the[12] --BV--> emperor[14]
  - ENTITYA[13] --compound--> emperor[14]
  - greeted[15] --ARG1--> emperor[14]
  - city[10] --loc--> greeted[15]
  - person[17] --loc--> greeted[15]
  - greeted[15] --ARG2--> person[17]
  - the[16] --BV--> person[17]
  - to[18] --ARG2--> person[17]
  - addressed[23] --ARG2--> person[17]
  - the[20] --BV--> edict[21]
  - addressed[23] --ARG2--> edict[21]
  - to[18] --ARG1--> addressed[23]
- sdp/pas
  - is[2] --verb_ARG1--> What[1]
  - ROOT[0] --root--> is[2]
  - is[2] --verb_ARG2--> meaning[4]
  - the[3] --det_ARG1--> meaning[4]
  - of[5] --prep_ARG1--> meaning[4]
  - of[5] --prep_ARG2--> name[7]
  - the[6] --det_ARG1--> name[7]
  - of[8] --prep_ARG1--> name[7]
  - of[8] --prep_ARG2--> city[10]
  - the[9] --det_ARG1--> city[10]
  - where[11] --conj_ARG1--> city[10]
  - the[12] --det_ARG1--> emperor[14]
  - ENTITYA[13] --noun_ARG1--> emperor[14]
  - greeted[15] --verb_ARG1--> emperor[14]
  - where[11] --conj_ARG2--> greeted[15]
  - greeted[15] --verb_ARG2--> person[17]
  - the[16] --det_ARG1--> person[17]
  - to[18] --prep_ARG2--> person[17]
  - whom[19] --relative_ARG1--> person[17]
  - the[20] --det_ARG1--> edict[21]
  - was[22] --aux_ARG1--> edict[21]
  - addressed[23] --verb_ARG2--> edict[21]
  - to[18] --prep_ARG1--> addressed[23]
  - was[22] --aux_ARG2--> addressed[23]
- sdp/psd
  - is[2] --PAT-arg--> What[1]
  - is[2] --ACT-arg--> meaning[4]
  - meaning[4] --PAT-arg--> name[7]
  - name[7] --APP--> city[10]
  - greeted[15] --LOC--> where[11]
  - emperor[14] --APP--> ENTITYA[13]
  - greeted[15] --ACT-arg--> emperor[14]
  - city[10] --RSTR--> greeted[15]
  - greeted[15] --PAT-arg--> person[17]
  - addressed[23] --ADDR-arg--> whom[19]
  - addressed[23] --PAT-arg--> edict[21]
  - person[17] --RSTR--> addressed[23]

## 4. Global Best Path
- Yongle ---- emperor ---- greeted ---- addressed ---- edict ---- person ---- city ---- name ---- meaning

## 5. Step5 Semantic Reasoning Paths
- p1: Yongle ---- emperor ---- greeted ---- addressed ---- edict ---- person ---- city ---- name ---- meaning
  - p1_e1: p1_n1 --greeted--> p1_n2
  - p1_e2: p1_n2 --addressed by the edict--> p1_n3
  - p1_e3: p1_n3 --has name--> p1_n4
  - p1_e4: p1_n4 --has meaning--> p1_n5

## 6. Step5 Atomic Questions
- q1: Who was greeted by the Yongle emperor?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: Who was the person addressed by the edict?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: What is the name of the city?
  - depends_on: q2
  - operation: lookup
  - semantic_edge_ids: p1_e3
- q4: What is the meaning of the name of the city?
  - depends_on: q3
  - operation: lookup
  - semantic_edge_ids: p1_e4

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who was greeted by the Yongle emperor?
  - depends_on: (none)
- q2: Who was the person addressed by the edict?
  - depends_on: q1
- q3: What is the name of the city?
  - depends_on: q2
- q4: What is the meaning of the name of the city?
  - depends_on: q3
