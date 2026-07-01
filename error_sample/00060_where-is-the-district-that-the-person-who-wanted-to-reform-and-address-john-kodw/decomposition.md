# DEPO Decomposition #60

- Dataset: `musique`
- Question: Where is the district that the person who wanted to reform and address John Kodwo Amissah's religion preached a sermon on Marian devotion before his death located?
- Gold answer: Saxony-Anhalt

## 1. Explicit Entities
- John Kodwo Amissah span=(71, 89)
- Marian span=(122, 128)

## 2. Entity Masking
- ENTITYA -> John Kodwo Amissah
- ENTITYB -> Marian

Masked question: Where is the district that the person who wanted to reform and address ENTITYA's religion preached a sermon on ENTITYB devotion before his death located?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Where is the district that the person who wanted to reform and address ENTITYA ' s religion preached a sermon on ENTITYB devotion before his death located ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - the[3] --BV--> district[4]
  - the[6] --BV--> person[7]
  - wanted[9] --ARG1--> person[7]
  - reform[11] --ARG1--> person[7]
  - address[13] --ARG1--> person[7]
  - preached[18] --ARG1--> person[7]
  - wanted[9] --ARG2--> reform[11]
  - reform[11] --_and_c--> address[13]
  - reform[11] --ARG2--> religion[17]
  - address[13] --ARG2--> religion[17]
  - ENTITYA[14] --poss--> religion[17]
  - district[4] --loc--> preached[18]
  - before[24] --ARG1--> preached[18]
  - preached[18] --ARG2--> sermon[20]
  - a[19] --BV--> sermon[20]
  - on[21] --ARG1--> sermon[20]
  - on[21] --ARG2--> devotion[23]
  - ENTITYB[22] --compound--> devotion[23]
  - before[24] --ARG2--> death[26]
  - his[25] --poss--> death[26]
  - district[4] --loc--> located[27]
- sdp/pas
  - is[2] --verb_ARG1--> Where[1]
  - ROOT[0] --root--> is[2]
  - is[2] --aux_ARG1--> district[4]
  - the[3] --det_ARG1--> district[4]
  - that[5] --conj_ARG1--> district[4]
  - the[6] --det_ARG1--> person[7]
  - who[8] --relative_ARG1--> person[7]
  - wanted[9] --verb_ARG1--> person[7]
  - reform[11] --verb_ARG1--> person[7]
  - address[13] --verb_ARG1--> person[7]
  - preached[18] --verb_ARG1--> person[7]
  - and[12] --coord_ARG1--> reform[11]
  - wanted[9] --verb_ARG2--> and[12]
  - to[10] --comp_ARG1--> and[12]
  - and[12] --coord_ARG2--> address[13]
  - '[15] --poss_ARG2--> ENTITYA[14]
  - s[16] --poss_ARG2--> ENTITYA[14]
  - reform[11] --verb_ARG2--> religion[17]
  - address[13] --verb_ARG2--> religion[17]
  - '[15] --poss_ARG1--> religion[17]
  - s[16] --poss_ARG1--> religion[17]
  - that[5] --conj_ARG2--> preached[18]
  - before[24] --prep_ARG1--> preached[18]
  - preached[18] --verb_ARG2--> sermon[20]
  - a[19] --det_ARG1--> sermon[20]
  - on[21] --prep_ARG1--> sermon[20]
  - on[21] --prep_ARG2--> devotion[23]
  - ENTITYB[22] --noun_ARG1--> devotion[23]
  - before[24] --prep_ARG2--> death[26]
  - his[25] --det_ARG1--> death[26]
  - before[24] --prep_ARG1--> located[27]
- sdp/psd
  - is[2] --PAT-arg--> district[4]
  - preached[18] --ACT-arg--> person[7]
  - wanted[9] --ACT-arg--> who[8]
  - reform[11] --ACT-arg--> who[8]
  - address[13] --ACT-arg--> who[8]
  - person[7] --RSTR--> wanted[9]
  - wanted[9] --PAT-arg--> reform[11]
  - and[12] --CONJ.member--> reform[11]
  - wanted[9] --PAT-arg--> address[13]
  - and[12] --CONJ.member--> address[13]
  - religion[17] --APP--> ENTITYA[14]
  - reform[11] --PAT-arg--> religion[17]
  - address[13] --PAT-arg--> religion[17]
  - district[4] --RSTR--> preached[18]
  - preached[18] --PAT-arg--> sermon[20]
  - located[27] --PAT-arg--> sermon[20]
  - devotion[23] --PAT-arg--> ENTITYB[22]
  - sermon[20] --PAT-arg--> devotion[23]
  - death[26] --APP--> his[25]
  - preached[18] --TWHEN--> death[26]
  - district[4] --RSTR--> located[27]

## 4. Global Best Path
- Marian ---- devotion ---- sermon ---- located ---- district ---- preached ---- person ---- address ---- religion ---- reform

## 5. Step5 Semantic Reasoning Paths
- p1: Marian ---- devotion ---- sermon ---- located ---- district ---- preached ---- person ---- address ---- religion ---- reform
  - p1_e1: p1_n1 --preached a sermon on--> p1_n2
  - p1_e2: p1_n2 --located in--> p1_n3

## 6. Step5 Atomic Questions
- q1: Where is the district that John Kodwo Amissah preached a sermon on Marian devotion located?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Where is the district that John Kodwo Amissah preached a sermon on Marian devotion located?
  - depends_on: (none)
