# DEPO Decomposition #69

- Dataset: `musique`
- Question: What was the wettest year of the administrative territorial entity that contains the administrative territorial entity where Drexel Heights is located?
- Gold answer: 1905

## 1. Explicit Entities
- Drexel Heights span=(125, 139)

## 2. Entity Masking
- ENTITYA -> Drexel Heights

Masked question: What was the wettest year of the administrative territorial entity that contains the administrative territorial entity where ENTITYA is located?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: What was the wettest year of the administrative territorial entity that contains the administrative territorial entity where ENTITYA is located ?

Mask token checks:
- ENTITYA: OK

Readable SDP edges:
- sdp/dm
  - was[2] --ARG1--> What[1]
  - was[2] --ARG2--> year[5]
  - the[3] --BV--> year[5]
  - wettest[4] --ARG1--> year[5]
  - of[6] --ARG1--> year[5]
  - of[6] --ARG2--> entity[10]
  - the[7] --BV--> entity[10]
  - administrative[8] --ARG1--> entity[10]
  - territorial[9] --ARG1--> entity[10]
  - contains[12] --ARG1--> entity[10]
  - contains[12] --ARG2--> entity[16]
  - the[13] --BV--> entity[16]
  - administrative[14] --ARG1--> entity[16]
  - territorial[15] --ARG1--> entity[16]
  - located[20] --ARG2--> ENTITYA[18]
- sdp/pas
  - was[2] --verb_ARG1--> What[1]
  - ROOT[0] --root--> was[2]
  - was[2] --verb_ARG2--> year[5]
  - the[3] --det_ARG1--> year[5]
  - wettest[4] --adj_ARG1--> year[5]
  - of[6] --prep_ARG1--> year[5]
  - of[6] --prep_ARG2--> entity[10]
  - the[7] --det_ARG1--> entity[10]
  - administrative[8] --adj_ARG1--> entity[10]
  - territorial[9] --adj_ARG1--> entity[10]
  - that[11] --relative_ARG1--> entity[10]
  - contains[12] --verb_ARG1--> entity[10]
  - contains[12] --verb_ARG2--> entity[16]
  - the[13] --det_ARG1--> entity[16]
  - administrative[14] --adj_ARG1--> entity[16]
  - territorial[15] --adj_ARG1--> entity[16]
  - where[17] --conj_ARG1--> entity[16]
  - is[19] --aux_ARG1--> ENTITYA[18]
  - located[20] --verb_ARG2--> ENTITYA[18]
  - where[17] --conj_ARG2--> located[20]
  - is[19] --aux_ARG2--> located[20]
- sdp/psd
  - was[2] --ACT-arg--> What[1]
  - year[5] --RSTR--> wettest[4]
  - was[2] --ACT-arg--> year[5]
  - entity[10] --RSTR--> administrative[8]
  - entity[10] --RSTR--> territorial[9]
  - contains[12] --ACT-arg--> that[11]
  - entity[10] --RSTR--> contains[12]
  - entity[16] --RSTR--> administrative[14]
  - entity[16] --RSTR--> territorial[15]
  - contains[12] --PAT-arg--> entity[16]
  - located[20] --LOC-arg--> where[17]
  - located[20] --PAT-arg--> ENTITYA[18]
  - entity[16] --RSTR--> located[20]

## 4. Global Best Path
- Drexel Heights ---- located ---- entity ---- contains ---- administrative ---- entity ---- year ---- wettest

## 5. Step5 Semantic Reasoning Paths
- p1: Drexel Heights ---- located ---- entity ---- contains ---- administrative ---- entity ---- year ---- wettest
  - p1_e1: p1_n1 --located in--> p1_n2
  - p1_e2: p1_n2 --has wettest year--> p1_n3

## 6. Step5 Atomic Questions
- q1: What is the administrative territorial entity containing Drexel Heights?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What was the wettest year of q1's answer?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What is the administrative territorial entity containing Drexel Heights?
  - depends_on: (none)
- q2: What was the wettest year of q1's answer?
  - depends_on: q1
