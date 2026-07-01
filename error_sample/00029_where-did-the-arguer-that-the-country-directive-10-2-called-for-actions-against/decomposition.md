# DEPO Decomposition #29

- Dataset: `musique`
- Question: Where did the arguer that the country Directive 10/2 called for actions against had become an imperialist power declare he would intervene in the Korean conflict?
- Gold answer: the Politburo

## 1. Explicit Entities
- Directive 10/2 span=(38, 52)
- Korean span=(146, 152)

## 2. Entity Masking
- ENTITYA -> Directive 10/2
- ENTITYB -> Korean

Masked question: Where did the arguer that the country ENTITYA called for actions against had become an imperialist power declare he would intervene in the ENTITYB conflict?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Where did the arguer that the country ENTITYA called for actions against had become an imperialist power declare he would intervene in the ENTITYB conflict ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - the[3] --BV--> arguer[4]
  - declare[18] --ARG1--> arguer[4]
  - the[6] --BV--> country[7]
  - called[9] --ARG2--> country[7]
  - against[12] --ARG2--> country[7]
  - called[9] --ARG1--> ENTITYA[8]
  - called[9] --ARG2--> actions[11]
  - against[12] --ARG2--> actions[11]
  - arguer[4] --loc--> become[14]
  - become[14] --ARG2--> power[17]
  - an[15] --BV--> power[17]
  - imperialist[16] --ARG1--> power[17]
  - Where[1] --loc--> declare[18]
  - intervene[21] --ARG1--> he[19]
  - declare[18] --ARG2--> would[20]
  - would[20] --ARG1--> intervene[21]
  - in[22] --ARG1--> intervene[21]
  - in[22] --ARG2--> conflict[25]
  - the[23] --BV--> conflict[25]
  - ENTITYB[24] --compound--> conflict[25]
- sdp/pas
  - did[2] --aux_ARG1--> arguer[4]
  - the[3] --det_ARG1--> arguer[4]
  - that[5] --relative_ARG1--> arguer[4]
  - declare[18] --verb_ARG1--> arguer[4]
  - the[6] --det_ARG1--> country[7]
  - called[9] --verb_ARG2--> country[7]
  - against[12] --prep_ARG2--> country[7]
  - had[13] --aux_ARG1--> country[7]
  - become[14] --verb_ARG1--> country[7]
  - the[6] --det_ARG1--> ENTITYA[8]
  - called[9] --verb_ARG1--> ENTITYA[8]
  - for[10] --prep_ARG1--> called[9]
  - for[10] --prep_ARG2--> actions[11]
  - against[12] --prep_ARG1--> actions[11]
  - had[13] --aux_ARG2--> become[14]
  - become[14] --verb_ARG2--> power[17]
  - an[15] --det_ARG1--> power[17]
  - imperialist[16] --adj_ARG1--> power[17]
  - ROOT[0] --root--> declare[18]
  - Where[1] --adj_ARG1--> declare[18]
  - did[2] --aux_ARG2--> declare[18]
  - would[20] --aux_ARG1--> he[19]
  - intervene[21] --verb_ARG1--> he[19]
  - declare[18] --verb_ARG2--> intervene[21]
  - would[20] --aux_ARG2--> intervene[21]
  - in[22] --prep_ARG1--> intervene[21]
  - in[22] --prep_ARG2--> conflict[25]
  - the[23] --det_ARG1--> conflict[25]
  - ENTITYB[24] --noun_ARG1--> conflict[25]
- sdp/psd
  - declare[18] --LOC--> Where[1]
  - declare[18] --ACT-arg--> arguer[4]
  - become[14] --ACT-arg--> country[7]
  - called[9] --ACT-arg--> ENTITYA[8]
  - arguer[4] --RSTR--> called[9]
  - country[7] --RSTR--> called[9]
  - called[9] --PAT-arg--> actions[11]
  - arguer[4] --RSTR--> become[14]
  - power[17] --RSTR--> imperialist[16]
  - become[14] --PAT-arg--> power[17]
  - ROOT[0] --root--> declare[18]
  - intervene[21] --ACT-arg--> he[19]
  - declare[18] --PAT-arg--> intervene[21]
  - conflict[25] --RSTR--> ENTITYB[24]
  - intervene[21] --PAT-arg--> conflict[25]

## 4. Global Best Path
- Korean ---- conflict ---- intervene ---- would ---- declare ---- become ---- arguer ---- called ---- country ---- against ---- actions

## 5. Step5 Semantic Reasoning Paths
- p1: Korean ---- conflict ---- intervene ---- would ---- declare ---- become ---- arguer ---- called ---- country ---- against ---- actions
  - p1_e1: p1_n1 --arguer of the Korean conflict--> p1_n2
  - p1_e2: p1_n2 --declared intervention location--> p1_n3

## 6. Step5 Atomic Questions
- q1: Who is the arguer of the Korean conflict?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: Where did q1's answer declare he would intervene?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who is the arguer of the Korean conflict?
  - depends_on: (none)
- q2: Where did q1's answer declare he would intervene?
  - depends_on: q1
