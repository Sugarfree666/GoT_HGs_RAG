# DEPO Decomposition #54

- Dataset: `musique`
- Question: What is the average salary of a working person with the same nationality as the producer of The Wild Women of Chastity Gulch?
- Gold answer: $59,039

## 1. Explicit Entities
- The Wild Women of Chastity Gulch span=(92, 124)

## 2. Entity Masking
- ENTITYA -> The Wild Women of Chastity Gulch

Masked question: What is the average salary of a working person with the same nationality as the producer of ENTITYA?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: What is the average salary of a working person with the same nationality as the producer of ENTITYA ?

Mask token checks:
- ENTITYA: OK

Readable SDP edges:
- sdp/dm
  - is[2] --ARG1--> What[1]
  - is[2] --ARG2--> salary[5]
  - the[3] --BV--> salary[5]
  - average[4] --ARG1--> salary[5]
  - of[6] --ARG1--> salary[5]
  - of[6] --ARG2--> person[9]
  - a[7] --BV--> person[9]
  - working[8] --ARG1--> person[9]
  - with[10] --ARG1--> person[9]
  - with[10] --ARG2--> nationality[13]
  - the[11] --BV--> nationality[13]
  - same[12] --ARG1--> nationality[13]
  - same[12] --than--> producer[16]
  - the[15] --BV--> producer[16]
  - producer[16] --ARG1--> ENTITYA[18]
- sdp/pas
  - is[2] --verb_ARG1--> What[1]
  - ROOT[0] --root--> is[2]
  - is[2] --verb_ARG2--> salary[5]
  - the[3] --det_ARG1--> salary[5]
  - average[4] --adj_ARG1--> salary[5]
  - of[6] --prep_ARG1--> salary[5]
  - of[6] --prep_ARG2--> person[9]
  - a[7] --det_ARG1--> person[9]
  - working[8] --verb_ARG1--> person[9]
  - with[10] --prep_ARG1--> person[9]
  - with[10] --prep_ARG2--> nationality[13]
  - the[11] --det_ARG1--> nationality[13]
  - same[12] --adj_ARG1--> nationality[13]
  - as[14] --prep_ARG1--> nationality[13]
  - as[14] --prep_ARG2--> producer[16]
  - the[15] --det_ARG1--> producer[16]
  - of[17] --prep_ARG1--> producer[16]
  - of[17] --prep_ARG2--> ENTITYA[18]
- sdp/psd
  - is[2] --PAT-arg--> What[1]
  - ROOT[0] --root--> is[2]
  - salary[5] --RSTR--> average[4]
  - is[2] --ACT-arg--> salary[5]
  - person[9] --RSTR--> working[8]
  - salary[5] --APP--> person[9]
  - nationality[13] --RSTR--> same[12]
  - person[9] --ACMP--> nationality[13]
  - same[12] --CPR--> producer[16]
  - producer[16] --PAT-arg--> ENTITYA[18]

## 4. Global Best Path
- The Wild Women of Chastity Gulch ---- producer ---- nationality ---- same ---- person ---- working ---- salary ---- average

## 5. Step5 Semantic Reasoning Paths
- p1: The Wild Women of Chastity Gulch ---- producer ---- nationality ---- same ---- person ---- working ---- salary ---- average
  - p1_e1: p1_n1 --produced by--> p1_n2
  - p1_e2: p1_n2 --has nationality--> p1_n3
  - p1_e3: p1_n3 --same nationality as--> p1_n4
  - p1_e4: p1_n4 --has salary--> p1_n5
  - p1_e5: p1_n5 --average of--> p1_n6

## 6. Step5 Atomic Questions
- q1: Who is the producer of The Wild Women of Chastity Gulch?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What is the nationality of q1's answer?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: Who is a working person with the same nationality as q2's answer?
  - depends_on: q2
  - operation: lookup
  - semantic_edge_ids: p1_e3
- q4: What is the salary of q3's answer?
  - depends_on: q3
  - operation: lookup
  - semantic_edge_ids: p1_e4
- q5: What is the average salary of q4's answer?
  - depends_on: q4
  - operation: aggregate
  - semantic_edge_ids: p1_e5

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: Who is the producer of The Wild Women of Chastity Gulch?
  - depends_on: (none)
- q2: What is the nationality of q1's answer?
  - depends_on: q1
- q3: Who is a working person with the same nationality as q2's answer?
  - depends_on: q2
- q4: What is the salary of q3's answer?
  - depends_on: q3
- q5: What is the average salary of q4's answer?
  - depends_on: q4
