# DEPO Decomposition #33

- Dataset: `musique`
- Question: How long are the council terms of the city that shares a county with Helvetia?
- Gold answer: four-year

## 1. Explicit Entities
- Helvetia span=(69, 77)

## 2. Entity Masking
- ENTITYA -> Helvetia

Masked question: How long are the council terms of the city that shares a county with ENTITYA?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: How long are the council terms of the city that shares a county with ENTITYA ?

Mask token checks:
- ENTITYA: OK

Readable SDP edges:
- sdp/dm
  - How[1] --measure--> long[2]
  - are[3] --ARG1--> long[2]
  - the[4] --BV--> terms[6]
  - council[5] --compound--> terms[6]
  - the[8] --BV--> city[9]
  - shares[11] --ARG1--> city[9]
  - with[14] --ARG1--> shares[11]
  - shares[11] --ARG2--> county[13]
  - a[12] --BV--> county[13]
  - with[14] --ARG2--> ENTITYA[15]
- sdp/pas
  - How[1] --adj_ARG1--> long[2]
  - are[3] --verb_ARG2--> long[2]
  - ROOT[0] --root--> are[3]
  - are[3] --verb_ARG1--> terms[6]
  - the[4] --det_ARG1--> terms[6]
  - council[5] --noun_ARG1--> terms[6]
  - of[7] --prep_ARG1--> terms[6]
  - of[7] --prep_ARG2--> city[9]
  - the[8] --det_ARG1--> city[9]
  - that[10] --relative_ARG1--> city[9]
  - shares[11] --verb_ARG1--> city[9]
  - with[14] --prep_ARG1--> shares[11]
  - shares[11] --verb_ARG2--> county[13]
  - a[12] --det_ARG1--> county[13]
  - with[14] --prep_ARG2--> ENTITYA[15]
- sdp/psd
  - long[2] --EXT--> How[1]
  - are[3] --THL--> long[2]
  - ROOT[0] --root--> are[3]
  - terms[6] --RSTR--> council[5]
  - are[3] --ACT-arg--> terms[6]
  - terms[6] --APP--> city[9]
  - shares[11] --ACT-arg--> that[10]
  - city[9] --RSTR--> shares[11]
  - shares[11] --PAT-arg--> county[13]
  - shares[11] --ADDR-arg--> ENTITYA[15]

## 4. Global Best Path
- Helvetia ---- shares ---- county ---- city ---- council ---- terms ---- long ---- How

## 5. Step5 Semantic Reasoning Paths
- p1: Helvetia ---- shares ---- county ---- city ---- council ---- terms ---- long ---- How
  - p1_e1: p1_n1 --shares a county with--> p1_n2
  - p1_e2: p1_n2 --is the city in--> p1_n3
  - p1_e3: p1_n3 --has council terms of--> p1_n4

## 6. Step5 Atomic Questions
- q1: What county does Helvetia share?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: What city is in the county obtained from q1's answer?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: How long are the council terms of the city obtained from q2's answer?
  - depends_on: q2
  - operation: lookup
  - semantic_edge_ids: p1_e3

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What county does Helvetia share?
  - depends_on: (none)
- q2: What city is in the county obtained from q1's answer?
  - depends_on: q1
- q3: How long are the council terms of the city obtained from q2's answer?
  - depends_on: q2
