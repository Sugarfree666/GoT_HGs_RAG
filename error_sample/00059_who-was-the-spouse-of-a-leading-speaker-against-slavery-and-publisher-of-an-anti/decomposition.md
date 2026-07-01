# DEPO Decomposition #59

- Dataset: `musique`
- Question: Who was the spouse of a leading speaker against slavery and publisher of an antislavery newspaper?
- Gold answer: Helen Pitts Douglass

## 1. Explicit Entities
(none)

## 2. Entity Masking
(none)

Masked question: Who was the spouse of a leading speaker against slavery and publisher of an antislavery newspaper?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Who was the spouse of a leading speaker against slavery and publisher of an antislavery newspaper ?

Readable SDP edges:
- sdp/dm
  - was[2] --ARG1--> Who[1]
  - was[2] --ARG2--> spouse[4]
  - the[3] --BV--> spouse[4]
  - of[5] --ARG1--> spouse[4]
  - of[5] --ARG2--> speaker[8]
  - a[6] --BV--> speaker[8]
  - leading[7] --ARG1--> speaker[8]
  - against[9] --ARG1--> speaker[8]
  - against[9] --ARG2--> slavery[10]
  - publisher[12] --ARG1--> newspaper[16]
  - of[13] --ARG2--> newspaper[16]
  - an[14] --BV--> newspaper[16]
  - antislavery[15] --compound--> newspaper[16]
- sdp/pas
  - was[2] --verb_ARG1--> Who[1]
  - ROOT[0] --root--> was[2]
  - was[2] --verb_ARG2--> spouse[4]
  - the[3] --det_ARG1--> spouse[4]
  - of[5] --prep_ARG1--> spouse[4]
  - of[5] --prep_ARG2--> speaker[8]
  - a[6] --det_ARG1--> speaker[8]
  - leading[7] --verb_ARG1--> speaker[8]
  - against[9] --prep_ARG1--> speaker[8]
  - and[11] --coord_ARG1--> speaker[8]
  - against[9] --prep_ARG2--> slavery[10]
  - was[2] --verb_ARG2--> and[11]
  - of[5] --prep_ARG2--> and[11]
  - and[11] --coord_ARG2--> publisher[12]
  - of[13] --prep_ARG1--> publisher[12]
  - of[13] --prep_ARG2--> newspaper[16]
  - an[14] --det_ARG1--> newspaper[16]
  - antislavery[15] --noun_ARG1--> newspaper[16]
- sdp/psd
  - was[2] --ACT-arg--> Who[1]
  - ROOT[0] --root--> was[2]
  - was[2] --PAT-arg--> spouse[4]
  - speaker[8] --RSTR--> leading[7]
  - spouse[4] --APP--> speaker[8]
  - and[11] --CONJ.member--> speaker[8]
  - speaker[8] --BEN--> slavery[10]
  - spouse[4] --APP--> publisher[12]
  - and[11] --CONJ.member--> publisher[12]
  - newspaper[16] --RSTR--> antislavery[15]
  - publisher[12] --PAT-arg--> newspaper[16]

## 4. Global Best Path
- spouse

## 5. Step5 Semantic Reasoning Paths
- p1: spouse
  - p1_e1: p1_n1 --has spouse--> p1_n2

## 6. Step5 Atomic Questions
- q1: Who is the leading speaker against slavery?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: (none)
- q2: Who is the spouse of q1's answer?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e1

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
Invalid DAG
- q1: lookup questions must include at least one semantic_edge_id.
