# DEPO Decomposition #11

- Dataset: `musique`
- Question: How many episodes are in season 5 of the series with The Bag or the Bat?
- Gold answer: 12

## 1. Explicit Entities
- The Bag or the Bat span=(53, 71)

## 2. Entity Masking
- ENTITYA -> The Bag or the Bat

Masked question: How many episodes are in season 5 of the series ENTITYA?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: How many episodes are in season 5 of the series ENTITYA ?

Mask token checks:
- ENTITYA: OK

Readable SDP edges:
- sdp/dm
  - many[2] --ARG1--> episodes[3]
  - in[5] --ARG1--> episodes[3]
  - in[5] --ARG2--> 5[7]
  - the[9] --BV--> series[10]
- sdp/pas
  - many[2] --adj_ARG1--> episodes[3]
  - are[4] --verb_ARG1--> episodes[3]
  - in[5] --prep_ARG1--> episodes[3]
  - are[4] --verb_ARG2--> in[5]
  - in[5] --prep_ARG2--> season[6]
  - 5[7] --adj_ARG1--> season[6]
  - of[8] --prep_ARG1--> season[6]
  - series[10] --noun_ARG1--> ENTITYA[11]
- sdp/psd
  - many[2] --EXT--> How[1]
  - episodes[3] --RSTR--> many[2]
  - are[4] --ACT-arg--> episodes[3]
  - ROOT[0] --root--> are[4]
  - are[4] --LOC-arg--> season[6]
  - season[6] --RSTR--> 5[7]

## 4. Global Best Path
- The Bag or the Bat ---- series

## 5. Step5 Semantic Reasoning Paths
- p1: The Bag or the Bat ---- series
  - p1_e1: p1_n1 --is part of--> p1_n2
  - p1_e2: p1_n2 --has--> p1_n3

## 6. Step5 Atomic Questions
- q1: What is season 5 of The Bag or the Bat?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: How many episodes are in q1's answer?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
Invalid DAG
- semantic_reasoning_paths[0].semantic_edges[1].support_tokens contains token not copied from source_token_path: 'season 5'.
