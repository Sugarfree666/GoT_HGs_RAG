# DEPO Decomposition #53

- Dataset: `musique`
- Question: Who did the person who married their half sister in the bible marry after the death of sarah?
- Gold answer: Keturah

## 1. Explicit Entities
(none)

## 2. Entity Masking
(none)

Masked question: Who did the person who married their half sister in the Bible marry after the death of Sarah?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Who did the person who married their half sister in the Bible marry after the death of Sarah ?

Readable SDP edges:
- sdp/dm
  - the[3] --BV--> person[4]
  - married[6] --ARG1--> person[4]
  - marry[13] --ARG1--> person[4]
  - in[10] --ARG1--> married[6]
  - married[6] --ARG2--> sister[9]
  - their[7] --poss--> sister[9]
  - in[10] --ARG2--> Bible[12]
  - the[11] --BV--> Bible[12]
  - after[14] --ARG1--> marry[13]
  - after[14] --ARG2--> death[16]
  - the[15] --BV--> death[16]
  - of[17] --ARG1--> death[16]
  - of[17] --ARG2--> Sarah[18]
- sdp/pas
  - did[2] --aux_ARG1--> person[4]
  - the[3] --det_ARG1--> person[4]
  - who[5] --relative_ARG1--> person[4]
  - married[6] --verb_ARG1--> person[4]
  - marry[13] --verb_ARG1--> person[4]
  - in[10] --prep_ARG1--> married[6]
  - married[6] --verb_ARG2--> sister[9]
  - their[7] --det_ARG1--> sister[9]
  - half[8] --adj_ARG1--> sister[9]
  - in[10] --prep_ARG2--> Bible[12]
  - the[11] --det_ARG1--> Bible[12]
  - ROOT[0] --root--> marry[13]
  - did[2] --aux_ARG2--> marry[13]
  - after[14] --prep_ARG1--> marry[13]
  - after[14] --prep_ARG2--> death[16]
  - the[15] --det_ARG1--> death[16]
  - of[17] --prep_ARG1--> death[16]
  - of[17] --prep_ARG2--> Sarah[18]
- sdp/psd
  - marry[13] --PAT-arg--> Who[1]
  - marry[13] --ACT-arg--> person[4]
  - married[6] --ACT-arg--> who[5]
  - person[4] --RSTR--> married[6]
  - sister[9] --APP--> their[7]
  - sister[9] --RSTR--> half[8]
  - married[6] --PAT-arg--> sister[9]
  - married[6] --LOC--> Bible[12]
  - ROOT[0] --root--> marry[13]
  - marry[13] --TWHEN--> death[16]
  - death[16] --ACT-arg--> Sarah[18]

## 4. Global Best Path
- Who

## 5. Step5 Semantic Reasoning Paths
- p1: Who
  - p1_e1: p1_n1 --married after the death of--> p1_n2

## 6. Step5 Atomic Questions
- q1: Who married after the death of Sarah?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
Invalid DAG
- semantic_reasoning_paths[0].semantic_edges[0].support_tokens contains token not copied from source_token_path: 'married'.
- semantic_reasoning_paths[0].semantic_edges[0].support_tokens contains token not copied from source_token_path: 'after'.
- semantic_reasoning_paths[0].semantic_edges[0].support_tokens contains token not copied from source_token_path: 'the'.
- semantic_reasoning_paths[0].semantic_edges[0].support_tokens contains token not copied from source_token_path: 'death'.
- semantic_reasoning_paths[0].semantic_edges[0].support_tokens contains token not copied from source_token_path: 'of'.
- semantic_reasoning_paths[0].semantic_edges[0].support_tokens contains token not copied from source_token_path: 'Sarah'.
