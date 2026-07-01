# DEPO Decomposition #37

- Dataset: `musique`
- Question: Who is the child of the president who suffered a major defeat when the body which approves members of the American cabinet refused to ratify the Versailles treaty?
- Gold answer: Jessie Woodrow Wilson

## 1. Explicit Entities
- Versailles span=(145, 155)

## 2. Entity Masking
- ENTITYA -> Versailles

Masked question: Who is the child of the president who suffered a major defeat when the body that approves members of the American cabinet refused to ratify the ENTITYA treaty?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Who is the child of the president who suffered a major defeat when the body that approves members of the American cabinet refused to ratify the ENTITYA treaty ?

Mask token checks:
- ENTITYA: OK

Readable SDP edges:
- sdp/dm
  - is[2] --ARG1--> Who[1]
  - is[2] --ARG2--> child[4]
  - the[3] --BV--> child[4]
  - of[5] --ARG1--> child[4]
  - the[6] --BV--> president[7]
  - suffered[9] --ARG1--> president[7]
  - when[13] --ARG1--> suffered[9]
  - suffered[9] --ARG2--> defeat[12]
  - a[10] --BV--> defeat[12]
  - major[11] --ARG1--> defeat[12]
  - the[14] --BV--> body[15]
  - approves[17] --ARG1--> body[15]
  - refused[23] --ARG1--> body[15]
  - ratify[25] --ARG1--> body[15]
  - approves[17] --ARG2--> members[18]
  - of[19] --ARG1--> members[18]
  - members[18] --ARG1--> cabinet[22]
  - the[20] --BV--> cabinet[22]
  - American[21] --ARG1--> cabinet[22]
  - when[13] --ARG2--> refused[23]
  - refused[23] --ARG2--> ratify[25]
  - ratify[25] --ARG2--> treaty[28]
  - the[26] --BV--> treaty[28]
  - ENTITYA[27] --compound--> treaty[28]
- sdp/pas
  - is[2] --verb_ARG1--> Who[1]
  - ROOT[0] --root--> is[2]
  - is[2] --verb_ARG2--> child[4]
  - the[3] --det_ARG1--> child[4]
  - of[5] --prep_ARG1--> child[4]
  - of[5] --prep_ARG2--> president[7]
  - the[6] --det_ARG1--> president[7]
  - who[8] --relative_ARG1--> president[7]
  - suffered[9] --verb_ARG1--> president[7]
  - when[13] --conj_ARG1--> suffered[9]
  - suffered[9] --verb_ARG2--> defeat[12]
  - a[10] --det_ARG1--> defeat[12]
  - major[11] --adj_ARG1--> defeat[12]
  - the[14] --det_ARG1--> body[15]
  - that[16] --relative_ARG1--> body[15]
  - approves[17] --verb_ARG1--> body[15]
  - refused[23] --verb_ARG1--> body[15]
  - ratify[25] --verb_ARG1--> body[15]
  - approves[17] --verb_ARG2--> members[18]
  - of[19] --prep_ARG1--> members[18]
  - of[19] --prep_ARG2--> cabinet[22]
  - the[20] --det_ARG1--> cabinet[22]
  - American[21] --adj_ARG1--> cabinet[22]
  - when[13] --conj_ARG2--> refused[23]
  - refused[23] --verb_ARG2--> ratify[25]
  - to[24] --comp_ARG1--> ratify[25]
  - ratify[25] --verb_ARG2--> treaty[28]
  - the[26] --det_ARG1--> treaty[28]
  - ENTITYA[27] --noun_ARG1--> treaty[28]
- sdp/psd
  - is[2] --ACT-arg--> Who[1]
  - ROOT[0] --root--> is[2]
  - is[2] --PAT-arg--> child[4]
  - child[4] --APP--> president[7]
  - suffered[9] --ACT-arg--> who[8]
  - child[4] --RSTR--> suffered[9]
  - president[7] --RSTR--> suffered[9]
  - defeat[12] --RSTR--> major[11]
  - suffered[9] --PAT-arg--> defeat[12]
  - refused[23] --ACT-arg--> body[15]
  - approves[17] --ACT-arg--> that[16]
  - body[15] --RSTR--> approves[17]
  - approves[17] --PAT-arg--> members[18]
  - cabinet[22] --APP--> American[21]
  - members[18] --APP--> cabinet[22]
  - suffered[9] --TWHEN--> refused[23]
  - refused[23] --PAT-arg--> ratify[25]
  - treaty[28] --RSTR--> ENTITYA[27]
  - ratify[25] --PAT-arg--> treaty[28]

## 4. Global Best Path
- Versailles ---- treaty ---- ratify ---- refused ---- body ---- approves ---- members ---- cabinet ---- American

## 5. Step5 Semantic Reasoning Paths
- p1: Versailles ---- treaty ---- ratify ---- refused ---- body ---- approves ---- members ---- cabinet ---- American
  - p1_e1: p1_n1 --body that refused to ratify--> p1_n2
  - p1_e2: p1_n2 --president of the body--> p1_n3
  - p1_e3: p1_n3 --child of--> p1_n4

## 6. Step5 Atomic Questions
- q1: What is the body that refused to ratify the Versailles treaty?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: Who is the president of the body that approves members of the American cabinet?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2
- q3: Who is the child of the president?
  - depends_on: q2
  - operation: lookup
  - semantic_edge_ids: p1_e3

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
Invalid DAG
- semantic_reasoning_paths[0].semantic_edges[2].support_tokens contains token not copied from source_token_path: 'child'.
- semantic_reasoning_paths[0].semantic_edges[2].support_tokens contains token not copied from source_token_path: 'president'.
