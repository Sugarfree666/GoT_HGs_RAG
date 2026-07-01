# DEPO Decomposition #7

- Dataset: `musique`
- Question: Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?
- Gold answer: the Mississippi River Delta

## 1. Explicit Entities
- Southeast Library span=(51, 68)
- Gulf of Mexico span=(98, 112)

## 2. Entity Masking
- ENTITYA -> Southeast Library
- ENTITYB -> Gulf of Mexico

Masked question: Where does the body of water by the city where the ENTITYA designer died empty into the ENTITYB?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Where does the body of water by the city where the ENTITYA designer died empty into the ENTITYB ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - the[3] --BV--> body[4]
  - empty[15] --ARG1--> body[4]
  - by[7] --ARG2--> city[9]
  - the[8] --BV--> city[9]
  - the[11] --BV--> designer[13]
  - ENTITYA[12] --compound--> designer[13]
  - died[14] --ARG1--> designer[13]
  - city[9] --loc--> died[14]
  - Where[1] --loc--> empty[15]
  - into[16] --ARG1--> empty[15]
  - into[16] --ARG2--> ENTITYB[18]
  - the[17] --BV--> ENTITYB[18]
- sdp/pas
  - does[2] --aux_ARG1--> body[4]
  - the[3] --det_ARG1--> body[4]
  - of[5] --prep_ARG1--> body[4]
  - empty[15] --verb_ARG1--> body[4]
  - of[5] --prep_ARG2--> water[6]
  - by[7] --prep_ARG2--> city[9]
  - the[8] --det_ARG1--> city[9]
  - where[10] --conj_ARG1--> city[9]
  - the[11] --det_ARG1--> designer[13]
  - ENTITYA[12] --noun_ARG1--> designer[13]
  - died[14] --verb_ARG1--> designer[13]
  - where[10] --conj_ARG2--> died[14]
  - Where[1] --adj_ARG1--> empty[15]
  - does[2] --aux_ARG2--> empty[15]
  - into[16] --prep_ARG1--> empty[15]
  - into[16] --prep_ARG2--> ENTITYB[18]
  - the[17] --det_ARG1--> ENTITYB[18]
- sdp/psd
  - empty[15] --ACT-arg--> body[4]
  - body[4] --APP--> water[6]
  - body[4] --LOC--> city[9]
  - designer[13] --PAT-arg--> ENTITYA[12]
  - died[14] --ACT-arg--> designer[13]
  - body[4] --RSTR--> died[14]
  - city[9] --RSTR--> died[14]
  - ROOT[0] --root--> empty[15]
  - empty[15] --PAT-arg--> ENTITYB[18]

## 4. Global Best Path
- Gulf of Mexico ---- empty ---- body ---- water ---- died ---- designer ---- Southeast Library

## 5. Step5 Semantic Reasoning Paths
- p1: Gulf of Mexico ---- empty ---- body ---- water ---- died ---- designer ---- Southeast Library
  - p1_e1: p1_n1 --identify body of water associated with--> p1_n2
  - p1_e2: p1_n2 --empty into--> p1_n3

## 6. Step5 Atomic Questions
- q1: What is the body of water associated with the Southeast Library designer who died?
  - depends_on: (none)
  - operation: lookup
  - semantic_edge_ids: p1_e1
- q2: Where does the body of water empty into the Gulf of Mexico?
  - depends_on: q1
  - operation: lookup
  - semantic_edge_ids: p1_e2

## 7. Step5 Legacy Action Trace
(none)

## 8. Atomic Question DAG
- q1: What is the body of water associated with the Southeast Library designer who died?
  - depends_on: (none)
- q2: Where does the body of water empty into the Gulf of Mexico?
  - depends_on: q1
