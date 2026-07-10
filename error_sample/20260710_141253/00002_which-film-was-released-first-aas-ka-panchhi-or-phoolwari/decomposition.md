# DEPO Decomposition #2

- Dataset: `2wikimultihopqa`
- Question: Which film was released first, Aas Ka Panchhi or Phoolwari?
- Gold answer: Phoolwari

## 1. Explicit Entities
- Aas Ka Panchhi span=(31, 45)
- Phoolwari span=(49, 58)

## 2. Entity Masking
- ENTITYA -> Aas Ka Panchhi
- ENTITYB -> Phoolwari

Masked question: Which film was released first, ENTITYA or ENTITYB?

## 3. HanLP SDP Graph
Model: hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE
Tokens: Which film was released first , ENTITYA or ENTITYB ?

Mask token checks:
- ENTITYA: OK
- ENTITYB: OK

Readable SDP edges:
- sdp/dm
  - Which[1] --BV--> film[2]
  - released[4] --ARG2--> film[2]
  - first[5] --ARG1--> released[4]
- sdp/pas
  - Which[1] --det_ARG1--> film[2]
  - was[3] --aux_ARG1--> film[2]
  - released[4] --verb_ARG2--> film[2]
  - was[3] --aux_ARG2--> released[4]
  - first[5] --adj_ARG1--> released[4]
  - ,[6] --punct_ARG1--> released[4]
  - or[8] --coord_ARG1--> ENTITYA[7]
  - or[8] --coord_ARG2--> ENTITYB[9]
- sdp/psd
  - film[2] --RSTR--> Which[1]
  - released[4] --PAT-arg--> film[2]
  - released[4] --TWHEN--> first[5]
  - or[8] --DISJ.member--> ENTITYA[7]
  - ,[6] --APPS.member--> or[8]
  - or[8] --DISJ.member--> ENTITYB[9]

## 4. Global Best Path
- P1: Aas Ka Panchhi ---- film ---- released
- P2: Phoolwari ---- film ---- released

## 5. Step5 Atomic Questions
- q1: When was Aas Ka Panchhi released?
  - depends_on: (none)
  - operation: 
  - output_type: 
- q2: When was Phoolwari released?
  - depends_on: (none)
  - operation: 
  - output_type: 
- q3: Which film was released first, q1's answer or q2's answer?
  - depends_on: q1, q2
  - operation: 
  - output_type: 

## 6. Atomic Question DAG
- q1: When was Aas Ka Panchhi released?
  - depends_on: (none)
- q2: When was Phoolwari released?
  - depends_on: (none)
- q3: Which film was released first, q1's answer or q2's answer?
  - depends_on: q1, q2
