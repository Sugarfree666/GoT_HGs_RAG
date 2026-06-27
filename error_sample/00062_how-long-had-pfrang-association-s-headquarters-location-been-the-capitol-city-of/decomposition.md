# DEPO Decomposition #62

- Dataset: `musique`
- Question: How long had Pfrang Association's headquarters location been the capitol city of the area where Guangling District is located?
- Gold answer: about 400 years

## 1. Explicit Entities
- Pfrang Association span=(13, 31)
- Guangling District span=(96, 114)

## 2. Entity Masking
- ENTITYA -> Pfrang Association
- ENTITYB -> Guangling District

Masked question: How long had ENTITYA's headquarters location been the capitol city of the area where ENTITYB is located?

## 3. Global Best Path
- Guangling District ---- located ---- area ---- capitol ---- city ---- long ---- location ---- headquarters ---- Pfrang Association

## 4. Step5 Action Trace
- q1: What area is Guangling District located in?
  - consume: Guangling District ---- located ---- area
  - produce: q1_answer
- q2: What is the capitol city of the area where q1's answer is located?
  - consume: q1_answer ---- capitol ---- city ---- location ---- headquarters ---- Pfrang Association
  - produce: q2_answer
- q3: How long had q2's answer been the capitol city?
  - consume: q2_answer ---- long
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: What area is Guangling District located in?
  - depends_on: (none)
- q2: What is the capitol city of the area where q1's answer is located?
  - depends_on: q1
- q3: How long had q2's answer been the capitol city?
  - depends_on: q2
