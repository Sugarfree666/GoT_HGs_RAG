# DEPO Decomposition #33

- Dataset: `musique`
- Question: How long are the council terms of the city that shares a county with Helvetia?
- Gold answer: four-year

## 1. Explicit Entities
- Helvetia span=(69, 77)

## 2. Entity Masking
- ENTITYA -> Helvetia

Masked question: How long are the council terms of the city that shares a county with ENTITYA?

## 3. Global Best Path
- Helvetia ---- shares ---- county ---- city ---- council ---- terms ---- long ---- How

## 4. Step5 Action Trace
- q1: What is the city that shares a county with Helvetia?
  - consume: Helvetia -> shares -> county -> city
  - produce: q1_answer
- q2: How long are the council terms of q1's answer?
  - consume: q1_answer -> council -> terms -> long
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What is the city that shares a county with Helvetia?
  - depends_on: (none)
- q2: How long are the council terms of q1's answer?
  - depends_on: q1
