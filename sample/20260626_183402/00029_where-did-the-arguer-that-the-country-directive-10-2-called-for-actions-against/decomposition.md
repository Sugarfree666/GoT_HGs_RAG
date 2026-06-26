# DEPO Decomposition #29

- Dataset: `musique`
- Question: Where did the arguer that the country Directive 10/2 called for actions against had become an imperialist power declare he would intervene in the Korean conflict?
- Gold answer: the Politburo

## 1. Explicit Entities
- Directive 10/2 span=(38, 52)
- Korean span=(146, 152)

## 2. Entity Masking
- ENTITYA -> Directive 10/2
- ENTITYB -> Korean

Masked question: Where did the arguer that the country ENTITYA called for actions against had become an imperialist power declare he would intervene in the ENTITYB conflict?

## 3. Global Best Path
- Korean ---- conflict ---- intervene ---- would ---- declare ---- become ---- arguer ---- called ---- country ---- against ---- actions

## 4. Step5 Action Trace
- q1: Who is the arguer that declared he would intervene in the Korean conflict?
  - consume: Korean -> conflict -> intervene -> declare -> arguer
  - produce: q1_answer
- q2: Where did q1's answer declare he would intervene in the conflict related to Directive 10/2?
  - consume: q1_answer ---- arguer -> called -> country -> Directive 10/2 -> actions -> against
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the arguer that declared he would intervene in the Korean conflict?
  - depends_on: (none)
- q2: Where did q1's answer declare he would intervene in the conflict related to Directive 10/2?
  - depends_on: q1
