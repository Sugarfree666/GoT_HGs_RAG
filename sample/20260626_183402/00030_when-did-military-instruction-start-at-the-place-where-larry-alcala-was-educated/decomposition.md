# DEPO Decomposition #30

- Dataset: `musique`
- Question: When did military instruction start at the place where Larry Alcala was educated?
- Gold answer: 1912

## 1. Explicit Entities
- Larry Alcala span=(55, 67)

## 2. Entity Masking
- ENTITYA -> Larry Alcala

Masked question: When did military instruction start at the place where ENTITYA was educated?

## 3. Global Best Path
- Larry Alcala ---- place ---- educated ---- start ---- When

## 4. Step5 Action Trace
- q1: What is the place where Larry Alcala was educated?
  - consume: Larry Alcala -> place
  - produce: q1_answer
- q2: When did military instruction start at q1's answer?
  - consume: q1_answer ---- place -> start
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What is the place where Larry Alcala was educated?
  - depends_on: (none)
- q2: When did military instruction start at q1's answer?
  - depends_on: q1
