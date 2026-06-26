# DEPO Decomposition #3

- Dataset: `musique`
- Question: What county is Erik Hort's birthplace a part of?
- Gold answer: Rockland County

## 1. Explicit Entities
- Erik Hort span=(15, 24)

## 2. Entity Masking
- ENTITYA -> Erik Hort

Masked question: What county is ENTITYA's birthplace a part of?

## 3. Global Best Path
- Erik Hort ---- birthplace ---- part ---- county

## 4. Step5 Action Trace
- q1: Where is Erik Hort's birthplace?
  - consume: Erik Hort -> birthplace
  - produce: q1_answer
- q2: What county is q1's answer a part of?
  - consume: q1_answer ---- birthplace -> part -> county
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Where is Erik Hort's birthplace?
  - depends_on: (none)
- q2: What county is q1's answer a part of?
  - depends_on: q1
