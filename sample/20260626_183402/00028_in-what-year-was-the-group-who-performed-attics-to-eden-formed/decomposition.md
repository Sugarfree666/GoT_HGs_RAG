# DEPO Decomposition #28

- Dataset: `musique`
- Question: In what year was the group who performed Attics To Eden formed?
- Gold answer: 2005

## 1. Explicit Entities
- Attics To Eden span=(41, 55)

## 2. Entity Masking
- ENTITYA -> Attics To Eden

Masked question: In what year was the group who performed ENTITYA formed?

## 3. Global Best Path
- Attics To Eden

## 4. Step5 Action Trace
- q1: Who is the group that performed the song Attics To Eden?
  - consume: Attics To Eden
  - produce: q1_answer
- q2: In what year was q1's answer formed?
  - consume: q1_answer
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the group that performed the song Attics To Eden?
  - depends_on: (none)
- q2: In what year was q1's answer formed?
  - depends_on: q1
