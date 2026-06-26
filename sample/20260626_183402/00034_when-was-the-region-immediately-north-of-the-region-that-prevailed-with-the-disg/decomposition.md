# DEPO Decomposition #34

- Dataset: `musique`
- Question: When was the region immediately north of the region that prevailed with the disgrace of Near East and the terrain feature on which shamal is located created?
- Gold answer: 1930

## 1. Explicit Entities
- Near East span=(88, 97)

## 2. Entity Masking
- ENTITYA -> Near East

Masked question: When was the region immediately north of the region that prevailed with the disgrace of ENTITYA and the terrain feature on which shamal is located created?

## 3. Global Best Path
- Near East

## 4. Step5 Action Trace
- q1: What is the region immediately north of the Near East?
  - consume: Near East
  - produce: q1_answer
- q2: When was the region that is immediately north of q1's answer created?
  - consume: q1_answer
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What is the region immediately north of the Near East?
  - depends_on: (none)
- q2: When was the region that is immediately north of q1's answer created?
  - depends_on: q1
