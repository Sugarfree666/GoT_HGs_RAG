# DEPO Decomposition #100

- Dataset: `musique`
- Question: When was the region immediately north of the region where the country in which Aluf can be found is located and the Persian Gulf established?
- Gold answer: 1932

## 1. Explicit Entities
- Aluf span=(79, 83)
- Persian Gulf span=(116, 128)

## 2. Entity Masking
- ENTITYA -> Aluf
- ENTITYB -> Persian Gulf

Masked question: When was the region immediately north of the region where the country in which ENTITYA can be found is located and the ENTITYB established?

## 3. Global Best Path
- Persian Gulf ---- established ---- region ---- north ---- found ---- country ---- can ---- region

## 4. Step5 Action Trace
- q1: What is the country in which Aluf can be found?
  - consume: country ---- where ---- Aluf ---- can ---- be ---- found
  - produce: q1_answer
- q2: What region is where q1's answer is located?
  - consume: region ---- where ---- q1_answer ---- is ---- located
  - produce: q2_answer
- q3: What region is immediately north of q2's answer?
  - consume: region ---- north ---- immediately ---- q2_answer
  - produce: q3_answer
- q4: When was the Persian Gulf established?
  - consume: Persian Gulf ---- established
  - produce: q4_answer
- q5: When was q3's answer established?
  - consume: q3_answer ---- q4_answer
  - produce: q5_answer

## 5. Atomic Question DAG
- q1: What is the country in which Aluf can be found?
  - depends_on: (none)
- q2: What region is where q1's answer is located?
  - depends_on: q1
- q3: What region is immediately north of q2's answer?
  - depends_on: q2
- q4: When was the Persian Gulf established?
  - depends_on: (none)
- q5: When was q3's answer established?
  - depends_on: q3, q4
