# DEPO Decomposition #15

- Dataset: `musique`
- Question: What year did the Governor of the city where the basilica named after the same saint as the one that Mantua Cathedral is dedicated to die?
- Gold answer: 1952

## 1. Explicit Entities
- Mantua Cathedral span=(101, 117)

## 2. Entity Masking
- ENTITYA -> Mantua Cathedral

Masked question: What year did the Governor of the city where the basilica named after the same saint as the one that ENTITYA is dedicated to die?

## 3. Global Best Path
- Mantua Cathedral ---- same ---- saint ---- basilica ---- city ---- named ---- Governor ---- die ---- year

## 4. Step5 Action Trace
- q1: What is the saint that Mantua Cathedral is dedicated to?
  - consume: Mantua Cathedral -> same -> saint
  - produce: q1_answer
- q2: What is the basilica named after q1's answer?
  - consume: q1_answer -> basilica
  - produce: q2_answer
- q3: What is the city where the basilica named after q2's answer is located?
  - consume: q2_answer -> city
  - produce: q3_answer
- q4: Who is the Governor of the city q3's answer?
  - consume: q3_answer -> Governor
  - produce: q4_answer
- q5: What year did q4's answer die?
  - consume: q4_answer -> die -> year
  - produce: q5_answer

## 5. Atomic Question DAG
- q1: What is the saint that Mantua Cathedral is dedicated to?
  - depends_on: (none)
- q2: What is the basilica named after q1's answer?
  - depends_on: q1
- q3: What is the city where the basilica named after q2's answer is located?
  - depends_on: q2
- q4: Who is the Governor of the city q3's answer?
  - depends_on: q3
- q5: What year did q4's answer die?
  - depends_on: q4
