# DEPO Decomposition #99

- Dataset: `musique`
- Question: How many times did the plague occur in the birth place of Concerto in C Major Op 3 6's composer?
- Gold answer: 22

## 1. Explicit Entities
- Concerto span=(58, 66)
- C Major Op span=(70, 80)

## 2. Entity Masking
- ENTITYA -> Concerto
- ENTITYB -> C Major Op

Masked question: How many times did the plague occur in the birthplace of the composer of ENTITYA in ENTITYB 3 6?

## 3. Global Best Path
- C Major Op ---- Concerto ---- composer ---- birthplace ---- occur ---- many ---- times ---- How

## 4. Step5 Action Trace
- q1: Who is the composer of Concerto in C Major Op?
  - consume: C Major Op ---- Concerto ---- composer
  - produce: q1_answer
- q2: What is the birthplace of q1's answer?
  - consume: q1_answer ---- birthplace
  - produce: q2_answer
- q3: How many times did the plague occur in q2's answer?
  - consume: q2_answer ---- occur
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: Who is the composer of Concerto in C Major Op?
  - depends_on: (none)
- q2: What is the birthplace of q1's answer?
  - depends_on: q1
- q3: How many times did the plague occur in q2's answer?
  - depends_on: q2
