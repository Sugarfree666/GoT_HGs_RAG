# DEPO Decomposition #5

- Dataset: `musique`
- Question: When was Lady Godiva's birthplace abolished?
- Gold answer: 918

## 1. Explicit Entities
- Lady Godiva span=(9, 20)

## 2. Entity Masking
- ENTITYA -> Lady Godiva

Masked question: When was ENTITYA's birthplace abolished?

## 3. Global Best Path
- Lady Godiva ---- birthplace ---- abolished

## 4. Step5 Action Trace
- q1: What is the birthplace of Lady Godiva?
  - consume: Lady Godiva -> birthplace
  - produce: q1_answer
- q2: When was q1's answer abolished?
  - consume: q1_answer ---- birthplace -> abolished
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What is the birthplace of Lady Godiva?
  - depends_on: (none)
- q2: When was q1's answer abolished?
  - depends_on: q1
