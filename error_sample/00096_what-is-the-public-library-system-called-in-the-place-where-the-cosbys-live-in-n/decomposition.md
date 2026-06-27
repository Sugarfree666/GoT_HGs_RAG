# DEPO Decomposition #96

- Dataset: `musique`
- Question: What is the public library system called in the place where the Cosbys live in New York?
- Gold answer: Brooklyn Public Library

## 1. Explicit Entities
- Cosbys span=(64, 70)
- New York span=(79, 87)

## 2. Entity Masking
- ENTITYA -> Cosbys
- ENTITYB -> New York

Masked question: What is the public library system called in the place where the ENTITYA live in ENTITYB?

## 3. Global Best Path
- Cosbys ---- live ---- New York ---- place ---- called ---- What

## 4. Step5 Action Trace
- q1: What is the place where the Cosbys live in New York?
  - consume: Cosbys ---- live ---- New York ---- place
  - produce: q1_answer
- q2: What is q1's answer called?
  - consume: q1_answer ---- called
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What is the place where the Cosbys live in New York?
  - depends_on: (none)
- q2: What is q1's answer called?
  - depends_on: q1
