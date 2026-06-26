# DEPO Decomposition #10

- Dataset: `musique`
- Question: What is the Till dom ensamma performer's birth date?
- Gold answer: 11 September 1962

## 1. Explicit Entities
- Till span=(12, 16)

## 2. Entity Masking
- ENTITYA -> Till

Masked question: What is the birth date of the ENTITYA dom ensamma performer?

## 3. Global Best Path
- Till ---- performer ---- ensamma ---- dom ---- date ---- birth

## 4. Step5 Action Trace
- q1: Who is the performer of Till?
  - consume: Till -> performer
  - produce: q1_answer
- q2: What is the birth date of q1's answer?
  - consume: q1_answer ---- performer -> birth date
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the performer of Till?
  - depends_on: (none)
- q2: What is the birth date of q1's answer?
  - depends_on: q1
