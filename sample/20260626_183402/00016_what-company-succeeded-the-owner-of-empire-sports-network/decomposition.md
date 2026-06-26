# DEPO Decomposition #16

- Dataset: `musique`
- Question: What company succeeded the owner of Empire Sports Network?
- Gold answer: Time Warner Cable

## 1. Explicit Entities
- Empire Sports Network span=(36, 57)

## 2. Entity Masking
- ENTITYA -> Empire Sports Network

Masked question: What company succeeded the owner of ENTITYA?

## 3. Global Best Path
- Empire Sports Network ---- owner ---- succeeded ---- company

## 4. Step5 Action Trace
- q1: Who is the owner of Empire Sports Network?
  - consume: Empire Sports Network -> owner
  - produce: q1_answer
- q2: What company succeeded q1's answer?
  - consume: q1_answer ---- owner -> succeeded -> company
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the owner of Empire Sports Network?
  - depends_on: (none)
- q2: What company succeeded q1's answer?
  - depends_on: q1
