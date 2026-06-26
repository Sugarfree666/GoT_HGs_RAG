# DEPO Decomposition #36

- Dataset: `musique`
- Question: What is the position of the 1st governor general of India?
- Gold answer: Governor-General of India

## 1. Explicit Entities
- India span=(52, 57)

## 2. Entity Masking
- ENTITYA -> India

Masked question: What is the position of the 1st governor general of ENTITYA?

## 3. Global Best Path
- India ---- general ---- 1st ---- governor ---- position

## 4. Step5 Action Trace
- q1: Who is the 1st governor general of India?
  - consume: India -> general -> 1st -> governor
  - produce: q1_answer
- q2: What is the position of q1's answer?
  - consume: q1_answer ---- governor -> position
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the 1st governor general of India?
  - depends_on: (none)
- q2: What is the position of q1's answer?
  - depends_on: q1
