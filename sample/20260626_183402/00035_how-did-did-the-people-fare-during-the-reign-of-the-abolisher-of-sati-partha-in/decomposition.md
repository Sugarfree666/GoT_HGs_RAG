# DEPO Decomposition #35

- Dataset: `musique`
- Question: How did did the people fare during the reign of the abolisher of sati partha in India?
- Gold answer: a long period of prosperity for the British people

## 1. Explicit Entities
- India span=(80, 85)

## 2. Entity Masking
- ENTITYA -> India

Masked question: How did the people fare during the reign of the abolisher of sati partha in ENTITYA?

## 3. Global Best Path
- India ---- abolisher ---- reign ---- fare ---- How

## 4. Step5 Action Trace
- q1: Who is the abolisher of sati in India?
  - consume: India -> abolisher
  - produce: q1_answer
- q2: What was the reign of q1's answer?
  - consume: q1_answer ---- abolisher -> reign
  - produce: q2_answer
- q3: How did the people fare during the reign of q2's answer?
  - consume: q2_answer ---- reign -> fare
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: Who is the abolisher of sati in India?
  - depends_on: (none)
- q2: What was the reign of q1's answer?
  - depends_on: q1
- q3: How did the people fare during the reign of q2's answer?
  - depends_on: q2
