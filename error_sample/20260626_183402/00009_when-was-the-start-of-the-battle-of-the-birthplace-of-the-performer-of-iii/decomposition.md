# DEPO Decomposition #9

- Dataset: `musique`
- Question: When was the start of the battle of the birthplace of the performer of III?
- Gold answer: December 14, 1814

## 1. Explicit Entities
- III span=(71, 74)

## 2. Entity Masking
- ENTITYA -> III

Masked question: When was the start of the battle of the birthplace of the performer of ENTITYA?

## 3. Global Best Path
- III ---- performer ---- birthplace ---- battle ---- start

## 4. Step5 Action Trace
- q1: Who is the performer of III?
  - consume: III -> performer
  - produce: q1_answer
- q2: What is the birthplace of q1's answer?
  - consume: q1_answer ---- performer -> birthplace
  - produce: q2_answer
- q3: When was the start of the battle of q2's answer?
  - consume: q2_answer ---- birthplace -> battle -> start
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: Who is the performer of III?
  - depends_on: (none)
- q2: What is the birthplace of q1's answer?
  - depends_on: q1
- q3: When was the start of the battle of q2's answer?
  - depends_on: q2
