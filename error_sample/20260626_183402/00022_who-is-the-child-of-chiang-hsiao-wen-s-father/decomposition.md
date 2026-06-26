# DEPO Decomposition #22

- Dataset: `musique`
- Question: Who is the child of Chiang Hsiao-wen's father?
- Gold answer: Chiang Hsiao-wu

## 1. Explicit Entities
- Chiang Hsiao-wen span=(20, 36)

## 2. Entity Masking
- ENTITYA -> Chiang Hsiao-wen

Masked question: Who is the child of ENTITYA's father?

## 3. Global Best Path
- Chiang Hsiao-wen ---- father ---- child

## 4. Step5 Action Trace
- q1: Who is the father of Chiang Hsiao-wen?
  - consume: Chiang Hsiao-wen -> father
  - produce: q1_answer
- q2: Who is the child of q1's answer?
  - consume: q1_answer ---- father -> child
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the father of Chiang Hsiao-wen?
  - depends_on: (none)
- q2: Who is the child of q1's answer?
  - depends_on: q1
