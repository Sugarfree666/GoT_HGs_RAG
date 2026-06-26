# DEPO Decomposition #26

- Dataset: `musique`
- Question: Who is the spouse of Young Man Luther's author?
- Gold answer: Joan Erikson

## 1. Explicit Entities
- Young Man Luther span=(21, 37)

## 2. Entity Masking
- ENTITYA -> Young Man Luther

Masked question: Who is the spouse of the author of ENTITYA?

## 3. Global Best Path
- Young Man Luther ---- author ---- spouse

## 4. Step5 Action Trace
- q1: Who is the author of Young Man Luther?
  - consume: Young Man Luther -> author
  - produce: q1_answer
- q2: Who is the spouse of q1's answer?
  - consume: q1_answer ---- author -> spouse
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the author of Young Man Luther?
  - depends_on: (none)
- q2: Who is the spouse of q1's answer?
  - depends_on: q1
