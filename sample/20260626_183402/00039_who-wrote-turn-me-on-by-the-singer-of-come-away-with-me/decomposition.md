# DEPO Decomposition #39

- Dataset: `musique`
- Question: Who wrote Turn Me On by the singer of Come Away with Me?
- Gold answer: John D. Loudermilk

## 1. Explicit Entities
- Turn Me On span=(10, 20)
- Come Away with Me span=(38, 55)

## 2. Entity Masking
- ENTITYA -> Turn Me On
- ENTITYB -> Come Away with Me

Masked question: Who wrote ENTITYA by the singer of ENTITYB?

## 3. Global Best Path
- Come Away with Me ---- singer ---- wrote ---- Turn Me On

## 4. Step5 Action Trace
- q1: Who is the singer of the song Come Away with Me?
  - consume: Come Away with Me -> singer
  - produce: q1_answer
- q2: Who wrote the song Turn Me On by q1's answer?
  - consume: q1_answer ---- singer -> wrote -> Turn Me On
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the singer of the song Come Away with Me?
  - depends_on: (none)
- q2: Who wrote the song Turn Me On by q1's answer?
  - depends_on: q1
