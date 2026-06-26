# DEPO Decomposition #4

- Dataset: `musique`
- Question: What year did the publisher of Labyrinth end?
- Gold answer: 1986

## 1. Explicit Entities
- Labyrinth span=(31, 40)

## 2. Entity Masking
- ENTITYA -> Labyrinth

Masked question: What year did the publisher of ENTITYA end?

## 3. Global Best Path
- Labyrinth ---- publisher ---- end ---- year

## 4. Step5 Action Trace
- q1: Who is the publisher of Labyrinth?
  - consume: Labyrinth -> publisher
  - produce: q1_answer
- q2: When did q1's answer end?
  - consume: q1_answer ---- publisher -> end
  - produce: q2_answer
- q3: What year did q2's answer end?
  - consume: q2_answer ---- end -> year
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: Who is the publisher of Labyrinth?
  - depends_on: (none)
- q2: When did q1's answer end?
  - depends_on: q1
- q3: What year did q2's answer end?
  - depends_on: q2
