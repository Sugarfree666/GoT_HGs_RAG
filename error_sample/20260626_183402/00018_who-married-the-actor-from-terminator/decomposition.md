# DEPO Decomposition #18

- Dataset: `musique`
- Question: Who married the actor from Terminator?
- Gold answer: Maria Shriver

## 1. Explicit Entities
- Terminator span=(27, 37)

## 2. Entity Masking
- ENTITYA -> Terminator

Masked question: Who married the actor from ENTITYA?

## 3. Global Best Path
- Terminator ---- actor ---- married

## 4. Step5 Action Trace
- q1: Who is the actor from the movie Terminator?
  - consume: Terminator -> actor
  - produce: q1_answer
- q2: Who is married to q1's answer?
  - consume: q1_answer ---- actor -> married
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the actor from the movie Terminator?
  - depends_on: (none)
- q2: Who is married to q1's answer?
  - depends_on: q1
