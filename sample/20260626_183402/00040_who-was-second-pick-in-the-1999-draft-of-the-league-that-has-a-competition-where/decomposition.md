# DEPO Decomposition #40

- Dataset: `musique`
- Question: Who was second pick in the 1999 draft of the league that has a competition where they give out the MLB MVP award after it?
- Gold answer: Josh Beckett

## 1. Explicit Entities
- MLB MVP span=(99, 106)

## 2. Entity Masking
- ENTITYA -> MLB MVP

Masked question: Who was the second pick in the 1999 draft of the league that has a competition where they give out the ENTITYA award after it?

## 3. Global Best Path
- MLB MVP ---- award ---- give ---- they ---- competition ---- league ---- draft ---- pick ---- second

## 4. Step5 Action Trace
- q1: What is the league that has a competition where they give out the MLB MVP award?
  - consume: MLB MVP -> competition -> league
  - produce: q1_answer
- q2: Who was the second pick in the 1999 draft of q1's answer?
  - consume: q1_answer -> draft -> pick -> second
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What is the league that has a competition where they give out the MLB MVP award?
  - depends_on: (none)
- q2: Who was the second pick in the 1999 draft of q1's answer?
  - depends_on: q1
