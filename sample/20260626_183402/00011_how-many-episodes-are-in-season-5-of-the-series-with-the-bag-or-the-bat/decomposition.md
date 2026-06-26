# DEPO Decomposition #11

- Dataset: `musique`
- Question: How many episodes are in season 5 of the series with The Bag or the Bat?
- Gold answer: 12

## 1. Explicit Entities
- The Bag or the Bat span=(53, 71)

## 2. Entity Masking
- ENTITYA -> The Bag or the Bat

Masked question: How many episodes are in season 5 of the series ENTITYA?

## 3. Global Best Path
- The Bag or the Bat ---- series

## 4. Step5 Action Trace
- q1: What is the series associated with The Bag or the Bat?
  - consume: The Bag or the Bat -> series
  - produce: q1_answer
- q2: How many episodes are in season 5 of q1's answer?
  - consume: q1_answer ---- series -> season 5 -> episodes
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What is the series associated with The Bag or the Bat?
  - depends_on: (none)
- q2: How many episodes are in season 5 of q1's answer?
  - depends_on: q1
