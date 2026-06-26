# DEPO Decomposition #23

- Dataset: `musique`
- Question: Where was the person who wrote about the rioting being a dividing factor in Birmingham educated?
- Gold answer: University of Glasgow

## 1. Explicit Entities
- Birmingham span=(76, 86)

## 2. Entity Masking
- ENTITYA -> Birmingham

Masked question: Where was the person who wrote about the rioting being a dividing factor in ENTITYA educated?

## 3. Global Best Path
- Birmingham ---- dividing ---- factor ---- rioting ---- wrote ---- educated ---- person

## 4. Step5 Action Trace
- q1: Who is the person who wrote about the rioting being a dividing factor in Birmingham?
  - consume: Birmingham -> dividing -> factor -> rioting -> wrote -> person
  - produce: q1_answer
- q2: Where was q1's answer educated?
  - consume: q1_answer -> educated
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the person who wrote about the rioting being a dividing factor in Birmingham?
  - depends_on: (none)
- q2: Where was q1's answer educated?
  - depends_on: q1
