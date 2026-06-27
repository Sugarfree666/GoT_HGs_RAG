# DEPO Decomposition #95

- Dataset: `musique`
- Question: Who had the lowest batting average in the league where the team with the most games in the series after which the MLB MVP is awarded played?
- Gold answer: Bill Bergen

## 1. Explicit Entities
- MLB MVP span=(114, 121)

## 2. Entity Masking
- ENTITYA -> MLB MVP

Masked question: Who had the lowest batting average in the league where the team with the most games in the series after which the ENTITYA is awarded played?

## 3. Global Best Path
- MLB MVP ---- awarded ---- which ---- played ---- league ---- team ---- series

## 4. Step5 Action Trace
- q1: Which league is the MLB MVP awarded in?
  - consume: MLB MVP ---- awarded
  - produce: q1_answer
- q2: What team played in the series in the league where q1's answer is?
  - consume: q1_answer ---- played ---- team ---- series
  - produce: q2_answer
- q3: Who had the lowest batting average in the league where q2's answer played?
  - consume: q2_answer ---- lowest ---- batting ---- average
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: Which league is the MLB MVP awarded in?
  - depends_on: (none)
- q2: What team played in the series in the league where q1's answer is?
  - depends_on: q1
- q3: Who had the lowest batting average in the league where q2's answer played?
  - depends_on: q2
