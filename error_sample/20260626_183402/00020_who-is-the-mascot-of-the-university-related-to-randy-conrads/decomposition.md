# DEPO Decomposition #20

- Dataset: `musique`
- Question: Who is the mascot of the university related to Randy Conrads?
- Gold answer: Benny Beaver

## 1. Explicit Entities
- Randy Conrads span=(47, 60)

## 2. Entity Masking
- ENTITYA -> Randy Conrads

Masked question: Who is the mascot of the university related to ENTITYA?

## 3. Global Best Path
- Randy Conrads ---- university ---- related ---- mascot

## 4. Step5 Action Trace
- q1: What is the university related to Randy Conrads?
  - consume: Randy Conrads -> university
  - produce: q1_answer
- q2: Who is the mascot of q1's answer?
  - consume: q1_answer ---- university -> mascot
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What is the university related to Randy Conrads?
  - depends_on: (none)
- q2: Who is the mascot of q1's answer?
  - depends_on: q1
