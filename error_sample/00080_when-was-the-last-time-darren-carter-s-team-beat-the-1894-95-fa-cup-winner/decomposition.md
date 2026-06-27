# DEPO Decomposition #80

- Dataset: `musique`
- Question: When was the last time Darren Carter's team beat the 1894-95 FA Cup winner?
- Gold answer: 1 December 2010

## 1. Explicit Entities
- Darren Carter span=(23, 36)
- FA Cup span=(61, 67)

## 2. Entity Masking
- ENTITYA -> Darren Carter
- ENTITYB -> FA Cup

Masked question: When was the last time ENTITYA's team beat the 1894-95 ENTITYB winner?

## 3. Global Best Path
- Darren Carter ---- team ---- beat ---- FA Cup ---- winner ---- time ---- last

## 4. Step5 Action Trace
- q1: What is the name of Darren Carter's team?
  - consume: Darren Carter ---- team
  - produce: q1_answer
- q2: When was the last time q1's answer beat the FA Cup winner?
  - consume: q1_answer ---- beat ---- FA Cup ---- winner
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What is the name of Darren Carter's team?
  - depends_on: (none)
- q2: When was the last time q1's answer beat the FA Cup winner?
  - depends_on: q1
