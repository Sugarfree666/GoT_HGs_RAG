# DEPO Decomposition #21

- Dataset: `musique`
- Question: What does seal stand for in the U.S. military branch that operates destroyers, as well as the USS Edsall?
- Gold answer: Sea, Air, and Land

## 1. Explicit Entities
- U.S span=(32, 35)
- USS Edsall span=(94, 104)

## 2. Entity Masking
- ENTITYA -> U.S
- ENTITYB -> USS Edsall

Masked question: What does seal stand for in the ENTITYA. military branch that operates destroyers, as well as the ENTITYB?

## 3. Global Best Path
- USS Edsall

## 4. Step5 Action Trace
- q1: What is the U.S. military branch that operates the USS Edsall?
  - consume: USS Edsall
  - produce: q1_answer
- q2: What does seal stand for in the U.S. military branch identified in q1's answer?
  - consume: q1_answer
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What is the U.S. military branch that operates the USS Edsall?
  - depends_on: (none)
- q2: What does seal stand for in the U.S. military branch identified in q1's answer?
  - depends_on: q1
