# DEPO Decomposition #41

- Dataset: `musique`
- Question: The Commission of Truth and Friendship established governments for Bustami's birth country and a second country that had who as president after declaring independence?
- Gold answer: Francisco Guterres

## 1. Explicit Entities
- The Commission of Truth and Friendship span=(0, 38)
- Bustami span=(67, 74)

## 2. Entity Masking
- ENTITYA -> The Commission of Truth and Friendship
- ENTITYB -> Bustami

Masked question: ENTITYA established governments for ENTITYB's birth country and a second country that had who as president after declaring independence?

## 3. Global Best Path
- Bustami ---- birth ---- country ---- governments ---- established ---- The Commission of Truth and Friendship

## 4. Step5 Action Trace
- q1: What is Bustami's birth country?
  - consume: Bustami ---- birth ---- country
  - produce: q1_answer
- q2: What governments were established by The Commission of Truth and Friendship for q1's answer?
  - consume: q1_answer ---- governments ---- established ---- The Commission of Truth and Friendship
  - produce: q2_answer
- q3: What is the second country that had who as president after declaring independence?
  - consume: q2_answer ---- declaring ---- independence
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: What is Bustami's birth country?
  - depends_on: (none)
- q2: What governments were established by The Commission of Truth and Friendship for q1's answer?
  - depends_on: q1
- q3: What is the second country that had who as president after declaring independence?
  - depends_on: q2
