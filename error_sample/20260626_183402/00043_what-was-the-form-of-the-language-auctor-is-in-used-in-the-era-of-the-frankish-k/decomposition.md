# DEPO Decomposition #43

- Dataset: `musique`
- Question: What was the form of the language Auctor is in, used in the era of the Frankish king who created the Holy Roman Empire, later known as?
- Gold answer: Medieval Latin

## 1. Explicit Entities
- Auctor span=(34, 40)
- Frankish span=(71, 79)
- Holy Roman Empire span=(101, 118)

## 2. Entity Masking
- ENTITYA -> Auctor
- ENTITYB -> Frankish
- ENTITYC -> Holy Roman Empire

Masked question: What was the form of the language ENTITYA is in, used in the era of the ENTITYB king who created the ENTITYC?

## 3. Global Best Path
- Frankish ---- king ---- Holy Roman Empire ---- created ---- era ---- form ---- used ---- language ---- Auctor

## 4. Step5 Action Trace
- q1: Who was the Frankish king who created the Holy Roman Empire?
  - consume: Frankish -> king
  - produce: q1_answer
- q2: What was the era of q1's answer?
  - consume: q1_answer ---- king -> era
  - produce: q2_answer
- q3: What was the form of the language Auctor is in, used in the era of q2's answer?
  - consume: q2_answer ---- era -> form ---- Auctor -> language
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: Who was the Frankish king who created the Holy Roman Empire?
  - depends_on: (none)
- q2: What was the era of q1's answer?
  - depends_on: q1
- q3: What was the form of the language Auctor is in, used in the era of q2's answer?
  - depends_on: q2
