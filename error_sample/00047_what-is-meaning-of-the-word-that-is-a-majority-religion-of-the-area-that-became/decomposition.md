# DEPO Decomposition #47

- Dataset: `musique`
- Question: what is meaning of the word that is a majority religion of the area that became India when the country origin of Mizraab was created in Arabic dictionary?
- Gold answer: the country of India

## 1. Explicit Entities
- India span=(80, 85)
- Mizraab span=(113, 120)
- Arabic span=(136, 142)

## 2. Entity Masking
- ENTITYA -> India
- ENTITYB -> Mizraab
- ENTITYC -> Arabic

Masked question: What is the meaning of the word that is a majority religion of the area that became ENTITYA when the country of ENTITYB was created in the ENTITYC dictionary?

## 3. Global Best Path
- Arabic ---- dictionary ---- created ---- became ---- religion ---- majority ---- area ---- word ---- meaning

## 4. Step5 Action Trace
- q1: What area became India?
  - consume: area ---- became ---- India
  - produce: q1_answer
- q2: What is the majority religion of q1's answer?
  - consume: q1_answer ---- majority ---- religion
  - produce: q2_answer
- q3: What was created in the Arabic dictionary?
  - consume: Arabic ---- dictionary ---- created
  - produce: q3_answer
- q4: What is the meaning of the word that is the majority religion of q2's answer?
  - consume: q2_answer ---- word ---- meaning
  - produce: q4_answer

## 5. Atomic Question DAG
- q1: What area became India?
  - depends_on: (none)
- q2: What is the majority religion of q1's answer?
  - depends_on: q1
- q3: What was created in the Arabic dictionary?
  - depends_on: (none)
- q4: What is the meaning of the word that is the majority religion of q2's answer?
  - depends_on: q2
