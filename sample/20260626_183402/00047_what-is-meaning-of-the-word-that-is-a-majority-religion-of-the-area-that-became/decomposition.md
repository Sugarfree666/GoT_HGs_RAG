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

Masked question: What is the meaning of the word that is a majority religion of the area that became ENTITYA when the country of origin of ENTITYB was created in the ENTITYC dictionary?

## 3. Global Best Path
- Mizraab ---- origin ---- country ---- created ---- became ---- religion ---- majority ---- area ---- word ---- meaning

## 4. Step5 Action Trace
- q1: What is the majority religion of the area that became India when the country of Mizraab was created?
  - consume: Mizraab -> origin -> country -> created -> became -> religion -> majority -> area
  - produce: q1_answer
- q2: What is the meaning of the word that corresponds to q1's answer in the Arabic dictionary?
  - consume: q1_answer ---- word -> meaning
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What is the majority religion of the area that became India when the country of Mizraab was created?
  - depends_on: (none)
- q2: What is the meaning of the word that corresponds to q1's answer in the Arabic dictionary?
  - depends_on: q1
