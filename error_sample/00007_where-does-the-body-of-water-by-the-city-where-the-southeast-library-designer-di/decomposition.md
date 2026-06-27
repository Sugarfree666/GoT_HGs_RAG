# DEPO Decomposition #7

- Dataset: `musique`
- Question: Where does the body of water by the city where the Southeast Library designer died empty into the Gulf of Mexico?
- Gold answer: the Mississippi River Delta

## 1. Explicit Entities
- Southeast Library span=(51, 68)
- Gulf of Mexico span=(98, 112)

## 2. Entity Masking
- ENTITYA -> Southeast Library
- ENTITYB -> Gulf of Mexico

Masked question: Where does the body of water by the city where the ENTITYA designer died empty into the ENTITYB?

## 3. Global Best Path
- Gulf of Mexico ---- empty ---- body ---- water ---- died ---- designer ---- Southeast Library

## 4. Step5 Action Trace
- q1: Who is the designer of the Southeast Library that died?
  - consume: Southeast Library ---- designer ---- died
  - produce: q1_answer
- q2: Where does the body of water associated with q1's answer empty into?
  - consume: body ---- water ---- empty ---- q1_answer
  - produce: q2_answer
- q3: What is the Gulf of Mexico?
  - consume: Gulf of Mexico
  - produce: q3_answer
- q4: Where does the body of water by q2's answer empty into the Gulf of Mexico?
  - consume: q2_answer ---- q3_answer
  - produce: q4_answer

## 5. Atomic Question DAG
- q1: Who is the designer of the Southeast Library that died?
  - depends_on: (none)
- q2: Where does the body of water associated with q1's answer empty into?
  - depends_on: q1
- q3: What is the Gulf of Mexico?
  - depends_on: (none)
- q4: Where does the body of water by q2's answer empty into the Gulf of Mexico?
  - depends_on: q2, q3
