# DEPO Decomposition #73

- Dataset: `musique`
- Question: Which county shares a border with the county where the most populous city in the state where Washington State Prison can be found is located?
- Gold answer: Presque Isle County

## 1. Explicit Entities
- Washington State Prison span=(93, 116)

## 2. Entity Masking
- ENTITYA -> Washington State Prison

Masked question: Which county shares a border with the county where the most populous city in the state where ENTITYA is located?

## 3. Global Best Path
- Washington State Prison ---- city ---- populous ---- state ---- located ---- county ---- shares ---- county

## 4. Step5 Action Trace
- q1: What is the most populous city in the state where Washington State Prison can be found?
  - consume: Washington State Prison ---- city ---- populous
  - produce: q1_answer
- q2: In which state is q1's answer located?
  - consume: q1_answer ---- state ---- located
  - produce: q2_answer
- q3: What county is located in q2's answer?
  - consume: q2_answer ---- county
  - produce: q3_answer
- q4: Which county shares a border with q3's answer?
  - consume: q3_answer ---- shares ---- county
  - produce: q4_answer

## 5. Atomic Question DAG
- q1: What is the most populous city in the state where Washington State Prison can be found?
  - depends_on: (none)
- q2: In which state is q1's answer located?
  - depends_on: q1
- q3: What county is located in q2's answer?
  - depends_on: q2
- q4: Which county shares a border with q3's answer?
  - depends_on: q3
