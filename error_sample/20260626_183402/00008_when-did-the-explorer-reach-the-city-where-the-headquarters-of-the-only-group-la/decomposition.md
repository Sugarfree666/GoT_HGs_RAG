# DEPO Decomposition #8

- Dataset: `musique`
- Question: When did the explorer reach the city where the headquarters of the only group larger than Vilaiyaadu Mankatha's record label is located?
- Gold answer: August 3, 1769

## 1. Explicit Entities
- Vilaiyaadu Mankatha span=(90, 109)

## 2. Entity Masking
- ENTITYA -> Vilaiyaadu Mankatha

Masked question: When did the explorer reach the city where the headquarters of the only group larger than ENTITYA's record label is located?

## 3. Global Best Path
- Vilaiyaadu Mankatha ---- label ---- located ---- city ---- headquarters ---- larger ---- group ---- only

## 4. Step5 Action Trace
- q1: What is the record label of Vilaiyaadu Mankatha?
  - consume: Vilaiyaadu Mankatha -> label
  - produce: q1_answer
- q2: Where is the headquarters of the only group larger than the record label of q1's answer located?
  - consume: q1_answer -> located
  - produce: q2_answer
- q3: What city is where the headquarters of the only group larger than the record label of Vilaiyaadu Mankatha is located?
  - consume: q2_answer -> city
  - produce: q3_answer
- q4: When did the explorer reach the city q3's answer?
  - consume: q3_answer
  - produce: q4_answer

## 5. Atomic Question DAG
- q1: What is the record label of Vilaiyaadu Mankatha?
  - depends_on: (none)
- q2: Where is the headquarters of the only group larger than the record label of q1's answer located?
  - depends_on: q1
- q3: What city is where the headquarters of the only group larger than the record label of Vilaiyaadu Mankatha is located?
  - depends_on: q2
- q4: When did the explorer reach the city q3's answer?
  - depends_on: q3
