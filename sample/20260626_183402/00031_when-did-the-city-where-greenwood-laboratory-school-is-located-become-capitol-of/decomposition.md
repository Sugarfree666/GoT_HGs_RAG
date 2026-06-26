# DEPO Decomposition #31

- Dataset: `musique`
- Question: When did the city where Greenwood Laboratory School is located become capitol of the state where the screenwriter of The Poor Boob was born?
- Gold answer: 1839

## 1. Explicit Entities
- Greenwood Laboratory School span=(24, 51)
- The Poor Boob span=(117, 130)

## 2. Entity Masking
- ENTITYA -> Greenwood Laboratory School
- ENTITYB -> The Poor Boob

Masked question: When did the city where ENTITYA is located become the capitol of the state where the screenwriter of ENTITYB was born?

## 3. Global Best Path
- The Poor Boob ---- screenwriter ---- state ---- born ---- capitol ---- become ---- city ---- located ---- Greenwood Laboratory School

## 4. Step5 Action Trace
- q1: Who is the screenwriter of The Poor Boob?
  - consume: The Poor Boob -> screenwriter -> state -> born
  - produce: q1_answer
- q2: What city became the capitol of the state where q1's answer was born?
  - consume: q1_answer ---- state -> capitol -> city
  - produce: q2_answer
- q3: When did q2's answer become the city where Greenwood Laboratory School is located?
  - consume: q2_answer -> located -> Greenwood Laboratory School
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: Who is the screenwriter of The Poor Boob?
  - depends_on: (none)
- q2: What city became the capitol of the state where q1's answer was born?
  - depends_on: q1
- q3: When did q2's answer become the city where Greenwood Laboratory School is located?
  - depends_on: q2
