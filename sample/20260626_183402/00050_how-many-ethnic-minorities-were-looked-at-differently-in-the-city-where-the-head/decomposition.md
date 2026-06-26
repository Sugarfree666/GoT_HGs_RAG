# DEPO Decomposition #50

- Dataset: `musique`
- Question: How many ethnic minorities were looked at differently in the city where the headquarters of the only group larger than Långa nätter's record label is located?
- Gold answer: two

## 1. Explicit Entities
- Långa nätter span=(119, 131)

## 2. Entity Masking
- ENTITYA -> Långa nätter

Masked question: How many ethnic minorities were looked at differently in the city where the headquarters of the only group larger than ENTITYA's record label is located?

## 3. Global Best Path
- Långa nätter ---- label ---- record ---- larger ---- group ---- only ---- headquarters ---- located ---- city

## 4. Step5 Action Trace
- q1: What is the record label of Långa nätter?
  - consume: Långa nätter -> label
  - produce: q1_answer
- q2: What is the only group larger than the record label of Långa nätter?
  - consume: q1_answer -> group ---- only
  - produce: q2_answer
- q3: Where is the headquarters of the only group larger than the record label of Långa nätter located?
  - consume: q2_answer -> headquarters
  - produce: q3_answer
- q4: In which city is the headquarters of the only group larger than Långa nätter's record label located?
  - consume: q3_answer -> city
  - produce: q4_answer
- q5: How many ethnic minorities were looked at differently in the city where the headquarters of the only group larger than Långa nätter's record label is located?
  - consume: q4_answer -> ethnic minorities ---- looked at differently
  - produce: q5_answer

## 5. Atomic Question DAG
- q1: What is the record label of Långa nätter?
  - depends_on: (none)
- q2: What is the only group larger than the record label of Långa nätter?
  - depends_on: q1
- q3: Where is the headquarters of the only group larger than the record label of Långa nätter located?
  - depends_on: q2
- q4: In which city is the headquarters of the only group larger than Långa nätter's record label located?
  - depends_on: q3
- q5: How many ethnic minorities were looked at differently in the city where the headquarters of the only group larger than Långa nätter's record label is located?
  - depends_on: q4
