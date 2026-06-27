# DEPO Decomposition #74

- Dataset: `musique`
- Question: What is the meaning of the name of the city where the Yongle emperor greeted the person to whom the edict was addressed?
- Gold answer: "Southern Capital"

## 1. Explicit Entities
- Yongle span=(54, 60)

## 2. Entity Masking
- ENTITYA -> Yongle

Masked question: What is the meaning of the name of the city where the ENTITYA emperor greeted the person to whom the edict was addressed?

## 3. Global Best Path
- Yongle ---- emperor ---- greeted ---- addressed ---- edict ---- person ---- city ---- name ---- meaning

## 4. Step5 Action Trace
- q1: Who is the person that the Yongle emperor greeted?
  - consume: Yongle ---- emperor ---- greeted ---- person
  - produce: q1_answer
- q2: What is the edict addressed to q1's answer?
  - consume: q1_answer ---- addressed ---- edict
  - produce: q2_answer
- q3: What city is associated with q2's answer?
  - consume: q2_answer ---- city
  - produce: q3_answer
- q4: What is the name of q3's answer?
  - consume: q3_answer ---- name
  - produce: q4_answer
- q5: What is the meaning of the name q4's answer?
  - consume: q4_answer ---- meaning
  - produce: q5_answer

## 5. Atomic Question DAG
- q1: Who is the person that the Yongle emperor greeted?
  - depends_on: (none)
- q2: What is the edict addressed to q1's answer?
  - depends_on: q1
- q3: What city is associated with q2's answer?
  - depends_on: q2
- q4: What is the name of q3's answer?
  - depends_on: q3
- q5: What is the meaning of the name q4's answer?
  - depends_on: q4
