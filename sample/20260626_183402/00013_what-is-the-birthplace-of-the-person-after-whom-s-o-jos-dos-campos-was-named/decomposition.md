# DEPO Decomposition #13

- Dataset: `musique`
- Question: What is the birthplace of the person after whom São José dos Campos was named?
- Gold answer: Nazareth

## 1. Explicit Entities
- São José dos Campos span=(48, 67)

## 2. Entity Masking
- ENTITYA -> São José dos Campos

Masked question: What is the birthplace of the person after whom ENTITYA was named?

## 3. Global Best Path
- São José dos Campos ---- named ---- person ---- birthplace

## 4. Step5 Action Trace
- q1: Who is the person after whom São José dos Campos was named?
  - consume: São José dos Campos -> named -> person
  - produce: q1_answer
- q2: What is the birthplace of q1's answer?
  - consume: q1_answer ---- person -> birthplace
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the person after whom São José dos Campos was named?
  - depends_on: (none)
- q2: What is the birthplace of q1's answer?
  - depends_on: q1
