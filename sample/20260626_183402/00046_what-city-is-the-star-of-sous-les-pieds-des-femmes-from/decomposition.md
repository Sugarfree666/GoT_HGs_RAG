# DEPO Decomposition #46

- Dataset: `musique`
- Question: What city is the star of Sous les pieds des femmes from?
- Gold answer: La Goulette

## 1. Explicit Entities
- Sous les pieds des femmes span=(25, 50)

## 2. Entity Masking
- ENTITYA -> Sous les pieds des femmes

Masked question: What city is the star of ENTITYA from?

## 3. Global Best Path
- Sous les pieds des femmes ---- star ---- city

## 4. Step5 Action Trace
- q1: Who is the star of the film Sous les pieds des femmes?
  - consume: Sous les pieds des femmes -> star
  - produce: q1_answer
- q2: What city is q1's answer from?
  - consume: q1_answer ---- star -> city
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the star of the film Sous les pieds des femmes?
  - depends_on: (none)
- q2: What city is q1's answer from?
  - depends_on: q1
