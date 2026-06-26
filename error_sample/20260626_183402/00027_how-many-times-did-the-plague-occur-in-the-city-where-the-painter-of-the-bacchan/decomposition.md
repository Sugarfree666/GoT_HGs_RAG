# DEPO Decomposition #27

- Dataset: `musique`
- Question: How many times did the plague occur in the city where the painter of The Bacchanal of the Andrians died?
- Gold answer: 22

## 1. Explicit Entities
- The Bacchanal of the Andrians span=(69, 98)

## 2. Entity Masking
- ENTITYA -> The Bacchanal of the Andrians

Masked question: How many times did the plague occur in the city where the painter of ENTITYA died?

## 3. Global Best Path
- The Bacchanal of the Andrians ---- painter ---- died ---- city ---- occur ---- times ---- How ---- many

## 4. Step5 Action Trace
- q1: Who is the painter of The Bacchanal of the Andrians?
  - consume: The Bacchanal of the Andrians -> painter
  - produce: q1_answer
- q2: In which city did q1's answer die?
  - consume: q1_answer ---- painter -> died -> city
  - produce: q2_answer
- q3: How many times did the plague occur in the city q2's answer?
  - consume: q2_answer ---- city -> occur -> times
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: Who is the painter of The Bacchanal of the Andrians?
  - depends_on: (none)
- q2: In which city did q1's answer die?
  - depends_on: q1
- q3: How many times did the plague occur in the city q2's answer?
  - depends_on: q2
