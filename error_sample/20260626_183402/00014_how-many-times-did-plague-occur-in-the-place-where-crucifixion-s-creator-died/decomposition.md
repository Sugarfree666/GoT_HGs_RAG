# DEPO Decomposition #14

- Dataset: `musique`
- Question: How many times did plague occur in the place where Crucifixion's creator died?
- Gold answer: 22

## 1. Explicit Entities
- Crucifixion span=(51, 62)

## 2. Entity Masking
- ENTITYA -> Crucifixion

Masked question: How many times did plague occur in the place where ENTITYA's creator died?

## 3. Global Best Path
- Crucifixion ---- creator ---- died ---- place ---- occur ---- times ---- many ---- How

## 4. Step5 Action Trace
- q1: Who is the creator of Crucifixion?
  - consume: Crucifixion -> creator
  - produce: q1_answer
- q2: In which place did q1's answer die?
  - consume: q1_answer -> died -> place
  - produce: q2_answer
- q3: How many times did plague occur in the place q2's answer?
  - consume: q2_answer -> occur -> times
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: Who is the creator of Crucifixion?
  - depends_on: (none)
- q2: In which place did q1's answer die?
  - depends_on: q1
- q3: How many times did plague occur in the place q2's answer?
  - depends_on: q2
