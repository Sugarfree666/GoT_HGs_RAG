# DEPO Decomposition #17

- Dataset: `musique`
- Question: When was the Palau de la Generalitat constructed in the city where Martin from the region where Perdiguera is located died?
- Gold answer: built in the 15th century

## 1. Explicit Entities
- Palau de la Generalitat span=(13, 36)
- Martin span=(67, 73)
- Perdiguera span=(96, 106)

## 2. Entity Masking
- ENTITYA -> Palau de la Generalitat
- ENTITYB -> Martin
- ENTITYC -> Perdiguera

Masked question: When was the ENTITYA constructed in the city where ENTITYB from the region where ENTITYC is located died?

## 3. Global Best Path
- Perdiguera ---- located ---- region ---- Martin ---- died ---- city ---- constructed ---- Palau de la Generalitat

## 4. Step5 Action Trace
- q1: In which city did Martin, from the region where Perdiguera is located, die?
  - consume: Perdiguera -> region -> Martin -> died -> city
  - produce: q1_answer
- q2: When was the Palau de la Generalitat constructed in q1's answer?
  - consume: q1_answer ---- constructed -> Palau de la Generalitat
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: In which city did Martin, from the region where Perdiguera is located, die?
  - depends_on: (none)
- q2: When was the Palau de la Generalitat constructed in q1's answer?
  - depends_on: q1
