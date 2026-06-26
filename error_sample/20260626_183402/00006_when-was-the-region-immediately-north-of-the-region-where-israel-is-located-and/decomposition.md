# DEPO Decomposition #6

- Dataset: `musique`
- Question: When was the region immediately north of the region where Israel is located and the location of the Battle of Qurah and Umm al Maradim created?
- Gold answer: 1930

## 1. Explicit Entities
- Israel span=(58, 64)
- Battle of Qurah and Umm al Maradim span=(100, 134)

## 2. Entity Masking
- ENTITYA -> Israel
- ENTITYB -> Battle of Qurah and Umm al Maradim

Masked question: When was the region immediately north of the region where ENTITYA is located and the location of the ENTITYB created?

## 3. Global Best Path
- Battle of Qurah and Umm al Maradim ---- location ---- created ---- region ---- Israel ---- region ---- located ---- north ---- immediately

## 4. Step5 Action Trace
- q1: What is the location of the Battle of Qurah and Umm al Maradim?
  - consume: Battle of Qurah and Umm al Maradim -> location
  - produce: q1_answer
- q2: What is the region immediately north of the location of the Battle of Qurah and Umm al Maradim?
  - consume: q1_answer ---- Israel -> region ---- region -> north ---- immediately
  - produce: q2_answer
- q3: When was the region immediately north of the region where Israel is located created?
  - consume: q2_answer -> created
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: What is the location of the Battle of Qurah and Umm al Maradim?
  - depends_on: (none)
- q2: What is the region immediately north of the location of the Battle of Qurah and Umm al Maradim?
  - depends_on: q1
- q3: When was the region immediately north of the region where Israel is located created?
  - depends_on: q2
