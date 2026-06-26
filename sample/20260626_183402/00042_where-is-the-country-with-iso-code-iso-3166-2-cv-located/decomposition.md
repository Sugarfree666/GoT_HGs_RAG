# DEPO Decomposition #42

- Dataset: `musique`
- Question: Where is the country with ISO code ISO 3166-2:CV located?
- Gold answer: central Atlantic Ocean

## 1. Explicit Entities
- ISO 3166-2:CV span=(35, 48)

## 2. Entity Masking
- ENTITYA -> ISO 3166-2:CV

Masked question: Where is the country with ISO code ENTITYA located?

## 3. Global Best Path
- ISO 3166-2:CV ---- code ---- ISO

## 4. Step5 Action Trace
- q1: What country has the ISO code ISO 3166-2:CV?
  - consume: ISO 3166-2:CV -> code
  - produce: q1_answer
- q2: Where is q1's answer located?
  - consume: q1_answer ---- country -> location
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What country has the ISO code ISO 3166-2:CV?
  - depends_on: (none)
- q2: Where is q1's answer located?
  - depends_on: q1
