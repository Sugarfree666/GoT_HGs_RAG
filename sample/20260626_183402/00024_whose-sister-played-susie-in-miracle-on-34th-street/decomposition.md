# DEPO Decomposition #24

- Dataset: `musique`
- Question: Whose sister played Susie in miracle on 34th street?
- Gold answer: Lana Wood

## 1. Explicit Entities
- Susie span=(20, 25)
- miracle on 34th street span=(29, 51)

## 2. Entity Masking
- ENTITYA -> Susie
- ENTITYB -> miracle on 34th street

Masked question: Whose sister played ENTITYA in ENTITYB?

## 3. Global Best Path
- Susie ---- played ---- sister

## 4. Step5 Action Trace
- q1: Who played the character Susie in miracle on 34th street?
  - consume: Susie -> played
  - produce: q1_answer
- q2: Whose sister is q1's answer?
  - consume: q1_answer ---- played -> sister
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who played the character Susie in miracle on 34th street?
  - depends_on: (none)
- q2: Whose sister is q1's answer?
  - depends_on: q1
