# DEPO Decomposition #48

- Dataset: `musique`
- Question: Who stars in the video "One Last Time" by the performer of Baby I?
- Gold answer: Matt Bennett

## 1. Explicit Entities
- One Last Time span=(24, 37)
- Baby I span=(59, 65)

## 2. Entity Masking
- ENTITYA -> One Last Time
- ENTITYB -> Baby I

Masked question: Who stars in the video 'ENTITYA' by the performer of 'ENTITYB'?

## 3. Global Best Path
- Baby I ---- performer ---- One Last Time ---- video ---- stars ---- Who

## 4. Step5 Action Trace
- q1: Who is the performer of the song Baby I?
  - consume: Baby I -> performer
  - produce: q1_answer
- q2: Who stars in the video One Last Time by q1's answer?
  - consume: q1_answer ---- One Last Time -> video -> stars
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the performer of the song Baby I?
  - depends_on: (none)
- q2: Who stars in the video One Last Time by q1's answer?
  - depends_on: q1
