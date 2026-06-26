# DEPO Decomposition #44

- Dataset: `musique`
- Question: When did the political party that favored harsh punishment of southern states after the Civil War, gain control of the House?
- Gold answer: the 2010 election

## 1. Explicit Entities
- Civil War span=(88, 97)
- House span=(119, 124)

## 2. Entity Masking
- ENTITYA -> Civil War
- ENTITYB -> House

Masked question: When did the political party that favored harsh punishment of southern states after the ENTITYA gain control of the ENTITYB?

## 3. Global Best Path
- House ---- control ---- gain ---- political ---- party ---- favored ---- punishment ---- states ---- southern

## 4. Step5 Action Trace
- q1: Which political party favored harsh punishment of southern states after the Civil War?
  - consume: House -> control
  - produce: q1_answer
- q2: When did q1's answer gain control of the House?
  - consume: q1_answer ---- control -> gain
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Which political party favored harsh punishment of southern states after the Civil War?
  - depends_on: (none)
- q2: When did q1's answer gain control of the House?
  - depends_on: q1
