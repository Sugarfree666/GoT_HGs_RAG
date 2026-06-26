# DEPO Decomposition #45

- Dataset: `musique`
- Question: When did the person chosen to be president of the confederacy end his fight in the Mexican-American war?
- Gold answer: 1848

## 1. Explicit Entities
- Mexican-American span=(83, 99)

## 2. Entity Masking
- ENTITYA -> Mexican-American

Masked question: When did the person chosen to be president of the confederacy end his fight in the ENTITYA war?

## 3. Global Best Path
- Mexican-American ---- war ---- fight ---- end ---- chosen ---- person ---- president ---- confederacy

## 4. Step5 Action Trace
- q1: When did the person chosen to be president of the confederacy end his fight in the Mexican-American war?
  - consume: Mexican-American -> war -> fight -> end
  - produce: q1_answer
- q2: Who is the person chosen to be president of the confederacy?
  - consume: q1_answer ---- chosen -> person -> president -> confederacy
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: When did the person chosen to be president of the confederacy end his fight in the Mexican-American war?
  - depends_on: (none)
- q2: Who is the person chosen to be president of the confederacy?
  - depends_on: q1
