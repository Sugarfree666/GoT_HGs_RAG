# DEPO Decomposition #60

- Dataset: `musique`
- Question: Where is the district that the person who wanted to reform and address John Kodwo Amissah's religion preached a sermon on Marian devotion before his death located?
- Gold answer: Saxony-Anhalt

## 1. Explicit Entities
- John Kodwo Amissah span=(71, 89)
- Marian span=(122, 128)

## 2. Entity Masking
- ENTITYA -> John Kodwo Amissah
- ENTITYB -> Marian

Masked question: Where is the district that the person who wanted to reform and address ENTITYA's religion preached a sermon on ENTITYB devotion before his death located?

## 3. Global Best Path
- Marian ---- devotion ---- sermon ---- located ---- district ---- preached ---- person ---- address ---- religion ---- reform

## 4. Step5 Action Trace
- q1: Who is the person that wanted to reform and address John Kodwo Amissah's religion?
  - consume: person ---- address ---- religion ---- reform
  - produce: q1_answer
- q2: What sermon on Marian devotion did q1's answer preach?
  - consume: Marian ---- devotion ---- sermon ---- preached ---- q1_answer
  - produce: q2_answer
- q3: Where is the district that q2's answer is located?
  - consume: district ---- located ---- q2_answer
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: Who is the person that wanted to reform and address John Kodwo Amissah's religion?
  - depends_on: (none)
- q2: What sermon on Marian devotion did q1's answer preach?
  - depends_on: q1
- q3: Where is the district that q2's answer is located?
  - depends_on: q2
