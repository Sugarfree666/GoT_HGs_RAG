# DEPO Decomposition #1

- Dataset: `musique`
- Question: When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?
- Gold answer: June 1982

## 1. Explicit Entities
- Messi span=(24, 29)
- Copa del Rey span=(41, 53)
- Barcelona span=(80, 89)

## 2. Entity Masking
- ENTITYA -> Messi
- ENTITYB -> Copa del Rey
- ENTITYC -> Barcelona

Masked question: When was the person who ENTITYA's goals in ENTITYB compared to get signed by ENTITYC?

## 3. Global Best Path
- Barcelona ---- signed ---- get ---- person ---- compared ---- goals ---- Messi

## 4. Step5 Action Trace
- q1: Who is the person compared to Messi's goals in Copa del Rey?
  - consume: Messi -> goals
  - produce: q1_answer
- q2: When was q1's answer signed by Barcelona?
  - consume: q1_answer ---- person -> signed -> get -> Barcelona
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who is the person compared to Messi's goals in Copa del Rey?
  - depends_on: (none)
- q2: When was q1's answer signed by Barcelona?
  - depends_on: q1
