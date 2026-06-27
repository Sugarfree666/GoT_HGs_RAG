# DEPO Decomposition #65

- Dataset: `musique`
- Question: Who was a prominent figure at the radio division of the network that created the version of The Biggest Loser set in the country where Seria is?
- Gold answer: Walter Sabo

## 1. Explicit Entities
- The Biggest Loser span=(92, 109)
- Seria span=(135, 140)

## 2. Entity Masking
- ENTITYA -> The Biggest Loser
- ENTITYB -> Seria

Masked question: Who was a prominent figure at the radio division of the network that created the version of ENTITYA set in the country where ENTITYB is?

## 3. Global Best Path
- Seria ---- country ---- set ---- created ---- figure ---- prominent ---- division ---- version ---- The Biggest Loser

## 4. Step5 Action Trace
- q1: What country is Seria in?
  - consume: Seria ---- country
  - produce: q1_answer
- q2: What version of The Biggest Loser was created in q1's answer?
  - consume: q1_answer ---- set ---- created ---- version ---- The Biggest Loser
  - produce: q2_answer
- q3: Who was a prominent figure at the radio division of q2's answer?
  - consume: q2_answer ---- division ---- figure ---- prominent
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: What country is Seria in?
  - depends_on: (none)
- q2: What version of The Biggest Loser was created in q1's answer?
  - depends_on: q1
- q3: Who was a prominent figure at the radio division of q2's answer?
  - depends_on: q2
