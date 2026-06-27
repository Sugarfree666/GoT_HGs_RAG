# DEPO Decomposition #93

- Dataset: `musique`
- Question: When did the majority party in the House of Representatives gain control of the body which determines rules of the US House and US Senate?
- Gold answer: January 2015

## 1. Explicit Entities
- House of Representatives span=(35, 59)
- US House span=(115, 123)
- US Senate span=(128, 137)

## 2. Entity Masking
- ENTITYA -> House of Representatives
- ENTITYB -> US House
- ENTITYC -> US Senate

Masked question: When did the majority party in the ENTITYA gain control of the body that determines the rules of the ENTITYB and ENTITYC?

## 3. Global Best Path
- P1: US House ---- rules ---- determines ---- body ---- control ---- gain ---- majority ---- party ---- House of Representatives
- P2: US Senate ---- rules ---- determines ---- body ---- control ---- gain ---- majority ---- party ---- House of Representatives

## 4. Step5 Action Trace
- q1: Who is the majority party in the House of Representatives?
  - consume: House of Representatives ---- majority ---- party
  - produce: q1_answer
- q2: When did q1's answer gain control of the body which determines rules of the US House?
  - consume: US House ---- rules ---- determines ---- body ---- control ---- gain ---- q1_answer
  - produce: q2_answer
- q3: When did q1's answer gain control of the body which determines rules of the US Senate?
  - consume: US Senate ---- rules ---- determines ---- body ---- control ---- gain ---- q1_answer
  - produce: q3_answer
- q4: When did the majority party in the House of Representatives gain control of the body which determines rules of the US House and US Senate, based on q2's answer and q3's answer?
  - consume: q2_answer ---- q3_answer
  - produce: q4_answer

## 5. Atomic Question DAG
- q1: Who is the majority party in the House of Representatives?
  - depends_on: (none)
- q2: When did q1's answer gain control of the body which determines rules of the US House?
  - depends_on: q1
- q3: When did q1's answer gain control of the body which determines rules of the US Senate?
  - depends_on: q1
- q4: When did the majority party in the House of Representatives gain control of the body which determines rules of the US House and US Senate, based on q2's answer and q3's answer?
  - depends_on: q2, q3
