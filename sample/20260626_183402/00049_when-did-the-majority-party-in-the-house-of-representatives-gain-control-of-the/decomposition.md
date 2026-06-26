# DEPO Decomposition #49

- Dataset: `musique`
- Question: When did the majority party in the House of Representatives gain control of the body which approves members of the Cabinet?
- Gold answer: January 2015

## 1. Explicit Entities
- House of Representatives span=(35, 59)
- Cabinet span=(115, 122)

## 2. Entity Masking
- ENTITYA -> House of Representatives
- ENTITYB -> Cabinet

Masked question: When did the majority party in the ENTITYA gain control of the body that approves members of the ENTITYB?

## 3. Global Best Path
- Cabinet ---- members ---- approves ---- body ---- control ---- gain ---- majority ---- party ---- House of Representatives

## 4. Step5 Action Trace
- q1: Who is the majority party in the House of Representatives?
  - consume: House of Representatives -> majority party
  - produce: q1_answer
- q2: What is the body that approves members of the Cabinet?
  - consume: q1_answer ---- Cabinet -> body -> approves
  - produce: q2_answer
- q3: When did the majority party in the House of Representatives gain control of the body that approves members of the Cabinet?
  - consume: q1_answer ---- q2_answer ---- control -> gain
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: Who is the majority party in the House of Representatives?
  - depends_on: (none)
- q2: What is the body that approves members of the Cabinet?
  - depends_on: q1
- q3: When did the majority party in the House of Representatives gain control of the body that approves members of the Cabinet?
  - depends_on: q1, q2
