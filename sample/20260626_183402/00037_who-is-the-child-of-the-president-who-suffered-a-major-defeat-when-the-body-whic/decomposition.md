# DEPO Decomposition #37

- Dataset: `musique`
- Question: Who is the child of the president who suffered a major defeat when the body which approves members of the American cabinet refused to ratify the Versailles treaty?
- Gold answer: Jessie Woodrow Wilson

## 1. Explicit Entities
- Versailles treaty span=(145, 162)

## 2. Entity Masking
- ENTITYA -> Versailles treaty

Masked question: Who is the child of the president who suffered a major defeat when the body which approves members of the American cabinet refused to ratify the ENTITYA?

## 3. Global Best Path
- Versailles treaty ---- ratify ---- refused ---- approves ---- members ---- body

## 4. Step5 Action Trace
- q1: What is the body that approves members of the American cabinet?
  - consume: Versailles treaty -> ratify -> refused -> approves -> members -> body
  - produce: q1_answer
- q2: Who is the child of the president associated with q1's answer?
  - consume: q1_answer ---- president -> child
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: What is the body that approves members of the American cabinet?
  - depends_on: (none)
- q2: Who is the child of the president associated with q1's answer?
  - depends_on: q1
