# DEPO Decomposition #51

- Dataset: `musique`
- Question: When did the country the top-ranking Warsaw Pact operatives came from, despite it being headquartered in the country where A Generation is set, agree to a unified Germany inside NATO?
- Gold answer: May 1990

## 1. Explicit Entities
- Warsaw Pact span=(37, 48)
- A Generation span=(123, 135)
- Germany span=(163, 170)
- NATO span=(178, 182)

## 2. Entity Masking
- ENTITYA -> Warsaw Pact
- ENTITYB -> A Generation
- ENTITYC -> Germany
- ENTITYD -> NATO

Masked question: When did the country that the top-ranking ENTITYA operatives came from, despite it being headquartered in the country where ENTITYB is set, agree to a unified ENTITYC inside ENTITYD?

## 3. Global Best Path
- NATO ---- inside ---- Germany ---- unified ---- agree ---- came ---- top-ranking ---- operatives ---- Warsaw Pact

## 4. Step5 Action Trace
- q1: Which country did the top-ranking operatives of the Warsaw Pact come from?
  - consume: top-ranking ---- operatives ---- Warsaw Pact
  - produce: q1_answer
- q2: In which country is A Generation set?
  - consume: A Generation ---- set ---- country
  - produce: q2_answer
- q3: When did q1's answer agree to a unified Germany inside NATO?
  - consume: Germany ---- unified ---- agree ---- inside ---- NATO ---- q1_answer
  - produce: q3_answer

## 5. Atomic Question DAG
- q1: Which country did the top-ranking operatives of the Warsaw Pact come from?
  - depends_on: (none)
- q2: In which country is A Generation set?
  - depends_on: (none)
- q3: When did q1's answer agree to a unified Germany inside NATO?
  - depends_on: q1
