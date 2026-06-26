# DEPO Decomposition #12

- Dataset: `musique`
- Question: How were the people from whom new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the country between Thailand and A Lim's country?
- Gold answer: The dynasty regrouped and defeated the Portuguese

## 1. Explicit Entities
- Somali Muslim Ajuran Empire span=(83, 110)
- Thailand span=(145, 153)
- A Lim span=(158, 163)

## 2. Entity Masking
- ENTITYA -> Somali Muslim Ajuran Empire
- ENTITYB -> Thailand
- ENTITYC -> A Lim

Masked question: How were the people from whom new coins were a proclamation of independence by the ENTITYA expelled from the country between ENTITYB and ENTITYC's country?

## 3. Global Best Path
- Somali Muslim Ajuran Empire ---- proclamation ---- coins ---- people ---- expelled ---- country

## 4. Step5 Action Trace
- q1: Who are the people from the Somali Muslim Ajuran Empire?
  - consume: Somali Muslim Ajuran Empire -> people
  - produce: q1_answer
- q2: How were q1's answer expelled from the country?
  - consume: q1_answer ---- people -> expelled
  - produce: q2_answer

## 5. Atomic Question DAG
- q1: Who are the people from the Somali Muslim Ajuran Empire?
  - depends_on: (none)
- q2: How were q1's answer expelled from the country?
  - depends_on: q1
