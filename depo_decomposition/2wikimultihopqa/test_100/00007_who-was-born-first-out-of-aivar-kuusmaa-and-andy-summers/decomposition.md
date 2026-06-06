# DEPO Decomposition #7

- Dataset: `2wikimultihopqa`
- Question: Who was born first out of Aivar Kuusmaa and Andy Summers?
- Gold answer: Andy Summers

## 1. Semantic-Normalized Question
Who was born first out of Aivar Kuusmaa and Andy Summers?

## 2. Mask Spans
- Aivar Kuusmaa (entity, Person)
- Andy Summers (entity, Person)

## 3. Selective Masked Question
Who was born first out of PersonA and PersonB?

## 4. CoreNLP Dependency Parse
- born[3] --nsubj:pass--> Who[1]
- born[3] --aux:pass--> was[2]
- born[3] --advmod--> first[4]
- PersonA[7] --case--> out[5]
- out[5] --fixed--> of[6]
- born[3] --obl:out_of--> PersonA[7]
- PersonB[9] --cc--> and[8]
- born[3] --obl:out_of--> PersonB[9]
- PersonA[7] --conj:and--> PersonB[9]
- born[3] --punct--> ?[10]

## 5. Undirected Dependency Graph
- Who[1] --nsubj:pass-- born[3]
- was[2] --aux:pass-- born[3]
- born[3] --advmod-- first[4]
- born[3] --obl:out_of-- Aivar Kuusmaa[7]
- born[3] --obl:out_of-- Andy Summers[9]
- born[3] --punct-- ?[10]
- out[5] --case-- Aivar Kuusmaa[7]
- out[5] --fixed-- of[6]
- Aivar Kuusmaa[7] --conj:and-- Andy Summers[9]
- and[8] --cc-- Andy Summers[9]

## 6. Entity Start Nodes
- e1: Aivar Kuusmaa graph_node_ids=['7']
- e2: Andy Summers graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Aivar Kuusmaa -- born -- first
- e1_p2 (e1): Aivar Kuusmaa -- born -- Who
- e1_p3 (e1): Aivar Kuusmaa -- born
- e1_p4 (e1): Aivar Kuusmaa -- out
- e1_p5 (e1): Aivar Kuusmaa -- born -- was
- e1_p6 (e1): Aivar Kuusmaa -- born -- ?
- e1_p7 (e1): Aivar Kuusmaa -- out -- of
- e1_p8 (e1): Aivar Kuusmaa -- born -- Andy Summers
- e1_p9 (e1): Aivar Kuusmaa -- Andy Summers
- e1_p10 (e1): Aivar Kuusmaa -- Andy Summers -- born -- first
- e1_p11 (e1): Aivar Kuusmaa -- Andy Summers -- born -- Who
- e1_p12 (e1): Aivar Kuusmaa -- Andy Summers -- born
- e1_p13 (e1): Aivar Kuusmaa -- born -- Andy Summers -- and
- e1_p14 (e1): Aivar Kuusmaa -- Andy Summers -- born -- was
- e1_p15 (e1): Aivar Kuusmaa -- Andy Summers -- born -- ?
- e1_p16 (e1): Aivar Kuusmaa -- Andy Summers -- and
- e2_p1 (e2): Andy Summers -- born -- first
- e2_p2 (e2): Andy Summers -- born -- Who
- e2_p3 (e2): Andy Summers -- born
- e2_p4 (e2): Andy Summers -- born -- was
- e2_p5 (e2): Andy Summers -- born -- ?
- e2_p6 (e2): Andy Summers -- and
- e2_p7 (e2): Andy Summers -- born -- Aivar Kuusmaa
- e2_p8 (e2): Andy Summers -- Aivar Kuusmaa
- e2_p9 (e2): Andy Summers -- born -- Aivar Kuusmaa -- out
- e2_p10 (e2): Andy Summers -- Aivar Kuusmaa -- born -- first
- e2_p11 (e2): Andy Summers -- born -- Aivar Kuusmaa -- out -- of
- e2_p12 (e2): Andy Summers -- Aivar Kuusmaa -- born -- Who
- e2_p13 (e2): Andy Summers -- Aivar Kuusmaa -- born
- e2_p14 (e2): Andy Summers -- Aivar Kuusmaa -- out
- e2_p15 (e2): Andy Summers -- Aivar Kuusmaa -- born -- was
- e2_p16 (e2): Andy Summers -- Aivar Kuusmaa -- born -- ?
- e2_p17 (e2): Andy Summers -- Aivar Kuusmaa -- out -- of

## 8. LLM Selected Entity Paths
- e1: e1_p1 Aivar Kuusmaa -- born -- first
  Reason: This path directly connects Aivar Kuusmaa to the concept of being born first, which is relevant to the question.
- e2: e2_p1 Andy Summers -- born -- first
  Reason: This path directly connects Andy Summers to the concept of being born first, which is relevant to the question.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "who",
  "answer_kind": "comparison_or_boolean",
  "answer_slot_hint": "date",
  "focus_predicate": "born",
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- aivar_kuusmaa: Aivar Kuusmaa (entity)
- birth_date_aivar: birth_date (value_slot)

Edges:
- aivar_kuusmaa -> birth_date_aivar (date of birth of Aivar Kuusmaa)

## 10. Atomic Subquestion DAG
- None: What is the birth date of Aivar Kuusmaa?
