# DEPO Decomposition #15

- Dataset: `2wikimultihopqa`
- Question: Where was the place of death of Maurice, Prince Of Orange's father?
- Gold answer: Delft

## 1. Semantic-Normalized Question
Where was the place of death of Maurice, Prince Of Orange's father?

## 2. Mask Spans
- Maurice, Prince Of Orange (entity, Person)

## 3. Selective Masked Question
Where was the place of death of PersonA's father?

## 4. CoreNLP Dependency Parse
- was[2] --advmod--> Where[1]
- place[4] --det--> the[3]
- was[2] --nsubj--> place[4]
- death[6] --case--> of[5]
- place[4] --nmod:of--> death[6]
- father[10] --case--> of[7]
- father[10] --nmod:poss--> PersonA[8]
- PersonA[8] --case--> 's[9]
- death[6] --nmod:of--> father[10]
- was[2] --punct--> ?[11]

## 5. Undirected Dependency Graph
- Where[1] --advmod-- was[2]
- was[2] --nsubj-- place[4]
- was[2] --punct-- ?[11]
- the[3] --det-- place[4]
- place[4] --nmod:of-- death[6]
- of[5] --case-- death[6]
- death[6] --nmod:of-- father[10]
- of[7] --case-- father[10]
- Maurice, Prince Of Orange[8] --nmod:poss-- father[10]
- Maurice, Prince Of Orange[8] --case-- 's[9]

## 6. Entity Start Nodes
- e1: Maurice, Prince Of Orange graph_node_ids=['8']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Maurice, Prince Of Orange -- father -- death -- place -- was -- Where
- e1_p2 (e1): Maurice, Prince Of Orange -- father -- death -- place
- e1_p3 (e1): Maurice, Prince Of Orange -- father -- death -- place -- was
- e1_p4 (e1): Maurice, Prince Of Orange -- father -- death -- place -- the
- e1_p5 (e1): Maurice, Prince Of Orange -- father -- death -- place -- was -- ?
- e1_p6 (e1): Maurice, Prince Of Orange -- father -- death
- e1_p7 (e1): Maurice, Prince Of Orange -- father -- death -- of
- e1_p8 (e1): Maurice, Prince Of Orange -- 's
- e1_p9 (e1): Maurice, Prince Of Orange -- father
- e1_p10 (e1): Maurice, Prince Of Orange -- father -- of

## 8. LLM Selected Entity Paths
- e1: e1_p1 Maurice, Prince Of Orange -- father -- death -- place -- was -- Where
  Reason: This path provides a complete reasoning chain from Maurice, Prince Of Orange to his father's place of death, including the necessary context of 'was' and 'where'.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "where",
  "answer_kind": "location",
  "answer_slot_hint": "death_place",
  "focus_predicate": "die"
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- maurice_prince_of_orange: Maurice, Prince Of Orange (entity)
- father: father (type_variable)
- death_place: death_place (value_slot)

Edges:
- maurice_prince_of_orange -> father (father of Maurice, Prince Of Orange)
- father -> death_place (place of death of the father)

## 10. Atomic Subquestion DAG
- None: Who is the father of Maurice, Prince Of Orange?
- None: Where did the father of Maurice, Prince Of Orange die?
