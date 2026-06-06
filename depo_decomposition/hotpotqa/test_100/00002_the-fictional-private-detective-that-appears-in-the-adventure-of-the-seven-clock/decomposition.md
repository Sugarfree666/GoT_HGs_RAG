# DEPO Decomposition #2

- Dataset: `hotpotqa`
- Question: The fictional private detective that appears in "The Adventure of the Seven Clocks" what written by whom?
- Gold answer: Sir Arthur Conan Doyle

## 1. Semantic-Normalized Question
The fictional private detective that appears in 'The Adventure of the Seven Clocks' was written by whom?

## 2. Mask Spans
- The Adventure of the Seven Clocks' (entity, Entity)

## 3. Selective Masked Question
The fictional private detective that appears in 'SomeEntityA was written by whom?

## 4. CoreNLP Dependency Parse
- detective[4] --det--> The[1]
- detective[4] --amod--> fictional[2]
- detective[4] --amod--> private[3]
- appears[6] --nsubj--> detective[4]
- written[11] --nsubj:pass--> detective[4]
- detective[4] --ref--> that[5]
- detective[4] --acl:relcl--> appears[6]
- SomeEntityA[9] --case--> in[7]
- SomeEntityA[9] --punct--> '[8]
- appears[6] --obl:in--> SomeEntityA[9]
- written[11] --aux:pass--> was[10]
- whom[13] --case--> by[12]
- written[11] --obl:agent--> whom[13]
- written[11] --punct--> ?[14]

## 5. Undirected Dependency Graph
- The[1] --det-- detective[4]
- fictional[2] --amod-- detective[4]
- private[3] --amod-- detective[4]
- detective[4] --nsubj/acl:relcl-- appears[6]
- detective[4] --nsubj:pass-- written[11]
- detective[4] --ref-- that[5]
- appears[6] --obl:in-- The Adventure of the Seven Clocks'[9]
- in[7] --case-- The Adventure of the Seven Clocks'[9]
- '[8] --punct-- The Adventure of the Seven Clocks'[9]
- was[10] --aux:pass-- written[11]
- written[11] --obl:agent-- whom[13]
- written[11] --punct-- ?[14]
- by[12] --case-- whom[13]

## 6. Entity Start Nodes
- e1: The Adventure of the Seven Clocks' graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): The Adventure of the Seven Clocks' -- appears -- detective -- fictional
- e1_p2 (e1): The Adventure of the Seven Clocks' -- appears -- detective -- private
- e1_p3 (e1): The Adventure of the Seven Clocks' -- appears -- detective -- written
- e1_p4 (e1): The Adventure of the Seven Clocks' -- appears -- detective -- written -- was
- e1_p5 (e1): The Adventure of the Seven Clocks' -- appears -- detective -- written -- whom
- e1_p6 (e1): The Adventure of the Seven Clocks' -- appears -- detective -- written -- ?
- e1_p7 (e1): The Adventure of the Seven Clocks' -- appears -- detective -- written -- whom -- by
- e1_p8 (e1): The Adventure of the Seven Clocks' -- appears -- detective
- e1_p9 (e1): The Adventure of the Seven Clocks' -- appears -- detective -- The
- e1_p10 (e1): The Adventure of the Seven Clocks' -- appears -- detective -- that
- e1_p11 (e1): The Adventure of the Seven Clocks' -- appears
- e1_p12 (e1): The Adventure of the Seven Clocks' -- in
- e1_p13 (e1): The Adventure of the Seven Clocks' -- '

## 8. LLM Selected Entity Paths
- e1: e1_p7 The Adventure of the Seven Clocks' -- appears -- detective -- written -- whom -- by
  Reason: This path follows the useful reasoning chain from the entity through the detective to the author, which is the final answer slot.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "who",
  "answer_kind": "person_or_entity",
  "answer_slot_hint": "written",
  "focus_predicate": null,
  "focus_noun": "written",
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- the_adventure_of_the_seven_clocks: The Adventure of the Seven Clocks (entity)
- detective: detective (type_variable)
- author: author (type_variable)

Edges:
- the_adventure_of_the_seven_clocks -> detective (appears in)
- detective -> author (written by)

## 10. Atomic Subquestion DAG
- None: Who is the detective that appears in The Adventure of the Seven Clocks?
- None: Who wrote the detective of The Adventure of the Seven Clocks?
