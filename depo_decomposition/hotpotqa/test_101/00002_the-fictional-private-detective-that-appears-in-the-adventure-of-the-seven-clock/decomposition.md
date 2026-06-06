# DEPO Decomposition #2

- Dataset: `hotpotqa`
- Question: The fictional private detective that appears in "The Adventure of the Seven Clocks" what written by whom?
- Gold answer: Sir Arthur Conan Doyle

## 1. Semantic-Normalized Question
The fictional private detective that appears in "The Adventure of the Seven Clocks" was written by whom?

## 2. Mask Spans
- The Adventure of the Seven Clocks (entity, Film)

## 3. Selective Masked Question
The fictional private detective that appears in "MovieA" was written by whom?

## 4. CoreNLP Dependency Parse
- detective[4] --det--> The[1]
- detective[4] --amod--> fictional[2]
- detective[4] --amod--> private[3]
- appears[6] --nsubj--> detective[4]
- written[12] --nsubj:pass--> detective[4]
- detective[4] --ref--> that[5]
- detective[4] --acl:relcl--> appears[6]
- MovieA[9] --case--> in[7]
- MovieA[9] --punct--> "[8]
- appears[6] --obl:in--> MovieA[9]
- MovieA[9] --punct--> "[10]
- written[12] --aux:pass--> was[11]
- whom[14] --case--> by[13]
- written[12] --obl:agent--> whom[14]
- written[12] --punct--> ?[15]

## 5. Undirected Dependency Graph
- The[1] --det-- detective[4]
- fictional[2] --amod-- detective[4]
- private[3] --amod-- detective[4]
- detective[4] --nsubj/acl:relcl-- appears[6]
- detective[4] --nsubj:pass-- written[12]
- detective[4] --ref-- that[5]
- appears[6] --obl:in-- The Adventure of the Seven Clocks[9]
- in[7] --case-- The Adventure of the Seven Clocks[9]
- "[8] --punct-- The Adventure of the Seven Clocks[9]
- The Adventure of the Seven Clocks[9] --punct-- "[10]
- was[11] --aux:pass-- written[12]
- written[12] --obl:agent-- whom[14]
- written[12] --punct-- ?[15]
- by[13] --case-- whom[14]

## 6. Entity Start Nodes
- e1: The Adventure of the Seven Clocks graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): The Adventure of the Seven Clocks -- appears -- detective -- fictional
- e1_p2 (e1): The Adventure of the Seven Clocks -- appears -- detective -- private
- e1_p3 (e1): The Adventure of the Seven Clocks -- appears -- detective -- written
- e1_p4 (e1): The Adventure of the Seven Clocks -- appears -- detective -- written -- was
- e1_p5 (e1): The Adventure of the Seven Clocks -- appears -- detective -- written -- whom
- e1_p6 (e1): The Adventure of the Seven Clocks -- appears -- detective -- written -- ?
- e1_p7 (e1): The Adventure of the Seven Clocks -- appears -- detective -- written -- whom -- by
- e1_p8 (e1): The Adventure of the Seven Clocks -- appears -- detective
- e1_p9 (e1): The Adventure of the Seven Clocks -- appears -- detective -- The
- e1_p10 (e1): The Adventure of the Seven Clocks -- appears -- detective -- that
- e1_p11 (e1): The Adventure of the Seven Clocks -- appears
- e1_p12 (e1): The Adventure of the Seven Clocks -- in
- e1_p13 (e1): The Adventure of the Seven Clocks -- "
- e1_p14 (e1): The Adventure of the Seven Clocks -- "

## 8. LLM Selected Entity Paths
- e1: e1_p7 The Adventure of the Seven Clocks -- appears -- detective -- written -- whom -- by
  Reason: This path follows the useful reasoning chain from the entity through the detective to the author, which is the final answer slot.

## 9. Selected Path Semantic Transduction
Nodes:
- the_adventure_of_the_seven_clocks: The Adventure of the Seven Clocks (entity)
- detective: detective (type_variable)
- author: author (type_variable)
- written_by: written_by (value_slot)

Edges:
- the_adventure_of_the_seven_clocks -> detective (appears in)
- detective -> author (written by)
- author -> written_by (author of the work)

## 10. Atomic Subquestion DAG
- None: Who is the detective that appears in The Adventure of the Seven Clocks?
- None: Who is the author of the detective of The Adventure of the Seven Clocks?
- None: Who is the author of the detective of The Adventure of the Seven Clocks?
