# DEPO Decomposition #4

- Dataset: `hotpotqa`
- Question: Sparking the Marian civil war, who helped the recently abdicated queen to escape her imprisonment?
- Gold answer: the Queen's gaoler

## 1. Semantic-Normalized Question
Sparking the Marian civil war, who helped the queen, who had recently abdicated, to escape her imprisonment?

## 2. Mask Spans
- Sparking the Marian (entity, SparkingTheMarian)

## 3. Selective Masked Question
SomeEntityA civil war, who helped the queen, who had recently abdicated, to escape her imprisonment?

## 4. CoreNLP Dependency Parse
- war[3] --compound--> SomeEntityA[1]
- war[3] --amod--> civil[2]
- war[3] --punct--> ,[4]
- helped[6] --nsubj--> who[5]
- war[3] --dep--> helped[6]
- queen[8] --det--> the[7]
- helped[6] --obj--> queen[8]
- abdicated[13] --nsubj--> queen[8]
- escape[16] --nsubj:xsubj--> queen[8]
- queen[8] --punct--> ,[9]
- queen[8] --ref--> who[10]
- abdicated[13] --aux--> had[11]
- abdicated[13] --advmod--> recently[12]
- queen[8] --acl:relcl--> abdicated[13]
- abdicated[13] --punct--> ,[14]
- escape[16] --mark--> to[15]
- abdicated[13] --xcomp--> escape[16]
- imprisonment[18] --nmod:poss--> her[17]
- escape[16] --obj--> imprisonment[18]
- war[3] --punct--> ?[19]

## 5. Undirected Dependency Graph
- Sparking the Marian[1] --compound-- war[3]
- civil[2] --amod-- war[3]
- war[3] --punct-- ,[4]
- war[3] --dep-- helped[6]
- war[3] --punct-- ?[19]
- who[5] --nsubj-- helped[6]
- helped[6] --obj-- queen[8]
- the[7] --det-- queen[8]
- queen[8] --nsubj/acl:relcl-- abdicated[13]
- queen[8] --nsubj:xsubj-- escape[16]
- queen[8] --punct-- ,[9]
- queen[8] --ref-- who[10]
- had[11] --aux-- abdicated[13]
- recently[12] --advmod-- abdicated[13]
- abdicated[13] --punct-- ,[14]
- abdicated[13] --xcomp-- escape[16]
- to[15] --mark-- escape[16]
- escape[16] --obj-- imprisonment[18]
- her[17] --nmod:poss-- imprisonment[18]

## 6. Entity Start Nodes
- e1: Sparking the Marian graph_node_ids=['1']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Sparking the Marian -- war -- helped -- queen -- abdicated -- escape -- imprisonment
- e1_p2 (e1): Sparking the Marian -- war -- helped -- queen -- escape -- abdicated -- had
- e1_p3 (e1): Sparking the Marian -- war -- helped -- queen -- escape -- abdicated -- recently
- e1_p4 (e1): Sparking the Marian -- war -- helped -- queen -- escape -- imprisonment -- her
- e1_p5 (e1): Sparking the Marian -- war -- helped -- queen -- abdicated -- had
- e1_p6 (e1): Sparking the Marian -- war -- helped -- queen -- abdicated -- recently
- e1_p7 (e1): Sparking the Marian -- war -- helped -- queen -- abdicated -- escape
- e1_p8 (e1): Sparking the Marian -- war -- helped -- queen -- escape -- abdicated
- e1_p9 (e1): Sparking the Marian -- war -- helped -- queen -- escape -- imprisonment
- e1_p10 (e1): Sparking the Marian -- war -- helped -- queen -- abdicated -- escape -- to
- e1_p11 (e1): Sparking the Marian -- war -- helped -- queen -- escape -- abdicated -- ,
- e1_p12 (e1): Sparking the Marian -- war -- helped -- queen -- abdicated
- e1_p13 (e1): Sparking the Marian -- war -- helped -- queen -- escape
- e1_p14 (e1): Sparking the Marian -- war -- helped -- queen -- abdicated -- ,
- e1_p15 (e1): Sparking the Marian -- war -- helped -- queen -- escape -- to
- e1_p16 (e1): Sparking the Marian -- war -- helped -- queen -- who
- e1_p17 (e1): Sparking the Marian -- war -- helped -- queen
- e1_p18 (e1): Sparking the Marian -- war -- helped -- queen -- the
- e1_p19 (e1): Sparking the Marian -- war -- helped -- queen -- ,
- e1_p20 (e1): Sparking the Marian -- war -- helped -- who
- e1_p21 (e1): Sparking the Marian -- war -- civil
- e1_p22 (e1): Sparking the Marian -- war -- helped
- e1_p23 (e1): Sparking the Marian -- war
- e1_p24 (e1): Sparking the Marian -- war -- ,
- e1_p25 (e1): Sparking the Marian -- war -- ?

## 8. LLM Selected Entity Paths
- e1: e1_p7 Sparking the Marian -- war -- helped -- queen -- abdicated -- escape
  Reason: This path follows the useful reasoning chain from the entity through the war, helped, queen, and escape, leading to the context of the question.

## 9. Selected Path Semantic Transduction
Nodes:
- sparking_the_marian: Sparking the Marian (entity)
- war: war (type_variable)
- helper: helper (type_variable)
- queen: queen (entity)
- abdicated: abdicated (type_variable)
- escape: escape (type_variable)
- imprisonment: imprisonment (type_variable)

Edges:
- sparking_the_marian -> war (event of)
- war -> helper (helped in)
- helper -> queen (helped)
- queen -> abdicated (who had)
- abdicated -> escape (to)
- escape -> imprisonment (from)

## 10. Atomic Subquestion DAG
- None: What event is associated with Sparking the Marian?
- None: Who helped the queen during the war of Sparking the Marian?
- None: Who is the queen that the helper of the war of Sparking the Marian helped to escape her imprisonment?
- None: Who had abdicated the queen of the helper of the war of Sparking the Marian?
- None: How did the queen of the helper of the war of Sparking the Marian escape?
- None: What is the imprisonment related to the escape of the abdicated queen of the helper of the war of Sparking the Marian?
