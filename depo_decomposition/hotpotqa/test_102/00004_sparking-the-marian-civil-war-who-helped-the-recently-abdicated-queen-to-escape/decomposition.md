# DEPO Decomposition #4

- Dataset: `hotpotqa`
- Question: Sparking the Marian civil war, who helped the recently abdicated queen to escape her imprisonment?
- Gold answer: the Queen's gaoler

## 1. Semantic-Normalized Question
Sparking the Marian civil war, who helped the queen who recently abdicated to escape her imprisonment?

## 2. Mask Spans
- Sparking the Marian (entity, SparkingTheMarian)

## 3. Selective Masked Question
SomeEntityA civil war, who helped the queen who recently abdicated to escape her imprisonment?

## 4. CoreNLP Dependency Parse
- war[3] --compound--> SomeEntityA[1]
- war[3] --amod--> civil[2]
- war[3] --punct--> ,[4]
- helped[6] --nsubj--> who[5]
- war[3] --dep--> helped[6]
- queen[8] --det--> the[7]
- helped[6] --obj--> queen[8]
- abdicated[11] --nsubj--> queen[8]
- escape[13] --nsubj:xsubj--> queen[8]
- queen[8] --ref--> who[9]
- abdicated[11] --advmod--> recently[10]
- queen[8] --acl:relcl--> abdicated[11]
- escape[13] --mark--> to[12]
- abdicated[11] --xcomp--> escape[13]
- imprisonment[15] --nmod:poss--> her[14]
- escape[13] --obj--> imprisonment[15]
- war[3] --punct--> ?[16]

## 5. Undirected Dependency Graph
- Sparking the Marian[1] --compound-- war[3]
- civil[2] --amod-- war[3]
- war[3] --punct-- ,[4]
- war[3] --dep-- helped[6]
- war[3] --punct-- ?[16]
- who[5] --nsubj-- helped[6]
- helped[6] --obj-- queen[8]
- the[7] --det-- queen[8]
- queen[8] --nsubj/acl:relcl-- abdicated[11]
- queen[8] --nsubj:xsubj-- escape[13]
- queen[8] --ref-- who[9]
- recently[10] --advmod-- abdicated[11]
- abdicated[11] --xcomp-- escape[13]
- to[12] --mark-- escape[13]
- escape[13] --obj-- imprisonment[15]
- her[14] --nmod:poss-- imprisonment[15]

## 6. Entity Start Nodes
- e1: Sparking the Marian graph_node_ids=['1']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Sparking the Marian -- war -- helped -- queen -- abdicated -- escape -- imprisonment
- e1_p2 (e1): Sparking the Marian -- war -- helped -- queen -- escape -- abdicated -- recently
- e1_p3 (e1): Sparking the Marian -- war -- helped -- queen -- escape -- imprisonment -- her
- e1_p4 (e1): Sparking the Marian -- war -- helped -- queen -- abdicated -- recently
- e1_p5 (e1): Sparking the Marian -- war -- helped -- queen -- abdicated -- escape
- e1_p6 (e1): Sparking the Marian -- war -- helped -- queen -- escape -- abdicated
- e1_p7 (e1): Sparking the Marian -- war -- helped -- queen -- escape -- imprisonment
- e1_p8 (e1): Sparking the Marian -- war -- helped -- queen -- abdicated -- escape -- to
- e1_p9 (e1): Sparking the Marian -- war -- helped -- queen -- abdicated
- e1_p10 (e1): Sparking the Marian -- war -- helped -- queen -- escape
- e1_p11 (e1): Sparking the Marian -- war -- helped -- queen -- escape -- to
- e1_p12 (e1): Sparking the Marian -- war -- helped -- queen -- who
- e1_p13 (e1): Sparking the Marian -- war -- helped -- queen
- e1_p14 (e1): Sparking the Marian -- war -- helped -- queen -- the
- e1_p15 (e1): Sparking the Marian -- war -- helped -- who
- e1_p16 (e1): Sparking the Marian -- war -- civil
- e1_p17 (e1): Sparking the Marian -- war -- helped
- e1_p18 (e1): Sparking the Marian -- war
- e1_p19 (e1): Sparking the Marian -- war -- ,
- e1_p20 (e1): Sparking the Marian -- war -- ?

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=escape
  Reason: The path effectively connects the entity to the queen and the escape action, covering the necessary predicates and answer intent.
- e1: e1_p2 score=85.0 valid=True terminal=escape
  Reason: The path includes the necessary elements to connect the entity to the queen and her escape, but lacks some directness in the chain.
- e1: e1_p3 score=90.0 valid=True terminal=escape
  Reason: The path effectively connects the entity to the queen and the escape action, covering the necessary predicates and answer intent.
- e1: e1_p4 score=80.0 valid=True terminal=abdicated
  Reason: The path connects the entity to the queen and her abdication but lacks a direct connection to the escape action.
- e1: e1_p5 score=85.0 valid=True terminal=escape
  Reason: The path effectively connects the entity to the queen and the escape action, covering the necessary predicates and answer intent.
- e1: e1_p6 score=80.0 valid=True terminal=escape
  Reason: The path connects the entity to the queen and the escape action but lacks some directness in the chain.
- e1: e1_p7 score=80.0 valid=True terminal=escape
  Reason: The path connects the entity to the queen and the escape action but lacks some directness in the chain.
- e1: e1_p8 score=75.0 valid=True terminal=escape
  Reason: The path connects the entity to the queen and the escape action but includes a preposition that detracts from its effectiveness.
- e1: e1_p9 score=70.0 valid=True terminal=abdicated
  Reason: The path connects the entity to the queen and her abdication but lacks a connection to the escape action.
- e1: e1_p10 score=70.0 valid=True terminal=escape
  Reason: The path connects the entity to the queen and the escape action but lacks a connection to the abdication.
- e1: e1_p11 score=75.0 valid=True terminal=escape
  Reason: The path connects the entity to the queen and the escape action but includes a preposition that detracts from its effectiveness.
- e1: e1_p12 score=60.0 valid=True terminal=who
  Reason: The path connects the entity to the queen but lacks coverage of the necessary actions and predicates.
- e1: e1_p13 score=50.0 valid=True terminal=queen
  Reason: The path connects the entity to the queen but lacks coverage of the necessary actions and predicates.
- e1: e1_p14 score=40.0 valid=True terminal=queen
  Reason: The path connects the entity to the queen but lacks coverage of the necessary actions and predicates.
- e1: e1_p15 score=50.0 valid=True terminal=who
  Reason: The path connects the entity to the queen but lacks coverage of the necessary actions and predicates.
- e1: e1_p16 score=30.0 valid=True terminal=war
  Reason: The path connects the entity to the war but lacks coverage of the necessary actions and predicates.
- e1: e1_p17 score=30.0 valid=True terminal=helped
  Reason: The path connects the entity to the helped action but lacks coverage of the necessary actions and predicates.
- e1: e1_p18 score=20.0 valid=True terminal=war
  Reason: The path connects the entity to the war but lacks coverage of the necessary actions and predicates.
- e1: e1_p19 score=10.0 valid=True terminal=war
  Reason: The path connects the entity to the war but lacks coverage of the necessary actions and predicates.
- e1: e1_p20 score=10.0 valid=True terminal=war
  Reason: The path connects the entity to the war but lacks coverage of the necessary actions and predicates.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p3

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p3'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- sparking_the_marian -> queen (helped)
- queen -> escape (to escape)
- escape -> imprisonment (from imprisonment)
### ast_ps2 (ps2)
- sparking_the_marian -> queen (helped)
- queen -> escape (action of escaping)
- escape -> imprisonment (from)

## 10. LLM Best AST Selection
- ast_ps1: score=0.92 valid=True reason=This AST effectively connects the entity to the queen and the escape action, covering all necessary predicates and answer intent.
- ast_ps2: score=0.9 valid=True reason=This AST also connects the entity to the queen and the escape action, but it includes a less direct relation hint.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- sparking_the_marian: Sparking the Marian (entity)
- queen: queen (type_variable)
- escape: escape (type_variable)
- imprisonment: imprisonment (value_slot)

Edges:
- sparking_the_marian -> queen (helped)
- queen -> escape (to escape)
- escape -> imprisonment (from imprisonment)

## 11. Atomic Subquestion DAG
- None: Who is the queen associated with Sparking the Marian?
- None: How did the queen of Sparking the Marian escape?
- None: What is the imprisonment of the queen of Sparking the Marian?
