# DEPO Decomposition #4

- Dataset: `hotpotqa`
- Question: Sparking the Marian civil war, who helped the recently abdicated queen to escape her imprisonment?
- Gold answer: the Queen's gaoler

## 1. Semantic-Normalized Question
Sparking the Marian civil war, who helped the queen who recently abdicated to escape her imprisonment?

## 2. Explicit Entities
- Sparking the Marian (Entity) span=(0, 19)

## 3. Entity Masking
- EntityA -> Sparking the Marian

EntityA civil war, who helped the queen who recently abdicated to escape her imprisonment?

## 4. CoreNLP Dependency Parse
- war[3] --compound--> EntityA[1]
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

## 6. Entity Start Nodes from Explicit Entities
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
  Reason: The path effectively connects the entity to the queen and the action of escaping, covering the necessary predicates and answer intent.
- e1: e1_p2 score=85.0 valid=True terminal=escape
  Reason: The path includes the necessary elements to connect the entity to the queen and her escape, but it slightly lacks in direct execution potential.
- e1: e1_p3 score=90.0 valid=True terminal=escape
  Reason: The path effectively connects the entity to the queen and the action of escaping, covering the necessary predicates and answer intent.
- e1: e1_p4 score=80.0 valid=True terminal=abdicated
  Reason: The path connects the entity to the queen and her abdication but lacks a direct connection to the escape action.
- e1: e1_p5 score=85.0 valid=True terminal=escape
  Reason: The path effectively connects the entity to the queen and the action of escaping, covering the necessary predicates and answer intent.
- e1: e1_p6 score=80.0 valid=True terminal=escape
  Reason: The path connects the entity to the queen and the action of escaping but lacks a direct connection to imprisonment.
- e1: e1_p7 score=80.0 valid=True terminal=escape
  Reason: The path connects the entity to the queen and the action of escaping but lacks a direct connection to abdication.
- e1: e1_p8 score=75.0 valid=True terminal=escape
  Reason: The path connects the entity to the queen and the action of escaping but includes a preposition that detracts from its effectiveness.
- e1: e1_p9 score=70.0 valid=True terminal=abdicated
  Reason: The path connects the entity to the queen and her abdication but lacks a connection to escape.
- e1: e1_p10 score=70.0 valid=True terminal=escape
  Reason: The path connects the entity to the queen and the action of escaping but lacks a connection to abdication.
- e1: e1_p11 score=75.0 valid=True terminal=escape
  Reason: The path connects the entity to the queen and the action of escaping but includes a preposition that detracts from its effectiveness.
- e1: e1_p12 score=70.0 valid=True terminal=who
  Reason: The path connects the entity to the queen and includes the wh-word 'who', but lacks a connection to escape.
- e1: e1_p13 score=60.0 valid=True terminal=queen
  Reason: The path connects the entity to the queen but lacks a connection to escape and does not cover the answer intent fully.
- e1: e1_p14 score=50.0 valid=True terminal=queen
  Reason: The path connects the entity to the queen but includes a determiner that detracts from its effectiveness and lacks a connection to escape.
- e1: e1_p15 score=50.0 valid=True terminal=who
  Reason: The path connects the entity to the action of helping but lacks a connection to the queen and escape.
- e1: e1_p16 score=30.0 valid=True terminal=war
  Reason: The path connects the entity to the war but does not cover any relevant cues or answer intent.
- e1: e1_p17 score=30.0 valid=True terminal=helped
  Reason: The path connects the entity to the action of helping but does not cover any relevant cues or answer intent.
- e1: e1_p18 score=0.0 valid=False
  Reason: The path is too short and does not connect to any relevant cues or answer intent.
- e1: e1_p19 score=0.0 valid=False
  Reason: The path includes punctuation and does not connect to any relevant cues or answer intent.
- e1: e1_p20 score=0.0 valid=False
  Reason: The path includes punctuation and does not connect to any relevant cues or answer intent.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p3

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p3'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Sparking the Marian civil war, who helped the recently abdicated queen to escape her imprisonment?
- ps1
  - e1_p1: Sparking the Marian -> war -> helped -> queen -> abdicated -> escape -> imprisonment
- ps2
  - e1_p3: Sparking the Marian -> war -> helped -> queen -> escape -> imprisonment -> her

Output:
- selected_path_set_ids: ['ps1', 'ps2']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who helped the queen escape? depends_on=[] support=['e1_p1']
- q2: Who is the recently abdicated queen? depends_on=[] support=['e1_p1']

## 10. Atomic Subquestion DAG
- None: Who helped the queen escape?
- None: Who is the recently abdicated queen?
