# DEPO Decomposition #20

- Dataset: `hotpotqa`
- Question: Where is the ice hockey team based that Zdeno Chára currently serving as captain of?
- Gold answer: Boston, Massachusetts

## 1. Semantic-Normalized Question
Where is the ice hockey team based that Zdeno Chára is currently serving as captain of?

## 2. Explicit Entities
- Zdeno Ch (Entity) span=(40, 48)

## 3. Entity Masking
- EntityA -> Zdeno Ch

Where is the ice hockey team based that EntityAára is currently serving as captain of?

## 4. CoreNLP Dependency Parse
- is[2] --advmod--> Where[1]
- based[7] --aux:pass--> is[2]
- team[6] --det--> the[3]
- team[6] --compound--> ice[4]
- team[6] --compound--> hockey[5]
- based[7] --nsubj:pass--> team[6]
- serving[12] --mark--> that[8]
- serving[12] --nsubj--> EntityAára[9]
- serving[12] --aux--> is[10]
- serving[12] --advmod--> currently[11]
- based[7] --ccomp--> serving[12]
- captain[14] --case--> as[13]
- serving[12] --obl:as--> captain[14]
- captain[14] --acl--> of[15]
- based[7] --punct--> ?[16]

## 5. Undirected Dependency Graph
- Where[1] --advmod-- is[2]
- is[2] --aux:pass-- based[7]
- the[3] --det-- team[6]
- ice[4] --compound-- team[6]
- hockey[5] --compound-- team[6]
- team[6] --nsubj:pass-- based[7]
- based[7] --ccomp-- serving[12]
- based[7] --punct-- ?[16]
- that[8] --mark-- serving[12]
- Zdeno Ch[9] --nsubj-- serving[12]
- is[10] --aux-- serving[12]
- currently[11] --advmod-- serving[12]
- serving[12] --obl:as-- captain[14]
- as[13] --case-- captain[14]
- captain[14] --acl-- of[15]

## 6. Entity Start Nodes from Explicit Entities
- e1: Zdeno Ch graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Zdeno Ch -- serving -- based -- team -- ice
- e1_p2 (e1): Zdeno Ch -- serving -- based -- team -- hockey
- e1_p3 (e1): Zdeno Ch -- serving -- based -- team
- e1_p4 (e1): Zdeno Ch -- serving -- captain -- as
- e1_p5 (e1): Zdeno Ch -- serving -- based -- team -- the
- e1_p6 (e1): Zdeno Ch -- serving -- based -- is -- Where
- e1_p7 (e1): Zdeno Ch -- serving -- based
- e1_p8 (e1): Zdeno Ch -- serving -- currently
- e1_p9 (e1): Zdeno Ch -- serving -- captain
- e1_p10 (e1): Zdeno Ch -- serving -- based -- is
- e1_p11 (e1): Zdeno Ch -- serving -- based -- ?
- e1_p12 (e1): Zdeno Ch -- serving -- captain -- of
- e1_p13 (e1): Zdeno Ch -- serving
- e1_p14 (e1): Zdeno Ch -- serving -- that
- e1_p15 (e1): Zdeno Ch -- serving -- is

## 8. LLM Path Scores
- e1: e1_p1 score=85.0 valid=True terminal=location
  Reason: The path starts from Zdeno Chára, reaches the team and based, covering the location intent.
- e1: e1_p2 score=90.0 valid=True terminal=location
  Reason: The path starts from Zdeno Chára, reaches the hockey team and based, effectively covering the location intent.
- e1: e1_p3 score=75.0 valid=True terminal=location
  Reason: The path starts from Zdeno Chára and reaches the team and based, but lacks some additional context.
- e1: e1_p4 score=30.0 valid=True terminal=captain
  Reason: The path starts from Zdeno Chára and reaches serving and captain, but does not address the location intent.
- e1: e1_p5 score=70.0 valid=True terminal=location
  Reason: The path starts from Zdeno Chára, reaches the team and based, but includes unnecessary noise.
- e1: e1_p6 score=40.0 valid=True terminal=location
  Reason: The path starts from Zdeno Chára and includes is and where, but lacks the necessary context for the team.
- e1: e1_p7 score=55.0 valid=True terminal=location
  Reason: The path starts from Zdeno Chára and reaches serving and based, but is too minimal.
- e1: e1_p8 score=50.0 valid=True terminal=location
  Reason: The path starts from Zdeno Chára and includes currently, but lacks the necessary context for the location.
- e1: e1_p9 score=60.0 valid=True terminal=captain
  Reason: The path starts from Zdeno Chára and reaches serving and captain, but does not address the location intent.
- e1: e1_p10 score=45.0 valid=True terminal=location
  Reason: The path starts from Zdeno Chára and includes is, but lacks the necessary context for the team.
- e1: e1_p11 score=20.0 valid=True terminal=location
  Reason: The path starts from Zdeno Chára but ends with a punctuation mark, failing to provide useful information.
- e1: e1_p12 score=35.0 valid=True terminal=captain
  Reason: The path starts from Zdeno Chára and reaches serving and captain, but does not address the location intent.
- e1: e1_p13 score=10.0 valid=True terminal=location
  Reason: The path starts from Zdeno Chára but is too minimal to provide useful information.
- e1: e1_p14 score=25.0 valid=True terminal=location
  Reason: The path starts from Zdeno Chára and includes that, but does not address the location intent.
- e1: e1_p15 score=15.0 valid=True terminal=location
  Reason: The path starts from Zdeno Chára but is too minimal to provide useful information.

## 8.1 Top-2 Paths per Entity
- e1: e1_p2, e1_p1

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p2'} mean_path_score=90.0
- ps2: {'e1': 'e1_p1'} mean_path_score=85.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Where is the ice hockey team based that Zdeno Chára currently serving as captain of?
- ps1
  - e1_p2: Zdeno Ch -> serving -> based -> team -> hockey
- ps2
  - e1_p1: Zdeno Ch -> serving -> based -> team -> ice

Output:
- selected_path_set_ids: ['ps1', 'ps2']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: What team is Zdeno Chára currently serving as captain of? depends_on=[] support=['e1_p2']
- q2: Where is the team that q1's answer is based? depends_on=['q1'] support=['e1_p2']

## 10. Atomic Subquestion DAG
- None: What team is Zdeno Chára currently serving as captain of?
- None: Where is the team that q1's answer is based?
