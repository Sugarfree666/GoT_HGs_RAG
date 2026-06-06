# DEPO Decomposition #20

- Dataset: `hotpotqa`
- Question: Where is the ice hockey team based that Zdeno Chára currently serving as captain of?
- Gold answer: Boston, Massachusetts

## 1. Semantic-Normalized Question
Where is the ice hockey team based that Zdeno Chára is currently serving as captain of?

## 2. Mask Spans
- Zdeno Ch (entity, ZdenoCh)

## 3. Selective Masked Question
Where is the ice hockey team based that SomeEntityAára is currently serving as captain of?

## 4. CoreNLP Dependency Parse
- is[2] --advmod--> Where[1]
- based[7] --aux:pass--> is[2]
- team[6] --det--> the[3]
- team[6] --compound--> ice[4]
- team[6] --compound--> hockey[5]
- based[7] --nsubj:pass--> team[6]
- serving[12] --mark--> that[8]
- serving[12] --nsubj--> SomeEntityAára[9]
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
- SomeEntityAára[9] --nsubj-- serving[12]
- is[10] --aux-- serving[12]
- currently[11] --advmod-- serving[12]
- serving[12] --obl:as-- captain[14]
- as[13] --case-- captain[14]
- captain[14] --acl-- of[15]

## 6. Entity Start Nodes
- e1: SomeEntityAára graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): SomeEntityAára -- serving -- based -- team -- ice
- e1_p2 (e1): SomeEntityAára -- serving -- based -- team -- hockey
- e1_p3 (e1): SomeEntityAára -- serving -- based -- team
- e1_p4 (e1): SomeEntityAára -- serving -- captain -- as
- e1_p5 (e1): SomeEntityAára -- serving -- based -- team -- the
- e1_p6 (e1): SomeEntityAára -- serving -- based -- is -- Where
- e1_p7 (e1): SomeEntityAára -- serving -- based
- e1_p8 (e1): SomeEntityAára -- serving -- currently
- e1_p9 (e1): SomeEntityAára -- serving -- captain
- e1_p10 (e1): SomeEntityAára -- serving -- based -- is
- e1_p11 (e1): SomeEntityAára -- serving -- based -- ?
- e1_p12 (e1): SomeEntityAára -- serving -- captain -- of
- e1_p13 (e1): SomeEntityAára -- serving
- e1_p14 (e1): SomeEntityAára -- serving -- that
- e1_p15 (e1): SomeEntityAára -- serving -- is

## 8. LLM Path Scores
- e1: e1_p1 score=85.0 valid=True terminal=location
  Reason: The path starts from SomeEntityAára, reaches team, and includes the based predicate, covering the location intent.
- e1: e1_p2 score=90.0 valid=True terminal=location
  Reason: The path starts from SomeEntityAára, reaches team, includes the hockey modifier, and covers the location intent effectively.
- e1: e1_p3 score=80.0 valid=True terminal=location
  Reason: The path starts from SomeEntityAára and reaches team, covering the location intent but lacks additional modifiers.
- e1: e1_p4 score=55.0 valid=True terminal=captain
  Reason: The path starts from SomeEntityAára and reaches captain, but it does not address the location intent.
- e1: e1_p5 score=70.0 valid=True terminal=location
  Reason: The path starts from SomeEntityAára, reaches team, and includes the determiner, but it does not effectively cover the location intent.
- e1: e1_p6 score=60.0 valid=True terminal=location
  Reason: The path starts from SomeEntityAára and includes the auxiliary is, but it does not effectively lead to the location intent.
- e1: e1_p7 score=50.0 valid=True terminal=location
  Reason: The path starts from SomeEntityAára and reaches based, but it does not lead to the location intent.
- e1: e1_p8 score=40.0 valid=True terminal=currently
  Reason: The path starts from SomeEntityAára and includes currently, but it does not address the location intent.
- e1: e1_p9 score=50.0 valid=True terminal=captain
  Reason: The path starts from SomeEntityAára and reaches captain, but it does not address the location intent.
- e1: e1_p10 score=55.0 valid=True terminal=location
  Reason: The path starts from SomeEntityAára and includes is, but it does not effectively lead to the location intent.
- e1: e1_p11 score=30.0 valid=True terminal=punctuation
  Reason: The path starts from SomeEntityAára and ends with punctuation, failing to address the location intent.
- e1: e1_p12 score=50.0 valid=True terminal=captain
  Reason: The path starts from SomeEntityAára and reaches captain, but it does not address the location intent.
- e1: e1_p13 score=20.0 valid=True terminal=serving
  Reason: The path starts from SomeEntityAára and only includes serving, failing to address the location intent.
- e1: e1_p14 score=25.0 valid=True terminal=that
  Reason: The path starts from SomeEntityAára and includes that, but it does not address the location intent.
- e1: e1_p15 score=30.0 valid=True terminal=is
  Reason: The path starts from SomeEntityAára and includes is, but it does not effectively lead to the location intent.

## 8.1 Top-2 Paths per Entity
- e1: e1_p2, e1_p1

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p2'} mean_path_score=90.0
- ps2: {'e1': 'e1_p1'} mean_path_score=85.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- zdeno_char -> team (team of Zdeno Chára)
- team -> location (location of the team)
### ast_ps2 (ps2)
- zdeno_chara -> team (team of Zdeno Chára)
- team -> location (location of the team)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively captures the relationship between Zdeno Chára and the team, leading to the location, thus supporting the decomposition into atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- zdeno_char: Zdeno Chára (entity)
- team: team (type_variable)
- location: location (value_slot)

Edges:
- zdeno_char -> team (team of Zdeno Chára)
- team -> location (location of the team)

## 11. Atomic Subquestion DAG
- None: What team is Zdeno Chára currently serving as captain of?
- None: Where is the team of Zdeno Chára based?
