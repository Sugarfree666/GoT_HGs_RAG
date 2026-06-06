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

## 8. LLM Selected Entity Paths
- e1: e1_p1 SomeEntityAára -- serving -- based -- team -- ice
  Reason: This path effectively connects the entity to the reasoning about the team based on the context of serving as captain.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "where",
  "answer_kind": "location",
  "answer_slot_hint": "location",
  "focus_predicate": null,
  "focus_noun": null,
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- zdeno_charra: Zdeno Chára (entity)
- ice_hockey_team: ice hockey team (type_variable)
- location: location (value_slot)

Edges:
- zdeno_charra -> ice_hockey_team (team that Zdeno Chára is serving as captain of)
- ice_hockey_team -> location (location of the ice hockey team)

## 10. Atomic Subquestion DAG
- None: What is the ice hockey team that Zdeno Chára is currently serving as captain of?
- None: Where is the ice hockey team of Zdeno Chára based?
