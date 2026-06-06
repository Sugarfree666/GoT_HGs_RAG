# DEPO Decomposition #14

- Dataset: `2wikimultihopqa`
- Question: Who is the child of the performer of song Me And Bobby Mcgee?
- Gold answer: Dean Miller

## 1. Semantic-Normalized Question
Who is the child of the performer of the song Me And Bobby Mcgee?

## 2. Mask Spans
- Me And Bobby Mcgee? (entity, Song)

## 3. Selective Masked Question
Who is the child of the performer of the song SongA

## 4. CoreNLP Dependency Parse
- Who[1] --cop--> is[2]
- child[4] --det--> the[3]
- Who[1] --nsubj--> child[4]
- performer[7] --case--> of[5]
- performer[7] --det--> the[6]
- child[4] --nmod:of--> performer[7]
- SongA[11] --case--> of[8]
- SongA[11] --det--> the[9]
- SongA[11] --compound--> song[10]
- performer[7] --nmod:of--> SongA[11]

## 5. Undirected Dependency Graph
- Who[1] --cop-- is[2]
- Who[1] --nsubj-- child[4]
- the[3] --det-- child[4]
- child[4] --nmod:of-- performer[7]
- of[5] --case-- performer[7]
- the[6] --det-- performer[7]
- performer[7] --nmod:of-- Me And Bobby Mcgee?[11]
- of[8] --case-- Me And Bobby Mcgee?[11]
- the[9] --det-- Me And Bobby Mcgee?[11]
- song[10] --compound-- Me And Bobby Mcgee?[11]

## 6. Entity Start Nodes
- e1: Me And Bobby Mcgee? graph_node_ids=['11']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Me And Bobby Mcgee? -- performer -- child -- Who
- e1_p2 (e1): Me And Bobby Mcgee? -- performer -- child -- Who -- is
- e1_p3 (e1): Me And Bobby Mcgee? -- performer -- child
- e1_p4 (e1): Me And Bobby Mcgee? -- performer -- child -- the
- e1_p5 (e1): Me And Bobby Mcgee? -- performer
- e1_p6 (e1): Me And Bobby Mcgee? -- song
- e1_p7 (e1): Me And Bobby Mcgee? -- performer -- of
- e1_p8 (e1): Me And Bobby Mcgee? -- performer -- the
- e1_p9 (e1): Me And Bobby Mcgee? -- of
- e1_p10 (e1): Me And Bobby Mcgee? -- the

## 8. LLM Selected Entity Paths
- e1: e1_p1 Me And Bobby Mcgee? -- performer -- child -- Who
  Reason: This path provides a complete reasoning chain from the song to its performer and then to the child, directly addressing the question.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "who",
  "answer_kind": "person_or_entity",
  "answer_slot_hint": null,
  "focus_predicate": null,
  "focus_noun": null,
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- me_and_bobby_mcgee: Me And Bobby Mcgee (entity)
- performer: performer (type_variable)
- child: child (type_variable)

Edges:
- me_and_bobby_mcgee -> performer (performer of Me And Bobby Mcgee)
- performer -> child (child of performer)

## 10. Atomic Subquestion DAG
- None: Who is the performer of the song Me And Bobby Mcgee?
- None: Who is the child of the performer of Me And Bobby Mcgee?
