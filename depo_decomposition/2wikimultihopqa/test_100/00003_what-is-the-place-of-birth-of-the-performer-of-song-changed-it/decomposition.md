# DEPO Decomposition #3

- Dataset: `2wikimultihopqa`
- Question: What is the place of birth of the performer of song Changed It?
- Gold answer: Port of Spain

## 1. Semantic-Normalized Question
What is the place of birth of the performer of the song Changed It?

## 2. Mask Spans
- Changed It? (entity, Song)

## 3. Selective Masked Question
What is the place of birth of the performer of the song SongA

## 4. CoreNLP Dependency Parse
- What[1] --cop--> is[2]
- place[4] --det--> the[3]
- What[1] --nsubj--> place[4]
- birth[6] --case--> of[5]
- place[4] --nmod:of--> birth[6]
- performer[9] --case--> of[7]
- performer[9] --det--> the[8]
- birth[6] --nmod:of--> performer[9]
- SongA[13] --case--> of[10]
- SongA[13] --det--> the[11]
- SongA[13] --compound--> song[12]
- performer[9] --nmod:of--> SongA[13]

## 5. Undirected Dependency Graph
- What[1] --cop-- is[2]
- What[1] --nsubj-- place[4]
- the[3] --det-- place[4]
- place[4] --nmod:of-- birth[6]
- of[5] --case-- birth[6]
- birth[6] --nmod:of-- performer[9]
- of[7] --case-- performer[9]
- the[8] --det-- performer[9]
- performer[9] --nmod:of-- Changed It?[13]
- of[10] --case-- Changed It?[13]
- the[11] --det-- Changed It?[13]
- song[12] --compound-- Changed It?[13]

## 6. Entity Start Nodes
- e1: Changed It? graph_node_ids=['13']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Changed It? -- performer -- birth -- place -- What
- e1_p2 (e1): Changed It? -- performer -- birth -- place -- What -- is
- e1_p3 (e1): Changed It? -- performer -- birth -- place
- e1_p4 (e1): Changed It? -- performer -- birth -- place -- the
- e1_p5 (e1): Changed It? -- performer -- birth
- e1_p6 (e1): Changed It? -- performer -- birth -- of
- e1_p7 (e1): Changed It? -- performer
- e1_p8 (e1): Changed It? -- song
- e1_p9 (e1): Changed It? -- performer -- of
- e1_p10 (e1): Changed It? -- performer -- the
- e1_p11 (e1): Changed It? -- of
- e1_p12 (e1): Changed It? -- the

## 8. LLM Selected Entity Paths
- e1: e1_p1 Changed It? -- performer -- birth -- place -- What
  Reason: This path provides a complete reasoning chain from the song 'Changed It?' to the performer's place of birth, which is the answer to the question.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "what",
  "answer_kind": "entity_or_attribute",
  "answer_slot_hint": null,
  "focus_predicate": "born",
  "focus_noun": null,
  "requires_value_slot": true
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- changed_it: Changed It (entity)
- performer: performer (type_variable)
- birthplace: birthplace (value_slot)

Edges:
- changed_it -> performer (performer of Changed It)
- performer -> birthplace (birthplace of performer)

## 10. Atomic Subquestion DAG
- None: Who is the performer of the song Changed It?
- None: What is the birthplace of the performer of Changed It?
