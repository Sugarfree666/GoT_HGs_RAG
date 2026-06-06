# DEPO Decomposition #14

- Dataset: `2wikimultihopqa`
- Question: Who is the child of the performer of song Me And Bobby Mcgee?
- Gold answer: Dean Miller

## 1. Semantic-Normalized Question
Who is the child of the performer of the song Me And Bobby Mcgee?

## 2. Explicit Entities
- Me And Bobby Mcgee (Song) span=(46, 64)

## 3. Entity Masking
- SongA -> Me And Bobby Mcgee

Who is the child of the performer of the song SongA?

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
- Who[1] --punct--> ?[12]

## 5. Undirected Dependency Graph
- Who[1] --cop-- is[2]
- Who[1] --nsubj-- child[4]
- Who[1] --punct-- ?[12]
- the[3] --det-- child[4]
- child[4] --nmod:of-- performer[7]
- of[5] --case-- performer[7]
- the[6] --det-- performer[7]
- performer[7] --nmod:of-- Me And Bobby Mcgee[11]
- of[8] --case-- Me And Bobby Mcgee[11]
- the[9] --det-- Me And Bobby Mcgee[11]
- song[10] --compound-- Me And Bobby Mcgee[11]

## 6. Entity Start Nodes from Explicit Entities
- e1: Me And Bobby Mcgee graph_node_ids=['11']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Me And Bobby Mcgee -- performer -- child -- Who
- e1_p2 (e1): Me And Bobby Mcgee -- performer -- child -- Who -- is
- e1_p3 (e1): Me And Bobby Mcgee -- performer -- child -- Who -- ?
- e1_p4 (e1): Me And Bobby Mcgee -- performer -- child
- e1_p5 (e1): Me And Bobby Mcgee -- performer -- child -- the
- e1_p6 (e1): Me And Bobby Mcgee -- performer
- e1_p7 (e1): Me And Bobby Mcgee -- song
- e1_p8 (e1): Me And Bobby Mcgee -- performer -- of
- e1_p9 (e1): Me And Bobby Mcgee -- performer -- the
- e1_p10 (e1): Me And Bobby Mcgee -- of
- e1_p11 (e1): Me And Bobby Mcgee -- the

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=child
  Reason: The path starts from Me And Bobby Mcgee, reaches performer, then child, and includes the who cue.
- e1: e1_p2 score=95.0 valid=True terminal=child
  Reason: The path starts from Me And Bobby Mcgee, reaches performer, then child, and includes the who and is cues.
- e1: e1_p3 score=90.0 valid=True terminal=child
  Reason: The path starts from Me And Bobby Mcgee, reaches performer, then child, and includes the who cue.
- e1: e1_p4 score=85.0 valid=True terminal=child
  Reason: The path starts from Me And Bobby Mcgee, reaches performer, then child, but misses the who cue.
- e1: e1_p5 score=70.0 valid=True terminal=child
  Reason: The path starts from Me And Bobby Mcgee, reaches performer, then child, but ends with a determiner and misses the who cue.
- e1: e1_p6 score=50.0 valid=True terminal=child
  Reason: The path starts from Me And Bobby Mcgee and reaches performer but does not cover child or the who cue.
- e1: e1_p7 score=40.0 valid=True terminal=child
  Reason: The path starts from Me And Bobby Mcgee and reaches song but does not cover performer, child, or the who cue.
- e1: e1_p8 score=30.0 valid=True terminal=child
  Reason: The path starts from Me And Bobby Mcgee and reaches performer but ends with a preposition and does not cover child or the who cue.
- e1: e1_p9 score=30.0 valid=True terminal=child
  Reason: The path starts from Me And Bobby Mcgee and reaches performer but ends with a determiner and does not cover child or the who cue.
- e1: e1_p10 score=20.0 valid=True terminal=child
  Reason: The path starts from Me And Bobby Mcgee and reaches of but does not cover performer, child, or the who cue.
- e1: e1_p11 score=20.0 valid=True terminal=child
  Reason: The path starts from Me And Bobby Mcgee and reaches the but does not cover performer, child, or the who cue.

## 8.1 Top-2 Paths per Entity
- e1: e1_p2, e1_p1

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p2'} mean_path_score=95.0
- ps2: {'e1': 'e1_p1'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- me_and_bobby_mcgee -> performer (performer of Me And Bobby Mcgee)
- performer -> child (child of performer)
- child -> child_value (child)
### ast_ps2 (ps2)
- me_and_bobby_mcgee -> performer (performer of Me And Bobby Mcgee)
- performer -> child (child of performer)
- child -> who (who is child)

## 10. LLM Best AST Selection
- ast_ps1: score=0.95 valid=True reason=This AST effectively captures the relationship between the song 'Me And Bobby Mcgee', its performer, and the performer's child, allowing for direct decomposition into atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- me_and_bobby_mcgee: Me And Bobby Mcgee (entity)
- performer: performer (type_variable)
- child: child (type_variable)
- child_value: child (value_slot)

Edges:
- me_and_bobby_mcgee -> performer (performer of Me And Bobby Mcgee)
- performer -> child (child of performer)
- child -> child_value (child)

## 11. Atomic Subquestion DAG
- None: Who is the performer of the song Me And Bobby Mcgee?
- None: Who is the child of the performer of Me And Bobby Mcgee?
- None: Who is the child of the performer of Me And Bobby Mcgee?
