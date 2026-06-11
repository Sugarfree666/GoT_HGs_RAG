# DEPO Decomposition #3

- Dataset: `2wikimultihopqa`
- Question: What is the place of birth of the performer of song Changed It?
- Gold answer: Port of Spain

## 1. Semantic-Normalized Question
What is the place of birth of the performer of the song Changed It?

## 2. Explicit Entities
- Changed It (Song) span=(56, 66)

## 3. Entity Masking
- SongA -> Changed It

What is the place of birth of the performer of the song SongA?

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
- What[1] --punct--> ?[14]

## 5. Undirected Dependency Graph
- What[1] --cop-- is[2]
- What[1] --nsubj-- place[4]
- What[1] --punct-- ?[14]
- the[3] --det-- place[4]
- place[4] --nmod:of-- birth[6]
- of[5] --case-- birth[6]
- birth[6] --nmod:of-- performer[9]
- of[7] --case-- performer[9]
- the[8] --det-- performer[9]
- performer[9] --nmod:of-- Changed It[13]
- of[10] --case-- Changed It[13]
- the[11] --det-- Changed It[13]
- song[12] --compound-- Changed It[13]

## 6. Entity Start Nodes from Explicit Entities
- e1: Changed It graph_node_ids=['13']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Changed It -- performer -- birth -- place -- What
- e1_p2 (e1): Changed It -- performer -- birth -- place -- What -- is
- e1_p3 (e1): Changed It -- performer -- birth -- place -- What -- ?
- e1_p4 (e1): Changed It -- performer -- birth -- place
- e1_p5 (e1): Changed It -- performer -- birth -- place -- the
- e1_p6 (e1): Changed It -- performer -- birth
- e1_p7 (e1): Changed It -- performer -- birth -- of
- e1_p8 (e1): Changed It -- performer
- e1_p9 (e1): Changed It -- song
- e1_p10 (e1): Changed It -- performer -- of
- e1_p11 (e1): Changed It -- performer -- the
- e1_p12 (e1): Changed It -- of
- e1_p13 (e1): Changed It -- the

## 7.5 Terminal Glue Path Pruning
Total raw paths: 13
Total kept paths: 5
Total pruned paths: 8
Total pruned ratio: 61.54%

### By Entity
- e1 / Changed It
  - raw: 13
  - kept: 5
  - pruned: 8
  - fallback_used: False
  - examples:
    - e1_p2: Changed It -> performer -> birth -> place -> What -> is [terminal=is, reason=terminal_glue_token]
    - e1_p3: Changed It -> performer -> birth -> place -> What -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p5: Changed It -> performer -> birth -> place -> the [terminal=the, reason=terminal_glue_token]
    - e1_p7: Changed It -> performer -> birth -> of [terminal=of, reason=terminal_glue_token]
    - e1_p10: Changed It -> performer -> of [terminal=of, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=place_of_birth
  Reason: The path starts from 'Changed It', covers the performer and the necessary roles leading to the place of birth, and includes the 'what' cue.
- e1: e1_p4 score=85.0 valid=True terminal=place_of_birth
  Reason: The path starts from 'Changed It', covers the performer and the necessary roles leading to the place of birth, and includes the 'what' cue.
- e1: e1_p6 score=75.0 valid=True terminal=place_of_birth
  Reason: The path starts from 'Changed It' and covers the performer and birth, but it misses the final role of place.
- e1: e1_p8 score=50.0 valid=True terminal=place_of_birth
  Reason: The path starts from 'Changed It' and covers the performer but does not reach the necessary roles of birth and place.
- e1: e1_p9 score=30.0 valid=True terminal=place_of_birth
  Reason: The path starts from 'Changed It' and connects to 'song', but it does not cover the necessary roles or answer intent.

## 8.1 Highest-Scored Path per Entity
- e1: e1_p1 score=90.0

## 8.2 Selected Path Set
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: What is the place of birth of the performer of song Changed It?
- ps1
  - e1_p1: Changed It -> performer -> birth -> place -> What

Output:
- reason: The DAG compiles each semantic reasoning edge into one atomic lookup.
- q1: Who performed Changed It? depends_on=[] support=[]
- q2: Where was the performer of Changed It born? depends_on=['q1'] support=[]

## 10. Atomic Subquestion DAG
- None: Who performed Changed It?
- None: Where was the performer of Changed It born?
