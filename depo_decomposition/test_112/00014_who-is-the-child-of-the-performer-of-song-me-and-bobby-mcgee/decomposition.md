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

## 7.5 Terminal Glue Path Pruning
Total raw paths: 11
Total kept paths: 4
Total pruned paths: 7
Total pruned ratio: 63.64%

### By Entity
- e1 / Me And Bobby Mcgee
  - raw: 11
  - kept: 4
  - pruned: 7
  - fallback_used: False
  - examples:
    - e1_p2: Me And Bobby Mcgee -> performer -> child -> Who -> is [terminal=is, reason=terminal_glue_token]
    - e1_p3: Me And Bobby Mcgee -> performer -> child -> Who -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p5: Me And Bobby Mcgee -> performer -> child -> the [terminal=the, reason=terminal_glue_token]
    - e1_p8: Me And Bobby Mcgee -> performer -> of [terminal=of, reason=terminal_glue_token]
    - e1_p9: Me And Bobby Mcgee -> performer -> the [terminal=the, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=child
  Reason: The path starts from the song, connects to the performer, and then to the child, effectively covering the question's intent.
- e1: e1_p4 score=85.0 valid=True terminal=child
  Reason: The path starts from the song, connects to the performer, and then to the child, but lacks the explicit wh cue.
- e1: e1_p6 score=60.0 valid=True terminal=performer
  Reason: The path only connects the song to the performer, missing the crucial child and wh cues.
- e1: e1_p7 score=50.0 valid=True terminal=song
  Reason: The path connects the song to itself but does not reach the performer or the child, missing key elements of the question.

## 8.1 Highest-Scored Path per Entity
- e1: e1_p1 score=90.0

## 8.2 Selected Path Set
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Who is the child of the performer of song Me And Bobby Mcgee?
- ps1
  - e1_p1: Me And Bobby Mcgee -> performer -> child -> Who

Output:
- reason: The DAG compiles each semantic reasoning edge into one atomic lookup.
- q1: Who performed Me And Bobby Mcgee? depends_on=[] support=[]
- q2: Who is the child of the performer of Me And Bobby Mcgee? depends_on=['q1'] support=[]

## 10. Atomic Subquestion DAG
- None: Who performed Me And Bobby Mcgee?
- None: Who is the child of the performer of Me And Bobby Mcgee?
