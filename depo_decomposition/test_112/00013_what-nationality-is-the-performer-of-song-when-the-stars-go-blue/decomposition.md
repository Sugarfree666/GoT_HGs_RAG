# DEPO Decomposition #13

- Dataset: `2wikimultihopqa`
- Question: What nationality is the performer of song When The Stars Go Blue?
- Gold answer: America

## 1. Semantic-Normalized Question
What nationality is the performer of the song When The Stars Go Blue?

## 2. Explicit Entities
- The Stars Go Blue (Song) span=(51, 68)

## 3. Entity Masking
- SongA -> The Stars Go Blue

What nationality is the performer of the song When SongA?

## 4. CoreNLP Dependency Parse
- nationality[2] --det--> What[1]
- is[3] --obj--> nationality[2]
- performer[5] --det--> the[4]
- is[3] --nsubj--> performer[5]
- song[8] --case--> of[6]
- song[8] --det--> the[7]
- performer[5] --nmod:of--> song[8]
- SongA[10] --advmod--> When[9]
- is[3] --dep--> SongA[10]
- is[3] --punct--> ?[11]

## 5. Undirected Dependency Graph
- What[1] --det-- nationality[2]
- nationality[2] --obj-- is[3]
- is[3] --nsubj-- performer[5]
- is[3] --dep-- The Stars Go Blue[10]
- is[3] --punct-- ?[11]
- the[4] --det-- performer[5]
- performer[5] --nmod:of-- song[8]
- of[6] --case-- song[8]
- the[7] --det-- song[8]
- When[9] --advmod-- The Stars Go Blue[10]

## 6. Entity Start Nodes from Explicit Entities
- e1: The Stars Go Blue graph_node_ids=['10']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): The Stars Go Blue -- is -- performer -- song
- e1_p2 (e1): The Stars Go Blue -- is -- performer -- song -- of
- e1_p3 (e1): The Stars Go Blue -- is -- performer -- song -- the
- e1_p4 (e1): The Stars Go Blue -- is -- nationality -- What
- e1_p5 (e1): The Stars Go Blue -- is -- nationality
- e1_p6 (e1): The Stars Go Blue -- is -- performer
- e1_p7 (e1): The Stars Go Blue -- is -- performer -- the
- e1_p8 (e1): The Stars Go Blue -- When
- e1_p9 (e1): The Stars Go Blue -- is
- e1_p10 (e1): The Stars Go Blue -- is -- ?

## 7.5 Terminal Glue Path Pruning
Total raw paths: 10
Total kept paths: 5
Total pruned paths: 5
Total pruned ratio: 50.00%

### By Entity
- e1 / The Stars Go Blue
  - raw: 10
  - kept: 5
  - pruned: 5
  - fallback_used: False
  - examples:
    - e1_p2: The Stars Go Blue -> is -> performer -> song -> of [terminal=of, reason=terminal_glue_token]
    - e1_p3: The Stars Go Blue -> is -> performer -> song -> the [terminal=the, reason=terminal_glue_token]
    - e1_p7: The Stars Go Blue -> is -> performer -> the [terminal=the, reason=terminal_glue_token]
    - e1_p9: The Stars Go Blue -> is [terminal=is, reason=terminal_glue_token]
    - e1_p10: The Stars Go Blue -> is -> ? [terminal=?, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=70.0 valid=True terminal=performer
  Reason: The path starts from the song and identifies the performer, but it does not reach the nationality, which is crucial for the answer.
- e1: e1_p4 score=85.0 valid=True terminal=nationality
  Reason: The path starts from the song, identifies the performer, and directly addresses the nationality, aligning well with the question intent.
- e1: e1_p5 score=80.0 valid=True terminal=nationality
  Reason: The path identifies the nationality but lacks the performer, which is necessary to fully support the question.
- e1: e1_p6 score=75.0 valid=True terminal=performer
  Reason: The path identifies the performer but does not address the nationality, which is essential for the answer.
- e1: e1_p8 score=30.0 valid=True terminal=none
  Reason: The path only connects the song to the word 'When', which does not contribute to answering the question.

## 8.1 Highest-Scored Path per Entity
- e1: e1_p4 score=85.0

## 8.2 Selected Path Set
- ps1: {'e1': 'e1_p4'} mean_path_score=85.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: What nationality is the performer of song When The Stars Go Blue?
- ps1
  - e1_p4: The Stars Go Blue -> is -> nationality -> What

Output:
- reason: The DAG compiles each semantic reasoning edge into one atomic lookup.
- q1: Who performed When The Stars Go Blue? depends_on=[] support=[]
- q2: What is the nationality of the performer of When The Stars Go Blue? depends_on=['q1'] support=[]

## 10. Atomic Subquestion DAG
- None: Who performed When The Stars Go Blue?
- None: What is the nationality of the performer of When The Stars Go Blue?
