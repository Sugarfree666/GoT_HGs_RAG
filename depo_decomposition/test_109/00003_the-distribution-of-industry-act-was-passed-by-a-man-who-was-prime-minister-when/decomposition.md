# DEPO Decomposition #3

- Dataset: `hotpotqa`
- Question: The Distribution of Industry act was passed by a man who was prime minister when?
- Gold answer: 1945 to 1951

## 1. Semantic-Normalized Question
When was the Distribution of Industry act passed by a man who was prime minister?

## 2. Explicit Entities
- Distribution of Industry (Entity) span=(13, 37)

## 3. Entity Masking
- EntityA -> Distribution of Industry

When was the EntityA act passed by a man who was prime minister?

## 4. CoreNLP Dependency Parse
- passed[6] --advmod--> When[1]
- passed[6] --aux:pass--> was[2]
- act[5] --det--> the[3]
- act[5] --compound--> EntityA[4]
- passed[6] --nsubj:pass--> act[5]
- man[9] --case--> by[7]
- man[9] --det--> a[8]
- passed[6] --obl:agent--> man[9]
- minister[13] --nsubj--> man[9]
- man[9] --ref--> who[10]
- minister[13] --cop--> was[11]
- minister[13] --amod--> prime[12]
- man[9] --acl:relcl--> minister[13]
- passed[6] --punct--> ?[14]

## 5. Undirected Dependency Graph
- When[1] --advmod-- passed[6]
- was[2] --aux:pass-- passed[6]
- the[3] --det-- act[5]
- Distribution of Industry[4] --compound-- act[5]
- act[5] --nsubj:pass-- passed[6]
- passed[6] --obl:agent-- man[9]
- passed[6] --punct-- ?[14]
- by[7] --case-- man[9]
- a[8] --det-- man[9]
- man[9] --nsubj/acl:relcl-- minister[13]
- man[9] --ref-- who[10]
- was[11] --cop-- minister[13]
- prime[12] --amod-- minister[13]

## 6. Entity Start Nodes from Explicit Entities
- e1: Distribution of Industry graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Distribution of Industry -- act -- passed -- man -- minister -- prime
- e1_p2 (e1): Distribution of Industry -- act -- passed -- man -- minister
- e1_p3 (e1): Distribution of Industry -- act -- passed -- man -- minister -- was
- e1_p4 (e1): Distribution of Industry -- act -- passed -- man -- who
- e1_p5 (e1): Distribution of Industry -- act -- passed -- man
- e1_p6 (e1): Distribution of Industry -- act -- passed -- man -- by
- e1_p7 (e1): Distribution of Industry -- act -- passed -- man -- a
- e1_p8 (e1): Distribution of Industry -- act -- passed -- When
- e1_p9 (e1): Distribution of Industry -- act -- passed
- e1_p10 (e1): Distribution of Industry -- act -- passed -- was
- e1_p11 (e1): Distribution of Industry -- act -- passed -- ?
- e1_p12 (e1): Distribution of Industry -- act
- e1_p13 (e1): Distribution of Industry -- act -- the

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, and includes the 'when' cue for the date.
- e1: e1_p2 score=85.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, and includes the 'when' cue for the date, but lacks the auxiliary 'was'.
- e1: e1_p3 score=95.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, includes 'was', and the 'when' cue for the date.
- e1: e1_p4 score=80.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, and includes the 'when' cue, but lacks the auxiliary 'was'.
- e1: e1_p5 score=70.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, but does not include any auxiliary or wh-word cues.
- e1: e1_p6 score=60.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, but ends with a preposition 'by', which does not contribute to the answer.
- e1: e1_p7 score=50.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, but ends with a determiner 'a', which does not contribute to the answer.
- e1: e1_p8 score=65.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, but ends with 'When', which is not a complete answer.
- e1: e1_p9 score=55.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry' and covers the 'passed' predicate, but is too short and lacks auxiliary or wh-word cues.
- e1: e1_p10 score=60.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, but ends with 'was', which does not contribute to the answer.
- e1: e1_p11 score=40.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, but ends with a punctuation mark '?', which does not contribute to the answer.
- e1: e1_p12 score=30.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry' but is too short, lacking any meaningful connections to the question.
- e1: e1_p13 score=25.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry' but ends with a determiner 'the', which does not contribute to the answer.

## 8.1 Top-2 Paths per Entity
- e1: e1_p3, e1_p1

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p3'} mean_path_score=95.0
- ps2: {'e1': 'e1_p1'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: The Distribution of Industry act was passed by a man who was prime minister when?
- ps1
  - e1_p3: Distribution of Industry -> act -> passed -> man -> minister -> was
- ps2
  - e1_p1: Distribution of Industry -> act -> passed -> man -> minister -> prime

Output:
- selected_path_set_ids: ['ps1', 'ps2']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who is the man who passed the Distribution of Industry act? depends_on=[] support=['e1_p3']
- q2: What was the position of q1's answer? depends_on=['q1'] support=['e1_p1']
- q3: When was q2's answer in office? depends_on=['q2'] support=['e1_p1']

## 10. Atomic Subquestion DAG
- None: Who is the man who passed the Distribution of Industry act?
- None: What was the position of q1's answer?
- None: When was q2's answer in office?
