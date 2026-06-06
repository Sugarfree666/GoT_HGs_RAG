# DEPO Decomposition #3

- Dataset: `hotpotqa`
- Question: The Distribution of Industry act was passed by a man who was prime minister when?
- Gold answer: 1945 to 1951

## 1. Semantic-Normalized Question
When was the Distribution of Industry act passed by a man who was prime minister?

## 2. Mask Spans
- Distribution of Industry (entity, DistributionOfIndustry)

## 3. Selective Masked Question
When was the SomeEntityA act passed by a man who was prime minister?

## 4. CoreNLP Dependency Parse
- passed[6] --advmod--> When[1]
- passed[6] --aux:pass--> was[2]
- act[5] --det--> the[3]
- act[5] --compound--> SomeEntityA[4]
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

## 6. Entity Start Nodes
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
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, and includes the 'who' cue, supporting the intent to find out when the act was passed.
- e1: e1_p2 score=85.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, and includes the 'who' cue, but lacks a direct connection to the auxiliary or additional context.
- e1: e1_p3 score=90.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, includes the auxiliary 'was', and the 'who' cue, supporting the intent to find out when the act was passed.
- e1: e1_p4 score=75.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, and includes the 'who' cue, but lacks a direct connection to the auxiliary or additional context.
- e1: e1_p5 score=70.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, but does not include any auxiliary or additional context, making it less effective.
- e1: e1_p6 score=60.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, but ends with a preposition 'by', which does not support the intent effectively.
- e1: e1_p7 score=55.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, but ends with a determiner 'a', which does not support the intent effectively.
- e1: e1_p8 score=70.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, and includes the 'When' cue, but lacks a direct connection to the auxiliary or additional context.
- e1: e1_p9 score=50.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry' and covers the 'passed' predicate, but is too short and lacks necessary context.
- e1: e1_p10 score=60.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry', covers the 'passed' predicate, but ends with an auxiliary 'was', which does not support the intent effectively.
- e1: e1_p11 score=40.0 valid=True terminal=date_passed
  Reason: The path starts from 'Distribution of Industry' and covers the 'passed' predicate, but ends with punctuation '?', which does not support the intent effectively.
- e1: e1_p12 score=30.0 valid=True terminal=date_passed
  Reason: The path is too short, only covering 'Distribution of Industry' and 'act', lacking any meaningful context or connection to the question.
- e1: e1_p13 score=30.0 valid=True terminal=date_passed
  Reason: The path is too short, only covering 'Distribution of Industry' and 'act', lacking any meaningful context or connection to the question.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p3

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p3'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- distribution_of_industry -> act (act of Distribution of Industry)
- act -> date_passed (date when the act was passed)
- act -> man (man who passed the act)
- man -> prime_minister (prime minister who was the man)
### ast_ps2 (ps2)
- distribution_of_industry -> act (act of Distribution of Industry)
- act -> man (passed by)
- man -> date_passed (date passed by the man)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively captures the necessary branches for the question, linking the act to the man who passed it and the date it was passed, allowing for straightforward decomposition into atomic subquestions.
- ast_ps2: score=0.92 valid=True reason=This AST also captures the necessary branches but uses 'passed by' which may introduce ambiguity in the decomposition process, making it slightly less preferable than ast_ps1.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- distribution_of_industry: Distribution of Industry (entity)
- act: act (type_variable)
- man: man (type_variable)
- prime_minister: prime minister (type_variable)
- date_passed: date_passed (value_slot)

Edges:
- distribution_of_industry -> act (act of Distribution of Industry)
- act -> date_passed (date when the act was passed)
- act -> man (man who passed the act)
- man -> prime_minister (prime minister who was the man)

## 11. Atomic Subquestion DAG
- None: What is the act of Distribution of Industry?
- None: When was the act of Distribution of Industry passed?
- None: Who is the man who passed the act of Distribution of Industry?
- None: Who is the prime minister of the man of the act of Distribution of Industry?
