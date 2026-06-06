# DEPO Decomposition #1

- Dataset: `2wikimultihopqa`
- Question: When did Lothair Ii's mother die?
- Gold answer: 20 March 851

## 1. Semantic-Normalized Question
When did Lothair II's mother die?

## 2. Mask Spans
- Lothair II's (entity, LothairIIS)

## 3. Selective Masked Question
When did SomeEntityA mother die?

## 4. CoreNLP Dependency Parse
- die[5] --advmod--> When[1]
- die[5] --aux--> did[2]
- mother[4] --compound--> SomeEntityA[3]
- die[5] --nsubj--> mother[4]
- die[5] --punct--> ?[6]

## 5. Undirected Dependency Graph
- When[1] --advmod-- die[5]
- did[2] --aux-- die[5]
- Lothair II's[3] --compound-- mother[4]
- mother[4] --nsubj-- die[5]
- die[5] --punct-- ?[6]

## 6. Entity Start Nodes
- e1: Lothair II's graph_node_ids=['3']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Lothair II's -- mother -- die -- When
- e1_p2 (e1): Lothair II's -- mother -- die
- e1_p3 (e1): Lothair II's -- mother -- die -- did
- e1_p4 (e1): Lothair II's -- mother -- die -- ?
- e1_p5 (e1): Lothair II's -- mother

## 8. LLM Path Scores
- e1: e1_p1 score=95.0 valid=True terminal=death_time
  Reason: The path starts from Lothair II's, reaches mother, covers the die predicate, and includes the when cue.
- e1: e1_p2 score=85.0 valid=True terminal=death_time
  Reason: The path starts from Lothair II's, reaches mother, and covers the die predicate but misses the when cue.
- e1: e1_p3 score=75.0 valid=True terminal=death_time
  Reason: The path starts from Lothair II's, reaches mother, covers the die predicate, but includes an auxiliary and misses the when cue.
- e1: e1_p4 score=30.0 valid=False terminal=death_time
  Reason: The path starts from Lothair II's, reaches mother, covers the die predicate, but ends with punctuation, making it invalid.
- e1: e1_p5 score=50.0 valid=False terminal=death_time
  Reason: The path starts from Lothair II's and reaches mother but is too short and lacks the die predicate and the when cue.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=95.0
- ps2: {'e1': 'e1_p2'} mean_path_score=85.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- lothair_ii -> mother (mother of Lothair II)
- mother -> death_date (date of death of the mother)
### ast_ps2 (ps2)
- lothair_ii -> mother (mother of Lothair II)
- mother -> death_date (date of death of the mother)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively captures the necessary branches for the question, covering the death of Lothair II's mother and includes the required temporal cue.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- lothair_ii: Lothair II (entity)
- mother: mother (type_variable)
- death_date: death_date (value_slot)

Edges:
- lothair_ii -> mother (mother of Lothair II)
- mother -> death_date (date of death of the mother)

## 11. Atomic Subquestion DAG
- None: Who is the mother of Lothair II?
- None: When did the mother of Lothair II die?
