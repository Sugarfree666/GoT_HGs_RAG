# DEPO Decomposition #7

- Dataset: `hotpotqa`
- Question: Which Walt Disney film was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?
- Gold answer: The Apple Dumpling Gang

## 1. Semantic-Normalized Question
Which Walt Disney film was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?

## 2. Mask Spans
- Walt Disney (entity, WaltDisney)
- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? (entity, Film)

## 3. Selective Masked Question
Which SomeEntityA film MovieA

## 4. CoreNLP Dependency Parse
- MovieA[4] --det--> Which[1]
- MovieA[4] --compound--> SomeEntityA[2]
- MovieA[4] --compound--> film[3]

## 5. Undirected Dependency Graph
- Which[1] --det-- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?[4]
- Walt Disney[2] --compound-- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?[4]
- film[3] --compound-- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?[4]

## 6. Entity Start Nodes
- e1: Walt Disney graph_node_ids=['2']
- e2: was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Walt Disney -- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes?
- e1_p2 (e1): Walt Disney -- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? -- film
- e1_p3 (e1): Walt Disney -- was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? -- Which
- e2_p1 (e2): was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? -- film
- e2_p2 (e2): was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? -- Which
- e2_p3 (e2): was produced first, The Apple Dumpling Gang or Something Wicked This Way Comes? -- Walt Disney

## 8. LLM Path Scores
- e1: e1_p1 score=70.0 valid=True terminal=film
  Reason: The path starts from Walt Disney and reaches the film question, but it lacks coverage of the production aspect.
- e1: e1_p2 score=85.0 valid=True terminal=film
  Reason: The path starts from Walt Disney, covers the film aspect, and includes the production cue, making it strong.
- e1: e1_p3 score=75.0 valid=True terminal=film
  Reason: The path starts from Walt Disney and includes the 'which' cue, but it does not fully cover the production aspect.
- e2: e2_p1 score=80.0 valid=True terminal=film
  Reason: The path starts from the production question and covers the film aspect well, supporting the answer intent.
- e2: e2_p2 score=60.0 valid=True terminal=film
  Reason: The path starts from the production question and includes 'which', but lacks coverage of the production aspect.
- e2: e2_p3 score=70.0 valid=True terminal=film
  Reason: The path starts from the production question and reaches Walt Disney, but it does not fully cover the film aspect.

## 8.1 Top-2 Paths per Entity
- e1: e1_p2, e1_p3
- e2: e2_p1, e2_p3

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p2', 'e2': 'e2_p1'} mean_path_score=82.5
- ps2: {'e1': 'e1_p2', 'e2': 'e2_p3'} mean_path_score=77.5
- ps3: {'e1': 'e1_p3', 'e2': 'e2_p1'} mean_path_score=77.5
- ps4: {'e1': 'e1_p3', 'e2': 'e2_p3'} mean_path_score=72.5

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- walt_disney -> film_1 (film produced by Walt Disney)
- walt_disney -> film_2 (film produced by Walt Disney)
- film_1 -> release_date (release date of The Apple Dumpling Gang)
- film_2 -> release_date (release date of Something Wicked This Way Comes)
### ast_ps2 (ps2)
- walt_disney -> film_1 (film produced by Walt Disney)
- walt_disney -> film_2 (film produced by Walt Disney)
- film_1 -> release_date (release date of The Apple Dumpling Gang)
- film_2 -> release_date (release date of Something Wicked This Way Comes)
### ast_ps3 (ps3)
- walt_disney -> film_a (produced film)
- walt_disney -> film_b (produced film)
- film_a -> film_e1 (film)
- film_a -> film_e2 (film)
- film_b -> film_e1 (film)
### ast_ps4 (ps4)
- walt_disney -> film_a (produced)
- walt_disney -> film_b (produced)
- film_a -> release_date (release_date of The Apple Dumpling Gang)
- film_b -> release_date (release_date of Something Wicked This Way Comes)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively covers the required facts about the films produced by Walt Disney and their release dates, allowing for straightforward decomposition into atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- walt_disney: Walt Disney (entity)
- film_1: The Apple Dumpling Gang (type_variable)
- film_2: Something Wicked This Way Comes (type_variable)
- release_date: release_date (value_slot)

Edges:
- walt_disney -> film_1 (film produced by Walt Disney)
- walt_disney -> film_2 (film produced by Walt Disney)
- film_1 -> release_date (release date of The Apple Dumpling Gang)
- film_2 -> release_date (release date of Something Wicked This Way Comes)

## 11. Atomic Subquestion DAG
- None: What is the film The Apple Dumpling Gang produced by Walt Disney?
- None: What is the release date of The Apple Dumpling Gang?
- None: What is the film Something Wicked This Way Comes produced by Walt Disney?
- None: What is the release date of Something Wicked This Way Comes?
