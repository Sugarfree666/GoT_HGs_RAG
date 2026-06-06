# DEPO Decomposition #15

- Dataset: `2wikimultihopqa`
- Question: Where was the place of death of Maurice, Prince Of Orange's father?
- Gold answer: Delft

## 1. Semantic-Normalized Question
Where was the place of death of Maurice, Prince Of Orange's father?

## 2. Mask Spans
- Maurice, Prince Of Orange (entity, Person)

## 3. Selective Masked Question
Where was the place of death of PersonA's father?

## 4. CoreNLP Dependency Parse
- was[2] --advmod--> Where[1]
- place[4] --det--> the[3]
- was[2] --nsubj--> place[4]
- death[6] --case--> of[5]
- place[4] --nmod:of--> death[6]
- father[10] --case--> of[7]
- father[10] --nmod:poss--> PersonA[8]
- PersonA[8] --case--> 's[9]
- death[6] --nmod:of--> father[10]
- was[2] --punct--> ?[11]

## 5. Undirected Dependency Graph
- Where[1] --advmod-- was[2]
- was[2] --nsubj-- place[4]
- was[2] --punct-- ?[11]
- the[3] --det-- place[4]
- place[4] --nmod:of-- death[6]
- of[5] --case-- death[6]
- death[6] --nmod:of-- father[10]
- of[7] --case-- father[10]
- Maurice, Prince Of Orange[8] --nmod:poss-- father[10]
- Maurice, Prince Of Orange[8] --case-- 's[9]

## 6. Entity Start Nodes
- e1: Maurice, Prince Of Orange graph_node_ids=['8']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Maurice, Prince Of Orange -- father -- death -- place -- was -- Where
- e1_p2 (e1): Maurice, Prince Of Orange -- father -- death -- place
- e1_p3 (e1): Maurice, Prince Of Orange -- father -- death -- place -- was
- e1_p4 (e1): Maurice, Prince Of Orange -- father -- death -- place -- the
- e1_p5 (e1): Maurice, Prince Of Orange -- father -- death -- place -- was -- ?
- e1_p6 (e1): Maurice, Prince Of Orange -- father -- death
- e1_p7 (e1): Maurice, Prince Of Orange -- father -- death -- of
- e1_p8 (e1): Maurice, Prince Of Orange -- 's
- e1_p9 (e1): Maurice, Prince Of Orange -- father
- e1_p10 (e1): Maurice, Prince Of Orange -- father -- of

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=place_of_death
  Reason: The path starts from Maurice, Prince Of Orange, covers the necessary roles leading to the place of death, and includes the where cue.
- e1: e1_p2 score=85.0 valid=True terminal=place_of_death
  Reason: The path effectively connects Maurice, Prince Of Orange to the place of death but lacks the explicit where cue.
- e1: e1_p3 score=90.0 valid=True terminal=place_of_death
  Reason: The path starts from Maurice, Prince Of Orange, and effectively leads to the place of death while including the where cue.
- e1: e1_p4 score=70.0 valid=True terminal=place_of_death
  Reason: The path connects Maurice, Prince Of Orange to the place of death but lacks the where cue and ends with a determiner.
- e1: e1_p5 score=65.0 valid=True terminal=place_of_death
  Reason: The path includes the where cue but ends with punctuation and does not fully connect to the place of death.
- e1: e1_p6 score=60.0 valid=True terminal=place_of_death
  Reason: The path connects Maurice, Prince Of Orange to father and death but does not reach the place of death or include the where cue.
- e1: e1_p7 score=55.0 valid=True terminal=place_of_death
  Reason: The path connects Maurice, Prince Of Orange to father and death but ends with a preposition and does not reach the place of death.
- e1: e1_p8 score=20.0 valid=False terminal=place_of_death
  Reason: The path is too short and does not connect to any relevant concepts.
- e1: e1_p9 score=30.0 valid=False terminal=place_of_death
  Reason: The path connects Maurice, Prince Of Orange to father but does not reach the place of death or include the where cue.
- e1: e1_p10 score=50.0 valid=False terminal=place_of_death
  Reason: The path connects Maurice, Prince Of Orange to father but does not reach the place of death or include the where cue.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p3

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p3'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- maurice_prince_of_orange -> father (father of Maurice, Prince Of Orange)
- father -> death_place (place of death of the father)
### ast_ps2 (ps2)
- maurice_prince_of_orange -> father (father of Maurice, Prince Of Orange)
- father -> death (death of father)
- death -> place_of_death (place of death)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively covers the necessary roles leading to the place of death of Maurice, Prince Of Orange's father, and includes the required 'where' cue.
- ast_ps2: score=0.95 valid=True reason=This AST also covers the necessary roles and includes the 'where' cue, but it introduces an additional node for 'death' which may complicate the decomposition slightly.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- maurice_prince_of_orange: Maurice, Prince Of Orange (entity)
- father: father (type_variable)
- death_place: death_place (value_slot)

Edges:
- maurice_prince_of_orange -> father (father of Maurice, Prince Of Orange)
- father -> death_place (place of death of the father)

## 11. Atomic Subquestion DAG
- None: Who is the father of Maurice, Prince Of Orange?
- None: Where did the father of Maurice, Prince Of Orange die?
