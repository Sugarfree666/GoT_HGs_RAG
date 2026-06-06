# DEPO Decomposition #15

- Dataset: `2wikimultihopqa`
- Question: Where was the place of death of Maurice, Prince Of Orange's father?
- Gold answer: Delft

## 1. Semantic-Normalized Question
Where was the place of death of the father of Maurice, Prince Of Orange?

## 2. Explicit Entities
- Maurice (Person) span=(46, 53)
- Prince Of Orange (Person) span=(55, 71)

## 3. Entity Masking
- PersonA -> Maurice
- PersonB -> Prince Of Orange

Where was the place of death of the father of PersonA, PersonB?

## 4. CoreNLP Dependency Parse
- was[2] --advmod--> Where[1]
- place[4] --det--> the[3]
- was[2] --nsubj--> place[4]
- death[6] --case--> of[5]
- place[4] --nmod:of--> death[6]
- father[9] --case--> of[7]
- father[9] --det--> the[8]
- death[6] --nmod:of--> father[9]
- PersonA[11] --case--> of[10]
- father[9] --nmod:of--> PersonA[11]
- father[9] --punct--> ,[12]
- father[9] --appos--> PersonB[13]
- was[2] --punct--> ?[14]

## 5. Undirected Dependency Graph
- Where[1] --advmod-- was[2]
- was[2] --nsubj-- place[4]
- was[2] --punct-- ?[14]
- the[3] --det-- place[4]
- place[4] --nmod:of-- death[6]
- of[5] --case-- death[6]
- death[6] --nmod:of-- father[9]
- of[7] --case-- father[9]
- the[8] --det-- father[9]
- father[9] --nmod:of-- Maurice[11]
- father[9] --punct-- ,[12]
- father[9] --appos-- Prince Of Orange[13]
- of[10] --case-- Maurice[11]

## 6. Entity Start Nodes from Explicit Entities
- e1: Maurice graph_node_ids=['11']
- e2: Prince Of Orange graph_node_ids=['13']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Maurice -- father -- death -- place -- was -- Where
- e1_p2 (e1): Maurice -- father -- death -- place
- e1_p3 (e1): Maurice -- father -- death -- place -- was
- e1_p4 (e1): Maurice -- father -- death -- place -- the
- e1_p5 (e1): Maurice -- father -- death -- place -- was -- ?
- e1_p6 (e1): Maurice -- father -- death
- e1_p7 (e1): Maurice -- father -- death -- of
- e1_p8 (e1): Maurice -- father
- e1_p9 (e1): Maurice -- father -- of
- e1_p10 (e1): Maurice -- father -- the
- e1_p11 (e1): Maurice -- father -- ,
- e1_p12 (e1): Maurice -- of
- e1_p13 (e1): Maurice -- father -- Prince Of Orange
- e2_p1 (e2): Prince Of Orange -- father -- death -- place -- was -- Where
- e2_p2 (e2): Prince Of Orange -- father -- death -- place
- e2_p3 (e2): Prince Of Orange -- father -- death -- place -- was
- e2_p4 (e2): Prince Of Orange -- father -- death -- place -- the
- e2_p5 (e2): Prince Of Orange -- father -- death -- place -- was -- ?
- e2_p6 (e2): Prince Of Orange -- father -- death
- e2_p7 (e2): Prince Of Orange -- father -- death -- of
- e2_p8 (e2): Prince Of Orange -- father
- e2_p9 (e2): Prince Of Orange -- father -- of
- e2_p10 (e2): Prince Of Orange -- father -- the
- e2_p11 (e2): Prince Of Orange -- father -- ,
- e2_p12 (e2): Prince Of Orange -- father -- Maurice
- e2_p13 (e2): Prince Of Orange -- father -- Maurice -- of

## 8. LLM Path Scores
- e1: e1_p1 score=95.0 valid=True terminal=place_of_death
  Reason: The path starts from Maurice, connects to father, death, and place, and includes the where cue.
- e1: e1_p2 score=85.0 valid=True terminal=place_of_death
  Reason: The path covers the necessary roles but misses the where cue.
- e1: e1_p3 score=90.0 valid=True terminal=place_of_death
  Reason: The path includes all necessary components and the where cue.
- e1: e1_p4 score=80.0 valid=True terminal=place_of_death
  Reason: The path covers the necessary roles but misses the where cue.
- e1: e1_p5 score=75.0 valid=True terminal=place_of_death
  Reason: The path includes all necessary components and the where cue, but ends with punctuation.
- e1: e1_p6 score=60.0 valid=True terminal=place_of_death
  Reason: The path is too short and misses key components.
- e1: e1_p7 score=55.0 valid=True terminal=place_of_death
  Reason: The path is too short and misses key components.
- e1: e1_p8 score=30.0 valid=False terminal=place_of_death
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p9 score=40.0 valid=False terminal=place_of_death
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p10 score=35.0 valid=False terminal=place_of_death
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p11 score=20.0 valid=False terminal=place_of_death
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p12 score=50.0 valid=False terminal=place_of_death
  Reason: The path is too short and does not provide any useful information.
- e1: e1_p13 score=0.0 valid=False
  Reason: missing from LLM output
- e2: e2_p1 score=95.0 valid=True terminal=place_of_death
  Reason: The path starts from Prince Of Orange, connects to father, death, and place, and includes the where cue.
- e2: e2_p2 score=85.0 valid=True terminal=place_of_death
  Reason: The path covers the necessary roles but misses the where cue.
- e2: e2_p3 score=90.0 valid=True terminal=place_of_death
  Reason: The path includes all necessary components and the where cue.
- e2: e2_p4 score=80.0 valid=True terminal=place_of_death
  Reason: The path covers the necessary roles but misses the where cue.
- e2: e2_p5 score=75.0 valid=True terminal=place_of_death
  Reason: The path includes all necessary components and the where cue, but ends with punctuation.
- e2: e2_p6 score=60.0 valid=True terminal=place_of_death
  Reason: The path is too short and misses key components.
- e2: e2_p7 score=55.0 valid=True terminal=place_of_death
  Reason: The path is too short and misses key components.
- e2: e2_p8 score=30.0 valid=False terminal=place_of_death
  Reason: The path is too short and does not provide any useful information.
- e2: e2_p9 score=40.0 valid=False terminal=place_of_death
  Reason: The path is too short and does not provide any useful information.
- e2: e2_p10 score=35.0 valid=False terminal=place_of_death
  Reason: The path is too short and does not provide any useful information.
- e2: e2_p11 score=20.0 valid=False terminal=place_of_death
  Reason: The path is too short and does not provide any useful information.
- e2: e2_p12 score=50.0 valid=False terminal=place_of_death
  Reason: The path is too short and does not provide any useful information.
- e2: e2_p13 score=70.0 valid=False terminal=place_of_death
  Reason: The path passes through another entity start and does not provide a direct answer.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p3
- e2: e2_p1, e2_p3

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1', 'e2': 'e2_p1'} mean_path_score=95.0
- ps2: {'e1': 'e1_p1', 'e2': 'e2_p3'} mean_path_score=92.5
- ps3: {'e1': 'e1_p3', 'e2': 'e2_p1'} mean_path_score=92.5
- ps4: {'e1': 'e1_p3', 'e2': 'e2_p3'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- maurice -> father (father of Maurice)
- father -> death_place (place of death of the father)
- prince_of_orange -> father_2 (father of Prince Of Orange)
- father_2 -> death_place_2 (place of death of the father)
### ast_ps2 (ps2)
- maurice -> father (father of Maurice)
- father -> death_place (place of death of the father)
- prince_of_orange -> father_2 (father of Prince Of Orange)
- father_2 -> death_place_2 (place of death of the father)
### ast_ps3 (ps3)
- maurice -> father (father of Maurice)
- father -> death_place (place of death of the father)
- prince_of_orange -> father_2 (father of Prince Of Orange)
- father_2 -> death_place_2 (place of death of the father)
### ast_ps4 (ps4)
- maurice -> father (father of Maurice)
- father -> death_place (place of death of the father)
- prince_of_orange -> father_2 (father of Prince Of Orange)
- father_2 -> death_place_2 (place of death of the father)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively covers the required branch facts of the original question, providing a clear path from Maurice to his father's place of death, and includes both entities with complete branches.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- maurice: Maurice (entity)
- father: father (type_variable)
- death_place: death_place (value_slot)
- prince_of_orange: Prince Of Orange (entity)
- father_2: father (type_variable)
- death_place_2: death_place (value_slot)

Edges:
- maurice -> father (father of Maurice)
- father -> death_place (place of death of the father)
- prince_of_orange -> father_2 (father of Prince Of Orange)
- father_2 -> death_place_2 (place of death of the father)

## 11. Atomic Subquestion DAG
- None: Who is the father of Maurice?
- None: Where was the father of Maurice's place of death?
- None: Who is the father of Prince Of Orange?
- None: Where did the father of Prince Of Orange die?
