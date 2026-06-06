# DEPO Decomposition #1

- Dataset: `2wikimultihopqa`
- Question: When did Lothair Ii's mother die?
- Gold answer: 20 March 851

## 1. Semantic-Normalized Question
When did Lothair II's mother die?

## 2. Explicit Entities
- Lothair II (Person) span=(9, 19)

## 3. Entity Masking
- PersonA -> Lothair II

When did PersonA's mother die?

## 4. CoreNLP Dependency Parse
- die[6] --advmod--> When[1]
- die[6] --aux--> did[2]
- mother[5] --nmod:poss--> PersonA[3]
- PersonA[3] --case--> 's[4]
- die[6] --nsubj--> mother[5]
- die[6] --punct--> ?[7]

## 5. Undirected Dependency Graph
- When[1] --advmod-- die[6]
- did[2] --aux-- die[6]
- Lothair II[3] --nmod:poss-- mother[5]
- Lothair II[3] --case-- 's[4]
- mother[5] --nsubj-- die[6]
- die[6] --punct-- ?[7]

## 6. Entity Start Nodes from Explicit Entities
- e1: Lothair II graph_node_ids=['3']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Lothair II -- mother -- die -- When
- e1_p2 (e1): Lothair II -- mother -- die
- e1_p3 (e1): Lothair II -- mother -- die -- did
- e1_p4 (e1): Lothair II -- mother -- die -- ?
- e1_p5 (e1): Lothair II -- mother
- e1_p6 (e1): Lothair II -- 's

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=death_time
  Reason: The path starts from Lothair II, reaches mother, covers the die predicate, and includes the when cue.
- e1: e1_p2 score=75.0 valid=True terminal=death_time
  Reason: The path starts from Lothair II, reaches mother, and covers the die predicate but misses the when cue.
- e1: e1_p3 score=70.0 valid=True terminal=death_time
  Reason: The path starts from Lothair II, reaches mother, covers the die predicate, but includes did instead of when.
- e1: e1_p4 score=30.0 valid=True terminal=death_time
  Reason: The path starts from Lothair II, reaches mother, covers the die predicate, but ends with a question mark, missing the when cue.
- e1: e1_p5 score=20.0 valid=True terminal=death_time
  Reason: The path starts from Lothair II and reaches mother but does not cover the die predicate or the when cue.
- e1: e1_p6 score=0.0 valid=False terminal=death_time
  Reason: The path starts from Lothair II but ends with a possessive marker, failing to cover any relevant cues.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p2'} mean_path_score=75.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- lothair_ii -> mother (mother of Lothair II)
- mother -> death_date (date of death of the mother)
### ast_ps2 (ps2)
- lothair_ii -> mother (mother of Lothair II)
- mother -> death_date (date of death of the mother)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively captures the necessary branches for the question, covering the relationship between Lothair II and his mother, and includes the required temporal aspect of the mother's death.
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
