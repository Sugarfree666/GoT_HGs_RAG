# DEPO Decomposition #7

- Dataset: `2wikimultihopqa`
- Question: Who was born first out of Aivar Kuusmaa and Andy Summers?
- Gold answer: Andy Summers

## 1. Semantic-Normalized Question
Who was born first out of Aivar Kuusmaa and Andy Summers?

## 2. Mask Spans
- Aivar Kuusmaa (entity, Person)
- Andy Summers (entity, Person)

## 3. Selective Masked Question
Who was born first out of PersonA and PersonB?

## 4. CoreNLP Dependency Parse
- born[3] --nsubj:pass--> Who[1]
- born[3] --aux:pass--> was[2]
- born[3] --advmod--> first[4]
- PersonA[7] --case--> out[5]
- out[5] --fixed--> of[6]
- born[3] --obl:out_of--> PersonA[7]
- PersonB[9] --cc--> and[8]
- born[3] --obl:out_of--> PersonB[9]
- PersonA[7] --conj:and--> PersonB[9]
- born[3] --punct--> ?[10]

## 5. Undirected Dependency Graph
- Who[1] --nsubj:pass-- born[3]
- was[2] --aux:pass-- born[3]
- born[3] --advmod-- first[4]
- born[3] --obl:out_of-- Aivar Kuusmaa[7]
- born[3] --obl:out_of-- Andy Summers[9]
- born[3] --punct-- ?[10]
- out[5] --case-- Aivar Kuusmaa[7]
- out[5] --fixed-- of[6]
- Aivar Kuusmaa[7] --conj:and-- Andy Summers[9]
- and[8] --cc-- Andy Summers[9]

## 6. Entity Start Nodes
- e1: Aivar Kuusmaa graph_node_ids=['7']
- e2: Andy Summers graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Aivar Kuusmaa -- born -- first
- e1_p2 (e1): Aivar Kuusmaa -- born -- Who
- e1_p3 (e1): Aivar Kuusmaa -- born
- e1_p4 (e1): Aivar Kuusmaa -- out
- e1_p5 (e1): Aivar Kuusmaa -- born -- was
- e1_p6 (e1): Aivar Kuusmaa -- born -- ?
- e1_p7 (e1): Aivar Kuusmaa -- out -- of
- e1_p8 (e1): Aivar Kuusmaa -- born -- Andy Summers
- e1_p9 (e1): Aivar Kuusmaa -- Andy Summers
- e1_p10 (e1): Aivar Kuusmaa -- Andy Summers -- born -- first
- e1_p11 (e1): Aivar Kuusmaa -- Andy Summers -- born -- Who
- e1_p12 (e1): Aivar Kuusmaa -- Andy Summers -- born
- e1_p13 (e1): Aivar Kuusmaa -- born -- Andy Summers -- and
- e1_p14 (e1): Aivar Kuusmaa -- Andy Summers -- born -- was
- e1_p15 (e1): Aivar Kuusmaa -- Andy Summers -- born -- ?
- e1_p16 (e1): Aivar Kuusmaa -- Andy Summers -- and
- e2_p1 (e2): Andy Summers -- born -- first
- e2_p2 (e2): Andy Summers -- born -- Who
- e2_p3 (e2): Andy Summers -- born
- e2_p4 (e2): Andy Summers -- born -- was
- e2_p5 (e2): Andy Summers -- born -- ?
- e2_p6 (e2): Andy Summers -- and
- e2_p7 (e2): Andy Summers -- born -- Aivar Kuusmaa
- e2_p8 (e2): Andy Summers -- Aivar Kuusmaa
- e2_p9 (e2): Andy Summers -- born -- Aivar Kuusmaa -- out
- e2_p10 (e2): Andy Summers -- Aivar Kuusmaa -- born -- first
- e2_p11 (e2): Andy Summers -- born -- Aivar Kuusmaa -- out -- of
- e2_p12 (e2): Andy Summers -- Aivar Kuusmaa -- born -- Who
- e2_p13 (e2): Andy Summers -- Aivar Kuusmaa -- born
- e2_p14 (e2): Andy Summers -- Aivar Kuusmaa -- out
- e2_p15 (e2): Andy Summers -- Aivar Kuusmaa -- born -- was
- e2_p16 (e2): Andy Summers -- Aivar Kuusmaa -- born -- ?
- e2_p17 (e2): Andy Summers -- Aivar Kuusmaa -- out -- of

## 8. LLM Path Scores
- e1: e1_p1 score=85.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, and includes the first cue, but lacks a direct connection to the other entity.
- e1: e1_p2 score=80.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, and includes the who cue, but does not connect to the other entity.
- e1: e1_p3 score=70.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa and reaches the born predicate, but it lacks coverage for the first cue.
- e1: e1_p4 score=30.0 valid=False terminal=birth_order
  Reason: The path stops at 'out', missing the necessary predicates and cues for the question.
- e1: e1_p5 score=85.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, and includes the was cue, but lacks a direct connection to the other entity.
- e1: e1_p6 score=30.0 valid=False terminal=birth_order
  Reason: The path stops at '?', missing the necessary predicates and cues for the question.
- e1: e1_p7 score=30.0 valid=False terminal=birth_order
  Reason: The path stops at 'of', missing the necessary predicates and cues for the question.
- e1: e1_p8 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, and includes Andy Summers, but lacks the first cue.
- e1: e1_p9 score=30.0 valid=False terminal=birth_order
  Reason: The path stops at 'Andy Summers', missing the necessary predicates and cues for the question.
- e1: e1_p10 score=95.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, includes Andy Summers, and covers the first cue.
- e1: e1_p11 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, includes Andy Summers, and covers the who cue.
- e1: e1_p12 score=85.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, but lacks coverage for the first cue.
- e1: e1_p13 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, includes Andy Summers, but lacks the first cue.
- e1: e1_p14 score=95.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, includes Andy Summers, and covers the was cue.
- e1: e1_p15 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Aivar Kuusmaa, reaches the born predicate, includes Andy Summers, and covers the ? cue.
- e1: e1_p16 score=30.0 valid=False terminal=birth_order
  Reason: The path stops at 'and', missing the necessary predicates and cues for the question.
- e2: e2_p1 score=85.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, and includes the first cue, but lacks a direct connection to the other entity.
- e2: e2_p2 score=80.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, and includes the who cue, but does not connect to the other entity.
- e2: e2_p3 score=70.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers and reaches the born predicate, but it lacks coverage for the first cue.
- e2: e2_p4 score=85.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, and includes the was cue, but lacks a direct connection to the other entity.
- e2: e2_p5 score=30.0 valid=False terminal=birth_order
  Reason: The path stops at '?', missing the necessary predicates and cues for the question.
- e2: e2_p6 score=30.0 valid=False terminal=birth_order
  Reason: The path stops at 'and', missing the necessary predicates and cues for the question.
- e2: e2_p7 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, and includes Aivar Kuusmaa, but lacks the first cue.
- e2: e2_p8 score=30.0 valid=False terminal=birth_order
  Reason: The path stops at 'Aivar Kuusmaa', missing the necessary predicates and cues for the question.
- e2: e2_p9 score=95.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, includes Aivar Kuusmaa, and covers the first cue.
- e2: e2_p10 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, includes Aivar Kuusmaa, and covers the who cue.
- e2: e2_p11 score=95.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, includes Aivar Kuusmaa, and covers the first cue.
- e2: e2_p12 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, includes Aivar Kuusmaa, but lacks the first cue.
- e2: e2_p13 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, includes Aivar Kuusmaa, but lacks the first cue.
- e2: e2_p14 score=95.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, includes Aivar Kuusmaa, and covers the was cue.
- e2: e2_p15 score=90.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, includes Aivar Kuusmaa, and covers the ? cue.
- e2: e2_p16 score=30.0 valid=False terminal=birth_order
  Reason: The path stops at 'and', missing the necessary predicates and cues for the question.
- e2: e2_p17 score=95.0 valid=True terminal=birth_order
  Reason: The path starts from Andy Summers, reaches the born predicate, includes Aivar Kuusmaa, and covers the first cue.

## 8.1 Top-2 Paths per Entity
- e1: e1_p10, e1_p14
- e2: e2_p11, e2_p14

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p10', 'e2': 'e2_p11'} mean_path_score=95.0
- ps2: {'e1': 'e1_p10', 'e2': 'e2_p14'} mean_path_score=95.0
- ps3: {'e1': 'e1_p14', 'e2': 'e2_p11'} mean_path_score=95.0
- ps4: {'e1': 'e1_p14', 'e2': 'e2_p14'} mean_path_score=95.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- aivar_kuusmaa -> birth_date_aivar (date of birth of Aivar Kuusmaa)
- andy_summers -> birth_date_andy (date of birth of Andy Summers)
### ast_ps2 (ps2)
- aivar_kuusmaa -> birth_date_aivar (date of birth of Aivar Kuusmaa)
- andy_summers -> birth_date_andy (date of birth of Andy Summers)
### ast_ps3 (ps3)
- aivar_kuusmaa -> birth_date_aivar (date of birth of Aivar Kuusmaa)
- andy_summers -> birth_date_andy (date of birth of Andy Summers)
### ast_ps4 (ps4)
- aivar_kuusmaa -> birth_date_aivar (date of birth of Aivar Kuusmaa)
- andy_summers -> birth_date_andy (date of birth of Andy Summers)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively captures the birth dates of both Aivar Kuusmaa and Andy Summers, allowing for direct comparison without generating a final operator question.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- aivar_kuusmaa: Aivar Kuusmaa (entity)
- andy_summers: Andy Summers (entity)
- birth_date_aivar: birth_date (value_slot)
- birth_date_andy: birth_date (value_slot)

Edges:
- aivar_kuusmaa -> birth_date_aivar (date of birth of Aivar Kuusmaa)
- andy_summers -> birth_date_andy (date of birth of Andy Summers)

## 11. Atomic Subquestion DAG
- None: What is the birth date of Aivar Kuusmaa?
- None: What is the birth date of Andy Summers?
