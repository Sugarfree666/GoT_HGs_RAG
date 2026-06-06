# DEPO Decomposition #5

- Dataset: `hotpotqa`
- Question: In which county is the town in which Raymond Robertsen was born ?
- Gold answer: Finnmark county,

## 1. Semantic-Normalized Question
In which county is the town in which Raymond Robertsen was born?

## 2. Mask Spans
- Raymond Robertsen (entity, Person)

## 3. Selective Masked Question
In which county is the town in which PersonA was born?

## 4. CoreNLP Dependency Parse
- which[2] --case--> In[1]
- town[6] --obl:in--> which[2]
- town[6] --nsubj--> county[3]
- town[6] --cop--> is[4]
- town[6] --det--> the[5]
- born[11] --obl:in--> town[6]
- which[8] --case--> in[7]
- town[6] --ref--> which[8]
- born[11] --nsubj:pass--> PersonA[9]
- born[11] --aux:pass--> was[10]
- town[6] --acl:relcl--> born[11]
- town[6] --punct--> ?[12]

## 5. Undirected Dependency Graph
- In[1] --case-- which[2]
- which[2] --obl:in-- town[6]
- county[3] --nsubj-- town[6]
- is[4] --cop-- town[6]
- the[5] --det-- town[6]
- town[6] --obl:in/acl:relcl-- born[11]
- town[6] --ref-- which[8]
- town[6] --punct-- ?[12]
- in[7] --case-- which[8]
- Raymond Robertsen[9] --nsubj:pass-- born[11]
- was[10] --aux:pass-- born[11]

## 6. Entity Start Nodes
- e1: Raymond Robertsen graph_node_ids=['9']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Raymond Robertsen -- born -- town -- county
- e1_p2 (e1): Raymond Robertsen -- born -- town -- which
- e1_p3 (e1): Raymond Robertsen -- born -- town -- which
- e1_p4 (e1): Raymond Robertsen -- born -- town -- which -- In
- e1_p5 (e1): Raymond Robertsen -- born -- town -- which -- in
- e1_p6 (e1): Raymond Robertsen -- born -- town
- e1_p7 (e1): Raymond Robertsen -- born -- town -- is
- e1_p8 (e1): Raymond Robertsen -- born -- town -- the
- e1_p9 (e1): Raymond Robertsen -- born -- town -- ?
- e1_p10 (e1): Raymond Robertsen -- born
- e1_p11 (e1): Raymond Robertsen -- born -- was

## 8. LLM Path Scores
- e1: e1_p1 score=95.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born, covers the town and county, and includes the which cue.
- e1: e1_p2 score=85.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born, covers the town, and includes the which cue, but does not reach county.
- e1: e1_p3 score=85.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born, covers the town, and includes the which cue, but does not reach county.
- e1: e1_p4 score=75.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born, covers the town, and includes the which cue, but ends with a preposition.
- e1: e1_p5 score=75.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born, covers the town, and includes the which cue, but ends with a preposition.
- e1: e1_p6 score=70.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen and reaches born and town, but does not include the which cue or reach county.
- e1: e1_p7 score=60.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen and reaches born and town, but ends with an auxiliary and does not include the which cue or reach county.
- e1: e1_p8 score=50.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen and reaches born and town, but ends with a determiner and does not include the which cue or reach county.
- e1: e1_p9 score=30.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen and reaches born and town, but ends with punctuation and does not include the which cue or reach county.
- e1: e1_p10 score=40.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen and reaches born, but does not cover the town or include the which cue.
- e1: e1_p11 score=50.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen and reaches born, but does not cover the town or include the which cue.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=95.0
- ps2: {'e1': 'e1_p2'} mean_path_score=85.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- raymond_robertsen -> town (town where Raymond Robertsen was born)
- town -> county (county of the town)
### ast_ps2 (ps2)
- raymond_robertsen -> town (town where Raymond Robertsen was born)
- town -> county (county of the town)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively captures the necessary branches for the original question, covering both the town and county aspects, and allows for the generation of executable atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- raymond_robertsen: Raymond Robertsen (entity)
- town: town (type_variable)
- county: county (value_slot)

Edges:
- raymond_robertsen -> town (town where Raymond Robertsen was born)
- town -> county (county of the town)

## 11. Atomic Subquestion DAG
- None: What is the town where Raymond Robertsen was born?
- None: In which county is the town of Raymond Robertsen located?
