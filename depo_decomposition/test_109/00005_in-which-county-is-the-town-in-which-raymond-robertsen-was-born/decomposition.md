# DEPO Decomposition #5

- Dataset: `hotpotqa`
- Question: In which county is the town in which Raymond Robertsen was born ?
- Gold answer: Finnmark county,

## 1. Semantic-Normalized Question
In which county is the town in which Raymond Robertsen was born?

## 2. Explicit Entities
- Raymond Robertsen (Person) span=(37, 54)

## 3. Entity Masking
- PersonA -> Raymond Robertsen

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

## 6. Entity Start Nodes from Explicit Entities
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
- e1: e1_p1 score=90.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born, covers the town and county, effectively supporting the question's intent.
- e1: e1_p2 score=85.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born, covers the town, and includes the wh cue 'which', but does not reach 'county'.
- e1: e1_p3 score=80.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born, covers the town, and includes the wh cue 'which', but does not reach 'county'.
- e1: e1_p4 score=75.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born, covers the town, includes the wh cue 'which', but ends with a preposition 'In' which does not contribute to the answer.
- e1: e1_p5 score=70.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born, covers the town, includes the wh cue 'which', but ends with 'in', which does not contribute to the answer.
- e1: e1_p6 score=60.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born and town, but does not include the necessary wh cue 'which' or reach 'county'.
- e1: e1_p7 score=50.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born and town, but ends with 'is', which does not contribute to the answer and lacks the necessary wh cue 'which'.
- e1: e1_p8 score=45.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born and town, but ends with 'the', which does not contribute to the answer and lacks the necessary wh cue 'which'.
- e1: e1_p9 score=30.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen, reaches born and town, but ends with '?', which does not contribute to the answer and lacks the necessary wh cue 'which'.
- e1: e1_p10 score=20.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen and reaches born, but is too short and does not cover the necessary cues or reach 'county'.
- e1: e1_p11 score=25.0 valid=True terminal=county
  Reason: The path starts from Raymond Robertsen and reaches born, but ends with 'was', which does not contribute to the answer and lacks the necessary wh cue 'which'.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p2'} mean_path_score=85.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: In which county is the town in which Raymond Robertsen was born ?
- ps1
  - e1_p1: Raymond Robertsen -> born -> town -> county
- ps2
  - e1_p2: Raymond Robertsen -> born -> town -> which

Output:
- selected_path_set_ids: ['ps1']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: In which town was Raymond Robertsen born? depends_on=[] support=['e1_p1']
- q2: In which county is q1's answer located? depends_on=['q1'] support=['e1_p1']

## 10. Atomic Subquestion DAG
- None: In which town was Raymond Robertsen born?
- None: In which county is q1's answer located?
