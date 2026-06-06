# DEPO Decomposition #8

- Dataset: `2wikimultihopqa`
- Question: Who is Raghnall Mac Ruaidhrí's paternal grandfather?
- Gold answer: Ailéan mac Ruaidhrí

## 1. Semantic-Normalized Question
Who is the paternal grandfather of Raghnall Mac Ruaidhrí?

## 2. Mask Spans
- Raghnall Mac Ruaidhr (entity, Person)

## 3. Selective Masked Question
Who is the paternal grandfather of PersonAí?

## 4. CoreNLP Dependency Parse
- Who[1] --cop--> is[2]
- grandfather[5] --det--> the[3]
- grandfather[5] --amod--> paternal[4]
- Who[1] --nsubj--> grandfather[5]
- PersonAí[7] --case--> of[6]
- grandfather[5] --nmod:of--> PersonAí[7]
- Who[1] --punct--> ?[8]

## 5. Undirected Dependency Graph
- Who[1] --cop-- is[2]
- Who[1] --nsubj-- grandfather[5]
- Who[1] --punct-- ?[8]
- the[3] --det-- grandfather[5]
- paternal[4] --amod-- grandfather[5]
- grandfather[5] --nmod:of-- PersonAí[7]
- of[6] --case-- PersonAí[7]

## 6. Entity Start Nodes
- e1: PersonAí graph_node_ids=['7']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): PersonAí -- grandfather -- paternal
- e1_p2 (e1): PersonAí -- grandfather -- Who
- e1_p3 (e1): PersonAí -- grandfather -- Who -- is
- e1_p4 (e1): PersonAí -- grandfather -- Who -- ?
- e1_p5 (e1): PersonAí -- grandfather
- e1_p6 (e1): PersonAí -- grandfather -- the
- e1_p7 (e1): PersonAí -- of

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from PersonAí, reaches grandfather, and includes the paternal modifier, effectively covering the answer intent.
- e1: e1_p2 score=75.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from PersonAí and reaches grandfather, but the inclusion of 'Who' does not contribute to the semantic chain execution.
- e1: e1_p3 score=85.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from PersonAí, reaches grandfather, and includes the 'is' predicate, which supports the answer intent.
- e1: e1_p4 score=70.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from PersonAí and reaches grandfather, but ends with a punctuation mark, which limits its executability.
- e1: e1_p5 score=60.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from PersonAí and reaches grandfather, but it lacks additional context or predicates to support a complete semantic chain.
- e1: e1_p6 score=50.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from PersonAí and reaches grandfather, but the inclusion of 'the' does not contribute to the semantic chain.
- e1: e1_p7 score=30.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from PersonAí and only includes 'of', which does not provide any useful information for the semantic chain.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p3

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p3'} mean_path_score=85.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- raghnall_mac_ruaidhri -> grandfather (grandfather of Raghnall Mac Ruaidhrí)
- grandfather -> paternal_grandfather (paternal grandfather of the grandfather)
### ast_ps2 (ps2)
- raghnall_mac_ruaidhri -> grandfather (grandfather of Raghnall Mac Ruaidhrí)
- grandfather -> paternal_grandfather (paternal grandfather of the grandfather)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively covers the original question by detailing the relationship from Raghnall Mac Ruaidhrí to his paternal grandfather, allowing for straightforward decomposition into atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- raghnall_mac_ruaidhri: Raghnall Mac Ruaidhrí (entity)
- grandfather: grandfather (type_variable)
- paternal_grandfather: paternal_grandfather (value_slot)

Edges:
- raghnall_mac_ruaidhri -> grandfather (grandfather of Raghnall Mac Ruaidhrí)
- grandfather -> paternal_grandfather (paternal grandfather of the grandfather)

## 11. Atomic Subquestion DAG
- None: Who is the grandfather of Raghnall Mac Ruaidhrí?
- None: Who is the paternal grandfather of the grandfather of Raghnall Mac Ruaidhrí?
