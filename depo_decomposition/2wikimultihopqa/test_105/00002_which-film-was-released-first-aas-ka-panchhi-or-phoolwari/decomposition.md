# DEPO Decomposition #2

- Dataset: `2wikimultihopqa`
- Question: Which film was released first, Aas Ka Panchhi or Phoolwari?
- Gold answer: Phoolwari

## 1. Semantic-Normalized Question
Which film was released first, Aas Ka Panchhi or Phoolwari?

## 2. Mask Spans
- was released first, Aas Ka Panchhi or Phoolwari? (entity, Film)

## 3. Selective Masked Question
Which film MovieA

## 4. CoreNLP Dependency Parse
- MovieA[3] --det--> Which[1]
- MovieA[3] --compound--> film[2]

## 5. Undirected Dependency Graph
- Which[1] --det-- was released first, Aas Ka Panchhi or Phoolwari?[3]
- film[2] --compound-- was released first, Aas Ka Panchhi or Phoolwari?[3]

## 6. Entity Start Nodes
- e1: was released first, Aas Ka Panchhi or Phoolwari? graph_node_ids=['3']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): was released first, Aas Ka Panchhi or Phoolwari? -- film
- e1_p2 (e1): was released first, Aas Ka Panchhi or Phoolwari? -- Which

## 8. LLM Path Scores
- e1: e1_p1 score=70.0 valid=True terminal=film_release
  Reason: The path connects the entity to 'film' but lacks coverage of the 'first' release aspect, which is crucial for the question intent.
- e1: e1_p2 score=50.0 valid=True terminal=film_release
  Reason: The path connects to 'Which' but does not adequately address the film or the 'first' release aspect, making it less relevant.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=70.0
- ps2: {'e1': 'e1_p2'} mean_path_score=50.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- aas_ka_panchhi -> release_date (release date of Aas Ka Panchhi)
- phoolwari -> release_date (release date of Phoolwari)
### ast_ps2 (ps2)
- aas_ka_panchhi -> release_date (release date of Aas Ka Panchhi)
- phoolwari -> release_date (release date of Phoolwari)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively connects both films to their release dates, allowing for the decomposition into atomic questions about each film's release date without generating a final comparison question.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- aas_ka_panchhi: Aas Ka Panchhi (entity)
- phoolwari: Phoolwari (entity)
- release_date: release_date (value_slot)

Edges:
- aas_ka_panchhi -> release_date (release date of Aas Ka Panchhi)
- phoolwari -> release_date (release date of Phoolwari)

## 11. Atomic Subquestion DAG
- None: What is the release date of Aas Ka Panchhi?
- None: What is the release date of Phoolwari?
