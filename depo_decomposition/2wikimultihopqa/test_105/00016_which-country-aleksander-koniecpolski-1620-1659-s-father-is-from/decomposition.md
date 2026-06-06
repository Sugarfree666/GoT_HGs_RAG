# DEPO Decomposition #16

- Dataset: `2wikimultihopqa`
- Question: Which country Aleksander Koniecpolski (1620–1659)'s father is from?
- Gold answer: Polish-Lithuanian Commonwealth

## 1. Semantic-Normalized Question
Which country is Aleksander Koniecpolski's (1620–1659) father from?

## 2. Mask Spans
- Aleksander Koniecpolski's (1620–1659) (entity, Country)

## 3. Selective Masked Question
Which country is CountryA father from?

## 4. CoreNLP Dependency Parse
- country[2] --det--> Which[1]
- father[5] --nsubj--> country[2]
- father[5] --cop--> is[3]
- father[5] --compound--> CountryA[4]
- father[5] --dep--> from[6]
- father[5] --punct--> ?[7]

## 5. Undirected Dependency Graph
- Which[1] --det-- country[2]
- country[2] --nsubj-- father[5]
- is[3] --cop-- father[5]
- Aleksander Koniecpolski's (1620–1659)[4] --compound-- father[5]
- father[5] --dep-- from[6]
- father[5] --punct-- ?[7]

## 6. Entity Start Nodes
- e1: Aleksander Koniecpolski's (1620–1659) graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Aleksander Koniecpolski's (1620–1659) -- father -- country -- Which
- e1_p2 (e1): Aleksander Koniecpolski's (1620–1659) -- father -- country
- e1_p3 (e1): Aleksander Koniecpolski's (1620–1659) -- father
- e1_p4 (e1): Aleksander Koniecpolski's (1620–1659) -- father -- is
- e1_p5 (e1): Aleksander Koniecpolski's (1620–1659) -- father -- from
- e1_p6 (e1): Aleksander Koniecpolski's (1620–1659) -- father -- ?

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=country
  Reason: The path starts from Aleksander Koniecpolski's father, covers the necessary roles, and includes the focus on the country.
- e1: e1_p2 score=85.0 valid=True terminal=country
  Reason: The path starts from Aleksander Koniecpolski's father and reaches the country, covering the necessary roles.
- e1: e1_p3 score=50.0 valid=True terminal=father
  Reason: The path only covers the entity and its father, missing the necessary connection to the country.
- e1: e1_p4 score=40.0 valid=True terminal=is
  Reason: The path includes 'is' but does not connect to the country, missing key elements of the question.
- e1: e1_p5 score=70.0 valid=True terminal=from
  Reason: The path connects 'father' to 'from', but does not reach the country, missing the final answer target.
- e1: e1_p6 score=20.0 valid=True terminal=?
  Reason: The path ends with a question mark, providing no useful information towards the answer.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p2'} mean_path_score=85.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- aleksander_koniecpolski -> father (father of Aleksander Koniecpolski)
- father -> country (country of the father)
### ast_ps2 (ps2)
- aleksander_koniecpolski -> father (father of Aleksander Koniecpolski)
- father -> country (country of the father)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively captures the relationship between Aleksander Koniecpolski and his father, leading to the country of origin, fulfilling all criteria for decomposition.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- aleksander_koniecpolski: Aleksander Koniecpolski (entity)
- father: father (type_variable)
- country: country (value_slot)

Edges:
- aleksander_koniecpolski -> father (father of Aleksander Koniecpolski)
- father -> country (country of the father)

## 11. Atomic Subquestion DAG
- None: What is the father of Aleksander Koniecpolski?
- None: What country is the father of Aleksander Koniecpolski from?
