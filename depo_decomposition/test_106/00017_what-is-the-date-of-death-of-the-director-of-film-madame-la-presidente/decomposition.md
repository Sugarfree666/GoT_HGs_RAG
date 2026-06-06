# DEPO Decomposition #17

- Dataset: `2wikimultihopqa`
- Question: What is the date of death of the director of film Madame La Presidente?
- Gold answer: 10 August 1960

## 1. Semantic-Normalized Question
What is the date of death of the director of the film Madame La Presidente?

## 2. Explicit Entities
- Madame La Presidente (Film) span=(54, 74)

## 3. Entity Masking
- FilmA -> Madame La Presidente

What is the date of death of the director of the film FilmA?

## 4. CoreNLP Dependency Parse
- What[1] --cop--> is[2]
- date[4] --det--> the[3]
- What[1] --nsubj--> date[4]
- death[6] --case--> of[5]
- date[4] --nmod:of--> death[6]
- director[9] --case--> of[7]
- director[9] --det--> the[8]
- death[6] --nmod:of--> director[9]
- FilmA[13] --case--> of[10]
- FilmA[13] --det--> the[11]
- FilmA[13] --compound--> film[12]
- director[9] --nmod:of--> FilmA[13]
- What[1] --punct--> ?[14]

## 5. Undirected Dependency Graph
- What[1] --cop-- is[2]
- What[1] --nsubj-- date[4]
- What[1] --punct-- ?[14]
- the[3] --det-- date[4]
- date[4] --nmod:of-- death[6]
- of[5] --case-- death[6]
- death[6] --nmod:of-- director[9]
- of[7] --case-- director[9]
- the[8] --det-- director[9]
- director[9] --nmod:of-- Madame La Presidente[13]
- of[10] --case-- Madame La Presidente[13]
- the[11] --det-- Madame La Presidente[13]
- film[12] --compound-- Madame La Presidente[13]

## 6. Entity Start Nodes from Explicit Entities
- e1: Madame La Presidente graph_node_ids=['13']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Madame La Presidente -- director -- death -- date -- What
- e1_p2 (e1): Madame La Presidente -- director -- death -- date -- What -- is
- e1_p3 (e1): Madame La Presidente -- director -- death -- date -- What -- ?
- e1_p4 (e1): Madame La Presidente -- director -- death -- date
- e1_p5 (e1): Madame La Presidente -- director -- death -- date -- the
- e1_p6 (e1): Madame La Presidente -- director -- death
- e1_p7 (e1): Madame La Presidente -- director -- death -- of
- e1_p8 (e1): Madame La Presidente -- director
- e1_p9 (e1): Madame La Presidente -- director -- of
- e1_p10 (e1): Madame La Presidente -- director -- the
- e1_p11 (e1): Madame La Presidente -- film
- e1_p12 (e1): Madame La Presidente -- of
- e1_p13 (e1): Madame La Presidente -- the

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente, reaches director, covers the death predicate, and includes the date cue.
- e1: e1_p2 score=90.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente, reaches director, covers the death predicate, includes the date cue, and has the auxiliary 'is' for completeness.
- e1: e1_p3 score=85.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente, reaches director, covers the death predicate, includes the date cue, but ends with punctuation which slightly reduces its score.
- e1: e1_p4 score=95.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente, reaches director, covers the death predicate, and includes the date cue without any noise.
- e1: e1_p5 score=70.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente, reaches director, covers the death predicate, and includes the date cue, but ends with a determiner which reduces its executability.
- e1: e1_p6 score=75.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente, reaches director, and covers the death predicate, but it does not include the date cue.
- e1: e1_p7 score=60.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente, reaches director, and covers the death predicate, but ends with a preposition which reduces its executability.
- e1: e1_p8 score=40.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente and reaches director, but it does not cover any relevant predicates or cues.
- e1: e1_p9 score=50.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente and reaches director, but it ends with a preposition and does not cover any relevant predicates or cues.
- e1: e1_p10 score=40.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente and reaches director, but it ends with a determiner and does not cover any relevant predicates or cues.
- e1: e1_p11 score=30.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente and reaches film, but it does not cover any relevant predicates or cues.
- e1: e1_p12 score=20.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente and reaches 'of', but it does not cover any relevant predicates or cues.
- e1: e1_p13 score=20.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente and reaches 'the', but it does not cover any relevant predicates or cues.

## 8.1 Top-2 Paths per Entity
- e1: e1_p4, e1_p1

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p4'} mean_path_score=95.0
- ps2: {'e1': 'e1_p1'} mean_path_score=90.0

## 9. Candidate Path-Set Semantic ASTs
### ast_ps1 (ps1)
- madame_la_presidente -> director (director of Madame La Presidente)
- director -> death_date (date of death of the director)
### ast_ps2 (ps2)
- madame_la_presidente -> director (director of Madame La Presidente)
- director -> death_date (date of death of the director)

## 10. LLM Best AST Selection
- ast_ps1: score=0.96 valid=True reason=This AST effectively captures the relationship between the film 'Madame La Presidente', its director, and the director's date of death, allowing for straightforward decomposition into atomic subquestions.
- best_candidate_id: ast_ps1
- selected_candidate_id: ast_ps1

## 10. Selected Semantic AST
Nodes:
- madame_la_presidente: Madame La Presidente (entity)
- director: director (type_variable)
- death_date: death_date (value_slot)

Edges:
- madame_la_presidente -> director (director of Madame La Presidente)
- director -> death_date (date of death of the director)

## 11. Atomic Subquestion DAG
- None: Who is the director of Madame La Presidente?
- None: When did the director of Madame La Presidente die?
