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

## 7.5 Terminal Glue Path Pruning
Total raw paths: 13
Total kept paths: 5
Total pruned paths: 8
Total pruned ratio: 61.54%

### By Entity
- e1 / Madame La Presidente
  - raw: 13
  - kept: 5
  - pruned: 8
  - fallback_used: False
  - examples:
    - e1_p2: Madame La Presidente -> director -> death -> date -> What -> is [terminal=is, reason=terminal_glue_token]
    - e1_p3: Madame La Presidente -> director -> death -> date -> What -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p5: Madame La Presidente -> director -> death -> date -> the [terminal=the, reason=terminal_glue_token]
    - e1_p7: Madame La Presidente -> director -> death -> of [terminal=of, reason=terminal_glue_token]
    - e1_p9: Madame La Presidente -> director -> of [terminal=of, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=95.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente, reaches director, covers the death predicate, and includes the date cue.
- e1: e1_p4 score=90.0 valid=True terminal=date_of_death
  Reason: The path starts from Madame La Presidente, reaches director, covers the death predicate, and includes the date cue.
- e1: e1_p6 score=75.0 valid=True terminal=death
  Reason: The path starts from Madame La Presidente and reaches director and death, but it does not include the date cue.
- e1: e1_p8 score=30.0 valid=False terminal=none
  Reason: The path stops too early at director and does not cover necessary roles or answer targets.
- e1: e1_p11 score=30.0 valid=False terminal=none
  Reason: The path only connects Madame La Presidente to film, missing critical roles and answer targets.

## 8.1 Highest-Scored Path per Entity
- e1: e1_p1 score=95.0

## 8.2 Selected Path Set
- ps1: {'e1': 'e1_p1'} mean_path_score=95.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: What is the date of death of the director of film Madame La Presidente?
- ps1
  - e1_p1: Madame La Presidente -> director -> death -> date -> What

Output:
- reason: The DAG compiles each semantic reasoning edge into one atomic lookup.
- q1: Who is the director of Madame La Presidente? depends_on=[] support=[]
- q2: When did the director of Madame La Presidente die? depends_on=['q1'] support=[]

## 10. Atomic Subquestion DAG
- None: Who is the director of Madame La Presidente?
- None: When did the director of Madame La Presidente die?
