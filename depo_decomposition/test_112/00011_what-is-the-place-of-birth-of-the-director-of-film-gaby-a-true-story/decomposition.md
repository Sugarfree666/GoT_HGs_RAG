# DEPO Decomposition #11

- Dataset: `2wikimultihopqa`
- Question: What is the place of birth of the director of film Gaby: A True Story?
- Gold answer: Mexico City

## 1. Semantic-Normalized Question
What is the place of birth of the director of the film Gaby: A True Story?

## 2. Explicit Entities
- Gaby: A True Story (Film) span=(55, 73)

## 3. Entity Masking
- FilmA -> Gaby: A True Story

What is the place of birth of the director of the film FilmA?

## 4. CoreNLP Dependency Parse
- What[1] --cop--> is[2]
- place[4] --det--> the[3]
- What[1] --nsubj--> place[4]
- birth[6] --case--> of[5]
- place[4] --nmod:of--> birth[6]
- director[9] --case--> of[7]
- director[9] --det--> the[8]
- birth[6] --nmod:of--> director[9]
- FilmA[13] --case--> of[10]
- FilmA[13] --det--> the[11]
- FilmA[13] --compound--> film[12]
- director[9] --nmod:of--> FilmA[13]
- What[1] --punct--> ?[14]

## 5. Undirected Dependency Graph
- What[1] --cop-- is[2]
- What[1] --nsubj-- place[4]
- What[1] --punct-- ?[14]
- the[3] --det-- place[4]
- place[4] --nmod:of-- birth[6]
- of[5] --case-- birth[6]
- birth[6] --nmod:of-- director[9]
- of[7] --case-- director[9]
- the[8] --det-- director[9]
- director[9] --nmod:of-- Gaby: A True Story[13]
- of[10] --case-- Gaby: A True Story[13]
- the[11] --det-- Gaby: A True Story[13]
- film[12] --compound-- Gaby: A True Story[13]

## 6. Entity Start Nodes from Explicit Entities
- e1: Gaby: A True Story graph_node_ids=['13']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Gaby: A True Story -- director -- birth -- place -- What
- e1_p2 (e1): Gaby: A True Story -- director -- birth -- place -- What -- is
- e1_p3 (e1): Gaby: A True Story -- director -- birth -- place -- What -- ?
- e1_p4 (e1): Gaby: A True Story -- director -- birth -- place
- e1_p5 (e1): Gaby: A True Story -- director -- birth -- place -- the
- e1_p6 (e1): Gaby: A True Story -- director -- birth
- e1_p7 (e1): Gaby: A True Story -- director -- birth -- of
- e1_p8 (e1): Gaby: A True Story -- director
- e1_p9 (e1): Gaby: A True Story -- director -- of
- e1_p10 (e1): Gaby: A True Story -- director -- the
- e1_p11 (e1): Gaby: A True Story -- film
- e1_p12 (e1): Gaby: A True Story -- of
- e1_p13 (e1): Gaby: A True Story -- the

## 7.5 Terminal Glue Path Pruning
Total raw paths: 13
Total kept paths: 5
Total pruned paths: 8
Total pruned ratio: 61.54%

### By Entity
- e1 / Gaby: A True Story
  - raw: 13
  - kept: 5
  - pruned: 8
  - fallback_used: False
  - examples:
    - e1_p2: Gaby: A True Story -> director -> birth -> place -> What -> is [terminal=is, reason=terminal_glue_token]
    - e1_p3: Gaby: A True Story -> director -> birth -> place -> What -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p5: Gaby: A True Story -> director -> birth -> place -> the [terminal=the, reason=terminal_glue_token]
    - e1_p7: Gaby: A True Story -> director -> birth -> of [terminal=of, reason=terminal_glue_token]
    - e1_p9: Gaby: A True Story -> director -> of [terminal=of, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=place_of_birth
  Reason: The path starts from the film, connects to the director, and leads to the place of birth, covering all necessary components.
- e1: e1_p4 score=85.0 valid=True terminal=place_of_birth
  Reason: The path effectively connects the film to its director and then to the place of birth, though it lacks the explicit wh cue.
- e1: e1_p6 score=75.0 valid=True terminal=place_of_birth
  Reason: The path connects the film to the director and then to birth, but it misses the explicit mention of place.
- e1: e1_p8 score=30.0 valid=True terminal=place_of_birth
  Reason: The path only connects the film to the director, lacking any reference to birth or place, making it insufficient.
- e1: e1_p11 score=20.0 valid=True terminal=place_of_birth
  Reason: The path connects the film to a general term 'film', which does not contribute to answering the question about the director's place of birth.

## 8.1 Highest-Scored Path per Entity
- e1: e1_p1 score=90.0

## 8.2 Selected Path Set
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: What is the place of birth of the director of film Gaby: A True Story?
- ps1
  - e1_p1: Gaby: A True Story -> director -> birth -> place -> What

Output:
- reason: The DAG compiles each semantic reasoning edge into one atomic lookup.
- q1: Who is the director of Gaby: A True Story? depends_on=[] support=[]
- q2: Where was the director born? depends_on=['q1'] support=[]

## 10. Atomic Subquestion DAG
- None: Who is the director of Gaby: A True Story?
- None: Where was the director born?
