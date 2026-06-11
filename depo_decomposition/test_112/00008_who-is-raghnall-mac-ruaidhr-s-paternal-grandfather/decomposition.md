# DEPO Decomposition #8

- Dataset: `2wikimultihopqa`
- Question: Who is Raghnall Mac Ruaidhrí's paternal grandfather?
- Gold answer: Ailéan mac Ruaidhrí

## 1. Semantic-Normalized Question
Who is the paternal grandfather of Raghnall Mac Ruaidhrí?

## 2. Explicit Entities
- Raghnall Mac Ruaidhrí (Person) span=(35, 56)

## 3. Entity Masking
- PersonA -> Raghnall Mac Ruaidhrí

Who is the paternal grandfather of PersonA?

## 4. CoreNLP Dependency Parse
- Who[1] --cop--> is[2]
- grandfather[5] --det--> the[3]
- grandfather[5] --amod--> paternal[4]
- Who[1] --nsubj--> grandfather[5]
- PersonA[7] --case--> of[6]
- grandfather[5] --nmod:of--> PersonA[7]
- Who[1] --punct--> ?[8]

## 5. Undirected Dependency Graph
- Who[1] --cop-- is[2]
- Who[1] --nsubj-- grandfather[5]
- Who[1] --punct-- ?[8]
- the[3] --det-- grandfather[5]
- paternal[4] --amod-- grandfather[5]
- grandfather[5] --nmod:of-- Raghnall Mac Ruaidhrí[7]
- of[6] --case-- Raghnall Mac Ruaidhrí[7]

## 6. Entity Start Nodes from Explicit Entities
- e1: Raghnall Mac Ruaidhrí graph_node_ids=['7']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Raghnall Mac Ruaidhrí -- grandfather -- paternal
- e1_p2 (e1): Raghnall Mac Ruaidhrí -- grandfather -- Who
- e1_p3 (e1): Raghnall Mac Ruaidhrí -- grandfather -- Who -- is
- e1_p4 (e1): Raghnall Mac Ruaidhrí -- grandfather -- Who -- ?
- e1_p5 (e1): Raghnall Mac Ruaidhrí -- grandfather
- e1_p6 (e1): Raghnall Mac Ruaidhrí -- grandfather -- the
- e1_p7 (e1): Raghnall Mac Ruaidhrí -- of

## 7.5 Terminal Glue Path Pruning
Total raw paths: 7
Total kept paths: 3
Total pruned paths: 4
Total pruned ratio: 57.14%

### By Entity
- e1 / Raghnall Mac Ruaidhrí
  - raw: 7
  - kept: 3
  - pruned: 4
  - fallback_used: False
  - examples:
    - e1_p3: Raghnall Mac Ruaidhrí -> grandfather -> Who -> is [terminal=is, reason=terminal_glue_token]
    - e1_p4: Raghnall Mac Ruaidhrí -> grandfather -> Who -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p6: Raghnall Mac Ruaidhrí -> grandfather -> the [terminal=the, reason=terminal_glue_token]
    - e1_p7: Raghnall Mac Ruaidhrí -> of [terminal=of, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from Raghnall Mac Ruaidhrí, reaches grandfather, and includes the necessary paternal modifier.
- e1: e1_p2 score=75.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from Raghnall Mac Ruaidhrí and reaches grandfather, but the use of 'Who' as a subject does not contribute to the semantic chain.
- e1: e1_p5 score=55.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from Raghnall Mac Ruaidhrí and reaches grandfather, but it lacks the necessary modifiers and cues to fully support the answer intent.

## 8.1 Highest-Scored Path per Entity
- e1: e1_p1 score=90.0

## 8.2 Selected Path Set
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Who is Raghnall Mac Ruaidhrí's paternal grandfather?
- ps1
  - e1_p1: Raghnall Mac Ruaidhrí -> grandfather -> paternal

Output:
- reason: The DAG compiles each semantic reasoning edge into one atomic lookup.
- q1: Who is the grandfather of Raghnall Mac Ruaidhrí? depends_on=[] support=[]
- q2: What is the relation type of Raghnall Mac Ruaidhrí's grandfather? depends_on=['q1'] support=[]

## 10. Atomic Subquestion DAG
- None: Who is the grandfather of Raghnall Mac Ruaidhrí?
- None: What is the relation type of Raghnall Mac Ruaidhrí's grandfather?
