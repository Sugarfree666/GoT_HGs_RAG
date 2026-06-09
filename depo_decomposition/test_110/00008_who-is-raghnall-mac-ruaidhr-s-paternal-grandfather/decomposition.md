# DEPO Decomposition #8

- Dataset: `2wikimultihopqa`
- Question: Who is Raghnall Mac Ruaidhrí's paternal grandfather?
- Gold answer: Ailéan mac Ruaidhrí

## 1. Semantic-Normalized Question
Who is the paternal grandfather of Raghnall Mac Ruaidhrí?

## 2. Explicit Entities
- Raghnall Mac Ruaidhr (Person) span=(35, 55)

## 3. Entity Masking
- PersonA -> Raghnall Mac Ruaidhr

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
- grandfather[5] --nmod:of-- Raghnall Mac Ruaidhr[7]
- of[6] --case-- Raghnall Mac Ruaidhr[7]

## 6. Entity Start Nodes from Explicit Entities
- e1: Raghnall Mac Ruaidhr graph_node_ids=['7']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Raghnall Mac Ruaidhr -- grandfather -- paternal
- e1_p2 (e1): Raghnall Mac Ruaidhr -- grandfather -- Who
- e1_p3 (e1): Raghnall Mac Ruaidhr -- grandfather -- Who -- is
- e1_p4 (e1): Raghnall Mac Ruaidhr -- grandfather -- Who -- ?
- e1_p5 (e1): Raghnall Mac Ruaidhr -- grandfather
- e1_p6 (e1): Raghnall Mac Ruaidhr -- grandfather -- the
- e1_p7 (e1): Raghnall Mac Ruaidhr -- of

## 7.5 Terminal Glue Path Pruning
Total raw paths: 7
Total kept paths: 3
Total pruned paths: 4
Total pruned ratio: 57.14%

### By Entity
- e1 / Raghnall Mac Ruaidhr
  - raw: 7
  - kept: 3
  - pruned: 4
  - fallback_used: False
  - examples:
    - e1_p3: Raghnall Mac Ruaidhr -> grandfather -> Who -> is [terminal=is, reason=terminal_glue_token]
    - e1_p4: Raghnall Mac Ruaidhr -> grandfather -> Who -> ? [terminal=?, reason=terminal_glue_token]
    - e1_p6: Raghnall Mac Ruaidhr -> grandfather -> the [terminal=the, reason=terminal_glue_token]
    - e1_p7: Raghnall Mac Ruaidhr -> of [terminal=of, reason=terminal_glue_token]

## 8. LLM Path Scores
- e1: e1_p1 score=90.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from Raghnall Mac Ruaidhr, reaches grandfather, and includes the paternal modifier, effectively supporting the intent to identify the paternal grandfather.
- e1: e1_p2 score=85.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from Raghnall Mac Ruaidhr, reaches grandfather, and includes the wh cue 'who', supporting the intent to identify the paternal grandfather.
- e1: e1_p5 score=70.0 valid=True terminal=paternal_grandfather
  Reason: The path starts from Raghnall Mac Ruaidhr and reaches grandfather, but it lacks the necessary wh cue and does not fully support the intent to identify the paternal grandfather.

## 8.1 Top-2 Paths per Entity
- e1: e1_p1, e1_p2

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p1'} mean_path_score=90.0
- ps2: {'e1': 'e1_p2'} mean_path_score=85.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: Who is Raghnall Mac Ruaidhrí's paternal grandfather?
- ps1
  - e1_p1: Raghnall Mac Ruaidhr -> grandfather -> paternal
- ps2
  - e1_p2: Raghnall Mac Ruaidhr -> grandfather -> Who

Output:
- selected_path_set_ids: ['ps1']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who is Raghnall Mac Ruaidhr's grandfather? depends_on=[] support=['e1_p1']
- q2: Who is the paternal grandfather of Raghnall Mac Ruaidhr? depends_on=['q1'] support=['e1_p1']

## 10. Atomic Subquestion DAG
- None: Who is Raghnall Mac Ruaidhr's grandfather?
- None: Who is the paternal grandfather of Raghnall Mac Ruaidhr?
