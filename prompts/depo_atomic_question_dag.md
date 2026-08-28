You decompose a complex question into an Atomic Question DAG using the supplied JSON input.

The input contains:
`original_question`: the complex question to decompose.
`question_entities`: a possibly incomplete list of explicit entity surface forms from the question.
`question_structure`: structural branches representing the original question, with adjacent phrases separated by -- .

Instructions:

1. An atomic lookup asks one direct relation between available anchors and returns exactly one new unknown; its wording must not contain another unresolved referent. A relational description such as "the parent of X" is not an available anchor: retrieve that referent first. A comparison, selection, verification, or aggregation over already retrieved values is also one atomic operation.
2. Generate the fewest nodes that form a connected DAG sufficient to answer `original_question`. Determine its target from the complete interrogative predicate, not from the wh-word or answer type alone, and reserve the target relation for the last node instead of absorbing it into an earlier identification. The last node must be the only leaf node and must return exactly that target, type, and granularity. Every earlier node must be an ancestor of the last node; if multiple values are requested, combine them in the last node rather than leave multiple leaves.
3. Preserve all stated entities, relations, relation directions, argument roles, candidates, and constraints, including negation and temporal or numeric restrictions. Keep each modifier attached to what it constrains. Constraints coordinated under one singular referent jointly identify one unknown and stay in the same atomic question; create parallel branches only for distinct unknowns whose results must be combined.
4. Use `original_question` as the sole source of meaning and `question_structure` as its semantic skeleton. Each branch represents a structural path through entities, relations, constraints, intermediate referents, and the target; preserve the supported paths and their convergence or parallelism. Use the original wording to interpret exact meaning, relation direction, and dependencies, and never introduce an unsupported fact. Treat `question_entities` only as a surface-form coverage aid.
5. Determine entity boundaries from `original_question`. Every explicit entity must appear in a semantically relevant atomic question with its exact original surface form, including capitalization, spelling, punctuation, parentheticals, appositives, titles, and internal conjunctions. Whenever an entity is mentioned, its complete surface is required; do not let `question_entities` split it, normalize it, replace it, or create a node solely for coverage.
6. Do not retrieve information already stated in `original_question`, introduce outside knowledge, or add an intermediate that is unnecessary for executing a direct relation.
7. A comparison, selection, verification, or aggregation may use only the exact values supplied by its prerequisite answers. Retrieve missing attributes first; for a derived value such as age or duration, retrieve its operands and compute it in a separate node before comparing. Begin every multi-input operation with `Based on qN's answer, ...` and mention each input literally.
8. Use ordered IDs `q1`, `q2`, ... in execution order, with dependencies only on earlier nodes. When consuming an earlier result, replace the resolved expression with the literal `qN's answer`; do not repeat that expression from the original question in its place. Then derive `depends_on` mechanically as the ordered list of IDs referenced literally in the question. Any mismatch is invalid.
9. Before output, conceptually substitute all prerequisite answers and verify that every node remains executable, every earlier node supports the last node, and the last question is answer-equivalent to `original_question`.

## Examples

### Example 1

Input:

```json
{
  "original_question": "Which film was released first, North Wind or South Rain?",
  "question_entities": ["North Wind", "South Rain"],
  "question_structure": [
    "North Wind -- film -- released -- first",
    "South Rain -- film -- released -- first"
  ]
}
```

Output:

```json
{"atomic_questions":[{"id":"q1","question":"When was North Wind released?","depends_on":[]},{"id":"q2","question":"When was South Rain released?","depends_on":[]},{"id":"q3","question":"Based on q1's answer and q2's answer, which film was released first: North Wind or South Rain?","depends_on":["q1","q2"]}]}
```

### Example 2

Input:

```json
{
  "original_question": "Where did Ada North's father study?",
  "question_entities": ["Ada North"],
  "question_structure": ["Ada North -- father -- study -- where"]
}
```

Output:

```json
{"atomic_questions":[{"id":"q1","question":"Who is Ada North's father?","depends_on":[]},{"id":"q2","question":"Where did q1's answer study?","depends_on":["q1"]}]}
```

### Example 3

Input:

```json
{
  "original_question": "Which film has the older director, Copper Sky or Silent Harbor?",
  "question_entities": ["Copper Sky", "Silent Harbor"],
  "question_structure": [
    "Copper Sky -- film -- director -- older",
    "Silent Harbor -- film -- director -- older"
  ]
}
```

Output:

```json
{"atomic_questions":[{"id":"q1","question":"Who directed Copper Sky?","depends_on":[]},{"id":"q2","question":"When was q1's answer born?","depends_on":["q1"]},{"id":"q3","question":"Who directed Silent Harbor?","depends_on":[]},{"id":"q4","question":"When was q3's answer born?","depends_on":["q3"]},{"id":"q5","question":"Based on q2's answer and q4's answer, which film has the older director: Copper Sky or Silent Harbor?","depends_on":["q2","q4"]}]}
```

### Example 4

Input:

```json
{
  "original_question": "Who lived longer, Ada North, 1st Countess of Example or Grace South (engineer)?",
  "question_entities": ["Ada North, 1st Countess of Example", "Grace South (engineer)"],
  "question_structure": [
    "Ada North, 1st Countess of Example -- lived -- longer",
    "Grace South (engineer) -- lived -- longer"
  ]
}
```

Output:

```json
{"atomic_questions":[{"id":"q1","question":"When was Ada North, 1st Countess of Example born?","depends_on":[]},{"id":"q2","question":"When did Ada North, 1st Countess of Example die?","depends_on":[]},{"id":"q3","question":"Based on q1's answer and q2's answer, how long did Ada North, 1st Countess of Example live?","depends_on":["q1","q2"]},{"id":"q4","question":"When was Grace South (engineer) born?","depends_on":[]},{"id":"q5","question":"When did Grace South (engineer) die?","depends_on":[]},{"id":"q6","question":"Based on q4's answer and q5's answer, how long did Grace South (engineer) live?","depends_on":["q4","q5"]},{"id":"q7","question":"Based on q3's answer and q6's answer, who lived longer: Ada North, 1st Countess of Example or Grace South (engineer)?","depends_on":["q3","q6"]}]}
```

### Example 5

Input:

```json
{
  "original_question": "What is the capital of the country containing the headquarters city of Northwind Labs?",
  "question_entities": ["Northwind Labs"],
  "question_structure": ["Northwind Labs -- headquarters -- city -- country -- capital"]
}
```

Output:

```json
{"atomic_questions":[{"id":"q1","question":"In which city is the headquarters of Northwind Labs?","depends_on":[]},{"id":"q2","question":"Which country contains q1's answer?","depends_on":["q1"]},{"id":"q3","question":"What is the capital of q2's answer?","depends_on":["q2"]}]}
```

### Example 6

Input:

```json
{
  "original_question": "When was the person whose sister played Mira in Copper Sky hired by South Lab?",
  "question_entities": ["Mira", "Copper Sky", "South Lab"],
  "question_structure": ["Copper Sky -- Mira -- played -- sister -- person -- hired -- South Lab -- when"]
}
```

Output:

```json
{"atomic_questions":[{"id":"q1","question":"Who played Mira in Copper Sky?","depends_on":[]},{"id":"q2","question":"Whose sister is q1's answer?","depends_on":["q1"]},{"id":"q3","question":"When was q2's answer hired by South Lab?","depends_on":["q2"]}]}
```

Return strict JSON only:
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "atomic natural-language question?",
      "depends_on": []
    }
  ]
}
Do not add any other top-level key or node field. Do not include reasoning, explanations, citations, or any text outside the JSON object.
