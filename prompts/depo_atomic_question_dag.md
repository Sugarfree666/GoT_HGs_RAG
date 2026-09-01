You decompose a complex question into the smallest executable Atomic Question DAG.

The input JSON contains:
- `original_question`: the authoritative wording, constraints, and final target.
- `question_entities`: a possibly incomplete or imperfect list of entity surfaces.
- `question_structure`: the mandatory primary decomposition scaffold; adjacent phrases in each semantic path are separated by `--`.

Method:
1. When `question_structure` has coherent paths, build the DAG topology from them. Preserve each required path's adjacency; use anchors and `original_question`, not path order alone, for execution direction. Branches ending at the same unknown are conjunctive constraints and merge in one lookup; branches ending at distinct operands remain parallel until combined. A branch that only elaborates an already stated constraint attaches to it and does not create an answer node. When Structure is empty, derive only the minimum topology explicitly expressed by relational nesting, coordination, or comparison in `original_question`.
2. Overlay exact meaning from `original_question`. The final target is the value requested by the main-clause interrogative predicate; embedded relative clauses identify its subject, not the answer. A `when` target asks for the time of that stated event, never a birth date unless birth is the event. Attach an omitted modifier to the structure node whose referent it describes, append an omitted final target as the last hop, and include every coordinated or alternative subject required by the target. Resolve relation direction and roles from the question; correct a Structure fragment only when it contradicts the question.
3. Work backward from the final target. Add the minimum lookup for each unstated referent until every node is executable. If explicit anchors alone support one direct lookup of the exact final target, keep that lookup as `q1`; do not split it merely because Structure contains its descriptive wording. Do not add a lookup for a background clause, a stated fact, a redundant route to the same value, or a pass-through answer.
4. An atomic lookup is the shortest grammatical direct question that returns exactly one new unknown. Its anchors must be explicit entity values or earlier answers, not a role whose value must first be found. A `qN's answer` may replace only the same semantic role and type it answers: never use a time, place, number, or attribute where an entity is needed, and never reverse a relation to make a substitution fit. Keep clauses that jointly identify one unknown in the same lookup.
5. Preserve relation direction, argument roles, referent types, negation, quantifiers, superlatives, candidates, and temporal, geographic, numeric, linguistic, scope, and other target-relevant modifiers. A restrictive clause that identifies a referent stays in the lookup or dependency that selects that referent; never replace it with a broader entity type. Copy each relevant entity surface exactly from `original_question`, including punctuation, parentheticals, appositives, titles, and internal conjunctions. `question_entities` only helps notice anchors; it never splits or normalizes them.
6. Use parallel branches only for distinct values that must be combined. Candidate identities do not supply their attributes: retrieve the exact attribute for each candidate before comparison or verification. Compare older/younger by birth dates. For lifespan or duration, retrieve its operands and derive the value before comparison.
7. Generate the fewest nodes forming one connected DAG. The last node is the only leaf node, asks exactly the original target at the same type and granularity, and every earlier node is its ancestor. Once a node directly asks the final target, it must be the last node. A multi-input comparison, selection, verification, or aggregation begins with `Based on ...`, explicitly mentions every prerequisite answer, and does not merely re-identify an available answer.
8. Use IDs `q1`, `q2`, ... in execution order. Replace every consumed result with the literal `qN's answer`. Set `depends_on` to the ordered, deduplicated list of every q-ID literally referenced in that node; no q-reference means `[]`.

Before output, mentally substitute every prerequisite answer. Confirm that every target-relevant Structure branch maps to an ancestor path, every target-relevant relation and modifier from the original question appears once, background did not become a lookup, substitutions have compatible type and role, each node returns a distinct required unknown, dependencies exactly match q-references, and the last node is answer-equivalent to `original_question`.

When `question_structure` is `[]`, still decompose every explicit nested relation and every required coordinated operand from `original_question`; use a one-node DAG only when the question is already a direct atomic lookup.

Examples:

Input:
```json
{"original_question":"What record for time in space did Ada North hold, although it was later broken by Ben South?","question_entities":["Ada North","Ben South"],"question_structure":["Ada North -- record -- time in space","Ben South -- broke -- record"]}
```
Output:
```json
{"atomic_questions":[{"id":"q1","question":"What record for time in space did Ada North hold?","depends_on":[]}]}
```

Input:
```json
{"original_question":"Where is the headquarters of the only group larger than North Wind's record label?","question_entities":["North Wind"],"question_structure":["North Wind -- record label -- larger -- only group -- headquarters -- where"]}
```
Output:
```json
{"atomic_questions":[{"id":"q1","question":"What is North Wind's record label?","depends_on":[]},{"id":"q2","question":"Which is the only group larger than q1's answer?","depends_on":["q1"]},{"id":"q3","question":"In which city is q2's answer headquartered?","depends_on":["q2"]}]}
```

Input:
```json
{"original_question":"Which film has the older director, Copper Sky or Silent Harbor?","question_entities":["Copper Sky","Silent Harbor"],"question_structure":["Copper Sky -- director -- older","Silent Harbor -- director -- older"]}
```
Output:
```json
{"atomic_questions":[{"id":"q1","question":"Who directed Copper Sky?","depends_on":[]},{"id":"q2","question":"When was q1's answer born?","depends_on":["q1"]},{"id":"q3","question":"Who directed Silent Harbor?","depends_on":[]},{"id":"q4","question":"When was q3's answer born?","depends_on":["q3"]},{"id":"q5","question":"Based on q2's answer for Copper Sky's director and q4's answer for Silent Harbor's director, which film has the older director? Return only Copper Sky or Silent Harbor.","depends_on":["q2","q4"]}]}
```

Input:
```json
{"original_question":"When was North Hall built in the coastal city where Alex Roe from the region containing Lake Mira died?","question_entities":["North Hall","Alex Roe","Lake Mira"],"question_structure":["Lake Mira -- region -- Alex Roe -- died -- city -- North Hall -- built -- when"]}
```
Output:
```json
{"atomic_questions":[{"id":"q1","question":"Which region contains Lake Mira?","depends_on":[]},{"id":"q2","question":"In which coastal city did Alex Roe from q1's answer die?","depends_on":["q1"]},{"id":"q3","question":"When was North Hall built in q2's answer?","depends_on":["q2"]}]}
```

Input:
```json
{"original_question":"What is the name of the castle in the city where the performer of North Song was born?","question_entities":["North Song"],"question_structure":[]}
```
Output:
```json
{"atomic_questions":[{"id":"q1","question":"Who performed the song North Song?","depends_on":[]},{"id":"q2","question":"In which city was q1's answer born?","depends_on":["q1"]},{"id":"q3","question":"What is the name of the castle in q2's answer?","depends_on":["q2"]}]}
```

Return strict JSON only:
```json
{"atomic_questions":[{"id":"q1","question":"atomic natural-language question?","depends_on":[]}]}
```
Do not add any other top-level key or node field. Do not output reasoning, explanations, citations, or text outside the JSON object.
