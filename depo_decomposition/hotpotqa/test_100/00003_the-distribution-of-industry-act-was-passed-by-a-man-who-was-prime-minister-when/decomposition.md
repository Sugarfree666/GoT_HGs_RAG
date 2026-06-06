# DEPO Decomposition #3

- Dataset: `hotpotqa`
- Question: The Distribution of Industry act was passed by a man who was prime minister when?
- Gold answer: 1945 to 1951

## 1. Semantic-Normalized Question
When was the Distribution of Industry act passed by a man who was prime minister?

## 2. Mask Spans
- Distribution of Industry (entity, DistributionOfIndustry)

## 3. Selective Masked Question
When was the SomeEntityA act passed by a man who was prime minister?

## 4. CoreNLP Dependency Parse
- passed[6] --advmod--> When[1]
- passed[6] --aux:pass--> was[2]
- act[5] --det--> the[3]
- act[5] --compound--> SomeEntityA[4]
- passed[6] --nsubj:pass--> act[5]
- man[9] --case--> by[7]
- man[9] --det--> a[8]
- passed[6] --obl:agent--> man[9]
- minister[13] --nsubj--> man[9]
- man[9] --ref--> who[10]
- minister[13] --cop--> was[11]
- minister[13] --amod--> prime[12]
- man[9] --acl:relcl--> minister[13]
- passed[6] --punct--> ?[14]

## 5. Undirected Dependency Graph
- When[1] --advmod-- passed[6]
- was[2] --aux:pass-- passed[6]
- the[3] --det-- act[5]
- Distribution of Industry[4] --compound-- act[5]
- act[5] --nsubj:pass-- passed[6]
- passed[6] --obl:agent-- man[9]
- passed[6] --punct-- ?[14]
- by[7] --case-- man[9]
- a[8] --det-- man[9]
- man[9] --nsubj/acl:relcl-- minister[13]
- man[9] --ref-- who[10]
- was[11] --cop-- minister[13]
- prime[12] --amod-- minister[13]

## 6. Entity Start Nodes
- e1: Distribution of Industry graph_node_ids=['4']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Distribution of Industry -- act -- passed -- man -- minister -- prime
- e1_p2 (e1): Distribution of Industry -- act -- passed -- man -- minister
- e1_p3 (e1): Distribution of Industry -- act -- passed -- man -- minister -- was
- e1_p4 (e1): Distribution of Industry -- act -- passed -- man -- who
- e1_p5 (e1): Distribution of Industry -- act -- passed -- man
- e1_p6 (e1): Distribution of Industry -- act -- passed -- man -- by
- e1_p7 (e1): Distribution of Industry -- act -- passed -- man -- a
- e1_p8 (e1): Distribution of Industry -- act -- passed -- When
- e1_p9 (e1): Distribution of Industry -- act -- passed
- e1_p10 (e1): Distribution of Industry -- act -- passed -- was
- e1_p11 (e1): Distribution of Industry -- act -- passed -- ?
- e1_p12 (e1): Distribution of Industry -- act
- e1_p13 (e1): Distribution of Industry -- act -- the

## 8. LLM Selected Entity Paths
- e1: e1_p8 Distribution of Industry -- act -- passed -- When
  Reason: This path effectively connects the entity 'Distribution of Industry' to the question of 'When' it was passed, providing a direct link to the temporal aspect of the inquiry.

## 9a. Answer Intent Extraction
```json
{
  "wh_cue": "when",
  "answer_kind": "temporal",
  "answer_slot_hint": "date",
  "focus_predicate": null
}
```

## 9b. Intent-Constrained Semantic Transduction
Nodes:
- distribution_of_industry: Distribution of Industry (entity)
- act: act (type_variable)
- passed_date: date (value_slot)

Edges:
- distribution_of_industry -> act (act of Distribution of Industry)
- act -> passed_date (date of passing the act)

## 10. Atomic Subquestion DAG
- None: What is the act of Distribution of Industry?
- None: When was the act of Distribution of Industry passed?
