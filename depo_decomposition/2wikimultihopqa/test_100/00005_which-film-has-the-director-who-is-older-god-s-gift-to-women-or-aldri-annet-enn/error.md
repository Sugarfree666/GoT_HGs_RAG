# DEPO Decomposition Error #5

- Dataset: `2wikimultihopqa`
- Question: Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Bråk?
- Gold answer: God'S Gift To Women
- Error type: `ValueError`

```text
LLM selected invalid entity-origin paths after retry: Selected path_id='e1_p9' passes through another entity start as an intermediate node. For parallel/common-answer questions, choose a path from this entity directly toward the answer slot or compared attribute, not through a different entity start.
```
