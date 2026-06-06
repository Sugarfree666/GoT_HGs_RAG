# DEPO Decomposition Error #10

- Dataset: `hotpotqa`
- Question: The 2000 ICC KnockOut Trophy was the debut of a Jamaican cricketer who is a right-handed what?
- Gold answer: middle order batsman
- Error type: `ValueError`

```text
LLM produced invalid selected-path semantic transduction after retry: Branch terminal 'batsman' label 'batsman' is incompatible with AnswerIntent(answer_kind='person_or_entity', answer_slot_hint=None).
```
