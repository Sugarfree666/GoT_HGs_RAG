# DEPO Decomposition Error #4

- Dataset: `hotpotqa`
- Question: Sparking the Marian civil war, who helped the recently abdicated queen to escape her imprisonment?
- Gold answer: the Queen's gaoler
- Error type: `ValueError`

```text
LLM produced invalid selected-path semantic transduction after retry: Branch terminal 'helper' label 'helper' is incompatible with AnswerIntent(answer_kind='person_or_entity', answer_slot_hint=None).
```
