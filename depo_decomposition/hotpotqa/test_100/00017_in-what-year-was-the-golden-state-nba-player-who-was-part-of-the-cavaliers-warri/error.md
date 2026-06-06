# DEPO Decomposition Error #17

- Dataset: `hotpotqa`
- Question: In what year was the Golden State NBA player, who was part of the Cavaliers-Warriors rivalry, named NBA Finals Most Valuable Player?
- Gold answer: 2015
- Error type: `ValueError`

```text
LLM produced invalid selected-path semantic transduction after retry: Branch terminal 'year' label 'year' is incompatible with AnswerIntent(answer_kind='person_or_entity', answer_slot_hint='year').
```
