# DEPO Decomposition Error #8

- Dataset: `hotpotqa`
- Question: Bethpage State Parkway begins with an interchange at which Long Island-based limited access highway?
- Gold answer: Southern State Parkway
- Error type: `ValueError`

```text
Grounded Atomic DAG support validation failed after retry: Node q2 support #1 cites node_texts not present in ps3/e1_p1: ['Long Island', 'based'].
Node q2 has no valid selected dependency path support.
```
