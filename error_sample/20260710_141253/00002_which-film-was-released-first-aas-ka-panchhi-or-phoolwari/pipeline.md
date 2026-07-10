# DEPO + HyperBranch #2

- Dataset: `2wikimultihopqa`
- Question: Which film was released first, Aas Ka Panchhi or Phoolwari?
- Gold answer: Phoolwari
- HyperBranch run: `D:\GitHub\HyperBranch\runs\depo_hypermemory\2wikimultihopqa\20260710_141253\00002_which-film-was-released-first-aas-ka-panchhi-or-phoolwari\hyperbranch_run`

## Atomic DAG
- q1: When was Aas Ka Panchhi released?
  - depends_on: (none)
- q2: When was Phoolwari released?
  - depends_on: (none)
- q3: Which film was released first, q1's answer or q2's answer?
  - depends_on: q1, q2

## Atomic Answers
- q1: INSUFFICIENT_EVIDENCE
  - question: When was Aas Ka Panchhi released?
  - confidence: 0.0
- q2: INSUFFICIENT_EVIDENCE
  - question: When was Phoolwari released?
  - confidence: 0.0
- q3: INSUFFICIENT_EVIDENCE
  - question: Which film was released first, q1's answer or q2's answer?
  - confidence: 0.0

## Final Answer
INSUFFICIENT_EVIDENCE

- confidence: 0.0