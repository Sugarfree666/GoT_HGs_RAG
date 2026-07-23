# DEPO HanLP-SDP Pipeline

DEPO is currently maintained as one HanLP-SDP decomposition mainline. Older
CoreNLP/OpenIE and Semantic Reasoning Path variants are not part of this active
pipeline.

```text
original question
-> explicit entity detection
-> deterministic ENTITYA/ENTITYB/... masking
-> HanLP DM/PAS/PSD parsing
-> query-focused token reasoning structure with selected path branch(es)
-> Step5 direct Atomic Question DAG generation
-> deterministic DAG parsing
```

The LLM is used for Step 2 explicit entity span detection and Step 5 atomic
question DAG generation. Placeholder assignment, overlap removal, and
`masked_question` construction are deterministic Python logic. Step 4 consumes
all three HanLP SDP views and emits a compact token graph plus selected
question-structure branch(es). Ordinary questions keep one path; comparison/candidate
questions may keep one path per candidate branch. Step 5 directly generates an
Atomic Question DAG from only:

1. `original_question`
2. `question_structure`

The `question_structure` value is produced by restoring the Step 4 selected
path(s) into original entity surface forms. Each structure branch is rendered in
the Step5 LLM prompt as text using `--`, for example `node1 -- node2 -- node3`.
The original question provides the authoritative semantics; the question
structure provides only a rough structural skeleton. It does not encode factual
direction, subject-object direction, or answer dependency direction. Step5 does
not receive the Step 2 entity surface list because entity recognition can be
noisy. Ordinary questions usually pass one structure branch; candidate/comparison
questions usually pass multiple branches. If Step 4 produces no selected path,
Step 5 is still called with the original question and an empty Question
structure section. Step 5 does not output semantic reasoning paths, support
spans, or DAG edges.

## Output Shape

The CLI prints:

1. Original Question
2. Explicit Entities
3. Entity Masking
4. HanLP SDP Parsing
5. Token Reasoning Structure
6. Atomic Question DAG

Step 4 is query-focused and stops at the token reasoning graph/path-cover
structure. Step 5 does not receive entity surface lists, all anchor paths, raw
SDP edges, masks, candidate sets, constraints, support spans, semantic edge
alignment metadata, or path indices.

Step 5 returns JSON atomic questions only:

```json
{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "natural-language atomic question?",
      "depends_on": [],
      "operation": "lookup",
      "output_type": "person"
    }
  ]
}
```

The program parses `atomic_questions` into `AtomicQuestionDAGResult`. Each
question becomes one DAG node. Edges and leaf nodes are derived deterministically
from `depends_on`. Step5 does not ask the LLM to emit semantic reasoning paths,
support spans, or final DAG edges.
The parser still rejects hard structural errors such as duplicate ids, unknown
dependencies, later-node dependencies, cycles, and unresolved ENTITY placeholders.
Mismatches between `depends_on` and textual `qN's answer` mentions are retained
as warnings instead of invalidating the DAG.

## Install

```powershell
pip install -r depo/requirements.txt
```

HanLP may download its configured model on first use. Set `HANLP_HOME` if you
want to control the cache directory, or pass `--hanlp-model` with a local model
path or HanLP pretrained constant.

## Run

Run one question from the repository root:

```powershell
python depo/main.py --question "Where was the director of film The Outlaw Express born?"
```

Run a questions file:

```powershell
python depo/main.py --questions-file questions/hotpotqa/questions.json
```

Enable Step 4 debug JSON:

```powershell
python depo/main.py --debug --debug-dir debug/hanlp_sdp --question "Who was born later, ENTITYA or ENTITYB?"
```

Skip Step 5 while debugging parsing and Step 4:

```powershell
python depo/main.py --skip-step5 --question "Where was the director of film The Outlaw Express born?"
```

Useful options:

```text
--question
--questions-file
--api-key
--base-url
--hanlp-model
--debug
--debug-dir
--skip-step5
```

## Tests

DEPO unit tests use mocked HanLP SDP results and fake LLM clients; they do not
require a live HanLP model.

```powershell
python -m unittest tests.test_hanlp_sdp_pipeline
python -m unittest tests.test_atomic_question_dag
python -m unittest discover -s tests
python -m compileall depo tests
```
