# DEPO HanLP-SDP Pipeline

DEPO is currently maintained as one HanLP-SDP decomposition mainline. Older
CoreNLP/OpenIE and Semantic Reasoning Path variants are not part of this active
pipeline.

```text
original question
-> explicit entity detection
-> deterministic ENTITYA/ENTITYB/... masking
-> HanLP DM/PAS/PSD parsing
-> query-focused token reasoning structure with one global best path
-> Step5 path contraction action trace
-> deterministic Atomic Question DAG conversion
```

The LLM is used for Step 2 explicit entity span detection and Step 5 atomic
path contraction action trace generation. Placeholder assignment, overlap removal, and
`masked_question` construction are deterministic Python logic. Step 4 consumes
all three HanLP SDP views and emits a compact token graph plus a single selected
Global Best Path. Step 5 sees only:

1. `original_question`
2. `explicit_entities`
3. `global_best_path`

`explicit_entities` are the original Step 2 entity surface strings. The
`global_best_path` is the Step 4 Global Best Path after ENTITY placeholders have
been deterministically restored to their original text.

## Output Shape

The CLI prints:

1. Original Question
2. Explicit Entities
3. Entity Masking
4. HanLP SDP Parsing
5. Token Reasoning Structure
6. Atomic Question DAG

Step 4 is query-focused and stops at the token reasoning graph/path-cover
structure. Step 5 does not receive all anchor paths, raw SDP edges, masks,
candidate sets, constraints, support spans, or path indices.

Step 5 returns JSON action trace only:

```json
{
  "actions": [
    {
      "id": "q1",
      "consume": ["path node", "relation"],
      "produce": "q1_answer",
      "question": "natural-language atomic question?"
    }
  ]
}
```

The program converts this action trace into the Atomic Question DAG. Each action
becomes one DAG node. `depends_on` is derived from `qN_answer` references in
`consume` and from `qN's answer` references in `question`. Edges and leaf nodes
are then derived from `depends_on`. Step5 no longer asks the LLM to emit
`support`, `start_index`, `end_index`, or final DAG edges.

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
