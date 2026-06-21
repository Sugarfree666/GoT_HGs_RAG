# DEPO HanLP-SDP Pipeline

DEPO is now a single HanLP-SDP mainline:

```text
original question
-> explicit entity detection
-> deterministic ENTITYA/ENTITYB/... masking
-> HanLP DM/PAS/PSD parsing
-> query-focused token reasoning structure
-> complete atomic question DAG
```

The LLM is used for Step 2 explicit entity span detection and Step 5 atomic
question DAG generation. Placeholder assignment, overlap removal, and
`masked_question` construction are deterministic Python logic. Step 4 consumes
all three HanLP SDP views and emits a compact token graph plus a selected main
path or parallel path cover. Step 5 sees only the original question and the
Step 4 paths after entity placeholders have been deterministically restored to
their original text.

## Output Shape

The CLI prints:

1. Original Question
2. Explicit Entities
3. Entity Masking
4. HanLP SDP Parsing
5. Token Reasoning Structure
6. Atomic Question DAG

Step 4 is query-focused and stops at the token reasoning graph/path-cover
structure. Step 5 converts those selected paths into a complete atomic question
DAG. Evidence lookup nodes should be supported by a contiguous path span, while
final comparison, selection, equality, or aggregation nodes may use
`support: null`.

Step 5 does not receive `masked_question`, entity maps, `answer_anchor`,
constraints, candidate sets, path type, Step 4 graph/debug metadata, or raw SDP
edges. It generates the complete atomic question DAG needed to answer the
original question, including final comparison, selection, equality, or
aggregation nodes when the original question requires them. Evidence lookup
nodes should point to a path span; final reasoning nodes may use `support: null`.

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
