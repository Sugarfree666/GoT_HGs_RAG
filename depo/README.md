# DEPO HanLP-SDP Pipeline

DEPO is now a single HanLP-SDP mainline:

```text
original question
-> explicit entity detection
-> deterministic ENTITYA/ENTITYB/... masking
-> HanLP DM/PAS/PSD parsing
-> query-focused token reasoning structure
```

The LLM is used only for explicit entity span detection. Placeholder assignment,
overlap removal, and `masked_question` construction are deterministic Python
logic. Step 4 consumes all three HanLP SDP views and emits a compact token graph
plus a selected main path or parallel path cover.

## Output Shape

The CLI prints:

1. Original Question
2. Explicit Entities
3. Entity Masking
4. HanLP SDP Parsing
5. Token Reasoning Structure

Step 4 is query-focused and stops at the token reasoning graph/path-cover
structure.

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

Useful options:

```text
--question
--questions-file
--api-key
--base-url
--hanlp-model
--debug
--debug-dir
```

## Tests

DEPO unit tests use mocked HanLP SDP results and fake LLM clients; they do not
require a live HanLP model.

```powershell
python -m unittest tests.test_hanlp_sdp_pipeline
python -m unittest discover -s tests
python -m compileall depo tests
```
