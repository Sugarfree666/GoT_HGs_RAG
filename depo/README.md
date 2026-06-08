# DEPO Question Decomposition Pipeline

This project implements a DEPO-style question decomposition pipeline where the
dependency graph stays aligned to a selectively masked CoreNLP parse, while LLM
anchor decisions are made on restored original question text.

## Architecture

1. **Semantic-normalized question**
   The question is lightly normalized for parser stability while preserving the
   original semantic content.

2. **Explicit entity detection**
   The LLM only identifies explicit named entities in the original question. It
   does not extract type variables, roles, relation phrases, answer slots,
   operators, parser-protection phrases, AST nodes, or subquestions. Single-token
   concrete entities such as `AlphaGo`, `Marufabad`, and `Phoolwari` are valid
   entities.

3. **Entity masking**
   The program masks all detected explicit entities, including single-token
   entities, using POS-hint placeholders such as `PersonA`, `FilmA`,
   `LocationA`, `GameA`, `WorkA`, or `EntityA`. Possessive suffixes outside the
   entity span are preserved, e.g. `John Middleton Murry's` becomes
   `PersonA's`.

4. **CoreNLP parse**
   CoreNLP parses the fully entity-masked question. The masked placeholders remain the
   internal graph tokens so token indices and dependency node IDs stay stable.

5. **Undirected dependency graph and restored graph**
   The CoreNLP dependency parse is converted to an unweighted undirected graph.
   Edge metadata preserves the original directed dependency labels for evidence
   and later LLM prompts. Graph node candidates are restored for LLM display. The internal graph still
   contains placeholders like `MovieA`, but the LLM sees candidate text directly
   from the original question:

   ```json
   {"node_id": "8", "text": "Ten9Eight: Shoot For The Moon"}
   ```

   It is never rendered as `MovieA [Ten9Eight: Shoot For The Moon]`.

6. **Entity start mapping**
   Entity start nodes are not re-detected. They are deterministically built from
   Step 2 explicit entity mappings and the corresponding placeholder graph
   nodes. There is no POS/proper-noun fallback in the main flow, so role or slot
   words such as `author`, `director`, `CEO`, `company`, and `nationality` cannot
   become entity starts unless Step 2 explicitly identified them as concrete
   named entities.

7. **Entity-origin path enumeration**
   For each entity start node, the program enumerates bounded simple paths from
   that entity over the undirected dependency graph. Paths may include syntactic
   noise such as determiners, prepositions, wh words, and punctuation because the
   later grounded DAG stage uses them as evidence. Prompt size is controlled by
   scoring and keeping the most useful paths per entity.

8. **LLM path scoring**
   The LLM receives grouped entity-origin paths and scores every path
   independently. It does not choose a single final path, generate candidate
   nodes, create a Problem Frame, build an AST, or generate atomic questions.

8.1. **Top-2 paths per entity**
   The program keeps the top two paths for each entity by score, with fallback
   to the best raw paths if an entity has no valid high-scoring path.

8.2. **Candidate path-set construction**
   The program builds Cartesian path-set candidates from the per-entity top-2
   paths. For two entities, top-2 by top-2 yields up to four path sets.

9. **Grounded Atomic DAG generation**
   The LLM generates the Atomic Subquestion DAG directly from only the original
   question and compact selected dependency path evidence built from top
   path-set candidates. Step 9 does not receive path scores, entity-start
   metadata, full graph edges, rejected paths, restored graph metadata, or a
   direct decomposition draft. It does not generate a Semantic AST, does not
   call a Best-AST judge, and does not produce a final comparison or operator
   question. Each atomic node must be one semantic lookup hop and must cite at
   least one supporting `path_set_id`, `path_id`, and `node_texts` segment from
   the supplied selected dependency paths.

10. **Atomic Subquestion DAG**
   The generated DAG contains only one-hop lookup subquestions and explicit
   dependencies between them. Final comparison, ranking, set, boolean, or
   synthesis reasoning is left to the HyperBranch final answer composer, which
   receives the original question plus atomic answers and evidence.

## Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Install Stanford CoreNLP for Stanza once:

```powershell
python -c "import stanza; stanza.install_corenlp()"
```

Run `questions.json`:

```powershell
python main.py
```

Run one question:

```powershell
python main.py --question "Do director of film Ten9Eight: Shoot For The Moon and director of film Sabotage (1936 Film) share the same nationality?"
```

Run with detailed intermediate output:

```powershell
python main.py --debug --question "Which actor is older?"
```

If Stanza cannot find CoreNLP, pass the CoreNLP directory:

```powershell
python main.py --corenlp-home "C:\path\to\corenlp"
```

If a managed port is occupied, choose another endpoint:

```powershell
python main.py --corenlp-url "http://localhost:9007"
```

## Tests

The unit tests use mocked `DependencyParse` objects and fake LLM clients; they
do not require a live CoreNLP server.

```powershell
python -m unittest
```
