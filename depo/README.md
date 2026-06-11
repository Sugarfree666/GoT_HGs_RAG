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

8.1. **Highest-scored path per entity**
   The program selects exactly one path for each explicit entity: the
   highest-scoring valid path, with fallback to the highest-scoring raw path if
   no valid high-scoring path exists.

8.2. **Selected path set**
   The selected best paths for all entities are combined into exactly one path
   set, `ps1`. For a two-entity question this means the two highest-scoring
   entity-specific paths are used directly.

9. **Semantic Reasoning Path induction**
   The LLM receives the original question, restored question, and compact
   selected dependency path evidence for `ps1`. It converts the syntactic
   dependency evidence into branch-level Semantic Reasoning Paths. These paths
   are not dependency paths and are not a Semantic AST: their nodes are semantic
   objects such as explicit entities, licensed intermediate objects
   (`performer`, `director`, `company`, `CEO`), or value slots (`nationality`,
   `birth_date`, `death_place`, `university`). Their edges are executable
   one-hop semantic relations such as `performer of song` or `place of death`.
   Final comparison, ranking, boolean, intersection, and common-answer intent is
   stored only as metadata for downstream composition.

10. **Atomic DAG compilation**
   The LLM compiles each Semantic Reasoning Path edge into exactly one atomic
   lookup question. Dependent questions must use explicit dependency variables
   such as `q1's answer`. Each atomic node keeps support copied from the source
   semantic edge and remains traceable to `source_semantic_path_id` and
   `source_semantic_edge_id`.
   The old direct dependency-evidence-to-DAG path is retained only as an
   explicit ablation fallback via `--direct-dag`; it is not the default.

11. **Atomic Subquestion DAG**
   The final DAG contains only one-hop lookup subquestions and explicit
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

Run the legacy direct-DAG ablation explicitly:

```powershell
python main.py --direct-dag --question "Which actor is older?"
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
