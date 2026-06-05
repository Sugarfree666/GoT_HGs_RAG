# DEPO Question Decomposition Pipeline

This project implements a DEPO-style question decomposition pipeline where the
dependency graph stays aligned to a selectively masked CoreNLP parse, while LLM
anchor decisions are made on restored original question text.

## Architecture

1. **Mask span extraction**
   Step 1 is not full anchor extraction. It only finds complex named entities
   and multi-word type/function noun phrases that should be masked to protect
   CoreNLP parsing. It does not select anchors, infer implicit variables, choose
   operators, split coordination, build an AST, or generate subquestions.

2. **Selective masking**
   Complex spans are replaced with POS-hint placeholders such as `MovieA`,
   `CompanyA`, `NetworkA`, or `TypeVarA`. Simple type variables such as
   `director`, `CEO`, `university`, `city`, and `nationality` normally remain
   in natural language.

3. **CoreNLP parse**
   CoreNLP parses the masked question. The masked placeholders remain the
   internal graph tokens so token indices and dependency node IDs stay stable.

4. **Undirected dependency graph**
   The CoreNLP dependency parse is converted to an unweighted undirected graph.
   Edge metadata preserves the original directed dependency labels for evidence
   and later LLM prompts, but the post-parse path search no longer uses
   dependency weights.

5. **Restored graph node candidates**
   Graph node candidates are restored for LLM display. The internal graph still
   contains placeholders like `MovieA`, but the LLM sees candidate text directly
   from the original question:

   ```json
   {"node_id": "8", "text": "Ten9Eight: Shoot For The Moon"}
   ```

   It is never rendered as `MovieA [Ten9Eight: Shoot For The Moon]`.

6. **Entity start detection**
   Known entity start nodes are selected deterministically from mask mappings,
   restored graph node metadata, and conservative proper-noun fallback rules.
   The LLM no longer generates candidate nodes and no Problem Frame is built.
   Role or slot words such as `author`, `director`, `CEO`, `company`, and
   `nationality` are not entity starts unless they are masked concrete entities.

7. **Entity-origin path enumeration**
   For each entity start node, the program enumerates bounded simple paths from
   that entity over the undirected dependency graph. Paths may include syntactic
   noise such as determiners, prepositions, wh words, and punctuation because the
   later AST stage prunes noise. Prompt size is controlled by scoring and keeping
   the most useful paths per entity.

8. **LLM entity-origin path selection**
   The LLM receives grouped paths and chooses exactly one provided `path_id` for
   each entity. It cannot invent paths, generate candidate nodes, create a
   Problem Frame, build an AST, or generate atomic questions. The program
   validates entity/path consistency and retries once on invalid output.

9. **Selected Path Semantic Transduction**
   The selected dependency paths are syntactic evidence, not the final semantic
   graph. The LLM converts each selected path into an executable branch-level
   relation chain whose nodes are query endpoints: fixed entities, licensed
   intermediate variables, and final value slots such as `death_date`,
   `birth_date`, `birthplace`, `university`, or `nationality`.

   Step 9 must not copy wh words, auxiliaries, prepositions, punctuation,
   comparison cues, or pure event predicates such as `die`, `born`, `worked`,
   `developed`, and `graduated` into AST nodes. Those tokens are converted into
   relation labels or value slots, for example `When + die -> death_date`.
   Possessive chains are kept executable, so `Lothair II's mother` becomes
   `Lothair II -> mother`, followed by `mother -> death_date`. DEPO validates
   this contract and retries once on invalid transductions.

10. **Execution DAG and atomic subquestion generation**
   The final semantic AST is compiled into a deterministic execution
   DAG. This code layer decides edge order, variable bindings such as `X1` and
   `X2` from ordinary path edges only.

   The LLM then receives only one compiled plan step at a time. For an edge
   step, it turns `known -> ask` into one atomic subquestion whose answer is the
   assigned variable. Final comparison, ranking, set, or logical reasoning is
   left to the HyperBranch final answer composer, which receives the original
   question plus the atomic answers and evidence. The LLM is no longer allowed
   to see and re-plan the full AST during subquestion generation, which prevents
   multi-hop fusion and accidental expansion of already-bound variables.

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
