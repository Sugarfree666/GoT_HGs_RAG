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

4. **Weighted undirected dependency graph**
   The existing dependency relation weight scheme is preserved. Core relations
   such as `nsubj`, `obj`, `iobj`, `ccomp`, and `xcomp` stay low weight;
   modifiers such as `nmod`, `obl`, `amod`, and `compound` stay medium weight;
   `det`, `punct`, and coordination penalties keep their previous behavior.
   This graph is still printed for compatibility and debugging.

5. **Restored graph node candidates**
   Graph node candidates are restored for LLM display. The internal graph still
   contains placeholders like `MovieA`, but the LLM sees candidate text directly
   from the original question:

   ```json
   {"node_id": "8", "text": "Ten9Eight: Shoot For The Moon"}
   ```

   It is never rendered as `MovieA [Ten9Eight: Shoot For The Moon]`.

6. **High-recall candidate nodes and Problem Frame**
   After CoreNLP parsing, the LLM proposes a high-recall candidate node pool and
   a Problem Frame. Candidate nodes are not final AST nodes. They may include
   entities, roles, slots, type qualifiers, operator cues, constraint values,
   coreference mentions, and other noisy but useful graph nodes. The Problem
   Frame declares the operator and the branch-level requirements.

7. **Candidate-projected graph and candidate paths**
   The CoreNLP dependency graph is converted to an unweighted undirected graph.
   Candidate nodes are projected onto it: two candidates receive a projected
   edge only if a short dependency bridge connects them without crossing another
   candidate. The program enumerates simple candidate paths, removes reverse
   duplicates, and keeps only paths that mention at least one requirement root
   or target.

8. **LLM path selection**
   The LLM selects exactly one filtered candidate path per requirement. It may
   only choose from provided `path_id` values. It cannot invent paths, generate
   an AST, or generate atomic questions. The program validates the selection and
   retries once on invalid output.

9. **Selected paths to AST skeleton**
   The AST skeleton is built by code from selected paths. Adjacent candidate
   nodes become one-hop AST edges. Shared surface variables in multiple branches
   are cloned, for example `director_r1`, `nationality_r1`, `director_r2`,
   `nationality_r2`. Root entities can remain shared or un-cloned. The operator
   and operator inputs come from the Problem Frame and branch terminals.

10. **Relation labeling**
   The LLM labels fixed AST edges with relation hints and confirms operator
   metadata. It cannot add, delete, merge, reorder, or shortcut skeleton nodes
   or edges. The validator rejects any structure not derived from selected
   paths. The fixed allowed operator set is:

   `NONE`, `COMPARE_SAME`, `COMPARE_DIFF`, `COMPARE_GREATER`,
   `COMPARE_LESS`, `ARGMAX`, `ARGMIN`, `INTERSECTION`, `UNION`,
   `DIFFERENCE`, `LOGICAL_AND`, `LOGICAL_OR`.

11. **Execution DAG and atomic subquestion generation**
   The final semantic AST is compiled into a deterministic execution
   DAG. This code layer decides edge order, variable bindings such as `X1` and
   `X2`. Non-`NONE` operators stay in the semantic AST, but they are not emitted
   as extra atomic DAG questions by default.

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
