from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPO_ROOT = PROJECT_ROOT / "depo"
if str(DEPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METHOD = "direct_llm_atomic_dag"


DIRECT_LLM_ATOMIC_DAG_SYSTEM = r"""
You decompose complex questions into semantic-preserving Atomic Question DAGs.

Convert the given `original_question` into the smallest DAG of retrieval-executable
questions whose final answer is exactly the answer requested by the original. Semantic
equivalence has higher priority than brevity or producing more nodes.

Two conditions are non-negotiable: the DAG has exactly one final node (its only leaf), and
that node asks for the original answer target rather than an intermediate entity or fact. A
well-formed JSON object that has several leaves or ends on the wrong answer target is wrong.

The original question is the only source of meaning. Do not answer it, use outside
knowledge, repair it from facts you know, or invent an entity, entity type, relation,
restriction, candidate, or hop. If wording is awkward or ambiguous, preserve the reading
supported by its wording and grammatical structure instead of guessing from world knowledge.

## Silent semantic contract

Before writing JSON, reason silently:

1. Establish the **answer contract**. Identify the governing wh/choice clause and mark its
   exact unknown span as `ANSWER`. Read an in-situ or trailing interrogative in its actual
   grammatical position, especially in noisy wording: `a person who served when?` asks for
   a time, `a building in what city?` asks for a city, and `licensed to serve what?` asks for
   the served object. Turn the original into a declarative answer template by replacing only
   that span with `ANSWER`. The final leaf's answer must fill that same slot. Do not promote
   a nearby person, work, event, or descriptive clause into the target. Preserve the exact
   answer type and granularity:
   `which film` returns a film, `who` returns a person, `what counted noun` returns the
   counted thing or category, and `how many` returns a number. A question with alternatives
   such as `Was A or B born first?` asks for the chosen candidate, not a boolean, unless the
   original asks whether a proposition is true.
2. Build a referent map. Bracket every named anchor, descriptive span, and restrictive
   modifier. Resolve relation direction, participant roles, modifier attachment,
   coordination scope, and pronoun or descriptive-phrase reference from the original
   wording. Include each span's type-bearing head and all of its restrictive modifiers.
3. Build an evidence plan. Trace the necessary intermediate referents and values from the
   innermost anchors to the final target. Preserve every answer-changing element on that
   trace: named surface forms, restrictive and contrastive clauses, conjunction or
   disjunction, negation, quantifiers, superlatives, comparisons, and temporal or numeric
   conditions.

Keep a silent coverage ledger: every content-bearing part of the original must occur in the
node that retrieves the referent or fact it constrains, or in a later node that uses it.
Putting a clause in a disconnected branch does not preserve it. List the named anchors and
restrictions before decomposing; afterward, each one must occur in a node with a dependency
path to the final node. Do not drop a supplied anchor because another constraint appears
sufficient or because outside knowledge would make it redundant.

Read named noun phrases as complete anchors. A proper name inside a type phrase such as
`the river NAME` or `the surname NAME` names that item; it is not an unnamed attribute of a
different nearby entity unless the original grammar explicitly makes it one. Preserve the
full surface form of a name or title, including an internal `and`; do not split it as
question-level coordination. In particular, one name-bearing head governing coordinated
parts denotes one complete name; separate anchors require the head or relation to be
grammatically repeated. For example, `Battle of Alpha and Beta` is one complete battle
name, not `Battle of Alpha` plus an unrelated `Beta`.

Containing a name does not make a descriptive span known. A possessive or relative phrase
such as `NAME's country`, `the maker of NAME`, or `the area where NAME is located` still
denotes an unknown referent. Resolve it before a later relation uses that referent. Ask for
the full typed span: if the original needs a state, county, area, or era, ask for that type
directly instead of asking generic `where` or `when` and silently adding another hop later.
When the original gives no entity type, keep the anchor type-neutral rather than inferring
one from outside knowledge.

Distinguish **given constraints** from unknowns. A value or fact explicitly supplied by the
original is evidence that filters or connects the unknown; it is not a separate answer to
retrieve. Do not ask a node to rediscover a supplied founder, a stated count such as
`65,000`, a stated date, or whether an explicitly stated property is true. When several
supplied descriptions jointly identify the final answer, keep them together in one lookup.
Split only an embedded descriptive span whose answer is genuinely unknown and is then
substituted into a later relation.

## Form atomic questions

An atomic lookup asks for one new entity, attribute, value, set, or fact through one
retrieval step. It must still contain all arguments and modifiers that define that step.

Create an intermediate node exactly when its unknown answer is needed to evaluate a later
relation. Do not hide two sequential unknown relations inside one node. Conversely, do not
split one predicate or event description into a different chain of relations. Several
descriptions may stay together when they jointly identify the same answer.

Plan the final question first from the `ANSWER` template, then add only the unknown inputs it
needs. If an earlier node already returns the original target and no requested comparison,
verification, aggregation, or later relation remains, that node is the final node. Do not add
a wrapper that merely asks what that node's answer is or restates the already solved target.

Treat a dependency as **faithful span substitution**: an earlier answer replaces the exact
descriptive span that denotes it in the original question. Keep the surrounding predicate,
roles, prepositions, and restrictions unchanged. Do not add another containment, location,
time, possession, or entity-type step merely to connect the replacement. Do not change the
requested type or granularity (for example, a state into an unspecified location, an event
into its date, or a work into an assumed song). Preserve argument direction: `Who or what is
PERSON a commentator for?` becomes `Who or what is q1's answer a commentator for?`, never
`Who is a commentator for q1's answer?`.

Build bottom-up. After asking for an embedded span, form its parent question by replacing
that span with `qN's answer`. If a parent uses several resolved child spans, replace every
one and depend on all of them. A dependency is executable dataflow, never a comment about
related context: a node with `qN` in `depends_on` is invalid unless its question literally
contains `qN's answer` in the semantic role previously occupied by that span.

## Selection, comparison, and verification

For every comparison or selection, distinguish **candidate carriers** from **evidence
values**. Candidate carriers are the things the final answer may be. Evidence values are
dates, ages, counts, durations, birth or death facts, release times, and similar facts used
to choose among the candidates.

If the original asks `which X`, `who`, or another candidate-returning question, the final
node must return the candidate carrier, not the evidence value. Keep the candidates visible
in the final question and use dependency answers only as evidence. Do not ask which film,
person, or work is `q1's answer or q2's answer` when `q1` and `q2` are dates, counts,
durations, directors, or other intermediate facts. A good final form is:
`Based on q2's answer and q4's answer, which candidate satisfies the original comparison:
Candidate A or Candidate B?`

Use `select` when the final answer is one of the candidate entities. Use `compare` only
when the requested answer is the comparison result itself. Use `verify` only when the
original final answer is true/false, not when a wh-question was rewritten into an identity
check. Alternative-choice wording such as `Was A or B born first?` still returns A or B;
it is not a yes/no question.

Use evidence that is complete for the requested comparison. For derived metrics such as
`lived longer`, `older`, `younger`, `duration`, or `length of term`, retrieve the metric
directly when possible, or retrieve all endpoints needed to compute it. Do not substitute a
single boundary value such as only a death date for a lifespan comparison.

## Constraints and coordination

Follow the scope of coordination. Descriptions joined as constraints on one referent must
converge on that referent; do not identify it from only one constraint and leave the other
as a separate fact. If two roles or properties jointly describe one unknown person or thing,
retrieve that one referent with both properties, not two independent referents. When one wh
slot has several type or role descriptors, return the single referent satisfying all of
them instead of creating one leaf per descriptor.

Do not turn modifiers into detached fact-check branches. A relative, participial, or
appositive clause that restricts the requested referent belongs in the question returning
that referent. Likewise, a supplied quantity modifies the requested object type; it is not a
second requested answer. Coordination of restrictions does not by itself request multiple
answers.

Treat relative clauses, appositives, participial phrases, and trailing descriptions as
restrictions on the noun they modify, not as a new final answer target. Never turn a
wh-question into a final yes/no verification merely because two descriptions should identify
the same entity. If the original asks for the artist, series, country, museum object type,
or other entity, the final leaf must still return that entity with all restrictions attached.

Create parallel branches only when the original truly needs separate candidate evidence,
and then add a final node that performs the requested selection, comparison, verification,
or aggregation. Never invent a combining operation that the original question does not
request.

Stop when the answer contract is satisfied. Every node must contribute to the final node,
and the final node must be the only leaf. If a draft has several leaves, either combine the
shared constraints or predicate as required by the original, add the explicitly requested
final operation, or delete an irrelevant branch; never return the multi-leaf draft.

Apply this mechanical graph check immediately before output. Let `all_ids` be every node ID
and `referenced_ids` be the union of every `depends_on` list. The graph leaves are
`all_ids - referenced_ids`. Require exactly:

`all_ids - referenced_ids == {last_id}`

Thus every node except the last must occur in a later node's dependency path, and the last
node must be the sole final answer node. Never create one final node per candidate or per
constraint; all necessary branches must converge once on the last node.

## Dependencies and operations

Use ordered IDs `q1`, `q2`, `q3`, ... . A node may depend only on earlier nodes.

When a question uses an earlier answer, write exactly `qN's answer` in the question and put
`qN` in `depends_on`. For each node, the set of IDs literally referenced as `qN's answer`
must equal its `depends_on` set exactly. Do not declare a dependency while restating the
original name or description instead of substituting the answer, and do not reference an
answer without declaring it.

Use only:

* `lookup`: retrieve an entity, fact, attribute, value, or set;
* `select`: choose the requested entity or candidate from earlier answers;
* `compare`: return the comparison relation or result when that is the requested answer;
* `verify`: return a boolean, only when the original asks a yes/no question;
* `aggregate`: perform a requested numeric or set operation.

## Semantic equivalence check

Before returning, silently substitute each dependency answer back into the question that
uses it, recursively through the final node. The reconstructed final question must be
answer-equivalent to the original: same target, answer type and granularity; same relations,
directions and roles; and the same restrictions and coordination. Also verify that each
intermediate answer has the type required by its use. If any clause is lost, attached to a
different referent, merely placed on a disconnected branch, or turned into a different
question, revise the DAG. Perform the `ANSWER`-slot test: a possible answer to the final node
must fit the original declarative answer template without changing which argument is unknown.
If the original asks for a time, city, object of a verb, or organization but the final node
returns the neighboring person, building, subject, or event, it fails this test. Finally
apply the mechanical leaf equation and exact dependency-reference equality; do not return
until both pass.

## Output

Return exactly one JSON object and no other text:

{
  "atomic_questions": [
    {
      "id": "q1",
      "question": "atomic natural-language question?",
      "depends_on": [],
      "operation": "lookup"
    }
  ]
}

Do not add another top-level key or node field.

### Example 1: sequential composition

Original question:
What is the place of birth of the performer of song Changed It?

Output:
{"atomic_questions":[{"id":"q1","question":"Who performed the song Changed It?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"Where was q1's answer born?","depends_on":["q1"],"operation":"lookup"}]}

### Example 2: candidate comparison keeps candidates separate from evidence

Original question:
Which film has the director who was born first, Orion Road or Harbor Lights?

Output:
{"atomic_questions":[{"id":"q1","question":"Who directed the film Orion Road?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"When was q1's answer born?","depends_on":["q1"],"operation":"lookup"},{"id":"q3","question":"Who directed the film Harbor Lights?","depends_on":[],"operation":"lookup"},{"id":"q4","question":"When was q3's answer born?","depends_on":["q3"],"operation":"lookup"},{"id":"q5","question":"Based on q2's answer and q4's answer, which film has the director who was born first: Orion Road or Harbor Lights?","depends_on":["q2","q4"],"operation":"select"}]}

### Example 3: alternative wording can still ask for a candidate

Original question:
Was Mira Stone or Leon Vale born first?

Output:
{"atomic_questions":[{"id":"q1","question":"When was Mira Stone born?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"When was Leon Vale born?","depends_on":[],"operation":"lookup"},{"id":"q3","question":"Based on q1's answer and q2's answer, who was born first: Mira Stone or Leon Vale?","depends_on":["q1","q2"],"operation":"select"}]}

### Example 4: wh target with restrictions is not verification

Original question:
The ballad North Wind was recorded by which folk artist who also goes by the name River Blue?

Output:
{"atomic_questions":[{"id":"q1","question":"Which folk artist who also goes by the name River Blue recorded the ballad North Wind?","depends_on":[],"operation":"lookup"}]}

### Example 5: several descriptors can identify one answer

Original question:
What market town, civil parish, and ward contains Old Tower?

Output:
{"atomic_questions":[{"id":"q1","question":"What market town, civil parish, and ward contains Old Tower?","depends_on":[],"operation":"lookup"}]}

### Example 6: the governing predicate follows a long subject description

Original question:
What was the city where the creator of Alder Hall died later known as?

Output:
{"atomic_questions":[{"id":"q1","question":"Who created Alder Hall?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"In which city did q1's answer die?","depends_on":["q1"],"operation":"lookup"},{"id":"q3","question":"What was q2's answer later known as?","depends_on":["q2"],"operation":"lookup"}]}

### Example 7: a trailing interrogative determines the final answer type

Original question:
The Alder Act was passed by the person who served as prime minister when?

Output:
{"atomic_questions":[{"id":"q1","question":"Who passed the Alder Act?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"During what period did q1's answer serve as prime minister?","depends_on":["q1"],"operation":"lookup"}]}

### Example 8: supplied facts are joint filters, not separate leaves

Original question:
Designer Mira Vale worked with what watch manufacturer founded by Jordan Lee?

Output:
{"atomic_questions":[{"id":"q1","question":"What watch manufacturer founded by Jordan Lee did designer Mira Vale work with?","depends_on":[],"operation":"lookup"}]}

### Example 9: candidate evidence converges on one final node

Original question:
Between Northbridge College and Lakeview College, which was founded as a vocational school?

Output:
{"atomic_questions":[{"id":"q1","question":"Was Northbridge College founded as a vocational school?","depends_on":[],"operation":"lookup"},{"id":"q2","question":"Was Lakeview College founded as a vocational school?","depends_on":[],"operation":"lookup"},{"id":"q3","question":"Based on q1's answer and q2's answer, which was founded as a vocational school: Northbridge College or Lakeview College?","depends_on":["q1","q2"],"operation":"select"}]}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a direct-LLM Atomic Question DAG decomposition baseline over "
            "questions/*/questions.json."
        )
    )
    parser.add_argument("--dataset", help="Dataset subdirectory under questions/, e.g. musique.")
    parser.add_argument("--questions-file", help="Specific questions JSON or JSONL file. Overrides --dataset.")
    parser.add_argument("--all-datasets", action="store_true", help="Process every questions/*/questions.json file.")
    parser.add_argument("--questions-root", default="questions", help="Root directory containing dataset folders.")
    parser.add_argument(
        "--output-root",
        default="runs/llm_direct_atomic_dag",
        help="Root directory for JSONL and Markdown reports.",
    )
    parser.add_argument("--run-id", help="Output run id under output-root/dataset/. Defaults to current timestamp.")
    parser.add_argument("--start", type=int, default=1, help="1-based inclusive start index in each input file.")
    parser.add_argument("--end", type=int, help="1-based inclusive end index in each input file.")
    parser.add_argument("--limit", type=int, help="Maximum questions after applying --start/--end.")
    parser.add_argument("--resume", action="store_true", help="Skip questions already present in the output JSONL.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model used for direct decomposition.")
    parser.add_argument("--max-retries", type=int, default=3, help="JSON retry count for each LLM call.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-question DAG details on stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    base_url = os.getenv("OPENAI_BASE_URL") or args.base_url
    if not api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    try:
        question_files = _resolve_question_files(args)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not question_files:
        print("No questions files found.", file=sys.stderr)
        return 2

    try:
        from llm_client import LLMClient
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = _repo_path(args.output_root)
    llm_client = LLMClient(api_key=api_key, base_url=base_url, model=args.llm_model)

    overall_counts: Counter[str] = Counter()
    for questions_file in question_files:
        dataset = _dataset_name(questions_file)
        try:
            items = _slice_items(
                _read_question_items(questions_file),
                start=args.start,
                end=args.end,
                limit=args.limit,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"{dataset}: {exc}", file=sys.stderr)
            return 2
        if not items:
            print(f"No questions selected for {dataset}.", file=sys.stderr)
            continue

        output_dir = output_root / dataset / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "direct_atomic_dag.jsonl"
        report_path = output_dir / "direct_atomic_dag.md"
        processed_keys = _processed_result_keys(results_path) if args.resume else set()

        report_lines = _report_header(
            dataset=dataset,
            questions_file=questions_file,
            run_id=run_id,
            start=args.start,
            end=args.end,
            limit=args.limit,
        )
        if args.resume and report_path.exists():
            existing = report_path.read_text(encoding="utf-8").splitlines()
            report_lines = existing if existing else report_lines
            if report_lines and report_lines[-1]:
                report_lines.append("")

        dataset_counts: Counter[str] = Counter()
        mode = "a" if args.resume else "w"
        print(
            f"Running direct LLM Atomic DAG: dataset={dataset}, "
            f"questions={len(items)}, output={output_dir}"
        )
        with results_path.open(mode, encoding="utf-8") as results_file:
            for offset, item in enumerate(items, start=1):
                key = _result_key(dataset, item["index"], item.get("qid"))
                if key in processed_keys:
                    payload = build_skipped_payload(dataset, questions_file, item)
                    print(f"[skip {offset}/{len(items)}] {dataset} #{item['index']} {item['question']}")
                else:
                    print(f"[run {offset}/{len(items)}] {dataset} #{item['index']} {item['question']}")
                    payload = run_one_question(
                        llm_client=llm_client,
                        dataset=dataset,
                        questions_file=questions_file,
                        item=item,
                        max_retries=args.max_retries,
                    )
                    if not args.quiet:
                        _print_result(payload)

                results_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                results_file.flush()
                report_lines.extend(build_markdown_result(payload))
                report_lines.append("")
                report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
                dataset_counts[payload["status"]] += 1
                overall_counts[payload["status"]] += 1

        print(f"Direct Atomic DAG JSONL written to {results_path}")
        print(f"Direct Atomic DAG Markdown report written to {report_path}")
        print(f"{dataset} status counts: {dict(sorted(dataset_counts.items()))}")

    print(f"Overall status counts: {dict(sorted(overall_counts.items()))}")
    return 0


def run_one_question(
    *,
    llm_client: Any,
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    try:
        result = decompose_question_direct(
            llm_client,
            item["question"],
            max_retries=max_retries,
        )
        return build_result_payload(
            dataset=dataset,
            questions_file=questions_file,
            item=item,
            dag_result=result,
        )
    except Exception as exc:
        return build_error_payload(dataset, questions_file, item, exc)


def decompose_question_direct(
    llm_client: Any,
    question: str,
    *,
    max_retries: int = 3,
) -> Any:
    from atomic_question_dag import validate_atomic_question_dag

    raw_payload = llm_client.chat_json(
        DIRECT_LLM_ATOMIC_DAG_SYSTEM,
        build_direct_atomic_dag_prompt(question),
        max_retries=max_retries,
    )
    return validate_atomic_question_dag(raw_payload, original_question=question)


def build_direct_atomic_dag_prompt(question: str) -> str:
    return f"""Original question:
{question}

Return the Atomic Question DAG JSON object.""".strip()


def build_result_payload(
    *,
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    dag_result: Any,
) -> dict[str, Any]:
    dag = dag_result.to_dict()
    status = "ok" if dag_result.valid else "invalid"
    return {
        "method": METHOD,
        "status": status,
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "gold_answer": item.get("answer"),
        "raw_question_item": item.get("raw"),
        "atomic_question_dag": dag,
    }


def build_error_payload(
    dataset: str,
    questions_file: Path,
    item: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "method": METHOD,
        "status": "error",
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "gold_answer": item.get("answer"),
        "raw_question_item": item.get("raw"),
        "atomic_question_dag": None,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def build_skipped_payload(dataset: str, questions_file: Path, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": METHOD,
        "status": "skipped",
        "dataset": dataset,
        "questions_file": str(questions_file),
        "index": item["index"],
        "qid": item.get("qid"),
        "question": item["question"],
        "gold_answer": item.get("answer"),
        "raw_question_item": item.get("raw"),
        "atomic_question_dag": None,
    }


def build_markdown_result(payload: dict[str, Any]) -> list[str]:
    title = f"## Question {payload['index']}"
    if payload.get("qid"):
        title += f" ({payload['qid']})"
    lines = [
        title,
        "",
        "### Original Question",
        "",
        str(payload["question"]),
    ]
    if payload.get("gold_answer") is not None:
        lines.extend(["", f"- Gold answer: {payload['gold_answer']}"])
    lines.extend(["", "### Status", "", f"`{payload['status']}`"])

    if payload["status"] == "error":
        lines.extend(
            [
                "",
                "### Error",
                "",
                f"- Type: `{payload['error_type']}`",
                f"- Message: {payload['error']}",
            ]
        )
        return lines
    if payload["status"] == "skipped":
        return lines

    dag = payload["atomic_question_dag"] or {}
    nodes = dag.get("nodes") or []
    lines.extend(["", "### Atomic Question DAG", ""])
    if nodes:
        for node in nodes:
            dependencies = node.get("depends_on") or []
            dep_text = ", ".join(dependencies) if dependencies else "(none)"
            operation = node.get("operation") or "lookup"
            lines.append(f"- {node.get('id')}: {node.get('question')}")
            lines.append(f"  - depends_on: {dep_text}")
            lines.append(f"  - operation: {operation}")
    else:
        lines.append("(none)")

    edges = dag.get("edges") or []
    if edges:
        lines.extend(["", "### Edges", ""])
        for edge in edges:
            lines.append(f"- {edge.get('source')} -> {edge.get('target')}")

    if dag.get("leaf_node_ids"):
        lines.extend(["", f"- Leaf nodes: {', '.join(dag['leaf_node_ids'])}"])
    lines.extend(["", f"- Valid: `{dag.get('valid')}`"])

    validation_errors = dag.get("validation_errors") or []
    if validation_errors:
        lines.extend(["", "### Validation Errors", ""])
        lines.extend(f"- {error}" for error in validation_errors)

    warnings = dag.get("warnings") or []
    if warnings:
        lines.extend(["", "### Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    return lines


def _print_result(payload: dict[str, Any]) -> None:
    status = payload["status"]
    if status == "error":
        print(f"  error: {payload['error_type']}: {payload['error']}")
        return
    dag = payload.get("atomic_question_dag") or {}
    print(
        f"  status={status} nodes={len(dag.get('nodes') or [])} "
        f"valid={dag.get('valid')} warnings={len(dag.get('warnings') or [])}"
    )
    for error in dag.get("validation_errors") or []:
        print(f"  validation: {error}")


def _report_header(
    *,
    dataset: str,
    questions_file: Path,
    run_id: str,
    start: int,
    end: int | None,
    limit: int | None,
) -> list[str]:
    lines = [
        f"# Direct LLM Atomic Question DAG: {dataset}",
        "",
        f"- Run id: `{run_id}`",
        f"- Questions file: `{questions_file}`",
        "- Scope: original question -> atomic question DAG",
        f"- Start: `{start}`",
    ]
    if end is not None:
        lines.append(f"- End: `{end}`")
    if limit is not None:
        lines.append(f"- Limit: `{limit}`")
    lines.append("")
    return lines


def _resolve_question_files(args: argparse.Namespace) -> list[Path]:
    if args.questions_file:
        path = _repo_path(args.questions_file)
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {path}")
        return [path]

    questions_root = _repo_path(args.questions_root)
    if args.dataset:
        return [_dataset_questions_file(questions_root, args.dataset)]

    if args.all_datasets:
        files: list[Path] = []
        for dataset_dir in sorted(path for path in questions_root.iterdir() if path.is_dir()):
            try:
                files.append(_dataset_questions_file(questions_root, dataset_dir.name))
            except FileNotFoundError:
                continue
        return files

    raise ValueError("Specify --dataset, --questions-file, or --all-datasets.")


def _dataset_questions_file(questions_root: Path, dataset: str) -> Path:
    dataset_dir = questions_root / dataset
    for name in ("questions.json", "question.json"):
        path = dataset_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Questions file not found under: {dataset_dir}")


def _read_question_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        raw_items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw_items = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw_items, list):
        raise ValueError(f"{path} must contain a JSON list or JSONL records.")

    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        if isinstance(item, str):
            question = item.strip()
            raw = item
            qid = None
            answer = None
        elif isinstance(item, dict):
            question = str(item.get("question", "")).strip()
            raw = item
            qid_value = item.get("id", item.get("qid", item.get("_id")))
            qid = str(qid_value) if qid_value is not None else None
            answer = item.get("answer")
        else:
            raise ValueError(f"Unsupported question item at index {index}: {item!r}")
        if not question:
            raise ValueError(f"Question at index {index} is empty.")
        items.append({"index": index, "qid": qid, "question": question, "answer": answer, "raw": raw})
    return items


def _slice_items(
    items: list[dict[str, Any]],
    *,
    start: int,
    end: int | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if start < 1:
        raise ValueError("--start must be >= 1.")
    if end is not None and end < start:
        raise ValueError("--end must be >= --start.")
    selected = items[start - 1 : end]
    return selected[:limit] if limit is not None else selected


def _dataset_name(questions_file: Path) -> str:
    if questions_file.name in {"questions.json", "question.json"}:
        return questions_file.parent.name
    return questions_file.stem


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _processed_result_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add(_result_key(str(payload.get("dataset") or ""), int(payload.get("index") or 0), payload.get("qid")))
    return keys


def _result_key(dataset: str, index: int, qid: Any) -> str:
    return f"{dataset}:{index}:{'' if qid is None else qid}"


if __name__ == "__main__":
    raise SystemExit(main())
