import asyncio
import json
import os
import re
import random
import sys
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union, List, Dict, Set, Tuple
from collections import Counter, defaultdict
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .hypermemory_prompt import (
    hyperedge_scoring_prompt,
    hypergraph_entity_pruning_prompt,
    hypergraph_prompt_evaluate,
)
from .llm import EmptyLLMResponseError


ENTITY_EXTRACTION_CACHE_FILE = "kv_store_entity_extraction_cache.jsonl"
ENTITY_EXTRACTION_STATE_FILE = "kv_store_entity_extraction_state.json"
CHECKPOINT_VERSION = 1


def entity_extraction_checkpoint_pending(working_dir: str, chunk_ids: list[str]) -> bool:
    state = _load_extraction_state(_extraction_state_path(working_dir))
    chunk_ids_hash = _chunk_ids_hash(chunk_ids)
    if state.get("chunk_ids_hash") == chunk_ids_hash:
        return not bool(state.get("merged"))

    cache_path = _extraction_cache_path(working_dir)
    return os.path.exists(cache_path) and os.path.getsize(cache_path) > 0


def _extraction_cache_path(working_dir: str) -> str:
    return os.path.join(working_dir, ENTITY_EXTRACTION_CACHE_FILE)


def _extraction_state_path(working_dir: str) -> str:
    return os.path.join(working_dir, ENTITY_EXTRACTION_STATE_FILE)


def _chunk_ids_hash(chunk_ids: list[str]) -> str:
    return compute_args_hash(sorted(chunk_ids))


def _load_extraction_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        logger.warning("Ignoring invalid entity extraction state file: %s", path)
        return {}


def _write_extraction_state(path: str, state: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_extraction_cache(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}

    cache: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid extraction checkpoint at %s:%s", path, line_number)
                continue

            chunk_id = record.get("chunk_id")
            if not chunk_id:
                continue
            cache[chunk_id] = {
                "nodes": record.get("nodes", {}),
                "edges": record.get("edges", {}),
            }
    return cache


def _append_extraction_cache(path: str, chunk_id: str, nodes: dict, edges: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"chunk_id": chunk_id, "nodes": nodes, "edges": edges},
                ensure_ascii=False,
            )
        )
        handle.write("\n")


def _graph_has_data(knowledge_graph_inst: BaseGraphStorage) -> bool:
    graph = getattr(knowledge_graph_inst, "_graph", None)
    if graph is None:
        return False
    return graph.number_of_nodes() > 0 or graph.number_of_edges() > 0


def _clear_graph_for_checkpoint_rebuild(knowledge_graph_inst: BaseGraphStorage) -> None:
    graph = getattr(knowledge_graph_inst, "_graph", None)
    if graph is None:
        logger.warning("Cannot clear graph storage for checkpoint rebuild; storage does not expose _graph.")
        return
    if graph.number_of_nodes() or graph.number_of_edges():
        logger.info(
            "Clearing partial graph before checkpoint rebuild: %s nodes, %s edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
    graph.clear()


def _find_empty_llm_response_error(exc: BaseException) -> Union[EmptyLLMResponseError, None]:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(current, EmptyLLMResponseError):
            return current

        last_attempt = getattr(current, "last_attempt", None)
        if last_attempt is not None:
            try:
                attempted_exc = last_attempt.exception()
            except Exception:
                attempted_exc = None
            if isinstance(attempted_exc, BaseException):
                pending.append(attempted_exc)

        cause = current.__cause__
        context = current.__context__
        if isinstance(cause, BaseException):
            pending.append(cause)
        if isinstance(context, BaseException):
            pending.append(context)
    return None


def _is_empty_llm_response_exception(exc: BaseException) -> bool:
    return _find_empty_llm_response_error(exc) is not None


def _empty_llm_response_reason(exc: BaseException) -> str:
    empty_error = _find_empty_llm_response_error(exc)
    if empty_error is None:
        return str(exc)
    finish_reason = getattr(empty_error, "finish_reason", None)
    if finish_reason:
        return f"finish_reason={finish_reason}"
    return str(empty_error)



def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    now_hyper_relation: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"entity"' or now_hyper_relation == "":
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 50.0
    )
    hyper_relation = now_hyper_relation
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        weight=weight,
        hyper_relation=hyper_relation,
        source_id=entity_source_id,
    )


async def _handle_single_hyperrelation_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 3 or record_attributes[0] != '"hyper-relation"':
        return None
    # add this record as edge
    knowledge_fragment = clean_str(record_attributes[1])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        hyper_relation="<hyperedge>"+knowledge_fragment,
        weight=weight,
        source_id=edge_source_id,
    )
    

async def _merge_hyperedges_then_upsert(
    hyperedge_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []

    already_hyperedge = await knowledge_graph_inst.get_node(hyperedge_name)
    if already_hyperedge is not None:
        already_weights.append(already_hyperedge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_hyperedge["source_id"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in nodes_data] + already_weights)
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    node_data = dict(
        role = "hyperedge",
        weight=weight,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        hyperedge_name,
        node_data=node_data,
    )
    node_data["hyperedge_name"] = hyperedge_name
    return node_data


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        role="entity",
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    edge_data = []
    
    for node in nodes_data:
        source_id = node["source_id"]
        hyper_relation = node["hyper_relation"]
        weight = node["weight"]
        
        already_weights = []
        already_source_ids = []
        
        if await knowledge_graph_inst.has_edge(hyper_relation, entity_name):
            already_edge = await knowledge_graph_inst.get_edge(hyper_relation, entity_name)
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
        
        weight = sum([weight] + already_weights)
        source_id = GRAPH_FIELD_SEP.join(
            set([source_id] + already_source_ids)
        )

        await knowledge_graph_inst.upsert_edge(
            hyper_relation,
            entity_name,
            edge_data=dict(
                role="link",
                weight=weight,
                source_id=source_id,
            ),
        )

        edge_data.append(dict(
            src_id=hyper_relation,
            tgt_id=entity_name,
            weight=weight,
        ))

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    hyperedge_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    max_concurrency = global_config["addon_params"].get("max_concurrency")
    chunk_semaphore = (
        asyncio.Semaphore(max_concurrency)
        if isinstance(max_concurrency, int) and max_concurrency > 0
        else None
    )
    resume_enabled = bool(global_config["addon_params"].get("resume"))

    ordered_chunks = list(chunks.items())
    ordered_chunk_ids = [chunk_id for chunk_id, _ in ordered_chunks]
    cached_results: dict[str, dict] = {}
    checkpoint_state: dict = {}
    checkpoint_matches = False
    cache_path = ""
    state_path = ""
    chunk_ids_hash = ""
    if resume_enabled:
        ordered_chunk_id_set = set(ordered_chunk_ids)
        working_dir = global_config["working_dir"]
        cache_path = _extraction_cache_path(working_dir)
        state_path = _extraction_state_path(working_dir)
        chunk_ids_hash = _chunk_ids_hash(ordered_chunk_ids)
        checkpoint_state = _load_extraction_state(state_path)
        checkpoint_matches = checkpoint_state.get("chunk_ids_hash") == chunk_ids_hash
        cached_results = {
            chunk_id: payload
            for chunk_id, payload in _load_extraction_cache(cache_path).items()
            if chunk_id in ordered_chunk_id_set
        }
        if (
            checkpoint_matches
            and checkpoint_state.get("merged")
            and len(cached_results) >= len(ordered_chunks)
            and _graph_has_data(knowledge_graph_inst)
        ):
            logger.info(
                "Entity extraction checkpoint already merged; skipping %s chunks.",
                len(ordered_chunks),
            )
            return knowledge_graph_inst

        if cached_results:
            logger.info(
                "Loaded entity extraction checkpoint with %s/%s completed chunks.",
                len(cached_results),
                len(ordered_chunks),
            )

    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = len(cached_results)
    already_entities = sum(len(result.get("nodes", {})) for result in cached_results.values())
    already_relations = sum(len(result.get("edges", {})) for result in cached_results.values())

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        def _finish_chunk(
            maybe_nodes: dict[str, list[dict]],
            maybe_edges: dict[str, list[dict]],
        ):
            nonlocal already_processed, already_entities, already_relations
            nodes = dict(maybe_nodes)
            edges = dict(maybe_edges)
            already_processed += 1
            already_entities += len(nodes)
            already_relations += len(edges)
            now_ticks = PROMPTS["process_tickers"][
                already_processed % len(PROMPTS["process_tickers"])
            ]
            stdout_encoding = sys.stdout.encoding or "utf-8"
            try:
                now_ticks.encode(stdout_encoding)
            except UnicodeEncodeError:
                now_ticks = "*"
            print(
                f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
                end="",
                flush=True,
            )
            return chunk_key, nodes, edges

        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        try:
            final_result = await use_llm_func(hint_prompt)
        except Exception as exc:
            if _is_empty_llm_response_exception(exc):
                logger.warning(
                    "Skipping chunk %s because initial entity extraction returned empty LLM content (%s).",
                    chunk_key,
                    _empty_llm_response_reason(exc),
                )
                return _finish_chunk({}, {})
            raise

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            try:
                glean_result = await use_llm_func(continue_prompt, history_messages=history)
            except Exception as exc:
                if _is_empty_llm_response_exception(exc):
                    logger.warning(
                        "Stopping entity gleaning for chunk %s at pass %s because continue extraction returned empty LLM content (%s).",
                        chunk_key,
                        now_glean_index + 1,
                        _empty_llm_response_reason(exc),
                    )
                    break
                raise

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            try:
                if_loop_result: str = await use_llm_func(
                    if_loop_prompt, history_messages=history
                )
            except Exception as exc:
                if _is_empty_llm_response_exception(exc):
                    logger.warning(
                        "Stopping entity gleaning for chunk %s at pass %s because loop check returned empty LLM content (%s).",
                        chunk_key,
                        now_glean_index + 1,
                        _empty_llm_response_reason(exc),
                    )
                    break
                raise
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        now_hyper_relation=""
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_relation = await _handle_single_hyperrelation_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[if_relation["hyper_relation"]].append(
                    if_relation
                )
                now_hyper_relation = if_relation["hyper_relation"]
                
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, now_hyper_relation
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

        return _finish_chunk(maybe_nodes, maybe_edges)

    async def _bounded_process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        if chunk_semaphore is None:
            return await _process_single_content(chunk_key_dp)
        async with chunk_semaphore:
            return await _process_single_content(chunk_key_dp)

    missing_chunks = [chunk for chunk in ordered_chunks if chunk[0] not in cached_results]
    if resume_enabled:
        _write_extraction_state(
            state_path,
            {
                "version": CHECKPOINT_VERSION,
                "chunk_count": len(ordered_chunks),
                "chunk_ids_hash": chunk_ids_hash,
                "extracted_count": len(cached_results),
                "extraction_complete": False,
                "merged": False,
            },
        )
    if missing_chunks and resume_enabled:
        logger.info(
            "Extracting %s chunks; %s chunks will be reused from checkpoint.",
            len(missing_chunks),
            len(cached_results),
        )
    elif missing_chunks:
        logger.info("Extracting %s chunks.", len(missing_chunks))
    else:
        logger.info("All %s chunks will be reused from extraction checkpoint.", len(ordered_chunks))

    for result in tqdm_async(
        asyncio.as_completed([_bounded_process_single_content(c) for c in missing_chunks]),
        total=len(missing_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        chunk_id, m_nodes, m_edges = await result
        cached_results[chunk_id] = {"nodes": m_nodes, "edges": m_edges}
        if resume_enabled:
            _append_extraction_cache(cache_path, chunk_id, m_nodes, m_edges)
        if resume_enabled and (len(cached_results) % 25 == 0 or len(cached_results) == len(ordered_chunks)):
            _write_extraction_state(
                state_path,
                {
                    "version": CHECKPOINT_VERSION,
                    "chunk_count": len(ordered_chunks),
                    "chunk_ids_hash": chunk_ids_hash,
                    "extracted_count": len(cached_results),
                    "extraction_complete": False,
                    "merged": False,
                },
            )

    if len(cached_results) < len(ordered_chunks):
        raise RuntimeError(
            f"Entity extraction checkpoint is incomplete: {len(cached_results)}/{len(ordered_chunks)} chunks."
        )

    if resume_enabled:
        _write_extraction_state(
            state_path,
            {
                "version": CHECKPOINT_VERSION,
                "chunk_count": len(ordered_chunks),
                "chunk_ids_hash": chunk_ids_hash,
                "extracted_count": len(cached_results),
                "extraction_complete": True,
                "merged": False,
            },
        )
    if resume_enabled and checkpoint_matches and not checkpoint_state.get("merged") and _graph_has_data(knowledge_graph_inst):
        _clear_graph_for_checkpoint_rebuild(knowledge_graph_inst)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for chunk_id in ordered_chunk_ids:
        cached_result = cached_results[chunk_id]
        m_nodes = cached_result.get("nodes", {})
        m_edges = cached_result.get("edges", {})
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
            
    logger.info("Inserting hyperedges into storage...")
    all_hyperedges_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_hyperedges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Inserting hyperedges",
        unit="entity",
    ):
        all_hyperedges_data.append(await result)
            
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    if not len(all_hyperedges_data) and not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any hyperedges and entities, maybe your LLM is not working"
        )
        return None

    if not len(all_hyperedges_data):
        logger.warning("Didn't extract any hyperedges")
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if hyperedge_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["hyperedge_name"], prefix="rel-"): {
                "content": dp["hyperedge_name"],
                "hyperedge_name": dp["hyperedge_name"],
            }
            for dp in all_hyperedges_data
        }
        await hyperedge_vdb.upsert(data_for_vdb)

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if resume_enabled:
        _write_extraction_state(
            state_path,
            {
                "version": CHECKPOINT_VERSION,
                "chunk_count": len(ordered_chunks),
                "chunk_ids_hash": chunk_ids_hash,
                "extracted_count": len(cached_results),
                "extraction_complete": True,
                "merged": True,
            },
        )

    return knowledge_graph_inst


async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if cached_response is not None:
        return cached_response
    
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        # entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )
    
    hint_prompt = entity_extract_prompt.format(
        **context_base, input_text="{input_text}"
    ).format(**context_base, input_text=query)

    final_result = await use_model_func(hint_prompt)

    logger.info("kw_prompt result:")
    print(final_result)
    hl_keywords, ll_keywords = [], []
    try:
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if len(record_attributes) == 3 and record_attributes[0] == '"hyper-relation"':
                hl_keywords.append("<hyperedge>"+clean_str(record_attributes[1]))
            elif len(record_attributes) == 5 and record_attributes[0] == '"entity"':
                ll_keywords.append(clean_str(record_attributes[1]).upper())
            else:
                continue
    # Handle parsing sample
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e} {final_result}")
        return PROMPTS["fail_response"]

    # Handdle keywords missing
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        # Let model handle it, it might still work if user provides entities manually
    if ll_keywords == []:
        logger.warning("low_level_keywords is empty, model might not work well.")
    
    ll_keywords = ", ".join(ll_keywords)
    hl_keywords = ", ".join(hl_keywords)

    # Build context using Think-on-Graph
    keywords = [ll_keywords, hl_keywords]
    context = await _build_query_context(
        query,  # Pass the full query
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        hyperedges_vdb,
        text_chunks_db,
        query_param,
        global_config,
    )

    if query_param.only_need_context:
        return context
    if not context or not context.strip():
        logger.warning("Context is empty after Traversal.")
        return PROMPTS["fail_response"]
        
    sys_prompt_temp = PROMPTS["rag_response_entities"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
    )
    if query_param.only_need_prompt:
        return sys_prompt
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return response


async def _parse_hyperedge_scoring(llm_output: str, width: int) -> List[Tuple[str, float]]:
    """Parses the LLM output for hyperedge scoring."""
    try:
        # Regex to find patterns like {2. "hyperedge text" (Score: 0.70)}
        pattern = re.compile(r'{\d+\.\s*"(.*?)"\s*\(Score:\s*([\d\.]+)\)\}', re.DOTALL)
        matches = pattern.findall(llm_output)
        
        scored_edges = []
        for match in matches:
            hyperedge_text = match[0].strip()
            score = float(match[1])
            scored_edges.append((f"<hyperedge>\"{hyperedge_text}\"", score))
            
        scored_edges.sort(key=lambda x: x[1], reverse=True)
        return scored_edges[:width]
    except Exception as e:
        logger.error(f"Error parsing hyperedge scoring output: {e}\nOutput: {llm_output}")
        return []

async def _build_query_context(
    query_full: str,
    query_keywords: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    hyperedges_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    width = 3
    depth = 3
    use_llm_func = global_config["llm_model_func"]
    tiktoken_model_name = global_config.get("tiktoken_model_name", "gpt-4o-mini")
    MAX_INPUT_TOKENS = 127000

    topic_entities_str = query_keywords[0]
    if not topic_entities_str:
        logger.warning("No topic entities found in query, falling back to empty context.")
        return ""

    topic_entities = [e.strip().upper() for e in topic_entities_str.split(',')]
    
    logger.info(f"Starting Think-on-Graph traversal with entities: {topic_entities}")

    # --- State variables ---
    frontier_entities: Set[str] = set(topic_entities)
    collected_entities: Dict[str, Dict] = {}
    collected_hyperedges: Dict[str, Dict] = {}
    visited_entities: Set[str] = set()
    visited_hyperedges: Set[str] = set()

    # Collect initial entities data
    for entity_name in frontier_entities:
        if entity_name not in collected_entities:
            node_data = await knowledge_graph_inst.get_node(entity_name)
            if node_data:
                collected_entities[entity_name] = node_data

    for d in range(depth):
        logger.info(f"--- Depth: {d+1}/{depth} ---")
        if not frontier_entities:
            logger.info("No more entities to explore. Stopping traversal.")
            break

        # List to store all generated paths (score, start_entity, hyperedge, end_entity) for this depth
        all_candidate_paths_this_depth = []

        # --- 1. & 2. Per-Entity Relation/Entity Search and Scoring ---
        async def process_frontier_entity(start_entity: str):
            """Processes one entity from the frontier to find candidate paths."""
            # 1. Relation Search & Prune (for this start_entity)
            entity_edge_list = await knowledge_graph_inst.get_node_edges(start_entity)
            if not entity_edge_list:
                return []
            
            candidate_hyperedges = {src for _, src in entity_edge_list if src not in visited_hyperedges}
            if not candidate_hyperedges:
                return []

            hyperedges_text_list = [h.replace('<hyperedge>', '') for h in candidate_hyperedges]
            prompt_template_for_scoring = hyperedge_scoring_prompt % (width, width)
            
            base_prompt_for_scoring = prompt_template_for_scoring.format(
                query=query_full,
                topic_entity=start_entity,
                hyperedges="{hyperedges_placeholder}"
            )
            base_tokens = len(encode_string_by_tiktoken(base_prompt_for_scoring.replace("{hyperedges_placeholder}", ""), model_name=tiktoken_model_name))
            available_tokens_for_hyperedges = MAX_INPUT_TOKENS - base_tokens

            hyperedges_for_prompt = [f'{i+1}. {h}' for i, h in enumerate(hyperedges_text_list)]
            
            truncated_hyperedges_list = truncate_list_by_token_size(
                hyperedges_for_prompt,
                lambda x: x,
                available_tokens_for_hyperedges
            )
            if len(truncated_hyperedges_list) < len(hyperedges_for_prompt):
                logger.warning(f"Truncating hyperedges for scoring prompt. Original: {len(hyperedges_for_prompt)}, Truncated: {len(truncated_hyperedges_list)}")

            formatted_prompt = prompt_template_for_scoring.format(
                query=query_full,
                topic_entity=start_entity,
                hyperedges="\n".join(truncated_hyperedges_list)
            )
            llm_response = await use_llm_func(formatted_prompt)
            selected_hyperedges_with_scores = await _parse_hyperedge_scoring(llm_response, width)

            if not selected_hyperedges_with_scores:
                return []

            paths_from_entity = []
            # 2. Entity Search & Score (for each selected entity-relation pair)
            for hyperedge_name, he_score in selected_hyperedges_with_scores:
                connected_entity_edges = await knowledge_graph_inst.get_node_edges(hyperedge_name)
                if not connected_entity_edges:
                    continue
                
                entities_in_edge = [
                    tgt for _, tgt in connected_entity_edges 
                    if tgt != start_entity and tgt not in visited_entities.union(frontier_entities)
                ]
                if not entities_in_edge:
                    continue
                
                # Score these candidate entities
                pruning_prompt_template = hypergraph_entity_pruning_prompt
                base_pruning_prompt = pruning_prompt_template.format(
                    query=query_full,
                    hyperedge=hyperedge_name.replace('<hyperedge>', '')[1:-1],
                    entities="{entities_placeholder}"
                )
                base_tokens = len(encode_string_by_tiktoken(base_pruning_prompt.replace("{entities_placeholder}", ""), model_name=tiktoken_model_name))
                available_tokens_for_entities = MAX_INPUT_TOKENS - base_tokens

                entities_for_prompt_strings = [entity[1:-1] for entity in entities_in_edge]
                truncated_entities_for_prompt = truncate_list_by_token_size(
                    entities_for_prompt_strings,
                    lambda x: x,
                    available_tokens_for_entities
                )

                if len(truncated_entities_for_prompt) < len(entities_for_prompt_strings):
                    logger.warning(f"Truncating entities for pruning prompt. Original: {len(entities_for_prompt_strings)}, Truncated: {len(truncated_entities_for_prompt)}")
                    truncated_entities_in_edge = []
                    truncated_set = set(truncated_entities_for_prompt)
                    for entity in entities_in_edge:
                        if entity[1:-1] in truncated_set:
                            truncated_entities_in_edge.append(entity)
                    entities_in_edge = truncated_entities_in_edge
                
                pruning_prompt = pruning_prompt_template.format(
                    query=query_full,
                    hyperedge=hyperedge_name.replace('<hyperedge>', '')[1:-1],
                    entities="; ".join([entity[1:-1] for entity in entities_in_edge])
                )
                llm_pruning_response = await use_llm_func(pruning_prompt)
                
                try:
                    score_line_match = re.search(r"Score:\s*([\d\.,\s]+)", llm_pruning_response)
                    if not score_line_match:
                        pruned_entities_with_scores = [(e, 1.0) for e in entities_in_edge]
                    else:
                        scores_str = score_line_match.group(1).split(',')
                        scores = [float(s.strip()) for s in scores_str]
                        if len(scores) != len(entities_in_edge):
                            scored_entities = sorted(zip(entities_in_edge[:len(scores)], scores), key=lambda x: x[1], reverse=True)
                        else:
                            scored_entities = sorted(zip(entities_in_edge, scores), key=lambda x: x[1], reverse=True)
                        pruned_entities_with_scores = scored_entities
                except Exception as e:
                    logger.error(f"Error parsing entity pruning scores: {e}")
                    pruned_entities_with_scores = [(e, 1.0) for e in entities_in_edge]

                for entity_name, e_score in pruned_entities_with_scores:
                    path_score = he_score * e_score
                    paths_from_entity.append((path_score, start_entity, hyperedge_name, entity_name))
            
            return paths_from_entity

        tasks = [process_frontier_entity(entity) for entity in frontier_entities]
        results_from_tasks = await asyncio.gather(*tasks)

        for paths in results_from_tasks:
            all_candidate_paths_this_depth.extend(paths)
        
        visited_entities.update(frontier_entities)

        if not all_candidate_paths_this_depth:
            logger.info("No new paths found in this depth. Stopping traversal.")
            break

        # --- 3. Global Pruning ---
        all_candidate_paths_this_depth.sort(key=lambda x: x[0], reverse=True)
        top_paths = all_candidate_paths_this_depth[:width]
        
        logger.info(f"Selected {len(top_paths)} entity paths for next depth.")

        if not top_paths:
            break

        # --- Update state and collect context for this depth ---
        next_frontier_entities = set()
        
        for _, start_entity, hyperedge_name, end_entity in top_paths:
            next_frontier_entities.add(end_entity)
            visited_hyperedges.add(hyperedge_name)

            if hyperedge_name not in collected_hyperedges:
                hyperedge_data = await knowledge_graph_inst.get_node(hyperedge_name)
                if hyperedge_data:
                    collected_hyperedges[hyperedge_name] = hyperedge_data
            
            if end_entity not in collected_entities:
                node_data = await knowledge_graph_inst.get_node(end_entity)
                if node_data:
                    collected_entities[end_entity] = node_data
            
            if hyperedge_name in collected_hyperedges:
                if 'related_entities' not in collected_hyperedges[hyperedge_name]:
                    collected_hyperedges[hyperedge_name]['related_entities'] = set()
                elif isinstance(collected_hyperedges[hyperedge_name]['related_entities'], str):
                    collected_hyperedges[hyperedge_name]['related_entities'] = {collected_hyperedges[hyperedge_name]['related_entities']}
                collected_hyperedges[hyperedge_name]['related_entities'].add(end_entity)

        # --- Stage 5: Evaluate if context is sufficient ---
        context_for_this_depth = []
        temp_related_entities = defaultdict(list)
        for _, _, he, entity in top_paths:
            temp_related_entities[he].append(entity)

        for he, entities in temp_related_entities.items():
             context_for_this_depth.append(f'{len(context_for_this_depth) + 1}. {he.replace("<hyperedge>", "")}\n   Connected Entities: [{", ".join([p_entity[1:-1] for p_entity in entities])}]')

        if not context_for_this_depth:
            logger.info("No new context was generated in this depth level.")
        else:
            eval_prompt_template = hypergraph_prompt_evaluate
            base_eval_prompt = eval_prompt_template.format(
                query=query_full,
                entity_descriptions="{entity_descriptions_placeholder}",
                hyperedges="{hyperedges_placeholder}"
            )
            base_tokens = len(encode_string_by_tiktoken(base_eval_prompt.replace("{entity_descriptions_placeholder}", "").replace("{hyperedges_placeholder}", ""), model_name=tiktoken_model_name))
            available_tokens = MAX_INPUT_TOKENS - base_tokens
            
            hyperedges_for_eval_list = context_for_this_depth
            
            truncated_hyperedges_for_eval_list = truncate_list_by_token_size(
                hyperedges_for_eval_list,
                lambda x: x,
                available_tokens
            )
            if len(truncated_hyperedges_for_eval_list) < len(hyperedges_for_eval_list):
                logger.warning(f"Truncating hyperedges for evaluation prompt. Original: {len(hyperedges_for_eval_list)}, Truncated: {len(truncated_hyperedges_for_eval_list)}")

            hyperedges_for_eval = "\n".join(truncated_hyperedges_for_eval_list)
            
            tokens_used_by_hyperedges = len(encode_string_by_tiktoken(hyperedges_for_eval, model_name=tiktoken_model_name))
            available_tokens_for_entities = available_tokens - tokens_used_by_hyperedges
            
            entity_descriptions_list = [f"{name}: {data.get('description', 'N/A')}" for name, data in collected_entities.items()]
            
            truncated_entity_descriptions_list = truncate_list_by_token_size(
                entity_descriptions_list,
                lambda x: x,
                available_tokens_for_entities
            )
            if len(truncated_entity_descriptions_list) < len(entity_descriptions_list):
                logger.warning(f"Truncating entity descriptions for evaluation prompt. Original: {len(entity_descriptions_list)}, Truncated: {len(truncated_entity_descriptions_list)}")

            entity_descriptions_for_eval = "\n".join(truncated_entity_descriptions_list)
            
            eval_prompt = hypergraph_prompt_evaluate.format(
                query=query_full,
                entity_descriptions=entity_descriptions_for_eval,
                hyperedges=hyperedges_for_eval
            )
            llm_eval_response = await use_llm_func(eval_prompt)
            logger.info(f"Sufficiency evaluation response: {llm_eval_response[:100]}...")
            if re.search(r'\{Yes\}', llm_eval_response, re.IGNORECASE):
                logger.info("Information is sufficient. Stopping traversal.")
                break

        frontier_entities = next_frontier_entities

    # --- 6. Format final context ---
    logger.info(f"Traversal finished. Collected {len(collected_entities)} entities and {len(collected_hyperedges)} hyperedges.")
    
    if not collected_entities and not collected_hyperedges:
        return ""
    
    # Finalize related_entities format
    for he_name in collected_hyperedges:
        if 'related_entities' in collected_hyperedges[he_name] and isinstance(collected_hyperedges[he_name]['related_entities'], set):
            collected_hyperedges[he_name]['related_entities'] = '|'.join(list(collected_hyperedges[he_name]['related_entities']))

    collected_entities = {
        name[1:-1]: {
            "role": data["role"],
            "entity_type": data["entity_type"][1:-1],
            "description": '<SEP>'.join([desc[1:-1] for desc in data["description"].split("<SEP>")]),
            "source_id": data["source_id"],
        }
        for name, data in collected_entities.items()
    }
    collected_hyperedges = {
        ''.join(['<hyperedge>', name.split('<hyperedge>')[1][1:-1]]): {
            "role": data["role"],
            "weight": data["weight"],
            "source_id": data["source_id"],
            "related_entities": '|'.join([rel_entity[1:-1] for rel_entity in data.get("related_entities", "").split('|')]),
        }
        for name, data in collected_hyperedges.items()
    }

    entities_section_list = [["id", "entity", "type", "description"]]
    entity_map = {name: i for i, name in enumerate(collected_entities.keys())}
    for name, data in collected_entities.items():
        entities_section_list.append(
            [
                entity_map[name],
                name,
                data.get("entity_type", "UNKNOWN"),
                data.get("description", "UNKNOWN"),
            ]
        )
    entities_context = list_of_list_to_csv(entities_section_list)

    relations_section_list = [["id", "hyperedge", "related_entities"]]
    for i, (name, data) in enumerate(collected_hyperedges.items()):
        relations_section_list.append(
            [
                i,
                name,
                data.get("related_entities", "")
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    source_ids = set()
    for data in collected_entities.values():
        source_ids.update(split_string_by_multi_markers(data.get("source_id", ""), [GRAPH_FIELD_SEP]))
    for data in collected_hyperedges.values():
        source_ids.update(split_string_by_multi_markers(data.get("source_id", ""), [GRAPH_FIELD_SEP]))
    
    text_units_data = [await text_chunks_db.get_by_id(sid) for sid in source_ids if sid]
    text_units_data = [t for t in text_units_data if t]

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(text_units_data):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    final_context = f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    # Get the tiktoken model name from global_config, default to 'gpt-4o-mini'
    tiktoken_model_name = global_config.get("tiktoken_model_name", "gpt-4o-mini")

    # Define the maximum allowed tokens for the context
    MAX_CONTEXT_TOKENS = 127000

    # Encode the context to tokens
    context_tokens = encode_string_by_tiktoken(final_context, model_name=tiktoken_model_name)

    # Check if truncation is needed
    if len(context_tokens) > MAX_CONTEXT_TOKENS:
        logger.warning(f"Context length ({len(context_tokens)} tokens) exceeds MAX_CONTEXT_TOKENS ({MAX_CONTEXT_TOKENS}). Truncating context.")
        # Truncate the tokens
        truncated_tokens = context_tokens[:MAX_CONTEXT_TOKENS]
        # Decode back to string
        final_context = decode_tokens_by_tiktoken(truncated_tokens, model_name=tiktoken_model_name)

    return final_context
