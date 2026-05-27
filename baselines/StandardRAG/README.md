# Standard RAG Baseline

Flat dense RAG baseline using `text-embedding-3-small` for question/chunk embeddings and `gpt-4o-mini` for answer generation.

It only retrieves raw chunks/passages from `vdb_chunks.json` or a newly built chunk index. It does not use entities, relations, hyperedges, hypergraphs, path search, or question decomposition.

Check or build the reusable chunk index:

```powershell
python scripts/build_standard_rag_index.py --dataset 2wikimultihopqa
```

## 2WikiMultiHopQA Example

```powershell
$env:OPENAI_API_KEY = "sk-..."
python scripts/run_standard_rag.py `
  --dataset 2wikimultihopqa `
  --question-file questions/2wikimultihopqa/hyperrag_query_test.json `
  --top-k 5 `
  --limit 100 `
  --runs-dir runs/StandardRAG/2wikimultihopqa_hyperrag_query_test `
  --output-path runs/StandardRAG/2wikimultihopqa_hyperrag_query_test/generated_answer.json
```

Evaluate with the existing script:

```powershell
python eval/get_score.py `
  --question-file questions/2wikimultihopqa/hyperrag_query_test.json `
  --runs-dir runs/StandardRAG/2wikimultihopqa_hyperrag_query_test `
  --limit 100 `
  --skip-rsim `
  --skip-gen `
  --output-dir eval/results/2wikimultihopqa/standard_rag
```
