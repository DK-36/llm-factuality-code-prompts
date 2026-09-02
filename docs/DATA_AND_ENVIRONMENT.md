# Data and Environment Contract

## Required benchmark input

Download `factcheck-GPT-benchmark.jsonl` from the [official FactCheck-GPT repository](https://github.com/yuxiaw/Factcheck-GPT/blob/main/factcheck-GPT-benchmark.jsonl) and place it at `data/factcheck_bench/raw/factcheck-GPT-benchmark.jsonl`.

The dissertation project used a 94-record JSONL file with SHA-256 `8bbf41a58ee1b7431fecd2a23f231f764e378e04edeb4d2a7a83949a7f40cf4b`. Verify the downloaded file against both identifiers before constructing claims. If the upstream file changes, preserve the downloaded version separately and record its hash; do not silently substitute a different benchmark revision.

The benchmark is described by Wang et al. in the [official ACL Anthology paper](https://aclanthology.org/2024.findings-emnlp.830/). This repository does not redistribute the dataset.

## Additional inputs created by the pipeline

Claim preparation creates the processed claim table and cohort manifest. Corpus preparation derives URL manifests from the benchmark, fetches benchmark-linked source documents, extracts text, chunks the text into 384-token passages with 64-token overlap, and creates evaluation-only qrels. Retrieval evaluation creates BM25, Dense, and Hybrid RRF rankings. All of these generated or downloaded files are ignored by Git.

The benchmark-linked source pages are live web resources. A fresh fetch can differ from the dissertation snapshot because a page may be edited, redirected, removed, blocked, or served differently; therefore the official benchmark enables a fresh methodological rerun but cannot by itself guarantee exact reconstruction of the original corpus.

## Python and package environment

The recorded project environment used Python 3.13.11. The directly imported project dependencies are pinned in `requirements.txt`: `ollama==0.6.2`, `python-dotenv==1.2.2`, `pypdf==6.14.2`, `numpy==2.5.1`, and `Pillow==12.3.0`.

The `requirements.txt` file covers this repository's Python code only. An Ollama service and the required local models must be installed separately. The code validates frozen model identities before formal writes where the original implementation recorded a digest.

## Frozen local-model identities

| Role | Model | Frozen identity |
|---|---|---|
| Primary verifier, CoVe generation, and same-family evaluation | `qwen3:8b` | SHA-256 digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41` |
| Dense passage and query encoding | `nomic-embed-text:v1.5` | Expected Ollama digest prefix `0a109f422b47` |
| Blind cross-family reliability adjudication | `llama3.1:8b` | SHA-256 digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` |

`CHAT_MODEL` and `OLLAMA_HOST` may be copied from `.env.example`. A real `.env` file is ignored and must never be committed.

## External VeriScore environment

The response-level extension uses official VeriScore 2.0.2 at commit `8714bca27b944b9659d6a966cdb92fb6fff8f72d`, plus `vendor_patches/veriscore_2.0.2_deepseek.patch`. Its frozen configuration is `data/factcheck_bench/cove/config/cove_external_veriscore_config.json`.

The launcher reads `DEEPSEEK_API_KEY` and `SERPER_API_KEY` from the environment and never writes their values into project metadata. This extension uses live model and search services, so even with the pinned local adapter its results are not guaranteed to be bit-for-bit deterministic.

## Input and output boundary

Only JSON method configurations are tracked below `data/`. Raw benchmark data, processed records, URL and split manifests, downloaded source documents, passages, qrels, dense embeddings, and retrieval runs remain ignored. Every experimental result, summary, Markdown report, CSV, and figure remains ignored under `outputs/`.

The repository contains no result fixture and no expected-output snapshot. Completion is established by each stage's schema, count, hash, leakage, and frozen-configuration checks rather than by comparing against a bundled result.
