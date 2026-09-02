# Evidence-Grounded LLM Factuality Verification and Revision

This repository is an anonymous source-only reproduction package for the dissertation methodology. It contains the formal code, prompts, frozen method configurations, data contract, and stage order, but intentionally contains no benchmark copy, downloaded corpus, model output, numerical result, report, figure, cache, credential, or local runtime state.

## Reproducibility status

The complete method can be reconstructed from the official FactCheck-Bench file plus the external resources declared in [Data and Environment](docs/DATA_AND_ENVIRONMENT.md). FactCheck-Bench by itself is not sufficient: the retrieval conditions also require fetching the benchmark-linked source documents, the model-backed stages require the declared Ollama models, and the separate VeriScore extension requires its pinned upstream checkout and API-backed search and model services.

This package supports methodological reproduction and a fresh rerun of the pipeline. It does not promise bit-for-bit reproduction because benchmark-linked web pages can change or disappear, local inference runtimes can change, and the VeriScore extension depends on live external services. Exact reconstruction of the original retrieval corpus would additionally require the frozen source-document snapshot, which is deliberately not published here.

## Methodological scope

Study I compares claim-level factuality verification under No Evidence, Benchmark-associated Evidence, and Retrieved Evidence conditions while keeping the claim cohort, verifier, decoding settings, and output schema fixed. The shared retrieval pipeline constructs a benchmark-grounded passage corpus, compares BM25, Dense Retrieval, and Hybrid Reciprocal Rank Fusion on development data, freezes Hybrid RRF, and evaluates retrieved-evidence verification at K=1, K=3, and K=5.

Study II applies Standard Chain-of-Verification through verification-question generation, context-independent answering, and response revision. Separate diagnostic stages align questions to canonical claims and assess answer–claim pairs, while post-revision stages extract revised claims, align initial and revised content, and reassess revised-claim factuality with frozen Hybrid top-five evidence.

The controlled revision comparison contains Branch A (Standard CoVe), Branch B (retrieved evidence during independent answering), Branch C (one additional revision without evidence), and Branch D (one bounded targeted revision using retrieved evidence and claim-level diagnostic feedback). The frozen internal identifier `d2` denotes the dissertation's active Branch D implementation and is not a fifth branch.

Reliability analysis includes normalized-exact human-label inheritance, blind cross-model semantic alignment, and independent passage-level factuality adjudication. External response-level evaluation is represented by the pinned official VeriScore integration and its K=1 to K=20 recall-target sensitivity analysis.

## Repository contents

| Path | Contents |
|---|---|
| `scripts/` | Descriptively named implementations of the formal methodological stages |
| `src/` | Shared cohort, retrieval, analysis, corpus-fetching, and CoVe modules |
| `prompts/` | The 17 formal model-visible and evaluation-only prompt templates |
| `data/factcheck_bench/*/config/` | Frozen retrieval, CoVe, reliability, and VeriScore method configurations only |
| `vendor_patches/` | The provider adapter used by the pinned official VeriScore protocol |
| `docs/METHODOLOGY_MAP.md` | Direct mapping from dissertation methods to code, configuration, and prompts |
| `docs/REPRODUCTION_FLOW.md` | Ordered dependency graph and command-level stage sequence |
| `docs/DATA_AND_ENVIRONMENT.md` | Input placement, integrity identifiers, model identities, dependencies, and reproducibility limits |
| `docs/CODE_INDEX.md` | File-by-file description of the formal implementation |
| `.env.example` | Environment-variable names without credentials |
| `requirements.txt` | Python package versions recorded from the dissertation project environment |

## Deliberate exclusions

The package excludes all automated tests, connectivity checks, synthetic smoke runs, historical development pilots, superseded Branch D generation code, development-only exploratory retrieval-depth code, record-inspection helpers, archived implementations, datasets, split manifests, qrels, downloaded documents, embeddings, model outputs, reports, result tables, figures, dissertation source files, local notes, real environment files, and secrets.

## Reading order

Start with [Data and Environment](docs/DATA_AND_ENVIRONMENT.md), follow [Reproduction Flow](docs/REPRODUCTION_FLOW.md), use [Methodology-to-Code Map](docs/METHODOLOGY_MAP.md) to check dissertation correspondence, and consult [Prompt Catalogue](prompts/README.md) for model-visible boundaries.

## Result boundary

No reported result is stored or demonstrated here. Running the published stages would create ignored files under `data/` and `outputs/`; those generated files are outside the submitted source artifact and must not be treated as part of this repository's evidence.
