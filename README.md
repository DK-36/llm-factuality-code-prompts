# Evidence-Grounded LLM Factuality Verification and Revision

This repository is an anonymous static research artifact containing the formal source code and prompts that correspond to the methodology of a study on evidence-grounded claim verification and CoVe-based factual revision.

The repository is intended for code and prompt inspection only. It is not a runnable software distribution and does not contain datasets, model outputs, generated reports, numerical results, figures, caches, credentials, or local runtime state.

## Methodological scope

Study I compares claim-level factuality verification under No Evidence, Benchmark-associated Evidence, and Retrieved Evidence conditions while keeping the claim cohort, verifier, decoding settings, and output schema fixed. The shared retrieval pipeline constructs a benchmark-grounded passage corpus, compares BM25, Dense Retrieval, and Hybrid Reciprocal Rank Fusion on development data, freezes Hybrid RRF, and evaluates retrieved-evidence verification at passage depths K=1, K=3, and K=5.

Study II applies Standard Chain-of-Verification through verification-question generation, context-independent answering, and response revision. Separate diagnostic stages align questions to canonical claims and assess answer–claim pairs, while post-revision stages extract revised claims, align initial and revised content, and reassess revised-claim factuality with frozen retrieved evidence.

The controlled revision comparison contains Branch A (Standard CoVe), Branch B (retrieved evidence during independent answering), Branch C (an additional revision without evidence), and Branch D (a bounded targeted revision using retrieved evidence and claim-level diagnostic feedback). Reliability analysis includes normalized-exact human-label inheritance, blind cross-model semantic alignment, and independent passage-level factuality adjudication. External response-level evaluation is represented by the official VeriScore integration and its K=1 to K=20 recall-target sensitivity analysis.

## Repository contents

| Path | Contents |
|---|---|
| `scripts/` | Descriptively named implementations of the formal methodological stages |
| `src/` | Shared cohort, retrieval, analysis, corpus-fetching, and CoVe utilities |
| `prompts/` | The 17 formal model-visible and evaluation-only prompt templates |
| `vendor_patches/` | The provider adapter used by the pinned official VeriScore protocol |
| `docs/METHODOLOGY_MAP.md` | Direct mapping from dissertation methods to code and prompts |
| `docs/CODE_INDEX.md` | File-by-file description of the migrated implementation |
| `requirements.txt` | Dependency disclosure only; no execution instructions are provided |

## Deliberate exclusions

The artifact excludes all test code, smoke-test utilities, historical pilots, superseded Branch D execution code, development-only exploratory retrieval-depth code, record-inspection helpers, archived implementations, datasets, split manifests, qrels, model outputs, reports, result tables, figures, dissertation source files, local notes, environment files, and secrets.

## Reading order

Start with `docs/METHODOLOGY_MAP.md`, then use `docs/CODE_INDEX.md` to locate each formal implementation, and finally consult `prompts/README.md` for prompt visibility and stage boundaries.

## Interpretation boundary

The source files retain the original data and output path contracts because those paths document experimental separation and leakage controls. The corresponding artifacts are intentionally absent, so the repository must not be interpreted as a reproduction package or as evidence of any reported result.
