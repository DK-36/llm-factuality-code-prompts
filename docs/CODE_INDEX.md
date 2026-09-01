# Code Index

The public filenames are descriptive rather than numbered. Internal stage labels such as B1–B6 and the `d2` artifact namespace are retained only where they are part of the frozen experimental schema.

## Formal scripts

| File | Purpose |
|---|---|
| `prepare_factcheck_bench_claims.py` | Prepares canonical claim records, stable identifiers, response-level splits, and nested Study I and Study II cohorts. |
| `run_no_evidence_claim_verification.py` | Runs claim-only factuality verification and computes condition-level reports. |
| `run_benchmark_evidence_claim_verification.py` | Runs factuality verification with benchmark-associated evidence text while withholding labels and evidence metadata. |
| `summarize_claim_verification.py` | Constructs the formal cross-condition Study I summary from frozen predictions. |
| `build_retrieval_corpus.py` | Builds retrieval splits, evidence and URL manifests, frozen documents, passages, qrels, and corpus audit summaries. |
| `evaluate_retrieval_methods.py` | Runs and compares BM25, Dense Retrieval, and Hybrid RRF under the two-level retrieval evaluation. |
| `run_retrieved_evidence_claim_verification.py` | Runs the frozen Hybrid retrieved-evidence verifier at K=1, K=3, or K=5 and performs the paired evidence-condition analysis. |
| `analyze_retrieval_depth.py` | Compares held-out retrieved-evidence verification across K=1, K=3, and K=5. |
| `run_standard_cove_and_post_revision_evaluation.py` | Implements Standard CoVe, diagnostic tracing, revised-claim extraction and alignment, evidence-grounded factuality reassessment, and the independent reliability protocol. |
| `run_controlled_cove_branches.py` | Implements Branch B evidence-supported answering, Branch C additional revision, and the active bounded Branch D targeted revision. |
| `validate_post_revision_evaluation.py` | Implements paired branch uncertainty, normalized-exact human-label inheritance, and blind cross-model checks. |
| `analyze_cove_diagnostic_traces.py` | Constructs the Standard-CoVe diagnostic groups, correction funnel, evidence-stage analysis, failure taxonomy, and separated evidence-layer synthesis. |
| `prepare_and_analyze_veriscore.py` | Exports final responses to the external protocol and validates and analyzes official VeriScore outputs. |
| `run_official_veriscore.py` | Runs the pinned official VeriScore environment and records protocol metadata without persisting credentials. |
| `analyze_veriscore_k_sensitivity.py` | Recomputes response-level VeriScore F1 for recall targets K=1 through K=20 from frozen response scores. |

## Shared modules

| File | Purpose |
|---|---|
| `factcheck_bench_pipeline.py` | Defines shared dataset paths, evidence normalization, frozen model identity, and cohort construction. |
| `factcheck_bench_analysis.py` | Implements binary verification metrics, paired transitions, response aggregation, and paired response-cluster bootstrap. |
| `factcheck_bench_corpus_fetch.py` | Implements guarded source-document fetching, redirect validation, content extraction, and frozen-document reprocessing. |
| `factcheck_bench_retrieval.py` | Implements corpus manifests, passage construction, qrels, and retrieval artifact utilities. |
| `factcheck_bench_retrieval_eval.py` | Implements BM25, dense encoding, Hybrid RRF, ranking, and two-level retrieval evaluation. |
| `factcheck_bench_cove.py` | Defines CoVe paths, response manifests, split validation, and branch-isolated artifact contracts. |

## Excluded code categories

The migration omits connectivity smoke tests, schema-inspection utilities, historical annotation-pilot preparation, development-only exploratory top-k analysis, JSONL display helpers, all automated tests, all archived scripts, the superseded Branch D prompt and execution entry point, and miscellaneous utility modules not imported by the formal pipeline.
