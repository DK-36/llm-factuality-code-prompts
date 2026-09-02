# Code Index

The public filenames are descriptive rather than numbered. Internal stage labels such as B1–B6 and the `d2` artifact namespace are retained only where they are part of the frozen experimental schema.

## Formal configurations

| File | Purpose |
|---|---|
| `retrieval_corpus_config.json` | Freezes the response-level split, URL normalization, guarded fetching, 384/64 passage construction, evidence alignment, and leakage policy. |
| `retrieval_evaluation_config.json` | Freezes BM25, Dense Retrieval, Hybrid RRF, evaluation depths, and development selection criteria. |
| `cove_experiment_config.json` | Freezes Standard CoVe B1–B6 prompts, schemas, model identities, decoding settings, split counts, and evaluation rules. |
| `cove_branch_experiment_config.json` | Defines shared branch inputs, Branch B and C interventions, and the deterministic Branch D candidate-selection precursor. |
| `cove_branch_d2_config.json` | Defines the bounded active Branch D revision stored in the frozen `d2` artifact namespace. |
| `cove_validation_config.json` | Defines paired uncertainty, exact-retained inheritance, blind cross-model checks, evidence tiers, and leakage controls. |
| `cove_validation_factuality_sampling_config.json` | Freezes deterministic sampling for the auxiliary factuality reliability check. |
| `cove_posthoc_analysis_config.json` | Defines the dissertation's diagnostic funnel, controlled evidence-stage effects, failure taxonomy, and separated evidence layers. |
| `cove_external_veriscore_config.json` | Pins the official VeriScore protocol, provider adapter, conditions, contrasts, and isolation boundary. |
| `cove_external_veriscore_k_sensitivity_config.json` | Freezes the K=1 to K=20 response-level sensitivity analysis after primary K=9. |

## Formal scripts

| File | Purpose |
|---|---|
| `prepare_factcheck_bench_claims.py` | Prepares canonical claim records, stable identifiers, response-level splits, and nested Study I and Study II cohorts. |
| `run_no_evidence_claim_verification.py` | Runs claim-only factuality verification and computes condition-level reports. |
| `run_benchmark_evidence_claim_verification.py` | Runs factuality verification with benchmark-associated evidence text while withholding labels and evidence metadata. |
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

The migration omits connectivity checks, synthetic smoke runs, schema-inspection utilities, historical annotation-pilot preparation, development-only exploratory top-k analysis, whole-dataset descriptive verifier summaries outside the reported held-out cohort, JSONL display helpers, all automated tests, all archived scripts, the superseded Branch D prompt and execution entry point, and miscellaneous utility modules not imported by the formal pipeline.
