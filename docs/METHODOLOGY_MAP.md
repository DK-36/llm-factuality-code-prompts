# Methodology-to-Code Map

This map follows the final dissertation methodology and prompt catalogue. It includes only stages that implement the reported data preparation, controlled verification conditions, retrieval protocol, CoVe revision design, post-revision evaluation, reliability analysis, or external response-level evaluation.

The executable order and input requirements are specified separately in `docs/REPRODUCTION_FLOW.md` and `docs/DATA_AND_ENVIRONMENT.md`. Frozen method choices are tracked under `data/factcheck_bench/retrieval/config/` and `data/factcheck_bench/cove/config/`.

## Shared data and retrieval foundation

| Methodological component | Implementation | Supporting module |
|---|---|---|
| FactCheck-Bench claim preparation, stable identifiers, human-label retention, and nested cohort construction | `scripts/prepare_factcheck_bench_claims.py` | `src/factcheck_bench_pipeline.py` |
| Source-URL canonicalisation, safe document collection, text extraction, 384-token passages, and 64-token overlap | `scripts/build_retrieval_corpus.py` | `src/factcheck_bench_corpus_fetch.py`, `src/factcheck_bench_retrieval.py` |
| Claim-only retrieval queries and source-document or strict-passage evaluation layers | `scripts/evaluate_retrieval_methods.py` | `src/factcheck_bench_retrieval_eval.py` |
| BM25 with fixed k1 and b, Dense Retrieval with nomic-embed-text:v1.5, and unweighted Hybrid RRF | `scripts/evaluate_retrieval_methods.py` | `src/factcheck_bench_retrieval_eval.py` |

Gold labels, benchmark evidence text, evidence stance, source mappings, and qrels are retained only where required for evaluation or corpus auditing and are excluded from retrieval ranking inputs.

## Study I: claim-level verification

| Condition or analysis | Implementation | Prompt |
|---|---|---|
| No Evidence | `scripts/run_no_evidence_claim_verification.py` | `prompts/no_evidence_verifier.txt` |
| Benchmark-associated Evidence | `scripts/run_benchmark_evidence_claim_verification.py` | `prompts/oracle_evidence_verifier.txt` |
| Retrieved Evidence with frozen Hybrid RRF | `scripts/run_retrieved_evidence_claim_verification.py` | `prompts/retrieved_evidence_verifier.txt` |
| Retrieved-output format normalization | `scripts/run_retrieved_evidence_claim_verification.py` | `prompts/retrieved_evidence_output_repair.txt` |
| Retrieved-evidence depths K=1, K=3, and K=5 | `scripts/run_retrieved_evidence_claim_verification.py`, `scripts/analyze_retrieval_depth.py` | `prompts/retrieved_evidence_verifier.txt` with only the declared passage count changed |
| Balanced Accuracy, Macro-F1, Overall Accuracy, abstention, paired transitions, and paired response-cluster bootstrap | `scripts/run_retrieved_evidence_claim_verification.py`, `scripts/analyze_retrieval_depth.py` | None |

Every eligible claim is evaluated independently, and outputs from one evidence condition are not supplied to another condition.

## Study II: Standard CoVe and diagnostic tracing

| Stage | Methodological role | Implementation | Prompt |
|---|---|---|---|
| B1 | Generate verification questions from the original question and initial response | `scripts/run_standard_cove_and_post_revision_evaluation.py` | `prompts/cove_question_planning.txt` |
| B2 | Align generated questions with label-free canonical claims for diagnostic coverage | `scripts/run_standard_cove_and_post_revision_evaluation.py` | `prompts/cove_question_claim_alignment.txt` |
| B3 | Answer each verification question independently without the initial response or other questions | `scripts/run_standard_cove_and_post_revision_evaluation.py` | `prompts/cove_independent_verification_answer.txt` |
| B4 | Evaluate each aligned answer–claim pair against benchmark-associated evidence with human labels withheld | `scripts/run_standard_cove_and_post_revision_evaluation.py` | `prompts/cove_answer_claim_evaluation.txt` |
| B5 | Revise the complete response from the initial response and B3 question–answer pairs | `scripts/run_standard_cove_and_post_revision_evaluation.py` | `prompts/cove_response_revision.txt` |
| B6a | Extract atomic factual claims from the revised response | `scripts/run_standard_cove_and_post_revision_evaluation.py` | `prompts/cove_revised_claim_extraction.txt` |
| B6b | Align canonical initial claims with revised content without judging factuality | `scripts/run_standard_cove_and_post_revision_evaluation.py` | `prompts/cove_gold_revised_claim_alignment.txt` |
| B6c | Reassess revised-claim factuality using frozen Hybrid top-five passages | `scripts/run_standard_cove_and_post_revision_evaluation.py` | `prompts/cove_revised_claim_factuality.txt` |

B2 and B4 are diagnostic-only stages and do not affect the Standard CoVe generation path. Post-revision states remain silver unless strengthened by the separately defined reliability layers.

## Controlled revision interventions

| Branch | Intervention | Implementation | Prompt |
|---|---|---|---|
| A | Standard CoVe without retrieved evidence in planning, independent answering, or revision | `scripts/run_standard_cove_and_post_revision_evaluation.py` | Standard CoVe prompts above |
| B | Answer the frozen Branch A verification questions with retrieved passages, then apply the unchanged standard revision step | `scripts/run_controlled_cove_branches.py` | `prompts/cove_grounded_verification_answer.txt`, `prompts/cove_response_revision.txt` |
| C | Apply one additional revision to the frozen Branch A response without retrieved evidence or diagnostic feedback | `scripts/run_controlled_cove_branches.py` | `prompts/cove_extra_revision_control.txt` |
| D | Apply one bounded targeted revision to the same Branch A response using diagnostic feedback and retrieved evidence | `scripts/run_controlled_cove_branches.py` | `prompts/cove_selective_verifier_revision_v2.txt` |

The Branch D candidate-selection precursor retains the internal key `d`, while the active bounded Branch D revision retains the frozen artifact key `d2`. Together they implement the dissertation's single Branch D condition; neither identifier denotes an additional reported branch.

## Revision outcomes, uncertainty, and reliability

| Methodological component | Implementation | Prompt if applicable |
|---|---|---|
| Strict Correction, Removal, Beneficial Disposition, Preservation, Factual Damage, and Deletion construction | `scripts/run_standard_cove_and_post_revision_evaluation.py`, `scripts/validate_post_revision_evaluation.py` | None |
| Paired response-cluster percentile bootstrap with 10,000 resamples | `src/factcheck_bench_analysis.py`, `scripts/validate_post_revision_evaluation.py` | None |
| Normalized-exact retained-claim layer with inherited binary human labels | `scripts/validate_post_revision_evaluation.py` | None |
| Blind cross-model semantic-alignment check | `scripts/validate_post_revision_evaluation.py` | `prompts/cove_targeted_blind_alignment_adjudication.txt` |
| Independent passage-level factuality check | `scripts/validate_post_revision_evaluation.py` | `prompts/cove_independent_revised_claim_adjudication.txt` |
| Standard-CoVe correction funnel, evidence-stage effects, failure taxonomy, and separated evidence layers | `scripts/analyze_cove_diagnostic_traces.py` | None |

Human-anchored, cross-model-supported, silver full-coverage, and external automatic evidence are kept separate and are not collapsed into a single validation score.

## External response-level evaluation

| Component | Implementation |
|---|---|
| Export the initial response and four final branch responses to the isolated official VeriScore protocol | `scripts/prepare_and_analyze_veriscore.py` |
| Run the pinned official VeriScore package through the documented provider adapter | `scripts/run_official_veriscore.py`, `vendor_patches/veriscore_2.0.2_deepseek.patch` |
| Analyze factual precision, supported-claim recall, and response-level F1 at the frozen shared target K=9 | `scripts/prepare_and_analyze_veriscore.py` |
| Recalculate response-level F1 for K=1 through K=20 from frozen response scores without new model, retrieval, or search calls | `scripts/analyze_veriscore_k_sensitivity.py` |
