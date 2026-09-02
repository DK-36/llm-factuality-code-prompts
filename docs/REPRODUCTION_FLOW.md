# Reproduction Flow

This document specifies the formal dependency order represented in the dissertation. The commands are instructions for a future reproducer; they were not executed while preparing this source-only repository, and no generated output is committed.

## Flow overview

`Official benchmark → canonical claims and response-level split → benchmark-linked corpus → development retrieval comparison → frozen Hybrid RRF → Study I verification → Standard CoVe → controlled Branch B/C/D interventions → shared post-revision evaluation → reliability synthesis → isolated VeriScore extension`

Development stages determine or validate frozen choices before any held-out stage is opened. Human labels, benchmark-associated evidence, source mappings, and qrels are evaluation-only fields and are not supplied to retrieval ranking or to generation unless the named methodology stage explicitly requires benchmark-associated evidence.

## 1. Prepare the canonical claims

Place the official file according to `docs/DATA_AND_ENVIRONMENT.md`, then construct stable claim and response identifiers, retain the human annotations for later evaluation, and create the response-level development and held-out membership.

```bash
python scripts/prepare_factcheck_bench_claims.py --scope full
```

## 2. Run the shared Study I baselines

The No Evidence condition must be generated first because the Benchmark-associated Evidence implementation checks that both conditions use the same model profile. The formal scripts support resumption without changing already compatible rows.

```bash
python scripts/run_no_evidence_claim_verification.py --scope full --cohort binary --resume
python scripts/run_benchmark_evidence_claim_verification.py --scope full --resume
```

The frozen internal name `oracle_evidence` and the prompt filename `oracle_evidence_verifier.txt` refer to the dissertation's Benchmark-associated Evidence condition; they are retained only because they are part of the stored schemas and fingerprints.

## 3. Build the benchmark-grounded retrieval corpus

The first two stages construct the response-level split and the evaluation-only evidence and URL manifests. `fetch-corpus` is the only networked stage. `reprocess-frozen-corpus` applies the frozen extractor to the fetched document snapshot before passage construction. Held-out qrels remain sealed until the retrieval configuration has been fixed from development data.

```bash
python scripts/build_retrieval_corpus.py prepare-retrieval-splits --scope full
python scripts/build_retrieval_corpus.py build-evidence-manifest --scope full
python scripts/build_retrieval_corpus.py fetch-corpus --scope full
python scripts/build_retrieval_corpus.py reprocess-frozen-corpus --scope full
python scripts/build_retrieval_corpus.py build-passages --scope full
python scripts/build_retrieval_corpus.py build-qrels --scope full --split dev
python scripts/build_retrieval_corpus.py summarize-corpus --scope full
```

## 4. Select and freeze the retriever on development data

Run every development ranking over the same claim-only queries and passage index. The configuration fixes BM25 at `k1=1.2` and `b=0.75`, Dense Retrieval at `nomic-embed-text:v1.5` with the declared query and document prefixes, and unweighted Hybrid RRF with constant 60 over the top 100 component ranks.

```bash
python scripts/evaluate_retrieval_methods.py prepare-two-level-qrels --scope full --split dev
python scripts/evaluate_retrieval_methods.py run-bm25-retrieval --scope full --split dev
python scripts/evaluate_retrieval_methods.py run-dense-retrieval --scope full --split dev
python scripts/evaluate_retrieval_methods.py run-hybrid-retrieval --scope full --split dev
python scripts/evaluate_retrieval_methods.py compare-retrieval --scope full --split dev
```

The tracked retrieval configuration already records Hybrid RRF as the frozen downstream retriever. The held-out corpus and retrieval evaluation require the explicit frozen-configuration acknowledgement.

```bash
python scripts/build_retrieval_corpus.py build-qrels --scope full --split heldout --confirm-config-frozen
python scripts/evaluate_retrieval_methods.py prepare-two-level-qrels --scope full --split heldout --confirm-config-frozen
python scripts/evaluate_retrieval_methods.py run-bm25-retrieval --scope full --split heldout --confirm-config-frozen
python scripts/evaluate_retrieval_methods.py run-dense-retrieval --scope full --split heldout --confirm-config-frozen
python scripts/evaluate_retrieval_methods.py run-hybrid-retrieval --scope full --split heldout --confirm-config-frozen
python scripts/evaluate_retrieval_methods.py compare-retrieval --scope full --split heldout --confirm-config-frozen
```

## 5. Complete Study I verification

Use the single frozen Hybrid run at K=1, K=3, and K=5. Each K changes only the number of retrieved passages exposed to the verifier. The K=5 report supplies the primary retrieved-evidence comparison, while the depth analysis performs the paired K comparison.

```bash
python scripts/run_retrieved_evidence_claim_verification.py --scope full --top-k 1 --resume
python scripts/run_retrieved_evidence_claim_verification.py --scope full --top-k 3 --resume
python scripts/run_retrieved_evidence_claim_verification.py --scope full --top-k 5 --resume
python scripts/analyze_retrieval_depth.py --scope full
```

## 6. Run Standard CoVe on development, then held-out data

The Standard CoVe generation path is B1 question planning, B3 independent answering, and B5 revision. B2 question–claim alignment and B4 evidence-grounded answer evaluation are diagnostic only and do not feed B5. B6a, B6b, and B6c evaluate the revised response through extraction, semantic alignment, and frozen Hybrid top-five factuality reassessment.

For each split, use `scripts/run_standard_cove_and_post_revision_evaluation.py` in this order: `prepare-inputs`, `run-questions`, `analyze-questions`, `run-alignment`, `analyze-alignment`, `run-answers`, `analyze-answers`, `run-answer-evaluation`, `analyze-answer-evaluation`, `run-revision`, `analyze-revision`, `run-revised-claim-extraction`, `analyze-revised-claim-extraction`, `run-revised-claim-alignment`, `analyze-revised-claim-alignment`, `prepare-revised-claim-evidence`, `run-revised-claim-factuality`, `analyze-revised-claim-factuality`, and `prepare-factuality-audit`.

Every model-backed held-out subcommand requires `--split heldout --confirm-config-frozen --resume`; deterministic held-out analysis subcommands use `--split heldout`. Development uses `--split dev` and is completed before the held-out confirmation is supplied. The `recover-*` subcommands are deterministic format normalization paths and are used only if a completed model response matches their explicitly validated recoverable schema.

## 7. Run the controlled CoVe branches

Branch A is the frozen Standard CoVe output from the previous section. On development and then held-out data, run `scripts/run_controlled_cove_branches.py` in this order: `audit`, `prepare-grounded-evidence`, `run-grounded-answers`, `run-grounded-revision`, `run-extra-revision`, `prepare-targeted-feedback-candidates`, `prepare-bounded-targeted-feedback`, `run-targeted-evidence-revision`, and `summarize`.

Branch B reuses the frozen Branch A questions, retrieves evidence for each question, produces independent grounded answers, and applies the unchanged standard revision prompt to the initial response. Branch C applies one additional evidence-free revision to the Branch A response. Branch D first derives deterministic candidate claims from Branch A B6c diagnostics, bounds the number and length of visible targets and evidence excerpts, and then applies one evidence-guided revision to the same Branch A response used by Branch C.

The precursor candidate artifact uses the internal key `d`; the executed dissertation Branch D uses the frozen artifact key `d2`. These are two internal stages of one reported Branch D condition, not two reported branches.

## 8. Apply the same post-revision evaluation to every branch

For branches `b`, `c`, and `d2`, repeat the B6a extraction, B6b alignment, B6c frozen Hybrid top-five factuality reassessment, and deterministic factuality audit through `scripts/run_standard_cove_and_post_revision_evaluation.py --branch <branch>`. Branch A already has these artifacts from Standard CoVe. The shared branch set for analysis is therefore `a`, `b`, `c`, and `d2`, displayed publicly as A, B, C, and D.

## 9. Run the formal uncertainty and reliability layers

Run `scripts/validate_post_revision_evaluation.py` in this order: `analyze-paired-statistics`, `build-exact-retained`, `prepare-targeted-audit`, `run-targeted-alignment`, `run-targeted-factuality`, and `analyze-targeted-validation`. The two model stages require `--confirm-config-frozen --resume`; all stages are held-out only.

The exact-retained layer inherits a binary human label only after normalized-exact matching. The blind Llama semantic and passage-factuality judgements remain auxiliary because the development calibration gate did not justify replacing the primary silver evaluation.

Then run `scripts/analyze_cove_diagnostic_traces.py` in this order: `build-non-factual-funnel`, `analyze-evidence-stage-effects`, `build-failure-taxonomy`, and `build-three-layer-report`. These stages are deterministic and keep human-anchored, cross-model-supported, and silver full-coverage evidence separate.

## 10. Run the isolated VeriScore extension

First export the initial and final responses with `python scripts/prepare_and_analyze_veriscore.py prepare --scope full --split heldout`. Install the official VeriScore checkout and environment at the pinned version declared in `docs/DATA_AND_ENVIRONMENT.md`, apply the tracked provider adapter, set the two credential environment variables, and launch `python scripts/run_official_veriscore.py --split heldout`.

The launcher invokes the official package out of process, records non-secret protocol metadata, and then imports the result through the project adapter. If importing separately, use `python scripts/prepare_and_analyze_veriscore.py analyze --scope full --split heldout --results <official-jsonl> --run-metadata <metadata-json>`.

After the primary shared K=9 analysis is fixed, run `python scripts/analyze_veriscore_k_sensitivity.py` to recalculate response-level F1 for K=1 through K=20 from the same frozen response scores without new extraction, search, verification, or retrieval calls.

## Completion and interpretation boundary

A fresh run is complete only when each stage's built-in cohort-size, unique-identifier, input-hash, model-digest, prompt-hash, frozen-configuration, schema, branch-isolation, and leakage checks pass. A completed run is still a new reproduction: differences caused by current web content, runtime implementations, or live external services must be reported rather than concealed.
