# Prompt Catalogue

These 17 prompt templates are the complete formal prompt set represented in the dissertation methodology and prompt appendix. Human labels are withheld from every generation, verification, revision, and model-based evaluation prompt; evaluation-only evidence is supplied only where the method explicitly requires it.

## Study I verification prompts

| Prompt | Model-visible input | Role |
|---|---|---|
| `no_evidence_verifier.txt` | One canonical claim | Claim-only verification using reliable internal knowledge |
| `oracle_evidence_verifier.txt` | One canonical claim and benchmark-associated evidence text | Benchmark-associated Evidence condition; `oracle` is retained only as the frozen implementation name |
| `retrieved_evidence_verifier.txt` | One canonical claim and the selected Hybrid passages | Retrieved Evidence condition at K=1, K=3, or K=5 |
| `retrieved_evidence_output_repair.txt` | An otherwise substantive verifier response with a format defect | Format-only normalization without changing the factual judgement |

## Standard CoVe prompts

| Prompt | Role |
|---|---|
| `cove_question_planning.txt` | Generate verification questions from the original question and initial response |
| `cove_independent_verification_answer.txt` | Answer one verification question without access to the initial response or other questions |
| `cove_response_revision.txt` | Revise the initial response from the original question and independent question–answer pairs |

## Diagnostic and post-revision evaluation prompts

| Prompt | Role |
|---|---|
| `cove_question_claim_alignment.txt` | Align generated questions to label-free canonical claims for diagnostic coverage |
| `cove_answer_claim_evaluation.txt` | Assess an independent answer–claim pair against benchmark-associated evidence |
| `cove_revised_claim_extraction.txt` | Extract atomic factual claims from a complete revised response |
| `cove_gold_revised_claim_alignment.txt` | Align canonical initial claims with revised content without factuality judgement |
| `cove_revised_claim_factuality.txt` | Judge one revised claim using frozen Hybrid top-five passages |

## Controlled intervention prompts

| Prompt | Role |
|---|---|
| `cove_grounded_verification_answer.txt` | Branch B independent answering with retrieved evidence |
| `cove_extra_revision_control.txt` | Branch C additional revision without retrieved evidence or diagnostic feedback |
| `cove_selective_verifier_revision_v2.txt` | Active Branch D bounded targeted revision using diagnostic feedback and evidence excerpts |

## Reliability-check prompts

| Prompt | Role |
|---|---|
| `cove_targeted_blind_alignment_adjudication.txt` | Blind cross-model semantic-presence assessment |
| `cove_independent_revised_claim_adjudication.txt` | Independent passage-by-passage support, refutation, or insufficiency assessment |

The superseded Branch D prompt and the development-only exploratory retrieval-depth prompt are deliberately excluded because neither belongs to the final reported method.
