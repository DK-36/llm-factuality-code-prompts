#!/usr/bin/env python3
"""Deterministic diagnostic synthesis for the frozen CoVe held-out experiment.

The stages in this module make no model calls and never modify Branch A/B/C/D artifacts. They materialise the reported NON_FACTUAL correction funnel, locate evidence effects by controlled stage, assign an explicit failure taxonomy, and produce a three-layer report that keeps human, cross-model, and silver evidence separate. Branch D is stored internally as ``d2`` for frozen-schema compatibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factcheck_bench_cove import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    cove_paths,
    load_json,
    load_jsonl,
    sha256_file,
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "factcheck_bench"
    / "cove"
    / "config"
    / "cove_posthoc_analysis_config.json"
)
VALIDATION_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_full"
    / "cove"
    / "branches"
    / "validation"
)
JSONL_DIR = VALIDATION_ROOT / "jsonl"
REPORTS_DIR = VALIDATION_ROOT / "reports"
VERIFIER_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_full"
    / "reports"
    / "12_retrieved_evidence_verifier_heldout_summary.json"
)
BRANCH_LABELS = {"a": "A", "b": "B", "c": "C", "d2": "D"}
STAGES = (
    "build-non-factual-funnel",
    "analyze-evidence-stage-effects",
    "build-failure-taxonomy",
    "build-three-layer-report",
)
BENEFICIAL_ERROR_OUTCOMES = {
    "ERROR_CORRECTED_CANDIDATE",
    "ERROR_REMOVED_BY_DELETION_CANDIDATE",
}
PRESERVED_FACTUAL_OUTCOMES = {
    "FACTUAL_RETAINED_CANDIDATE",
    "FACTUAL_PRESERVED_AFTER_CHANGE_CANDIDATE",
}
FACTUAL_HARM_OUTCOMES = {
    "FACTUAL_DAMAGED_CANDIDATE",
    "FACTUAL_DELETED_CANDIDATE",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def output_path(name: str, split: str, suffix: str) -> Path:
    root = JSONL_DIR if suffix == "jsonl" else REPORTS_DIR
    return root / f"{name}_{split}.{suffix}"


def config() -> dict[str, Any]:
    cfg = load_json(CONFIG_PATH)
    if cfg.get("schema_version") != "fcb_cove_posthoc_analysis_config_v1":
        raise ValueError("Unexpected post-hoc analysis config schema")
    if cfg.get("split") != "heldout":
        raise ValueError("Post-hoc analysis is frozen to heldout")
    return cfg


def branch_paths(branch: str):
    return cove_paths(PROJECT_ROOT, "full", branch)


def branch_artifacts(branch: str, split: str) -> dict[str, Any]:
    paths = branch_paths(branch)
    file_paths = {
        "initial": paths.initial_claim_outcomes(split),
        "added": paths.added_claim_outcomes(split),
        "revision": paths.revision_results(split),
    }
    missing = [relative(path) for path in file_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing branch artifacts for {branch}: {missing}")
    return {
        "paths": file_paths,
        "initial": load_jsonl(file_paths["initial"]),
        "added": load_jsonl(file_paths["added"]),
        "revision": load_jsonl(file_paths["revision"]),
    }


def load_cohort(split: str) -> dict[str, dict[str, Any]]:
    cfg = config()
    contexts = {
        branch: branch_artifacts(branch, split)
        for branch in cfg["active_branches"]
    }
    expected = cfg["expected_cohort"]
    initial_sets = []
    for branch, context in contexts.items():
        rows = context["initial"]
        if len(context["revision"]) != expected["responses"]:
            raise ValueError(f"Unexpected response count for branch {branch}")
        if len(rows) != expected["initial_claims"]:
            raise ValueError(f"Unexpected initial-claim count for branch {branch}")
        counts = Counter(row["human_label"] for row in rows)
        if counts["NON_FACTUAL"] != expected["non_factual_initial_claims"]:
            raise ValueError(f"Unexpected NON_FACTUAL count for branch {branch}")
        if counts["FACTUAL"] != expected["factual_initial_claims"]:
            raise ValueError(f"Unexpected FACTUAL count for branch {branch}")
        initial_sets.append({row["initial_claim_id"] for row in rows})
    if any(ids != initial_sets[0] for ids in initial_sets[1:]):
        raise ValueError("Branch initial-claim sets are not identical")
    return contexts


def source_fingerprints(paths: list[Path]) -> dict[str, str]:
    return {relative(path): sha256_file(path) for path in paths}


def pct(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def funnel_paths(split: str) -> tuple[Path, Path, Path]:
    return (
        output_path("V7_non_factual_correction_funnel", split, "jsonl"),
        output_path("V7_non_factual_correction_funnel_summary", split, "json"),
        output_path("V7_non_factual_correction_funnel", split, "md"),
    )


def _usable_b4(row: dict[str, Any], settings: dict[str, Any]) -> bool:
    return (
        row.get("status") == "ok"
        and row.get("oracle_evidence_usable") is True
        and row.get("alignment_validity") in settings["usable_alignment_validity"]
        and row.get("evidence_sufficiency")
        in settings["usable_evidence_sufficiency"]
    )


def _broad_challenge(row: dict[str, Any], settings: dict[str, Any]) -> bool:
    return (
        _usable_b4(row, settings)
        and row.get("answer_stance") == settings["required_challenge_stance"]
        and row.get("answer_correctness")
        in settings["broad_challenge_correctness"]
    )


def _strict_challenge(row: dict[str, Any], settings: dict[str, Any]) -> bool:
    return (
        _usable_b4(row, settings)
        and row.get("answer_stance") == settings["required_challenge_stance"]
        and row.get("answer_correctness")
        in settings["strict_challenge_correctness"]
        and row.get("evidence_sufficiency")
        == settings["strict_challenge_requires_evidence_sufficiency"]
    )


def classify_mechanism(row: dict[str, Any]) -> str:
    outcome = row["terminal_silver_outcome"]
    if outcome == "ERROR_CORRECTED_CANDIDATE":
        return "SUCCESS_STRICT_CORRECTION"
    if outcome == "ERROR_REMOVED_BY_DELETION_CANDIDATE":
        return "SUCCESS_ERROR_REMOVAL_BY_DELETION"
    if outcome == "SILVER_LABEL_DISAGREEMENT_CANDIDATE":
        return "EVALUATOR_LABEL_CONFLICT"
    if outcome in {
        "UNRESOLVED_REVISED_FACTUALITY",
        "UNRESOLVED_EXTRACTION_OMISSION",
    }:
        return "FINAL_FACTUALITY_UNRESOLVED"
    if not row["useful_question_covered"]:
        return "PLANNER_MISS"
    if not row["has_usable_b4_evaluation"]:
        return "VERIFICATION_NOT_EVALUABLE"
    if not row["has_broad_supported_challenge"]:
        return "VERIFICATION_DID_NOT_PRODUCE_SUPPORTED_CHALLENGE"
    if not row["revision_action_observed"]:
        return "REVISION_OMISSION_AFTER_CHALLENGE"
    return "REVISION_UNSUCCESSFUL_AFTER_CHALLENGE"


def build_non_factual_funnel(split: str, dry_run: bool) -> int:
    cfg = config()
    settings = cfg["non_factual_funnel"]
    contexts = load_cohort(split)
    base = contexts["a"]
    base_paths = branch_paths("a")
    pair_path = base_paths.alignment_pairs(split)
    b4_path = base_paths.answer_claim_evaluation_results(split)
    exact_path = output_path("V2_exact_retained_claims", split, "jsonl")
    for path in (pair_path, b4_path, exact_path):
        if not path.exists():
            raise FileNotFoundError(relative(path))

    useful_relations = set(settings["useful_question_relations"])
    useful_pairs: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(pair_path):
        if row.get("human_label") == "NON_FACTUAL" and row.get("relation") in useful_relations:
            useful_pairs[row["claim_id"]].append(row)
    evaluations: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(b4_path):
        evaluations[row["claim_id"]].append(row)
    exact_by_claim = {
        row["initial_claim_id"]: row
        for row in load_jsonl(exact_path)
        if row["branch"] == "a" and row["human_label"] == "NON_FACTUAL"
    }
    nf_rows = sorted(
        (row for row in base["initial"] if row["human_label"] == "NON_FACTUAL"),
        key=lambda row: row["initial_claim_id"],
    )
    rows: list[dict[str, Any]] = []
    for initial in nf_rows:
        claim_id = initial["initial_claim_id"]
        pairs = useful_pairs.get(claim_id, [])
        question_ids = sorted({row["question_id"] for row in pairs})
        relevant_evals = [
            row for row in evaluations.get(claim_id, [])
            if row.get("question_id") in question_ids
        ]
        usable = [row for row in relevant_evals if _usable_b4(row, settings)]
        broad = [row for row in relevant_evals if _broad_challenge(row, settings)]
        strict = [row for row in relevant_evals if _strict_challenge(row, settings)]
        relation = initial["relation"]
        outcome = initial["provisional_outcome"]
        record = {
            "schema_version": "fcb_cove_non_factual_funnel_row_v1",
            "split": split,
            "branch": "a",
            "response_id": initial["response_id"],
            "initial_claim_id": claim_id,
            "initial_claim": initial["initial_claim"],
            "human_label": "NON_FACTUAL",
            "useful_question_covered": bool(pairs),
            "useful_question_ids": question_ids,
            "useful_question_count": len(question_ids),
            "direct_pair_count": sum(row["relation"] == "DIRECT" for row in pairs),
            "partial_pair_count": sum(row["relation"] == "PARTIAL" for row in pairs),
            "has_usable_b4_evaluation": bool(usable),
            "usable_b4_evaluation_ids": sorted(row["evaluation_id"] for row in usable),
            "has_broad_supported_challenge": bool(broad),
            "broad_supported_challenge_ids": sorted(row["evaluation_id"] for row in broad),
            "has_strict_supported_challenge": bool(strict),
            "strict_supported_challenge_ids": sorted(row["evaluation_id"] for row in strict),
            "revision_relation": relation,
            "revision_action_observed": relation in settings["revision_action_relations"],
            "terminal_silver_outcome": outcome,
            "strict_correction_candidate": outcome == settings["strict_correction_outcome"],
            "error_deletion_candidate": outcome == settings["deletion_outcome"],
            "beneficial_error_disposition_candidate": outcome in BENEFICIAL_ERROR_OUTCOMES,
            "exact_retained_gold": claim_id in exact_by_claim,
            "exact_retained_disposition": (
                exact_by_claim[claim_id]["exact_transition_disposition"]
                if claim_id in exact_by_claim else None
            ),
            "evaluation_scope": {
                "planning_alignment": "B2_same_model_silver",
                "answer_challenge": "B4_oracle_evidence_grounded_same_model_silver",
                "terminal_outcome": "B6b_B6c_same_protocol_silver",
                "exact_retained": "human_gold_when_present"
            },
        }
        record["mechanism_taxonomy"] = classify_mechanism(record)
        rows.append(record)

    expected = cfg["expected_cohort"]["non_factual_initial_claims"]
    if len(rows) != expected or len({row["initial_claim_id"] for row in rows}) != expected:
        raise ValueError("NON_FACTUAL funnel denominator or ID uniqueness failed")

    nested = {
        "initial_non_factual": len(rows),
        "useful_question_covered": sum(row["useful_question_covered"] for row in rows),
        "covered_and_b4_evaluable": sum(
            row["useful_question_covered"] and row["has_usable_b4_evaluation"]
            for row in rows
        ),
        "covered_evaluable_and_broad_challenge": sum(
            row["useful_question_covered"]
            and row["has_usable_b4_evaluation"]
            and row["has_broad_supported_challenge"]
            for row in rows
        ),
        "covered_evaluable_challenged_and_revision_action": sum(
            row["useful_question_covered"]
            and row["has_usable_b4_evaluation"]
            and row["has_broad_supported_challenge"]
            and row["revision_action_observed"]
            for row in rows
        ),
        "full_path_strict_correction": sum(
            row["useful_question_covered"]
            and row["has_usable_b4_evaluation"]
            and row["has_broad_supported_challenge"]
            and row["revision_action_observed"]
            and row["strict_correction_candidate"]
            for row in rows
        ),
        "full_path_correction_or_deletion": sum(
            row["useful_question_covered"]
            and row["has_usable_b4_evaluation"]
            and row["has_broad_supported_challenge"]
            and row["revision_action_observed"]
            and row["beneficial_error_disposition_candidate"]
            for row in rows
        ),
    }
    terminal = Counter(row["terminal_silver_outcome"] for row in rows)
    taxonomy = Counter(row["mechanism_taxonomy"] for row in rows)
    path_diagnostics = {
        "strict_corrections_outside_full_path": sum(
            row["strict_correction_candidate"]
            and not (
                row["useful_question_covered"]
                and row["has_usable_b4_evaluation"]
                and row["has_broad_supported_challenge"]
                and row["revision_action_observed"]
            )
            for row in rows
        ),
        "deletions_outside_full_path": sum(
            row["error_deletion_candidate"]
            and not (
                row["useful_question_covered"]
                and row["has_usable_b4_evaluation"]
                and row["has_broad_supported_challenge"]
                and row["revision_action_observed"]
            )
            for row in rows
        ),
        "broad_challenge_but_no_beneficial_terminal_outcome": sum(
            row["has_broad_supported_challenge"]
            and not row["beneficial_error_disposition_candidate"]
            for row in rows
        ),
        "exact_retained_gold_errors": sum(row["exact_retained_gold"] for row in rows),
    }
    summary = {
        "schema_version": "fcb_cove_non_factual_funnel_summary_v1",
        "status": "complete",
        "split": split,
        "branch": "a",
        "denominator": len(rows),
        "nested_funnel_counts": nested,
        "nested_funnel_rates_over_initial_non_factual": {
            key: pct(value, len(rows)) for key, value in nested.items()
        },
        "terminal_silver_outcome_counts": dict(terminal),
        "mechanism_taxonomy_counts": dict(taxonomy),
        "path_diagnostics": path_diagnostics,
        "definitions": settings,
        "interpretation": (
            "The funnel is a diagnostic path over standard CoVe. B2, B4, and "
            "B6 labels are silver. A stage count is not human-gold correctness, "
            "and terminal corrections outside the complete measured path are "
            "reported separately rather than forced into a causal story."
        ),
        "source_fingerprints": source_fingerprints([
            pair_path,
            b4_path,
            base["paths"]["initial"],
            exact_path,
            CONFIG_PATH,
        ]),
        "generated_at": utc_now(),
    }
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    rows_path, summary_path, md_path = funnel_paths(split)
    atomic_write_jsonl(rows_path, rows)
    atomic_write_json(summary_path, summary)
    labels = [
        ("Initial NON_FACTUAL claims", "initial_non_factual"),
        ("Useful question covered", "useful_question_covered"),
        ("Covered + B4 evaluable", "covered_and_b4_evaluable"),
        ("Covered + evaluable + broad supported challenge", "covered_evaluable_and_broad_challenge"),
        ("Previous stages + revision action", "covered_evaluable_challenged_and_revision_action"),
        ("Full path + strict correction", "full_path_strict_correction"),
        ("Full path + correction or deletion", "full_path_correction_or_deletion"),
    ]
    lines = [
        "# V7 — Standard-CoVe NON_FACTUAL Correction Funnel",
        "",
        "This report follows the 100 held-out human-labelled NON_FACTUAL initial "
        "claims through the frozen standard-CoVe mechanism. All intermediate "
        "judgments remain silver unless the Exact-retained layer applies.",
        "",
        "| Nested stage | Claims | Rate over initial NON_FACTUAL |",
        "|---|---:|---:|",
    ]
    for label, key in labels:
        value = nested[key]
        lines.append(f"| {label} | {value} | {fmt_pct(pct(value, len(rows)))} |")
    lines.extend([
        "",
        "## Terminal outcomes",
        "",
        "| Outcome | Claims |",
        "|---|---:|",
    ])
    for key, value in sorted(terminal.items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend([
        "",
        "## Deterministic mechanism taxonomy",
        "",
        "| Category | Claims |",
        "|---|---:|",
    ])
    for key in cfg["failure_taxonomy"]["priority"]:
        lines.append(f"| `{key}` | {taxonomy.get(key, 0)} |")
    lines.extend([
        "",
        "The taxonomy assigns terminal success first. For an unsuccessful claim, "
        "it then records the earliest observable bottleneck. It does not prove "
        "that the labelled stage caused the failure.",
        "",
    ])
    atomic_write_text(md_path, "\n".join(lines))
    print(json.dumps({
        "stage": "V7_non_factual_correction_funnel",
        "status": "complete",
        "denominator": len(rows),
        "nested_funnel_counts": nested,
        "terminal_silver_outcome_counts": dict(terminal),
        "mechanism_taxonomy_counts": dict(taxonomy),
        "report": relative(md_path),
    }, ensure_ascii=False, indent=2))
    return 0


def _answer_status_effect(split: str) -> dict[str, Any]:
    paths = branch_paths("a")
    standard_path = paths.verification_answer_results(split)
    grounded_path = (
        PROJECT_ROOT
        / "outputs"
        / "factcheck_bench_full"
        / "cove"
        / "branches"
        / "branch_b"
        / "jsonl"
        / f"Q3_grounded_verification_answers_{split}.jsonl"
    )
    standard = {row["question_id"]: row for row in load_jsonl(standard_path)}
    grounded = {row["question_id"]: row for row in load_jsonl(grounded_path)}
    if set(standard) != set(grounded):
        raise ValueError("Standard and grounded answer question sets differ")
    matrix = Counter(
        (standard[qid]["answer_status"], grounded[qid]["answer_status"])
        for qid in sorted(standard)
    )
    return {
        "paired_question_count": len(standard),
        "standard_status_counts": dict(Counter(row["answer_status"] for row in standard.values())),
        "grounded_status_counts": dict(Counter(row["answer_status"] for row in grounded.values())),
        "paired_status_matrix": {
            f"{before}_TO_{after}": count
            for (before, after), count in sorted(matrix.items())
        },
        "grounded_rows_with_citations": sum(
            bool(row.get("cited_passage_ranks")) for row in grounded.values()
        ),
        "interpretation": (
            "Answer status is a model self-report, not correctness. This contrast "
            "measures behavioural change under evidence only."
        ),
        "source_paths": [standard_path, grounded_path],
    }


def _branch_terminal_summary(context: dict[str, Any]) -> dict[str, Any]:
    initial = context["initial"]
    nf = [row for row in initial if row["human_label"] == "NON_FACTUAL"]
    factual = [row for row in initial if row["human_label"] == "FACTUAL"]
    added = context["added"]
    return {
        "non_factual_denominator": len(nf),
        "strict_correction": sum(
            row["provisional_outcome"] == "ERROR_CORRECTED_CANDIDATE" for row in nf
        ),
        "error_deletion": sum(
            row["provisional_outcome"] == "ERROR_REMOVED_BY_DELETION_CANDIDATE" for row in nf
        ),
        "beneficial_error_disposition": sum(
            row["provisional_outcome"] in BENEFICIAL_ERROR_OUTCOMES for row in nf
        ),
        "factual_denominator": len(factual),
        "factual_preserved": sum(
            row["provisional_outcome"] in PRESERVED_FACTUAL_OUTCOMES for row in factual
        ),
        "factual_damaged": sum(
            row["provisional_outcome"] == "FACTUAL_DAMAGED_CANDIDATE" for row in factual
        ),
        "factual_deleted": sum(
            row["provisional_outcome"] == "FACTUAL_DELETED_CANDIDATE" for row in factual
        ),
        "added_claim_denominator": len(added),
        "added_errors": sum(row["provisional_outcome"] == "NEW_ERROR_CANDIDATE" for row in added),
        "initial_outcome_counts": dict(Counter(row["provisional_outcome"] for row in initial)),
        "added_outcome_counts": dict(Counter(row["provisional_outcome"] for row in added)),
    }


def evidence_paths(split: str) -> tuple[Path, Path]:
    return (
        output_path("V8_evidence_stage_effects", split, "json"),
        output_path("V8_evidence_stage_effects", split, "md"),
    )


def analyze_evidence_stage_effects(split: str, dry_run: bool) -> int:
    cfg = config()
    contexts = load_cohort(split)
    verifier = load_json(VERIFIER_SUMMARY_PATH)
    v1_path = output_path("V1_paired_branch_statistics", split, "json")
    v1 = load_json(v1_path)
    answer_effect = _answer_status_effect(split)
    branch_terminal = {
        branch: _branch_terminal_summary(context)
        for branch, context in contexts.items()
    }
    d2_feedback_path = (
        PROJECT_ROOT
        / "outputs"
        / "factcheck_bench_full"
        / "cove"
        / "branches"
        / "branch_d2"
        / "reports"
        / f"branch_d2_feedback_{split}_summary.json"
    )
    branch_b_generation_path = (
        PROJECT_ROOT
        / "outputs"
        / "factcheck_bench_full"
        / "cove"
        / "branches"
        / "branch_b"
        / "reports"
        / f"branch_b_generation_{split}_summary.json"
    )
    d2_feedback = load_json(d2_feedback_path)
    branch_b_generation = load_json(branch_b_generation_path)
    verifier_metrics = {
        name: {
            metric: verifier["metrics"][name][metric]
            for metric in ("accuracy_including_abstentions_and_errors", "balanced_accuracy", "macro_f1", "coverage")
        }
        for name in ("no_evidence", "oracle_evidence", "retrieved_evidence")
    }
    paired = verifier["paired_response_cluster_bootstrap"]["paired_difference_intervals"]
    branch_intervals = v1["branch_rate_cluster_bootstrap"]["paired_difference_intervals"]
    stages = [
        {
            "stage_id": "E1_claim_level_verification",
            "controlled_contrast": "same claims/model/prompt family; evidence source changes",
            "evidence_condition": "No Evidence vs Benchmark-associated Evidence vs Hybrid retrieved top-5",
            "result": {
                "metrics": verifier_metrics,
                "paired_intervals": paired,
            },
            "evidence_strength": "HUMAN_ANCHORED",
            "interpretation": (
                "Benchmark-associated and retrieved evidence improve verifier classification over "
                "No Evidence; retrieved-minus-benchmark-associated remains inconclusive."
            ),
        },
        {
            "stage_id": "E2_verification_question_planning",
            "controlled_contrast": "none",
            "evidence_condition": "B1 questions are frozen and shared",
            "result": {"questions_changed_by_evidence": 0},
            "evidence_strength": "DESIGN_INVARIANT",
            "interpretation": "Evidence is introduced only after question planning.",
        },
        {
            "stage_id": "E3_verification_answer_generation",
            "controlled_contrast": "Branch B grounded answers minus Branch A parametric answers on identical B1 questions",
            "evidence_condition": "Hybrid retrieved top-5 passages are visible only in Branch B",
            "result": answer_effect,
            "evidence_strength": "BEHAVIOURAL_SELF_REPORT_ONLY",
            "interpretation": answer_effect["interpretation"],
        },
        {
            "stage_id": "E4_grounded_cove_revision",
            "controlled_contrast": "B minus A",
            "evidence_condition": "evidence grounds verification answers; final revision prompt does not see passages",
            "result": {
                "branch_a": branch_terminal["a"],
                "branch_b": branch_terminal["b"],
                "paired_interval": branch_intervals["b_minus_a"],
                "branch_b_generation": branch_b_generation,
            },
            "evidence_strength": "SILVER_RESPONSE_CLUSTER_BOOTSTRAP",
            "interpretation": (
                "The preservation advantage is stable, while the beneficial-error "
                "difference is inconclusive. Branch B used one fallback and extensive "
                "format-only recovery, which remains a sensitivity limitation."
            ),
        },
        {
            "stage_id": "E5_extra_revision_control",
            "controlled_contrast": "C minus A",
            "evidence_condition": "one generic extra revision call without evidence or verifier feedback",
            "result": {
                "branch_a": branch_terminal["a"],
                "branch_c": branch_terminal["c"],
                "paired_interval": branch_intervals["c_minus_a"],
            },
            "evidence_strength": "SILVER_RESPONSE_CLUSTER_BOOTSTRAP",
            "interpretation": "An extra unguided revision call does not show a stable benefit over A.",
        },
        {
            "stage_id": "E6_selective_post_cove_feedback",
            "controlled_contrast": "D minus C",
            "evidence_condition": "same extra-call budget; only D receives bounded claim-level verifier feedback and retrieved excerpts",
            "result": {
                "branch_c": branch_terminal["c"],
                "branch_d2": branch_terminal["d2"],
                "paired_interval": branch_intervals["d_minus_c"],
                "feedback_exposure": d2_feedback,
            },
            "evidence_strength": "SILVER_RESPONSE_CLUSTER_BOOTSTRAP",
            "interpretation": (
                "D improves both beneficial error disposition and factual preservation "
                "over the unguided extra-revision control under the same silver protocol."
            ),
        },
    ]
    all_sources = [VERIFIER_SUMMARY_PATH, v1_path, d2_feedback_path, branch_b_generation_path]
    all_sources.extend(answer_effect.pop("source_paths"))
    all_sources.extend(
        path
        for context in contexts.values()
        for path in context["paths"].values()
    )
    summary = {
        "schema_version": "fcb_cove_evidence_stage_effects_v1",
        "status": "complete",
        "split": split,
        "stage_effects": stages,
        "branch_terminal_silver_summary": branch_terminal,
        "causal_boundaries": [
            "E1 uses human claim labels and supports an evidence-availability claim.",
            "E3 answer statuses are self-reports and cannot be interpreted as correctness.",
            "B-minus-A isolates answer grounding but Branch B has one fallback and format-only recovery sensitivity.",
            "C-minus-A controls for one additional unguided revision call.",
            "D-minus-C is the primary selective-feedback contrast under equal extra-call count.",
        ],
        "source_fingerprints": source_fingerprints(all_sources + [CONFIG_PATH]),
        "generated_at": utc_now(),
    }
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    json_path, md_path = evidence_paths(split)
    atomic_write_json(json_path, summary)
    vm = verifier_metrics
    bma = branch_intervals["b_minus_a"]["metrics"]
    dmc = branch_intervals["d_minus_c"]["metrics"]
    lines = [
        "# V8 — Where Evidence Changes the Pipeline",
        "",
        "| Stage | Controlled comparison | What can be concluded |",
        "|---|---|---|",
        "| Claim verifier | No Evidence / Benchmark-associated Evidence / Retrieved Evidence | Evidence improves human-labelled claim classification; retrieved versus benchmark-associated evidence is inconclusive. |",
        "| Question planning | Shared B1 questions | Evidence has no opportunity to change planning in the frozen design. |",
        "| Verification answers | B versus A, identical questions | Evidence changes answer behaviour, but self-reported status is not correctness. |",
        "| Grounded CoVe revision | B minus A | Main stable effect is improved factual preservation, not a clearly resolved correction gain. |",
        "| Extra-call control | C minus A | A generic extra revision does not show stable improvement. |",
        "| Selective feedback | D minus C | Under the silver protocol, bounded verifier feedback improves both error disposition and preservation. |",
        "",
        "## Claim-level verifier",
        "",
        "| Setting | Accuracy | Balanced accuracy | Macro-F1 | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for setting in ("no_evidence", "oracle_evidence", "retrieved_evidence"):
        row = vm[setting]
        lines.append(
            f"| `{setting}` | {fmt_pct(row['accuracy_including_abstentions_and_errors'])} | "
            f"{fmt_pct(row['balanced_accuracy'])} | {fmt_pct(row['macro_f1'])} | "
            f"{fmt_pct(row['coverage'])} |"
        )
    lines.extend([
        "",
        "## Verification-answer behaviour",
        "",
        f"The paired question set contains {answer_effect['paired_question_count']} questions. "
        f"Standard CoVe statuses are `{answer_effect['standard_status_counts']}`; grounded "
        f"statuses are `{answer_effect['grounded_status_counts']}`. This is behavioural "
        "evidence only and is not used as an answer-accuracy result.",
        "",
        "## Downstream branch contrasts",
        "",
        "| Contrast | Error-disposition delta (95% CI) | Preservation delta (95% CI) |",
        "|---|---:|---:|",
        f"| B − A | {100*bma['beneficial_error_disposition_rate']['point_estimate']:+.2f} pp "
        f"[{100*bma['beneficial_error_disposition_rate']['lower']:+.2f}, {100*bma['beneficial_error_disposition_rate']['upper']:+.2f}] | "
        f"{100*bma['factual_preservation_rate']['point_estimate']:+.2f} pp "
        f"[{100*bma['factual_preservation_rate']['lower']:+.2f}, {100*bma['factual_preservation_rate']['upper']:+.2f}] |",
        f"| D − C | {100*dmc['beneficial_error_disposition_rate']['point_estimate']:+.2f} pp "
        f"[{100*dmc['beneficial_error_disposition_rate']['lower']:+.2f}, {100*dmc['beneficial_error_disposition_rate']['upper']:+.2f}] | "
        f"{100*dmc['factual_preservation_rate']['point_estimate']:+.2f} pp "
        f"[{100*dmc['factual_preservation_rate']['lower']:+.2f}, {100*dmc['factual_preservation_rate']['upper']:+.2f}] |",
        "",
        "The branch effects remain same-protocol silver results. They locate where "
        "evidence is associated with improvement but do not convert revised claims "
        "into human gold.",
        "",
    ])
    atomic_write_text(md_path, "\n".join(lines))
    print(json.dumps({
        "stage": "V8_evidence_stage_effects",
        "status": "complete",
        "paired_question_count": answer_effect["paired_question_count"],
        "branch_terminal_silver_summary": branch_terminal,
        "report": relative(md_path),
    }, ensure_ascii=False, indent=2))
    return 0


def taxonomy_paths(split: str) -> tuple[Path, Path, Path]:
    return (
        output_path("V9_failure_taxonomy", split, "jsonl"),
        output_path("V9_failure_taxonomy_summary", split, "json"),
        output_path("V9_failure_taxonomy", split, "md"),
    )


def build_failure_taxonomy(split: str, dry_run: bool) -> int:
    cfg = config()
    contexts = load_cohort(split)
    funnel_path, funnel_summary_path, _ = funnel_paths(split)
    if not funnel_path.exists() or not funnel_summary_path.exists():
        raise FileNotFoundError("Run build-non-factual-funnel before taxonomy")
    funnel = load_jsonl(funnel_path)
    exact_path = output_path("V2_exact_retained_claims", split, "jsonl")
    v6_path = output_path("V6_targeted_validation_summary", split, "json")
    exact_rows = load_jsonl(exact_path)
    v6 = load_json(v6_path)
    rows: list[dict[str, Any]] = []
    for row in funnel:
        rows.append({
            "schema_version": "fcb_cove_failure_taxonomy_row_v1",
            "unit_type": "STANDARD_COVE_NON_FACTUAL_MECHANISM",
            "split": split,
            "branch": "a",
            "response_id": row["response_id"],
            "claim_id": row["initial_claim_id"],
            "claim": row["initial_claim"],
            "human_label": "NON_FACTUAL",
            "category": row["mechanism_taxonomy"],
            "terminal_silver_outcome": row["terminal_silver_outcome"],
            "useful_question_covered": row["useful_question_covered"],
            "has_usable_b4_evaluation": row["has_usable_b4_evaluation"],
            "has_broad_supported_challenge": row["has_broad_supported_challenge"],
            "revision_action_observed": row["revision_action_observed"],
            "exact_retained_gold": row["exact_retained_gold"],
        })
    collateral_summary: dict[str, Any] = {}
    for branch, context in contexts.items():
        factual = [row for row in context["initial"] if row["human_label"] == "FACTUAL"]
        added = context["added"]
        collateral_summary[branch] = {
            "factual_preserved": sum(row["provisional_outcome"] in PRESERVED_FACTUAL_OUTCOMES for row in factual),
            "factual_damaged": sum(row["provisional_outcome"] == "FACTUAL_DAMAGED_CANDIDATE" for row in factual),
            "factual_deleted": sum(row["provisional_outcome"] == "FACTUAL_DELETED_CANDIDATE" for row in factual),
            "factual_unresolved": sum(
                row["provisional_outcome"] not in PRESERVED_FACTUAL_OUTCOMES | FACTUAL_HARM_OUTCOMES
                for row in factual
            ),
            "new_error_candidates": sum(row["provisional_outcome"] == "NEW_ERROR_CANDIDATE" for row in added),
            "added_factual_candidates": sum(row["provisional_outcome"] == "ADDED_FACTUAL_CANDIDATE" for row in added),
            "added_unresolved": sum(row["provisional_outcome"] == "ADDED_CLAIM_UNRESOLVED_CANDIDATE" for row in added),
        }
        for item in factual:
            if item["provisional_outcome"] not in FACTUAL_HARM_OUTCOMES:
                continue
            rows.append({
                "schema_version": "fcb_cove_failure_taxonomy_row_v1",
                "unit_type": "FACTUAL_COLLATERAL_HARM_CANDIDATE",
                "split": split,
                "branch": branch,
                "response_id": item["response_id"],
                "claim_id": item["initial_claim_id"],
                "claim": item["initial_claim"],
                "human_label": "FACTUAL",
                "category": (
                    "FACTUAL_DELETION_CANDIDATE"
                    if item["provisional_outcome"] == "FACTUAL_DELETED_CANDIDATE"
                    else "FACTUAL_DAMAGE_CANDIDATE"
                ),
                "terminal_silver_outcome": item["provisional_outcome"],
                "evaluation_strength": "SILVER_ONLY_UNLESS_EXACT_LAYER_OVERRIDES",
            })
        for item in added:
            if item["provisional_outcome"] != "NEW_ERROR_CANDIDATE":
                continue
            rows.append({
                "schema_version": "fcb_cove_failure_taxonomy_row_v1",
                "unit_type": "NEW_ERROR_CANDIDATE",
                "split": split,
                "branch": branch,
                "response_id": item["response_id"],
                "claim_id": item["revised_claim_id"],
                "claim": item["revised_claim"],
                "human_label": None,
                "category": "NEW_ERROR_CANDIDATE",
                "terminal_silver_outcome": item["provisional_outcome"],
                "evaluation_strength": "SILVER_ONLY",
            })

    mechanism_counts = Counter(row["mechanism_taxonomy"] for row in funnel)
    exact_disagreements = [
        row for row in exact_rows if row.get("qwen_agrees_with_inherited") is False
    ]
    evaluation_failures = {
        "exact_retained_qwen_gold_disagreement_branch_rows": len(exact_disagreements),
        "exact_retained_qwen_gold_disagreement_unique_initial_claims": len({row["initial_claim_id"] for row in exact_disagreements}),
        "blind_alignment_relation_disagreements": v6["targeted_alignment_agreement"].get("relation_disagreement", 0),
        "blind_alignment_relation_agreements": v6["targeted_alignment_agreement"].get("exact_relation_agreement", 0),
        "llama_targeted_non_factual_predictions": v6["targeted_factuality_prediction_counts"].get("NON_FACTUAL", 0),
        "cross_model_confirmed_non_factual": v6["cross_model_confirmed_prediction_counts"].get("NON_FACTUAL", 0),
    }
    examples: dict[str, list[dict[str, Any]]] = {}
    for category in cfg["failure_taxonomy"]["priority"]:
        candidates = [row for row in funnel if row["mechanism_taxonomy"] == category]
        examples[category] = [
            {
                "claim_id": row["initial_claim_id"],
                "response_id": row["response_id"],
                "claim": row["initial_claim"],
                "terminal_silver_outcome": row["terminal_silver_outcome"],
            }
            for row in candidates[:2]
        ]
    summary = {
        "schema_version": "fcb_cove_failure_taxonomy_summary_v1",
        "status": "complete",
        "split": split,
        "standard_cove_non_factual_denominator": len(funnel),
        "mechanism_taxonomy_counts": dict(mechanism_counts),
        "mechanism_taxonomy_priority": cfg["failure_taxonomy"]["priority"],
        "branch_collateral_summary": collateral_summary,
        "evaluation_failure_diagnostics": evaluation_failures,
        "deterministic_examples": examples,
        "interpretation": cfg["failure_taxonomy"]["interpretation"],
        "limitations": [
            "The taxonomy is deterministic but its B2/B4/B6 inputs are silver.",
            "The earliest observed bottleneck is descriptive and not proof of causation.",
            "A corrected terminal claim is counted as success even if the measured verification path was incomplete.",
            "Collateral and added-error categories remain candidate outcomes unless the Exact-retained gold layer overrides them."
        ],
        "source_fingerprints": source_fingerprints([
            funnel_path, funnel_summary_path, exact_path, v6_path, CONFIG_PATH,
            *[path for context in contexts.values() for path in context["paths"].values()],
        ]),
        "generated_at": utc_now(),
    }
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    jsonl_path, summary_path, md_path = taxonomy_paths(split)
    atomic_write_jsonl(jsonl_path, rows)
    atomic_write_json(summary_path, summary)
    lines = [
        "# V9 — CoVe Failure Taxonomy",
        "",
        "The mechanism taxonomy covers all 100 standard-CoVe held-out "
        "NON_FACTUAL claims. Success is assigned first; unsuccessful claims receive "
        "the earliest observable bottleneck in the frozen pipeline.",
        "",
        "## Standard-CoVe NON_FACTUAL taxonomy",
        "",
        "| Category | Claims |",
        "|---|---:|",
    ]
    for category in cfg["failure_taxonomy"]["priority"]:
        lines.append(f"| `{category}` | {mechanism_counts.get(category, 0)} |")
    lines.extend([
        "",
        "## Collateral outcomes by branch",
        "",
        "| Branch | Factual preserved | Factual damaged | Factual deleted | New errors |",
        "|---|---:|---:|---:|---:|",
    ])
    for branch in cfg["active_branches"]:
        item = collateral_summary[branch]
        lines.append(
            f"| {BRANCH_LABELS[branch]} | {item['factual_preserved']} | "
            f"{item['factual_damaged']} | {item['factual_deleted']} | "
            f"{item['new_error_candidates']} |"
        )
    lines.extend([
        "",
        "## Evaluator failure diagnostics",
        "",
    ])
    for key, value in evaluation_failures.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "These categories are diagnostic. They should not be interpreted as a "
        "human-verified causal decomposition.",
        "",
    ])
    atomic_write_text(md_path, "\n".join(lines))
    print(json.dumps({
        "stage": "V9_failure_taxonomy",
        "status": "complete",
        "mechanism_taxonomy_counts": dict(mechanism_counts),
        "branch_collateral_summary": collateral_summary,
        "evaluation_failure_diagnostics": evaluation_failures,
        "report": relative(md_path),
    }, ensure_ascii=False, indent=2))
    return 0


def three_layer_paths(split: str) -> tuple[Path, Path]:
    return (
        output_path("V10_three_layer_research_report", split, "json"),
        output_path("V10_three_layer_research_report", split, "md"),
    )


def build_three_layer_report(split: str, dry_run: bool) -> int:
    cfg = config()
    paths = {
        "v1": output_path("V1_paired_branch_statistics", split, "json"),
        "v2": output_path("V2_exact_retained_summary", split, "json"),
        "v6": output_path("V6_targeted_validation_summary", split, "json"),
        "v7": funnel_paths(split)[1],
        "v8": evidence_paths(split)[0],
        "v9": taxonomy_paths(split)[1],
    }
    missing = [relative(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prerequisite reports: {missing}")
    reports = {name: load_json(path) for name, path in paths.items()}
    v1 = reports["v1"]
    v2 = reports["v2"]
    v6 = reports["v6"]
    v7 = reports["v7"]
    v8 = reports["v8"]
    v9 = reports["v9"]
    exact_binary = sum(
        value["gold_inherited_binary"] for value in v2["branch_summaries"].values()
    )
    dmc = v1["branch_rate_cluster_bootstrap"]["paired_difference_intervals"]["d_minus_c"]["metrics"]
    bma = v1["branch_rate_cluster_bootstrap"]["paired_difference_intervals"]["b_minus_a"]["metrics"]
    layers = {
        "HUMAN_ANCHORED": {
            "scope": "initial benchmark claims plus normalized-exact retained revised claims",
            "initial_binary_claims": cfg["expected_cohort"]["binary_initial_claims"],
            "exact_retained_binary_branch_rows": exact_binary,
            "exact_retained_total_branch_rows": v2["total_rows"],
            "qwen_gold_disagreement_branch_rows": sum(
                item["qwen_gold_disagreements"] for item in v2["branch_summaries"].values()
            ),
            "claims_supported": [
                "Study I evidence-condition verifier comparison uses human labels.",
                "Exact-retained revised claims inherit labels without semantic judgment.",
            ],
            "claims_not_supported": [
                "Human-gold factuality for semantically modified or added revised claims.",
                "A whole-response human-gold net factual gain score."
            ],
        },
        "CROSS_MODEL_SUPPORTED": {
            "scope": "targeted C-D blind Llama sensitivity",
            "alignment_calls": v6["targeted_alignment_calls"],
            "alignment_exact_agreements": v6["targeted_alignment_agreement"].get("exact_relation_agreement", 0),
            "alignment_disagreements": v6["targeted_alignment_agreement"].get("relation_disagreement", 0),
            "factuality_calls": v6["targeted_factuality_calls"],
            "cross_model_confirmed_total": v6["cross_model_confirmed_total"],
            "cross_model_confirmed_prediction_counts": v6["cross_model_confirmed_prediction_counts"],
            "claims_supported": [
                "A small set of revised claims receives independent support confirmation.",
                "Evaluator-family sensitivity is directly observable."
            ],
            "claims_not_supported": [
                "Balanced independent error confirmation.",
                "Formal net factual gain, because the Llama development calibration gate failed."
            ],
        },
        "SILVER_FULL_COVERAGE": {
            "scope": "full four-branch B2/B4/B6 evaluation with response-cluster uncertainty",
            "branch_terminal_summary": v8["branch_terminal_silver_summary"],
            "d_minus_c": dmc,
            "b_minus_a": bma,
            "standard_cove_funnel": v7["nested_funnel_counts"],
            "standard_cove_failure_taxonomy": v9["mechanism_taxonomy_counts"],
            "claims_supported": [
                "D has a response-cluster-stable same-protocol advantage over C.",
                "B's stable advantage over A is primarily factual preservation.",
                "Standard CoVe loses errors at multiple stages between question coverage and correction."
            ],
            "claims_not_supported": [
                "Human-gold net factual gain.",
                "Evaluator-independent causal attribution of each individual failure."
            ],
        },
    }
    summary = {
        "schema_version": "fcb_cove_three_layer_research_report_v1",
        "status": "complete",
        "split": split,
        "aggregation_rule": cfg["three_layer_policy"]["aggregation_rule"],
        "layers": layers,
        "overall_findings": [
            "External evidence improves claim-level verification on the human-labelled held-out cohort.",
            "Standard CoVe has limited strict correction and substantial collateral damage under the silver mechanism evaluation.",
            "Grounding verification answers is associated more clearly with preservation than with additional error correction.",
            "Bounded selective feedback outperforms an unguided extra revision under the same additional-call design in the silver evaluation.",
            "Exact retention exposes concrete same-model evaluator errors, while cross-model adjudication lacks refutation coverage.",
        ],
        "reporting_rule": (
            "Always name the layer beside a result. Never merge human, cross-model, "
            "and silver counts into one accuracy or net-gain number."
        ),
        "source_fingerprints": source_fingerprints(list(paths.values()) + [CONFIG_PATH]),
        "generated_at": utc_now(),
    }
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    json_path, md_path = three_layer_paths(split)
    atomic_write_json(json_path, summary)
    exact_disagreements = layers["HUMAN_ANCHORED"]["qwen_gold_disagreement_branch_rows"]
    lines = [
        "# V10 — Three-Layer CoVe Research Report",
        "",
        "No single score combines the three layers. Every conclusion below names "
        "the evidence strength that supports it.",
        "",
        "## Layer 1 — Human-anchored",
        "",
        f"- Initial binary human anchors: {cfg['expected_cohort']['binary_initial_claims']}.",
        f"- Exact-retained binary branch rows: {exact_binary}.",
        f"- Qwen–gold disagreement branch rows exposed by exact retention: {exact_disagreements}.",
        "- Valid use: claim-level verifier performance and exact-retained revised claims.",
        "- Invalid use: whole-response human-gold net factual gain.",
        "",
        "## Layer 2 — Cross-model supported",
        "",
        f"- Blind alignment agreement: {layers['CROSS_MODEL_SUPPORTED']['alignment_exact_agreements']}/"
        f"{layers['CROSS_MODEL_SUPPORTED']['alignment_calls']}.",
        f"- Cross-model-confirmed revised claims: {v6['cross_model_confirmed_total']} "
        f"`{v6['cross_model_confirmed_prediction_counts']}`.",
        "- All confirmed claims are factual; no NON_FACTUAL claim is independently confirmed.",
        "- This layer is auxiliary because the independent verifier failed its frozen development calibration gate.",
        "",
        "## Layer 3 — Silver full coverage",
        "",
        "| Branch | Strict corrections | Correction/deletion | Factual preserved | Factual damaged | Factual deleted | Added errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in cfg["active_branches"]:
        row = v8["branch_terminal_silver_summary"][branch]
        lines.append(
            f"| {BRANCH_LABELS[branch]} | {row['strict_correction']} | "
            f"{row['beneficial_error_disposition']} | {row['factual_preserved']} | "
            f"{row['factual_damaged']} | {row['factual_deleted']} | {row['added_errors']} |"
        )
    lines.extend([
        "",
        f"D minus C beneficial error disposition: {100*dmc['beneficial_error_disposition_rate']['point_estimate']:+.2f} pp "
        f"(95% CI {100*dmc['beneficial_error_disposition_rate']['lower']:+.2f} to "
        f"{100*dmc['beneficial_error_disposition_rate']['upper']:+.2f}).",
        "",
        f"D minus C factual preservation: {100*dmc['factual_preservation_rate']['point_estimate']:+.2f} pp "
        f"(95% CI {100*dmc['factual_preservation_rate']['lower']:+.2f} to "
        f"{100*dmc['factual_preservation_rate']['upper']:+.2f}).",
        "",
        "These are response-cluster-stable same-protocol silver effects, not "
        "human-gold net factual gain.",
        "",
        "## Defensible synthesis",
        "",
        "External evidence clearly improves claim verification. In CoVe, its most "
        "consistent downstream benefit is reducing collateral damage and preserving "
        "factual content. Selective evidence-based feedback also outperforms an "
        "unguided extra revision under the silver protocol. However, exact-gold "
        "coverage is small and the independent model fails to confirm errors, so a "
        "formal whole-response human-gold net-gain claim remains unsupported.",
        "",
    ])
    atomic_write_text(md_path, "\n".join(lines))
    print(json.dumps({
        "stage": "V10_three_layer_research_report",
        "status": "complete",
        "layers": {
            key: {
                "scope": value["scope"],
                "claims_supported": value["claims_supported"],
            }
            for key, value in layers.items()
        },
        "report": relative(md_path),
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one deterministic CoVe diagnostic-synthesis stage."
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--scope", choices=("full",), default="full")
    parser.add_argument("--split", choices=("heldout",), default="heldout")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "build-non-factual-funnel":
        return build_non_factual_funnel(args.split, args.dry_run)
    if args.stage == "analyze-evidence-stage-effects":
        return analyze_evidence_stage_effects(args.split, args.dry_run)
    if args.stage == "build-failure-taxonomy":
        return build_failure_taxonomy(args.split, args.dry_run)
    if args.stage == "build-three-layer-report":
        return build_three_layer_report(args.split, args.dry_run)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
