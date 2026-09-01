#!/usr/bin/env python3
"""Isolated post-hoc validation for the frozen CoVe branch experiment.

This module never changes A/B/C/D2 generation or B6 evaluation artifacts.  It
adds: paired response-cluster uncertainty, fallback sensitivity, a conservative
normalized-exact gold-inheritance layer, and a targeted blind Llama audit whose
semantic-alignment and factuality calls are separate and stateless.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ollama import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factcheck_bench_analysis import paired_response_cluster_bootstrap  # noqa: E402
from factcheck_bench_cove import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json_hash,
    cove_paths,
    load_json,
    load_jsonl,
    sha256_file,
    sha256_text,
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "factcheck_bench"
    / "cove"
    / "config"
    / "cove_validation_config.json"
)
SAMPLING_CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "factcheck_bench"
    / "cove"
    / "config"
    / "cove_validation_factuality_sampling_config.json"
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
BRANCH_LABELS = {"a": "A", "b": "B", "c": "C", "d2": "D"}
BENEFICIAL_ERROR_OUTCOMES = {
    "ERROR_CORRECTED_CANDIDATE",
    "ERROR_REMOVED_BY_DELETION_CANDIDATE",
}
PRESERVED_FACTUAL_OUTCOMES = {
    "FACTUAL_RETAINED_CANDIDATE",
    "FACTUAL_PRESERVED_AFTER_CHANGE_CANDIDATE",
}
UNRESOLVED_OUTCOMES = {
    "UNRESOLVED_UNKNOWN_INITIAL_ANCHOR",
    "UNRESOLVED_REVISED_FACTUALITY",
    "SILVER_LABEL_DISAGREEMENT_CANDIDATE",
}
ALIGNMENT_RELATIONS = {
    "EQUIVALENT",
    "MODIFIED",
    "PARTIAL",
    "PRESENT_UNEXTRACTED",
    "ABSENT",
}
PASSAGE_RELATIONS = {"SUPPORTS", "REFUTES", "INSUFFICIENT"}
STAGES = (
    "analyze-paired-statistics",
    "build-exact-retained",
    "prepare-targeted-audit",
    "run-targeted-alignment",
    "run-targeted-factuality",
    "analyze-targeted-validation",
)
MODEL_STAGES = {"run-targeted-alignment", "run-targeted-factuality"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def config() -> dict[str, Any]:
    value = load_json(CONFIG_PATH)
    if value.get("schema_version") != "fcb_cove_targeted_validation_config_v1":
        raise ValueError("Unexpected CoVe validation config schema")
    if value.get("split") != "heldout":
        raise ValueError("The validation layer is frozen to heldout")
    return value


def sampling_config() -> dict[str, Any]:
    value = load_json(SAMPLING_CONFIG_PATH)
    if value.get("schema_version") != "fcb_cove_targeted_factuality_sampling_config_v1":
        raise ValueError("Unexpected targeted factuality sampling config schema")
    return value


def branch_paths(branch: str):
    return cove_paths(PROJECT_ROOT, "full", branch)


def output_path(name: str, split: str, suffix: str = "jsonl") -> Path:
    directory = JSONL_DIR if suffix == "jsonl" else REPORTS_DIR
    return directory / f"{name}_{split}.{suffix}"


def exact_path(split: str) -> Path:
    return output_path("V2_exact_retained_claims", split)


def alignment_manifest_path(split: str) -> Path:
    return output_path("V3_targeted_alignment_manifest", split)


def factuality_manifest_path(split: str) -> Path:
    return output_path("V3_targeted_factuality_manifest", split)


def alignment_results_path(split: str) -> Path:
    return output_path("V4_blind_llama_alignment", split)


def factuality_results_path(split: str) -> Path:
    return output_path("V5_blind_llama_factuality", split)


def validation_tiers_path(split: str) -> Path:
    return output_path("V6_revised_claim_validation_tiers", split)


def normalized_exact_text(text: str) -> str:
    """Conservative normalized exact form; no punctuation/fuzzy semantics."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def stable_key(seed: int, stratum: str, stable_id: str) -> str:
    return sha256_text(f"{seed}|{stratum}|{stable_id}")


def load_branch_context(branch: str, split: str) -> dict[str, Any]:
    paths = branch_paths(branch)
    required = {
        "revisions": paths.revision_results(split),
        "extractions": paths.revised_claim_extraction_results(split),
        "alignments": paths.revised_claim_alignment_results(split),
        "evidence": paths.revised_claim_evidence(split),
        "factuality": paths.revised_claim_factuality_results(split),
        "audit": paths.factuality_audit_manifest(split),
        "initial": paths.initial_claim_outcomes(split),
        "added": paths.added_claim_outcomes(split),
    }
    missing = [relative(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing branch artifacts: {missing}")
    rows = {name: load_jsonl(path) for name, path in required.items()}
    if len(rows["revisions"]) != 72 or len(rows["initial"]) != 510:
        raise ValueError(
            f"Unexpected heldout branch cohort for {branch}: "
            f"responses={len(rows['revisions'])}, initial={len(rows['initial'])}"
        )
    claim_count = sum(
        len(row.get("revised_claims") or [])
        for row in rows["extractions"]
        if row.get("status") == "ok"
    )
    if claim_count != len(rows["factuality"]) or claim_count != len(rows["audit"]):
        raise ValueError(f"Incomplete revised-claim evaluation for {branch}")
    return {
        **rows,
        "paths": required,
        "initial_by_id": {row["initial_claim_id"]: row for row in rows["initial"]},
        "extraction_by_response": {
            row["response_id"]: row for row in rows["extractions"]
        },
        "alignment_by_response": {
            row["response_id"]: row for row in rows["alignments"]
        },
        "evidence_by_id": {
            row["revised_claim_id"]: row for row in rows["evidence"]
        },
        "factuality_by_id": {
            row["revised_claim_id"]: row for row in rows["factuality"]
        },
        "audit_by_id": {row["revised_claim_id"]: row for row in rows["audit"]},
    }


def load_all_contexts(split: str) -> dict[str, dict[str, Any]]:
    cfg = config()
    contexts = {
        branch: load_branch_context(branch, split)
        for branch in cfg["active_branches"]
    }
    sets = [set(value["initial_by_id"]) for value in contexts.values()]
    if any(item != sets[0] for item in sets[1:]):
        raise ValueError("Four branches do not share the exact initial-claim set")
    return contexts


def outcome_prediction(row: dict[str, Any]) -> str:
    label = row["human_label"]
    outcome = row["provisional_outcome"]
    if outcome in UNRESOLVED_OUTCOMES:
        return "UNKNOWN"
    if label == "NON_FACTUAL":
        return "NON_FACTUAL" if outcome in BENEFICIAL_ERROR_OUTCOMES else "FACTUAL"
    if label == "FACTUAL":
        return "FACTUAL" if outcome in PRESERVED_FACTUAL_OUTCOMES else "NON_FACTUAL"
    return "UNKNOWN"


def paired_contrast(
    baseline_rows: dict[str, dict[str, Any]],
    treatment_rows: dict[str, dict[str, Any]],
    *,
    excluded_response_ids: set[str] | None = None,
) -> dict[str, Any]:
    excluded = excluded_response_ids or set()
    ids = [
        claim_id
        for claim_id, row in baseline_rows.items()
        if row["response_id"] not in excluded
    ]
    if set(baseline_rows) != set(treatment_rows):
        raise ValueError("Paired contrast claim-ID mismatch")
    error_ids = [cid for cid in ids if baseline_rows[cid]["human_label"] == "NON_FACTUAL"]
    factual_ids = [cid for cid in ids if baseline_rows[cid]["human_label"] == "FACTUAL"]
    error_gains = sum(
        baseline_rows[cid]["provisional_outcome"] not in BENEFICIAL_ERROR_OUTCOMES
        and treatment_rows[cid]["provisional_outcome"] in BENEFICIAL_ERROR_OUTCOMES
        for cid in error_ids
    )
    error_regressions = sum(
        baseline_rows[cid]["provisional_outcome"] in BENEFICIAL_ERROR_OUTCOMES
        and treatment_rows[cid]["provisional_outcome"] not in BENEFICIAL_ERROR_OUTCOMES
        for cid in error_ids
    )
    factual_rescues = sum(
        baseline_rows[cid]["provisional_outcome"] not in PRESERVED_FACTUAL_OUTCOMES
        and treatment_rows[cid]["provisional_outcome"] in PRESERVED_FACTUAL_OUTCOMES
        for cid in factual_ids
    )
    factual_harms = sum(
        baseline_rows[cid]["provisional_outcome"] in PRESERVED_FACTUAL_OUTCOMES
        and treatment_rows[cid]["provisional_outcome"] not in PRESERVED_FACTUAL_OUTCOMES
        for cid in factual_ids
    )
    return {
        "paired_claim_count": len(ids),
        "response_count": len({baseline_rows[cid]["response_id"] for cid in ids}),
        "excluded_response_ids": sorted(excluded),
        "non_factual_claim_count": len(error_ids),
        "factual_claim_count": len(factual_ids),
        "error_adverse_to_beneficial": error_gains,
        "error_beneficial_to_adverse": error_regressions,
        "net_beneficial_error_change": error_gains - error_regressions,
        "factual_harm_to_preserved": factual_rescues,
        "factual_preserved_to_harm": factual_harms,
        "net_factual_preservation_change": factual_rescues - factual_harms,
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def branch_rate_cluster_bootstrap(
    contexts: dict[str, dict[str, Any]],
    comparisons: list[tuple[str, str, str]],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Bootstrap correction/deletion and preservation rates by response."""

    response_ids = sorted({row["response_id"] for row in contexts["a"]["initial"]})
    cluster_counts: dict[str, dict[str, tuple[int, int, int, int]]] = {}
    point_rates: dict[str, dict[str, float]] = {}
    for branch, context in contexts.items():
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in context["initial"]:
            grouped[row["response_id"]].append(row)
        branch_counts = {}
        for response_id in response_ids:
            rows = grouped[response_id]
            nf = [row for row in rows if row["human_label"] == "NON_FACTUAL"]
            factual = [row for row in rows if row["human_label"] == "FACTUAL"]
            branch_counts[response_id] = (
                sum(row["provisional_outcome"] in BENEFICIAL_ERROR_OUTCOMES for row in nf),
                len(nf),
                sum(row["provisional_outcome"] in PRESERVED_FACTUAL_OUTCOMES for row in factual),
                len(factual),
            )
        cluster_counts[branch] = branch_counts
        totals = [sum(values[index] for values in branch_counts.values()) for index in range(4)]
        point_rates[branch] = {
            "beneficial_error_disposition_rate": totals[0] / totals[1],
            "factual_preservation_rate": totals[2] / totals[3],
        }
    draws = {
        name: {
            "beneficial_error_disposition_rate": [],
            "factual_preservation_rate": [],
        }
        for name, _, _ in comparisons
    }
    rng = random.Random(seed)
    for _ in range(samples):
        selected = [rng.choice(response_ids) for _ in response_ids]
        rates: dict[str, dict[str, float]] = {}
        for branch in contexts:
            totals = [
                sum(cluster_counts[branch][response_id][index] for response_id in selected)
                for index in range(4)
            ]
            rates[branch] = {
                "beneficial_error_disposition_rate": totals[0] / totals[1],
                "factual_preservation_rate": totals[2] / totals[3],
            }
        for name, treatment, baseline in comparisons:
            for metric in draws[name]:
                draws[name][metric].append(rates[treatment][metric] - rates[baseline][metric])
    alpha = 1.0 - confidence_level
    paired = {}
    for name, treatment, baseline in comparisons:
        paired[name] = {
            "treatment": treatment,
            "baseline": baseline,
            "metrics": {},
        }
        for metric, values in draws[name].items():
            point = point_rates[treatment][metric] - point_rates[baseline][metric]
            lower = _percentile(values, alpha / 2.0)
            upper = _percentile(values, 1.0 - alpha / 2.0)
            paired[name]["metrics"][metric] = {
                "point_estimate": point,
                "lower": lower,
                "upper": upper,
                "includes_zero": lower <= 0.0 <= upper,
                "valid_replicates": len(values),
            }
    return {
        "method": "paired_response_cluster_percentile_bootstrap",
        "cluster_unit": "response_id",
        "response_cluster_count": len(response_ids),
        "samples": samples,
        "seed": seed,
        "confidence_level": confidence_level,
        "point_rates": point_rates,
        "paired_difference_intervals": paired,
    }


def analyze_paired_statistics(split: str, dry_run: bool) -> int:
    cfg = config()
    contexts = load_all_contexts(split)
    records = [
        {
            "claim_id": row["initial_claim_id"],
            "response_id": row["response_id"],
            "human_label": row["human_label"],
        }
        for row in contexts["a"]["initial"]
    ]
    result_sets = {
        branch: {
            claim_id: {"prediction": outcome_prediction(row), "status": "ok"}
            for claim_id, row in context["initial_by_id"].items()
            if row["human_label"] in {"FACTUAL", "NON_FACTUAL"}
        }
        for branch, context in contexts.items()
    }
    comparisons = [
        ("b_minus_a", "b", "a"),
        ("c_minus_a", "c", "a"),
        ("d_minus_a", "d2", "a"),
        ("b_minus_c", "b", "c"),
        ("d_minus_c", "d2", "c"),
        ("d_minus_b", "d2", "b"),
    ]
    settings = cfg["paired_statistics"]
    bootstrap = paired_response_cluster_bootstrap(
        records,
        result_sets,
        comparisons,
        samples=int(settings["bootstrap_samples"]),
        seed=int(settings["bootstrap_seed"]),
        confidence_level=float(settings["confidence_level"]),
    )
    rate_bootstrap = branch_rate_cluster_bootstrap(
        contexts,
        comparisons,
        samples=int(settings["bootstrap_samples"]),
        seed=int(settings["bootstrap_seed"]),
        confidence_level=float(settings["confidence_level"]),
    )
    fallback_ids = {
        branch: {
            row["response_id"]
            for row in context["revisions"]
            if row.get("fallback_applied") is True
        }
        for branch, context in contexts.items()
    }
    sensitivity: dict[str, Any] = {}
    for name, treatment, baseline in comparisons:
        all_result = paired_contrast(
            contexts[baseline]["initial_by_id"],
            contexts[treatment]["initial_by_id"],
        )
        excluded = fallback_ids[treatment] | fallback_ids[baseline]
        sensitivity[name] = {
            "all_heldout": all_result,
            "fallback_excluded": paired_contrast(
                contexts[baseline]["initial_by_id"],
                contexts[treatment]["initial_by_id"],
                excluded_response_ids=excluded,
            ),
        }
    report = {
        "schema_version": "fcb_cove_paired_branch_statistics_v1",
        "status": "complete",
        "split": split,
        "outcome_interpretation": {
            "NON_FACTUAL_recall": "beneficial correction-or-deletion rate",
            "FACTUAL_recall": "factual preservation rate",
            "balanced_accuracy": "mean of those two branch rates",
            "labels_remain": "same_protocol_silver_llm_assisted",
        },
        "paired_response_cluster_bootstrap": bootstrap,
        "branch_rate_cluster_bootstrap": rate_bootstrap,
        "fallback_response_ids": {
            branch: sorted(values) for branch, values in fallback_ids.items()
        },
        "fallback_excluded_sensitivity": sensitivity,
        "config_sha256": sha256_file(CONFIG_PATH),
        "generated_at": utc_now(),
    }
    if dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    json_path = output_path("V1_paired_branch_statistics", split, "json")
    md_path = output_path("V1_paired_branch_statistics", split, "md")
    atomic_write_json(json_path, report)
    lines = [
        "# V1 — Paired Branch Statistics",
        "",
        f"- Split: `{split}`",
        f"- Response clusters: {bootstrap['response_cluster_count']}",
        f"- Binary initial claims: {bootstrap['claim_count']}",
        f"- Bootstrap replicates: {bootstrap['samples']:,}",
        f"- Seed: `{bootstrap['seed']}`",
        "- Evaluation labels: same-protocol silver candidates",
        "",
        "| Contrast | Error-disposition delta (95% CI) | Preservation delta (95% CI) |",
        "|---|---:|---:|",
    ]
    for name, _, _ in comparisons:
        metrics = rate_bootstrap["paired_difference_intervals"][name]["metrics"]
        error = metrics["beneficial_error_disposition_rate"]
        factual = metrics["factual_preservation_rate"]
        lines.append(
            f"| `{name}` | {100*error['point_estimate']:+.2f} pp "
            f"[{100*error['lower']:+.2f}, {100*error['upper']:+.2f}] | "
            f"{100*factual['point_estimate']:+.2f} pp "
            f"[{100*factual['lower']:+.2f}, {100*factual['upper']:+.2f}] |"
        )
    lines.extend([
        "",
        "The bootstrap resamples complete responses, so claims from one long "
        "answer never enter different replicates independently. Intervals quantify "
        "response-sampling uncertainty; they do not correct silver-label error.",
        "",
        "Fallback-excluded counts are stored in the JSON companion and are a "
        "sensitivity analysis, not a replacement cohort.",
        "",
    ])
    atomic_write_text(md_path, "\n".join(lines))
    print(json.dumps({
        "stage": "V1_paired_branch_statistics",
        "status": "complete",
        "json": relative(json_path),
        "markdown": relative(md_path),
    }, indent=2))
    return 0


def build_exact_retained(split: str, dry_run: bool) -> int:
    cfg = config()
    contexts = load_all_contexts(split)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for branch, context in contexts.items():
        revised_by_response = {
            response_id: row.get("revised_claims") or []
            for response_id, row in context["extraction_by_response"].items()
        }
        seen_revised: set[str] = set()
        branch_rows: list[dict[str, Any]] = []
        for initial in context["initial"]:
            normalized = normalized_exact_text(initial["initial_claim"])
            candidates = [
                claim
                for claim in revised_by_response[initial["response_id"]]
                if normalized_exact_text(claim["claim"]) == normalized
            ]
            if len(candidates) > 1:
                raise ValueError(
                    f"Ambiguous normalized-exact match: {branch} "
                    f"{initial['initial_claim_id']}"
                )
            if not candidates:
                continue
            revised = candidates[0]
            revised_id = revised["claim_id"]
            if revised_id in seen_revised:
                raise ValueError(
                    f"One revised claim exact-matched multiple anchors: "
                    f"{branch} {revised_id}"
                )
            seen_revised.add(revised_id)
            human_label = initial["human_label"]
            tier = (
                "GOLD_INHERITED"
                if human_label in cfg["exact_retained"]["binary_human_labels_inherit"]
                else "UNRESOLVED"
            )
            qwen = context["factuality_by_id"][revised_id]
            record = {
                "schema_version": "fcb_cove_exact_retained_claim_v1",
                "branch": branch,
                "branch_evaluation_only": True,
                "response_id": initial["response_id"],
                "initial_claim_id": initial["initial_claim_id"],
                "initial_claim": initial["initial_claim"],
                "revised_claim_id": revised_id,
                "revised_claim": revised["claim"],
                "normalization_version": cfg["exact_retained"]["normalization_version"],
                "normalized_text_sha256": sha256_text(normalized),
                "human_label": human_label,
                "human_label_evaluation_only": True,
                "inherited_prediction": human_label if tier == "GOLD_INHERITED" else "UNKNOWN",
                "validation_tier": tier,
                "qwen_prediction_evaluation_only": qwen.get("prediction"),
                "qwen_agrees_with_inherited": (
                    qwen.get("prediction") == human_label
                    if tier == "GOLD_INHERITED"
                    else None
                ),
                "silver_provisional_outcome_evaluation_only": initial["provisional_outcome"],
                "exact_transition_disposition": (
                    "ERROR_RETAINED_GOLD"
                    if human_label == "NON_FACTUAL"
                    else (
                        "FACTUAL_RETAINED_GOLD"
                        if human_label == "FACTUAL"
                        else "UNKNOWN_ANCHOR_UNRESOLVED"
                    )
                ),
                "source_b6a_sha256": sha256_file(context["paths"]["extractions"]),
                "source_initial_outcomes_sha256": sha256_file(context["paths"]["initial"]),
            }
            branch_rows.append(record)
            rows.append(record)
        summaries[branch] = {
            "exact_retained_claims": len(branch_rows),
            "gold_inherited_binary": sum(
                row["validation_tier"] == "GOLD_INHERITED" for row in branch_rows
            ),
            "human_label_counts": dict(Counter(row["human_label"] for row in branch_rows)),
            "qwen_gold_disagreements": sum(
                row["qwen_agrees_with_inherited"] is False for row in branch_rows
            ),
            "silver_outcome_counts": dict(
                Counter(row["silver_provisional_outcome_evaluation_only"] for row in branch_rows)
            ),
            "silver_beneficial_error_candidates_invalidated": sum(
                row["human_label"] == "NON_FACTUAL"
                and row["silver_provisional_outcome_evaluation_only"]
                in BENEFICIAL_ERROR_OUTCOMES
                for row in branch_rows
            ),
            "silver_factual_harm_candidates_invalidated": sum(
                row["human_label"] == "FACTUAL"
                and row["silver_provisional_outcome_evaluation_only"]
                not in PRESERVED_FACTUAL_OUTCOMES
                for row in branch_rows
            ),
        }
    report = {
        "schema_version": "fcb_cove_exact_retained_summary_v1",
        "status": "complete",
        "split": split,
        "normalization": cfg["exact_retained"],
        "branch_summaries": summaries,
        "total_rows": len(rows),
        "interpretation": (
            "Only normalized-exact text retention inherits the initial human "
            "label. B6b EQUIVALENT and all semantic/fuzzy matches are excluded."
        ),
        "config_sha256": sha256_file(CONFIG_PATH),
        "generated_at": utc_now(),
    }
    if dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    atomic_write_jsonl(exact_path(split), rows)
    json_path = output_path("V2_exact_retained_summary", split, "json")
    md_path = output_path("V2_exact_retained_summary", split, "md")
    atomic_write_json(json_path, report)
    lines = [
        "# V2 — Exact-Retained Gold-Inheritance Layer",
        "",
        "Only NFKC/casefold/whitespace-normalized exact matches are accepted.",
        "Semantic equivalence, embeddings, fuzzy matching, and B6b labels are not used.",
        "",
        "| Branch | Exact retained | Binary gold inherited | Qwen disagreements | Silver factual harms invalidated |",
        "|---|---:|---:|---:|---:|",
    ]
    for branch in cfg["active_branches"]:
        item = summaries[branch]
        lines.append(
            f"| {BRANCH_LABELS[branch]} | {item['exact_retained_claims']} | "
            f"{item['gold_inherited_binary']} | {item['qwen_gold_disagreements']} | "
            f"{item['silver_factual_harm_candidates_invalidated']} |"
        )
    lines.extend([
        "",
        "A retained human-labelled NON_FACTUAL claim remains an error; exact "
        "retention is not correction. A retained FACTUAL claim is a gold-anchored "
        "preservation. UNKNOWN anchors remain unresolved.",
        "",
    ])
    atomic_write_text(md_path, "\n".join(lines))
    print(json.dumps({
        "stage": "V2_exact_retained",
        "status": "complete",
        "rows": len(rows),
        "branch_summaries": summaries,
        "output": relative(exact_path(split)),
    }, ensure_ascii=False, indent=2))
    return 0


def _stratified_sample(
    groups: dict[str, list[str]], total: int, seed: int, prefix: str
) -> list[str]:
    available = sum(len(values) for values in groups.values())
    if total >= available:
        return sorted({item for values in groups.values() for item in values})
    selected: list[str] = []
    remaining = total
    ordered_groups = sorted(groups)
    for index, name in enumerate(ordered_groups):
        values = sorted(
            set(groups[name]),
            key=lambda item: stable_key(seed, f"{prefix}:{name}", item),
        )
        if not values:
            continue
        groups_left = len(ordered_groups) - index
        if index == len(ordered_groups) - 1:
            take = min(remaining, len(values))
        else:
            proportional = round(total * len(values) / available)
            take = min(len(values), max(1, proportional))
            take = min(take, max(0, remaining - (groups_left - 1)))
        selected.extend(values[:take])
        remaining -= take
    if remaining:
        leftovers = sorted(
            {
                item for values in groups.values() for item in values
                if item not in selected
            },
            key=lambda item: stable_key(seed, f"{prefix}:remainder", item),
        )
        selected.extend(leftovers[:remaining])
    return selected[:total]


def prepare_targeted_audit(split: str, dry_run: bool) -> int:
    cfg = config()
    if not exact_path(split).exists():
        raise FileNotFoundError("Run build-exact-retained first")
    contexts = load_all_contexts(split)
    audit_cfg = cfg["targeted_audit"]
    seed = int(audit_cfg["selection_seed"])
    c_rows = contexts["c"]["initial_by_id"]
    d_rows = contexts["d2"]["initial_by_id"]
    non_factual_differences: list[str] = []
    factual_groups: dict[str, list[str]] = defaultdict(list)
    controls_by_label: dict[str, list[str]] = defaultdict(list)
    for claim_id, c_row in c_rows.items():
        d_row = d_rows[claim_id]
        label = c_row["human_label"]
        if label == "NON_FACTUAL":
            c_good = c_row["provisional_outcome"] in BENEFICIAL_ERROR_OUTCOMES
            d_good = d_row["provisional_outcome"] in BENEFICIAL_ERROR_OUTCOMES
            if c_good != d_good:
                non_factual_differences.append(claim_id)
            elif c_row["provisional_outcome"] == d_row["provisional_outcome"]:
                controls_by_label[label].append(claim_id)
        elif label == "FACTUAL":
            c_good = c_row["provisional_outcome"] in PRESERVED_FACTUAL_OUTCOMES
            d_good = d_row["provisional_outcome"] in PRESERVED_FACTUAL_OUTCOMES
            if c_good != d_good:
                factual_groups["D_RESCUE" if d_good else "D_HARM"].append(claim_id)
            elif c_row["provisional_outcome"] == d_row["provisional_outcome"]:
                controls_by_label[label].append(claim_id)
    factual_selected = _stratified_sample(
        dict(factual_groups),
        int(audit_cfg["factual_directional_difference_claim_sample"]),
        seed,
        "factual_difference",
    )
    control_selected = _stratified_sample(
        dict(controls_by_label),
        int(audit_cfg["stable_control_claim_sample"]),
        seed,
        "stable_control",
    )
    reasons_by_claim: defaultdict[str, set[str]] = defaultdict(set)
    for claim_id in non_factual_differences:
        reasons_by_claim[claim_id].add("PRIMARY_NON_FACTUAL_DIRECTIONAL_DIFFERENCE")
    for claim_id in factual_selected:
        reasons_by_claim[claim_id].add("PRIMARY_FACTUAL_DIRECTIONAL_DIFFERENCE_SAMPLE")
    for claim_id in control_selected:
        reasons_by_claim[claim_id].add("STABLE_CONTROL_SAMPLE")
    for branch in audit_cfg["primary_branches"]:
        for row in contexts[branch]["initial"]:
            if row.get("alignment_structure_recovery") or row.get("relation") == "PRESENT_UNEXTRACTED":
                reasons_by_claim[row["initial_claim_id"]].add(
                    "STRUCTURE_RECOVERY_OR_PRESENT_UNEXTRACTED"
                )
    alignment_units: list[dict[str, Any]] = []
    for claim_id in sorted(reasons_by_claim):
        for branch in audit_cfg["primary_branches"]:
            row = contexts[branch]["initial_by_id"][claim_id]
            extraction = contexts[branch]["extraction_by_response"][row["response_id"]]
            unit_id = "aln_" + sha256_text(f"{branch}|{claim_id}")[:20]
            alignment_units.append({
                "schema_version": "fcb_cove_targeted_alignment_unit_v1",
                "audit_unit_id": unit_id,
                "split": split,
                "branch_evaluation_only": branch,
                "selection_reasons_evaluation_only": sorted(reasons_by_claim[claim_id]),
                "response_id": row["response_id"],
                "initial_claim_id": claim_id,
                "initial_claim": row["initial_claim"],
                "revised_response": extraction["revised_response"],
                "revised_claims": [
                    {"claim_id": item["claim_id"], "claim": item["claim"]}
                    for item in extraction["revised_claims"]
                ],
                "human_label_evaluation_only": row["human_label"],
                "qwen_b6b_relation_evaluation_only": row["relation"],
                "qwen_b6b_revised_claim_ids_evaluation_only": row["revised_claim_ids"],
                "silver_outcome_evaluation_only": row["provisional_outcome"],
                "model_input_fields": cfg["leakage_policy"]["alignment_model_input_fields"],
                "withheld_fields": cfg["leakage_policy"]["alignment_withheld_fields"],
            })
    exact_revised = {
        (row["branch"], row["revised_claim_id"])
        for row in load_jsonl(exact_path(split))
        if row["validation_tier"] == "GOLD_INHERITED"
    }
    factuality_reasons: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for unit in alignment_units:
        branch = unit["branch_evaluation_only"]
        for revised_id in unit["qwen_b6b_revised_claim_ids_evaluation_only"]:
            for reason in unit["selection_reasons_evaluation_only"]:
                factuality_reasons[(branch, revised_id)].add(f"LINKED_TO_{reason}")
    for branch in audit_cfg["primary_branches"]:
        for row in contexts[branch]["added"]:
            if row["provisional_outcome"] == "NEW_ERROR_CANDIDATE":
                factuality_reasons[(branch, row["revised_claim_id"])].add(
                    "PRIMARY_BRANCH_NEW_ERROR_CANDIDATE"
                )
        for row in contexts[branch]["audit"]:
            if row["deterministic_flags"]:
                factuality_reasons[(branch, row["revised_claim_id"])].add(
                    "DETERMINISTIC_QWEN_POLICY_FLAG"
                )
    if audit_cfg["exclude_exact_retained_from_factuality_calls"]:
        for key in list(factuality_reasons):
            if key in exact_revised:
                del factuality_reasons[key]
    uncapped_factuality_candidates = len(factuality_reasons)
    sampling = sampling_config()
    reason_priority = list(sampling["reason_priority"])
    priority_index = {reason: index for index, reason in enumerate(reason_priority)}
    grouped_candidates: defaultdict[str, list[str]] = defaultdict(list)
    for branch, revised_id in factuality_reasons:
        reasons = factuality_reasons[(branch, revised_id)]
        primary_reason = min(
            reasons,
            key=lambda reason: (priority_index.get(reason, len(priority_index)), reason),
        )
        grouped_candidates[f"{branch}|{primary_reason}"].append(f"{branch}|{revised_id}")
    selected_keys = set(
        _stratified_sample(
            dict(grouped_candidates),
            int(sampling["maximum_factuality_units"]),
            int(sampling["sampling_seed"]),
            "targeted_factuality_compute_budget",
        )
    )
    factuality_reasons = defaultdict(
        set,
        {
            key: reasons
            for key, reasons in factuality_reasons.items()
            if f"{key[0]}|{key[1]}" in selected_keys
        },
    )
    factuality_units: list[dict[str, Any]] = []
    for branch, revised_id in sorted(factuality_reasons):
        context = contexts[branch]
        evidence = context["evidence_by_id"][revised_id]
        qwen = context["factuality_by_id"][revised_id]
        audit = context["audit_by_id"][revised_id]
        unit_id = "fac_" + sha256_text(f"{branch}|{revised_id}")[:20]
        factuality_units.append({
            "schema_version": "fcb_cove_targeted_factuality_unit_v1",
            "audit_unit_id": unit_id,
            "split": split,
            "branch_evaluation_only": branch,
            "selection_reasons_evaluation_only": sorted(
                factuality_reasons[(branch, revised_id)]
            ),
            "response_id": qwen["response_id"],
            "revised_claim_id": revised_id,
            "revised_claim": qwen["revised_claim"],
            "evidence_items": evidence["items"],
            "evidence_normalized_text": evidence["normalized_text"],
            "evidence_normalized_sha256": evidence["normalized_sha256"],
            "qwen_prediction_evaluation_only": qwen["prediction"],
            "qwen_confidence_evaluation_only": qwen["confidence"],
            "qwen_rationale_evaluation_only": qwen["rationale"],
            "qwen_deterministic_flags_evaluation_only": audit["deterministic_flags"],
            "model_input_fields": cfg["leakage_policy"]["factuality_model_input_fields"],
            "withheld_fields": cfg["leakage_policy"]["factuality_withheld_fields"],
        })
    summary = {
        "schema_version": "fcb_cove_targeted_audit_manifest_summary_v1",
        "status": "complete",
        "split": split,
        "primary_comparison": cfg["primary_comparison"],
        "selection": {
            "non_factual_directional_difference_claims": len(non_factual_differences),
            "factual_directional_difference_available": sum(map(len, factual_groups.values())),
            "factual_directional_difference_selected": len(factual_selected),
            "stable_control_selected": len(control_selected),
            "unique_initial_claims": len(reasons_by_claim),
            "alignment_units": len(alignment_units),
            "factuality_units": len(factuality_units),
            "uncapped_factuality_candidates": uncapped_factuality_candidates,
            "factuality_compute_cap": int(sampling["maximum_factuality_units"]),
        },
        "alignment_selection_reason_counts": dict(Counter(
            reason for row in alignment_units
            for reason in row["selection_reasons_evaluation_only"]
        )),
        "factuality_selection_reason_counts": dict(Counter(
            reason for row in factuality_units
            for reason in row["selection_reasons_evaluation_only"]
        )),
        "isolation": {
            "alignment_and_factuality_manifests_built_together_before_model_calls": True,
            "factuality_selection_does_not_read_llama_alignment": True,
            "branch_identity_withheld_from_model": True,
            "qwen_and_gold_fields_withheld_from_model": True,
            "original_branch_artifacts_modified": False,
        },
        "config_sha256": sha256_file(CONFIG_PATH),
        "factuality_sampling_config_sha256": sha256_file(SAMPLING_CONFIG_PATH),
        "generated_at": utc_now(),
    }
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    atomic_write_jsonl(alignment_manifest_path(split), alignment_units)
    atomic_write_jsonl(factuality_manifest_path(split), factuality_units)
    json_path = output_path("V3_targeted_audit_manifest_summary", split, "json")
    md_path = output_path("V3_targeted_audit_manifest_summary", split, "md")
    atomic_write_json(json_path, summary)
    atomic_write_text(md_path, "\n".join([
        "# V3 — Targeted Blind-Audit Manifest",
        "",
        f"- Primary contrast: C versus D",
        f"- Unique initial claims: {len(reasons_by_claim)}",
        f"- Independent alignment calls: {len(alignment_units)}",
        f"- Independent factuality calls: {len(factuality_units)}",
        "",
        "The two manifests are frozen before either model stage runs. Alignment "
        "cannot select factuality inputs, and factuality cannot affect alignment.",
        "Branch identity, human labels, Qwen outputs, and provisional outcomes are "
        "stored only for later joins and are absent from both prompts.",
        "",
    ]))
    print(json.dumps({
        "stage": "V3_targeted_audit_manifest",
        "status": "complete",
        **summary["selection"],
        "alignment_manifest": relative(alignment_manifest_path(split)),
        "factuality_manifest": relative(factuality_manifest_path(split)),
    }, indent=2))
    return 0


def response_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def preflight_model(cfg: dict[str, Any], host: str) -> tuple[Client, str]:
    settings = cfg["independent_adjudicator"]
    client = Client(host=host, timeout=float(settings["timeout_seconds"]))
    response = client.list()
    available: dict[str, str | None] = {}
    for item in response_value(response, "models") or []:
        name = response_value(item, "model") or response_value(item, "name")
        if isinstance(name, str):
            available[name] = response_value(item, "digest")
    model = settings["model"]
    if model not in available:
        raise ValueError(f"Independent model not installed: {model}")
    digest = available[model]
    if digest != settings["expected_model_digest"]:
        raise ValueError(
            f"Independent model digest changed: expected={settings['expected_model_digest']} "
            f"actual={digest}"
        )
    return client, str(digest)


def alignment_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["relation", "matched_revised_claim_ids", "rationale"],
        "properties": {
            "relation": {"type": "string", "enum": sorted(ALIGNMENT_RELATIONS)},
            "matched_revised_claim_ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "rationale": {"type": "string", "minLength": 3, "maxLength": 300},
        },
    }


def factuality_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["passage_assessments"],
        "properties": {
            "passage_assessments": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["passage_rank", "relation", "rationale"],
                    "properties": {
                        "passage_rank": {"type": "integer", "minimum": 1, "maximum": 5},
                        "relation": {"type": "string", "enum": sorted(PASSAGE_RELATIONS)},
                        "rationale": {"type": "string", "minLength": 3, "maxLength": 220},
                    },
                },
            }
        },
    }


def parse_alignment(raw: str, unit: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {
        "relation", "matched_revised_claim_ids", "rationale"
    }:
        raise ValueError("Alignment output has unexpected fields")
    relation = value["relation"]
    ids = value["matched_revised_claim_ids"]
    rationale = " ".join(str(value["rationale"]).split())
    allowed = {row["claim_id"] for row in unit["revised_claims"]}
    if relation not in ALIGNMENT_RELATIONS:
        raise ValueError("Invalid alignment relation")
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or not set(ids) <= allowed:
        raise ValueError("Invalid matched revised-claim IDs")
    if relation in {"EQUIVALENT", "MODIFIED", "PARTIAL"} and not ids:
        raise ValueError(f"{relation} requires a matched revised claim")
    if relation in {"PRESENT_UNEXTRACTED", "ABSENT"} and ids:
        raise ValueError(f"{relation} requires an empty matched ID list")
    if not 3 <= len(rationale) <= 300:
        raise ValueError("Invalid alignment rationale")
    return {"relation": relation, "matched_revised_claim_ids": ids, "rationale": rationale}


def parse_factuality(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"passage_assessments"}:
        raise ValueError("Factuality output has unexpected fields")
    rows = value["passage_assessments"]
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("Exactly five passage assessments are required")
    normalized = []
    for row in sorted(rows, key=lambda item: item.get("passage_rank", 0)):
        if set(row) != {"passage_rank", "relation", "rationale"}:
            raise ValueError("Invalid passage-assessment fields")
        rank = row["passage_rank"]
        relation = row["relation"]
        rationale = " ".join(str(row["rationale"]).split())
        if type(rank) is not int or relation not in PASSAGE_RELATIONS:
            raise ValueError("Invalid passage assessment")
        if not 3 <= len(rationale) <= 220:
            raise ValueError("Invalid passage rationale")
        normalized.append({"passage_rank": rank, "relation": relation, "rationale": rationale})
    if [row["passage_rank"] for row in normalized] != [1, 2, 3, 4, 5]:
        raise ValueError("Passage ranks must be 1 through 5")
    relations = {row["relation"] for row in normalized}
    if "SUPPORTS" in relations and "REFUTES" not in relations:
        prediction, status = "FACTUAL", "DIRECT_SUPPORT"
    elif "REFUTES" in relations and "SUPPORTS" not in relations:
        prediction, status = "NON_FACTUAL", "DIRECT_REFUTATION"
    elif "SUPPORTS" in relations and "REFUTES" in relations:
        prediction, status = "UNKNOWN", "CONFLICTING_DIRECT_EVIDENCE"
    else:
        prediction, status = "UNKNOWN", "INSUFFICIENT_DIRECT_EVIDENCE"
    return {
        "passage_assessments": normalized,
        "prediction": prediction,
        "evidence_status": status,
    }


def call_model(
    client: Client,
    cfg: dict[str, Any],
    prompt: str,
    schema: dict[str, Any],
    num_predict: int,
) -> tuple[str, dict[str, Any]]:
    settings = cfg["independent_adjudicator"]
    response = client.chat(
        model=settings["model"],
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        think=False,
        format=schema,
        options={
            "temperature": settings["temperature"],
            "seed": settings["seed"],
            "num_predict": num_predict,
        },
    )
    message = response_value(response, "message")
    content = response_value(message, "content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned no message content")
    metadata = {
        key: response_value(response, key)
        for key in (
            "model", "created_at", "done", "done_reason", "total_duration",
            "load_duration", "prompt_eval_count", "prompt_eval_duration",
            "eval_count", "eval_duration",
        )
        if response_value(response, key) is not None
    }
    return content.strip(), metadata


def run_model_stage(
    split: str,
    stage: str,
    *,
    resume: bool,
    dry_run: bool,
    host: str,
) -> int:
    cfg = config()
    settings = cfg["independent_adjudicator"]
    if stage == "alignment":
        manifest_path = alignment_manifest_path(split)
        results_path = alignment_results_path(split)
        prompt_path = PROJECT_ROOT / settings["alignment_prompt_path"]
        schema = alignment_schema()
        num_predict = int(settings["alignment_num_predict"])
    else:
        manifest_path = factuality_manifest_path(split)
        results_path = factuality_results_path(split)
        prompt_path = PROJECT_ROOT / settings["factuality_prompt_path"]
        schema = factuality_schema()
        num_predict = int(settings["factuality_num_predict"])
    if not manifest_path.exists():
        raise FileNotFoundError("Run prepare-targeted-audit first")
    units = load_jsonl(manifest_path)
    template = prompt_path.read_text(encoding="utf-8")
    placeholders = (
        {"{initial_claim_json}", "{revised_response_json}", "{revised_claims_json}"}
        if stage == "alignment"
        else {"{revised_claim_json}", "{retrieved_evidence_text}"}
    )
    if any(template.count(item) != 1 for item in placeholders):
        raise ValueError(f"Prompt placeholders invalid: {prompt_path}")
    if dry_run:
        print(json.dumps({
            "stage": f"targeted_{stage}",
            "unit_count": len(units),
            "model": settings["model"],
            "manifest": relative(manifest_path),
            "heldout_original_artifacts_touched": False,
        }, indent=2))
        return 0
    client, digest = preflight_model(cfg, host)
    run_payload = {
        "stage": f"targeted_blind_llama_{stage}_v1",
        "split": split,
        "model": settings["model"],
        "model_digest": digest,
        "temperature": settings["temperature"],
        "seed": settings["seed"],
        "num_predict": num_predict,
        "prompt_sha256": sha256_file(prompt_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(CONFIG_PATH),
        "schema_sha256": canonical_json_hash(schema),
    }
    if stage == "factuality":
        run_payload["factuality_sampling_config_sha256"] = sha256_file(
            SAMPLING_CONFIG_PATH
        )
    fingerprint = canonical_json_hash(run_payload)
    existing = load_jsonl(results_path) if results_path.exists() else []
    if existing and not resume:
        raise FileExistsError(f"Use --resume for existing output: {results_path}")
    expected_ids = {row["audit_unit_id"] for row in units}
    if len(expected_ids) != len(units):
        raise ValueError("Audit manifest contains duplicate unit IDs")
    by_id: dict[str, dict[str, Any]] = {}
    for row in existing:
        unit_id = row.get("audit_unit_id")
        if unit_id not in expected_ids or unit_id in by_id:
            raise ValueError(f"Unexpected/duplicate existing audit row: {unit_id}")
        if row.get("run_fingerprint") != fingerprint:
            raise ValueError(f"Existing audit fingerprint mismatch: {unit_id}")
        by_id[unit_id] = row
    pending = [row for row in units if by_id.get(row["audit_unit_id"], {}).get("status") != "ok"]
    print(
        f"Targeted blind {stage}: total={len(units)}, "
        f"retained_ok={len(units)-len(pending)}, pending={len(pending)}",
        flush=True,
    )
    order = {row["audit_unit_id"]: index + 1 for index, row in enumerate(units)}
    consecutive = 0
    for unit in pending:
        unit_id = unit["audit_unit_id"]
        position = order[unit_id]
        print(f"[{position}/{len(units)}] {unit_id} {stage} ...", flush=True)
        if stage == "alignment":
            prompt = (
                template.replace("{initial_claim_json}", json.dumps(unit["initial_claim"], ensure_ascii=False))
                .replace("{revised_response_json}", json.dumps(unit["revised_response"], ensure_ascii=False))
                .replace("{revised_claims_json}", json.dumps(unit["revised_claims"], ensure_ascii=False))
            )
            parser = lambda raw: parse_alignment(raw, unit)
        else:
            prompt = (
                template.replace("{revised_claim_json}", json.dumps(unit["revised_claim"], ensure_ascii=False))
                .replace("{retrieved_evidence_text}", unit["evidence_normalized_text"])
            )
            parser = parse_factuality
        raw: str | None = None
        metadata: dict[str, Any] = {}
        error: Exception | None = None
        attempts = 0
        started = time.perf_counter()
        for attempt in range(int(settings["max_retries"]) + 1):
            attempts = attempt + 1
            try:
                raw, metadata = call_model(client, cfg, prompt, schema, num_predict)
                error = None
                break
            except Exception as caught:
                error = caught
                if attempt < int(settings["max_retries"]):
                    time.sleep(1.0)
        result = {
            "schema_version": f"fcb_cove_targeted_{stage}_result_v1",
            "audit_unit_id": unit_id,
            "split": split,
            "branch_evaluation_only": unit["branch_evaluation_only"],
            "response_id": unit["response_id"],
            "model_input_fields": unit["model_input_fields"],
            "withheld_fields": unit["withheld_fields"],
            **run_payload,
            "run_fingerprint": fingerprint,
            "attempts": attempts,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "raw_model_output": raw,
            "ollama_metadata": metadata,
            "created_at": utc_now(),
        }
        if stage == "alignment":
            result["initial_claim_id"] = unit["initial_claim_id"]
        else:
            result["revised_claim_id"] = unit["revised_claim_id"]
        if error is not None:
            result.update({"status": "request_error", "error": f"{type(error).__name__}: {error}"})
        else:
            try:
                parsed = parser(raw or "")
                result.update({"status": "ok", "error": None, **parsed})
            except Exception as caught:
                result.update({"status": "parse_error", "error": f"{type(caught).__name__}: {caught}"})
        by_id[unit_id] = result
        atomic_write_jsonl(results_path, [by_id[row["audit_unit_id"]] for row in units if row["audit_unit_id"] in by_id])
        if result["status"] == "ok":
            consecutive = 0
            detail = result["relation"] if stage == "alignment" else result["prediction"]
            print(f"[{position}/{len(units)}] success, {detail}, {result['latency_seconds']:.1f}s", flush=True)
        else:
            consecutive = consecutive + 1 if result["status"] == "request_error" else 0
            print(f"[{position}/{len(units)}] {result['status']}: {result['error']}", flush=True)
            if consecutive >= int(settings["max_consecutive_request_errors"]):
                print("Stopping after consecutive request errors; rerun with --resume.", file=sys.stderr, flush=True)
                break
    status_counts = Counter(by_id.get(row["audit_unit_id"], {}).get("status", "missing") for row in units)
    print(json.dumps({
        "stage": f"targeted_blind_{stage}",
        "unit_count": len(units),
        "status_counts": dict(status_counts),
        "output": relative(results_path),
    }, indent=2))
    return 0 if status_counts == Counter({"ok": len(units)}) else 2


def derive_target_transition(
    human_label: str,
    alignment: dict[str, Any],
    tier_by_revised: dict[tuple[str, str], dict[str, Any]],
    branch: str,
    *,
    eligible_tiers: set[str] | None = None,
    require_cross_model_absence: bool = False,
    qwen_relation: str | None = None,
) -> str:
    relation = alignment["relation"]
    if relation == "ABSENT":
        if require_cross_model_absence and qwen_relation != "ABSENT":
            return "UNRESOLVED"
        return "BENEFICIAL" if human_label == "NON_FACTUAL" else "HARMFUL"
    if relation == "PRESENT_UNEXTRACTED":
        return "UNRESOLVED"
    selected_rows = [
        tier_by_revised.get((branch, revised_id), {})
        for revised_id in alignment["matched_revised_claim_ids"]
    ]
    if eligible_tiers is not None and any(
        row.get("validation_tier") not in eligible_tiers for row in selected_rows
    ):
        return "UNRESOLVED"
    labels = [row.get("resolved_prediction") for row in selected_rows]
    if not labels or any(label not in {"FACTUAL", "NON_FACTUAL"} for label in labels):
        return "UNRESOLVED"
    all_factual = all(label == "FACTUAL" for label in labels)
    if human_label == "NON_FACTUAL":
        return "BENEFICIAL" if all_factual and relation != "EQUIVALENT" else "ADVERSE"
    if human_label == "FACTUAL":
        return "PRESERVED" if all_factual and relation in {"EQUIVALENT", "MODIFIED"} else "HARMFUL"
    return "UNRESOLVED"


def analyze_targeted_validation(split: str, dry_run: bool) -> int:
    cfg = config()
    contexts = load_all_contexts(split)
    for path in (exact_path(split), alignment_results_path(split), factuality_results_path(split)):
        if not path.exists():
            raise FileNotFoundError(path)
    exact_rows = load_jsonl(exact_path(split))
    exact_by_revised = {
        (row["branch"], row["revised_claim_id"]): row for row in exact_rows
    }
    llama_factuality = {
        (row["branch_evaluation_only"], row["revised_claim_id"]): row
        for row in load_jsonl(factuality_results_path(split))
    }
    tier_rows: list[dict[str, Any]] = []
    for branch, context in contexts.items():
        for qwen in context["factuality"]:
            revised_id = qwen["revised_claim_id"]
            key = (branch, revised_id)
            exact = exact_by_revised.get(key)
            independent = llama_factuality.get(key)
            flags = context["audit_by_id"][revised_id]["deterministic_flags"]
            if exact and exact["validation_tier"] == "GOLD_INHERITED":
                tier = "GOLD_INHERITED"
                resolved = exact["inherited_prediction"]
                basis = "normalized_exact_initial_claim_human_label"
            elif (
                independent
                and independent.get("status") == "ok"
                and independent.get("prediction") in {"FACTUAL", "NON_FACTUAL"}
                and independent.get("prediction") == qwen.get("prediction")
                and not flags
            ):
                tier = "CROSS_MODEL_CONFIRMED"
                resolved = qwen["prediction"]
                basis = "qwen_llama_exact_label_agreement_without_policy_flag"
            elif qwen.get("status") == "ok" and qwen.get("prediction") in {"FACTUAL", "NON_FACTUAL"}:
                tier = "SILVER_ONLY"
                resolved = qwen["prediction"]
                basis = "qwen_only_or_independent_not_confirming"
            else:
                tier = "UNRESOLVED"
                resolved = "UNKNOWN"
                basis = "no_decisive_eligible_prediction"
            tier_rows.append({
                "schema_version": "fcb_cove_revised_claim_validation_tier_v1",
                "branch": branch,
                "response_id": qwen["response_id"],
                "revised_claim_id": revised_id,
                "revised_claim": qwen["revised_claim"],
                "validation_tier": tier,
                "resolved_prediction": resolved,
                "resolution_basis": basis,
                "qwen_prediction": qwen.get("prediction"),
                "qwen_deterministic_flags": flags,
                "llama_targeted": independent is not None,
                "llama_status": independent.get("status") if independent else None,
                "llama_prediction": independent.get("prediction") if independent else None,
                "llama_evidence_status": independent.get("evidence_status") if independent else None,
                "exact_retained_initial_claim_id": exact.get("initial_claim_id") if exact else None,
                "formal_human_gold_net_gain_eligible": tier == "GOLD_INHERITED",
            })
    tier_by_revised = {(row["branch"], row["revised_claim_id"]): row for row in tier_rows}
    alignment_manifest = {row["audit_unit_id"]: row for row in load_jsonl(alignment_manifest_path(split))}
    alignment_rows = load_jsonl(alignment_results_path(split))
    agreement = Counter()
    transition_rows: list[dict[str, Any]] = []
    for result in alignment_rows:
        unit = alignment_manifest[result["audit_unit_id"]]
        if result.get("status") != "ok":
            agreement["technical_or_parse_failure"] += 1
            continue
        if result["relation"] == unit["qwen_b6b_relation_evaluation_only"]:
            agreement["exact_relation_agreement"] += 1
        else:
            agreement["relation_disagreement"] += 1
        branch = unit["branch_evaluation_only"]
        transition_rows.append({
            "branch": branch,
            "initial_claim_id": unit["initial_claim_id"],
            "human_label": unit["human_label_evaluation_only"],
            "selection_reasons": unit["selection_reasons_evaluation_only"],
            "llama_relation": result["relation"],
            "llama_matched_revised_claim_ids": result["matched_revised_claim_ids"],
            "qwen_relation": unit["qwen_b6b_relation_evaluation_only"],
            "qwen_silver_outcome": unit["silver_outcome_evaluation_only"],
            "silver_assisted_transition_state": derive_target_transition(
                unit["human_label_evaluation_only"], result, tier_by_revised, branch
            ),
            "strong_tier_transition_state": derive_target_transition(
                unit["human_label_evaluation_only"],
                result,
                tier_by_revised,
                branch,
                eligible_tiers={"GOLD_INHERITED", "CROSS_MODEL_CONFIRMED"},
                require_cross_model_absence=True,
                qwen_relation=unit["qwen_b6b_relation_evaluation_only"],
            ),
        })
    by_claim_branch = {(row["initial_claim_id"], row["branch"]): row for row in transition_rows}
    paired_targeted = Counter()
    paired_strong = Counter()
    paired_resolved = 0
    paired_strong_resolved = 0
    for claim_id in sorted({row["initial_claim_id"] for row in transition_rows}):
        c = by_claim_branch.get((claim_id, "c"))
        d = by_claim_branch.get((claim_id, "d2"))
        if not c or not d:
            continue
        if c["human_label"] not in {"FACTUAL", "NON_FACTUAL"}:
            paired_targeted["unknown_anchor_pair"] += 1
            paired_strong["unknown_anchor_pair"] += 1
            continue
        c_state, d_state = c["silver_assisted_transition_state"], d["silver_assisted_transition_state"]
        if "UNRESOLVED" in {c_state, d_state}:
            paired_targeted["unresolved_pair"] += 1
        else:
            paired_resolved += 1
            if c["human_label"] == "NON_FACTUAL":
                if c_state != "BENEFICIAL" and d_state == "BENEFICIAL":
                    paired_targeted["D_error_gain"] += 1
                elif c_state == "BENEFICIAL" and d_state != "BENEFICIAL":
                    paired_targeted["D_error_regression"] += 1
                else:
                    paired_targeted["same_error_disposition"] += 1
            elif c["human_label"] == "FACTUAL":
                if c_state != "PRESERVED" and d_state == "PRESERVED":
                    paired_targeted["D_factual_rescue"] += 1
                elif c_state == "PRESERVED" and d_state != "PRESERVED":
                    paired_targeted["D_factual_harm"] += 1
                else:
                    paired_targeted["same_factual_disposition"] += 1
        c_strong = c["strong_tier_transition_state"]
        d_strong = d["strong_tier_transition_state"]
        if "UNRESOLVED" in {c_strong, d_strong}:
            paired_strong["unresolved_pair"] += 1
        else:
            paired_strong_resolved += 1
            if c["human_label"] == "NON_FACTUAL":
                if c_strong != "BENEFICIAL" and d_strong == "BENEFICIAL":
                    paired_strong["D_error_gain"] += 1
                elif c_strong == "BENEFICIAL" and d_strong != "BENEFICIAL":
                    paired_strong["D_error_regression"] += 1
                else:
                    paired_strong["same_error_disposition"] += 1
            elif c["human_label"] == "FACTUAL":
                if c_strong != "PRESERVED" and d_strong == "PRESERVED":
                    paired_strong["D_factual_rescue"] += 1
                elif c_strong == "PRESERVED" and d_strong != "PRESERVED":
                    paired_strong["D_factual_harm"] += 1
                else:
                    paired_strong["same_factual_disposition"] += 1
    branch_tiers = {
        branch: dict(Counter(row["validation_tier"] for row in tier_rows if row["branch"] == branch))
        for branch in cfg["active_branches"]
    }
    alignment_relation_matrix = Counter(
        (
            alignment_manifest[row["audit_unit_id"]][
                "qwen_b6b_relation_evaluation_only"
            ],
            row.get("relation"),
        )
        for row in alignment_rows
        if row.get("status") == "ok"
    )
    factuality_manifest = {
        row["audit_unit_id"]: row
        for row in load_jsonl(factuality_manifest_path(split))
    }
    factuality_label_matrix = Counter(
        (
            factuality_manifest[row["audit_unit_id"]][
                "qwen_prediction_evaluation_only"
            ],
            row.get("prediction"),
        )
        for row in llama_factuality.values()
        if row.get("status") == "ok"
    )
    summary = {
        "schema_version": "fcb_cove_targeted_validation_summary_v1",
        "status": "complete",
        "split": split,
        "tier_policy": cfg["tier_policy"],
        "branch_tier_counts": branch_tiers,
        "exact_gold_inherited_total": sum(row["validation_tier"] == "GOLD_INHERITED" for row in tier_rows),
        "cross_model_confirmed_total": sum(row["validation_tier"] == "CROSS_MODEL_CONFIRMED" for row in tier_rows),
        "cross_model_confirmed_prediction_counts": dict(Counter(
            row["resolved_prediction"]
            for row in tier_rows
            if row["validation_tier"] == "CROSS_MODEL_CONFIRMED"
        )),
        "targeted_factuality_calls": len(llama_factuality),
        "targeted_factuality_prediction_counts": dict(Counter(
            row.get("prediction") for row in llama_factuality.values()
        )),
        "targeted_alignment_calls": len(alignment_rows),
        "targeted_alignment_agreement": dict(agreement),
        "targeted_alignment_agreement_rate": (
            agreement.get("exact_relation_agreement", 0) / len(alignment_rows)
            if alignment_rows else None
        ),
        "targeted_alignment_llama_relation_counts": dict(Counter(
            row.get("relation")
            for row in alignment_rows
            if row.get("status") == "ok"
        )),
        "targeted_alignment_qwen_by_llama_matrix": {
            f"{qwen}->{llama}": count
            for (qwen, llama), count in sorted(alignment_relation_matrix.items())
        },
        "targeted_factuality_qwen_by_llama_matrix": {
            f"{qwen}->{llama}": count
            for (qwen, llama), count in sorted(factuality_label_matrix.items())
        },
        "targeted_transition_rows": len(transition_rows),
        "paired_silver_assisted_resolved_claims": paired_resolved,
        "paired_silver_assisted_transition_counts": dict(paired_targeted),
        "paired_strong_tier_resolved_claims": paired_strong_resolved,
        "paired_strong_tier_transition_counts": dict(paired_strong),
        "calibration_boundary": {
            "llama_development_gate_passed": cfg["independent_adjudicator"]["development_calibration_gate_passed"],
            "cross_model_layer_is": cfg["independent_adjudicator"]["interpretation"],
            "formal_net_gain_eligible": False,
            "reason": (
                "The Llama factuality protocol failed the preregistered dev gate; "
                "targeted agreement is a sensitivity layer, not a gold replacement."
            ),
        },
        "source_fingerprints": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "factuality_sampling_config_sha256": sha256_file(SAMPLING_CONFIG_PATH),
            "exact_retained_sha256": sha256_file(exact_path(split)),
            "alignment_manifest_sha256": sha256_file(alignment_manifest_path(split)),
            "factuality_manifest_sha256": sha256_file(factuality_manifest_path(split)),
            "alignment_results_sha256": sha256_file(alignment_results_path(split)),
            "factuality_results_sha256": sha256_file(factuality_results_path(split)),
        },
        "generated_at": utc_now(),
    }
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    atomic_write_jsonl(validation_tiers_path(split), tier_rows)
    atomic_write_jsonl(output_path("V6_targeted_transition_diagnostics", split), transition_rows)
    json_path = output_path("V6_targeted_validation_summary", split, "json")
    md_path = output_path("V6_targeted_validation_summary", split, "md")
    atomic_write_json(json_path, summary)
    lines = [
        "# V6 — Targeted Validation and Evidence Tiers",
        "",
        "| Branch | Gold inherited | Cross-model confirmed | Silver only | Unresolved |",
        "|---|---:|---:|---:|---:|",
    ]
    for branch in cfg["active_branches"]:
        counts = branch_tiers[branch]
        lines.append(
            f"| {BRANCH_LABELS[branch]} | {counts.get('GOLD_INHERITED',0)} | "
            f"{counts.get('CROSS_MODEL_CONFIRMED',0)} | {counts.get('SILVER_ONLY',0)} | "
            f"{counts.get('UNRESOLVED',0)} |"
        )
    lines.extend([
        "",
        f"- Blind alignment exact-relation agreement: "
        f"{agreement.get('exact_relation_agreement',0)}/{len(alignment_rows)}",
        f"- Llama factuality predictions: `{dict(Counter(row.get('prediction') for row in llama_factuality.values()))}`",
        f"- Cross-model confirmed labels: `{dict(Counter(row['resolved_prediction'] for row in tier_rows if row['validation_tier'] == 'CROSS_MODEL_CONFIRMED'))}`",
        f"- Silver-assisted paired claims with resolved C and D transitions: {paired_resolved}",
        f"- Silver-assisted paired counts: `{dict(paired_targeted)}`",
        f"- Strong-tier paired claims with resolved C and D transitions: {paired_strong_resolved}",
        f"- Strong-tier paired counts: `{dict(paired_strong)}`",
        "",
        "`GOLD_INHERITED` is the only tier backed directly by an existing human "
        "label, and only for a normalized-exact retained claim. "
        "`CROSS_MODEL_CONFIRMED` remains auxiliary because the Llama protocol did "
        "not pass its frozen development calibration gate. The full branch net-gain "
        "claim therefore remains ineligible for formal human-gold interpretation.",
        "",
    ])
    atomic_write_text(md_path, "\n".join(lines))
    print(json.dumps({
        "stage": "V6_targeted_validation",
        "status": "complete",
        "branch_tier_counts": branch_tiers,
        "targeted_alignment_agreement": dict(agreement),
        "paired_silver_assisted_transition_counts": dict(paired_targeted),
        "paired_strong_tier_transition_counts": dict(paired_strong),
        "report": relative(json_path),
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one isolated post-hoc CoVe validation stage."
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--scope", choices=("full",), default="full")
    parser.add_argument("--split", choices=("heldout",), default="heldout")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-config-frozen", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    args = parser.parse_args(argv)
    if args.resume and args.stage not in MODEL_STAGES:
        parser.error("--resume is valid only for model-backed validation stages")
    if args.stage in MODEL_STAGES and not args.confirm_config_frozen:
        parser.error("Held-out model validation requires --confirm-config-frozen")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "analyze-paired-statistics":
        return analyze_paired_statistics(args.split, args.dry_run)
    if args.stage == "build-exact-retained":
        return build_exact_retained(args.split, args.dry_run)
    if args.stage == "prepare-targeted-audit":
        return prepare_targeted_audit(args.split, args.dry_run)
    if args.stage == "run-targeted-alignment":
        return run_model_stage(
            args.split, "alignment", resume=args.resume, dry_run=args.dry_run,
            host=args.ollama_host,
        )
    if args.stage == "run-targeted-factuality":
        return run_model_stage(
            args.split, "factuality", resume=args.resume, dry_run=args.dry_run,
            host=args.ollama_host,
        )
    if args.stage == "analyze-targeted-validation":
        return analyze_targeted_validation(args.split, args.dry_run)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
