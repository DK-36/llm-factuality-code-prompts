#!/usr/bin/env python3
"""Analyze paired held-out Study I effects across retrieved-evidence depths.

This analysis performs no model inference and no retrieval. It compares stored
predictions for Hybrid RRF top-1, top-3, and top-5 against each other and the
frozen No Evidence and Benchmark-associated Evidence conditions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factcheck_bench_analysis import (  # noqa: E402
    compute_binary_metrics,
    paired_response_cluster_bootstrap,
)


SCHEMA_VERSION = "retrieval_depth_heldout_pairwise_v1"
EXPECTED_CLAIMS = 468
EXPECTED_RESPONSES = 72
EXPECTED_LABEL_COUNTS = {"FACTUAL": 371, "NON_FACTUAL": 97}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_722
CONFIDENCE_LEVEL = 0.95

GOLD_PATH = PROJECT_ROOT / "data/factcheck_bench/processed/fcb_gold_claims_full.jsonl"
SPLIT_PATH = (
    PROJECT_ROOT
    / "data/factcheck_bench/retrieval/manifests/retrieval_split_manifest.jsonl"
)
NO_EVIDENCE_PATH = (
    PROJECT_ROOT
    / "outputs/factcheck_bench_full/jsonl/08b_no_evidence_verifier_results.jsonl"
)
BENCHMARK_PATH = (
    PROJECT_ROOT
    / "outputs/factcheck_bench_full/jsonl/08c_oracle_evidence_verifier_results.jsonl"
)
K1_PATH = (
    PROJECT_ROOT
    / "outputs/factcheck_bench_full/retrieved_evidence_k1_heldout/jsonl/12_retrieved_evidence_verifier_heldout_k01_results.jsonl"
)
K3_PATH = (
    PROJECT_ROOT
    / "outputs/factcheck_bench_full/retrieved_evidence_k3_heldout/jsonl/12_retrieved_evidence_verifier_heldout_k03_results.jsonl"
)
K5_PATH = (
    PROJECT_ROOT
    / "outputs/factcheck_bench_full/jsonl/12_retrieved_evidence_verifier_heldout_results.jsonl"
)
K1_SUMMARY = (
    PROJECT_ROOT
    / "outputs/factcheck_bench_full/retrieved_evidence_k1_heldout/reports/12_retrieved_evidence_verifier_heldout_k01_summary.json"
)
K3_SUMMARY = (
    PROJECT_ROOT
    / "outputs/factcheck_bench_full/retrieved_evidence_k3_heldout/reports/12_retrieved_evidence_verifier_heldout_k03_summary.json"
)
K5_SUMMARY = (
    PROJECT_ROOT
    / "outputs/factcheck_bench_full/reports/12_retrieved_evidence_verifier_heldout_summary.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/factcheck_bench_full/retrieval_depth_heldout_sensitivity/reports"
)
JSON_REPORT = OUTPUT_ROOT / "retrieval_depth_heldout_pairwise_summary.json"
MD_REPORT = OUTPUT_ROOT / "retrieval_depth_heldout_pairwise_report.md"

SETTINGS = (
    ("no_evidence", "No evidence", NO_EVIDENCE_PATH, "no_evidence", None),
    (
        "benchmark",
        "Benchmark-associated evidence",
        BENCHMARK_PATH,
        "oracle_evidence",
        None,
    ),
    ("retrieved_k1", "Retrieved evidence K=1", K1_PATH, "retrieved_evidence", 1),
    ("retrieved_k3", "Retrieved evidence K=3", K3_PATH, "retrieved_evidence", 3),
    ("retrieved_k5", "Retrieved evidence K=5", K5_PATH, "retrieved_evidence", 5),
)

# Every estimand is after minus before. Positive depth contrasts therefore
# favour the larger K, and positive benchmark contrasts favour retrieval.
COMPARISONS = (
    ("benchmark_minus_no", "Benchmark - No evidence", "benchmark", "no_evidence"),
    ("k1_minus_no", "Retrieved K=1 - No evidence", "retrieved_k1", "no_evidence"),
    ("k3_minus_no", "Retrieved K=3 - No evidence", "retrieved_k3", "no_evidence"),
    ("k5_minus_no", "Retrieved K=5 - No evidence", "retrieved_k5", "no_evidence"),
    ("k1_minus_benchmark", "Retrieved K=1 - Benchmark", "retrieved_k1", "benchmark"),
    ("k3_minus_benchmark", "Retrieved K=3 - Benchmark", "retrieved_k3", "benchmark"),
    ("k5_minus_benchmark", "Retrieved K=5 - Benchmark", "retrieved_k5", "benchmark"),
    ("k3_minus_k1", "Retrieved K=3 - K=1", "retrieved_k3", "retrieved_k1"),
    ("k5_minus_k3", "Retrieved K=5 - K=3", "retrieved_k5", "retrieved_k3"),
)

SUMMARY_CHECKS = (
    ("benchmark_minus_no", K5_SUMMARY, "oracle_minus_no_evidence"),
    ("k1_minus_no", K1_SUMMARY, "retrieved_minus_no_evidence"),
    ("k3_minus_no", K3_SUMMARY, "retrieved_minus_no_evidence"),
    ("k5_minus_no", K5_SUMMARY, "retrieved_minus_no_evidence"),
    ("k1_minus_benchmark", K1_SUMMARY, "retrieved_minus_oracle"),
    ("k3_minus_benchmark", K3_SUMMARY, "retrieved_minus_oracle"),
    ("k5_minus_benchmark", K5_SUMMARY, "retrieved_minus_oracle"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def index_unique(
    rows: Sequence[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in indexed:
            raise ValueError(f"{label} contains an invalid or duplicate {key}: {value!r}")
        indexed[value] = row
    return indexed


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())
        temporary = Path(sink.name)
    temporary.replace(path)
    path.chmod(0o644)


def load_inputs() -> tuple[
    list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]], dict[str, Any]
]:
    gold = index_unique(read_jsonl(GOLD_PATH), "claim_id", "gold claims")
    split = index_unique(read_jsonl(SPLIT_PATH), "claim_id", "split manifest")
    claim_ids = sorted(
        claim_id
        for claim_id, row in split.items()
        if row.get("split") == "heldout"
        and row.get("in_primary_matched_cohort") is True
    )
    if len(claim_ids) != EXPECTED_CLAIMS or not set(claim_ids) <= set(gold):
        raise ValueError("Held-out cohort does not match the frozen Study I design")
    records = [gold[claim_id] for claim_id in claim_ids]
    response_ids = {str(row["response_id"]) for row in records}
    label_counts = Counter(str(row["human_label"]) for row in records)
    if (
        len(response_ids) != EXPECTED_RESPONSES
        or dict(label_counts) != EXPECTED_LABEL_COUNTS
    ):
        raise ValueError("Held-out response or label counts are incompatible")

    result_sets: dict[str, dict[str, dict[str, Any]]] = {}
    profiles: dict[str, tuple[Any, ...]] = {}
    fingerprints: dict[str, list[str]] = {}
    profile_fields = (
        "model",
        "model_digest",
        "temperature",
        "seed",
        "num_predict",
        "think",
    )
    for key, _, path, expected_setting, expected_k in SETTINGS:
        source = index_unique(read_jsonl(path), "claim_id", f"{key} predictions")
        selected = {
            claim_id: source[claim_id]
            for claim_id in claim_ids
            if claim_id in source
        }
        if set(selected) != set(claim_ids):
            raise ValueError(f"{key} does not cover the exact held-out claim set")
        for claim_id, row in selected.items():
            reference = gold[claim_id]
            if (
                row.get("setting") != expected_setting
                or row.get("response_id") != reference.get("response_id")
                or row.get("gold_claim") != reference.get("gold_claim")
                or (
                    "human_label" in row
                    and row.get("human_label") != reference.get("human_label")
                )
                or row.get("status") != "ok"
                or row.get("prediction")
                not in {"FACTUAL", "NON_FACTUAL", "UNKNOWN"}
            ):
                raise ValueError(f"{key} contains an incompatible row for {claim_id}")
            if expected_k is not None:
                item_count = len(row.get("retrieved_evidence", {}).get("items", []))
                if item_count != expected_k:
                    raise ValueError(
                        f"{key} has {item_count} passages for {claim_id}; "
                        f"expected {expected_k}"
                    )
        profile_values = {
            tuple(row.get(field) for field in profile_fields)
            for row in selected.values()
        }
        if len(profile_values) != 1:
            raise ValueError(f"{key} contains multiple verifier profiles")
        profiles[key] = next(iter(profile_values))
        fingerprints[key] = sorted(
            {str(row.get("run_fingerprint")) for row in selected.values()}
        )
        result_sets[key] = selected
    if len(set(profiles.values())) != 1:
        raise ValueError(f"Verifier profiles differ across settings: {profiles}")

    return records, result_sets, {
        "claims": len(records),
        "responses": len(response_ids),
        "gold_label_counts": dict(label_counts),
        "verifier_profile": dict(zip(profile_fields, next(iter(profiles.values())))),
        "run_fingerprints": fingerprints,
    }


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def verify_existing_intervals(paired: Mapping[str, Any]) -> None:
    for comparison_key, summary_path, historical_key in SUMMARY_CHECKS:
        historical = read_json(summary_path)
        historical_metrics = historical[
            "paired_response_cluster_bootstrap"
        ]["paired_difference_intervals"][historical_key]["metrics"]
        current_metrics = paired[comparison_key]["metrics"]
        for metric in ("balanced_accuracy", "macro_f1"):
            for field in ("point_estimate", "lower", "upper"):
                if not close(
                    current_metrics[metric][field], historical_metrics[metric][field]
                ):
                    raise ValueError(
                        f"Recomputed {comparison_key} {metric} {field} differs "
                        f"from {summary_path}"
                    )


def build_report() -> dict[str, Any]:
    records, result_sets, audit = load_inputs()
    metrics = {
        key: compute_binary_metrics(records, predictions)
        for key, predictions in result_sets.items()
    }
    bootstrap = paired_response_cluster_bootstrap(
        records,
        result_sets,
        tuple((key, after, before) for key, _, after, before in COMPARISONS),
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
        confidence_level=CONFIDENCE_LEVEL,
    )
    paired = bootstrap["paired_difference_intervals"]
    verify_existing_intervals(paired)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "complete",
        "analysis_role": "secondary_heldout_retrieval_depth_sensitivity",
        "formal_primary_top_k": 5,
        "cohort": audit,
        "setting_labels": {key: label for key, label, _, _, _ in SETTINGS},
        "comparison_labels": {key: label for key, label, _, _ in COMPARISONS},
        "setting_metrics": metrics,
        "paired_response_cluster_bootstrap": bootstrap,
        "interpretation_boundaries": [
            "The formal Retrieved Evidence condition remains Hybrid RRF top-5.",
            "Top-1 and top-3 are secondary held-out sensitivity conditions and are not used for reselection.",
            "An interval including zero is inconclusive and is not evidence of equivalence.",
            "The analysis reuses stored predictions and performs no model inference or retrieval.",
        ],
    }


def pp(value: Any) -> str:
    return f"{100.0 * float(value):+.2f}"


def markdown(report: Mapping[str, Any]) -> str:
    paired = report["paired_response_cluster_bootstrap"][
        "paired_difference_intervals"
    ]
    lines = [
        "# Held-out retrieval-depth paired sensitivity",
        "",
        "Status: **complete**. This secondary analysis reuses the same 468 held-out "
        "claims in 72 response clusters. The formal Hybrid RRF top-5 condition "
        "remains unchanged.",
        "",
        "| Comparison (first minus second) | Balanced Accuracy delta (95% CI), pp | Macro-F1 delta (95% CI), pp |",
        "|---|---:|---:|",
    ]
    for key, label, _, _ in COMPARISONS:
        metrics = paired[key]["metrics"]
        balanced = metrics["balanced_accuracy"]
        macro = metrics["macro_f1"]
        lines.append(
            f"| {label} | {pp(balanced['point_estimate'])} "
            f"[{pp(balanced['lower'])}, {pp(balanced['upper'])}] | "
            f"{pp(macro['point_estimate'])} "
            f"[{pp(macro['lower'])}, {pp(macro['upper'])}] |"
        )
    lines.extend(
        [
            "",
            f"Intervals use {BOOTSTRAP_SAMPLES:,} paired response-cluster percentile "
            f"resamples with seed {BOOTSTRAP_SEED} and a {CONFIDENCE_LEVEL:.0%} "
            "confidence level. Positive values favour the first-named condition.",
            "",
            "An interval including zero is interpreted as inconclusive, not as evidence "
            "that the two conditions are equivalent.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze held-out paired differences across retrieved-evidence depths "
            "without model calls."
        )
    )
    parser.add_argument("--scope", choices=("full",), default="full")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and compute results without writing reports.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report()
    if not args.dry_run:
        atomic_write(
            JSON_REPORT,
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        atomic_write(MD_REPORT, markdown(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "claims": report["cohort"]["claims"],
                "responses": report["cohort"]["responses"],
                "comparisons": len(COMPARISONS),
                "bootstrap_resamples": BOOTSTRAP_SAMPLES,
                "json_report": str(JSON_REPORT.relative_to(PROJECT_ROOT)),
                "markdown_report": str(MD_REPORT.relative_to(PROJECT_ROOT)),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
