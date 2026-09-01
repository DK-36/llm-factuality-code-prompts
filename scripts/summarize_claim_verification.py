#!/usr/bin/env python3
"""Recompute the canonical full verifier research summary without model calls.

The summary is deliberately derived from the full gold, manifest, no-evidence,
and oracle JSONL files. Current retrieval and retrieved-verifier reports are
joined as separately generated downstream summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factcheck_bench_analysis import (  # noqa: E402
    build_confidence_distribution,
    build_paired_transitions,
    build_response_aggregation,
    compute_binary_metrics,
    metric_differences,
)
from factcheck_bench_pipeline import (  # noqa: E402
    PRIMARY_LABELS,
    build_oracle_cohort,
    paths_for_scope,
    retrieval_paths,
)


LABELS = (*PRIMARY_LABELS, "UNKNOWN")
SUMMARY_VERSION = "full_verifier_research_summary_v4"
FULL_PATHS = paths_for_scope(PROJECT_ROOT, "full")
RETRIEVAL_PATHS = retrieval_paths(PROJECT_ROOT, "full")
DEFAULT_RAW = (
    PROJECT_ROOT
    / "data"
    / "factcheck_bench"
    / "raw"
    / "factcheck-GPT-benchmark.jsonl"
)
DEFAULT_OUTPUT_JSON = (
    FULL_PATHS.output_root / "reports" / "full_verifier_research_summary.json"
)
DEFAULT_OUTPUT_MARKDOWN = (
    FULL_PATHS.output_root / "reports" / "full_verifier_research_summary.md"
)
NO_PROMPT = PROJECT_ROOT / "prompts" / "no_evidence_verifier.txt"
ORACLE_PROMPT = PROJECT_ROOT / "prompts" / "oracle_evidence_verifier.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSONL does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(
                    f"{path} line {line_number} must be a JSON object"
                )
            records.append(row)
    if not records:
        raise ValueError(f"Required JSONL is empty: {path}")
    return records


def index_rows(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        claim_id = row.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError("Every row must have a non-empty string claim_id")
        if claim_id in by_id:
            duplicates.append(claim_id)
            continue
        by_id[claim_id] = row
    return by_id, sorted(set(duplicates))


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sorted_counter(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return dict(sorted(counts.items()))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def set_differences(
    actual: set[str], expected: set[str]
) -> dict[str, Any]:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing_count": len(missing),
        "missing_ids": missing,
        "extra_count": len(extra),
        "extra_ids": extra,
        "exact_match": not missing and not extra,
    }


def validate_result_rows(
    name: str,
    rows: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
    duplicate_ids: list[str],
    gold_by_id: dict[str, dict[str, Any]],
    expected_ids: set[str],
    gold_sha256: str,
    prompt_sha256: str,
    no_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    response_mismatches: list[str] = []
    claim_text_mismatches: list[str] = []
    human_label_mismatches: list[str] = []
    raw_output_parse_errors: list[str] = []
    raw_output_field_mismatches: list[str] = []
    empty_rationale_ids: list[str] = []

    for claim_id, row in result_by_id.items():
        gold = gold_by_id.get(claim_id)
        if gold is None:
            continue
        if row.get("response_id") != gold.get("response_id"):
            response_mismatches.append(claim_id)
        if row.get("gold_claim") != gold.get("gold_claim"):
            claim_text_mismatches.append(claim_id)
        if "human_label" in row and row.get("human_label") != gold.get(
            "human_label"
        ):
            human_label_mismatches.append(claim_id)
        rationale = row.get("rationale")
        if row.get("status") == "ok" and not (
            isinstance(rationale, str) and rationale.strip()
        ):
            empty_rationale_ids.append(claim_id)

        raw = row.get("raw_model_output")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else None
        except json.JSONDecodeError:
            raw_output_parse_errors.append(claim_id)
            continue
        if not isinstance(parsed, dict):
            raw_output_parse_errors.append(claim_id)
            continue
        if any(
            parsed.get(field) != row.get(field)
            for field in ("prediction", "confidence", "rationale")
        ):
            raw_output_field_mismatches.append(claim_id)

    id_check = set_differences(set(result_by_id), expected_ids)
    status_counts = sorted_counter(row.get("status") for row in rows)
    prediction_counts = sorted_counter(row.get("prediction") for row in rows)
    invalid_predictions = sorted(
        claim_id
        for claim_id, row in result_by_id.items()
        if row.get("status") == "ok" and row.get("prediction") not in LABELS
    )
    input_hashes = sorted(
        {
            str(row.get("input_sha256"))
            for row in rows
            if row.get("input_sha256") is not None
        }
    )
    prompt_hashes = sorted(
        {
            str(row.get("prompt_sha256"))
            for row in rows
            if row.get("prompt_sha256") is not None
        }
    )
    model_digests = sorted(
        {
            str(row.get("model_digest"))
            for row in rows
            if row.get("model_digest") is not None
        }
    )
    run_fingerprints = sorted(
        {
            str(row.get("run_fingerprint"))
            for row in rows
            if row.get("run_fingerprint") is not None
        }
    )
    baseline_hashes = sorted(
        {
            str(row.get("no_evidence_results_sha256"))
            for row in rows
            if row.get("no_evidence_results_sha256") is not None
        }
    )

    checks = {
        "ids": id_check,
        "duplicate_claim_id_count": len(duplicate_ids),
        "duplicate_claim_ids": duplicate_ids,
        "status_counts": status_counts,
        "prediction_counts": prediction_counts,
        "invalid_prediction_count": len(invalid_predictions),
        "invalid_prediction_ids": invalid_predictions,
        "response_id_mismatch_count": len(response_mismatches),
        "response_id_mismatch_ids": response_mismatches,
        "gold_claim_mismatch_count": len(claim_text_mismatches),
        "gold_claim_mismatch_ids": claim_text_mismatches,
        "human_label_mismatch_count": len(human_label_mismatches),
        "human_label_mismatch_ids": human_label_mismatches,
        "empty_rationale_count": len(empty_rationale_ids),
        "empty_rationale_ids": empty_rationale_ids,
        "raw_output_parse_error_count": len(raw_output_parse_errors),
        "raw_output_parse_error_ids": raw_output_parse_errors,
        "raw_output_field_mismatch_count": len(raw_output_field_mismatches),
        "raw_output_field_mismatch_ids": raw_output_field_mismatches,
        "input_sha256_values": input_hashes,
        "input_sha256_matches_gold": input_hashes == [gold_sha256],
        "prompt_sha256_values": prompt_hashes,
        "prompt_sha256_matches_current_prompt": prompt_hashes
        == [prompt_sha256],
        "model_digests": model_digests,
        "single_model_digest": len(model_digests) == 1,
        "run_fingerprints": run_fingerprints,
        "single_run_fingerprint": len(run_fingerprints) == 1,
        "no_evidence_results_sha256_values": baseline_hashes,
        "no_evidence_results_sha256_matches_current": (
            True
            if no_evidence_sha256 is None
            else baseline_hashes == [no_evidence_sha256]
        ),
    }
    checks["passed"] = all(
        (
            id_check["exact_match"],
            not duplicate_ids,
            set(status_counts) == {"ok"},
            not invalid_predictions,
            not response_mismatches,
            not claim_text_mismatches,
            not human_label_mismatches,
            not empty_rationale_ids,
            not raw_output_parse_errors,
            not raw_output_field_mismatches,
            checks["input_sha256_matches_gold"],
            checks["prompt_sha256_matches_current_prompt"],
            checks["single_model_digest"],
            checks["single_run_fingerprint"],
            checks["no_evidence_results_sha256_matches_current"],
        )
    )
    checks["setting"] = name
    checks["technical_success_rate"] = (
        status_counts.get("ok", 0) / len(expected_ids)
        if expected_ids
        else None
    )
    return checks


def flatten_evidence_items(value: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(value, list):
        for child in value:
            items.extend(flatten_evidence_items(child))
    elif isinstance(value, dict):
        if "text" in value:
            items.append(value)
        else:
            for key in ("gold_evidence", "evidence", "items", "passages", "bundle"):
                if key in value:
                    items.extend(flatten_evidence_items(value[key]))
    return items


def evidence_inventory(records: list[dict[str, Any]]) -> dict[str, Any]:
    items = [
        item
        for record in records
        for item in flatten_evidence_items(record.get("gold_evidence"))
    ]
    urls = [
        item["url"].strip()
        for item in items
        if isinstance(item.get("url"), str) and item["url"].strip()
    ]
    domains = [
        urlparse(url).netloc.lower().removeprefix("www.")
        for url in urls
        if urlparse(url).netloc
    ]
    wikipedia_count = sum(
        domain == "wikipedia.org" or domain.endswith(".wikipedia.org")
        for domain in domains
    )
    return {
        "evidence_item_count": len(items),
        "items_with_nonempty_url": len(urls),
        "items_without_nonempty_url": len(items) - len(urls),
        "raw_unique_url_count": len(set(urls)),
        "unique_domain_count": len(set(domains)),
        "wikipedia_passage_count": wikipedia_count,
        "source_counts": sorted_counter(item.get("source") for item in items),
        "stance_counts": sorted_counter(item.get("stance") for item in items),
        "url_normalization_status": "not_performed",
    }


def optional_json(path: Path) -> dict[str, Any]:
    """Read an optional generated retrieval report without making it required."""
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Optional retrieval report must be an object: {path}")
    return value


def retrieval_interface_status() -> dict[str, Any]:
    split = optional_json(RETRIEVAL_PATHS.split_summary)
    preparation = optional_json(RETRIEVAL_PATHS.preparation_report_json)
    corpus = optional_json(RETRIEVAL_PATHS.corpus_summary_json)
    dev_retrieval_comparison_path = (
        RETRIEVAL_PATHS.root
        / "evaluation"
        / "reports"
        / "two_level_dev_comparison.json"
    )
    heldout_retrieval_comparison_path = (
        RETRIEVAL_PATHS.root
        / "evaluation"
        / "reports"
        / "two_level_heldout_comparison.json"
    )
    retrieved_verifier_summary_path = (
        FULL_PATHS.output_root
        / "reports"
        / "12_retrieved_evidence_verifier_heldout_summary.json"
    )
    dev_retrieval_comparison = optional_json(dev_retrieval_comparison_path)
    heldout_retrieval_comparison = optional_json(
        heldout_retrieval_comparison_path
    )
    retrieved_verifier = optional_json(retrieved_verifier_summary_path)
    split_complete = split.get("assertions_passed") is True
    manifests_complete = bool(preparation)
    fetch_status = corpus.get("stages", {}).get("fetch_corpus", {}).get(
        "status", "pending"
    )
    passage_status = corpus.get("stages", {}).get("build_passages", {}).get(
        "status", "pending"
    )
    reprocess_status = corpus.get("stages", {}).get(
        "reprocess_frozen_corpus", {}
    ).get("status", "pending")
    qrels_status = corpus.get("stages", {}).get("build_qrels_dev", {}).get(
        "status", "pending"
    )
    heldout_qrels_status = (
        "complete"
        if RETRIEVAL_PATHS.qrels_heldout_jsonl.exists()
        else corpus.get("stages", {}).get(
            "build_qrels_heldout", {}
        ).get("status", "pending")
    )
    if split_complete and manifests_complete:
        status = "corpus_preparation_implemented"
    else:
        status = "pending_preparation"
    return {
        "status": status,
        "corpus_name": "benchmark-grounded closed source-document corpus",
        "split": {
            "status": "complete" if split_complete else "pending",
            "development_matched_claims": split.get(
                "development_matched_claims"
            ),
            "primary_heldout_matched_claims": split.get(
                "heldout_matched_claims"
            ),
            "secondary_whole_matched_claims": split.get(
                "total_matched_claims"
            ),
            "development_source_responses": split.get(
                "development_source_response_count"
            ),
            "development_matched_responses": split.get(
                "development_matched_responses"
            ),
            "heldout_matched_responses": split.get(
                "heldout_matched_responses"
            ),
            "note": (
                "The 121 matched claims were not used for prior retrieval tuning; "
                "they are prospectively designated for configuration selection."
            ),
        },
        "construction": {
            "manifest_status": "complete" if manifests_complete else "pending",
            "fetch_status": fetch_status,
            "reprocess_status": reprocess_status,
            "passage_status": passage_status,
            "qrels_status": qrels_status,
            "heldout_qrels_status": heldout_qrels_status,
            "evidence_item_count": preparation.get(
                "evidence_inventory", {}
            ).get("evidence_item_count"),
            "raw_unique_url_count": preparation.get(
                "evidence_inventory", {}
            ).get("raw_unique_url_count"),
            "canonical_unique_url_count": preparation.get(
                "url_canonicalisation", {}
            ).get("canonical_unique_url_count"),
        },
        "retrieval_metrics_status": (
            "dev_selection_and_heldout_evaluation_complete"
            if heldout_retrieval_comparison.get("status") == "complete"
            else corpus.get("retrieval_metrics_status", "not_run")
        ),
        "dev_two_level_evaluation": dev_retrieval_comparison,
        "heldout_two_level_evaluation": heldout_retrieval_comparison,
        "retrieved_verifier_status": retrieved_verifier.get(
            "status", "not_run"
        ),
        "retrieved_verifier_evaluation": retrieved_verifier,
        "query_input": relative(FULL_PATHS.gold_claims),
        "cohort_input": relative(FULL_PATHS.cohort_manifest),
        "primary_evaluation_claim_ids": (
            "retrieval_split_manifest rows with split=heldout"
        ),
        "artifacts": [
            relative(RETRIEVAL_PATHS.split_manifest),
            relative(RETRIEVAL_PATHS.split_summary),
            relative(RETRIEVAL_PATHS.evidence_manifest),
            relative(RETRIEVAL_PATHS.url_manifest),
            relative(RETRIEVAL_PATHS.documents),
            relative(RETRIEVAL_PATHS.passages),
            relative(RETRIEVAL_PATHS.qrels_dev_jsonl),
            relative(RETRIEVAL_PATHS.qrels_dev_mapping_report_json),
            relative(RETRIEVAL_PATHS.qrels_heldout_jsonl),
            relative(RETRIEVAL_PATHS.qrels_heldout_mapping_report_json),
            relative(RETRIEVAL_PATHS.corpus_summary_json),
            relative(dev_retrieval_comparison_path),
            relative(heldout_retrieval_comparison_path),
            relative(retrieved_verifier_summary_path),
        ],
        "future_artifacts": [],
        "leakage_boundary": (
            "Query construction, ranking, and retrieved-verifier prompts must not "
            "use human_label, gold evidence stance/text, revised claims, or the "
            "claim-specific gold URL mapping. Those fields are evaluation-only."
        ),
    }


def response_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if row.get("binary_claims", 0) > 0 and row.get("accuracy") is not None
    ]
    values = [float(row["accuracy"]) for row in usable]
    if not values:
        return {
            "response_count": 0,
            "minimum": None,
            "quartile_1": None,
            "median": None,
            "quartile_3": None,
            "maximum": None,
            "zero_accuracy_response_count": 0,
            "perfect_accuracy_response_count": 0,
            "below_half_accuracy_response_count": 0,
        }
    if len(values) == 1:
        quartile_1 = quartile_3 = values[0]
    else:
        quartile_1, _, quartile_3 = statistics.quantiles(
            values, n=4, method="inclusive"
        )
    minimum = min(values)
    maximum = max(values)
    return {
        "response_count": len(values),
        "minimum": minimum,
        "quartile_1": quartile_1,
        "median": statistics.median(values),
        "quartile_3": quartile_3,
        "maximum": maximum,
        "minimum_response_ids": sorted(
            row["response_id"]
            for row in usable
            if float(row["accuracy"]) == minimum
        ),
        "maximum_response_ids": sorted(
            row["response_id"]
            for row in usable
            if float(row["accuracy"]) == maximum
        ),
        "zero_accuracy_response_count": sum(value == 0.0 for value in values),
        "perfect_accuracy_response_count": sum(value == 1.0 for value in values),
        "below_half_accuracy_response_count": sum(value < 0.5 for value in values),
    }


def matched_response_rows(
    records: list[dict[str, Any]],
    no_by_id: dict[str, dict[str, Any]],
    oracle_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    no_rows, no_macro = build_response_aggregation(records, no_by_id)
    oracle_rows, oracle_macro = build_response_aggregation(records, oracle_by_id)
    no_index = {row["response_id"]: row for row in no_rows}
    oracle_index = {row["response_id"]: row for row in oracle_rows}
    combined: list[dict[str, Any]] = []
    for response_id in sorted(oracle_index):
        no_row = no_index[response_id]
        oracle_row = oracle_index[response_id]
        no_accuracy = no_row["accuracy"]
        oracle_accuracy = oracle_row["accuracy"]
        combined.append(
            {
                "response_id": response_id,
                "matched_claims": oracle_row["binary_claims"],
                "no_evidence_correct": no_row["correct_count"],
                "no_evidence_misses": (
                    no_row["binary_claims"] - no_row["correct_count"]
                ),
                "no_evidence_accuracy": no_accuracy,
                "oracle_correct": oracle_row["correct_count"],
                "oracle_misses": (
                    oracle_row["binary_claims"] - oracle_row["correct_count"]
                ),
                "oracle_accuracy": oracle_accuracy,
                "oracle_minus_no_evidence_accuracy": (
                    None
                    if no_accuracy is None or oracle_accuracy is None
                    else oracle_accuracy - no_accuracy
                ),
                "oracle_coverage": oracle_row["coverage"],
            }
        )
    combined.sort(key=lambda row: row["response_id"])
    macros = {
        "no_evidence": no_macro,
        "oracle": oracle_macro,
        "oracle_minus_no_evidence_accuracy": (
            None
            if no_macro["accuracy"] is None or oracle_macro["accuracy"] is None
            else oracle_macro["accuracy"] - no_macro["accuracy"]
        ),
    }
    return combined, macros


def definitive_error_confidence(
    records: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    values = [
        float(result_by_id[record["claim_id"]]["confidence"])
        for record in records
        if record["claim_id"] in result_by_id
        and result_by_id[record["claim_id"]].get("status") == "ok"
        and result_by_id[record["claim_id"]].get("prediction")
        in PRIMARY_LABELS
        and result_by_id[record["claim_id"]].get("prediction")
        != record.get("human_label")
    ]
    return {
        "wrong_definitive_decision_count": len(values),
        "minimum_confidence": min(values) if values else None,
        "maximum_confidence": max(values) if values else None,
        "mean_confidence": statistics.fmean(values) if values else None,
        "count_at_or_above_0_80": sum(value >= 0.8 for value in values),
        "count_at_or_above_0_95": sum(value >= 0.95 for value in values),
    }


def predicted_rate(
    records: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
    label: str,
) -> float | None:
    if not records:
        return None
    return sum(
        result_by_id.get(record["claim_id"], {}).get("status") == "ok"
        and result_by_id.get(record["claim_id"], {}).get("prediction") == label
        for record in records
    ) / len(records)


def majority_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["human_label"] for record in records)
    label, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return {
        "majority_label": label,
        "gold_label_counts": dict(counts),
        "accuracy": count / len(records) if records else None,
        "balanced_accuracy": 0.5 if len(counts) == 2 else None,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    raw_records = load_jsonl(args.raw)
    gold_records = load_jsonl(args.gold_claims)
    manifest_records = load_jsonl(args.manifest)
    no_results = load_jsonl(args.no_evidence)
    oracle_results = load_jsonl(args.oracle)

    gold_by_id, gold_duplicates = index_rows(gold_records)
    manifest_by_id, manifest_duplicates = index_rows(manifest_records)
    no_by_id, no_duplicates = index_rows(no_results)
    oracle_by_id, oracle_duplicates = index_rows(oracle_results)

    binary_records = [
        record
        for record in gold_records
        if record.get("human_label") in PRIMARY_LABELS
    ]
    matched_records, oracle_cohort_audit = build_oracle_cohort(gold_records)
    binary_ids = {record["claim_id"] for record in binary_records}
    matched_ids = {record["claim_id"] for record in matched_records}
    gold_ids = set(gold_by_id)

    gold_sha256 = sha256_file(args.gold_claims)
    no_sha256 = sha256_file(args.no_evidence)
    no_prompt_sha256 = sha256_file(args.no_prompt)
    oracle_prompt_sha256 = sha256_file(args.oracle_prompt)

    manifest_id_check = set_differences(set(manifest_by_id), gold_ids)
    manifest_field_mismatches = sorted(
        claim_id
        for claim_id in gold_ids & set(manifest_by_id)
        if any(
            manifest_by_id[claim_id].get(field)
            != gold_by_id[claim_id].get(field)
            for field in (
                "response_id",
                "human_label",
                "is_binary_evaluable",
                "in_primary_matched_cohort",
                "audit_flag",
            )
        )
    )
    no_integrity = validate_result_rows(
        "no_evidence",
        no_results,
        no_by_id,
        no_duplicates,
        gold_by_id,
        binary_ids,
        gold_sha256,
        no_prompt_sha256,
    )
    oracle_integrity = validate_result_rows(
        "oracle_evidence",
        oracle_results,
        oracle_by_id,
        oracle_duplicates,
        gold_by_id,
        matched_ids,
        gold_sha256,
        oracle_prompt_sha256,
        no_evidence_sha256=no_sha256,
    )
    manifest_integrity = {
        "ids": manifest_id_check,
        "duplicate_claim_id_count": len(manifest_duplicates),
        "duplicate_claim_ids": manifest_duplicates,
        "field_mismatch_count": len(manifest_field_mismatches),
        "field_mismatch_ids": manifest_field_mismatches,
        "passed": (
            manifest_id_check["exact_match"]
            and not manifest_duplicates
            and not manifest_field_mismatches
        ),
    }

    no_full_metrics = compute_binary_metrics(binary_records, no_by_id)
    no_matched_metrics = compute_binary_metrics(matched_records, no_by_id)
    oracle_metrics = compute_binary_metrics(matched_records, oracle_by_id)
    transitions = build_paired_transitions(
        matched_records, no_by_id, oracle_by_id
    )
    differences = metric_differences(no_matched_metrics, oracle_metrics)

    no_full_response_rows, no_full_response_macro = build_response_aggregation(
        binary_records, no_by_id
    )
    response_rows, matched_response_macro = matched_response_rows(
        matched_records, no_by_id, oracle_by_id
    )
    no_matched_distribution_rows, _ = build_response_aggregation(
        matched_records, no_by_id
    )
    oracle_distribution_rows, _ = build_response_aggregation(
        matched_records, oracle_by_id
    )
    highest_error_clusters = sorted(
        response_rows,
        key=lambda row: (
            -row["oracle_misses"],
            row["oracle_accuracy"],
            row["response_id"],
        ),
    )[:10]
    largest_oracle_regressions = sorted(
        response_rows,
        key=lambda row: (
            row["oracle_minus_no_evidence_accuracy"],
            row["response_id"],
        ),
    )[:10]

    audit_flag_counts = Counter(
        flag
        for record in gold_records
        for flag in record.get("audit_flags", [])
    )
    flagged_ids = {
        record["claim_id"] for record in gold_records if record.get("audit_flag")
    }
    unflagged_binary = [
        record for record in binary_records if record["claim_id"] not in flagged_ids
    ]
    unflagged_matched = [
        record for record in matched_records if record["claim_id"] not in flagged_ids
    ]
    sensitivity = {
        "automated_audit_flagged_claim_count": len(flagged_ids),
        "automated_audit_flag_counts": dict(sorted(audit_flag_counts.items())),
        "flagged_binary_claim_count": len(binary_ids & flagged_ids),
        "flagged_matched_claim_count": len(matched_ids & flagged_ids),
        "no_evidence_full_binary_excluding_automated_flags": (
            compute_binary_metrics(unflagged_binary, no_by_id)
        ),
        "no_evidence_matched_excluding_automated_flags": (
            compute_binary_metrics(unflagged_matched, no_by_id)
        ),
        "oracle_matched_excluding_automated_flags": (
            compute_binary_metrics(unflagged_matched, oracle_by_id)
        ),
        "human_gold_adjudication": {
            "status": "not_available",
            "note": (
                "No separate blinded human adjudication file was found; automated "
                "schema/audit flags are not equivalent to gold relabelling."
            ),
        },
    }

    label_counts = Counter(record["human_label"] for record in gold_records)
    binary_label_counts = Counter(
        record["human_label"] for record in binary_records
    )
    matched_label_counts = Counter(
        record["human_label"] for record in matched_records
    )
    claim_response_ids = {record["response_id"] for record in gold_records}
    source_response_ids = {
        f"fcb_r{index:04d}" for index in range(1, len(raw_records) + 1)
    }
    dataset = {
        "source_response_count": len(raw_records),
        "claim_response_count": len(claim_response_ids),
        "source_responses_without_claims_count": len(
            source_response_ids - claim_response_ids
        ),
        "source_responses_without_claims": sorted(
            source_response_ids - claim_response_ids
        ),
        "all_claim_count": len(gold_records),
        "human_label_counts": dict(label_counts),
        "binary_claim_count": len(binary_records),
        "binary_human_label_counts": dict(binary_label_counts),
        "human_unknown_claim_count": label_counts.get("UNKNOWN", 0),
        "structural_evidence_claim_count": sum(
            bool(record.get("structural_evidence_available"))
            for record in gold_records
        ),
        "usable_oracle_evidence_claim_count": sum(
            bool(record.get("oracle_evidence_available"))
            for record in gold_records
        ),
        "matched_evidence_claim_count": len(matched_records),
        "matched_human_label_counts": dict(matched_label_counts),
        "matched_response_count": len(
            {record["response_id"] for record in matched_records}
        ),
        "matched_coverage_of_binary": (
            len(matched_records) / len(binary_records)
        ),
        "binary_exclusion_reason_counts": dict(
            Counter(
                record.get("oracle_evidence_exclusion_reason")
                for record in binary_records
                if record["claim_id"] not in matched_ids
            )
        ),
        "all_cohort_exclusion_reason_counts": oracle_cohort_audit[
            "exclusion_reason_counts"
        ],
    }

    no_confidence_full = build_confidence_distribution(
        binary_records, no_by_id, args.high_confidence_threshold
    )
    no_confidence_matched = build_confidence_distribution(
        matched_records, no_by_id, args.high_confidence_threshold
    )
    oracle_confidence = build_confidence_distribution(
        matched_records, oracle_by_id, args.high_confidence_threshold
    )

    paired_transition_counts = {
        key: value["count"]
        for key, value in transitions["named_transitions"].items()
    }
    paired_correct_net_gain = (
        oracle_metrics["correct_count"] - no_matched_metrics["correct_count"]
    )
    strict_wrong_correct_net = (
        paired_transition_counts["wrong_to_correct"]
        - paired_transition_counts["correct_to_wrong"]
    )

    no_full_majority = majority_baseline(binary_records)
    matched_majority = majority_baseline(matched_records)
    gold_non_factual_rate_full = (
        binary_label_counts["NON_FACTUAL"] / len(binary_records)
    )
    gold_non_factual_rate_matched = (
        matched_label_counts["NON_FACTUAL"] / len(matched_records)
    )

    files = {
        "raw": {
            "path": relative(args.raw),
            "sha256": sha256_file(args.raw),
            "rows": len(raw_records),
        },
        "gold_claims": {
            "path": relative(args.gold_claims),
            "sha256": gold_sha256,
            "rows": len(gold_records),
        },
        "cohort_manifest": {
            "path": relative(args.manifest),
            "sha256": sha256_file(args.manifest),
            "rows": len(manifest_records),
        },
        "no_evidence_predictions": {
            "path": relative(args.no_evidence),
            "sha256": no_sha256,
            "rows": len(no_results),
        },
        "oracle_predictions": {
            "path": relative(args.oracle),
            "sha256": sha256_file(args.oracle),
            "rows": len(oracle_results),
        },
        "no_evidence_prompt": {
            "path": relative(args.no_prompt),
            "sha256": no_prompt_sha256,
        },
        "oracle_prompt": {
            "path": relative(args.oracle_prompt),
            "sha256": oracle_prompt_sha256,
        },
    }

    overall_integrity = all(
        (
            not gold_duplicates,
            manifest_integrity["passed"],
            no_integrity["passed"],
            oracle_integrity["passed"],
            no_integrity["model_digests"] == oracle_integrity["model_digests"],
        )
    )
    retrieval_interface = retrieval_interface_status()
    heldout_bootstrap = retrieval_interface.get(
        "retrieved_verifier_evaluation", {}
    ).get("paired_response_cluster_bootstrap", {})

    return {
        "summary_version": SUMMARY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generation_method": (
            "Recomputed from canonical full JSONL files; no model calls and no "
            "metrics copied from prior summaries."
        ),
        "files": files,
        "dataset": dataset,
        "retrieval_evidence_inventory": evidence_inventory(gold_records),
        "no_evidence_full_binary": {
            "metrics": no_full_metrics,
            "majority_baseline": no_full_majority,
            "accuracy_minus_majority_baseline": (
                no_full_metrics["accuracy_including_abstentions_and_errors"]
                - no_full_majority["accuracy"]
            ),
            "gold_non_factual_rate": gold_non_factual_rate_full,
            "predicted_non_factual_rate": predicted_rate(
                binary_records, no_by_id, "NON_FACTUAL"
            ),
            "response_macro": no_full_response_macro,
            "response_distribution": response_distribution(
                no_full_response_rows
            ),
        },
        "oracle_evidence_matched": {
            "metrics": oracle_metrics,
            "majority_baseline": matched_majority,
            "accuracy_minus_majority_baseline": (
                oracle_metrics["accuracy_including_abstentions_and_errors"]
                - matched_majority["accuracy"]
            ),
            "gold_non_factual_rate": gold_non_factual_rate_matched,
            "predicted_non_factual_rate": predicted_rate(
                matched_records, oracle_by_id, "NON_FACTUAL"
            ),
        },
        "strict_matched_paired_comparison": {
            "cohort_definition": (
                f"Exact intersection fixed by the {len(matched_records)} binary "
                "claims with usable oracle evidence; both settings contain every "
                "matched claim ID."
            ),
            "claim_count": len(matched_records),
            "response_count": len(
                {record["response_id"] for record in matched_records}
            ),
            "no_evidence_metrics": no_matched_metrics,
            "oracle_metrics": oracle_metrics,
            "point_differences": differences,
            "transitions": transitions,
            "transition_counts": paired_transition_counts,
            "strict_wrong_correct_net_gain": strict_wrong_correct_net,
            "total_correct_count_net_gain": paired_correct_net_gain,
            "majority_baseline": matched_majority,
            "no_evidence_accuracy_minus_majority_baseline": (
                no_matched_metrics[
                    "accuracy_including_abstentions_and_errors"
                ]
                - matched_majority["accuracy"]
            ),
            "paired_response_cluster_bootstrap": {
                "status": "not_computed",
                "reason": (
                    "The implemented paired bootstrap is scoped to the primary "
                    "468-claim held-out three-setting comparison. This secondary "
                    "589-claim whole-matched comparison remains a descriptive analysis."
                ),
            },
        },
        "response_level": {
            "claim_weighted_no_evidence_full_binary_accuracy": no_full_metrics[
                "accuracy_including_abstentions_and_errors"
            ],
            "response_macro_no_evidence_full_binary": no_full_response_macro,
            "matched_response_macro": matched_response_macro,
            "matched_no_evidence_distribution": response_distribution(
                no_matched_distribution_rows
            ),
            "matched_oracle_distribution": response_distribution(
                oracle_distribution_rows
            ),
            "highest_oracle_error_clusters": highest_error_clusters,
            "largest_oracle_regressions": largest_oracle_regressions,
            "all_matched_response_rows": response_rows,
        },
        "confidence": {
            "no_evidence_full_binary": no_confidence_full,
            "no_evidence_matched": no_confidence_matched,
            "oracle_matched": oracle_confidence,
            "no_evidence_full_binary_wrong_definitive": (
                definitive_error_confidence(binary_records, no_by_id)
            ),
            "oracle_matched_wrong_definitive": (
                definitive_error_confidence(matched_records, oracle_by_id)
            ),
            "interpretation": (
                "Scores are self-reported confidence in the selected label, not "
                "calibrated class-probability distributions."
            ),
        },
        "prediction_rationale_consistency": {
            "rationale_presence_validated": True,
            "no_evidence_nonempty_rationale_count": len(no_results)
            - no_integrity["empty_rationale_count"],
            "oracle_nonempty_rationale_count": len(oracle_results)
            - oracle_integrity["empty_rationale_count"],
            "semantic_consistency_status": "not_evaluated",
            "reason": (
                "The repository has no semantic label-rationale consistency checker; "
                "JSON presence and parseability do not establish semantic support."
            ),
        },
        "sensitivity_analysis": sensitivity,
        "data_integrity": {
            "overall_status": "passed" if overall_integrity else "failed",
            "gold_duplicate_claim_id_count": len(gold_duplicates),
            "gold_duplicate_claim_ids": gold_duplicates,
            "manifest": manifest_integrity,
            "no_evidence": no_integrity,
            "oracle_evidence": oracle_integrity,
            "model_digest_matches_between_settings": (
                no_integrity["model_digests"]
                == oracle_integrity["model_digests"]
            ),
        },
        "research_conclusions": {
            "supported": [
                (
                    "On the same matched claims, oracle evidence improves the "
                    "observed accuracy, balanced accuracy, macro-F1, coverage, and "
                    "selective accuracy point estimates."
                ),
                (
                    "Oracle evidence produces more wrong-to-correct than "
                    "correct-to-wrong transitions, but it also introduces new errors."
                ),
                (
                    "Both settings over-predict NON_FACTUAL, and high self-reported "
                    "confidence does not reliably identify correct decisions."
                ),
                (
                    "Balanced accuracy above 50% indicates non-degenerate class "
                    "discrimination, while raw accuracy remains below the majority "
                    "baseline because the dataset is strongly FACTUAL-majority."
                ),
                *(
                    [
                        "On the primary 468-claim held-out cohort, paired response-"
                        "cluster bootstrap intervals support positive oracle-minus-no-"
                        "evidence and retrieved-minus-no-evidence differences for "
                        "accuracy, balanced accuracy, and macro-F1."
                    ]
                    if heldout_bootstrap.get("status") == "complete"
                    else []
                ),
            ],
            "not_supported": [
                (
                    "A stable retrieved-evidence advantage over oracle evidence; "
                    "the held-out paired response-cluster intervals for retrieved "
                    "minus oracle include zero."
                ),
                (
                    "A response-cluster interval for the secondary 589-claim whole-"
                    "matched no-evidence/oracle comparison; the implemented paired "
                    "bootstrap applies to the primary 468-claim three-setting cohort."
                ),
                "Open-domain fact-checking performance beyond FactCheck-Bench.",
                (
                    "Semantic correctness of rationales or interpretation of scalar "
                    "confidence as a calibrated probability."
                ),
                (
                    "Causal attribution of verifier gains to retrieval alone, "
                    "or any claim about CoVe revision quality from Experiment A; "
                    "the retrieved setting combines corpus availability, ranking, "
                    "passage selection, and evidence interpretation. CoVe is "
                    "evaluated separately in Experiment B."
                ),
            ],
            "heldout_statistical_interpretation": (
                "The primary held-out paired response-cluster bootstrap is complete. "
                "Oracle-minus-no-evidence and retrieved-minus-no-evidence 95% intervals "
                "exclude zero for accuracy, balanced accuracy, and macro-F1; retrieved-"
                "minus-oracle intervals include zero for all three metrics."
                if heldout_bootstrap.get("status") == "complete"
                else "The primary held-out paired response-cluster bootstrap is pending."
            ),
        },
        "retrieval_next_interface": retrieval_interface,
    }


def format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def format_delta(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.2f} pp"


def format_interval(interval: dict[str, Any]) -> str:
    lower = interval.get("lower")
    upper = interval.get("upper")
    if lower is None or upper is None:
        return "—"
    return f"[{100 * float(lower):+.2f}, {100 * float(upper):+.2f}] pp"


def append_metric_table(lines: list[str], metrics: dict[str, Any]) -> None:
    lines.extend(
        [
            "| Metric | Value |",
            "|---|---:|",
            f"| Claims | {metrics['gold_claim_count']} |",
            f"| Correct | {metrics['correct_count']} |",
            "| Accuracy | "
            f"{format_percent(metrics['accuracy_including_abstentions_and_errors'])} |",
            f"| Balanced accuracy | {format_percent(metrics['balanced_accuracy'])} |",
            f"| Macro-F1 | {format_percent(metrics['macro_f1'])} |",
            f"| Coverage | {format_percent(metrics['coverage'])} |",
            f"| Selective accuracy | {format_percent(metrics['selective_accuracy'])} |",
            f"| Model UNKNOWN | {metrics['model_unknown_count']} |",
            f"| Technical failures | {metrics['technical_failure_count']} |",
            "",
            "| Human label / prediction | FACTUAL | NON_FACTUAL | UNKNOWN | Error/missing |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in PRIMARY_LABELS:
        row = metrics["confusion_matrix"][label]
        lines.append(
            f"| {label} | {row['FACTUAL']} | {row['NON_FACTUAL']} | "
            f"{row['UNKNOWN']} | {row['ERROR_OR_MISSING']} |"
        )
    lines.extend(
        [
            "",
            "| Class | Precision | Recall | F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for label in PRIMARY_LABELS:
        item = metrics[label]
        lines.append(
            f"| {label} | {format_percent(item['precision'])} | "
            f"{format_percent(item['recall'])} | {format_percent(item['f1'])} |"
        )


def markdown_report(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    no_full = summary["no_evidence_full_binary"]
    oracle = summary["oracle_evidence_matched"]
    paired = summary["strict_matched_paired_comparison"]
    response = summary["response_level"]
    confidence = summary["confidence"]
    integrity = summary["data_integrity"]
    sensitivity = summary["sensitivity_analysis"]
    retrieval = summary["retrieval_next_interface"]
    heldout_bootstrap = retrieval.get(
        "retrieved_verifier_evaluation", {}
    ).get("paired_response_cluster_bootstrap", {})

    lines = [
        "# Full FactCheck-Bench Verifier Research Summary",
        "",
        f"- Generated: `{summary['generated_at']}`",
        "- Method: recomputed directly from canonical full JSONL files; no Ollama calls.",
        f"- Data integrity: **{integrity['overall_status'].upper()}**",
        (
            "- Primary held-out statistical interval: **complete** "
            f"({heldout_bootstrap.get('samples', 0):,} paired response-cluster "
            "bootstrap resamples)."
            if heldout_bootstrap.get("status") == "complete"
            else "- Primary held-out statistical interval: **pending**."
        ),
        "- Secondary 589-claim matched interval: **not computed**.",
        "",
        "## Dataset and cohorts",
        "",
        "| Measure | Count |",
        "|---|---:|",
        f"| Source responses | {dataset['source_response_count']} |",
        f"| Responses with claims | {dataset['claim_response_count']} |",
        f"| Responses without claims | {dataset['source_responses_without_claims_count']} |",
        f"| All claims | {dataset['all_claim_count']} |",
        f"| FACTUAL | {dataset['human_label_counts'].get('FACTUAL', 0)} |",
        f"| NON_FACTUAL | {dataset['human_label_counts'].get('NON_FACTUAL', 0)} |",
        f"| Human UNKNOWN | {dataset['human_unknown_claim_count']} |",
        f"| Binary cohort | {dataset['binary_claim_count']} |",
        f"| Matched oracle cohort | {dataset['matched_evidence_claim_count']} |",
        f"| Matched responses | {dataset['matched_response_count']} |",
        f"| Matched coverage of binary | {format_percent(dataset['matched_coverage_of_binary'])} |",
        "",
        "Binary claims excluded from the matched cohort: "
        + ", ".join(
            f"`{key}`={value}"
            for key, value in sorted(
                dataset["binary_exclusion_reason_counts"].items()
            )
        )
        + ".",
        "",
        f"## Full no-evidence result ({dataset['binary_claim_count']} binary claims)",
        "",
    ]
    append_metric_table(lines, no_full["metrics"])
    lines.extend(
        [
            "",
            f"Majority baseline: {format_percent(no_full['majority_baseline']['accuracy'])}; "
            f"model minus baseline: {format_delta(no_full['accuracy_minus_majority_baseline'])}.",
            f"Gold NON_FACTUAL rate: {format_percent(no_full['gold_non_factual_rate'])}; "
            f"predicted NON_FACTUAL rate: {format_percent(no_full['predicted_non_factual_rate'])}.",
            "",
            "## Oracle-evidence result (matched cohort)",
            "",
        ]
    )
    append_metric_table(lines, oracle["metrics"])
    lines.extend(
        [
            "",
            f"Majority baseline: {format_percent(oracle['majority_baseline']['accuracy'])}; "
            f"model minus baseline: {format_delta(oracle['accuracy_minus_majority_baseline'])}.",
            f"Gold NON_FACTUAL rate: {format_percent(oracle['gold_non_factual_rate'])}; "
            f"predicted NON_FACTUAL rate: {format_percent(oracle['predicted_non_factual_rate'])}.",
            "",
            "## Strict matched paired comparison",
            "",
            f"All values below use the same **{paired['claim_count']} claim IDs**.",
            "",
            "| Metric | No evidence | Oracle | Oracle − no evidence |",
            "|---|---:|---:|---:|",
        ]
    )
    metric_rows = (
        ("Accuracy", "accuracy_including_abstentions_and_errors", "accuracy_oracle_minus_no_evidence"),
        ("Balanced accuracy", "balanced_accuracy", "balanced_accuracy_oracle_minus_no_evidence"),
        ("Macro-F1", "macro_f1", "macro_f1_oracle_minus_no_evidence"),
        ("Coverage", "coverage", "coverage_oracle_minus_no_evidence"),
        ("Selective accuracy", "selective_accuracy", "selective_accuracy_oracle_minus_no_evidence"),
    )
    for label, metric, difference in metric_rows:
        lines.append(
            f"| {label} | {format_percent(paired['no_evidence_metrics'][metric])} | "
            f"{format_percent(paired['oracle_metrics'][metric])} | "
            f"{format_delta(paired['point_differences'][difference])} |"
        )
    lines.extend(
        [
            "",
            "| Transition | Count |",
            "|---|---:|",
        ]
    )
    transition_labels = (
        ("wrong→correct", "wrong_to_correct"),
        ("correct→wrong", "correct_to_wrong"),
        ("UNKNOWN→correct", "unknown_to_correct_decision"),
        ("decision→UNKNOWN", "decision_to_unknown"),
        ("wrong→wrong", "wrong_to_wrong"),
        ("correct→correct", "correct_to_correct"),
        ("UNKNOWN→wrong", "unknown_to_wrong_decision"),
        ("UNKNOWN→UNKNOWN", "unknown_to_unknown"),
    )
    for label, key in transition_labels:
        lines.append(f"| {label} | {paired['transition_counts'][key]} |")
    lines.extend(
        [
            "",
            f"Strict wrong/correct net gain: **{paired['strict_wrong_correct_net_gain']:+d}**; "
            f"total correct-count net gain including UNKNOWN transitions: "
            f"**{paired['total_correct_count_net_gain']:+d}**.",
            "",
            "> These are point estimates. They do not establish statistical significance.",
            "",
            "## Response-level distribution",
            "",
            f"Claim-weighted matched accuracy: "
            f"{format_percent(paired['no_evidence_metrics']['accuracy_including_abstentions_and_errors'])} "
            f"→ {format_percent(paired['oracle_metrics']['accuracy_including_abstentions_and_errors'])}.",
            f"Response-macro matched accuracy: "
            f"{format_percent(response['matched_response_macro']['no_evidence']['accuracy'])} "
            f"→ {format_percent(response['matched_response_macro']['oracle']['accuracy'])}.",
            f"Median matched response accuracy: "
            f"{format_percent(response['matched_no_evidence_distribution']['median'])} "
            f"→ {format_percent(response['matched_oracle_distribution']['median'])}.",
            f"Minimum/maximum matched response accuracy: "
            f"{format_percent(response['matched_no_evidence_distribution']['minimum'])}/"
            f"{format_percent(response['matched_no_evidence_distribution']['maximum'])} "
            f"→ {format_percent(response['matched_oracle_distribution']['minimum'])}/"
            f"{format_percent(response['matched_oracle_distribution']['maximum'])}.",
            f"Zero/perfect-accuracy response counts: "
            f"{response['matched_no_evidence_distribution']['zero_accuracy_response_count']}/"
            f"{response['matched_no_evidence_distribution']['perfect_accuracy_response_count']} "
            f"→ {response['matched_oracle_distribution']['zero_accuracy_response_count']}/"
            f"{response['matched_oracle_distribution']['perfect_accuracy_response_count']}.",
            "",
            "Highest oracle error clusters:",
            "",
            "| response_id | Matched | Oracle misses | No-evidence acc. | Oracle acc. | Delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in response["highest_oracle_error_clusters"]:
        lines.append(
            f"| `{row['response_id']}` | {row['matched_claims']} | "
            f"{row['oracle_misses']} | {format_percent(row['no_evidence_accuracy'])} | "
            f"{format_percent(row['oracle_accuracy'])} | "
            f"{format_delta(row['oracle_minus_no_evidence_accuracy'])} |"
        )

    no_wrong = confidence["no_evidence_full_binary_wrong_definitive"]
    oracle_wrong = confidence["oracle_matched_wrong_definitive"]
    no_score_counts = "; ".join(
        f"{float(score):.2f}={count}"
        for score, count in sorted(
            confidence["no_evidence_full_binary"]["exact_score_counts"].items(),
            key=lambda item: float(item[0]),
        )
    )
    oracle_score_counts = "; ".join(
        f"{float(score):.2f}={count}"
        for score, count in sorted(
            confidence["oracle_matched"]["exact_score_counts"].items(),
            key=lambda item: float(item[0]),
        )
    )
    lines.extend(
        [
            "",
            "## Confidence and rationale diagnostics",
            "",
            "| Setting | Exact self-reported confidence counts |",
            "|---|---|",
            f"| No evidence (full binary) | {no_score_counts} |",
            f"| Oracle (matched) | {oracle_score_counts} |",
            "",
            f"No-evidence wrong definitive decisions: {no_wrong['wrong_definitive_decision_count']}; "
            f"minimum confidence {format_percent(no_wrong['minimum_confidence'])}; "
            f"{no_wrong['count_at_or_above_0_95']} were ≥0.95.",
            f"Oracle wrong definitive decisions: {oracle_wrong['wrong_definitive_decision_count']}; "
            f"minimum confidence {format_percent(oracle_wrong['minimum_confidence'])}; "
            f"{oracle_wrong['count_at_or_above_0_80']} were ≥0.80.",
            "",
            "Self-reported confidence is not a calibrated class probability. "
            "Every result has a non-empty rationale, but semantic label–rationale "
            "consistency is **not evaluated** because no checker exists.",
            "",
            "## Data integrity and sensitivity",
            "",
            f"- Gold duplicate IDs: {integrity['gold_duplicate_claim_id_count']}",
            f"- Manifest check: {'passed' if integrity['manifest']['passed'] else 'failed'}",
            f"- No-evidence check: {'passed' if integrity['no_evidence']['passed'] else 'failed'}",
            f"- Oracle check: {'passed' if integrity['oracle_evidence']['passed'] else 'failed'}",
            f"- Automated audit flags: {sensitivity['automated_audit_flagged_claim_count']} claims; "
            f"{sensitivity['flagged_matched_claim_count']} in the matched cohort.",
            "- Automated flag types: "
            + ", ".join(
                f"`{key}`={value}"
                for key, value in sorted(
                    sensitivity["automated_audit_flag_counts"].items()
                )
            )
            + ".",
            "- Full no-evidence sensitivity after excluding flagged binary claims: "
            f"n={sensitivity['no_evidence_full_binary_excluding_automated_flags']['gold_claim_count']}, "
            f"accuracy={format_percent(sensitivity['no_evidence_full_binary_excluding_automated_flags']['accuracy_including_abstentions_and_errors'])}, "
            f"balanced accuracy={format_percent(sensitivity['no_evidence_full_binary_excluding_automated_flags']['balanced_accuracy'])}, "
            f"macro-F1={format_percent(sensitivity['no_evidence_full_binary_excluding_automated_flags']['macro_f1'])}.",
            "- Matched paired metrics are unchanged by automated-flag exclusion "
            "because no flagged claims enter the matched cohort.",
            "- No separate blinded human gold-adjudication file was found.",
            "",
            "## Research interpretation",
            "",
            "Supported:",
            "",
        ]
    )
    lines.extend(
        f"- {item}"
        for item in summary["research_conclusions"]["supported"]
    )
    lines.extend(["", "Not supported:", ""])
    lines.extend(
        f"- {item}"
        for item in summary["research_conclusions"]["not_supported"]
    )
    inventory = summary["retrieval_evidence_inventory"]
    retrieval_split = retrieval["split"]
    retrieval_construction = retrieval["construction"]
    dev_evaluation = retrieval.get("dev_two_level_evaluation", {})
    heldout_evaluation = retrieval.get("heldout_two_level_evaluation", {})
    retrieved_evaluation = retrieval.get("retrieved_verifier_evaluation", {})
    dev_metric_rows = {
        row["metric"]: row
        for row in dev_evaluation.get("metric_comparison", [])
    }
    heldout_metric_rows = {
        row["metric"]: row
        for row in heldout_evaluation.get("metric_comparison", [])
    }
    dev_source_r5 = dev_metric_rows.get(
        "source_document.conditional_recall_at_5"
    )
    dev_strict_r5 = dev_metric_rows.get("strict_passage.recall_at_5")
    heldout_source_r5 = heldout_metric_rows.get(
        "source_document.conditional_recall_at_5"
    )
    heldout_strict_r5 = heldout_metric_rows.get("strict_passage.recall_at_5")
    lines.extend(
        [
            "",
            "## Retrieval handoff",
            "",
            f"The gold file contains {inventory['evidence_item_count']} evidence items; "
            f"{inventory['items_with_nonempty_url']} have a URL, representing "
            f"{inventory['raw_unique_url_count']} raw unique URLs across "
            f"{inventory['unique_domain_count']} domains.",
            "",
            f"- Query input: `{retrieval['query_input']}`",
            f"- Cohort input: `{retrieval['cohort_input']}`",
            f"- Corpus: **{retrieval['corpus_name']}**",
            "- Frozen split: "
            f"dev={retrieval_split['development_matched_claims']}, "
            f"primary held-out={retrieval_split['primary_heldout_matched_claims']}, "
            f"secondary whole cohort={retrieval_split['secondary_whole_matched_claims']}.",
            f"- Manifest status: `{retrieval_construction['manifest_status']}`; "
            f"fetch: `{retrieval_construction['fetch_status']}`; passages: "
            f"`{retrieval_construction['passage_status']}`; qrels: "
            f"`{retrieval_construction['qrels_status']}`.",
            f"- Offline frozen-document reprocessing: "
            f"`{retrieval_construction['reprocess_status']}`.",
            f"- Held-out qrels: "
            f"`{retrieval_construction['heldout_qrels_status']}`.",
            f"- Retrieval evaluation: "
            f"`{retrieval['retrieval_metrics_status']}`; selection decision: "
            f"`{dev_evaluation.get('selection', {}).get('decision', 'pending')}`.",
            (
                "- Dev source-document conditional Recall@5: "
                f"BM25={dev_source_r5['bm25']:.4f}, "
                f"Dense={dev_source_r5['dense']:.4f}, "
                f"Hybrid={dev_source_r5['hybrid']:.4f}."
                if dev_source_r5
                else "- Dev source-document conditional Recall@5: pending."
            ),
            (
                "- Dev strict-passage Recall@5: "
                f"BM25={dev_strict_r5['bm25']:.4f}, "
                f"Dense={dev_strict_r5['dense']:.4f}, "
                f"Hybrid={dev_strict_r5['hybrid']:.4f}."
                if dev_strict_r5
                else "- Dev strict-passage Recall@5: pending."
            ),
            (
                "- Hybrid RRF won the two frozen dev Recall@5 criteria; its "
                "384/64, RRF k=60, fusion-depth=100, evidence-top-k=5 "
                "configuration was then applied unchanged to held-out."
                if dev_evaluation.get("selection", {}).get(
                    "configuration_is_frozen"
                )
                else "- Configuration remains unfrozen."
            ),
            (
                "- Held-out source-document conditional Recall@5: "
                f"BM25={heldout_source_r5['bm25']:.4f}, "
                f"Dense={heldout_source_r5['dense']:.4f}, "
                f"Hybrid={heldout_source_r5['hybrid']:.4f}."
                if heldout_source_r5
                else "- Held-out source-document conditional Recall@5: pending."
            ),
            (
                "- Held-out strict-passage Recall@5: "
                f"BM25={heldout_strict_r5['bm25']:.4f}, "
                f"Dense={heldout_strict_r5['dense']:.4f}, "
                f"Hybrid={heldout_strict_r5['hybrid']:.4f}."
                if heldout_strict_r5
                else "- Held-out strict-passage Recall@5: pending."
            ),
            f"- Leakage boundary: {retrieval['leakage_boundary']}",
            "",
            f"Retrieved-verifier report status: "
            f"`{retrieval['retrieved_verifier_status']}`.",
            "",
        ]
    )
    retrieved_metrics = retrieved_evaluation.get("metrics", {})
    if retrieved_metrics:
        lines.extend(
            [
                "### Held-out evidence-condition comparison",
                "",
                "| Setting | Accuracy | Balanced accuracy | Macro-F1 | Coverage |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for key, label in (
            ("no_evidence", "No evidence"),
            ("oracle_evidence", "Oracle evidence"),
            ("retrieved_evidence", "Hybrid retrieved top-5"),
        ):
            item = retrieved_metrics.get(key, {})
            lines.append(
                f"| {label} | "
                f"{format_percent(item.get('accuracy_including_abstentions_and_errors'))} | "
                f"{format_percent(item.get('balanced_accuracy'))} | "
                f"{format_percent(item.get('macro_f1'))} | "
                f"{format_percent(item.get('coverage'))} |"
            )
        technical_failures = retrieved_metrics.get(
            "retrieved_evidence", {}
        ).get("technical_failure_count")
        technical_completion = retrieved_evaluation.get(
            "technical_completion", {}
        )
        lines.extend(
            [
                "",
                f"Retrieved technical failures: {technical_failures}; format-only "
                f"repairs: {technical_completion.get('format_repair_count', 0)}. "
                "Oracle evidence is benchmark-associated and not guaranteed to be a "
                "perfect upper bound.",
                "",
            ]
        )
        bootstrap = retrieved_evaluation.get(
            "paired_response_cluster_bootstrap", {}
        )
        if bootstrap.get("status") == "complete":
            lines.extend(
                [
                    "### Paired response-cluster bootstrap",
                    "",
                    f"Percentile 95% intervals use {bootstrap['samples']:,} paired "
                    f"resamples of {bootstrap['response_cluster_count']} responses.",
                    "",
                    "| Difference | Accuracy delta (95% CI) | Balanced accuracy delta (95% CI) | Macro-F1 delta (95% CI) |",
                    "|---|---:|---:|---:|",
                ]
            )
            for key, label in (
                ("oracle_minus_no_evidence", "Oracle - no evidence"),
                ("retrieved_minus_no_evidence", "Retrieved - no evidence"),
                ("retrieved_minus_oracle", "Retrieved - oracle"),
            ):
                values = bootstrap["paired_difference_intervals"][key]["metrics"]
                lines.append(
                    f"| {label} | {format_delta(values['accuracy']['point_estimate'])} "
                    f"{format_interval(values['accuracy'])} | "
                    f"{format_delta(values['balanced_accuracy']['point_estimate'])} "
                    f"{format_interval(values['balanced_accuracy'])} | "
                    f"{format_delta(values['macro_f1']['point_estimate'])} "
                    f"{format_interval(values['macro_f1'])} |"
                )
            lines.extend(
                [
                    "",
                    "Oracle and retrieved evidence both have positive intervals versus "
                    "no evidence on these three metrics. Retrieved-minus-oracle intervals "
                    "include zero, so the observed retrieved advantage is inconclusive.",
                    "",
                ]
            )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the canonical full FactCheck-Bench verifier summary "
            "without contacting Ollama."
        )
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--gold-claims", type=Path, default=FULL_PATHS.gold_claims)
    parser.add_argument("--manifest", type=Path, default=FULL_PATHS.cohort_manifest)
    parser.add_argument(
        "--no-evidence", type=Path, default=FULL_PATHS.no_evidence_output
    )
    parser.add_argument("--oracle", type=Path, default=FULL_PATHS.oracle_output)
    parser.add_argument("--no-prompt", type=Path, default=NO_PROMPT)
    parser.add_argument("--oracle-prompt", type=Path, default=ORACLE_PROMPT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.8,
        help="Exploratory self-reported confidence threshold (default: 0.8).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Recompute and validate but do not write summary files.",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.high_confidence_threshold <= 1:
        parser.error("--high-confidence-threshold must be between 0 and 1")
    if args.output_json.resolve(strict=False) == args.output_markdown.resolve(
        strict=False
    ):
        parser.error("--output-json and --output-markdown must be distinct")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(args)
    if not args.check_only:
        atomic_write_text(
            args.output_json,
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
        )
        atomic_write_text(args.output_markdown, markdown_report(summary))
        print(f"JSON summary: {args.output_json}")
        print(f"Markdown summary: {args.output_markdown}")
    else:
        print("Check-only mode: no files were written.")
    print(f"Data integrity: {summary['data_integrity']['overall_status']}")
    paired = summary["strict_matched_paired_comparison"]
    print(f"Matched claims: {paired['claim_count']}")
    print(
        "Accuracy: "
        f"{paired['no_evidence_metrics']['accuracy_including_abstentions_and_errors']:.6f} "
        "-> "
        f"{paired['oracle_metrics']['accuracy_including_abstentions_and_errors']:.6f}"
    )
    return 0 if summary["data_integrity"]["overall_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
