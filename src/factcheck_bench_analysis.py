"""Reusable claim- and response-level FactCheck-Bench metrics."""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


LABELS = ("FACTUAL", "NON_FACTUAL", "UNKNOWN")
PRIMARY_LABELS = ("FACTUAL", "NON_FACTUAL")
PAIRED_STATES = (
    "CORRECT_DECISION",
    "WRONG_DECISION",
    "UNKNOWN",
    "TECHNICAL_ERROR_OR_MISSING",
)
BOOTSTRAP_METRIC_FIELDS = {
    "accuracy": "accuracy_including_abstentions_and_errors",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "macro_f1",
    "coverage": "coverage",
    "selective_accuracy": "selective_accuracy",
}


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def safe_mean(values: Iterable[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return statistics.fmean(usable) if usable else None


def f1_from_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    predicted_count = tp + fp
    gold_support = tp + fn
    if gold_support == 0:
        precision = recall = f1 = None
    else:
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / gold_support
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def prediction_bucket(result: dict[str, Any] | None) -> str:
    if result is None or result.get("status") != "ok":
        return "ERROR_OR_MISSING"
    prediction = result.get("prediction")
    return str(prediction) if prediction in LABELS else "ERROR_OR_MISSING"


def compute_binary_metrics(
    records: Iterable[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Score binary human labels while retaining abstentions/failures."""
    primary = [
        record
        for record in records
        if record.get("human_label") in PRIMARY_LABELS
    ]
    pairs = [
        (record, result_by_id.get(str(record["claim_id"])))
        for record in primary
    ]
    correct = sum(
        result is not None
        and result.get("status") == "ok"
        and result.get("prediction") == record.get("human_label")
        for record, result in pairs
    )
    answered = [
        (record, result)
        for record, result in pairs
        if result is not None
        and result.get("status") == "ok"
        and result.get("prediction") in PRIMARY_LABELS
    ]
    unknown = sum(
        result is not None
        and result.get("status") == "ok"
        and result.get("prediction") == "UNKNOWN"
        for _, result in pairs
    )
    failures = sum(
        result is None or result.get("status") != "ok"
        for _, result in pairs
    )

    per_class: dict[str, dict[str, Any]] = {}
    for label in PRIMARY_LABELS:
        tp = sum(
            record.get("human_label") == label
            and result is not None
            and result.get("status") == "ok"
            and result.get("prediction") == label
            for record, result in pairs
        )
        fp = sum(
            record.get("human_label") != label
            and result is not None
            and result.get("status") == "ok"
            and result.get("prediction") == label
            for record, result in pairs
        )
        fn = sum(
            record.get("human_label") == label
            and not (
                result is not None
                and result.get("status") == "ok"
                and result.get("prediction") == label
            )
            for record, result in pairs
        )
        per_class[label] = f1_from_counts(tp, fp, fn)

    columns = (*LABELS, "ERROR_OR_MISSING")
    confusion = {
        label: {column: 0 for column in columns}
        for label in PRIMARY_LABELS
    }
    for record, result in pairs:
        confusion[str(record["human_label"])][prediction_bucket(result)] += 1

    total = len(primary)
    return {
        "cohort_definition": "human FACTUAL/NON_FACTUAL claims",
        "gold_claim_count": total,
        "gold_label_counts": dict(
            Counter(str(record["human_label"]) for record in primary)
        ),
        "correct_count": correct,
        "accuracy_including_abstentions_and_errors": safe_ratio(correct, total),
        "balanced_accuracy": safe_mean(
            per_class[label]["recall"] for label in PRIMARY_LABELS
        ),
        "answered_count": len(answered),
        "coverage": safe_ratio(len(answered), total),
        "selective_accuracy": safe_ratio(correct, len(answered)),
        "model_unknown_count": unknown,
        "abstention_rate": safe_ratio(unknown, total),
        "technical_failure_count": failures,
        "FACTUAL": per_class["FACTUAL"],
        "NON_FACTUAL": per_class["NON_FACTUAL"],
        "macro_f1": safe_mean(
            per_class[label]["f1"] for label in PRIMARY_LABELS
        ),
        "confusion_matrix": confusion,
    }


def build_response_aggregation(
    records: Iterable[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return per-response metrics and an equal-response macro summary."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["response_id"])].append(record)

    rows: list[dict[str, Any]] = []
    for response_id in sorted(grouped):
        response_records = grouped[response_id]
        metrics = compute_binary_metrics(response_records, result_by_id)
        rows.append(
            {
                "response_id": response_id,
                "binary_claims": metrics["gold_claim_count"],
                "correct_count": metrics["correct_count"],
                "accuracy": metrics[
                    "accuracy_including_abstentions_and_errors"
                ],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "coverage": metrics["coverage"],
                "selective_accuracy": metrics["selective_accuracy"],
                "model_unknown_count": metrics["model_unknown_count"],
                "technical_failure_count": metrics["technical_failure_count"],
            }
        )

    nonempty = [row for row in rows if row["binary_claims"] > 0]
    macro = {
        "cohort_definition": "equal-weight mean across responses with binary claims",
        "response_count": len(nonempty),
        "accuracy": safe_mean(row["accuracy"] for row in nonempty),
        "balanced_accuracy": safe_mean(
            row["balanced_accuracy"] for row in nonempty
        ),
        "macro_f1": safe_mean(row["macro_f1"] for row in nonempty),
        "coverage": safe_mean(row["coverage"] for row in nonempty),
        "selective_accuracy": safe_mean(
            row["selective_accuracy"] for row in nonempty
        ),
    }
    return rows, macro


def build_confidence_distribution(
    records: Iterable[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
    high_confidence_threshold: float,
) -> dict[str, Any]:
    """Summarize exact scalar self-reports without treating them as probabilities."""
    pairs = [
        (record, result_by_id.get(str(record["claim_id"])))
        for record in records
        if record.get("human_label") in PRIMARY_LABELS
    ]
    ok_pairs = [
        (record, result)
        for record, result in pairs
        if result is not None and result.get("status") == "ok"
    ]
    exact_scores: Counter[str] = Counter()
    by_prediction: dict[str, list[float]] = defaultdict(list)
    correct_scores: list[float] = []
    incorrect_scores: list[float] = []
    high_confidence_error_ids: list[str] = []
    for record, result in ok_pairs:
        score = float(result["confidence"])
        exact_scores[f"{score:.6g}"] += 1
        by_prediction[str(result["prediction"])].append(score)
        if result["prediction"] == record["human_label"]:
            correct_scores.append(score)
        else:
            incorrect_scores.append(score)
            if score >= high_confidence_threshold:
                high_confidence_error_ids.append(str(record["claim_id"]))

    return {
        "semantics": "self-reported confidence that the selected label is appropriate",
        "valid_prediction_count": len(ok_pairs),
        "exact_score_counts": dict(
            sorted(exact_scores.items(), key=lambda item: float(item[0]))
        ),
        "by_prediction": {
            label: {
                "count": len(by_prediction.get(label, [])),
                "mean": safe_mean(by_prediction.get(label, [])),
            }
            for label in LABELS
        },
        "mean_confidence_correct": safe_mean(correct_scores),
        "mean_confidence_incorrect_or_abstained": safe_mean(incorrect_scores),
        "high_confidence_threshold": high_confidence_threshold,
        "high_confidence_error_count": len(high_confidence_error_ids),
        "high_confidence_error_claim_ids": high_confidence_error_ids,
        "calibration_warning": (
            "This scalar is not a full class-probability distribution; multiclass "
            "Brier score, log loss, and ECE are not computed."
        ),
    }


def paired_state(
    record: dict[str, Any],
    result: dict[str, Any] | None,
) -> str:
    """Map one gold/result pair to a mutually exclusive paired state."""
    if result is None or result.get("status") != "ok":
        return "TECHNICAL_ERROR_OR_MISSING"
    prediction = result.get("prediction")
    if prediction == "UNKNOWN":
        return "UNKNOWN"
    if prediction == record.get("human_label"):
        return "CORRECT_DECISION"
    if prediction in PRIMARY_LABELS:
        return "WRONG_DECISION"
    return "TECHNICAL_ERROR_OR_MISSING"


def build_paired_transitions(
    records: Iterable[dict[str, Any]],
    before_by_id: dict[str, dict[str, Any]],
    after_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build an exhaustive transition matrix for two paired settings."""
    matrix = {
        before: {after: 0 for after in PAIRED_STATES}
        for before in PAIRED_STATES
    }
    state_pair_ids: dict[str, list[str]] = defaultdict(list)
    for record in records:
        claim_id = str(record["claim_id"])
        before = paired_state(record, before_by_id.get(claim_id))
        after = paired_state(record, after_by_id.get(claim_id))
        matrix[before][after] += 1
        state_pair_ids[f"{before}->{after}"].append(claim_id)

    def transition(before: str, after: str) -> dict[str, Any]:
        claim_ids = state_pair_ids.get(f"{before}->{after}", [])
        return {"count": len(claim_ids), "claim_ids": claim_ids}

    decision_to_unknown_ids = sorted(
        state_pair_ids.get("CORRECT_DECISION->UNKNOWN", [])
        + state_pair_ids.get("WRONG_DECISION->UNKNOWN", [])
    )
    named = {
        "wrong_to_correct": transition(
            "WRONG_DECISION", "CORRECT_DECISION"
        ),
        "correct_to_wrong": transition(
            "CORRECT_DECISION", "WRONG_DECISION"
        ),
        "unknown_to_correct_decision": transition(
            "UNKNOWN", "CORRECT_DECISION"
        ),
        "decision_to_unknown": {
            "count": len(decision_to_unknown_ids),
            "claim_ids": decision_to_unknown_ids,
        },
        "wrong_to_wrong": transition("WRONG_DECISION", "WRONG_DECISION"),
        "correct_to_correct": transition(
            "CORRECT_DECISION", "CORRECT_DECISION"
        ),
        "unknown_to_wrong_decision": transition(
            "UNKNOWN", "WRONG_DECISION"
        ),
        "unknown_to_unknown": transition("UNKNOWN", "UNKNOWN"),
    }
    return {
        "state_definitions": {
            "CORRECT_DECISION": (
                "status=ok and prediction equals the human binary label"
            ),
            "WRONG_DECISION": (
                "status=ok and prediction is the opposite definitive label"
            ),
            "UNKNOWN": "status=ok and prediction=UNKNOWN",
            "TECHNICAL_ERROR_OR_MISSING": (
                "missing row or non-ok technical status"
            ),
        },
        "state_matrix_before_to_after": matrix,
        # Backward-compatible key used by the oracle report schema.
        "state_matrix_no_evidence_to_oracle": matrix,
        "state_pair_claim_ids": dict(sorted(state_pair_ids.items())),
        "named_transitions": named,
        "matrix_total": sum(sum(row.values()) for row in matrix.values()),
    }


def metric_differences(
    before: dict[str, Any],
    after: dict[str, Any],
    prefix: str = "oracle_minus_no_evidence",
) -> dict[str, float | None]:
    """Subtract common scalar verifier metrics using explicit output names."""
    scalar_fields = {
        "accuracy": "accuracy_including_abstentions_and_errors",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "macro_f1",
        "coverage": "coverage",
        "selective_accuracy": "selective_accuracy",
    }
    differences: dict[str, float | None] = {}
    for output_name, source_name in scalar_fields.items():
        before_value = before.get(source_name)
        after_value = after.get(source_name)
        differences[f"{output_name}_{prefix}"] = (
            None
            if before_value is None or after_value is None
            else float(after_value) - float(before_value)
        )
    return differences


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a percentile from no values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_metric_counts(
    records: Sequence[dict[str, Any]],
    result_by_id: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics = compute_binary_metrics(records, dict(result_by_id))
    return {
        "total": metrics["gold_claim_count"],
        "correct": metrics["correct_count"],
        "answered": metrics["answered_count"],
        "unknown": metrics["model_unknown_count"],
        "failures": metrics["technical_failure_count"],
        "per_class": {
            label: {
                "tp": metrics[label]["true_positive"],
                "fp": metrics[label]["false_positive"],
                "fn": metrics[label]["false_negative"],
            }
            for label in PRIMARY_LABELS
        },
    }


def _metrics_from_cluster_counts(counts: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    total = sum(int(item["total"]) for item in counts)
    correct = sum(int(item["correct"]) for item in counts)
    answered = sum(int(item["answered"]) for item in counts)
    per_class = {}
    for label in PRIMARY_LABELS:
        per_class[label] = f1_from_counts(
            sum(int(item["per_class"][label]["tp"]) for item in counts),
            sum(int(item["per_class"][label]["fp"]) for item in counts),
            sum(int(item["per_class"][label]["fn"]) for item in counts),
        )
    return {
        "accuracy": safe_ratio(correct, total),
        "balanced_accuracy": safe_mean(
            per_class[label]["recall"] for label in PRIMARY_LABELS
        ),
        "macro_f1": safe_mean(
            per_class[label]["f1"] for label in PRIMARY_LABELS
        ),
        "coverage": safe_ratio(answered, total),
        "selective_accuracy": safe_ratio(correct, answered),
    }


def paired_response_cluster_bootstrap(
    records: Sequence[dict[str, Any]],
    result_sets: Mapping[str, Mapping[str, dict[str, Any]]],
    comparisons: Sequence[tuple[str, str, str]],
    *,
    samples: int = 10_000,
    seed: int = 20_260_722,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Percentile intervals from paired response-level cluster resamples.

    ``comparisons`` entries are ``(output_name, after_setting, before_setting)``.
    Every setting uses the identical resampled response IDs in each iteration.
    All claims belonging to a selected response are retained together, while
    aggregate metrics remain claim-weighted within each bootstrap replicate.
    """
    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    primary = [
        record for record in records if record.get("human_label") in PRIMARY_LABELS
    ]
    claim_ids = [str(record["claim_id"]) for record in primary]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Bootstrap records contain duplicate claim IDs")
    required_ids = set(claim_ids)
    if not result_sets:
        raise ValueError("At least one result set is required")
    for setting, result_by_id in result_sets.items():
        if set(result_by_id) != required_ids:
            raise ValueError(
                f"{setting} must contain the exact bootstrap claim-ID set"
            )
    for name, after, before in comparisons:
        if not name or after not in result_sets or before not in result_sets:
            raise ValueError(f"Invalid paired comparison: {(name, after, before)!r}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in primary:
        grouped[str(record["response_id"])].append(record)
    response_ids = sorted(grouped)
    if not response_ids:
        raise ValueError("Bootstrap cohort has no response clusters")

    cluster_counts = {
        setting: {
            response_id: _cluster_metric_counts(
                grouped[response_id], result_by_id
            )
            for response_id in response_ids
        }
        for setting, result_by_id in result_sets.items()
    }
    point_metrics = {
        setting: compute_binary_metrics(primary, dict(result_by_id))
        for setting, result_by_id in result_sets.items()
    }
    setting_samples = {
        setting: {metric: [] for metric in BOOTSTRAP_METRIC_FIELDS}
        for setting in result_sets
    }
    difference_samples = {
        name: {metric: [] for metric in BOOTSTRAP_METRIC_FIELDS}
        for name, _, _ in comparisons
    }

    rng = random.Random(seed)
    for _ in range(samples):
        selected = [rng.choice(response_ids) for _ in response_ids]
        replicate: dict[str, dict[str, float | None]] = {}
        for setting in result_sets:
            replicate[setting] = _metrics_from_cluster_counts(
                [cluster_counts[setting][response_id] for response_id in selected]
            )
            for metric, value in replicate[setting].items():
                if value is not None:
                    setting_samples[setting][metric].append(float(value))
        for name, after, before in comparisons:
            for metric in BOOTSTRAP_METRIC_FIELDS:
                after_value = replicate[after][metric]
                before_value = replicate[before][metric]
                if after_value is not None and before_value is not None:
                    difference_samples[name][metric].append(
                        float(after_value) - float(before_value)
                    )

    alpha = 1.0 - confidence_level

    def interval(values: Sequence[float], point: float | None) -> dict[str, Any]:
        return {
            "point_estimate": point,
            "lower": _percentile(values, alpha / 2.0),
            "upper": _percentile(values, 1.0 - alpha / 2.0),
            "valid_replicates": len(values),
            "includes_zero": (
                None
                if point is None
                else _percentile(values, alpha / 2.0) <= 0.0
                <= _percentile(values, 1.0 - alpha / 2.0)
            ),
        }

    setting_intervals = {}
    for setting, metrics in point_metrics.items():
        setting_intervals[setting] = {
            metric: interval(
                setting_samples[setting][metric], metrics.get(source_field)
            )
            for metric, source_field in BOOTSTRAP_METRIC_FIELDS.items()
        }

    paired_intervals = {}
    for name, after, before in comparisons:
        paired_intervals[name] = {
            "after": after,
            "before": before,
            "estimand": f"{after}_minus_{before}",
            "metrics": {
                metric: interval(
                    difference_samples[name][metric],
                    (
                        None
                        if point_metrics[after].get(source_field) is None
                        or point_metrics[before].get(source_field) is None
                        else float(point_metrics[after][source_field])
                        - float(point_metrics[before][source_field])
                    ),
                )
                for metric, source_field in BOOTSTRAP_METRIC_FIELDS.items()
            },
        }

    return {
        "status": "complete",
        "method": "paired_response_cluster_percentile_bootstrap",
        "cluster_unit": "response_id",
        "pairing": (
            "Identical response-ID resamples are used for every setting in each "
            "replicate; all claims in a selected response remain together."
        ),
        "estimand_weighting": "claim_weighted_within_each_cluster_resample",
        "interval_type": "percentile",
        "confidence_level": confidence_level,
        "samples": samples,
        "seed": seed,
        "response_cluster_count": len(response_ids),
        "claim_count": len(primary),
        "setting_intervals": setting_intervals,
        "paired_difference_intervals": paired_intervals,
        "limitations": [
            "Intervals quantify sampling variation across observed responses only.",
            "They do not correct benchmark-label, qrel, corpus, or model biases.",
            "An interval including zero is reported as inconclusive, not proof of equivalence.",
        ],
    }
