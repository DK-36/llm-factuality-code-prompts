#!/usr/bin/env python3
"""Export frozen CoVe responses to VeriScore and analyse official output.

This adapter deliberately does not vendor, call, or modify VeriScore. X1 creates
the official four-field JSONL input and an auditable local unit manifest. The
official package runs in a separate environment. X2 validates the returned rows
against X1 before calculating the published response-level F1@K aggregation and
paired response bootstrap intervals.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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
    sha256_text,
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "factcheck_bench"
    / "cove"
    / "config"
    / "cove_external_veriscore_config.json"
)
ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_full"
    / "cove"
    / "external_evaluation"
    / "veriscore"
)
INPUT_DIR = ROOT / "input"
VENDOR_DIR = ROOT / "vendor_output"
REPORTS_DIR = ROOT / "reports"
STAGES = ("prepare", "analyze")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def config() -> dict[str, Any]:
    cfg = load_json(CONFIG_PATH)
    if cfg.get("schema_version") != "fcb_cove_external_veriscore_config_v1":
        raise ValueError("Unexpected external VeriScore config schema")
    return cfg


def paths(split: str) -> dict[str, Path]:
    return {
        "official_input": INPUT_DIR / f"factcheck_bench_cove_{split}.jsonl",
        "unit_manifest": INPUT_DIR / f"X1_veriscore_units_{split}.jsonl",
        "metadata_template": INPUT_DIR / f"veriscore_run_metadata_template_{split}.json",
        "export_json": REPORTS_DIR / f"X1_veriscore_export_{split}_summary.json",
        "export_md": REPORTS_DIR / f"X1_veriscore_export_{split}.md",
        "response_scores": VENDOR_DIR / f"X2_veriscore_response_scores_{split}.jsonl",
        "analysis_json": REPORTS_DIR / f"X2_veriscore_analysis_{split}_summary.json",
        "analysis_md": REPORTS_DIR / f"X2_veriscore_analysis_{split}.md",
    }


def _require_text(row: dict[str, Any], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty {key} in {context}")
    return value.strip()


def _unique_by(rows: Iterable[dict[str, Any]], key: str, context: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = _require_text(row, key, context)
        if value in result:
            raise ValueError(f"Duplicate {key}={value!r} in {context}")
        result[value] = row
    return result


def _load_frozen_responses(split: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    cfg = config()
    expected = int(cfg["splits"][split]["expected_responses"])
    base_paths = cove_paths(PROJECT_ROOT, "full", "a")
    manifest_rows = [
        row for row in load_jsonl(base_paths.response_manifest)
        if row.get("split") == split
    ]
    manifest = _unique_by(manifest_rows, "response_id", "CoVe response manifest")
    if len(manifest) != expected:
        raise ValueError(f"Expected {expected} {split} responses, found {len(manifest)}")

    revisions: dict[str, dict[str, dict[str, Any]]] = {}
    source_paths = {"response_manifest": base_paths.response_manifest}
    for branch in ("a", "b", "c", "d2"):
        revision_path = cove_paths(PROJECT_ROOT, "full", branch).revision_results(split)
        rows = load_jsonl(revision_path)
        revisions[branch] = _unique_by(rows, "response_id", f"Branch {branch} revisions")
        source_paths[f"branch_{branch}_revisions"] = revision_path
        if set(revisions[branch]) != set(manifest):
            missing = sorted(set(manifest) - set(revisions[branch]))
            extra = sorted(set(revisions[branch]) - set(manifest))
            raise ValueError(
                f"Branch {branch} response set mismatch; missing={missing}, extra={extra}"
            )

    conditions = {item["condition_id"]: item for item in cfg["conditions"]}
    expected_conditions = {"initial", "a", "b", "c", "d2"}
    if set(conditions) != expected_conditions:
        raise ValueError("External evaluator conditions do not match the frozen design")

    units: list[dict[str, Any]] = []
    for response_id in sorted(manifest):
        source = manifest[response_id]
        question = _require_text(source, "original_question", response_id)
        for condition_id in ("initial", "a", "b", "c", "d2"):
            condition = conditions[condition_id]
            if condition_id == "initial":
                response = _require_text(source, "initial_response", response_id)
                response_source = "cove_response_manifest.initial_response"
                fallback_applied = False
            else:
                revision = revisions[condition_id][response_id]
                response = _require_text(revision, "revised_response", response_id)
                if revision.get("status") != "ok":
                    raise ValueError(
                        f"Branch {condition_id} response {response_id} is not status=ok"
                    )
                expected_hash = revision.get("revised_response_sha256")
                if expected_hash and sha256_text(response) != expected_hash:
                    raise ValueError(
                        f"Branch {condition_id} response hash mismatch for {response_id}"
                    )
                response_source = relative(source_paths[f"branch_{condition_id}_revisions"])
                fallback_applied = revision.get("fallback_applied") is True
            unit_id = f"veriscore_{split}_{response_id}_{condition_id}"
            units.append({
                "schema_version": "fcb_cove_veriscore_unit_v1",
                "evaluation_unit_id": unit_id,
                "response_id": response_id,
                "source_record_index": source.get("source_record_index"),
                "split": split,
                "condition_id": condition_id,
                "condition_display_name": condition["display_name"],
                "veriscore_model_name": condition["veriscore_model_name"],
                "prompt_source": cfg["prompt_source"],
                "question": question,
                "response": response,
                "question_sha256": sha256_text(question),
                "response_sha256": sha256_text(response),
                "response_source": response_source,
                "fallback_applied": fallback_applied,
                "gold_fields_included": [],
            })

    expected_units = expected * len(conditions)
    if len(units) != expected_units:
        raise ValueError(f"Expected {expected_units} units, found {len(units)}")
    fingerprints = {
        relative(path): sha256_file(path) for path in source_paths.values()
    }
    return units, fingerprints


def prepare(split: str, dry_run: bool) -> int:
    cfg = config()
    out = paths(split)
    units, source_fingerprints = _load_frozen_responses(split)
    official_rows = [{
        "question": row["question"],
        "response": row["response"],
        "model": row["veriscore_model_name"],
        "prompt_source": row["prompt_source"],
    } for row in units]
    condition_counts: defaultdict[str, int] = defaultdict(int)
    fallback_counts: defaultdict[str, int] = defaultdict(int)
    for row in units:
        condition_counts[row["condition_id"]] += 1
        fallback_counts[row["condition_id"]] += int(row["fallback_applied"])
    summary = {
        "schema_version": "fcb_cove_veriscore_export_summary_v1",
        "status": "external_run_pending",
        "scope": "full",
        "split": split,
        "response_count": len({row["response_id"] for row in units}),
        "condition_count": len(condition_counts),
        "evaluation_unit_count": len(units),
        "condition_counts": dict(condition_counts),
        "fallback_counts": dict(fallback_counts),
        "official_input_fields": ["question", "response", "model", "prompt_source"],
        "official_protocol": cfg["official_protocol"],
        "isolation": cfg["isolation"],
        "source_fingerprints": source_fingerprints,
        "config_sha256": sha256_file(CONFIG_PATH),
        "generated_at": utc_now(),
        "artifacts": {key: relative(value) for key, value in out.items() if key in {
            "official_input", "unit_manifest", "metadata_template", "export_json", "export_md"
        }},
    }
    metadata_template = {
        "schema_version": "fcb_external_veriscore_run_metadata_v1",
        "status": "REPLACE_WITH_complete_BEFORE_X2",
        "split": split,
        "official_repository": cfg["official_protocol"]["official_repository"],
        "official_repository_commit": cfg["official_protocol"]["official_repository_commit"],
        "package_version": cfg["official_protocol"]["expected_package_version"],
        "extraction_model": cfg["official_protocol"]["extraction_model"],
        "verification_model": cfg["official_protocol"]["verification_model"],
        "model_access_mode": cfg["official_protocol"]["model_access_mode"],
        "provider": cfg["official_protocol"]["provider"],
        "provider_base_url": cfg["official_protocol"]["provider_base_url"],
        "thinking_mode": cfg["official_protocol"]["thinking_mode"],
        "temperature": cfg["official_protocol"]["temperature"],
        "provider_adapter_path": cfg["official_protocol"]["provider_adapter"]["path"],
        "provider_adapter_sha256": cfg["official_protocol"]["provider_adapter"]["sha256"],
        "label_n": cfg["official_protocol"]["label_n"],
        "search_provider": cfg["official_protocol"]["search_provider"],
        "search_res_num": cfg["official_protocol"]["search_res_num"],
        "run_started_at": None,
        "run_completed_at": None,
        "notes": None,
    }
    lines = [
        f"# X1 — VeriScore Export ({split})",
        "",
        f"- Responses: {summary['response_count']}",
        f"- Conditions: {summary['condition_count']} (Initial, A, B, C, D)",
        f"- Official input rows: {summary['evaluation_unit_count']}",
        "- Status: external official VeriScore run pending",
        "",
        "## Isolation boundary",
        "",
        "X1 copies only the original question and frozen response text. It does not "
        "reuse B6a claims, Hybrid passages, qrels, gold claims, labels, or evidence. "
        "The official VeriScore package must run out of process with the same "
        "extractor, verifier, search settings, and input file for every condition.",
        "",
        "## Frozen scoring protocol",
        "",
        f"- Verification labels: {cfg['official_protocol']['label_n']}",
        f"- Search results per claim: {cfg['official_protocol']['search_res_num']}",
        "- Shared K: upper median claim count across all five conditions in this split",
        "- Primary metric: response-level F1@shared median K",
        "",
        "The external result is a supplementary automatic response-level metric. It "
        "does not replace human adjudication and is not merged into V10's three "
        "evidence-strength layers.",
        "",
    ]
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    atomic_write_jsonl(out["unit_manifest"], units)
    atomic_write_jsonl(out["official_input"], official_rows)
    atomic_write_json(out["metadata_template"], metadata_template)
    summary["official_input_sha256"] = sha256_file(out["official_input"])
    summary["unit_manifest_sha256"] = sha256_file(out["unit_manifest"])
    atomic_write_json(out["export_json"], summary)
    atomic_write_text(out["export_md"], "\n".join(lines))
    print(json.dumps({
        "stage": "X1_prepare_veriscore_export",
        "status": "complete",
        "split": split,
        "responses": summary["response_count"],
        "conditions": summary["condition_count"],
        "rows": summary["evaluation_unit_count"],
        "official_input": relative(out["official_input"]),
        "metadata_template": relative(out["metadata_template"]),
    }, ensure_ascii=False, indent=2))
    return 0


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values:
        raise ValueError(f"No values available for {key}")
    return statistics.fmean(values)


def _validate_metadata(path: Path, split: str, cfg: dict[str, Any]) -> dict[str, Any]:
    metadata = load_json(path)
    if metadata.get("schema_version") != "fcb_external_veriscore_run_metadata_v1":
        raise ValueError("Unexpected VeriScore run metadata schema")
    if metadata.get("status") != "complete":
        raise ValueError("VeriScore run metadata status must be 'complete'")
    if metadata.get("split") != split:
        raise ValueError("VeriScore run metadata split mismatch")
    required = (
        "official_repository_commit", "package_version", "extraction_model",
        "verification_model", "model_access_mode", "provider",
        "provider_base_url", "thinking_mode", "provider_adapter_path",
        "provider_adapter_sha256", "run_started_at", "run_completed_at",
    )
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ValueError(f"Incomplete VeriScore run metadata fields: {missing}")
    protocol = cfg["official_protocol"]
    expected_text = {
        "official_repository": protocol["official_repository"],
        "official_repository_commit": protocol["official_repository_commit"],
        "package_version": protocol["expected_package_version"],
        "extraction_model": protocol["extraction_model"],
        "verification_model": protocol["verification_model"],
        "model_access_mode": protocol["model_access_mode"],
        "provider": protocol["provider"],
        "provider_base_url": protocol["provider_base_url"],
        "thinking_mode": protocol["thinking_mode"],
        "provider_adapter_path": protocol["provider_adapter"]["path"],
        "provider_adapter_sha256": protocol["provider_adapter"]["sha256"],
    }
    for key, expected in expected_text.items():
        if metadata.get(key) != expected:
            raise ValueError(f"VeriScore metadata {key} does not match frozen config")
    for key in ("label_n", "search_res_num"):
        if int(metadata.get(key, -1)) != int(protocol[key]):
            raise ValueError(f"VeriScore metadata {key} does not match frozen config")
    if float(metadata.get("temperature", -1)) != float(protocol["temperature"]):
        raise ValueError("VeriScore metadata temperature does not match frozen config")
    if metadata.get("search_provider") != protocol["search_provider"]:
        raise ValueError("VeriScore metadata search_provider does not match frozen config")
    return metadata


def _score_official_rows(
    units: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
    label_n: int,
) -> tuple[list[dict[str, Any]], int]:
    if len(units) != len(official_rows):
        raise ValueError(
            f"Official output has {len(official_rows)} rows; X1 expects {len(units)}"
        )
    allowed = {"supported", "unsupported"} if label_n == 2 else {
        "supported", "contradicted", "inconclusive"
    }
    scored: list[dict[str, Any]] = []
    for index, (unit, result) in enumerate(zip(units, official_rows, strict=True), start=1):
        checks = {
            "question": unit["question"],
            "response": unit["response"],
            "model": unit["veriscore_model_name"],
            "prompt_source": unit["prompt_source"],
        }
        for key, expected in checks.items():
            if result.get(key) != expected:
                raise ValueError(f"Official output row {index} {key} does not match X1")
        if result.get("abstained") is True:
            raise ValueError(f"Unexpected abstained result at official output row {index}")
        claims = result.get("all_claims")
        verifications = result.get("claim_verification_result")
        if not isinstance(claims, list):
            raise ValueError(f"Missing all_claims list at official output row {index}")
        if not isinstance(verifications, list):
            raise ValueError(f"Missing claim_verification_result at row {index}")
        labels: list[str] = []
        for item in verifications:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid verification item at row {index}")
            label = str(item.get("verification_result", "")).strip().lower().rstrip(".")
            if label not in allowed:
                raise ValueError(f"Unexpected VeriScore label {label!r} at row {index}")
            labels.append(label)
        claim_count = len(claims)
        supported = sum(label == "supported" for label in labels)
        if len(verifications) > claim_count:
            raise ValueError(f"More verification results than claims at row {index}")
        scored.append({
            "schema_version": "fcb_cove_veriscore_response_score_v1",
            "evaluation_unit_id": unit["evaluation_unit_id"],
            "response_id": unit["response_id"],
            "split": unit["split"],
            "condition_id": unit["condition_id"],
            "condition_display_name": unit["condition_display_name"],
            "question_sha256": unit["question_sha256"],
            "response_sha256": unit["response_sha256"],
            "verifiable_claim_count": claim_count,
            "verified_claim_count": len(verifications),
            "supported_claim_count": supported,
            "unsupported_claim_count": claim_count - supported,
            "factual_precision": supported / claim_count if claim_count else 0.0,
            "no_verifiable_claim_sentinel": (
                not claims or claims == ["No verifiable claim."]
            ),
            "official_label_counts": {
                label: labels.count(label) for label in sorted(set(labels))
            },
        })
    claim_counts = sorted(row["verifiable_claim_count"] for row in scored)
    shared_k = claim_counts[len(claim_counts) // 2]
    if shared_k <= 0:
        raise ValueError("Shared VeriScore K must be positive")
    for row in scored:
        recall = min(row["supported_claim_count"] / shared_k, 1.0)
        precision = row["factual_precision"]
        f1 = 0.0 if recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        row["shared_median_k"] = shared_k
        row["recall_at_shared_median_k"] = recall
        row["veriscore_f1_at_shared_median_k"] = f1
    return scored, shared_k


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, str]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    by_condition: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_condition[row["condition_id"]][row["response_id"]] = row
    response_ids = sorted(next(iter(by_condition.values())))
    if any(set(values) != set(response_ids) for values in by_condition.values()):
        raise ValueError("VeriScore response sets are not paired across conditions")
    metrics = [
        "veriscore_f1_at_shared_median_k",
        "factual_precision",
        "supported_claim_count",
        "verifiable_claim_count",
    ]
    samples = int(settings["bootstrap_samples"])
    confidence = float(settings["confidence_level"])
    rng = random.Random(int(settings["bootstrap_seed"]))
    output: dict[str, Any] = {}
    for comparison in comparisons:
        name = comparison["name"]
        treatment = comparison["treatment"]
        baseline = comparison["baseline"]
        metric_output: dict[str, Any] = {}
        for metric in metrics:
            paired_values = [
                float(by_condition[treatment][rid][metric])
                - float(by_condition[baseline][rid][metric])
                for rid in response_ids
            ]
            draws = []
            for _ in range(samples):
                selected = [rng.randrange(len(response_ids)) for _ in response_ids]
                draws.append(statistics.fmean(paired_values[index] for index in selected))
            alpha = 1.0 - confidence
            lower = _percentile(draws, alpha / 2.0)
            upper = _percentile(draws, 1.0 - alpha / 2.0)
            metric_output[metric] = {
                "point_estimate": statistics.fmean(paired_values),
                "lower": lower,
                "upper": upper,
                "includes_zero": lower <= 0.0 <= upper,
                "valid_replicates": len(draws),
            }
        output[name] = {
            "treatment": treatment,
            "baseline": baseline,
            "metrics": metric_output,
        }
    return {
        "method": "paired_response_percentile_bootstrap",
        "sampling_unit": "response_id",
        "response_count": len(response_ids),
        "samples": samples,
        "seed": int(settings["bootstrap_seed"]),
        "confidence_level": confidence,
        "comparisons": output,
    }


def _fmt(value: float, percent: bool = False) -> str:
    return f"{100.0 * value:.2f}%" if percent else f"{value:.3f}"


def analyze(split: str, results_path: Path, metadata_path: Path, dry_run: bool) -> int:
    cfg = config()
    out = paths(split)
    if not out["unit_manifest"].exists() or not out["official_input"].exists():
        raise FileNotFoundError("Run the X1 prepare stage before X2 analysis")
    metadata = _validate_metadata(metadata_path, split, cfg)
    units = load_jsonl(out["unit_manifest"])
    official_rows = load_jsonl(results_path)
    scored, shared_k = _score_official_rows(
        units, official_rows, int(metadata["label_n"])
    )
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[row["condition_id"]].append(row)
    condition_summary = {}
    for condition in cfg["conditions"]:
        condition_id = condition["condition_id"]
        rows = grouped[condition_id]
        condition_summary[condition_id] = {
            "display_name": condition["display_name"],
            "response_count": len(rows),
            "mean_verifiable_claim_count": _mean(rows, "verifiable_claim_count"),
            "mean_supported_claim_count": _mean(rows, "supported_claim_count"),
            "mean_factual_precision": _mean(rows, "factual_precision"),
            "mean_recall_at_shared_median_k": _mean(rows, "recall_at_shared_median_k"),
            "mean_veriscore_f1_at_shared_median_k": _mean(
                rows, "veriscore_f1_at_shared_median_k"
            ),
            "no_verifiable_claim_sentinel_count": sum(
                row["no_verifiable_claim_sentinel"] for row in rows
            ),
        }
    bootstrap = _paired_bootstrap(
        scored, cfg["analysis"]["paired_comparisons"], cfg["analysis"]
    )
    report = {
        "schema_version": "fcb_cove_external_veriscore_analysis_v1",
        "status": "complete",
        "evidence_strength": "EXTERNAL_AUTOMATIC_RESPONSE_LEVEL_SUPPLEMENT",
        "scope": "full",
        "split": split,
        "shared_median_k": shared_k,
        "response_count": len({row["response_id"] for row in scored}),
        "evaluation_unit_count": len(scored),
        "condition_summary": condition_summary,
        "paired_response_bootstrap": bootstrap,
        "external_run_metadata": metadata,
        "source_fingerprints": {
            relative(CONFIG_PATH): sha256_file(CONFIG_PATH),
            relative(out["official_input"]): sha256_file(out["official_input"]),
            relative(out["unit_manifest"]): sha256_file(out["unit_manifest"]),
            relative(results_path): sha256_file(results_path),
            relative(metadata_path): sha256_file(metadata_path),
        },
        "interpretation_boundary": (
            "VeriScore independently re-extracts verifiable claims and checks them "
            "against Serper search results. It is an external automatic whole-response "
            "sensitivity metric, not a human-gold claim transition label."
        ),
        "generated_at": utc_now(),
    }
    lines = [
        f"# X2 — External VeriScore Analysis ({split})",
        "",
        "- Evidence strength: `EXTERNAL_AUTOMATIC_RESPONSE_LEVEL_SUPPLEMENT`",
        f"- Paired responses per condition: {report['response_count']}",
        f"- Shared median K: {shared_k}",
        f"- Official package version: `{metadata['package_version']}`",
        f"- Extraction model: `{metadata['extraction_model']}`",
        f"- Verification model: `{metadata['verification_model']}`",
        "",
        "## Condition results",
        "",
        "| Condition | Verifiable claims | Supported claims | Precision | Recall@K | F1@K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in cfg["conditions"]:
        item = condition_summary[condition["condition_id"]]
        lines.append(
            f"| {item['display_name']} | {item['mean_verifiable_claim_count']:.2f} | "
            f"{item['mean_supported_claim_count']:.2f} | "
            f"{_fmt(item['mean_factual_precision'], True)} | "
            f"{_fmt(item['mean_recall_at_shared_median_k'], True)} | "
            f"{_fmt(item['mean_veriscore_f1_at_shared_median_k'], True)} |"
        )
    lines.extend([
        "",
        "## Paired differences",
        "",
        "| Contrast | F1@K difference (95% CI) | Precision difference (95% CI) |",
        "|---|---:|---:|",
    ])
    for comparison in cfg["analysis"]["paired_comparisons"]:
        item = bootstrap["comparisons"][comparison["name"]]["metrics"]
        f1 = item["veriscore_f1_at_shared_median_k"]
        precision = item["factual_precision"]
        lines.append(
            f"| `{comparison['name']}` | {100*f1['point_estimate']:+.2f} pp "
            f"({100*f1['lower']:+.2f}, {100*f1['upper']:+.2f}) | "
            f"{100*precision['point_estimate']:+.2f} pp "
            f"({100*precision['lower']:+.2f}, {100*precision['upper']:+.2f}) |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        report["interpretation_boundary"],
        "The result must remain separate from V10's Human-anchored, Cross-model-",
        "supported, and Silver full-coverage layers. Unsupported in VeriScore means "
        "not supported by the retrieved search snippets; it does not necessarily mean "
        "the claim is false.",
        "",
    ])
    if dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    atomic_write_jsonl(out["response_scores"], scored)
    atomic_write_json(out["analysis_json"], report)
    atomic_write_text(out["analysis_md"], "\n".join(lines))
    print(json.dumps({
        "stage": "X2_analyze_veriscore",
        "status": "complete",
        "split": split,
        "shared_median_k": shared_k,
        "report": relative(out["analysis_md"]),
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or analyse an isolated official VeriScore run."
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--scope", choices=("full",), default="full")
    parser.add_argument("--split", choices=("dev", "heldout"), default="heldout")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "analyze":
        if args.results is None or args.run_metadata is None:
            parser.error("analyze requires --results and --run-metadata")
    elif args.results is not None or args.run_metadata is not None:
        parser.error("--results and --run-metadata are valid only for analyze")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "prepare":
        return prepare(args.split, args.dry_run)
    if args.stage == "analyze":
        return analyze(
            args.split,
            args.results.resolve(),
            args.run_metadata.resolve(),
            args.dry_run,
        )
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
