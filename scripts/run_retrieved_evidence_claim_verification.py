#!/usr/bin/env python3
"""Run the frozen Hybrid retrieved-evidence verifier on held-out claims.

The model sees only a claim and passage text. Gold labels, qrels, evidence
stances, and URL mappings are joined only after inference for evaluation.
The formal top-5 run remains the default; top-1 and top-3 are isolated held-out
sensitivity conditions with separate outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factcheck_bench_analysis import (  # noqa: E402
    build_paired_transitions,
    build_response_aggregation,
    compute_binary_metrics,
    metric_differences,
    paired_response_cluster_bootstrap,
)


def _import_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _import_script(
    "fcb_no_evidence_verification",
    PROJECT_ROOT / "scripts" / "run_no_evidence_claim_verification.py",
)

SETTING = "retrieved_evidence"
RESULT_SCHEMA_VERSION = "retrieved_evidence_result_v1"
REPORT_SCHEMA_VERSION = "retrieved_evidence_report_v2"
FORMAT_REPAIR_PROMPT_VERSION = "retrieved_evidence_format_repair_v1"
EXPECTED_CLAIMS = 468
EXPECTED_RESPONSES = 72
PRIMARY_TOP_K = 5
SUPPORTED_TOP_K = (1, 3, PRIMARY_TOP_K)
TOP_K_PHRASES = {
    1: "one passage",
    3: "three passages",
    PRIMARY_TOP_K: "five passages",
}
PROMPT_VERSIONS = {
    1: "retrieved_evidence_hybrid_top1_heldout_sensitivity_v1",
    3: "retrieved_evidence_hybrid_top3_heldout_sensitivity_v1",
    PRIMARY_TOP_K: "retrieved_evidence_hybrid_top5_v1",
}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_722
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

GOLD = PROJECT_ROOT / "data/factcheck_bench/processed/fcb_gold_claims_full.jsonl"
SPLITS = PROJECT_ROOT / "data/factcheck_bench/retrieval/manifests/retrieval_split_manifest.jsonl"
PASSAGES = PROJECT_ROOT / "data/factcheck_bench/retrieval/passages.jsonl"
HYBRID_RUN = PROJECT_ROOT / "data/factcheck_bench/retrieval/evaluation/runs/hybrid_rrf_heldout.jsonl"
RETRIEVAL_COMPARISON = PROJECT_ROOT / "data/factcheck_bench/retrieval/evaluation/reports/two_level_heldout_comparison.json"
SOURCE_QRELS = PROJECT_ROOT / "data/factcheck_bench/retrieval/evaluation/qrels_source_heldout.jsonl"
STRICT_QRELS = PROJECT_ROOT / "data/factcheck_bench/retrieval/evaluation/qrels_strict_passage_heldout.jsonl"
NO_RESULTS = PROJECT_ROOT / "outputs/factcheck_bench_full/jsonl/08b_no_evidence_verifier_results.jsonl"
ORACLE_RESULTS = PROJECT_ROOT / "outputs/factcheck_bench_full/jsonl/08c_oracle_evidence_verifier_results.jsonl"
PROMPT = PROJECT_ROOT / "prompts/retrieved_evidence_verifier.txt"
FORMAT_REPAIR_PROMPT = PROJECT_ROOT / "prompts/retrieved_evidence_output_repair.txt"
PRIMARY_OUTPUT = PROJECT_ROOT / "outputs/factcheck_bench_full/jsonl/12_retrieved_evidence_verifier_heldout_results.jsonl"
PRIMARY_JSON_REPORT = PROJECT_ROOT / "outputs/factcheck_bench_full/reports/12_retrieved_evidence_verifier_heldout_summary.json"
PRIMARY_MD_REPORT = PROJECT_ROOT / "outputs/factcheck_bench_full/reports/12_retrieved_evidence_verifier_heldout_report.md"
SENSITIVITY_OUTPUT_ROOTS = {
    1: PROJECT_ROOT / "outputs/factcheck_bench_full/retrieved_evidence_k1_heldout",
    3: PROJECT_ROOT / "outputs/factcheck_bench_full/retrieved_evidence_k3_heldout",
}


def output_paths(top_k: int) -> tuple[Path, Path, Path]:
    if top_k == PRIMARY_TOP_K:
        return PRIMARY_OUTPUT, PRIMARY_JSON_REPORT, PRIMARY_MD_REPORT
    if top_k in SENSITIVITY_OUTPUT_ROOTS:
        output_root = SENSITIVITY_OUTPUT_ROOTS[top_k]
        return (
            output_root
            / f"jsonl/12_retrieved_evidence_verifier_heldout_k{top_k:02d}_results.jsonl",
            output_root
            / f"reports/12_retrieved_evidence_verifier_heldout_k{top_k:02d}_summary.json",
            output_root
            / f"reports/12_retrieved_evidence_verifier_heldout_k{top_k:02d}_report.md",
        )
    raise ValueError(f"Unsupported retrieved-evidence top-k: {top_k}")


def effective_prompt_template(template: str, top_k: int) -> str:
    if template.count("five passages") != 1:
        raise ValueError(
            "Canonical retrieved-evidence prompt must contain exactly one "
            "'five passages' declaration"
        )
    try:
        phrase = TOP_K_PHRASES[top_k]
    except KeyError as error:
        raise ValueError(f"Unsupported retrieved-evidence top-k: {top_k}") from error
    return template.replace("five passages", phrase)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def unique_by(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} contains invalid {key}: {value!r}")
        if value in output:
            raise ValueError(f"{label} contains duplicate {key}: {value}")
        output[value] = row
    return output


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_result(sink: Any, row: dict[str, Any]) -> None:
    sink.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    sink.flush()
    os.fsync(sink.fileno())


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    gold_by_id = unique_by(read_jsonl(GOLD), "claim_id", "gold claims")
    split_rows = read_jsonl(SPLITS)
    heldout_ids = {
        str(row["claim_id"])
        for row in split_rows
        if row.get("split") == "heldout" and row.get("in_primary_matched_cohort") is True
    }
    if len(heldout_ids) != EXPECTED_CLAIMS:
        raise ValueError(f"Expected {EXPECTED_CLAIMS} held-out matched claims, found {len(heldout_ids)}")
    records = [gold_by_id[claim_id] for claim_id in sorted(heldout_ids)]
    if {row.get("human_label") for row in records} - {"FACTUAL", "NON_FACTUAL"}:
        raise ValueError("Held-out cohort contains non-binary human labels")
    response_ids = {str(row["response_id"]) for row in records}
    if len(response_ids) != EXPECTED_RESPONSES:
        raise ValueError(f"Expected {EXPECTED_RESPONSES} held-out responses, found {len(response_ids)}")

    comparison = json.loads(RETRIEVAL_COMPARISON.read_text(encoding="utf-8"))
    selection = comparison.get("selection", {})
    selected = selection.get("selected_configuration", {})
    if (
        comparison.get("split") != "heldout"
        or selection.get("configuration_is_frozen") is not True
        or selection.get("decision") != "frozen_hybrid_rrf_evaluated_without_reselection"
        or selected.get("retriever") != "hybrid_rrf"
        or selected.get("evidence_top_k") != PRIMARY_TOP_K
    ):
        raise ValueError("Held-out retrieval comparison does not confirm the frozen Hybrid top-5 configuration")
    return records, gold_by_id, comparison


def build_retrieved_evidence(
    records: list[dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, set[str]]]:
    record_ids = {str(row["claim_id"]) for row in records}
    passages = unique_by(read_jsonl(PASSAGES), "passage_id", "passages")
    ranked: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(HYBRID_RUN):
        query_id = str(row.get("query_id"))
        if query_id in record_ids and int(row.get("rank", 0)) <= top_k:
            ranked[query_id].append(row)
    if set(ranked) != record_ids:
        missing = sorted(record_ids - set(ranked))
        raise ValueError(f"Hybrid run lacks top-{top_k} rows for {len(missing)} held-out claims: {missing[:5]}")

    bundles: dict[str, dict[str, Any]] = {}
    retrieved_passages: dict[str, set[str]] = {}
    retrieved_docs: dict[str, set[str]] = {}
    for record in records:
        claim_id = str(record["claim_id"])
        run_rows = sorted(ranked[claim_id], key=lambda row: int(row["rank"]))
        if [int(row["rank"]) for row in run_rows] != list(range(1, top_k + 1)):
            raise ValueError(f"Hybrid top-{top_k} ranks are not contiguous for {claim_id}")
        items: list[dict[str, Any]] = []
        visible: list[str] = []
        for run_row in run_rows:
            passage_id = str(run_row["passage_id"])
            passage = passages.get(passage_id)
            if passage is None:
                raise ValueError(f"Hybrid run references unknown passage: {passage_id}")
            text = str(passage.get("text", "")).strip()
            if not text:
                raise ValueError(f"Retrieved passage has empty text: {passage_id}")
            rank = int(run_row["rank"])
            visible.append(f"Passage {rank} text (JSON-encoded): {json.dumps(text, ensure_ascii=False)}")
            items.append(
                {
                    "rank": rank,
                    "passage_id": passage_id,
                    "doc_id": str(run_row["doc_id"]),
                    "retrieval_score": float(run_row["score"]),
                    "text": text,
                    "text_sha256": _base.sha256_text(text),
                }
            )
        normalized_text = "\n\n".join(visible)
        bundles[claim_id] = {
            "retriever": "hybrid_rrf",
            "split": "heldout",
            "top_k": top_k,
            "items": items,
            "normalized_text": normalized_text,
            "normalized_sha256": _base.sha256_text(normalized_text),
            "model_visible_fields": ["text"],
            "evaluation_only_fields_omitted_from_prompt": [
                "human_label", "gold_evidence_text", "gold_evidence_stance", "gold_url_mapping", "qrels"
            ],
        }
        retrieved_passages[claim_id] = {item["passage_id"] for item in items}
        retrieved_docs[claim_id] = {item["doc_id"] for item in items}
    return bundles, retrieved_passages, retrieved_docs


def load_historical(path: Path, records: list[dict[str, Any]], setting: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    needed = {str(row["claim_id"]) for row in records}
    all_rows = read_jsonl(path)
    selected = {str(row["claim_id"]): row for row in all_rows if str(row.get("claim_id")) in needed}
    if set(selected) != needed:
        raise ValueError(f"{setting} results are incomplete for the held-out cohort")
    profiles = {}
    for field in ("model", "model_digest", "temperature", "seed", "num_predict", "think"):
        values = {row.get(field) for row in selected.values()}
        if len(values) != 1:
            raise ValueError(f"{setting} has inconsistent {field}: {values}")
        profiles[field] = next(iter(values))
    if any(row.get("setting") != setting for row in selected.values()):
        raise ValueError(f"Unexpected setting in {path}")
    return selected, profiles


def qrel_sets(path: Path, key: str) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(path):
        output[str(row["query_id"])].add(str(row[key]))
    return output


def build_prompt(template: str, record: dict[str, Any], bundle: dict[str, Any]) -> str:
    return template.replace(
        "{gold_claim_json}", json.dumps(str(record["gold_claim"]).strip(), ensure_ascii=False)
    ).replace("{retrieved_evidence_text}", str(bundle["normalized_text"]))


def _format_repair_candidate(raw: str) -> dict[str, Any]:
    """Validate that only rationale length prevents normal strict parsing."""
    try:
        _base.parse_model_output(raw)
    except ValueError as error:
        if str(error) not in {
            "rationale cannot exceed 240 characters.",
            "rationale cannot exceed 35 words.",
        }:
            raise
    else:
        raise ValueError("Output does not require format repair")
    parsed = json.loads(
        raw.strip(),
        object_pairs_hook=_base.reject_duplicate_keys,
        parse_constant=_base.reject_json_constant,
    )
    if not isinstance(parsed, dict) or set(parsed) != {
        "prediction",
        "confidence",
        "rationale",
    }:
        raise ValueError("Format repair requires the exact output object schema")
    # A short placeholder lets the canonical parser validate every field other
    # than the over-length rationale before any repair call is allowed.
    validation_copy = dict(parsed)
    validation_copy["rationale"] = "Format repair required."
    validated = _base.parse_model_output(
        json.dumps(validation_copy, ensure_ascii=False, allow_nan=False)
    )
    return {
        "prediction": validated["prediction"],
        "confidence": validated["confidence"],
        "rationale": str(parsed["rationale"]),
    }


def repair_output_format(
    raw: str,
    config: dict[str, Any],
    client: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Shorten only an over-length rationale while freezing the decision."""
    original = _format_repair_candidate(raw)
    template = FORMAT_REPAIR_PROMPT.read_text(encoding="utf-8")
    prompt = (
        template.replace(
            "{prediction_json}",
            json.dumps(original["prediction"], ensure_ascii=False),
        )
        .replace(
            "{confidence_json}",
            json.dumps(original["confidence"], ensure_ascii=False),
        )
        .replace("{rationale_json}", json.dumps(original["rationale"], ensure_ascii=False))
    )
    repaired_raw, ollama_metadata = _base.call_ollama(client, config, prompt)
    repaired = _base.parse_model_output(repaired_raw)
    if repaired["prediction"] != original["prediction"]:
        raise ValueError("Format repair changed prediction")
    if repaired["confidence"] != original["confidence"]:
        raise ValueError("Format repair changed confidence")
    metadata = {
        "method": "model_format_only_rationale_shortening",
        "prompt_version": FORMAT_REPAIR_PROMPT_VERSION,
        "prompt_sha256": _base.sha256_file(FORMAT_REPAIR_PROMPT),
        "prediction_preserved": True,
        "confidence_preserved": True,
        "original_rationale_word_count": len(original["rationale"].split()),
        "original_rationale_character_count": len(original["rationale"]),
        "repaired_rationale_word_count": len(repaired["rationale"].split()),
        "repaired_rationale_character_count": len(repaired["rationale"]),
        "repair_raw_model_output": repaired_raw,
        "repair_ollama_metadata": ollama_metadata,
        "repaired_at": utc_now(),
    }
    return repaired, metadata


def run_config(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    bundles: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    digest: str,
    top_k: int,
    effective_template: str,
) -> dict[str, Any]:
    manifest = [
        {
            "claim_id": row["claim_id"],
            "response_id": row["response_id"],
            "claim_sha256": _base.sha256_text(str(row["gold_claim"])),
            "retrieved_evidence_sha256": bundles[str(row["claim_id"])]["normalized_sha256"],
        }
        for row in records
    ]
    payload = {
        "setting": SETTING,
        "split": "heldout",
        "retriever": "hybrid_rrf",
        "top_k": top_k,
        "model": profile["model"],
        "model_digest": digest,
        "temperature": profile["temperature"],
        "seed": profile["seed"],
        "num_predict": profile["num_predict"],
        "think": False,
        "timeout_seconds": args.timeout,
        "max_retries": args.max_retries,
        "prompt_version": PROMPT_VERSIONS[top_k],
        "prompt_sha256": _base.sha256_text(effective_template),
        "output_schema_sha256": _base.canonical_json_hash(_base.OUTPUT_SCHEMA),
        "cohort_sha256": _base.canonical_json_hash(manifest),
        "gold_input_sha256": _base.sha256_file(GOLD),
        "split_manifest_sha256": _base.sha256_file(SPLITS),
        "passages_sha256": _base.sha256_file(PASSAGES),
        "hybrid_run_sha256": _base.sha256_file(HYBRID_RUN),
        "retrieval_comparison_sha256": _base.sha256_file(RETRIEVAL_COMPARISON),
    }
    if top_k != PRIMARY_TOP_K:
        payload["analysis_role"] = "secondary_heldout_retrieval_depth_sensitivity"
        payload["formal_primary_top_k"] = PRIMARY_TOP_K
    payload["run_fingerprint"] = _base.canonical_json_hash(payload)
    return payload


def result_base(record: dict[str, Any], bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "claim_id": record["claim_id"],
        "response_id": record["response_id"],
        "gold_claim": record["gold_claim"],
        "human_label": record["human_label"],
        "claim_sha256": _base.sha256_text(str(record["gold_claim"])),
        "retrieved_evidence": bundle,
        "setting": SETTING,
        "model_input_fields": ["gold_claim", "retrieved_passage_text"],
        **config,
    }


def process(record: dict[str, Any], bundle: dict[str, Any], template: str, config: dict[str, Any], client: Any) -> dict[str, Any]:
    output = result_base(record, bundle, config)
    started = time.perf_counter()
    raw: str | None = None
    metadata: dict[str, Any] = {}
    error: Exception | None = None
    attempts = 0
    for attempt in range(config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw, metadata = _base.call_ollama(client, config, build_prompt(template, record, bundle))
            error = None
            break
        except Exception as exc:
            error = exc
            if attempt < config["max_retries"]:
                time.sleep(1.0)
    output.update(
        {
            "attempts": attempts,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "raw_model_output": raw,
            "ollama_metadata": metadata,
            "created_at": utc_now(),
        }
    )
    if error is not None:
        output.update(status="request_error", prediction=None, confidence=None, rationale=None, error=f"{type(error).__name__}: {error}")
        return output
    format_repair = None
    try:
        parsed = _base.parse_model_output(raw or "")
    except Exception as exc:
        try:
            parsed, format_repair = repair_output_format(raw or "", config, client)
        except Exception:
            output.update(status="parse_error", prediction=None, confidence=None, rationale=None, error=f"{type(exc).__name__}: {exc}")
            return output
    output.update(
        status="ok",
        prediction=parsed["prediction"],
        confidence=parsed["confidence"],
        rationale=parsed["rationale"],
        error=None,
        format_repair=format_repair,
        latency_seconds=round(time.perf_counter() - started, 4),
    )
    return output


def repair_existing_parse_error(
    row: dict[str, Any],
    config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    if row.get("status") != "parse_error":
        return row
    raw = row.get("raw_model_output")
    if not isinstance(raw, str) or not raw.strip():
        return row
    try:
        parsed, repair = repair_output_format(raw, config, client)
    except Exception:
        return row
    repaired = dict(row)
    repaired.update(
        status="ok",
        prediction=parsed["prediction"],
        confidence=parsed["confidence"],
        rationale=parsed["rationale"],
        error=None,
        format_repair=repair,
    )
    return repaired


def evidence_diagnostics(
    records: list[dict[str, Any]],
    retrieved: dict[str, dict[str, Any]],
    retrieved_passages: dict[str, set[str]],
    retrieved_docs: dict[str, set[str]],
    top_k: int,
) -> dict[str, Any]:
    source = qrel_sets(SOURCE_QRELS, "doc_id")
    strict = qrel_sets(STRICT_QRELS, "passage_id")
    buckets: dict[str, dict[str, int]] = {
        f"source_document_hit_at_{top_k}": {"claims": 0, "correct": 0},
        f"source_document_miss_or_unavailable_at_{top_k}": {"claims": 0, "correct": 0},
        f"strict_passage_hit_at_{top_k}": {"claims": 0, "correct": 0},
        f"strict_passage_miss_or_unavailable_at_{top_k}": {"claims": 0, "correct": 0},
    }
    for record in records:
        claim_id = str(record["claim_id"])
        result = retrieved.get(claim_id)
        is_correct = (
            result is not None
            and result.get("status") == "ok"
            and result.get("prediction") == record["human_label"]
        )
        source_hit = bool(source.get(claim_id, set()) & retrieved_docs[claim_id])
        strict_hit = bool(strict.get(claim_id, set()) & retrieved_passages[claim_id])
        for name, hit in (("source_document", source_hit), ("strict_passage", strict_hit)):
            key = f"{name}_{'hit' if hit else 'miss_or_unavailable'}_at_{top_k}"
            buckets[key]["claims"] += 1
            buckets[key]["correct"] += int(is_correct)
    for values in buckets.values():
        values["accuracy"] = values["correct"] / values["claims"] if values["claims"] else None
    return buckets


def _gold_stance_bucket(record: dict[str, Any]) -> str:
    stances = {
        str(item.get("stance", "")).strip().lower()
        for item in record.get("gold_evidence", [])
        if isinstance(item, dict) and str(item.get("stance", "")).strip()
    }
    has_refute = "refute" in stances
    has_partial = "partially-support" in stances
    has_support = bool(stances & {"completely-support", "human-validated"})
    if has_refute and (has_support or has_partial):
        return "mixed_with_refute"
    if has_refute:
        return "refute_only"
    if has_partial:
        return "partial_support_without_refute"
    if has_support:
        return "support_or_human_validated_only"
    return "other_or_missing"


def _concise_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    prediction_counts = {
        prediction: sum(
            int(metrics["confusion_matrix"][label][prediction])
            for label in ("FACTUAL", "NON_FACTUAL")
        )
        for prediction in (
            "FACTUAL",
            "NON_FACTUAL",
            "UNKNOWN",
            "ERROR_OR_MISSING",
        )
    }
    return {
        "correct_count": metrics["correct_count"],
        "accuracy": metrics["accuracy_including_abstentions_and_errors"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "coverage": metrics["coverage"],
        "selective_accuracy": metrics["selective_accuracy"],
        "factual_recall": metrics["FACTUAL"]["recall"],
        "non_factual_recall": metrics["NON_FACTUAL"]["recall"],
        "prediction_counts": prediction_counts,
    }


def _stratum_report(
    records: list[dict[str, Any]],
    result_sets: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "claim_count": len(records),
        "response_count": len({str(record["response_id"]) for record in records}),
        "gold_label_counts": dict(
            Counter(str(record["human_label"]) for record in records)
        ),
        "settings": {
            setting: _concise_metrics(compute_binary_metrics(records, results))
            for setting, results in result_sets.items()
        },
    }


def build_stratified_diagnostics(
    records: list[dict[str, Any]],
    result_sets: dict[str, dict[str, dict[str, Any]]],
    retrieved_passages: dict[str, set[str]],
    retrieved_docs: dict[str, set[str]],
    top_k: int,
) -> dict[str, Any]:
    source = qrel_sets(SOURCE_QRELS, "doc_id")
    strict = qrel_sets(STRICT_QRELS, "passage_id")

    def source_bucket(record: dict[str, Any]) -> str:
        claim_id = str(record["claim_id"])
        if claim_id not in source:
            return "no_source_qrel_or_unavailable"
        if source[claim_id] & retrieved_docs[claim_id]:
            return f"source_document_hit_at_{top_k}"
        return f"source_document_eligible_miss_at_{top_k}"

    def strict_bucket(record: dict[str, Any]) -> str:
        claim_id = str(record["claim_id"])
        if claim_id not in strict:
            return "no_strict_passage_qrel_or_unavailable"
        if strict[claim_id] & retrieved_passages[claim_id]:
            return f"strict_passage_hit_at_{top_k}"
        return f"strict_passage_eligible_miss_at_{top_k}"

    assignments = {
        "gold_label": lambda record: str(record["human_label"]).lower(),
        "gold_evidence_stance": _gold_stance_bucket,
        "source_document_retrieval": source_bucket,
        "strict_passage_retrieval": strict_bucket,
    }
    groups = {}
    for group_name, assign in assignments.items():
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            buckets[assign(record)].append(record)
        if sum(len(bucket) for bucket in buckets.values()) != len(records):
            raise AssertionError(f"{group_name} strata do not partition the cohort")
        groups[group_name] = {
            name: _stratum_report(bucket, result_sets)
            for name, bucket in sorted(buckets.items())
        }
    return {
        "status": "complete",
        "cohort_claim_count": len(records),
        "field_usage": (
            "Gold labels, evidence stances, and qrels are evaluation-only and "
            "were never exposed to retrieval ranking or verifier prompts."
        ),
        "definitions": {
            f"source_document_hit_at_{top_k}": (
                f"At least one gold source-qrel document contributed a retrieved top-{top_k} passage."
            ),
            f"source_document_eligible_miss_at_{top_k}": (
                f"A source qrel exists, but no relevant document appears in top-{top_k}."
            ),
            f"strict_passage_hit_at_{top_k}": (
                f"At least one strict gold-aligned qrel passage appears in top-{top_k}."
            ),
            f"strict_passage_eligible_miss_at_{top_k}": (
                f"A strict passage qrel exists, but no such passage appears in top-{top_k}."
            ),
            "stance_groups": (
                "Exclusive claim-level buckets derived from benchmark evidence stances; "
                "human-validated is grouped with support, while partial support and mixed "
                "refute bundles remain separate."
            ),
        },
        "groups": groups,
        "limitations": [
            "Qrels are incomplete and a miss can still contain unjudged valid evidence.",
            "Stratified results are descriptive and some buckets are small.",
            "Differences between hit and miss strata are associations, not causal effects.",
        ],
    }


def build_report(
    records: list[dict[str, Any]],
    retrieved_rows: list[dict[str, Any]],
    no_rows: dict[str, dict[str, Any]],
    oracle_rows: dict[str, dict[str, Any]],
    config: dict[str, Any],
    retrieved_passages: dict[str, set[str]],
    retrieved_docs: dict[str, set[str]],
    top_k: int,
    output: Path,
) -> dict[str, Any]:
    retrieved = {str(row["claim_id"]): row for row in retrieved_rows}
    result_sets = {
        "no_evidence": no_rows,
        "oracle_evidence": oracle_rows,
        "retrieved_evidence": retrieved,
    }
    metrics = {
        setting: compute_binary_metrics(records, results)
        for setting, results in result_sets.items()
    }
    response_rows, response_macro = build_response_aggregation(records, retrieved)
    benefit_recovery: dict[str, float | None] = {}
    for metric_name, key in (("accuracy", "accuracy_including_abstentions_and_errors"), ("balanced_accuracy", "balanced_accuracy"), ("macro_f1", "macro_f1")):
        no_value = metrics["no_evidence"].get(key)
        oracle_value = metrics["oracle_evidence"].get(key)
        retrieved_value = metrics["retrieved_evidence"].get(key)
        denominator = None if no_value is None or oracle_value is None else oracle_value - no_value
        benefit_recovery[metric_name] = None if denominator in (None, 0) or retrieved_value is None else (retrieved_value - no_value) / denominator
    bootstrap = paired_response_cluster_bootstrap(
        records,
        result_sets,
        (
            ("oracle_minus_no_evidence", "oracle_evidence", "no_evidence"),
            ("retrieved_minus_no_evidence", "retrieved_evidence", "no_evidence"),
            ("retrieved_minus_oracle", "retrieved_evidence", "oracle_evidence"),
        ),
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
        confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
    )
    repairs = [
        {
            "claim_id": row["claim_id"],
            "response_id": row["response_id"],
            **row["format_repair"],
        }
        for row in retrieved_rows
        if isinstance(row.get("format_repair"), dict)
    ]
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "complete" if len(retrieved) == len(records) and all(row.get("status") == "ok" for row in retrieved.values()) else "incomplete",
        "cohort": {"split": "heldout", "claim_count": len(records), "response_count": len({row["response_id"] for row in records}), "gold_label_counts": dict(Counter(str(row["human_label"]) for row in records))},
        "run_config": config,
        "metrics": metrics,
        "metric_differences": {
            "retrieved_minus_no_evidence": metric_differences(metrics["no_evidence"], metrics["retrieved_evidence"], "retrieved_minus_no_evidence"),
            "retrieved_minus_oracle": metric_differences(metrics["oracle_evidence"], metrics["retrieved_evidence"], "retrieved_minus_oracle"),
        },
        "oracle_benefit_recovery_ratio": benefit_recovery,
        "paired_transitions": {
            "no_evidence_to_retrieved": build_paired_transitions(records, no_rows, retrieved),
            "oracle_to_retrieved": build_paired_transitions(records, oracle_rows, retrieved),
        },
        "paired_response_cluster_bootstrap": bootstrap,
        "retrieval_conditioned_verifier_accuracy": evidence_diagnostics(
            records, retrieved, retrieved_passages, retrieved_docs, top_k
        ),
        "stratified_diagnostics": build_stratified_diagnostics(
            records,
            result_sets,
            retrieved_passages,
            retrieved_docs,
            top_k,
        ),
        "retrieved_response_macro": response_macro,
        "retrieved_per_response": response_rows,
        "technical_completion": {
            "technical_failure_count": metrics["retrieved_evidence"]["technical_failure_count"],
            "format_repair_count": len(repairs),
            "format_repairs": repairs,
            "repair_policy": (
                "Only a strict-JSON result whose sole validation failure is rationale "
                "length may enter the format-only repair. Prediction and confidence "
                "must remain exactly unchanged."
            ),
        },
        "artifacts": {
            "results": str(output.relative_to(PROJECT_ROOT)),
            "hybrid_run": str(HYBRID_RUN.relative_to(PROJECT_ROOT)),
            "passages": str(PASSAGES.relative_to(PROJECT_ROOT)),
            "no_evidence_results": str(NO_RESULTS.relative_to(PROJECT_ROOT)),
            "oracle_results": str(ORACLE_RESULTS.relative_to(PROJECT_ROOT)),
        },
        "interpretation": [
            (
                "Hybrid top-5 was selected on the 121-claim development set and applied "
                "unchanged as the formal held-out configuration."
                if top_k == PRIMARY_TOP_K
                else f"This isolated top-{top_k} held-out run is a secondary "
                "retrieval-depth sensitivity check; the formal top-5 configuration "
                "remains unchanged."
            ),
            "Gold labels, qrels, evidence stances, and URL mappings were excluded from model input and joined only for evaluation.",
            "Benchmark-associated evidence is not guaranteed complete; benefit recovery is therefore diagnostic, not a causal decomposition.",
            "Paired percentile confidence intervals resample response_id clusters and keep all claims from a response together.",
            "Retrieval-hit and evidence-stance strata are descriptive associations and use evaluation-only metadata.",
        ],
    }


def pct(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):.2f}%"


def ci(value: dict[str, Any]) -> str:
    return f"[{pct(value['lower'])}, {pct(value['upper'])}]"


def markdown(report: dict[str, Any]) -> str:
    top_k = int(report["run_config"]["top_k"])
    if top_k != PRIMARY_TOP_K:
        row = report["metrics"]["retrieved_evidence"]
        lines = [
            f"# Held-out Retrieved Evidence sensitivity at K={top_k}",
            "",
            f"Status: **{report['status']}**. Cohort: {report['cohort']['claim_count']} claims "
            f"across {report['cohort']['response_count']} responses.",
            "",
            "This is an isolated secondary sensitivity check. The formal Study I "
            "Retrieved Evidence configuration remains Hybrid RRF top-5.",
            "",
            "## Retrieved Evidence results",
            "",
            "| Balanced Accuracy | Macro-F1 | Accuracy |",
            "|---:|---:|---:|",
            f"| {pct(row['balanced_accuracy'])} | {pct(row['macro_f1'])} | "
            f"{pct(row['accuracy_including_abstentions_and_errors'])} |",
            "",
            f"Technical failures: **{report['technical_completion']['technical_failure_count']}**. "
            f"Format-only repairs: **{report['technical_completion']['format_repair_count']}**.",
            "",
            "The same 468 held-out claims, frozen Hybrid RRF ranking, verifier model, "
            "prompt protocol and decoding configuration were retained; only the retrieved "
            f"evidence bundle was truncated to the top-{top_k} passages.",
            "",
        ]
        return "\n".join(lines)
    metrics = report["metrics"]
    bootstrap = report["paired_response_cluster_bootstrap"]
    lines = [
        "# Held-out retrieved-evidence verifier evaluation",
        "",
        f"Status: **{report['status']}**. Cohort: {report['cohort']['claim_count']} claims across {report['cohort']['response_count']} responses.",
        f"Technical failures: **{report['technical_completion']['technical_failure_count']}**. "
        f"Format-only repairs: **{report['technical_completion']['format_repair_count']}**.",
        "",
        "## Three-setting comparison",
        "",
        "| Setting | Accuracy | Balanced accuracy | Macro-F1 | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("no_evidence", "No evidence"), ("oracle_evidence", "Benchmark-associated evidence"), ("retrieved_evidence", "Hybrid retrieved top-5")):
        row = metrics[key]
        lines.append(f"| {label} | {pct(row['accuracy_including_abstentions_and_errors'])} | {pct(row['balanced_accuracy'])} | {pct(row['macro_f1'])} | {pct(row['coverage'])} |")
    lines.extend(
        [
            "",
            "## Paired response-cluster bootstrap",
            "",
            f"Percentile 95% intervals use {bootstrap['samples']:,} paired resamples "
            f"of all {bootstrap['response_cluster_count']} response clusters "
            f"(seed `{bootstrap['seed']}`). All claims from a sampled response remain together.",
            "",
            "| Setting | Accuracy 95% CI | Balanced accuracy 95% CI | Macro-F1 95% CI | Coverage 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, label in (("no_evidence", "No evidence"), ("oracle_evidence", "Benchmark-associated evidence"), ("retrieved_evidence", "Hybrid retrieved top-5")):
        intervals = bootstrap["setting_intervals"][key]
        lines.append(
            f"| {label} | {ci(intervals['accuracy'])} | "
            f"{ci(intervals['balanced_accuracy'])} | {ci(intervals['macro_f1'])} | "
            f"{ci(intervals['coverage'])} |"
        )
    lines.extend(
        [
            "",
            "| Paired difference | Accuracy delta (95% CI) | Balanced accuracy delta (95% CI) | Macro-F1 delta (95% CI) |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("oracle_minus_no_evidence", "Benchmark-associated - no evidence"),
        ("retrieved_minus_no_evidence", "Retrieved - no evidence"),
        ("retrieved_minus_oracle", "Retrieved - benchmark-associated"),
    ):
        intervals = bootstrap["paired_difference_intervals"][key]["metrics"]
        lines.append(
            f"| {label} | {pct(intervals['accuracy']['point_estimate'])} "
            f"{ci(intervals['accuracy'])} | "
            f"{pct(intervals['balanced_accuracy']['point_estimate'])} "
            f"{ci(intervals['balanced_accuracy'])} | "
            f"{pct(intervals['macro_f1']['point_estimate'])} "
            f"{ci(intervals['macro_f1'])} |"
        )
    lines.extend(["", "## Paired changes", ""])
    for key, label in (("no_evidence_to_retrieved", "No evidence → retrieved"), ("oracle_to_retrieved", "Benchmark-associated → retrieved")):
        named = report["paired_transitions"][key]["named_transitions"]
        lines.append(f"- {label}: wrong→correct {named['wrong_to_correct']['count']}; correct→wrong {named['correct_to_wrong']['count']}; wrong→wrong {named['wrong_to_wrong']['count']}.")
    lines.extend(["", "## Retrieval-conditioned verifier accuracy", ""])
    for name, values in report["retrieval_conditioned_verifier_accuracy"].items():
        lines.append(f"- {name}: {values['correct']}/{values['claims']} correct ({pct(values['accuracy'])}).")
    strata = report["stratified_diagnostics"]["groups"]
    lines.extend(
        [
            "",
            "## Stratified diagnostics",
            "",
            "Within a single gold-label stratum, accuracy is that class's recall.",
            "",
            "| Gold label | Claims | No evidence | Benchmark-associated | Retrieved |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bucket, label in (("factual", "FACTUAL"), ("non_factual", "NON_FACTUAL")):
        item = strata["gold_label"][bucket]
        lines.append(
            f"| {label} | {item['claim_count']} | "
            f"{pct(item['settings']['no_evidence']['accuracy'])} | "
            f"{pct(item['settings']['oracle_evidence']['accuracy'])} | "
            f"{pct(item['settings']['retrieved_evidence']['accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "| Retrieval stratum | Claims | No evidence acc. | Benchmark-associated acc. | Retrieved acc. |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    retrieval_labels = {
        "source_document_hit_at_5": "Source document hit@5",
        "source_document_eligible_miss_at_5": "Source document eligible miss@5",
        "no_source_qrel_or_unavailable": "No source qrel / unavailable",
        "strict_passage_hit_at_5": "Strict passage hit@5",
        "strict_passage_eligible_miss_at_5": "Strict passage eligible miss@5",
        "no_strict_passage_qrel_or_unavailable": "No strict qrel / unavailable",
    }
    for group in ("source_document_retrieval", "strict_passage_retrieval"):
        for bucket, item in strata[group].items():
            lines.append(
                f"| {retrieval_labels[bucket]} | {item['claim_count']} | "
                f"{pct(item['settings']['no_evidence']['accuracy'])} | "
                f"{pct(item['settings']['oracle_evidence']['accuracy'])} | "
                f"{pct(item['settings']['retrieved_evidence']['accuracy'])} |"
            )
    lines.extend(
        [
            "",
            "| Gold evidence stance bundle | Claims | No evidence acc. | Benchmark-associated acc. | Retrieved acc. |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bucket, item in strata["gold_evidence_stance"].items():
        lines.append(
            f"| {bucket} | {item['claim_count']} | "
            f"{pct(item['settings']['no_evidence']['accuracy'])} | "
            f"{pct(item['settings']['oracle_evidence']['accuracy'])} | "
            f"{pct(item['settings']['retrieved_evidence']['accuracy'])} |"
        )
    if report["technical_completion"]["format_repairs"]:
        lines.extend(["", "## Technical completion", ""])
        for repair in report["technical_completion"]["format_repairs"]:
            lines.append(
                f"- `{repair['claim_id']}`: format-only rationale shortening "
                f"({repair['original_rationale_word_count']} to "
                f"{repair['repaired_rationale_word_count']} words); prediction and "
                "confidence were preserved exactly."
            )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], json_report: Path, md_report: Path) -> None:
    json_report.parent.mkdir(parents=True, exist_ok=True)
    json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_report.write_text(markdown(report), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Hybrid verifier on 468 held-out claims.")
    parser.add_argument("--scope", choices=("full",), default="full")
    parser.add_argument(
        "--top-k",
        type=int,
        choices=SUPPORTED_TOP_K,
        default=PRIMARY_TOP_K,
        help=(
            "Use 5 for the formal run (default), or 1/3 for isolated held-out "
            "sensitivity checks."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Keep compatible successful rows and retry missing/non-ok rows.")
    parser.add_argument("--report-only", action="store_true", help="Regenerate reports without Ollama calls.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the first model message without calls or writes.")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-consecutive-request-errors", type=int, default=3)
    args = parser.parse_args(argv)
    if args.report_only and (args.resume or args.dry_run):
        parser.error("--report-only cannot be combined with --resume or --dry-run")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    top_k = int(args.top_k)
    output, json_report, md_report = output_paths(top_k)
    records, _, _ = load_inputs()
    bundles, retrieved_passages, retrieved_docs = build_retrieved_evidence(records, top_k)
    no_rows, no_profile = load_historical(NO_RESULTS, records, "no_evidence")
    oracle_rows, oracle_profile = load_historical(ORACLE_RESULTS, records, "oracle_evidence")
    if no_profile != oracle_profile:
        raise ValueError(f"No-evidence and oracle model profiles differ: {no_profile} vs {oracle_profile}")
    prompt_template = PROMPT.read_text(encoding="utf-8")
    for placeholder in ("{gold_claim_json}", "{retrieved_evidence_text}"):
        if prompt_template.count(placeholder) != 1:
            raise ValueError(
                f"Retrieved verifier prompt placeholder is invalid: {placeholder}"
            )
    template = effective_prompt_template(prompt_template, top_k)
    repair_template = FORMAT_REPAIR_PROMPT.read_text(encoding="utf-8")
    for placeholder in ("{prediction_json}", "{confidence_json}", "{rationale_json}"):
        if repair_template.count(placeholder) != 1:
            raise ValueError(f"Format-repair prompt placeholder is invalid: {placeholder}")

    observed_digest = str(no_profile["model_digest"])
    if not args.report_only and not args.dry_run:
        client = _base.Client(host=args.ollama_host, timeout=args.timeout)
        observed_digest = _base.preflight_ollama(client, str(no_profile["model"]))
        if observed_digest != no_profile["model_digest"]:
            raise ValueError(f"Installed model digest differs from historical runs: {observed_digest} vs {no_profile['model_digest']}")
    else:
        client = None
    config = run_config(
        args, records, bundles, no_profile, observed_digest, top_k, template
    )

    if args.dry_run:
        first = records[0]
        print(json.dumps({"validated_claims": len(records), "responses": EXPECTED_RESPONSES, "retriever": "hybrid_rrf", "top_k": top_k, "run_fingerprint": config["run_fingerprint"], "output": str(output.relative_to(PROJECT_ROOT))}, indent=2))
        print("MODEL MESSAGE START")
        print(build_prompt(template, first, bundles[str(first["claim_id"])]))
        print("MODEL MESSAGE END")
        return 0

    existing = read_jsonl(output) if output.exists() else []
    if existing and not (args.resume or args.report_only):
        raise FileExistsError(f"{output} exists; use --resume or --report-only")
    for row in existing:
        if row.get("run_fingerprint") != config["run_fingerprint"]:
            raise ValueError("Existing retrieved-verifier output has an incompatible run fingerprint")
    normalized_existing = []
    for row in existing:
        if args.resume and row.get("status") == "parse_error":
            print(
                f"[repair] attempting format-only repair for {row.get('claim_id')} ...",
                flush=True,
            )
            repaired = repair_existing_parse_error(row, config, client)
            if repaired.get("status") == "ok":
                print(
                    f"[repair] success prediction={repaired.get('prediction')} "
                    f"confidence={repaired.get('confidence')}",
                    flush=True,
                )
            else:
                print("[repair] not repairable; normal retry remains pending.", flush=True)
            row = repaired
        normalized_existing.append(row)
    existing = normalized_existing
    successful = {str(row["claim_id"]): row for row in existing if row.get("status") == "ok"}

    if not args.report_only:
        atomic_write_jsonl(output, [successful[claim_id] for claim_id in sorted(successful)])
        pending = [row for row in records if str(row["claim_id"]) not in successful]
        absolute_position = {
            str(record["claim_id"]): index
            for index, record in enumerate(records, 1)
        }
        print(f"Retrieved verifier: {len(successful)} compatible successes retained; {len(pending)} pending.", flush=True)
        consecutive_errors = 0
        with output.open("a", encoding="utf-8") as sink:
            for record in pending:
                claim_id = str(record["claim_id"])
                index = absolute_position[claim_id]
                print(f"[{index}/{len(records)}] verifying {claim_id} with Hybrid top-{top_k} ...", flush=True)
                row = process(record, bundles[claim_id], template, config, client)
                append_result(sink, row)
                print(f"[{index}/{len(records)}] {row['status']} prediction={row.get('prediction')} latency={row['latency_seconds']:.2f}s", flush=True)
                consecutive_errors = consecutive_errors + 1 if row["status"] == "request_error" else 0
                if consecutive_errors >= args.max_consecutive_request_errors:
                    print("Stopped after consecutive request failures; rerun with --resume.", file=sys.stderr)
                    break

    current = read_jsonl(output)
    by_id = unique_by(current, "claim_id", "retrieved verifier results")
    ordered = [by_id[str(record["claim_id"])] for record in records if str(record["claim_id"]) in by_id]
    report = build_report(
        records,
        ordered,
        no_rows,
        oracle_rows,
        config,
        retrieved_passages,
        retrieved_docs,
        top_k,
        output,
    )
    write_reports(report, json_report, md_report)
    print(json.dumps({
        "status": report["status"],
        "completed": len(ordered),
        "expected": len(records),
        "results": str(output.relative_to(PROJECT_ROOT)),
        "report": str(md_report.relative_to(PROJECT_ROOT)),
    }, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
