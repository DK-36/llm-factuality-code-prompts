"""Shared FactCheck-Bench paths, evidence normalization, and cohorts.

This module deliberately contains no model client.  Data preparation and both
verifier settings import the same definitions so that ``pilot`` and ``full``
cannot silently disagree about paths or the matched oracle-evidence cohort.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCOPES = ("pilot", "full")
PRIMARY_LABELS = ("FACTUAL", "NON_FACTUAL")
NORMALIZATION_VERSION = "oracle_evidence_normalization_v1"
COHORT_VERSION = "binary_with_usable_oracle_evidence_v1"
FROZEN_QWEN3_8B_DIGEST = (
    "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"
)
EVIDENCE_BUNDLE_KEYS = (
    "gold_evidence",
    "evidence",
    "items",
    "passages",
    "bundle",
)


@dataclass(frozen=True)
class DatasetPaths:
    """All scope-dependent paths used by the verifier pipeline."""

    scope: str
    source_input: Path
    gold_claims: Path
    cohort_manifest: Path
    preparation_report: Path
    output_root: Path
    no_evidence_output: Path
    no_evidence_report: Path
    no_evidence_markdown: Path
    oracle_output: Path
    oracle_report: Path
    oracle_markdown: Path
    oracle_smoke_output: Path
    oracle_smoke_report: Path
    oracle_smoke_markdown: Path


@dataclass(frozen=True)
class RetrievalPaths:
    """Canonical full-scope paths for Experiment A corpus artifacts."""

    scope: str
    root: Path
    config: Path
    manifests_dir: Path
    reports_dir: Path
    raw_documents_dir: Path
    split_manifest: Path
    split_summary: Path
    evidence_manifest: Path
    url_manifest: Path
    preparation_report_json: Path
    preparation_report_markdown: Path
    canonicalisation_report: Path
    fetch_report: Path
    reprocess_report_json: Path
    reprocess_report_markdown: Path
    documents: Path
    passages: Path
    passage_build_report: Path
    qrels_jsonl: Path
    qrels_tsv: Path
    qrels_mapping_audit: Path
    qrels_mapping_report_json: Path
    qrels_mapping_report_markdown: Path
    qrels_dev_jsonl: Path
    qrels_dev_tsv: Path
    qrels_dev_mapping_audit: Path
    qrels_dev_mapping_report_json: Path
    qrels_dev_mapping_report_markdown: Path
    qrels_heldout_jsonl: Path
    qrels_heldout_tsv: Path
    qrels_heldout_mapping_audit: Path
    qrels_heldout_mapping_report_json: Path
    qrels_heldout_mapping_report_markdown: Path
    corpus_summary_json: Path
    corpus_summary_markdown: Path


def paths_for_scope(project_root: Path, scope: str) -> DatasetPaths:
    """Resolve deterministic project-relative paths for ``pilot`` or ``full``."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")

    data_root = project_root / "data" / "factcheck_bench"
    if scope == "pilot":
        source_input = data_root / "processed" / "fcb_annotation_pilot_20.jsonl"
        gold_claims = data_root / "processed" / "fcb_gold_claims_pilot_20.jsonl"
        manifest = data_root / "processed" / "fcb_cohort_manifest_pilot_20.jsonl"
        output_root = project_root / "outputs" / "factcheck_bench_pilot"
    else:
        source_input = data_root / "raw" / "factcheck-GPT-benchmark.jsonl"
        gold_claims = data_root / "processed" / "fcb_gold_claims_full.jsonl"
        manifest = data_root / "processed" / "fcb_cohort_manifest_full.jsonl"
        output_root = project_root / "outputs" / "factcheck_bench_full"

    return DatasetPaths(
        scope=scope,
        source_input=source_input,
        gold_claims=gold_claims,
        cohort_manifest=manifest,
        preparation_report=output_root / "reports" / "07_gold_claims_summary.json",
        output_root=output_root,
        no_evidence_output=(
            output_root / "jsonl" / "08b_no_evidence_verifier_results.jsonl"
        ),
        no_evidence_report=(
            output_root / "reports" / "08b_no_evidence_verifier_summary.json"
        ),
        no_evidence_markdown=(
            output_root / "reports" / "08b_no_evidence_verifier_report.md"
        ),
        oracle_output=(
            output_root / "jsonl" / "08c_oracle_evidence_verifier_results.jsonl"
        ),
        oracle_report=(
            output_root / "reports" / "08c_oracle_evidence_verifier_summary.json"
        ),
        oracle_markdown=(
            output_root / "reports" / "08c_oracle_evidence_verifier_report.md"
        ),
        oracle_smoke_output=(
            output_root / "jsonl" / "08c_oracle_evidence_smoke.jsonl"
        ),
        oracle_smoke_report=(
            output_root / "reports" / "08c_oracle_evidence_smoke_summary.json"
        ),
        oracle_smoke_markdown=(
            output_root / "reports" / "08c_oracle_evidence_smoke_report.md"
        ),
    )


def retrieval_paths(project_root: Path, scope: str = "full") -> RetrievalPaths:
    """Resolve Experiment A paths without creating directories.

    Retrieval development and held-out membership is defined only for the full
    canonical cohort.  Keeping this full-only prevents the historical pilot
    gold file from becoming a second, competing split source.
    """
    if scope != "full":
        raise ValueError("Experiment A retrieval artifacts support full scope only")

    root = project_root / "data" / "factcheck_bench" / "retrieval"
    manifests = root / "manifests"
    reports = root / "reports"
    return RetrievalPaths(
        scope=scope,
        root=root,
        config=root / "config" / "retrieval_corpus_config.json",
        manifests_dir=manifests,
        reports_dir=reports,
        raw_documents_dir=root / "raw_documents",
        split_manifest=manifests / "retrieval_split_manifest.jsonl",
        split_summary=manifests / "retrieval_split_summary.json",
        evidence_manifest=manifests / "evidence_manifest.jsonl",
        url_manifest=manifests / "url_manifest.jsonl",
        preparation_report_json=manifests / "preparation_report.json",
        preparation_report_markdown=manifests / "preparation_report.md",
        canonicalisation_report=reports / "url_canonicalisation_report.json",
        fetch_report=reports / "fetch_report.json",
        reprocess_report_json=reports / "frozen_document_reprocess_report.json",
        reprocess_report_markdown=reports / "frozen_document_reprocess_report.md",
        documents=root / "documents.jsonl",
        passages=root / "passages.jsonl",
        passage_build_report=reports / "passage_build_report.json",
        qrels_jsonl=root / "qrels.jsonl",
        qrels_tsv=root / "qrels.tsv",
        qrels_mapping_audit=reports / "qrels_mapping_audit.jsonl",
        qrels_mapping_report_json=reports / "qrels_mapping_report.json",
        qrels_mapping_report_markdown=reports / "qrels_mapping_report.md",
        qrels_dev_jsonl=root / "qrels_dev.jsonl",
        qrels_dev_tsv=root / "qrels_dev.tsv",
        qrels_dev_mapping_audit=reports / "qrels_dev_mapping_audit.jsonl",
        qrels_dev_mapping_report_json=reports / "qrels_dev_mapping_report.json",
        qrels_dev_mapping_report_markdown=reports / "qrels_dev_mapping_report.md",
        qrels_heldout_jsonl=root / "qrels_heldout.jsonl",
        qrels_heldout_tsv=root / "qrels_heldout.tsv",
        qrels_heldout_mapping_audit=(
            reports / "qrels_heldout_mapping_audit.jsonl"
        ),
        qrels_heldout_mapping_report_json=(
            reports / "qrels_heldout_mapping_report.json"
        ),
        qrels_heldout_mapping_report_markdown=(
            reports / "qrels_heldout_mapping_report.md"
        ),
        corpus_summary_json=reports / "corpus_summary.json",
        corpus_summary_markdown=reports / "corpus_summary.md",
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def normalize_evidence_text(value: Any) -> tuple[str | None, str | None]:
    """Return stable passage text or a precise exclusion reason."""
    if not isinstance(value, str):
        return None, "text_not_string"
    text = " ".join(value.split())
    if not text:
        return None, "empty_or_whitespace_text"
    if re.fullmatch(r"(?i)(?:url|link)\s*:?\s*", text):
        return None, "marker_only_text"
    if re.fullmatch(r"(?i)(?:https?://|www\.)\S+", text):
        return None, "url_only_text"
    if not any(character.isalnum() for character in text):
        return None, "punctuation_only_text"
    return text, None


def _collect_evidence_items(
    value: Any,
    path: str,
    items: list[tuple[str, dict[str, Any]]],
    invalid: list[dict[str, Any]],
) -> None:
    if isinstance(value, list):
        for index, child in enumerate(value):
            _collect_evidence_items(child, f"{path}[{index}]", items, invalid)
        return
    if isinstance(value, dict):
        if "text" in value:
            items.append((path, value))
            return
        nested_keys = [key for key in EVIDENCE_BUNDLE_KEYS if key in value]
        if nested_keys:
            for key in nested_keys:
                _collect_evidence_items(value[key], f"{path}.{key}", items, invalid)
            return
        invalid.append({"path": path, "reason": "object_has_no_text_or_bundle"})
        return
    invalid.append(
        {"path": path, "reason": f"unsupported_item_type:{type(value).__name__}"}
    )


def normalize_oracle_evidence(value: Any) -> dict[str, Any]:
    """Normalize an evidence bundle using the frozen oracle-v1 semantics.

    Only ``text`` is rendered into the model-visible block.  Trace metadata is
    retained separately, and item-level ``raw`` is deliberately never copied.
    """
    if value is None:
        return {
            "normalization_version": NORMALIZATION_VERSION,
            "status": "no_evidence_bundle",
            "normalized_text": None,
            "normalized_sha256": None,
            "total_item_count": 0,
            "valid_item_count": 0,
            "items": [],
            "skipped_items": [{"path": "gold_evidence", "reason": "missing"}],
            "model_visible_fields": ["text"],
        }

    candidates: list[tuple[str, dict[str, Any]]] = []
    skipped: list[dict[str, Any]] = []
    _collect_evidence_items(value, "gold_evidence", candidates, skipped)
    status = "no_evidence_bundle" if not candidates and not skipped else "pending"

    valid_items: list[dict[str, Any]] = []
    prompt_blocks: list[str] = []
    for original_index, (path, item) in enumerate(candidates, start=1):
        text, reason = normalize_evidence_text(item.get("text"))
        if reason is not None:
            skipped.append({"path": path, "reason": reason})
            continue

        url = item.get("url")
        source = item.get("source")
        stance = item.get("stance")
        rank = item.get("rank")
        metadata = {
            "evidence_index": len(valid_items) + 1,
            "original_item_index": original_index,
            "path": path,
            "text": text,
            "text_sha256": sha256_text(text or ""),
            "url": url.strip() if isinstance(url, str) and url.strip() else None,
            "source": (
                source.strip() if isinstance(source, str) and source.strip() else None
            ),
            "rank": rank if isinstance(rank, int) and not isinstance(rank, bool) else None,
            "stance": (
                stance.strip() if isinstance(stance, str) and stance.strip() else None
            ),
        }
        valid_items.append(metadata)
        prompt_blocks.append(
            f"Evidence {metadata['evidence_index']} text (JSON-encoded): "
            f"{json.dumps(text, ensure_ascii=False)}"
        )

    normalized_text = "\n".join(prompt_blocks) if prompt_blocks else None
    if valid_items:
        status = "ok"
    elif status == "pending":
        status = "no_valid_evidence_text"

    result = {
        "normalization_version": NORMALIZATION_VERSION,
        "status": status,
        "normalized_text": normalized_text,
        "normalized_sha256": None,
        "total_item_count": len(candidates),
        "valid_item_count": len(valid_items),
        "items": valid_items,
        "skipped_items": skipped,
        "model_visible_fields": ["text"],
    }
    if normalized_text is not None:
        hash_payload = {
            key: child
            for key, child in result.items()
            if key != "normalized_sha256"
        }
        result["normalized_sha256"] = canonical_json_hash(hash_payload)
    return result


def _cohort_details(record: dict[str, Any]) -> dict[str, Any]:
    label = record.get("human_label")
    is_binary = label in PRIMARY_LABELS
    has_evidence_field = "gold_evidence" in record
    raw_bundle = record.get("gold_evidence")
    structural = isinstance(raw_bundle, (list, dict)) and bool(raw_bundle)
    normalized = normalize_oracle_evidence(raw_bundle if has_evidence_field else None)

    if not is_binary:
        reason = "non_binary_human_label"
    elif not has_evidence_field:
        reason = "missing_evidence_field"
    elif normalized["status"] == "no_evidence_bundle":
        reason = "empty_evidence_bundle"
    elif normalized["status"] != "ok":
        reason = "no_valid_evidence_text"
    else:
        reason = None

    return {
        "is_binary": is_binary,
        "human_unknown": label == "UNKNOWN",
        "structural_evidence_available": structural,
        "oracle_evidence_available": normalized["status"] == "ok",
        "in_primary_matched_cohort": is_binary and normalized["status"] == "ok",
        "oracle_evidence_exclusion_reason": reason,
        "normalization": normalized,
    }


def annotate_claim_cohorts(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Add stable cohort fields and return a compact manifest plus summary."""
    annotated: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for record in records:
        claim_id = record.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError("Every claim must have a non-empty claim_id.")
        if claim_id in seen_ids:
            raise ValueError(f"Duplicate claim_id: {claim_id}")
        seen_ids.add(claim_id)

        details = _cohort_details(record)
        normalized = details["normalization"]
        memberships = ["all"]
        if details["is_binary"]:
            memberships.append("binary")
        if details["human_unknown"]:
            memberships.append("human_unknown")
        if details["in_primary_matched_cohort"]:
            memberships.append("matched")

        audit_flags: list[str] = []
        raw_texts = record.get("raw_auto_evidence")
        raw_urls = record.get("raw_auto_evidence_urls")
        raw_stances = record.get("raw_auto_evidence_stances")
        if all(isinstance(value, list) for value in (raw_texts, raw_urls, raw_stances)):
            if not (len(raw_texts) == len(raw_urls) == len(raw_stances)):
                audit_flags.append("auto_evidence_alignment_mismatch")
        if (
            details["structural_evidence_available"]
            and not details["oracle_evidence_available"]
        ):
            audit_flags.append("structural_evidence_without_usable_text")

        row = dict(record)
        row.update(
            {
                "is_binary_evaluable": details["is_binary"],
                "human_unknown": details["human_unknown"],
                "structural_evidence_available": details[
                    "structural_evidence_available"
                ],
                "oracle_evidence_available": details["oracle_evidence_available"],
                "in_primary_matched_cohort": details[
                    "in_primary_matched_cohort"
                ],
                "cohort_memberships": memberships,
                "oracle_evidence_normalization_status": normalized["status"],
                "oracle_evidence_valid_item_count": normalized["valid_item_count"],
                "oracle_evidence_normalized_sha256": normalized[
                    "normalized_sha256"
                ],
                "oracle_evidence_exclusion_reason": details[
                    "oracle_evidence_exclusion_reason"
                ],
                "audit_flag": bool(audit_flags),
                "audit_flags": audit_flags,
            }
        )
        annotated.append(row)

        manifest.append(
            {
                "cohort_version": COHORT_VERSION,
                "claim_id": claim_id,
                "response_id": record.get("response_id"),
                "human_label": record.get("human_label"),
                "is_binary_evaluable": details["is_binary"],
                "human_unknown": details["human_unknown"],
                "structural_evidence_available": details[
                    "structural_evidence_available"
                ],
                "oracle_evidence_available": details["oracle_evidence_available"],
                "in_primary_matched_cohort": details[
                    "in_primary_matched_cohort"
                ],
                "cohort_memberships": memberships,
                "oracle_evidence_normalization_status": normalized["status"],
                "oracle_evidence_valid_item_count": normalized["valid_item_count"],
                "oracle_evidence_exclusion_reason": details[
                    "oracle_evidence_exclusion_reason"
                ],
                "audit_flag": bool(audit_flags),
                "audit_flags": audit_flags,
            }
        )

    label_counts = Counter(row.get("human_label") for row in annotated)
    exclusion_counts = Counter(
        row["oracle_evidence_exclusion_reason"]
        for row in annotated
        if row["oracle_evidence_exclusion_reason"] is not None
    )
    summary = {
        "cohort_version": COHORT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "all_claim_count": len(annotated),
        "binary_claim_count": sum(row["is_binary_evaluable"] for row in annotated),
        "matched_claim_count": sum(
            row["in_primary_matched_cohort"] for row in annotated
        ),
        "human_unknown_claim_count": sum(row["human_unknown"] for row in annotated),
        "structural_evidence_claim_count": sum(
            row["structural_evidence_available"] for row in annotated
        ),
        "oracle_evidence_available_claim_count": sum(
            row["oracle_evidence_available"] for row in annotated
        ),
        "claim_response_count": len(
            {row.get("response_id") for row in annotated if row.get("response_id")}
        ),
        "human_label_counts": dict(label_counts),
        "matched_human_label_counts": dict(
            Counter(
                row.get("human_label")
                for row in annotated
                if row["in_primary_matched_cohort"]
            )
        ),
        "matched_response_count": len(
            {
                row.get("response_id")
                for row in annotated
                if row["in_primary_matched_cohort"]
            }
        ),
        "cohort_exclusion_reason_counts": dict(exclusion_counts),
        "audit_flagged_claim_count": sum(row["audit_flag"] for row in annotated),
        "unique_claim_id_count": len(seen_ids),
    }
    return annotated, manifest, summary


def build_oracle_cohort(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the normalized binary matched cohort used by the oracle verifier."""
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    structural_evidence_binary_count = 0
    evidence_available_binary_count = 0

    for record in records:
        details = _cohort_details(record)
        is_binary = details["is_binary"]
        if is_binary and record.get("evidence_available") is True:
            evidence_available_binary_count += 1
        if is_binary and details["structural_evidence_available"]:
            structural_evidence_binary_count += 1

        if details["in_primary_matched_cohort"]:
            enriched = dict(record)
            enriched["_oracle_evidence"] = details["normalization"]
            eligible.append(enriched)
            continue

        normalized = details["normalization"]
        exclusions.append(
            {
                "claim_id": record["claim_id"],
                "response_id": record["response_id"],
                "human_label": record["human_label"],
                "reason": details["oracle_evidence_exclusion_reason"],
                "normalization_status": (
                    None if not is_binary else normalized["status"]
                ),
                "item_exclusion_reasons": (
                    []
                    if not is_binary
                    else [item["reason"] for item in normalized["skipped_items"]]
                ),
            }
        )

    exclusion_reason_counts = Counter(item["reason"] for item in exclusions)
    item_exclusion_reason_counts = Counter(
        reason
        for item in exclusions
        for reason in item["item_exclusion_reasons"]
    )
    audit = {
        "cohort_version": COHORT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "total_input_claims": len(records),
        "total_input_responses": len({record["response_id"] for record in records}),
        "binary_claims": sum(record.get("human_label") in PRIMARY_LABELS for record in records),
        "human_unknown_claims": sum(record.get("human_label") == "UNKNOWN" for record in records),
        "evidence_available_binary_claims": evidence_available_binary_count,
        "nonempty_evidence_bundle_binary_claims": structural_evidence_binary_count,
        "oracle_eligible_claims": len(eligible),
        "oracle_eligible_responses": len({record["response_id"] for record in eligible}),
        "oracle_eligible_gold_labels": dict(
            Counter(record["human_label"] for record in eligible)
        ),
        "exclusion_reason_counts": dict(exclusion_reason_counts),
        "item_exclusion_reason_counts": dict(item_exclusion_reason_counts),
        "exclusions": exclusions,
    }
    return eligible, audit
