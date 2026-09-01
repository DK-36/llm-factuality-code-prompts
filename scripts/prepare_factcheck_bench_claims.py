#!/usr/bin/env python3
"""Prepare one unified pilot/full FactCheck-Bench claim benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factcheck_bench_pipeline import (  # noqa: E402
    SCOPES,
    annotate_claim_cohorts,
    canonical_json_hash,
    paths_for_scope,
)


REQUIRED_CLAIM_FIELDS = [
    # These fields are required to construct the gold verifier benchmark.
    "claims_factuality_label",
    "human_evidence",
    "auto_evidence",
    "auto_evidence_url",
    "stance_claim_autoEvid",
]

OPTIONAL_CLAIM_FIELDS = [
    # These fields are retained only as audit metadata. Some original
    # FactCheck-Bench sentences contain revision metadata even when the
    # original `claims` list is empty, so they must not be required to
    # align perfectly with the original claims.
    "claim_checkworthiness",
    "if_automatic_evidence_enough_to_verify",
    "if_claim_needs_edit",
    "revised_claims",
    "usedEvidence_index_in_revision",
    "if_most_important_claim",
]

NA_VALUES = {"", "na", "n/a", "none", "null", "not available"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_path(path: Path) -> str:
    """Use repository-relative paths in portable generated reports."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise TypeError("JSON root must be a list.")
        if not all(isinstance(item, dict) for item in data):
            raise TypeError("Every JSON array item must be an object.")
        return data

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if not isinstance(item, dict):
                raise TypeError(
                    f"Line {line_number} must contain a JSON object."
                )

            records.append(item)

    return records


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_string(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if text.lower() in NA_VALUES:
        return None

    return text


def parse_human_evidence(value: Any) -> dict[str, Any] | None:
    """
    FactCheck-Bench human_evidence often stores:
        URL
        evidence text

    Keep the original string while separating URL and text.
    """
    raw = clean_string(value)
    if raw is None:
        return None

    urls = re.findall(r"https?://[^\s]+", raw)
    url = urls[0].rstrip(".,);]") if urls else None

    text = raw
    if url:
        text = text.replace(url, "", 1).strip()

    return {
        "source": "human",
        "text": text or raw,
        "url": url,
        "stance": "human-validated",
        "raw": raw,
    }


def normalise_stance(value: Any) -> str | None:
    text = clean_string(value)
    if text is None:
        return None

    return text.lower().replace("_", "-").strip()


def normalise_factuality_label(
    value: Any,
    record_index: int,
    sentence_key: str,
    claim_index: int,
) -> tuple[str, bool | None]:
    """
    Preserve FactCheck-Bench's human factuality annotation.

    Returns:
        ("FACTUAL", True)
        ("NON_FACTUAL", False)
        ("UNKNOWN", None)

    UNKNOWN is retained for auditability but should be excluded from
    binary verifier metrics unless the experiment explicitly defines a
    third output class.
    """
    if value is True:
        return "FACTUAL", True

    if value is False:
        return "NON_FACTUAL", False

    if isinstance(value, str):
        normalised = value.strip().lower().replace("-", "_")

        if normalised in {"true", "factual"}:
            return "FACTUAL", True

        if normalised in {"false", "non_factual", "nonfactual"}:
            return "NON_FACTUAL", False

        if normalised in {"unknown", "uncertain", "undetermined"}:
            return "UNKNOWN", None

    raise TypeError(
        f"Record {record_index}, {sentence_key}, claim {claim_index + 1}: "
        "factuality label must be true, false, or 'unknown', "
        f"but got {value!r}."
    )


def validate_sentence(
    record_index: int,
    sentence_key: str,
    sentence: dict[str, Any],
) -> int:
    claims = sentence.get("claims")

    if not isinstance(claims, list):
        raise TypeError(
            f"Record {record_index}, {sentence_key}: 'claims' must be a list."
        )

    claim_count = len(claims)

    # A sentence with no original claims contributes no claim-level example.
    # FactCheck-Bench may still contain revised_claims or other sentence-level
    # revision metadata for such a sentence; that metadata is deliberately
    # ignored here rather than treated as a schema error.
    if claim_count == 0:
        return 0

    # Only fields needed to build the gold claim/evidence benchmark must align
    # exactly with the original claims.
    for field in REQUIRED_CLAIM_FIELDS:
        values = sentence.get(field)

        if not isinstance(values, list):
            raise TypeError(
                f"Record {record_index}, {sentence_key}: "
                f"'{field}' must be a list."
            )

        if len(values) != claim_count:
            raise ValueError(
                f"Record {record_index}, {sentence_key}: "
                f"{claim_count} claims but {len(values)} values in '{field}'."
            )

    # Optional audit metadata may be absent or have a different length in the
    # original benchmark. It is read safely later and never used as gold input.
    for field in OPTIONAL_CLAIM_FIELDS:
        values = sentence.get(field)

        if values is not None and not isinstance(values, list):
            raise TypeError(
                f"Record {record_index}, {sentence_key}: "
                f"optional field '{field}' must be a list when present."
            )

    return claim_count


def get_optional_claim_value(
    sentence: dict[str, Any],
    field: str,
    claim_index: int,
) -> Any:
    """Return optional claim metadata without assuming perfect alignment."""
    values = sentence.get(field)

    if not isinstance(values, list):
        return None

    if claim_index >= len(values):
        return None

    return values[claim_index]


def build_evidence_bundle(
    sentence: dict[str, Any],
    claim_index: int,
    record_index: int,
    sentence_key: str,
) -> list[dict[str, Any]]:
    evidence_bundle: list[dict[str, Any]] = []

    auto_texts = sentence["auto_evidence"][claim_index]
    auto_urls = sentence["auto_evidence_url"][claim_index]
    auto_stances = sentence["stance_claim_autoEvid"][claim_index]

    if not isinstance(auto_texts, list):
        raise TypeError(
            f"Record {record_index}, {sentence_key}, "
            f"claim {claim_index + 1}: "
            "'auto_evidence' item must be a list."
        )

    if not isinstance(auto_urls, list):
        raise TypeError(
            f"Record {record_index}, {sentence_key}, "
            f"claim {claim_index + 1}: "
            "'auto_evidence_url' item must be a list."
        )

    if not isinstance(auto_stances, list):
        raise TypeError(
            f"Record {record_index}, {sentence_key}, "
            f"claim {claim_index + 1}: "
            "'stance_claim_autoEvid' item must be a list."
        )

    # Retain only auto-retrieved passages that humans annotated as relevant.
    for rank, text in enumerate(auto_texts, start=1):
        list_index = rank - 1
        url = auto_urls[list_index] if list_index < len(auto_urls) else None
        stance = (
            auto_stances[list_index]
            if list_index < len(auto_stances)
            else None
        )
        evidence_text = clean_string(text)
        if evidence_text is None:
            continue

        evidence_url = clean_string(url)
        normalised_stance = normalise_stance(stance)

        # A retrieved passage without a corresponding human stance is not
        # oracle evidence. Its original text/URL/stance lists remain in the
        # output's raw audit fields.
        if normalised_stance is None:
            continue

        if normalised_stance == "irrelevant":
            continue

        evidence_bundle.append(
            {
                "source": "annotated_auto",
                "rank": rank,
                "text": evidence_text,
                "url": evidence_url,
                "stance": normalised_stance,
            }
        )

    human_item = parse_human_evidence(
        sentence["human_evidence"][claim_index]
    )
    if human_item is not None:
        evidence_bundle.append(human_item)

    return evidence_bundle


def evidence_source_label(
    evidence_bundle: list[dict[str, Any]],
) -> str:
    sources = {item["source"] for item in evidence_bundle}

    if sources == {"human"}:
        return "human"
    if sources == {"annotated_auto"}:
        return "annotated_auto"
    if sources == {"human", "annotated_auto"}:
        return "human+annotated_auto"
    return "none"


def flatten_records(
    source_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for record_index, record in enumerate(source_records, start=1):
        response_id = f"fcb_r{record_index:04d}"
        sentences = record.get("sentences")

        if not isinstance(sentences, dict):
            raise TypeError(
                f"Record {record_index}: 'sentences' must be an object."
            )

        for sentence_key, sentence in sentences.items():
            if not isinstance(sentence, dict):
                raise TypeError(
                    f"Record {record_index}, {sentence_key}: "
                    "sentence value must be an object."
                )

            claim_count = validate_sentence(
                record_index,
                sentence_key,
                sentence,
            )

            for claim_index in range(claim_count):
                raw_label = sentence["claims_factuality_label"][claim_index]
                human_label, human_label_bool = normalise_factuality_label(
                    value=raw_label,
                    record_index=record_index,
                    sentence_key=sentence_key,
                    claim_index=claim_index,
                )

                evidence_bundle = build_evidence_bundle(
                    sentence=sentence,
                    claim_index=claim_index,
                    record_index=record_index,
                    sentence_key=sentence_key,
                )

                claim_number = claim_index + 1
                claim_id = (
                    f"{response_id}_{sentence_key}_c{claim_number:02d}"
                )

                output.append(
                    {
                        "claim_id": claim_id,
                        "response_id": response_id,
                        "source_record_index": record_index,
                        "sentence_id": sentence_key,
                        "claim_index_in_sentence": claim_number,
                        "prompt": record.get("prompt"),
                        "source_response": record.get("response"),
                        "source_sentence": sentence.get("text"),
                        "gold_claim": sentence["claims"][claim_index],
                        "human_label": human_label,
                        "human_label_bool": human_label_bool,
                        "human_label_raw": raw_label,
                        "is_binary_evaluable": (
                            human_label in {"FACTUAL", "NON_FACTUAL"}
                        ),
                        "gold_evidence": evidence_bundle,
                        "gold_evidence_texts": [
                            item["text"] for item in evidence_bundle
                        ],
                        "evidence_available": bool(evidence_bundle),
                        "evidence_source": evidence_source_label(
                            evidence_bundle
                        ),
                        "auto_evidence_sufficient": (
                            get_optional_claim_value(
                                sentence,
                                "if_automatic_evidence_enough_to_verify",
                                claim_index,
                            )
                        ),
                        "claim_checkworthiness": (
                            get_optional_claim_value(
                                sentence,
                                "claim_checkworthiness",
                                claim_index,
                            )
                        ),
                        "claim_needs_edit": (
                            get_optional_claim_value(
                                sentence,
                                "if_claim_needs_edit",
                                claim_index,
                            )
                        ),
                        "revised_claim": (
                            get_optional_claim_value(
                                sentence,
                                "revised_claims",
                                claim_index,
                            )
                        ),
                        "revision_evidence_index": (
                            get_optional_claim_value(
                                sentence,
                                "usedEvidence_index_in_revision",
                                claim_index,
                            )
                        ),
                        "claim_importance": (
                            get_optional_claim_value(
                                sentence,
                                "if_most_important_claim",
                                claim_index,
                            )
                        ),
                        "raw_auto_evidence": (
                            sentence["auto_evidence"][claim_index]
                        ),
                        "raw_auto_evidence_urls": (
                            sentence["auto_evidence_url"][claim_index]
                        ),
                        "raw_auto_evidence_stances": (
                            sentence["stance_claim_autoEvid"][claim_index]
                        ),
                        "raw_human_evidence": (
                            sentence["human_evidence"][claim_index]
                        ),
                    }
                )

    return output


def build_summary(
    source_records: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    label_counts = Counter(item["human_label"] for item in claims)
    source_counts = Counter(item["evidence_source"] for item in claims)
    sufficiency_counts = Counter(
        str(item["auto_evidence_sufficient"]).lower()
        for item in claims
    )

    no_evidence_ids = [
        item["claim_id"]
        for item in claims
        if not item["evidence_available"]
    ]
    binary_evaluable_ids = [
        item["claim_id"]
        for item in claims
        if item["is_binary_evaluable"]
    ]
    unknown_ids = [
        item["claim_id"]
        for item in claims
        if item["human_label"] == "UNKNOWN"
    ]
    usable_oracle_evidence_ids = [
        item["claim_id"]
        for item in claims
        if item.get("oracle_evidence_available") is True
    ]
    matched_ids = [
        item["claim_id"]
        for item in claims
        if item.get("in_primary_matched_cohort") is True
    ]
    alignment_warnings: list[dict[str, Any]] = []
    excluded_without_stance_count = 0
    missing_url_count = 0

    for item in claims:
        auto_texts = item["raw_auto_evidence"]
        auto_urls = item["raw_auto_evidence_urls"]
        auto_stances = item["raw_auto_evidence_stances"]

        if not (
            len(auto_texts) == len(auto_urls) == len(auto_stances)
        ):
            alignment_warnings.append(
                {
                    "claim_id": item["claim_id"],
                    "auto_evidence_text_count": len(auto_texts),
                    "auto_evidence_url_count": len(auto_urls),
                    "auto_evidence_stance_count": len(auto_stances),
                }
            )

        for evidence_index, text in enumerate(auto_texts):
            if clean_string(text) is None:
                continue

            url = (
                auto_urls[evidence_index]
                if evidence_index < len(auto_urls)
                else None
            )
            if clean_string(url) is None:
                missing_url_count += 1

            stance = (
                auto_stances[evidence_index]
                if evidence_index < len(auto_stances)
                else None
            )
            if normalise_stance(stance) is None:
                excluded_without_stance_count += 1

        missing_url_count += sum(
            evidence["source"] == "human" and evidence["url"] is None
            for evidence in item["gold_evidence"]
        )

    return {
        "input_file": report_path(input_path),
        "output_file": report_path(output_path),
        "source_response_count": len(source_records),
        "gold_claim_count": len(claims),
        "total_claim_count": len(claims),
        "binary_evaluable_claim_count": len(binary_evaluable_ids),
        "factual_claim_count": label_counts["FACTUAL"],
        "non_factual_claim_count": label_counts["NON_FACTUAL"],
        "unknown_claim_count": len(unknown_ids),
        "unknown_claim_ids": unknown_ids,
        "human_label_counts": dict(label_counts),
        "evidence_source_counts": dict(source_counts),
        "auto_evidence_sufficiency_counts": dict(sufficiency_counts),
        "claims_without_gold_evidence_count": len(no_evidence_ids),
        "claims_without_gold_evidence_ids": no_evidence_ids,
        "claims_without_oracle_evidence_count": len(no_evidence_ids),
        "claims_without_oracle_evidence_ids": no_evidence_ids,
        "usable_oracle_evidence_claim_count": len(
            usable_oracle_evidence_ids
        ),
        "claims_without_usable_oracle_evidence_count": (
            len(claims) - len(usable_oracle_evidence_ids)
        ),
        "matched_claim_count": len(matched_ids),
        "matched_claim_ids": matched_ids,
        "evidence_text_url_stance_alignment_warning_count": len(
            alignment_warnings
        ),
        "evidence_text_url_stance_alignment_warnings": alignment_warnings,
        "evidence_passages_excluded_without_human_stance_count": (
            excluded_without_stance_count
        ),
        "evidence_passages_with_missing_urls_count": missing_url_count,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten FactCheck-Bench into a claim-level verifier benchmark."
        )
    )

    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="pilot",
        help=(
            "Resolve scope-specific defaults. pilot preserves historical paths; "
            "full reads the complete raw benchmark and writes separate outputs."
        ),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input FactCheck-Bench JSONL/JSON (overrides --scope default).",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output claim-level JSONL (overrides --scope default).",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output QA summary JSON (overrides --scope default).",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output compact cohort manifest JSONL (overrides scope default).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse, validate, and print counts without writing any files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing gold/manifest/report files. This can invalidate "
            "prediction fingerprints and is therefore never implicit."
        ),
    )

    args = parser.parse_args(argv)
    defaults = paths_for_scope(PROJECT_ROOT, args.scope)
    args.input = args.input or defaults.source_input
    args.output = args.output or defaults.gold_claims
    args.report = args.report or defaults.preparation_report
    args.manifest = args.manifest or defaults.cohort_manifest

    destinations = {
        args.output.resolve(strict=False),
        args.report.resolve(strict=False),
        args.manifest.resolve(strict=False),
    }
    if len(destinations) != 3:
        parser.error("--output, --report, and --manifest must be distinct")
    if args.input.resolve(strict=False) in destinations:
        parser.error("Refusing to overwrite the input file")
    if args.scope == "full":
        pilot = paths_for_scope(PROJECT_ROOT, "pilot")
        pilot_output_root = pilot.output_root.resolve(strict=False)
        protected_pilot_paths = {
            pilot.gold_claims.resolve(strict=False),
            pilot.cohort_manifest.resolve(strict=False),
        }
        collisions = [
            path
            for path in destinations
            if path in protected_pilot_paths
            or path == pilot_output_root
            or pilot_output_root in path.parents
        ]
        if collisions:
            parser.error(
                "Full scope may not write pilot data or output paths: "
                f"{sorted(str(path) for path in collisions)}"
            )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    source_records = load_records(args.input)
    claim_records = flatten_records(source_records)

    if not claim_records:
        raise ValueError("No claims were extracted.")

    claim_records, cohort_manifest, cohort_summary = annotate_claim_cohorts(
        claim_records
    )

    summary = build_summary(
        source_records=source_records,
        claims=claim_records,
        input_path=args.input,
        output_path=args.output,
    )
    claim_response_ids = {
        item["response_id"] for item in claim_records
    }
    source_response_ids = {
        f"fcb_r{index:04d}"
        for index in range(1, len(source_records) + 1)
    }
    summary.update(
        {
            "scope": args.scope,
            "input_sha256": sha256_file(args.input),
            "manifest_file": report_path(args.manifest),
            "cohort_manifest_sha256": canonical_json_hash(cohort_manifest),
            "claim_records_sha256": canonical_json_hash(claim_records),
            "claim_response_count": len(claim_response_ids),
            "source_responses_without_claims_count": len(
                source_response_ids - claim_response_ids
            ),
            "source_responses_without_claims": sorted(
                source_response_ids - claim_response_ids
            ),
            "cohorts": cohort_summary,
            "field_semantics": {
                "evidence_available": (
                    "legacy structural non-empty bundle flag"
                ),
                "oracle_evidence_available": (
                    "normalization status is ok with at least one usable text"
                ),
                "in_primary_matched_cohort": (
                    "binary human label and oracle_evidence_available"
                ),
            },
        }
    )

    if not args.dry_run:
        existing = [
            path
            for path in (args.output, args.manifest, args.report)
            if path.exists()
        ]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Refusing to replace existing prepared artifacts without "
                f"--overwrite: {[str(path) for path in existing]}"
            )
        save_jsonl(claim_records, args.output)
        save_jsonl(cohort_manifest, args.manifest)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print("=" * 72)
    print("FACTCHECK-BENCH GOLD CLAIM PREPARATION")
    print("=" * 72)
    print(f"Scope: {args.scope}")
    print(f"Mode: {'dry-run (no writes)' if args.dry_run else 'write'}")
    print(f"Source responses: {summary['source_response_count']}")
    print(f"Responses with claims: {summary['claim_response_count']}")
    print(f"Gold claims: {summary['gold_claim_count']}")
    print(
        "Binary-evaluable claims: "
        f"{summary['binary_evaluable_claim_count']}"
    )
    print(f"Unknown claims: {summary['unknown_claim_count']}")
    print(f"Matched claims: {summary['cohorts']['matched_claim_count']}")
    print(f"Labels: {summary['human_label_counts']}")
    print(f"Evidence sources: {summary['evidence_source_counts']}")
    print(
        "Claims without gold evidence: "
        f"{summary['claims_without_gold_evidence_count']}"
    )
    print(
        "Evidence alignment warnings: "
        f"{summary['evidence_text_url_stance_alignment_warning_count']}"
    )
    print(
        "Passages excluded without human stance: "
        f"{summary['evidence_passages_excluded_without_human_stance_count']}"
    )
    print(
        "Evidence passages with missing URLs: "
        f"{summary['evidence_passages_with_missing_urls_count']}"
    )
    print(f"Output: {args.output}")
    print(f"Cohort manifest: {args.manifest}")
    print(f"QA report: {args.report}")
    if args.dry_run:
        print("No files were written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
