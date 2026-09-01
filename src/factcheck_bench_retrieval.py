"""Experiment A corpus preparation, chunking, and qrels utilities.

The module is deliberately model-free.  Gold labels, evidence text, stance, and
claim-to-URL mappings are persisted for QA/evaluation, but no function here
constructs a retrieval query or ranks a passage with those fields.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from factcheck_bench_pipeline import (
    PRIMARY_LABELS,
    RetrievalPaths,
    canonical_json_hash,
    normalize_evidence_text,
    normalize_oracle_evidence,
    paths_for_scope,
    retrieval_paths,
    sha256_text,
)


RETRIEVAL_SCHEMA_VERSION = "fcb_retrieval_artifacts_v1"
SPLIT_EXPECTATION_ERROR = (
    "Frozen retrieval split counts do not match the canonical files; refusing "
    "to write partial or silently redefined artifacts."
)
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])(?:[\"'’”）\]]*)\s+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not permitted: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_json(path: Path, *, allow_missing: bool = False) -> dict[str, Any]:
    if not path.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Required JSON does not exist: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"Required JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(f"{path} line {line_number} must be an object")
            rows.append(row)
    return rows


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(value)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            for row in rows:
                target.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                        default=_json_default,
                    )
                    + "\n"
                )
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def project_relative(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def config_for_paths(paths: RetrievalPaths) -> dict[str, Any]:
    config = load_json(paths.config)
    if config.get("schema_version") != "fcb_retrieval_config_v1":
        raise ValueError("Unsupported retrieval config schema_version")
    if config.get("corpus", {}).get("scope") != "full":
        raise ValueError("Retrieval config must be full scope")
    return config


def _normalise_hostname(hostname: str) -> str:
    return hostname.rstrip(".").encode("idna").decode("ascii").lower()


def canonicalize_url(raw_url: Any, settings: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic candidate URL without resolving redirects."""
    version = str(settings.get("version", "unknown"))
    result: dict[str, Any] = {
        "version": version,
        "raw_url": raw_url if isinstance(raw_url, str) else None,
        "canonical_url": None,
        "status": "invalid",
        "error": None,
        "fragment_removed": False,
        "tracking_parameters_removed": [],
        "trailing_slash_changed": False,
    }
    if not isinstance(raw_url, str) or not raw_url.strip():
        result["error"] = "missing_or_non_string_url"
        return result

    raw = html.unescape(raw_url.strip())
    try:
        parsed = urlsplit(raw)
    except ValueError as error:
        result["error"] = f"url_parse_error:{error}"
        return result
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        result["error"] = f"unsupported_scheme:{scheme or 'missing'}"
        return result
    if parsed.username is not None or parsed.password is not None:
        result["error"] = "userinfo_not_permitted"
        return result
    if not parsed.hostname:
        result["error"] = "missing_hostname"
        return result
    try:
        hostname = _normalise_hostname(parsed.hostname)
        port = parsed.port
    except (UnicodeError, ValueError) as error:
        result["error"] = f"invalid_hostname_or_port:{error}"
        return result
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    host_display = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host_display if port is None or default_port else f"{host_display}:{port}"

    path = parsed.path or "/"
    if settings.get("normalise_trailing_slash", True) and path != "/":
        without = path.rstrip("/")
        if without != path:
            result["trailing_slash_changed"] = True
            path = without or "/"

    exact_tracking = {
        str(item).casefold()
        for item in settings.get("tracking_parameter_names", [])
    }
    prefix_tracking = tuple(
        str(item).casefold()
        for item in settings.get("tracking_parameter_prefixes", [])
    )
    kept_parameters: list[tuple[str, str]] = []
    removed: list[str] = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        folded = key.casefold()
        if folded in exact_tracking or any(
            folded.startswith(prefix) for prefix in prefix_tracking
        ):
            removed.append(key)
        else:
            kept_parameters.append((key, value))
    query = urlencode(kept_parameters, doseq=True)
    fragment = "" if settings.get("remove_fragment", True) else parsed.fragment
    result["fragment_removed"] = bool(parsed.fragment and not fragment)
    result["tracking_parameters_removed"] = removed
    result["canonical_url"] = urlunsplit((scheme, netloc, path, query, fragment))
    result["status"] = "ok"
    return result


def canonicalisation_sanity_checks(settings: dict[str, Any]) -> dict[str, Any]:
    cases = [
        (
            "HTTPS://Example.COM/path/#section",
            "https://example.com/path",
        ),
        (
            "https://Example.com/?utm_source=x&id=7&fbclid=y",
            "https://example.com/?id=7",
        ),
        (
            "http://EXAMPLE.com:80/search?q=a+b&lang=en",
            "http://example.com/search?q=a+b&lang=en",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for raw, expected in cases:
        actual = canonicalize_url(raw, settings)
        second = canonicalize_url(actual["canonical_url"], settings)
        passed = (
            actual["status"] == "ok"
            and actual["canonical_url"] == expected
            and second["canonical_url"] == expected
        )
        rows.append(
            {
                "raw_url": raw,
                "expected": expected,
                "actual": actual["canonical_url"],
                "idempotent": second["canonical_url"] == actual["canonical_url"],
                "passed": passed,
            }
        )
    return {
        "case_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "cases": rows,
    }


def _index_unique(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Every row must have a non-empty {key}")
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    return result


def _response_id_for_index(index: int) -> str:
    return f"fcb_r{index:04d}"


def calculate_retrieval_split(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently recompute and assert the preregistered 121/468 split."""
    full = paths_for_scope(project_root, "full")
    gold = load_jsonl(full.gold_claims)
    manifest = load_jsonl(full.cohort_manifest)
    raw = load_jsonl(full.source_input)
    no_evidence_results = load_jsonl(full.no_evidence_output)
    oracle_results = load_jsonl(full.oracle_output)
    gold_by_id = _index_unique(gold, "claim_id")
    manifest_by_id = _index_unique(manifest, "claim_id")
    if set(gold_by_id) != set(manifest_by_id):
        raise ValueError("Gold and cohort manifest claim ID sets differ")

    split_config = config["split"]
    dev_raw_count = int(split_config["development_raw_response_count"])
    if dev_raw_count <= 0 or dev_raw_count >= len(raw):
        raise ValueError("development_raw_response_count must split the raw dataset")
    dev_source_ids = {
        _response_id_for_index(index) for index in range(1, dev_raw_count + 1)
    }
    heldout_source_ids = {
        _response_id_for_index(index)
        for index in range(dev_raw_count + 1, len(raw) + 1)
    }

    recomputed_ids: set[str] = set()
    binary_ids: set[str] = set()
    canonical_flag_ids: set[str] = set()
    manifest_flag_ids: set[str] = set()
    split_rows: list[dict[str, Any]] = []
    for record in gold:
        claim_id = record["claim_id"]
        manifest_record = manifest_by_id[claim_id]
        source_index = record.get("source_record_index")
        response_id = record.get("response_id")
        if not isinstance(source_index, int) or source_index < 1:
            raise ValueError(f"Invalid source_record_index for {claim_id}")
        if response_id != _response_id_for_index(source_index):
            raise ValueError(f"response/source index mismatch for {claim_id}")
        normalized_evidence = normalize_oracle_evidence(record.get("gold_evidence"))
        independently_matched = (
            record.get("human_label") in PRIMARY_LABELS
            and normalized_evidence["status"] == "ok"
        )
        if record.get("human_label") in PRIMARY_LABELS:
            binary_ids.add(claim_id)
        expected_manifest_fields = {
            "response_id": response_id,
            "human_label": record.get("human_label"),
            "is_binary_evaluable": record.get("human_label") in PRIMARY_LABELS,
            "oracle_evidence_available": normalized_evidence["status"] == "ok",
            "oracle_evidence_normalization_status": normalized_evidence["status"],
            "oracle_evidence_valid_item_count": normalized_evidence[
                "valid_item_count"
            ],
            "in_primary_matched_cohort": independently_matched,
        }
        manifest_mismatches = {
            key: {"gold_recomputed": expected, "manifest": manifest_record.get(key)}
            for key, expected in expected_manifest_fields.items()
            if manifest_record.get(key) != expected
        }
        if manifest_mismatches:
            raise ValueError(
                f"Gold/cohort manifest schema drift for {claim_id}: "
                f"{manifest_mismatches}"
            )
        if independently_matched:
            recomputed_ids.add(claim_id)
        if record.get("in_primary_matched_cohort") is True:
            canonical_flag_ids.add(claim_id)
        if manifest_record.get("in_primary_matched_cohort") is True:
            manifest_flag_ids.add(claim_id)
        if not independently_matched:
            continue
        split = "dev" if response_id in dev_source_ids else "heldout"
        if split == "heldout" and response_id not in heldout_source_ids:
            raise ValueError(f"Matched response outside raw split: {response_id}")
        split_rows.append(
            {
                "schema_version": RETRIEVAL_SCHEMA_VERSION,
                "split_definition_version": split_config["definition_version"],
                "claim_id": claim_id,
                "response_id": response_id,
                "source_record_index": source_index,
                "split": split,
                "has_usable_oracle_evidence": True,
                "in_primary_matched_cohort": True,
            }
        )

    if not (
        recomputed_ids == canonical_flag_ids == manifest_flag_ids
    ):
        raise ValueError(
            "Independently recomputed matched IDs disagree with canonical flags"
        )
    no_evidence_by_id = _index_unique(no_evidence_results, "claim_id")
    oracle_by_id = _index_unique(oracle_results, "claim_id")
    if set(no_evidence_by_id) != binary_ids:
        raise ValueError(
            "Full no-evidence result IDs do not equal the recomputed binary cohort"
        )
    if set(oracle_by_id) != recomputed_ids:
        raise ValueError(
            "Full oracle result IDs do not equal the recomputed strict matched cohort"
        )
    for setting, result_rows in (
        ("no_evidence", no_evidence_results),
        ("oracle_evidence", oracle_results),
    ):
        for result in result_rows:
            claim_id = result["claim_id"]
            gold_record = gold_by_id[claim_id]
            if result.get("response_id") != gold_record.get("response_id"):
                raise ValueError(f"{setting} response_id drift for {claim_id}")
            if result.get("setting") != setting or result.get("status") != "ok":
                raise ValueError(
                    f"Canonical {setting} result is not a successful fixed-setting "
                    f"row for {claim_id}"
                )
    split_by_claim = _index_unique(split_rows, "claim_id")
    dev_rows = [row for row in split_rows if row["split"] == "dev"]
    heldout_rows = [row for row in split_rows if row["split"] == "heldout"]
    dev_claim_ids = {row["claim_id"] for row in dev_rows}
    heldout_claim_ids = {row["claim_id"] for row in heldout_rows}
    dev_matched_response_ids = {row["response_id"] for row in dev_rows}
    heldout_matched_response_ids = {row["response_id"] for row in heldout_rows}
    if dev_claim_ids & heldout_claim_ids:
        raise ValueError("Dev and held-out claim IDs overlap")
    if dev_matched_response_ids & heldout_matched_response_ids:
        raise ValueError("Dev and held-out response IDs overlap")
    if len(split_by_claim) != len(recomputed_ids):
        raise ValueError("Split does not cover every matched claim exactly once")

    computed = {
        "total_matched_claims": len(split_rows),
        "development_matched_claims": len(dev_rows),
        "heldout_matched_claims": len(heldout_rows),
        "total_matched_responses": len(
            dev_matched_response_ids | heldout_matched_response_ids
        ),
        "development_matched_responses": len(dev_matched_response_ids),
        "heldout_matched_responses": len(heldout_matched_response_ids),
    }
    expected = {
        key.removeprefix("expected_"): int(value)
        for key, value in split_config.items()
        if key.startswith("expected_")
    }
    mismatches = {
        key: {"expected": value, "actual": computed.get(key)}
        for key, value in expected.items()
        if computed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{SPLIT_EXPECTATION_ERROR} {mismatches}")

    summary = {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "split_definition_version": split_config["definition_version"],
        "generated_at": utc_now(),
        "scope": "full",
        "definition": (
            "The first 20 raw responses define retrieval development; all "
            "remaining raw responses define primary held-out. Assignment is by "
            "response_id, never random claim sampling."
        ),
        "purpose": (
            "This split is preregistered before retrieval configuration selection; "
            "the 121 claims were not previously used to tune retrieval."
        ),
        "raw_source_response_count": len(raw),
        "development_source_response_count": len(dev_source_ids),
        "heldout_source_response_count": len(heldout_source_ids),
        **computed,
        "development_source_response_ids": sorted(dev_source_ids),
        "heldout_source_response_ids_sha256": canonical_json_hash(
            sorted(heldout_source_ids)
        ),
        "development_matched_response_ids": sorted(dev_matched_response_ids),
        "heldout_matched_response_ids": sorted(heldout_matched_response_ids),
        "development_claim_ids_sha256": canonical_json_hash(sorted(dev_claim_ids)),
        "heldout_claim_ids_sha256": canonical_json_hash(sorted(heldout_claim_ids)),
        "all_matched_claim_ids_sha256": canonical_json_hash(
            sorted(recomputed_ids)
        ),
        "claim_split_disjoint": not bool(dev_claim_ids & heldout_claim_ids),
        "response_split_disjoint": not bool(
            dev_matched_response_ids & heldout_matched_response_ids
        ),
        "canonical_flag_mismatch_count": 0,
        "canonical_prediction_result_audit": {
            "no_evidence_rows": len(no_evidence_results),
            "no_evidence_unique_claims": len(no_evidence_by_id),
            "no_evidence_matches_binary_cohort": True,
            "oracle_rows": len(oracle_results),
            "oracle_unique_claims": len(oracle_by_id),
            "oracle_matches_strict_matched_cohort": True,
            "paired_strict_matched_claims": len(recomputed_ids),
            "all_rows_status_ok": True,
        },
        "expected_counts": expected,
        "assertions_passed": True,
        "input_files": {
            "raw": {
                "path": project_relative(project_root, full.source_input),
                "sha256": sha256_file(full.source_input),
            },
            "gold_claims": {
                "path": project_relative(project_root, full.gold_claims),
                "sha256": sha256_file(full.gold_claims),
            },
            "cohort_manifest": {
                "path": project_relative(project_root, full.cohort_manifest),
                "sha256": sha256_file(full.cohort_manifest),
            },
            "no_evidence_results": {
                "path": project_relative(project_root, full.no_evidence_output),
                "sha256": sha256_file(full.no_evidence_output),
            },
            "oracle_results": {
                "path": project_relative(project_root, full.oracle_output),
                "sha256": sha256_file(full.oracle_output),
            },
        },
        "config_sha256": canonical_json_hash(config),
    }
    return split_rows, summary


def prepare_retrieval_splits(
    project_root: Path,
    paths: RetrievalPaths,
    config: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    rows, summary = calculate_retrieval_split(project_root, config)
    summary["artifact_sha256"] = canonical_json_hash(rows)
    if not dry_run:
        atomic_write_jsonl(paths.split_manifest, rows)
        atomic_write_json(paths.split_summary, summary)
    return summary


def _split_for_source_index(source_index: int, config: dict[str, Any]) -> str:
    cutoff = int(config["split"]["development_raw_response_count"])
    return "dev" if source_index <= cutoff else "heldout"


def _domain_for_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        return (urlsplit(url).hostname or "").lower() or None
    except ValueError:
        return None


def _evidence_manifest_markdown(report: dict[str, Any]) -> str:
    counts = report["evidence_inventory"]
    coverage = report["matched_claim_url_coverage"]
    canonical = report["url_canonicalisation"]
    return "\n".join(
        [
            "# Experiment A corpus preparation report",
            "",
            f"- Corpus: **{report['corpus_name']}**",
            f"- Evidence items: **{counts['evidence_item_count']}**",
            f"- Items with URL: **{counts['items_with_url']}**",
            f"- Items without URL: **{counts['items_without_url']}**",
            f"- Raw unique URL strings: **{counts['raw_unique_url_count']}**",
            f"- Raw domains: **{counts['raw_domain_count']}**",
            f"- Canonical URL candidates: **{canonical['canonical_unique_url_count']}**",
            f"- Invalid URL strings: **{canonical['invalid_url_count']}**",
            "",
            "## Strict matched claim URL coverage",
            "",
            "| Split | Claims | At least one URL | No URL |",
            "|---|---:|---:|---:|",
            f"| dev | {coverage['dev']['claim_count']} | {coverage['dev']['with_url']} | {coverage['dev']['without_url']} |",
            f"| heldout | {coverage['heldout']['claim_count']} | {coverage['heldout']['with_url']} | {coverage['heldout']['without_url']} |",
            f"| total | {coverage['total']['claim_count']} | {coverage['total']['with_url']} | {coverage['total']['without_url']} |",
            "",
            "Gold evidence text, stance, human label, and claim-specific URL mapping "
            "are evaluation-only. They must not enter retrieval queries or ranking.",
            "",
            "Source-document fetching, passage construction, and qrel mapping are "
            "reported separately and may remain pending until their explicit stages run.",
            "",
        ]
    )


def _existing_fetch_state_is_auditable(
    project_root: Path,
    existing_url: dict[str, Any],
    document_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> bool:
    """Use the fetcher's authoritative provenance check before preserving success."""
    if existing_url.get("fetch_status") != "success":
        return True
    doc_id = existing_url.get("doc_id")
    document = document_by_id.get(doc_id) if isinstance(doc_id, str) else None
    if not document or document.get("fetch_status") != "success":
        return False
    from factcheck_bench_corpus_fetch import (
        _canonical_json_sha256,
        _config_section,
        _fetch_pipeline_sha256,
        _verified_resume_record,
    )

    fetch_config_sha256 = _canonical_json_sha256(dict(_config_section(config)))
    requested_url = existing_url.get("canonical_url") or existing_url.get("raw_url")
    return _verified_resume_record(
        document,
        project_root,
        manifest_row=existing_url,
        requested_url=requested_url if isinstance(requested_url, str) else None,
        fetch_config_sha256=fetch_config_sha256,
        fetch_pipeline_sha256=_fetch_pipeline_sha256(fetch_config_sha256),
    )


def _url_manifest_preparation_projection(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Hash immutable URL preparation fields, excluding mutable fetch state."""
    fields = (
        "schema_version",
        "url_id",
        "raw_url",
        "raw_url_sha256",
        "canonical_url",
        "canonical_url_sha256",
        "canonicalisation_version",
        "canonicalisation_status",
        "canonicalisation_error",
        "canonicalisation_changes",
        "domain",
        "canonical_duplicate_group",
        "trace",
        "trace_usage",
    )
    return [{field: row.get(field) for field in fields} for row in rows]


def build_evidence_and_url_manifests(
    project_root: Path,
    paths: RetrievalPaths,
    config: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build evaluation-only evidence traces and unique raw-URL fetch rows."""
    full = paths_for_scope(project_root, "full")
    gold = load_jsonl(full.gold_claims)
    manifest = load_jsonl(full.cohort_manifest)
    manifest_by_id = _index_unique(manifest, "claim_id")
    split_rows, split_summary = calculate_retrieval_split(project_root, config)
    split_by_id = _index_unique(split_rows, "claim_id")
    settings = config["url_canonicalisation"]
    existing_url_rows = load_jsonl(paths.url_manifest, allow_missing=True)
    existing_url_by_raw = (
        _index_unique(existing_url_rows, "raw_url") if existing_url_rows else {}
    )
    existing_documents = load_jsonl(paths.documents, allow_missing=True)
    existing_document_by_id = (
        _index_unique(existing_documents, "doc_id") if existing_documents else {}
    )
    reset_stale_fetch_state_count = 0
    sanity = canonicalisation_sanity_checks(settings)
    if not sanity["all_passed"]:
        raise ValueError("URL canonicalisation sanity checks failed")

    evidence_rows: list[dict[str, Any]] = []
    raw_url_traces: dict[str, dict[str, Any]] = {}
    claim_has_url: dict[str, bool] = defaultdict(bool)
    for record in gold:
        claim_id = record["claim_id"]
        source_index = record.get("source_record_index")
        if not isinstance(source_index, int):
            raise ValueError(f"Missing source_record_index for {claim_id}")
        response_id = record.get("response_id")
        split = _split_for_source_index(source_index, config)
        matched = claim_id in split_by_id
        evidence_bundle = record.get("gold_evidence")
        if not isinstance(evidence_bundle, list):
            raise TypeError(f"gold_evidence must be a list for {claim_id}")
        for index, item in enumerate(evidence_bundle, start=1):
            if not isinstance(item, dict):
                raise TypeError(f"Evidence {claim_id}/{index} must be an object")
            evidence_id = f"{claim_id}_e{index:03d}"
            text, exclusion_reason = normalize_evidence_text(item.get("text"))
            raw_url = (
                item.get("url").strip()
                if isinstance(item.get("url"), str) and item.get("url").strip()
                else None
            )
            canonical = canonicalize_url(raw_url, settings)
            if raw_url:
                claim_has_url[claim_id] = True
            row = {
                "schema_version": RETRIEVAL_SCHEMA_VERSION,
                "evidence_id": evidence_id,
                "evidence_index": index,
                "claim_id": claim_id,
                "response_id": response_id,
                "source_record_index": source_index,
                "split": split,
                "in_primary_matched_cohort": matched,
                "raw_url": raw_url,
                "canonical_url": canonical["canonical_url"],
                "canonicalisation_status": canonical["status"],
                "canonicalisation_error": canonical["error"],
                "raw_url_sha256": sha256_text(raw_url) if raw_url else None,
                "canonical_url_sha256": (
                    sha256_text(canonical["canonical_url"])
                    if canonical["canonical_url"]
                    else None
                ),
                "gold_evidence_text": text,
                "gold_stance": item.get("stance"),
                "human_label": record.get("human_label"),
                "usable_text": text is not None,
                "unusable_text_reason": exclusion_reason,
                "source_metadata": {
                    "source": item.get("source"),
                    "rank": item.get("rank"),
                    "original_item_keys": sorted(item),
                },
                "field_policy": {
                    "gold_evidence_text": "evaluation_only",
                    "gold_stance": "evaluation_only",
                    "human_label": "evaluation_only",
                    "raw_url": "evaluation_only_claim_url_mapping",
                    "canonical_url": "evaluation_only_claim_url_mapping",
                },
            }
            evidence_rows.append(row)
            if not raw_url:
                continue
            trace = raw_url_traces.setdefault(
                raw_url,
                {
                    "evidence_ids": set(),
                    "claim_ids": set(),
                    "response_ids": set(),
                    "splits": set(),
                    "matched_evidence_ids": set(),
                    "canonical": canonical,
                },
            )
            trace["evidence_ids"].add(evidence_id)
            trace["claim_ids"].add(claim_id)
            trace["response_ids"].add(response_id)
            trace["splits"].add(split)
            if matched:
                trace["matched_evidence_ids"].add(evidence_id)

    url_rows: list[dict[str, Any]] = []
    for raw_url in sorted(raw_url_traces):
        trace = raw_url_traces[raw_url]
        canonical = trace["canonical"]
        raw_hash = sha256_text(raw_url)
        canonical_url = canonical["canonical_url"]
        canonical_hash = sha256_text(canonical_url) if canonical_url else None
        url_row = {
            "schema_version": RETRIEVAL_SCHEMA_VERSION,
                "url_id": f"url_{raw_hash[:20]}",
                "raw_url": raw_url,
                "raw_url_sha256": raw_hash,
                "canonical_url": canonical_url,
                "canonical_url_sha256": canonical_hash,
                "canonicalisation_version": settings["version"],
                "canonicalisation_status": canonical["status"],
                "canonicalisation_error": canonical["error"],
                "canonicalisation_changes": {
                    "fragment_removed": canonical["fragment_removed"],
                    "tracking_parameters_removed": canonical[
                        "tracking_parameters_removed"
                    ],
                    "trailing_slash_changed": canonical[
                        "trailing_slash_changed"
                    ],
                },
                "domain": _domain_for_url(canonical_url or raw_url),
                "canonical_duplicate_group": (
                    f"canonical_{canonical_hash[:20]}" if canonical_hash else None
                ),
                "trace": {
                    "evidence_ids": sorted(trace["evidence_ids"]),
                    "claim_ids": sorted(trace["claim_ids"]),
                    "response_ids": sorted(trace["response_ids"]),
                    "splits": sorted(trace["splits"]),
                    "matched_evidence_ids": sorted(trace["matched_evidence_ids"]),
                },
                "trace_usage": "evaluation_only_gold_url_mapping",
                "fetch_status": "pending",
                "fetch_attempts": 0,
                "fetched_at": None,
                "status_code": None,
                "content_type": None,
                "redirect_chain": [],
                "final_url": None,
                "doc_id": None,
                "raw_path": None,
                "raw_content_sha256": None,
                "content_hash": None,
                "content_duplicate_group": None,
            "error": None,
        }
        existing = existing_url_by_raw.get(raw_url)
        preserve_existing = bool(
            existing
            and existing.get("canonical_url") == canonical_url
            and _existing_fetch_state_is_auditable(
                project_root, existing, existing_document_by_id, config
            )
        )
        if existing and not preserve_existing and existing.get("fetch_status") == "success":
            reset_stale_fetch_state_count += 1
        if preserve_existing:
            immutable_preparation_fields = {
                "schema_version",
                "url_id",
                "raw_url",
                "raw_url_sha256",
                "canonical_url",
                "canonical_url_sha256",
                "canonicalisation_version",
                "canonicalisation_status",
                "canonicalisation_error",
                "canonicalisation_changes",
                "domain",
                "canonical_duplicate_group",
                "trace",
                "trace_usage",
            }
            for field, value in existing.items():
                if field not in immutable_preparation_fields:
                    url_row[field] = value
        url_rows.append(url_row)

    canonical_urls = [
        row["canonical_url"]
        for row in url_rows
        if row["canonicalisation_status"] == "ok"
    ]
    url_domain_values = [row["domain"] for row in url_rows if row["domain"]]
    evidence_source_counts = Counter(
        row["source_metadata"]["source"] for row in evidence_rows
    )
    stance_counts = Counter(row["gold_stance"] for row in evidence_rows)
    fetch_status_counts = Counter(row["fetch_status"] for row in url_rows)
    usable_reason_counts = Counter(
        row["unusable_text_reason"]
        for row in evidence_rows
        if row["unusable_text_reason"]
    )

    matched_claims_by_split: dict[str, list[str]] = {"dev": [], "heldout": []}
    for row in split_rows:
        matched_claims_by_split[row["split"]].append(row["claim_id"])

    def coverage(split: str) -> dict[str, int]:
        claims = matched_claims_by_split[split]
        with_url = sum(claim_has_url[claim_id] for claim_id in claims)
        return {
            "claim_count": len(claims),
            "with_url": with_url,
            "without_url": len(claims) - with_url,
        }

    dev_coverage = coverage("dev")
    heldout_coverage = coverage("heldout")
    total_coverage = {
        key: dev_coverage[key] + heldout_coverage[key]
        for key in dev_coverage
    }
    report = {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "scope": "full",
        "corpus_name": config["corpus"]["name"],
        "corpus_definition": (
            "A benchmark-grounded closed collection built from every source URL "
            "present in canonical FactCheck-Bench gold evidence. It is not "
            "unrestricted open-web retrieval."
        ),
        "evidence_inventory": {
            "evidence_item_count": len(evidence_rows),
            "usable_text_count": sum(row["usable_text"] for row in evidence_rows),
            "unusable_text_count": sum(
                not row["usable_text"] for row in evidence_rows
            ),
            "unusable_text_reason_counts": dict(sorted(usable_reason_counts.items())),
            "items_with_url": sum(row["raw_url"] is not None for row in evidence_rows),
            "items_without_url": sum(
                row["raw_url"] is None for row in evidence_rows
            ),
            "raw_unique_url_count": len(url_rows),
            "raw_domain_count": len(set(url_domain_values)),
            "source_counts": dict(sorted(evidence_source_counts.items())),
            "stance_counts": dict(sorted(stance_counts.items())),
        },
        "matched_evidence_inventory": {
            "evidence_item_count": sum(
                row["in_primary_matched_cohort"] for row in evidence_rows
            ),
            "items_with_url": sum(
                row["in_primary_matched_cohort"] and row["raw_url"] is not None
                for row in evidence_rows
            ),
            "raw_unique_url_count": len(
                {
                    row["raw_url"]
                    for row in evidence_rows
                    if row["in_primary_matched_cohort"] and row["raw_url"]
                }
            ),
        },
        "matched_claim_url_coverage": {
            "dev": dev_coverage,
            "heldout": heldout_coverage,
            "total": total_coverage,
        },
        "url_canonicalisation": {
            "version": settings["version"],
            "raw_unique_url_count": len(url_rows),
            "valid_url_count": len(canonical_urls),
            "invalid_url_count": len(url_rows) - len(canonical_urls),
            "canonical_unique_url_count": len(set(canonical_urls)),
            "canonical_duplicate_raw_url_count": len(canonical_urls)
            - len(set(canonical_urls)),
            "fragments_removed_count": sum(
                row["canonicalisation_changes"]["fragment_removed"]
                for row in url_rows
            ),
            "tracking_parameters_removed_count": sum(
                len(row["canonicalisation_changes"]["tracking_parameters_removed"])
                for row in url_rows
            ),
            "trailing_slash_changed_count": sum(
                row["canonicalisation_changes"]["trailing_slash_changed"]
                for row in url_rows
            ),
            "idempotence_failure_count": sum(
                canonicalize_url(row["canonical_url"], settings)["canonical_url"]
                != row["canonical_url"]
                for row in url_rows
                if row["canonical_url"]
            ),
            "sanity_checks": sanity,
        },
        "split_summary": {
            key: split_summary[key]
            for key in (
                "total_matched_claims",
                "development_matched_claims",
                "heldout_matched_claims",
                "total_matched_responses",
                "development_matched_responses",
                "heldout_matched_responses",
                "development_source_response_count",
                "heldout_source_response_count",
            )
        },
        "artifact_counts": {
            "evidence_manifest_rows": len(evidence_rows),
            "url_manifest_rows": len(url_rows),
        },
        "artifact_hashes": {
            "evidence_manifest_canonical_sha256": canonical_json_hash(
                evidence_rows
            ),
            "url_manifest_preparation_sha256": canonical_json_hash(
                _url_manifest_preparation_projection(url_rows)
            ),
        },
        "input_hashes": {
            "gold_claims_sha256": sha256_file(full.gold_claims),
            "cohort_manifest_sha256": sha256_file(full.cohort_manifest),
            "config_sha256": canonical_json_hash(config),
        },
        "field_policy": config["leakage_policy"],
        "fetch_status": (
            "pending_until_fetch_corpus_stage"
            if fetch_status_counts == {"pending": len(url_rows)}
            else "preserved_from_existing_url_manifest"
        ),
        "fetch_status_counts": dict(sorted(fetch_status_counts.items())),
        "stale_success_rows_reset_to_pending": reset_stale_fetch_state_count,
    }
    canonical_report = {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "generated_at": report["generated_at"],
        **report["url_canonicalisation"],
        "invalid_urls": [
            {
                "url_id": row["url_id"],
                "raw_url": row["raw_url"],
                "error": row["canonicalisation_error"],
            }
            for row in url_rows
            if row["canonicalisation_status"] != "ok"
        ],
    }
    if not dry_run:
        atomic_write_jsonl(paths.evidence_manifest, evidence_rows)
        atomic_write_jsonl(paths.url_manifest, url_rows)
        atomic_write_json(paths.preparation_report_json, report)
        atomic_write_text(
            paths.preparation_report_markdown,
            _evidence_manifest_markdown(report),
        )
        atomic_write_json(paths.canonicalisation_report, canonical_report)
    return report


def token_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


def _boundary_token_indices(
    text: str, tokens: list[tuple[int, int]]
) -> set[int]:
    """Return token-end indices that coincide with paragraph/sentence boundaries."""
    boundary_chars = {match.start() for match in re.finditer(r"\n\s*\n", text)}
    boundary_chars.update(
        match.start() for match in SENTENCE_BOUNDARY_RE.finditer(text)
    )
    boundaries: set[int] = set()
    token_index = 0
    for boundary in sorted(boundary_chars):
        while token_index < len(tokens) and tokens[token_index][1] <= boundary:
            token_index += 1
        if token_index > 0:
            boundaries.add(token_index)
    return boundaries


def _span_metadata(
    blocks: list[dict[str, Any]], char_start: int, char_end: int
) -> dict[str, Any]:
    overlapping = [
        block
        for block in blocks
        if isinstance(block.get("char_start"), int)
        and isinstance(block.get("char_end"), int)
        and block["char_end"] > char_start
        and block["char_start"] < char_end
    ]
    sections = [
        str(block["section"])
        for block in overlapping
        if isinstance(block.get("section"), str) and block["section"].strip()
    ]
    pages = sorted(
        {
            int(block["page"])
            for block in overlapping
            if isinstance(block.get("page"), int)
        }
    )
    return {
        "section": sections[0] if sections else None,
        "page": pages[0] if len(pages) == 1 else None,
        "page_start": pages[0] if pages else None,
        "page_end": pages[-1] if pages else None,
    }


def chunk_document(
    document: dict[str, Any],
    *,
    chunk_size: int,
    chunk_overlap: int,
    minimum_boundary_fraction: float,
    chunking_version: str = "ad_hoc_boundary_chunking_v1",
    chunk_config_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic boundary-aware passages from one clean document."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >=0 and smaller than chunk_size")
    if not 0 < minimum_boundary_fraction <= 1:
        raise ValueError("minimum_boundary_fraction must be in (0, 1]")
    doc_id = document.get("doc_id")
    text = document.get("clean_text")
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError("Document requires a non-empty doc_id")
    if not isinstance(text, str) or not text.strip():
        return []
    chunk_spec = {
        "chunking_version": chunking_version,
        "tokenizer": TOKEN_RE.pattern,
        "chunk_size_tokens": chunk_size,
        "chunk_overlap_tokens": chunk_overlap,
        "minimum_boundary_fraction": minimum_boundary_fraction,
    }
    chunk_fingerprint = chunk_config_fingerprint or canonical_json_hash(chunk_spec)
    tokens = token_spans(text)
    if not tokens:
        return []
    boundaries = _boundary_token_indices(text, tokens)
    blocks = document.get("content_blocks")
    if not isinstance(blocks, list):
        blocks = []

    passages: list[dict[str, Any]] = []
    start_token = 0
    while start_token < len(tokens):
        target_end = min(len(tokens), start_token + chunk_size)
        end_token = target_end
        if target_end < len(tokens):
            minimum = start_token + max(
                1, math.ceil(chunk_size * minimum_boundary_fraction)
            )
            candidates = [
                boundary
                for boundary in boundaries
                if minimum <= boundary <= target_end
            ]
            if candidates:
                end_token = max(candidates)
        if end_token <= start_token:
            end_token = target_end

        char_start = tokens[start_token][0]
        char_end = tokens[end_token - 1][1]
        while char_start < char_end and text[char_start].isspace():
            char_start += 1
        while char_end > char_start and text[char_end - 1].isspace():
            char_end -= 1
        passage_text = text[char_start:char_end]
        metadata = _span_metadata(blocks, char_start, char_end)
        passage_index = len(passages) + 1
        passages.append(
            {
                "schema_version": RETRIEVAL_SCHEMA_VERSION,
                "passage_id": (
                    f"{doc_id}_ch{chunk_fingerprint[:12]}_p{passage_index:05d}"
                ),
                "doc_id": doc_id,
                "canonical_url": document.get("canonical_url"),
                "final_url": document.get("final_url"),
                "title": document.get("title"),
                "section": metadata["section"],
                "text": passage_text,
                "token_count": end_token - start_token,
                "token_start": start_token,
                "token_end": end_token,
                "char_start": char_start,
                "char_end": char_end,
                "page": metadata["page"],
                "page_start": metadata["page_start"],
                "page_end": metadata["page_end"],
                "chunk_config_fingerprint": chunk_fingerprint,
            }
        )
        if end_token >= len(tokens):
            break
        next_start = max(start_token + 1, end_token - chunk_overlap)
        start_token = next_start
    return passages


def _successful_document_integrity_errors(
    project_root: Path, document: dict[str, Any]
) -> list[str]:
    """Validate frozen bytes and extracted text before blessing passages."""
    errors: list[str] = []
    clean_text = document.get("clean_text")
    clean_hash = document.get("clean_text_sha256")
    content_hash = document.get("content_hash")
    if not isinstance(clean_text, str) or not clean_text.strip():
        errors.append("clean_text_missing")
    elif not isinstance(clean_hash, str) or sha256_text(clean_text) != clean_hash:
        errors.append("clean_text_sha256_mismatch")
    if isinstance(clean_hash, str) and content_hash != clean_hash:
        errors.append("content_hash_mismatch")
    raw_path_value = document.get("raw_path")
    expected_raw_hash = document.get("raw_sha256") or document.get(
        "raw_content_sha256"
    )
    if not isinstance(raw_path_value, str) or not isinstance(expected_raw_hash, str):
        errors.append("raw_provenance_missing")
    else:
        raw_path = Path(raw_path_value)
        if not raw_path.is_absolute():
            raw_path = project_root / raw_path
        try:
            raw_path.resolve().relative_to(project_root.resolve())
        except ValueError:
            errors.append("raw_path_outside_project")
        else:
            if not raw_path.is_file():
                errors.append("raw_file_missing")
            elif sha256_file(raw_path) != expected_raw_hash:
                errors.append("raw_sha256_mismatch")
    if not isinstance(document.get("fetch_config_sha256"), str):
        errors.append("fetch_config_fingerprint_missing")
    if not isinstance(document.get("extractor_version"), str):
        errors.append("extractor_version_missing")
    return errors


def build_passages(
    project_root: Path,
    paths: RetrievalPaths,
    config: dict[str, Any],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    limit: int | None = None,
    artifact_namespace: str = "canonical",
    dry_run: bool = False,
) -> dict[str, Any]:
    if artifact_namespace not in {"canonical", "smoke"}:
        raise ValueError("artifact_namespace must be canonical or smoke")
    output_passages = (
        paths.passages
        if artifact_namespace == "canonical"
        else paths.root / "smoke" / "passages.jsonl"
    )
    output_report = (
        paths.passage_build_report
        if artifact_namespace == "canonical"
        else paths.root / "smoke" / "reports" / "passage_build_report.json"
    )
    if limit is not None and artifact_namespace == "canonical" and not dry_run:
        raise ValueError(
            "A limited passage build may not overwrite canonical passages.jsonl; "
            "use artifact_namespace='smoke'."
        )
    documents = load_jsonl(paths.documents)
    document_by_id = _index_unique(documents, "doc_id")
    url_rows = load_jsonl(paths.url_manifest)
    url_by_manifest_id = _index_unique(url_rows, "url_id")
    for document in documents:
        if document.get("fetch_status") != "success":
            continue
        if document.get("orphaned_from_url_manifest") is True:
            continue
        integrity_errors = _successful_document_integrity_errors(
            project_root, document
        )
        if integrity_errors:
            raise ValueError(
                f"Successful document failed frozen-artifact validation: "
                f"{document['doc_id']} ({', '.join(integrity_errors)})"
            )
        manifest_id = document.get("source_url_manifest_id")
        manifest_row = (
            url_by_manifest_id.get(manifest_id)
            if isinstance(manifest_id, str)
            else None
        )
        if (
            manifest_row is None
            or manifest_row.get("doc_id") != document.get("doc_id")
            or not _existing_fetch_state_is_auditable(
                project_root, manifest_row, document_by_id, config
            )
        ):
            raise ValueError(
                "Successful document is stale relative to the current URL/fetch/"
                f"extractor configuration: {document['doc_id']}"
            )
    for document in documents:
        alias = document.get("duplicate_of_doc_id")
        if not alias:
            continue
        target = document_by_id.get(alias)
        if target is None or target.get("duplicate_of_doc_id"):
            raise ValueError(
                f"Document duplicate aliases must be flat and target an existing "
                f"primary: {document['doc_id']} -> {alias}"
            )
        if target.get("fetch_status") != "success":
            raise ValueError(f"Duplicate primary is not successful: {alias}")
    chunk_config = config["chunking"]
    resolved_size = int(
        chunk_size
        if chunk_size is not None
        else chunk_config["chunk_size_tokens"]
    )
    resolved_overlap = int(
        chunk_overlap
        if chunk_overlap is not None
        else chunk_config["chunk_overlap_tokens"]
    )
    minimum_fraction = float(chunk_config["minimum_boundary_fraction"])
    chunk_spec = {
        "chunking_version": chunk_config["version"],
        "tokenizer": chunk_config["tokenizer"],
        "chunk_size_tokens": resolved_size,
        "chunk_overlap_tokens": resolved_overlap,
        "minimum_boundary_fraction": minimum_fraction,
    }
    chunk_config_fingerprint = canonical_json_hash(chunk_spec)
    all_successful_documents = [
        row
        for row in documents
        if row.get("fetch_status") == "success"
        and isinstance(row.get("clean_text"), str)
        and row["clean_text"].strip()
    ]
    successful_documents = [
        row
        for row in all_successful_documents
        if not row.get("duplicate_of_doc_id")
        and row.get("orphaned_from_url_manifest") is not True
    ]
    successful_documents.sort(key=lambda row: row["doc_id"])
    selected = successful_documents[:limit] if limit is not None else successful_documents
    passage_rows = [
        passage
        for document in selected
        for passage in chunk_document(
            document,
            chunk_size=resolved_size,
            chunk_overlap=resolved_overlap,
            minimum_boundary_fraction=minimum_fraction,
            chunking_version=chunk_config["version"],
            chunk_config_fingerprint=chunk_config_fingerprint,
        )
    ]
    passage_ids = [row["passage_id"] for row in passage_rows]
    if len(passage_ids) != len(set(passage_ids)):
        raise ValueError("Duplicate stable passage IDs generated")
    index_field_allowlist = set(
        config["leakage_policy"]["passage_index_field_allowlist"]
    )
    observed_passage_fields = {
        field for row in passage_rows for field in row
    }
    unexpected_index_fields = observed_passage_fields - index_field_allowlist
    if unexpected_index_fields:
        raise ValueError(
            "Passage artifact contains fields outside the frozen index allowlist: "
            f"{sorted(unexpected_index_fields)}"
        )
    forbidden_fields = set(config["leakage_policy"]["forbidden_ranking_inputs"])
    if observed_passage_fields & forbidden_fields:
        raise ValueError("Evaluation-only fields leaked into passage artifacts")
    pending_document_count = sum(
        row.get("fetch_status") in {"pending", "stale_pending"}
        for row in documents
    )
    if len(selected) != len(successful_documents):
        build_status = "partial_limit"
    elif pending_document_count:
        build_status = "partial_source_fetch"
    else:
        build_status = "complete"
    report = {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": build_status,
        "document_record_count": len(documents),
        "pending_document_count": pending_document_count,
        "successful_document_count": len(all_successful_documents),
        "primary_document_count_after_deduplication": len(successful_documents),
        "duplicate_documents_excluded_from_passages": len(
            all_successful_documents
        )
        - len(successful_documents),
        "selected_document_count": len(selected),
        "passage_count": len(passage_rows),
        "documents_with_passages": len(
            {row["doc_id"] for row in passage_rows}
        ),
        "documents_without_passages": len(selected)
        - len({row["doc_id"] for row in passage_rows}),
        "chunking": {
            "version": chunk_config["version"],
            "tokenizer": chunk_config["tokenizer"],
            "chunk_size_tokens": resolved_size,
            "chunk_overlap_tokens": resolved_overlap,
            "minimum_boundary_fraction": minimum_fraction,
            "fingerprint": chunk_config_fingerprint,
        },
        "artifact_namespace": artifact_namespace,
        "output_passages": project_relative(project_root, output_passages),
        "index_field_allowlist": sorted(index_field_allowlist),
        "input_documents_sha256": sha256_file(paths.documents),
        "passages_canonical_sha256": canonical_json_hash(passage_rows),
        "config_sha256": canonical_json_hash(config),
    }
    if not dry_run:
        atomic_write_jsonl(output_passages, passage_rows)
        report["passages_file_sha256"] = sha256_file(output_passages)
        atomic_write_json(output_report, report)
    return report


def alignment_normalize(value: str) -> str:
    text = html.unescape(unicodedata.normalize("NFKC", value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _lexical_alignment_score(evidence: str, passage: str) -> float:
    evidence_tokens = alignment_normalize(evidence).split()
    passage_tokens = alignment_normalize(passage).split()
    if not evidence_tokens or not passage_tokens:
        return 0.0
    evidence_counts = Counter(evidence_tokens)
    passage_counts = Counter(passage_tokens)
    overlap = sum(
        min(count, passage_counts[token])
        for token, count in evidence_counts.items()
    )
    precision = overlap / len(passage_tokens)
    recall = overlap / len(evidence_tokens)
    token_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    sequence = SequenceMatcher(
        None,
        alignment_normalize(evidence),
        alignment_normalize(passage),
        autojunk=False,
    ).ratio()
    containment = recall
    return max(containment * 0.8 + token_f1 * 0.2, sequence)


def _sentence_substring_score(evidence: str, passage: str) -> float:
    evidence_sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", evidence)
        if sentence.strip()
    ]
    normalized_passage = alignment_normalize(passage)
    eligible = [
        alignment_normalize(sentence)
        for sentence in evidence_sentences
        if len(alignment_normalize(sentence).split()) >= 4
    ]
    if not eligible:
        return 0.0
    matched = [sentence for sentence in eligible if sentence in normalized_passage]
    return len(matched) / len(eligible)


def _relevance_for_stance(stance: Any) -> tuple[int, bool]:
    normalized = str(stance).strip().casefold() if stance is not None else ""
    if normalized in {"completely-support", "refute", "human-validated"}:
        return 2, normalized == "human-validated"
    if normalized == "partially-support":
        return 1, False
    return 1, True


def _best_alignment_candidates(
    evidence: dict[str, Any],
    passages: list[dict[str, Any]],
    alignment_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    text = evidence["gold_evidence_text"]
    normalized_evidence = alignment_normalize(text)
    exact = [
        {
            "passage": passage,
            "method": "exact_normalized",
            "score": 1.0,
            "audit_status": "auto_accepted",
            "positive_qrel": True,
        }
        for passage in passages
        if normalized_evidence
        and normalized_evidence in alignment_normalize(passage["text"])
    ]
    if exact:
        return exact, "successfully_mapped"

    sentence_candidates: list[dict[str, Any]] = []
    auto_threshold = float(
        alignment_config.get("sentence_substring_auto_accept_threshold", 1.0)
    )
    candidate_threshold = float(
        alignment_config.get("sentence_substring_candidate_threshold", 0.5)
    )
    for passage in passages:
        score = _sentence_substring_score(text, passage["text"])
        if score >= candidate_threshold:
            auto_accepted = score >= auto_threshold
            sentence_candidates.append(
                {
                    "passage": passage,
                    "method": (
                        "sentence_substring"
                        if auto_accepted
                        else "sentence_substring_candidate"
                    ),
                    "score": score,
                    "audit_status": (
                        "auto_accepted"
                        if auto_accepted
                        else "needs_manual_review"
                    ),
                    "positive_qrel": auto_accepted,
                }
            )
    if sentence_candidates:
        maximum = max(row["score"] for row in sentence_candidates)
        best = [row for row in sentence_candidates if row["score"] == maximum]
        if all(row["positive_qrel"] for row in best):
            return best, "successfully_mapped"
        if len(best) > 1:
            for row in best:
                row["audit_status"] = "ambiguous_needs_manual_review"
            return best, "ambiguous"
        return best, "sentence_substring_candidate_needs_manual_review"

    scored = sorted(
        (
            {
                "passage": passage,
                "method": "fuzzy_lexical_candidate",
                "score": _lexical_alignment_score(text, passage["text"]),
                "audit_status": "needs_manual_review",
                "positive_qrel": False,
            }
            for passage in passages
        ),
        key=lambda row: (-row["score"], row["passage"]["passage_id"]),
    )
    threshold = float(alignment_config["fuzzy_candidate_threshold"])
    candidates = [row for row in scored if row["score"] >= threshold]
    candidates = candidates[: int(alignment_config["maximum_fuzzy_candidates"])]
    if not candidates:
        return [], "document_available_but_evidence_unmatched"
    top = candidates[0]["score"]
    margin = float(alignment_config["fuzzy_ambiguity_margin"])
    near_top = [row for row in candidates if top - row["score"] <= margin]
    if len(near_top) > 1:
        for row in near_top:
            row["audit_status"] = "ambiguous_needs_manual_review"
        return near_top, "ambiguous"
    return [candidates[0]], "fuzzy_candidate_needs_manual_review"


def _qrels_markdown(report: dict[str, Any]) -> str:
    counts = report["mapping_status_counts"]
    lines = [
        "# Gold evidence-to-passage mapping QA",
        "",
        f"- Status: **{report['status']}**",
        f"- Selected matched claims: **{report['selected_matched_claim_count']}**",
        f"- Selected usable evidence items: **{report['selected_usable_evidence_count']}**",
        f"- Positive qrel pairs: **{report['positive_qrel_pair_count']}**",
        f"- Audit-only candidate pairs: **{report['audit_candidate_pair_count']}**",
        "",
        "| Mapping outcome | Evidence items |",
        "|---|---:|",
    ]
    lines.extend(f"| `{key}` | {value} |" for key, value in sorted(counts.items()))
    lines.extend(
        [
            "",
            "Fuzzy lexical and partial multi-sentence substring candidates are "
            "never promoted to positive qrels without manual audit. Support and "
            "refute evidence may both receive relevance 2; partial support receives "
            "relevance 1.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_current_passage_artifact(
    paths: RetrievalPaths,
    passage_input: Path,
    document_input: Path,
    passage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Refuse qrels construction from stale or unproven passages."""
    report_path = (
        paths.passage_build_report
        if passage_input == paths.passages
        else passage_input.parent / "reports" / "passage_build_report.json"
    )
    report = load_json(report_path)
    if report.get("input_documents_sha256") != sha256_file(document_input):
        raise ValueError(
            "Passages are stale: their report does not match current "
            "documents.jsonl. Re-run build-passages."
        )
    if report.get("passages_canonical_sha256") != canonical_json_hash(
        passage_rows
    ):
        raise ValueError(
            "Passages are stale or modified: their canonical hash does not "
            "match the passage build report. Re-run build-passages."
        )
    fingerprint = report.get("chunking", {}).get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Passage build report is missing a chunk fingerprint")
    row_fingerprints = {
        row.get("chunk_config_fingerprint") for row in passage_rows
    }
    if passage_rows and row_fingerprints != {fingerprint}:
        raise ValueError(
            "Passage rows do not share the chunk fingerprint recorded in the "
            "passage build report."
        )
    return report


def build_qrels(
    project_root: Path,
    paths: RetrievalPaths,
    config: dict[str, Any],
    *,
    split_scope: str = "dev",
    confirm_config_frozen: bool = False,
    limit: int | None = None,
    artifact_namespace: str = "canonical",
    input_passages: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if split_scope not in {"dev", "heldout", "all"}:
        raise ValueError("split_scope must be dev, heldout, or all")
    if split_scope in {"heldout", "all"} and not confirm_config_frozen:
        raise ValueError(
            "Held-out qrels are sealed until retrieval/chunk configuration is "
            "frozen; pass confirm_config_frozen=True only for the one-time "
            "held-out evaluation or the secondary all-cohort analysis."
        )
    if artifact_namespace not in {"canonical", "smoke"}:
        raise ValueError("artifact_namespace must be canonical or smoke")
    if limit is not None and artifact_namespace == "canonical" and not dry_run:
        raise ValueError(
            "A limited qrels build may not overwrite canonical artifacts; use "
            "artifact_namespace='smoke'."
        )
    passage_input = input_passages or paths.passages
    if artifact_namespace == "smoke":
        output_root = paths.root / "smoke"
        output_reports = output_root / "reports"
        qrels_jsonl = output_root / f"qrels_{split_scope}.jsonl"
        qrels_tsv = output_root / f"qrels_{split_scope}.tsv"
        mapping_audit_path = output_reports / f"qrels_{split_scope}_mapping_audit.jsonl"
        mapping_report_json = output_reports / f"qrels_{split_scope}_mapping_report.json"
        mapping_report_markdown = output_reports / f"qrels_{split_scope}_mapping_report.md"
    elif split_scope == "dev":
        qrels_jsonl = paths.qrels_dev_jsonl
        qrels_tsv = paths.qrels_dev_tsv
        mapping_audit_path = paths.qrels_dev_mapping_audit
        mapping_report_json = paths.qrels_dev_mapping_report_json
        mapping_report_markdown = paths.qrels_dev_mapping_report_markdown
    elif split_scope == "heldout":
        qrels_jsonl = paths.qrels_heldout_jsonl
        qrels_tsv = paths.qrels_heldout_tsv
        mapping_audit_path = paths.qrels_heldout_mapping_audit
        mapping_report_json = paths.qrels_heldout_mapping_report_json
        mapping_report_markdown = paths.qrels_heldout_mapping_report_markdown
    else:
        qrels_jsonl = paths.qrels_jsonl
        qrels_tsv = paths.qrels_tsv
        mapping_audit_path = paths.qrels_mapping_audit
        mapping_report_json = paths.qrels_mapping_report_json
        mapping_report_markdown = paths.qrels_mapping_report_markdown
    evidence_rows = load_jsonl(paths.evidence_manifest)
    url_rows = load_jsonl(paths.url_manifest)
    document_rows = load_jsonl(paths.documents)
    passage_rows = load_jsonl(passage_input)
    _index_unique(evidence_rows, "evidence_id")
    url_by_raw = _index_unique(url_rows, "raw_url")
    document_by_id = _index_unique(document_rows, "doc_id")
    _index_unique(passage_rows, "passage_id")
    for document in document_rows:
        alias = document.get("duplicate_of_doc_id")
        if not alias:
            continue
        target = document_by_id.get(alias)
        if target is None:
            raise ValueError(
                f"Duplicate alias target is missing for {document['doc_id']}: {alias}"
            )
        if target.get("duplicate_of_doc_id"):
            raise ValueError(
                f"Duplicate alias chain is not allowed: {document['doc_id']} -> "
                f"{alias} -> {target.get('duplicate_of_doc_id')}"
            )
        if target.get("fetch_status") != "success":
            raise ValueError(
                f"Duplicate alias target is not a successful document: {alias}"
            )
    passage_build_report = _validate_current_passage_artifact(
        paths, passage_input, paths.documents, passage_rows
    )
    if split_scope in {"heldout", "all"} and (
        artifact_namespace != "canonical"
        or passage_input != paths.passages
        or passage_build_report.get("status") != "complete"
    ):
        raise ValueError(
            "Held-out/all qrels require the complete canonical passage artifact "
            "after every URL row reached a terminal fetch state."
        )
    passages_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for passage in passage_rows:
        passages_by_doc[passage["doc_id"]].append(passage)

    all_matched_claim_ids = sorted(
        {
            row["claim_id"]
            for row in evidence_rows
            if row.get("in_primary_matched_cohort") is True
        }
    )
    matched_claim_ids = sorted(
        {
            row["claim_id"]
            for row in evidence_rows
            if row.get("in_primary_matched_cohort") is True
            and (split_scope == "all" or row.get("split") == split_scope)
        }
    )
    selected_claim_ids = set(
        matched_claim_ids[:limit] if limit is not None else matched_claim_ids
    )
    selected_evidence = [
        row
        for row in evidence_rows
        if row.get("claim_id") in selected_claim_ids and row.get("usable_text")
    ]
    mapping_audit: list[dict[str, Any]] = []
    raw_qrel_rows: list[dict[str, Any]] = []
    for evidence in selected_evidence:
        base_audit = {
            "schema_version": RETRIEVAL_SCHEMA_VERSION,
            "evidence_id": evidence["evidence_id"],
            "query_id": evidence["claim_id"],
            "response_id": evidence["response_id"],
            "split": evidence["split"],
            "raw_url": evidence.get("raw_url"),
            "gold_stance": evidence.get("gold_stance"),
            "field_usage": "evaluation_only",
        }
        raw_url = evidence.get("raw_url")
        if not raw_url:
            mapping_audit.append(
                {**base_audit, "mapping_status": "missing_url", "candidates": []}
            )
            continue
        url_row = url_by_raw.get(raw_url)
        if url_row is None:
            mapping_audit.append(
                {
                    **base_audit,
                    "mapping_status": "url_manifest_missing",
                    "candidates": [],
                }
            )
            continue
        trace_doc_id = url_row.get("doc_id")
        doc_id = url_row.get("duplicate_of_doc_id") or trace_doc_id
        fetch_status = url_row.get("fetch_status")
        if fetch_status == "success" and isinstance(trace_doc_id, str):
            trace_document = document_by_id.get(trace_doc_id)
            if trace_document is None:
                raise ValueError(
                    f"Successful URL trace points to missing document: {trace_doc_id}"
                )
            expected_primary = (
                trace_document.get("duplicate_of_doc_id") or trace_doc_id
            )
            if doc_id != expected_primary:
                raise ValueError(
                    "URL/document duplicate alias mismatch for "
                    f"{raw_url}: {doc_id!r} != {expected_primary!r}"
                )
        if fetch_status != "success" or not isinstance(doc_id, str):
            mapping_audit.append(
                {
                    **base_audit,
                    "mapping_status": "url_fetch_failure",
                    "fetch_status": fetch_status,
                    "doc_id": doc_id,
                    "trace_doc_id": trace_doc_id,
                    "candidates": [],
                }
            )
            continue
        document = document_by_id.get(doc_id)
        candidates_for_doc = passages_by_doc.get(doc_id, [])
        if document is None or not candidates_for_doc:
            mapping_audit.append(
                {
                    **base_audit,
                    "mapping_status": "document_available_but_no_passages",
                    "fetch_status": fetch_status,
                    "doc_id": doc_id,
                    "trace_doc_id": trace_doc_id,
                    "candidates": [],
                }
            )
            continue
        candidates, mapping_status = _best_alignment_candidates(
            evidence, candidates_for_doc, config["alignment"]
        )
        relevance, stance_audit = _relevance_for_stance(
            evidence.get("gold_stance")
        )
        candidate_summaries: list[dict[str, Any]] = []
        for candidate in candidates:
            audit_status = candidate["audit_status"]
            if stance_audit and audit_status == "auto_accepted":
                audit_status = "auto_accepted_stance_semantics_need_audit"
            positive = candidate.get("positive_qrel") is True
            row = {
                "schema_version": RETRIEVAL_SCHEMA_VERSION,
                "query_id": evidence["claim_id"],
                "passage_id": candidate["passage"]["passage_id"],
                "relevance": relevance if positive else 0,
                "stance": evidence.get("gold_stance"),
                "alignment_method": candidate["method"],
                "alignment_score": round(float(candidate["score"]), 6),
                "audit_status": audit_status,
                "evidence_ids": [evidence["evidence_id"]],
                "split": evidence["split"],
                "field_usage": "evaluation_only_qrel",
            }
            raw_qrel_rows.append(row)
            candidate_summaries.append(
                {
                    "passage_id": row["passage_id"],
                    "alignment_method": row["alignment_method"],
                    "alignment_score": row["alignment_score"],
                    "audit_status": row["audit_status"],
                    "proposed_relevance": relevance,
                    "qrel_relevance": row["relevance"],
                }
            )
        mapping_audit.append(
            {
                **base_audit,
                "mapping_status": mapping_status,
                "fetch_status": fetch_status,
                "doc_id": doc_id,
                "trace_doc_id": trace_doc_id,
                "candidates": candidate_summaries,
            }
        )

    method_priority = {
        "exact_normalized": 3,
        "sentence_substring": 2,
        "sentence_substring_candidate": 1,
        "fuzzy_lexical_candidate": 1,
    }
    audit_priority = {
        "ambiguous_needs_manual_review": 4,
        "needs_manual_review": 3,
        "auto_accepted_stance_semantics_need_audit": 2,
        "auto_accepted": 1,
    }
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw_qrel_rows:
        key = (row["query_id"], row["passage_id"])
        current = aggregated.get(key)
        if current is None:
            aggregate = dict(row)
            aggregate["stances"] = [row["stance"]]
            aggregate["alignment_methods"] = [row["alignment_method"]]
            aggregated[key] = aggregate
            continue
        current["relevance"] = max(current["relevance"], row["relevance"])
        current["alignment_score"] = max(
            current["alignment_score"], row["alignment_score"]
        )
        current["evidence_ids"] = sorted(
            set(current["evidence_ids"]) | set(row["evidence_ids"])
        )
        current["stances"] = sorted(
            {str(value) for value in current["stances"] + [row["stance"]]}
        )
        current["alignment_methods"] = sorted(
            set(current["alignment_methods"]) | {row["alignment_method"]}
        )
        if method_priority[row["alignment_method"]] > method_priority[
            current["alignment_method"]
        ]:
            current["alignment_method"] = row["alignment_method"]
        if audit_priority.get(row["audit_status"], 0) > audit_priority.get(
            current["audit_status"], 0
        ):
            current["audit_status"] = row["audit_status"]
    qrel_rows = sorted(
        aggregated.values(),
        key=lambda row: (row["query_id"], row["passage_id"]),
    )
    for row in qrel_rows:
        row["stance"] = (
            row["stances"][0] if len(row["stances"]) == 1 else "mixed"
        )

    mapping_counts = Counter(row["mapping_status"] for row in mapping_audit)
    fetch_failure_status_counts = Counter(
        row.get("fetch_status")
        for row in mapping_audit
        if row["mapping_status"] == "url_fetch_failure"
    )
    mapped_claim_ids = {
        row["query_id"]
        for row in qrel_rows
        if row["relevance"] > 0
    }
    pending_fetch_mapping_count = sum(
        row["mapping_status"] == "url_fetch_failure"
        and row.get("fetch_status") in {None, "pending", "stale_pending"}
        for row in mapping_audit
    )
    if len(selected_claim_ids) != len(matched_claim_ids):
        mapping_build_status = "partial_limit"
    elif pending_fetch_mapping_count:
        mapping_build_status = "partial_source_fetch"
    else:
        mapping_build_status = "complete_with_recorded_unmatched"
    report = {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": mapping_build_status,
        "split_scope": split_scope,
        "heldout_configuration_frozen_confirmation": bool(
            confirm_config_frozen if split_scope in {"heldout", "all"} else False
        ),
        "artifact_namespace": artifact_namespace,
        "alignment_version": config["alignment"]["version"],
        "whole_cohort_matched_claim_count": len(all_matched_claim_ids),
        "matched_claim_count": len(matched_claim_ids),
        "selected_matched_claim_count": len(selected_claim_ids),
        "selected_usable_evidence_count": len(selected_evidence),
        "mapping_audit_row_count": len(mapping_audit),
        "mapping_status_counts": dict(sorted(mapping_counts.items())),
        "pending_fetch_mapping_count": pending_fetch_mapping_count,
        "fetch_failure_status_counts": dict(
            sorted((str(key), value) for key, value in fetch_failure_status_counts.items())
        ),
        "positive_qrel_pair_count": sum(
            row["relevance"] > 0 for row in qrel_rows
        ),
        "audit_candidate_pair_count": sum(
            row["relevance"] == 0 for row in qrel_rows
        ),
        "query_count_with_positive_qrels": len(mapped_claim_ids),
        "query_count_without_positive_qrels": len(selected_claim_ids)
        - len(mapped_claim_ids),
        "qrel_relevance_counts": dict(
            sorted(Counter(str(row["relevance"]) for row in qrel_rows).items())
        ),
        "alignment_method_counts": dict(
            sorted(Counter(row["alignment_method"] for row in qrel_rows).items())
        ),
        "audit_status_counts": dict(
            sorted(Counter(row["audit_status"] for row in qrel_rows).items())
        ),
        "split_positive_qrel_counts": dict(
            sorted(
                Counter(
                    row["split"] for row in qrel_rows if row["relevance"] > 0
                ).items()
            )
        ),
        "input_hashes": {
            "evidence_manifest_sha256": sha256_file(paths.evidence_manifest),
            "url_manifest_sha256": sha256_file(paths.url_manifest),
            "documents_sha256": sha256_file(paths.documents),
            "passages_sha256": sha256_file(passage_input),
            "passages_canonical_sha256": canonical_json_hash(passage_rows),
            "config_sha256": canonical_json_hash(config),
        },
        "input_passages": project_relative(project_root, passage_input),
        "passage_build_fingerprint": passage_build_report.get("chunking", {}).get(
            "fingerprint"
        ),
        "output_qrels": project_relative(project_root, qrels_jsonl),
        "policy": {
            "support_and_refute_can_be_relevance_2": True,
            "partial_support_relevance": 1,
            "fuzzy_candidates_are_positive_qrels": False,
            "fuzzy_candidates_require_manual_audit": True,
            "partial_sentence_candidates_are_positive_qrels": False,
            "sentence_substring_auto_accept_threshold": config["alignment"].get(
                "sentence_substring_auto_accept_threshold", 1.0
            ),
            "semantic_alignment": "not_implemented; lexical candidates only",
        },
    }
    if not dry_run:
        atomic_write_jsonl(qrels_jsonl, qrel_rows)
        tsv_lines = [
            "query_id\tpassage_id\trelevance\tstance\talignment_method\t"
            "alignment_score\taudit_status\tevidence_ids\n"
        ]
        for row in qrel_rows:
            tsv_lines.append(
                "\t".join(
                    [
                        row["query_id"],
                        row["passage_id"],
                        str(row["relevance"]),
                        str(row["stance"]),
                        row["alignment_method"],
                        str(row["alignment_score"]),
                        row["audit_status"],
                        ";".join(row["evidence_ids"]),
                    ]
                )
                + "\n"
            )
        atomic_write_text(qrels_tsv, "".join(tsv_lines))
        atomic_write_jsonl(mapping_audit_path, mapping_audit)
        report["qrels_file_sha256"] = sha256_file(qrels_jsonl)
        atomic_write_json(mapping_report_json, report)
        atomic_write_text(
            mapping_report_markdown, _qrels_markdown(report)
        )
    return report


def _corpus_summary_markdown(summary: dict[str, Any]) -> str:
    stages = summary["stages"]
    fetch = summary["fetch"]
    return "\n".join(
        [
            "# Experiment A corpus status",
            "",
            f"Corpus: **{summary['corpus_name']}**",
            "",
            "| Stage | Status |",
            "|---|---|",
            *[
                f"| `{name}` | `{value['status']}` |"
                for name, value in stages.items()
            ],
            "",
            f"- URL manifest rows: {fetch['url_manifest_rows']}",
            f"- Unique canonical URL candidates: {fetch['unique_canonical_candidate_count']}",
            f"- Pending URL rows: {fetch['pending_url_rows']}",
            f"- Successful frozen documents: {fetch['successful_document_count']}",
            f"- Empty-content documents: {fetch['fetch_status_counts'].get('empty_content', 0)}",
            f"- Empty-content documents recovered offline: {summary['reprocessing']['recovered_empty_content_count']}",
            f"- Passages: {summary['passages']['passage_count']}",
            f"- Development positive qrel pairs: {summary['qrels']['dev']['positive_pair_count']}",
            (
                "- Optional combined whole-cohort qrels: "
                f"`{stages['build_qrels_all_secondary']['status']}` "
                "(not required for development selection or held-out evaluation)"
            ),
            "",
            (
                "Two-level retrieval metrics: "
                f"`{summary['retrieval_metrics_status']}`. "
                "Retrieved-verifier status: "
                f"`{summary['retrieved_verifier_status']}`."
            ),
            "",
        ]
    )


def _passage_artifact_status(
    paths: RetrievalPaths,
    passage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not paths.passages.exists() and not paths.passage_build_report.exists():
        return {"status": "pending", "stale_reasons": []}
    reasons: list[str] = []
    report = load_json(paths.passage_build_report, allow_missing=True)
    if not paths.passages.exists() or not report:
        reasons.append("passage_or_report_missing")
    else:
        if not paths.documents.exists():
            reasons.append("documents_missing")
        elif report.get("input_documents_sha256") != sha256_file(paths.documents):
            reasons.append("documents_hash_mismatch")
        if report.get("passages_canonical_sha256") != canonical_json_hash(
            passage_rows
        ):
            reasons.append("passages_hash_mismatch")
        fingerprint = report.get("chunking", {}).get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            reasons.append("passage_report_missing_chunk_fingerprint")
        fingerprints = {
            row.get("chunk_config_fingerprint") for row in passage_rows
        }
        if passage_rows and fingerprints != {fingerprint}:
            reasons.append("chunk_fingerprint_mismatch")
    return {
        "status": "stale" if reasons else report.get("status", "complete"),
        "stale_reasons": reasons,
        "report": report,
    }


def _qrel_artifact_status(
    paths: RetrievalPaths,
    config: dict[str, Any],
    *,
    split_scope: str,
    qrels_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if not qrels_path.exists() and not report_path.exists():
        return {"status": "pending", "stale_reasons": [], "rows": []}
    reasons: list[str] = []
    report = load_json(report_path, allow_missing=True)
    rows = load_jsonl(qrels_path, allow_missing=True)
    if not qrels_path.exists() or not report:
        reasons.append("qrels_or_report_missing")
    else:
        if report.get("split_scope") != split_scope:
            reasons.append("split_scope_mismatch")
        if split_scope in {"heldout", "all"} and report.get(
            "heldout_configuration_frozen_confirmation"
        ) is not True:
            reasons.append("heldout_not_explicitly_unsealed")
        current_inputs = {
            "evidence_manifest_sha256": paths.evidence_manifest,
            "url_manifest_sha256": paths.url_manifest,
            "documents_sha256": paths.documents,
            "passages_sha256": paths.passages,
        }
        recorded = report.get("input_hashes", {})
        for field, path in current_inputs.items():
            if not path.exists() or recorded.get(field) != sha256_file(path):
                reasons.append(f"{field}_mismatch")
        if recorded.get("config_sha256") != canonical_json_hash(config):
            reasons.append("config_hash_mismatch")
        if report.get("qrels_file_sha256") != sha256_file(qrels_path):
            reasons.append("qrels_hash_mismatch")
    return {
        "status": "stale" if reasons else report.get("status", "complete"),
        "stale_reasons": reasons,
        "rows": rows,
        "report": report,
    }


def summarize_corpus(
    project_root: Path,
    paths: RetrievalPaths,
    config: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    split_summary = load_json(paths.split_summary, allow_missing=True)
    split_rows = load_jsonl(paths.split_manifest, allow_missing=True)
    preparation = load_json(paths.preparation_report_json, allow_missing=True)
    reprocess_report = load_json(paths.reprocess_report_json, allow_missing=True)
    evidence_rows = load_jsonl(paths.evidence_manifest, allow_missing=True)
    url_rows = load_jsonl(paths.url_manifest, allow_missing=True)
    document_rows = load_jsonl(paths.documents, allow_missing=True)
    passage_rows = load_jsonl(paths.passages, allow_missing=True)
    dev_retrieval_comparison_path = (
        paths.root / "evaluation" / "reports" / "two_level_dev_comparison.json"
    )
    heldout_retrieval_comparison_path = (
        paths.root
        / "evaluation"
        / "reports"
        / "two_level_heldout_comparison.json"
    )
    retrieved_verifier_summary_path = (
        project_root
        / "outputs"
        / "factcheck_bench_full"
        / "reports"
        / "12_retrieved_evidence_verifier_heldout_summary.json"
    )
    dev_retrieval_comparison = load_json(
        dev_retrieval_comparison_path, allow_missing=True
    )
    heldout_retrieval_comparison = load_json(
        heldout_retrieval_comparison_path, allow_missing=True
    )
    retrieved_verifier_summary = load_json(
        retrieved_verifier_summary_path, allow_missing=True
    )
    dev_retrieval_evaluation_status = (
        "complete"
        if dev_retrieval_comparison.get("status") == "complete"
        and dev_retrieval_comparison.get("split") == "dev"
        else "pending"
    )
    heldout_retrieval_evaluation_status = (
        "complete"
        if heldout_retrieval_comparison.get("status") == "complete"
        and heldout_retrieval_comparison.get("split") == "heldout"
        else "pending"
    )
    passage_state = _passage_artifact_status(paths, passage_rows)
    qrel_states = {
        "dev": _qrel_artifact_status(
            paths,
            config,
            split_scope="dev",
            qrels_path=paths.qrels_dev_jsonl,
            report_path=paths.qrels_dev_mapping_report_json,
        ),
        "heldout": _qrel_artifact_status(
            paths,
            config,
            split_scope="heldout",
            qrels_path=paths.qrels_heldout_jsonl,
            report_path=paths.qrels_heldout_mapping_report_json,
        ),
        "all": _qrel_artifact_status(
            paths,
            config,
            split_scope="all",
            qrels_path=paths.qrels_jsonl,
            report_path=paths.qrels_mapping_report_json,
        ),
    }

    current_config_hash = canonical_json_hash(config)
    full_paths = paths_for_scope(project_root, "full")
    split_stale_reasons: list[str] = []
    if split_summary or split_rows:
        if not split_summary or not split_rows:
            split_stale_reasons.append("split_manifest_or_summary_missing")
        if split_summary.get("config_sha256") != current_config_hash:
            split_stale_reasons.append("config_hash_mismatch")
        if split_summary.get("artifact_sha256") != canonical_json_hash(split_rows):
            split_stale_reasons.append("split_manifest_hash_mismatch")
        if len(split_rows) != split_summary.get("total_matched_claims"):
            split_stale_reasons.append("split_row_count_mismatch")
        canonical_split_inputs = {
            "raw": full_paths.source_input,
            "gold_claims": full_paths.gold_claims,
            "cohort_manifest": full_paths.cohort_manifest,
            "no_evidence_results": full_paths.no_evidence_output,
            "oracle_results": full_paths.oracle_output,
        }
        recorded_inputs = split_summary.get("input_files", {})
        for name, path in canonical_split_inputs.items():
            if (
                not path.exists()
                or recorded_inputs.get(name, {}).get("sha256") != sha256_file(path)
            ):
                split_stale_reasons.append(f"canonical_{name}_hash_mismatch")
    preparation_stale_reasons: list[str] = []
    if preparation or evidence_rows or url_rows:
        if not preparation or not evidence_rows or not url_rows:
            preparation_stale_reasons.append(
                "evidence_url_manifest_or_report_missing"
            )
        if preparation.get("input_hashes", {}).get(
            "config_sha256"
        ) != current_config_hash:
            preparation_stale_reasons.append("config_hash_mismatch")
        canonical_preparation_inputs = {
            "gold_claims_sha256": full_paths.gold_claims,
            "cohort_manifest_sha256": full_paths.cohort_manifest,
        }
        for field, path in canonical_preparation_inputs.items():
            if (
                not path.exists()
                or preparation.get("input_hashes", {}).get(field)
                != sha256_file(path)
            ):
                preparation_stale_reasons.append(
                    f"canonical_{field}_mismatch"
                )
        artifact_hashes = preparation.get("artifact_hashes", {})
        if artifact_hashes.get(
            "evidence_manifest_canonical_sha256"
        ) != canonical_json_hash(evidence_rows):
            preparation_stale_reasons.append("evidence_manifest_hash_mismatch")
        if artifact_hashes.get(
            "url_manifest_preparation_sha256"
        ) != canonical_json_hash(_url_manifest_preparation_projection(url_rows)):
            preparation_stale_reasons.append("url_manifest_hash_mismatch")

    fetch_counts = Counter(row.get("fetch_status") for row in url_rows)
    pending = fetch_counts.get("pending", 0) + fetch_counts.get(
        "stale_pending", 0
    )
    valid_candidates = sum(
        row.get("canonicalisation_status") == "ok" for row in url_rows
    )
    unique_canonical_candidates = len(
        {
            row.get("canonical_url")
            for row in url_rows
            if row.get("canonicalisation_status") == "ok"
            and row.get("canonical_url")
        }
    )
    successful_documents = sum(
        row.get("fetch_status") == "success" for row in document_rows
    )
    reprocess_stale_reasons: list[str] = []
    if reprocess_report:
        if not paths.documents.exists() or reprocess_report.get(
            "output_documents_sha256"
        ) != sha256_file(paths.documents):
            reprocess_stale_reasons.append("documents_hash_mismatch")
        if not paths.url_manifest.exists() or reprocess_report.get(
            "output_url_manifest_sha256"
        ) != sha256_file(paths.url_manifest):
            reprocess_stale_reasons.append("url_manifest_hash_mismatch")
    document_by_id = (
        _index_unique(document_rows, "doc_id") if document_rows else {}
    )
    stale_success_rows = [
        row.get("url_id")
        for row in url_rows
        if row.get("fetch_status") == "success"
        and not _existing_fetch_state_is_auditable(
            project_root, row, document_by_id, config
        )
    ]
    if not url_rows:
        fetch_status = "pending"
    elif stale_success_rows:
        fetch_status = "stale_manifest_document_mismatch"
    elif pending:
        fetch_status = "partial" if len(document_rows) else "pending"
    else:
        fetch_status = "complete_with_recorded_failures"

    summary = {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "corpus_name": config["corpus"]["name"],
        "scope": "full",
        "stages": {
            "prepare_retrieval_splits": {
                "status": (
                    "stale"
                    if split_stale_reasons
                    else "complete"
                    if split_summary.get("assertions_passed") is True
                    else "pending"
                ),
                "artifact": project_relative(project_root, paths.split_manifest),
                "stale_reasons": split_stale_reasons,
            },
            "build_evidence_manifest": {
                "status": (
                    "stale"
                    if preparation_stale_reasons
                    else "complete"
                    if preparation
                    else "pending"
                ),
                "artifact": project_relative(project_root, paths.evidence_manifest),
                "stale_reasons": preparation_stale_reasons,
            },
            "fetch_corpus": {
                "status": fetch_status,
                "artifact": project_relative(project_root, paths.documents),
            },
            "reprocess_frozen_corpus": {
                "status": (
                    "stale"
                    if reprocess_stale_reasons
                    else reprocess_report.get("status", "complete")
                    if reprocess_report
                    else "pending"
                ),
                "artifact": project_relative(
                    project_root, paths.reprocess_report_json
                ),
                "stale_reasons": reprocess_stale_reasons,
            },
            "build_passages": {
                "status": passage_state["status"],
                "artifact": project_relative(project_root, paths.passages),
                "stale_reasons": passage_state["stale_reasons"],
            },
            "build_qrels_dev": {
                "status": qrel_states["dev"]["status"],
                "artifact": project_relative(project_root, paths.qrels_dev_jsonl),
                "stale_reasons": qrel_states["dev"]["stale_reasons"],
            },
            "build_qrels_heldout": {
                "status": qrel_states["heldout"]["status"],
                "artifact": project_relative(
                    project_root, paths.qrels_heldout_jsonl
                ),
                "stale_reasons": qrel_states["heldout"]["stale_reasons"],
            },
            "build_qrels_all_secondary": {
                "status": qrel_states["all"]["status"],
                "artifact": project_relative(project_root, paths.qrels_jsonl),
                "stale_reasons": qrel_states["all"]["stale_reasons"],
                "required_for_primary_experiment": False,
            },
            "evaluate_retrieval_dev_two_level": {
                "status": dev_retrieval_evaluation_status,
                "artifact": project_relative(
                    project_root, dev_retrieval_comparison_path
                ),
            },
            "evaluate_retrieval_heldout_two_level": {
                "status": heldout_retrieval_evaluation_status,
                "artifact": project_relative(
                    project_root, heldout_retrieval_comparison_path
                ),
            },
            "run_retrieved_verifier_heldout": {
                "status": retrieved_verifier_summary.get(
                    "status", "pending"
                ),
                "artifact": project_relative(
                    project_root, retrieved_verifier_summary_path
                ),
            },
        },
        "split": {
            "development_matched_claims": split_summary.get(
                "development_matched_claims"
            ),
            "heldout_matched_claims": split_summary.get(
                "heldout_matched_claims"
            ),
            "total_matched_claims": split_summary.get("total_matched_claims"),
        },
        "preparation": {
            "evidence_item_count": preparation.get("evidence_inventory", {}).get(
                "evidence_item_count"
            ),
            "items_with_url": preparation.get("evidence_inventory", {}).get(
                "items_with_url"
            ),
            "raw_unique_url_count": preparation.get("evidence_inventory", {}).get(
                "raw_unique_url_count"
            ),
            "canonical_unique_url_count": preparation.get(
                "url_canonicalisation", {}
            ).get("canonical_unique_url_count"),
        },
        "fetch": {
            "url_manifest_rows": len(url_rows),
            "valid_canonical_candidate_rows": valid_candidates,
            "unique_canonical_candidate_count": unique_canonical_candidates,
            "pending_url_rows": pending,
            "fetch_status_counts": dict(
                sorted((str(key), value) for key, value in fetch_counts.items())
            ),
            "document_record_count": len(document_rows),
            "successful_document_count": successful_documents,
            "stale_success_manifest_row_count": len(stale_success_rows),
            "content_duplicate_document_count": sum(
                bool(row.get("is_content_duplicate")) for row in document_rows
            ),
        },
        "reprocessing": {
            "status": (
                "stale"
                if reprocess_stale_reasons
                else reprocess_report.get("status", "complete")
                if reprocess_report
                else "pending"
            ),
            "recovered_empty_content_count": reprocess_report.get(
                "recovered_empty_content_count", 0
            ),
            "downgraded_success_count": reprocess_report.get(
                "downgraded_success_count", 0
            ),
            "network_was_used": reprocess_report.get("network_was_used"),
            "stale_reasons": reprocess_stale_reasons,
        },
        "passages": {
            "passage_count": len(passage_rows),
            "document_count": len({row.get("doc_id") for row in passage_rows}),
        },
        "qrels": {
            split: {
                "status": state["status"],
                "pair_count": len(state["rows"]),
                "positive_pair_count": sum(
                    int(row.get("relevance", 0)) > 0 for row in state["rows"]
                ),
                "audit_candidate_pair_count": sum(
                    int(row.get("relevance", 0)) == 0 for row in state["rows"]
                ),
                "mapping_status_counts": state.get("report", {}).get(
                    "mapping_status_counts", {}
                ),
                "stale_reasons": state["stale_reasons"],
            }
            for split, state in qrel_states.items()
        },
        "retrieval_metrics_status": (
            "complete_dev_and_heldout_two_level"
            if dev_retrieval_evaluation_status == "complete"
            and heldout_retrieval_evaluation_status == "complete"
            else "complete_dev_two_level"
            if dev_retrieval_evaluation_status == "complete"
            else "not_run"
        ),
        "retrieval_evaluation": {
            "development": dev_retrieval_comparison,
            "heldout": heldout_retrieval_comparison,
        },
        "retrieved_verifier_status": retrieved_verifier_summary.get(
            "status", "not_run"
        ),
        "leakage_policy": config["leakage_policy"],
        "config_sha256": current_config_hash,
    }
    if not dry_run:
        atomic_write_json(paths.corpus_summary_json, summary)
        atomic_write_text(
            paths.corpus_summary_markdown, _corpus_summary_markdown(summary)
        )
    return summary


def default_paths_and_config(
    project_root: Path, scope: str = "full"
) -> tuple[RetrievalPaths, dict[str, Any]]:
    paths = retrieval_paths(project_root, scope)
    return paths, config_for_paths(paths)
