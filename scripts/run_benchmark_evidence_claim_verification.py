#!/usr/bin/env python3
"""Run the pilot or full oracle-evidence factuality verifier with local Ollama.

This stage reuses the proven request/parser primitives from 08b while keeping
an independent oracle result schema.  The model receives only ``gold_claim``
and normalized evidence passage text.  Gold labels and evidence annotation
metadata are attached to the persisted result only after inference.

Recommended sequence:

1. Run ``--smoke-test --dry-run`` (no calls and no writes).
2. Run ``--smoke-test`` into the automatically selected smoke paths.
3. Inspect the smoke reports.
4. Run or resume the full automatically derived oracle cohort.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factcheck_bench_pipeline import (  # noqa: E402
    COHORT_VERSION as SHARED_COHORT_VERSION,
    NORMALIZATION_VERSION as SHARED_NORMALIZATION_VERSION,
    SCOPES,
    build_oracle_cohort as shared_build_oracle_cohort,
    normalize_evidence_text as shared_normalize_evidence_text,
    normalize_oracle_evidence as shared_normalize_oracle_evidence,
    paths_for_scope,
)
from factcheck_bench_analysis import (  # noqa: E402
    PAIRED_STATES,
    build_paired_transitions,
    metric_differences,
)

BASE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "run_no_evidence_claim_verification.py"
)


def _load_08b_module() -> Any:
    spec = importlib.util.spec_from_file_location("fcb_no_evidence_08b", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import 08b verifier primitives from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_08b_module()

LABELS = _base.LABELS
PRIMARY_LABELS = _base.PRIMARY_LABELS
OUTPUT_SCHEMA = _base.OUTPUT_SCHEMA
canonical_json_hash = _base.canonical_json_hash
load_jsonl_objects = _base.load_jsonl_objects
parse_model_output = _base.parse_model_output
sha256_file = _base.sha256_file
sha256_text = _base.sha256_text
preflight_ollama = _base.preflight_ollama
report_path = _base.report_path

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "factcheck_bench"
    / "processed"
    / "fcb_gold_claims_pilot_20.jsonl"
)
DEFAULT_NO_EVIDENCE_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_pilot"
    / "jsonl"
    / "08b_no_evidence_verifier_results.jsonl"
)
DEFAULT_NO_EVIDENCE_JSON_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_pilot"
    / "reports"
    / "08b_no_evidence_verifier_summary.json"
)
DEFAULT_NO_EVIDENCE_MARKDOWN_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_pilot"
    / "reports"
    / "08b_no_evidence_verifier_report.md"
)
NO_EVIDENCE_PROMPT = PROJECT_ROOT / "prompts" / "no_evidence_verifier.txt"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_pilot"
    / "jsonl"
    / "08c_oracle_evidence_verifier_results.jsonl"
)
DEFAULT_JSON_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_pilot"
    / "reports"
    / "08c_oracle_evidence_verifier_summary.json"
)
DEFAULT_MARKDOWN_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_pilot"
    / "reports"
    / "08c_oracle_evidence_verifier_report.md"
)
DEFAULT_SMOKE_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_pilot"
    / "jsonl"
    / "08c_oracle_evidence_smoke.jsonl"
)
DEFAULT_SMOKE_JSON_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_pilot"
    / "reports"
    / "08c_oracle_evidence_smoke_summary.json"
)
DEFAULT_SMOKE_MARKDOWN_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_pilot"
    / "reports"
    / "08c_oracle_evidence_smoke_report.md"
)
DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "oracle_evidence_verifier.txt"
FULL_PATHS = paths_for_scope(PROJECT_ROOT, "full")

PROMPT_PLACEHOLDERS = {"{gold_claim_json}", "{oracle_evidence_text}"}
PROMPT_VERSION = "oracle_evidence_v1"
NORMALIZATION_VERSION = SHARED_NORMALIZATION_VERSION
COHORT_VERSION = SHARED_COHORT_VERSION
OUTPUT_SCHEMA_VERSION = "oracle_evidence_output_v1"
RESULT_SCHEMA_VERSION = "oracle_evidence_result_v1"
REPORT_VERSION = "oracle_evidence_report_v1"
SETTING = "oracle_evidence"

RESULT_ALLOWED_FIELDS = {
    "result_schema_version",
    "claim_id",
    "response_id",
    "gold_claim",
    "human_label",
    "claim_sha256",
    "oracle_evidence",
    "setting",
    "model_input_fields",
    "model",
    "model_digest",
    "expected_no_evidence_model_digest",
    "temperature",
    "seed",
    "num_predict",
    "think",
    "timeout_seconds",
    "max_retries",
    "max_consecutive_request_errors",
    "prompt_version",
    "prompt_sha256",
    "evidence_normalization_version",
    "cohort_version",
    "cohort_sha256",
    "output_schema_version",
    "output_schema_sha256",
    "input_sha256",
    "no_evidence_results_sha256",
    "run_fingerprint",
    "attempts",
    "latency_seconds",
    "raw_model_output",
    "ollama_metadata",
    "created_at",
    "status",
    "prediction",
    "confidence",
    "rationale",
    "error",
}

FORBIDDEN_PROMPT_METADATA_FIELDS = {
    "human_label",
    "human_label_bool",
    "human_label_raw",
    "gold_evidence_texts",
    "evidence_available",
    "evidence_source",
    "auto_evidence_sufficient",
    "claim_needs_edit",
    "revised_claim",
    "revision_evidence_index",
    "raw_auto_evidence",
    "raw_auto_evidence_urls",
    "raw_auto_evidence_stances",
    "raw_human_evidence",
    "prompt",
    "source_response",
    "source_sentence",
}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FactCheck-Bench oracle-evidence verifier with Ollama."
    )
    parser.add_argument("--scope", choices=SCOPES, default="pilot")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--markdown-report", type=Path, default=None
    )
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--no-evidence-results",
        type=Path,
        default=None,
        help="Existing 08b results used for digest checks and paired comparison.",
    )
    parser.add_argument("--model", default=os.getenv("CHAT_MODEL", "qwen3:8b"))
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=None)
    selection.add_argument("--claim-id", action="append", default=None)
    selection.add_argument(
        "--smoke-test",
        action="store_true",
        help="Select one eligible FACTUAL and one eligible NON_FACTUAL claim.",
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-consecutive-request-errors", type=int, default=3)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--typical-error-limit", type=int, default=10)

    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true")
    output_mode.add_argument("--overwrite", action="store_true")

    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--dry-run", "--preview", action="store_true", help="No calls or writes."
    )
    run_mode.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate reports from an existing oracle result JSONL.",
    )

    args = parser.parse_args(argv)
    output_explicit = args.output is not None
    report_explicit = args.report is not None
    markdown_explicit = args.markdown_report is not None
    defaults = paths_for_scope(PROJECT_ROOT, args.scope)
    args.input = args.input or defaults.gold_claims
    args.no_evidence_results = (
        args.no_evidence_results or defaults.no_evidence_output
    )
    args.output = args.output or defaults.oracle_output
    args.report = args.report or defaults.oracle_report
    args.markdown_report = args.markdown_report or defaults.oracle_markdown
    validate_args(parser, args)
    args.selection_explicit = bool(
        args.limit is not None or args.claim_id is not None or args.smoke_test
    )
    if args.smoke_test:
        if not output_explicit:
            args.output = defaults.oracle_smoke_output
        if not report_explicit:
            args.report = defaults.oracle_smoke_report
        if not markdown_explicit:
            args.markdown_report = defaults.oracle_smoke_markdown
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.claim_id is not None and len(set(args.claim_id)) != len(args.claim_id):
        parser.error("--claim-id values must be unique")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.num_predict < 1:
        parser.error("--num-predict must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if args.max_consecutive_request_errors < 1:
        parser.error("--max-consecutive-request-errors must be at least 1")
    if not 0.0 <= args.high_confidence_threshold <= 1.0:
        parser.error("--high-confidence-threshold must be between 0 and 1")
    if args.typical_error_limit < 0:
        parser.error("--typical-error-limit cannot be negative")
    if args.report_only and (args.resume or args.overwrite):
        parser.error("--report-only cannot be combined with --resume/--overwrite")


def canonical_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def validate_write_path_safety(
    args: argparse.Namespace,
    partial_selection: bool,
) -> None:
    """Protect immutable inputs/history and formal paths from accidental aliases."""
    destinations = {
        "output": canonical_path(args.output),
        "json_report": canonical_path(args.report),
        "markdown_report": canonical_path(args.markdown_report),
    }
    if len(set(destinations.values())) != len(destinations):
        raise ValueError(f"Output/report paths must be distinct: {destinations}")

    scope_paths = paths_for_scope(
        PROJECT_ROOT, getattr(args, "scope", "pilot")
    )
    protected_sources = {
        canonical_path(args.input),
        canonical_path(args.prompt),
        canonical_path(args.no_evidence_results),
        canonical_path(BASE_SCRIPT),
        canonical_path(NO_EVIDENCE_PROMPT),
        canonical_path(DEFAULT_NO_EVIDENCE_JSON_REPORT),
        canonical_path(DEFAULT_NO_EVIDENCE_MARKDOWN_REPORT),
        canonical_path(scope_paths.no_evidence_report),
        canonical_path(scope_paths.no_evidence_markdown),
    }
    collisions = {
        name: str(path)
        for name, path in destinations.items()
        if path in protected_sources
    }
    if collisions:
        raise ValueError(
            "Refusing to overwrite an input, prompt, script, or historical 08b "
            f"artifact: {collisions}"
        )

    if partial_selection:
        formal_paths = {
            canonical_path(scope_paths.oracle_output),
            canonical_path(scope_paths.oracle_report),
            canonical_path(scope_paths.oracle_markdown),
        }
        formal_collisions = {
            name: str(path)
            for name, path in destinations.items()
            if path in formal_paths
        }
        if formal_collisions:
            raise ValueError(
                "A partial/smoke run may not use a formal full-run output/report "
                f"path: {formal_collisions}"
            )

    if getattr(args, "scope", "pilot") == "full":
        pilot_paths = paths_for_scope(PROJECT_ROOT, "pilot")
        pilot_root = canonical_path(pilot_paths.output_root)
        pilot_collisions = {
            name: str(path)
            for name, path in destinations.items()
            if path == pilot_root or pilot_root in path.parents
        }
        if pilot_collisions:
            raise ValueError(
                "Full scope may not write historical pilot artifacts: "
                f"{pilot_collisions}"
            )


# Public compatibility names now delegate to the shared data module. Keeping
# these names preserves existing tests/imports and, more importantly, makes the
# preprocessing manifest and runtime oracle cohort use identical code.
normalize_evidence_text = shared_normalize_evidence_text
normalize_oracle_evidence = shared_normalize_oracle_evidence


def load_input_records(path: Path) -> list[dict[str, Any]]:
    """Reuse 08b input invariants, then validate evidence container metadata."""
    records = _base.load_input_records(path)
    for line_number, record in enumerate(records, start=1):
        if "gold_evidence" not in record:
            continue
        evidence = record["gold_evidence"]
        if not isinstance(evidence, (list, dict)) and evidence is not None:
            raise TypeError(
                f"Input line {line_number}: gold_evidence must be list/object/null."
            )
        available = record.get("evidence_available")
        if available is not None and not isinstance(available, bool):
            raise TypeError(
                f"Input line {line_number}: evidence_available must be boolean."
            )
    return records


build_oracle_cohort = shared_build_oracle_cohort


def select_records(
    eligible_records: list[dict[str, Any]],
    cohort_audit: dict[str, Any],
    limit: int | None = None,
    claim_ids: list[str] | None = None,
    smoke_test: bool = False,
) -> list[dict[str, Any]]:
    by_id = {record["claim_id"]: record for record in eligible_records}
    if claim_ids is not None:
        missing = [claim_id for claim_id in claim_ids if claim_id not in by_id]
        if missing:
            excluded_by_id = {
                item["claim_id"]: item["reason"]
                for item in cohort_audit["exclusions"]
            }
            details = {claim_id: excluded_by_id.get(claim_id, "not_in_input") for claim_id in missing}
            raise ValueError(f"Requested claim IDs are not oracle-eligible: {details}")
        return [by_id[claim_id] for claim_id in claim_ids]
    if smoke_test:
        selected = []
        for label in PRIMARY_LABELS:
            match = next(
                (record for record in eligible_records if record["human_label"] == label),
                None,
            )
            if match is None:
                raise ValueError(f"No eligible {label} claim is available for smoke test.")
            selected.append(match)
        return selected
    if limit is not None:
        return eligible_records[:limit]
    return eligible_records


def load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"Prompt file is empty: {path}")
    placeholders = set(re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", template))
    if placeholders != PROMPT_PLACEHOLDERS:
        raise ValueError(
            f"Prompt placeholders must be exactly {sorted(PROMPT_PLACEHOLDERS)}; "
            f"found {sorted(placeholders)}"
        )
    for placeholder in PROMPT_PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ValueError(f"Prompt must contain exactly one {placeholder!r}.")
    return template


def build_prompt(
    template: str,
    gold_claim: str,
    oracle_evidence: dict[str, Any],
) -> str:
    if not isinstance(gold_claim, str) or not gold_claim.strip():
        raise ValueError("gold_claim must be a non-empty string.")
    if oracle_evidence.get("status") != "ok":
        raise ValueError("oracle_evidence must have status='ok'.")
    evidence_text = oracle_evidence.get("normalized_text")
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        raise ValueError("oracle_evidence normalized_text must be non-empty.")
    replacements = {
        "{gold_claim_json}": json.dumps(gold_claim.strip(), ensure_ascii=False),
        "{oracle_evidence_text}": evidence_text,
    }
    pattern = re.compile(r"\{(?:gold_claim_json|oracle_evidence_text)\}")
    return pattern.sub(lambda match: replacements[match.group(0)], template)


def load_no_evidence_baseline(
    path: Path,
    input_by_id: dict[str, dict[str, Any]],
    expected_input_sha256: str | None = None,
    required_claim_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = load_jsonl_objects(path)
    seen: set[str] = set()
    for row in records:
        claim_id = row.get("claim_id")
        if not isinstance(claim_id, str) or claim_id not in input_by_id:
            raise ValueError(f"08b output contains unknown claim_id: {claim_id!r}")
        if claim_id in seen:
            raise ValueError(f"08b output contains duplicate claim_id: {claim_id}")
        seen.add(claim_id)
        missing_fields = sorted(_base.RESULT_ALLOWED_FIELDS - set(row))
        extra_fields = sorted(set(row) - _base.RESULT_ALLOWED_FIELDS)
        if missing_fields or extra_fields:
            raise ValueError(
                "08b output fields do not match the historical no-evidence schema; "
                f"missing={missing_fields}, extra={extra_fields}."
            )
        if row.get("result_schema_version") != _base.RESULT_SCHEMA_VERSION:
            raise ValueError(f"08b result schema mismatch for {claim_id}.")
        if row.get("setting") != _base.SETTING:
            raise ValueError(f"08b setting mismatch for {claim_id}.")
        if row.get("model_input_fields") != ["gold_claim"]:
            raise ValueError(f"08b model_input_fields mismatch for {claim_id}.")
        if expected_input_sha256 is not None and row.get("input_sha256") != expected_input_sha256:
            raise ValueError(f"08b input hash mismatch for {claim_id}.")
        source = input_by_id[claim_id]
        if row.get("response_id") != source["response_id"]:
            raise ValueError(f"08b response_id mismatch for {claim_id}.")
        if row.get("gold_claim") != source["gold_claim"]:
            raise ValueError(f"08b gold_claim mismatch for {claim_id}.")
        if row.get("status") == "ok":
            parse_model_output(
                json.dumps(
                    {
                        "prediction": row.get("prediction"),
                        "confidence": row.get("confidence"),
                        "rationale": row.get("rationale"),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )

    required_ids = set(input_by_id) if required_claim_ids is None else required_claim_ids
    unknown_required_ids = sorted(required_ids - set(input_by_id))
    if unknown_required_ids:
        raise ValueError(
            "Required baseline claim IDs are absent from the gold input: "
            f"{unknown_required_ids}"
        )
    missing_baseline_ids = sorted(required_ids - seen)
    if missing_baseline_ids:
        raise ValueError(
            "08b result file is incomplete for the required paired cohort; "
            f"missing claim IDs: {missing_baseline_ids}"
        )

    profile_fields = (
        "model",
        "model_digest",
        "temperature",
        "seed",
        "num_predict",
        "think",
        "input_sha256",
        "prompt_version",
        "output_schema_version",
        "run_fingerprint",
    )
    profile: dict[str, Any] = {}
    for field in profile_fields:
        values = {row.get(field) for row in records}
        if len(values) != 1:
            raise ValueError(f"08b output has inconsistent {field}: {values}")
        profile[field] = next(iter(values))
    digest = profile.get("model_digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError("08b output does not contain one valid model digest.")
    return records, profile


def validate_model_config_against_baseline(
    args: argparse.Namespace,
    baseline_profile: dict[str, Any],
) -> None:
    requested = {
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "num_predict": args.num_predict,
        "think": False,
    }
    mismatches = {
        key: {"requested": value, "08b": baseline_profile.get(key)}
        for key, value in requested.items()
        if value != baseline_profile.get(key)
    }
    if mismatches:
        raise ValueError(
            "Oracle model configuration must match the historical 08b run: "
            f"{mismatches}"
        )


def cohort_manifest_hash(eligible_records: list[dict[str, Any]]) -> str:
    manifest = [
        {
            "claim_id": record["claim_id"],
            "response_id": record["response_id"],
            "human_label": record["human_label"],
            "gold_claim_sha256": sha256_text(record["gold_claim"]),
            "oracle_evidence_sha256": record["_oracle_evidence"]["normalized_sha256"],
        }
        for record in eligible_records
    ]
    return canonical_json_hash(manifest)


def build_run_config(
    args: argparse.Namespace,
    input_sha256: str,
    prompt_sha256: str,
    no_evidence_results_sha256: str,
    cohort_sha256: str,
    model_digest: str,
    expected_model_digest: str,
) -> dict[str, Any]:
    schema_sha256 = canonical_json_hash(OUTPUT_SCHEMA)
    payload = {
        "setting": SETTING,
        "model": args.model,
        "model_digest": model_digest,
        "expected_no_evidence_model_digest": expected_model_digest,
        "ollama_host": args.ollama_host,
        "temperature": args.temperature,
        "seed": args.seed,
        "num_predict": args.num_predict,
        "think": False,
        "timeout_seconds": args.timeout,
        "max_retries": args.max_retries,
        "max_consecutive_request_errors": args.max_consecutive_request_errors,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "evidence_normalization_version": NORMALIZATION_VERSION,
        "cohort_version": COHORT_VERSION,
        "cohort_sha256": cohort_sha256,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "output_schema_sha256": schema_sha256,
        "input_sha256": input_sha256,
        "no_evidence_results_sha256": no_evidence_results_sha256,
    }
    return {**payload, "run_fingerprint": canonical_json_hash(payload)}


def create_result_base(
    record: dict[str, Any], run_config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "claim_id": record["claim_id"],
        "response_id": record["response_id"],
        "gold_claim": record["gold_claim"],
        "human_label": record["human_label"],
        "claim_sha256": sha256_text(record["gold_claim"]),
        "oracle_evidence": record["_oracle_evidence"],
        "setting": SETTING,
        "model_input_fields": ["gold_claim", "oracle_evidence_text"],
        "model": run_config["model"],
        "model_digest": run_config["model_digest"],
        "expected_no_evidence_model_digest": run_config[
            "expected_no_evidence_model_digest"
        ],
        "temperature": run_config["temperature"],
        "seed": run_config["seed"],
        "num_predict": run_config["num_predict"],
        "think": False,
        "timeout_seconds": run_config["timeout_seconds"],
        "max_retries": run_config["max_retries"],
        "max_consecutive_request_errors": run_config[
            "max_consecutive_request_errors"
        ],
        "prompt_version": run_config["prompt_version"],
        "prompt_sha256": run_config["prompt_sha256"],
        "evidence_normalization_version": run_config[
            "evidence_normalization_version"
        ],
        "cohort_version": run_config["cohort_version"],
        "cohort_sha256": run_config["cohort_sha256"],
        "output_schema_version": run_config["output_schema_version"],
        "output_schema_sha256": run_config["output_schema_sha256"],
        "input_sha256": run_config["input_sha256"],
        "no_evidence_results_sha256": run_config[
            "no_evidence_results_sha256"
        ],
        "run_fingerprint": run_config["run_fingerprint"],
    }


def process_record(
    record: dict[str, Any],
    prompt_template: str,
    run_config: dict[str, Any],
    client: Any,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    result = create_result_base(record, run_config)
    prompt = build_prompt(
        prompt_template, record["gold_claim"], record["_oracle_evidence"]
    )
    started = time.perf_counter()
    raw_output: str | None = None
    metadata: dict[str, Any] = {}
    request_error: Exception | None = None
    attempts = 0

    for attempt in range(run_config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw_output, metadata = _base.call_ollama(client, run_config, prompt)
            request_error = None
            break
        except Exception as error:  # transport or malformed Ollama envelope
            request_error = error
            if attempt < run_config["max_retries"]:
                sleep_fn(1.0)

    result.update(
        {
            "attempts": attempts,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "raw_model_output": raw_output,
            "ollama_metadata": metadata,
            "created_at": utc_now(),
        }
    )
    if request_error is not None:
        result.update(
            {
                "status": "request_error",
                "prediction": None,
                "confidence": None,
                "rationale": None,
                "error": f"{type(request_error).__name__}: {request_error}",
            }
        )
        return result
    try:
        parsed = parse_model_output(raw_output or "")
    except Exception as error:
        result.update(
            {
                "status": "parse_error",
                "prediction": None,
                "confidence": None,
                "rationale": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return result
    result.update(parsed)
    result["status"] = "ok"
    result["error"] = None
    return result


def validate_result_record(
    row: dict[str, Any],
    eligible_by_id: dict[str, dict[str, Any]],
    expected_run_config: dict[str, Any],
) -> None:
    claim_id = row.get("claim_id")
    if not isinstance(claim_id, str) or claim_id not in eligible_by_id:
        raise ValueError(f"Oracle output contains unknown/ineligible claim_id: {claim_id!r}")
    missing = sorted(RESULT_ALLOWED_FIELDS - set(row))
    extra = sorted(set(row) - RESULT_ALLOWED_FIELDS)
    if missing or extra:
        raise ValueError(
            f"Output fields do not match {RESULT_SCHEMA_VERSION}; "
            f"missing={missing}, extra={extra}."
        )

    config_fields = (
        "setting",
        "model",
        "model_digest",
        "expected_no_evidence_model_digest",
        "temperature",
        "seed",
        "num_predict",
        "think",
        "timeout_seconds",
        "max_retries",
        "max_consecutive_request_errors",
        "prompt_version",
        "prompt_sha256",
        "evidence_normalization_version",
        "cohort_version",
        "cohort_sha256",
        "output_schema_version",
        "output_schema_sha256",
        "input_sha256",
        "no_evidence_results_sha256",
        "run_fingerprint",
    )
    mismatches = [
        field
        for field in config_fields
        if row.get(field) != expected_run_config.get(field)
    ]
    if mismatches:
        raise ValueError(f"Stored run metadata mismatch for {claim_id}: {mismatches}")

    source = eligible_by_id[claim_id]
    if row.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unexpected result_schema_version for {claim_id}.")
    if row.get("model_input_fields") != ["gold_claim", "oracle_evidence_text"]:
        raise ValueError(f"Unexpected model_input_fields for {claim_id}.")
    if row.get("response_id") != source["response_id"]:
        raise ValueError(f"response_id mismatch for {claim_id}.")
    if row.get("gold_claim") != source["gold_claim"]:
        raise ValueError(f"gold_claim mismatch for {claim_id}.")
    if row.get("human_label") != source["human_label"]:
        raise ValueError(f"human_label mismatch for {claim_id}.")
    if row.get("claim_sha256") != sha256_text(source["gold_claim"]):
        raise ValueError(f"claim_sha256 mismatch for {claim_id}.")
    if row.get("oracle_evidence") != source["_oracle_evidence"]:
        raise ValueError(f"oracle_evidence mismatch for {claim_id}.")

    evidence = row["oracle_evidence"]
    if evidence.get("status") != "ok" or not evidence.get("normalized_text"):
        raise ValueError(f"Invalid persisted oracle evidence for {claim_id}.")
    stored_hash = evidence.get("normalized_sha256")
    evidence_payload = {
        key: value for key, value in evidence.items() if key != "normalized_sha256"
    }
    if stored_hash != canonical_json_hash(evidence_payload):
        raise ValueError(f"oracle_evidence hash mismatch for {claim_id}.")
    if evidence.get("model_visible_fields") != ["text"]:
        raise ValueError(f"Unexpected model-visible evidence fields for {claim_id}.")
    if any("raw" in item for item in evidence.get("items", [])):
        raise ValueError(f"Raw evidence leaked into persisted normalized bundle: {claim_id}")

    status = row.get("status")
    if status not in {"ok", "request_error", "parse_error"}:
        raise ValueError(f"Invalid output status for {claim_id}: {status!r}")
    if status == "ok":
        parsed = parse_model_output(
            json.dumps(
                {
                    "prediction": row.get("prediction"),
                    "confidence": row.get("confidence"),
                    "rationale": row.get("rationale"),
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        if row.get("error") is not None:
            raise ValueError(f"Successful output has a non-null error for {claim_id}.")
        for key, value in parsed.items():
            if row.get(key) != value:
                raise ValueError(f"Normalized {key} mismatch for {claim_id}.")
    else:
        if any(
            row.get(key) is not None
            for key in ("prediction", "confidence", "rationale")
        ):
            raise ValueError(f"Technical failure contains a prediction for {claim_id}.")
        if not isinstance(row.get("error"), str) or not row["error"].strip():
            raise ValueError(f"Technical failure has no error message for {claim_id}.")


def load_existing_results(
    path: Path,
    eligible_by_id: dict[str, dict[str, Any]],
    expected_run_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = load_jsonl_objects(path)
    seen: set[str] = set()
    for row in rows:
        validate_result_record(row, eligible_by_id, expected_run_config)
        claim_id = row["claim_id"]
        if claim_id in seen:
            raise ValueError(f"Duplicate claim_id in oracle output: {claim_id}")
        seen.add(claim_id)
    return rows


def write_result(file: TextIO, result: dict[str, Any]) -> None:
    missing = sorted(RESULT_ALLOWED_FIELDS - set(result))
    extra = sorted(set(result) - RESULT_ALLOWED_FIELDS)
    if missing or extra:
        raise ValueError(f"Refusing to write invalid result fields: missing={missing}, extra={extra}")
    file.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    file.flush()


def atomic_write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for row in rows:
                write_result(temporary, row)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def safe_mean(values: Iterable[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return statistics.fmean(available) if available else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999)))
    return ordered[index]


def f1_from_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    support = tp + fn
    predicted = tp + fp
    if support == 0:
        precision = recall = f1 = None
    else:
        precision = tp / predicted if predicted else 0.0
        recall = tp / support
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "support": support,
        "predicted_count": predicted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def prediction_bucket(result: dict[str, Any] | None) -> str:
    if result is None or result.get("status") != "ok":
        return "ERROR_OR_MISSING"
    prediction = result.get("prediction")
    return prediction if prediction in LABELS else "ERROR_OR_MISSING"


def compute_binary_metrics(
    records: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    primary = [record for record in records if record["human_label"] in PRIMARY_LABELS]
    pairs = [(record, result_by_id.get(record["claim_id"])) for record in primary]
    correct = sum(
        result is not None
        and result.get("status") == "ok"
        and result.get("prediction") == record["human_label"]
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
        result is None or result.get("status") != "ok" for _, result in pairs
    )

    per_class: dict[str, dict[str, Any]] = {}
    for label in PRIMARY_LABELS:
        tp = sum(
            record["human_label"] == label
            and result is not None
            and result.get("status") == "ok"
            and result.get("prediction") == label
            for record, result in pairs
        )
        fp = sum(
            record["human_label"] != label
            and result is not None
            and result.get("status") == "ok"
            and result.get("prediction") == label
            for record, result in pairs
        )
        fn = sum(
            record["human_label"] == label
            and not (
                result is not None
                and result.get("status") == "ok"
                and result.get("prediction") == label
            )
            for record, result in pairs
        )
        per_class[label] = f1_from_counts(tp, fp, fn)

    confusion_columns = (*LABELS, "ERROR_OR_MISSING")
    confusion = {
        label: {column: 0 for column in confusion_columns} for label in PRIMARY_LABELS
    }
    for record, result in pairs:
        confusion[record["human_label"]][prediction_bucket(result)] += 1

    total = len(primary)
    macro_f1 = safe_mean(per_class[label]["f1"] for label in PRIMARY_LABELS)
    balanced_accuracy = safe_mean(
        per_class[label]["recall"] for label in PRIMARY_LABELS
    )
    return {
        "cohort_definition": "same human FACTUAL/NON_FACTUAL claim IDs",
        "gold_claim_count": total,
        "gold_label_counts": dict(Counter(record["human_label"] for record in primary)),
        "correct_count": correct,
        "accuracy_including_abstentions_and_errors": safe_ratio(correct, total),
        "balanced_accuracy": balanced_accuracy,
        "answered_count": len(answered),
        "coverage": safe_ratio(len(answered), total),
        "selective_accuracy": safe_ratio(correct, len(answered)),
        "model_unknown_count": unknown,
        "abstention_rate": safe_ratio(unknown, total),
        "technical_failure_count": failures,
        "FACTUAL": per_class["FACTUAL"],
        "NON_FACTUAL": per_class["NON_FACTUAL"],
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
    }


def build_response_aggregation(
    input_records: list[dict[str, Any]],
    eligible_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
    no_evidence_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    input_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in input_records:
        input_by_response[record["response_id"]].append(record)
    for record in eligible_records:
        eligible_by_response[record["response_id"]].append(record)
    for record in selected_records:
        selected_by_response[record["response_id"]].append(record)

    rows = []
    for response_id in sorted(input_by_response):
        all_rows = input_by_response[response_id]
        binary_rows = [row for row in all_rows if row["human_label"] in PRIMARY_LABELS]
        structural = [
            row
            for row in binary_rows
            if isinstance(row.get("gold_evidence"), (list, dict))
            and bool(row.get("gold_evidence"))
        ]
        eligible = eligible_by_response.get(response_id, [])
        selected = selected_by_response.get(response_id, [])
        metrics = compute_binary_metrics(selected, result_by_id)
        no_metrics = (
            compute_binary_metrics(selected, no_evidence_by_id)
            if no_evidence_by_id is not None
            else None
        )
        rows.append(
            {
                "response_id": response_id,
                "total_claims": len(all_rows),
                "binary_claims": len(binary_rows),
                "nonempty_evidence_bundle_binary_claims": len(structural),
                "oracle_eligible_claims": len(eligible),
                "oracle_eligibility_coverage_of_binary": safe_ratio(
                    len(eligible), len(binary_rows)
                ),
                "selected_claims_in_this_run": len(selected),
                "correct_count": metrics["correct_count"],
                "accuracy": metrics["accuracy_including_abstentions_and_errors"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "coverage": metrics["coverage"],
                "selective_accuracy": metrics["selective_accuracy"],
                "model_unknown_count": metrics["model_unknown_count"],
                "technical_failure_count": metrics["technical_failure_count"],
                "no_evidence_accuracy": (
                    None
                    if no_metrics is None
                    else no_metrics["accuracy_including_abstentions_and_errors"]
                ),
                "oracle_minus_no_evidence_accuracy": (
                    None
                    if no_metrics is None
                    or metrics["accuracy_including_abstentions_and_errors"] is None
                    or no_metrics[
                        "accuracy_including_abstentions_and_errors"
                    ] is None
                    else metrics["accuracy_including_abstentions_and_errors"]
                    - no_metrics[
                        "accuracy_including_abstentions_and_errors"
                    ]
                ),
            }
        )
    return rows


def build_confidence_analysis(
    selected_records: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
    high_confidence_threshold: float,
) -> dict[str, Any]:
    pairs = [
        (record, result_by_id.get(record["claim_id"]))
        for record in selected_records
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
    for record, result in ok_pairs:
        score = float(result["confidence"])
        exact_scores[f"{score:.6g}"] += 1
        by_prediction[str(result["prediction"])].append(score)
        if result["prediction"] == record["human_label"]:
            correct_scores.append(score)
        else:
            incorrect_scores.append(score)

    high_confidence_errors = [
        record["claim_id"]
        for record, result in ok_pairs
        if result["prediction"] != record["human_label"]
        and float(result["confidence"]) >= high_confidence_threshold
    ]
    return {
        "semantics": "self-reported confidence that the selected label is appropriate",
        "valid_prediction_count": len(ok_pairs),
        "exact_score_counts": dict(sorted(exact_scores.items(), key=lambda item: float(item[0]))),
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
        "high_confidence_error_count": len(high_confidence_errors),
        "high_confidence_error_claim_ids": high_confidence_errors,
        "calibration_warning": (
            "This scalar is not a full class-probability distribution; multiclass "
            "Brier score, log loss, and ECE are not computed."
        ),
    }


def evidence_excerpt(record: dict[str, Any], max_chars: int = 260) -> str:
    texts = [item["text"] for item in record["_oracle_evidence"]["items"]]
    text = " | ".join(texts)
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def build_typical_errors(
    selected_records: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    rows = []
    for record in selected_records:
        result = result_by_id.get(record["claim_id"])
        if (
            result is None
            or result.get("status") != "ok"
            or result.get("prediction") == record["human_label"]
        ):
            continue
        rows.append(
            {
                "claim_id": record["claim_id"],
                "response_id": record["response_id"],
                "gold_claim": record["gold_claim"],
                "human_label": record["human_label"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "rationale": result["rationale"],
                "evidence_excerpt": evidence_excerpt(record),
            }
        )
    rows.sort(key=lambda row: (-float(row["confidence"]), row["claim_id"]))
    return rows[:limit]


def build_summary(
    input_records: list[dict[str, Any]],
    eligible_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    cohort_audit: dict[str, Any],
    all_results: list[dict[str, Any]],
    no_evidence_results: list[dict[str, Any]],
    run_config: dict[str, Any],
    args: argparse.Namespace,
    started_at: str | None,
    finished_at: str,
) -> dict[str, Any]:
    selected_ids = [record["claim_id"] for record in selected_records]
    selected_id_set = set(selected_ids)
    result_by_id = {
        row["claim_id"]: row
        for row in all_results
        if row["claim_id"] in selected_id_set
    }
    no_evidence_by_id = {row["claim_id"]: row for row in no_evidence_results}
    paired_target_ids = [
        claim_id for claim_id in selected_ids if claim_id in no_evidence_by_id
    ]
    paired_result_ids = [
        claim_id for claim_id in paired_target_ids if claim_id in result_by_id
    ]
    paired_result_id_set = set(paired_result_ids)
    formal_full_scope = getattr(args, "scope", "pilot") == "full"
    paired_record_id_set = (
        set(paired_target_ids) if formal_full_scope else paired_result_id_set
    )
    paired_records = [
        record
        for record in selected_records
        if record["claim_id"] in paired_record_id_set
    ]

    status_counts = Counter(
        "missing"
        if record["claim_id"] not in result_by_id
        else str(result_by_id[record["claim_id"]].get("status"))
        for record in selected_records
    )
    prediction_counts = Counter(
        str(result_by_id[record["claim_id"]]["prediction"])
        for record in selected_records
        if record["claim_id"] in result_by_id
        and result_by_id[record["claim_id"]].get("status") == "ok"
    )
    oracle_metrics = compute_binary_metrics(selected_records, result_by_id)
    no_evidence_metrics = compute_binary_metrics(paired_records, no_evidence_by_id)
    matched_oracle_metrics = compute_binary_metrics(paired_records, result_by_id)
    paired_transitions = build_paired_transitions(
        paired_records, no_evidence_by_id, result_by_id
    )
    response_rows = build_response_aggregation(
        input_records,
        eligible_records,
        selected_records,
        result_by_id,
        no_evidence_by_id,
    )
    response_rows_with_selected = [
        row for row in response_rows if row["selected_claims_in_this_run"] > 0
    ]
    response_macro = {
        "cohort_definition": (
            "equal-weight mean across responses with selected matched claims"
        ),
        "response_count": len(response_rows_with_selected),
        "no_evidence_accuracy": safe_mean(
            row["no_evidence_accuracy"] for row in response_rows_with_selected
        ),
        "oracle_accuracy": safe_mean(
            row["accuracy"] for row in response_rows_with_selected
        ),
        "oracle_minus_no_evidence_accuracy": safe_mean(
            row["oracle_minus_no_evidence_accuracy"]
            for row in response_rows_with_selected
        ),
        "oracle_balanced_accuracy": safe_mean(
            row["balanced_accuracy"] for row in response_rows_with_selected
        ),
        "oracle_macro_f1": safe_mean(
            row["macro_f1"] for row in response_rows_with_selected
        ),
        "oracle_coverage": safe_mean(
            row["coverage"] for row in response_rows_with_selected
        ),
    }
    latencies = [
        float(row["latency_seconds"])
        for row in result_by_id.values()
        if isinstance(row.get("latency_seconds"), (int, float))
    ]
    result_count = len(result_by_id)
    ok_count = status_counts["ok"]

    return {
        "report_version": REPORT_VERSION,
        "generated_at": finished_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "setting": SETTING,
        "completion_status": (
            "complete"
            if result_count == len(selected_records) and ok_count == len(selected_records)
            else "incomplete"
        ),
        "row_completion_status": (
            "complete" if result_count == len(selected_records) else "incomplete"
        ),
        "prediction_status": (
            "all_parsed" if ok_count == len(selected_records) else "has_failures"
        ),
        "files": {
            "input": report_path(args.input),
            "output": report_path(args.output),
            "prompt": report_path(args.prompt),
            "no_evidence_results": report_path(args.no_evidence_results),
            "json_report": report_path(args.report),
            "markdown_report": report_path(args.markdown_report),
        },
        "run_config": run_config,
        "input_and_cohort": {
            **cohort_audit,
            "scope": getattr(args, "scope", "pilot"),
            "selection_source": getattr(args, "selection_source", "unknown"),
            "selected_claim_count": len(selected_records),
            "selected_claim_ids": selected_ids,
            "actual_result_count": result_count,
        },
        "technical_status": {
            "status_counts": dict(status_counts),
            "prediction_counts": dict(prediction_counts),
            "technical_success_rate": safe_ratio(ok_count, len(selected_records)),
            "unique_result_claim_ids": len(set(result_by_id)),
            "duplicate_result_claim_ids": 0,
            "duplicate_validation": "validated_by_loader",
            "missing_selected_claim_ids": [
                claim_id for claim_id in selected_ids if claim_id not in result_by_id
            ],
            "result_ids_outside_selection": sorted(
                {row["claim_id"] for row in all_results} - selected_id_set
            ),
        },
        "oracle_primary_binary_metrics": oracle_metrics,
        "paired_comparison": {
            "cohort_definition": (
                "full normalized matched target cohort; missing oracle rows remain "
                "technical failures in paired denominators"
                if formal_full_scope
                else "intersection of actual oracle result IDs and existing 08b "
                "no-evidence result IDs within the selected oracle target cohort"
            ),
            "paired_target_claim_count": len(paired_target_ids),
            "paired_target_claim_ids": paired_target_ids,
            "matched_claim_count": len(paired_records),
            "matched_claim_ids": [
                record["claim_id"] for record in paired_records
            ],
            "complete_pair_count": len(paired_result_ids),
            "complete_pair_claim_ids": paired_result_ids,
            "paired_target_ids_missing_oracle_results": [
                claim_id for claim_id in paired_target_ids if claim_id not in result_by_id
            ],
            "oracle_selected_ids_missing_from_no_evidence": [
                claim_id for claim_id in selected_ids if claim_id not in no_evidence_by_id
            ],
            "no_evidence_result_ids_outside_oracle_selection": sorted(
                set(no_evidence_by_id) - selected_id_set
            ),
            "no_evidence_metrics_on_matched_ids": no_evidence_metrics,
            "oracle_metrics_on_matched_ids": matched_oracle_metrics,
            "point_differences": metric_differences(
                no_evidence_metrics, matched_oracle_metrics
            ),
            "transitions": paired_transitions,
            "paired_response_cluster_bootstrap": {
                "status": "not_implemented",
                "todo": (
                    "Use identical response_id cluster resamples for no-evidence "
                    "and oracle metric differences; no reusable project tool exists."
                ),
            },
        },
        "response_aggregation": response_rows,
        "response_macro_metrics": response_macro,
        "confidence_analysis": build_confidence_analysis(
            selected_records, result_by_id, args.high_confidence_threshold
        ),
        "latency_seconds": {
            "count": len(latencies),
            "mean": safe_mean(latencies),
            "median": statistics.median(latencies) if latencies else None,
            "p95_nearest_rank": percentile(latencies, 0.95),
        },
        "typical_errors": build_typical_errors(
            selected_records, result_by_id, args.typical_error_limit
        ),
        "leakage_audit": {
            "model_input_fields": ["gold_claim", "oracle_evidence_text"],
            "model_visible_evidence_item_fields": ["text"],
            "evidence_metadata_saved_but_not_sent": [
                "url",
                "source",
                "rank",
                "stance",
            ],
            "item_raw_saved": False,
            "gold_label_not_in_constructed_model_prompt": True,
            "forbidden_record_fields_not_sent": sorted(FORBIDDEN_PROMPT_METADATA_FIELDS),
            "evaluation_join_key": "claim_id",
        },
        "interpretation_notes": [
            f"The {cohort_audit['evidence_available_binary_claims']} structurally "
            "evidence-available binary claims include "
            f"{cohort_audit['exclusion_reason_counts'].get('no_valid_evidence_text', 0)} "
            "bundles with only punctuation, marker, or URL text; the runnable "
            f"oracle cohort therefore contains {cohort_audit['oracle_eligible_claims']} claims.",
            "Model UNKNOWN and technical failures remain in overall accuracy and "
            "class-recall denominators.",
            "Coverage is the fraction receiving a valid FACTUAL/NON_FACTUAL decision.",
            f"Paired metrics never compare the broader {cohort_audit['binary_claims']}-"
            "claim no-evidence cohort directly with the matched oracle cohort.",
            "Evidence stance is trace metadata only and was not sent to the model.",
            "Response-level bootstrap intervals remain a documented follow-up TODO.",
        ],
    }


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_escape(value: Any, max_chars: int = 260) -> str:
    text = " ".join(str(value).split())
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text.replace("|", "\\|")


def build_markdown_report(summary: dict[str, Any]) -> str:
    cohort = summary["input_and_cohort"]
    technical = summary["technical_status"]
    metrics = summary["oracle_primary_binary_metrics"]
    paired = summary["paired_comparison"]
    no_metrics = paired["no_evidence_metrics_on_matched_ids"]
    matched_oracle = paired["oracle_metrics_on_matched_ids"]
    differences = paired["point_differences"]
    transitions = paired["transitions"]["named_transitions"]
    confidence = summary["confidence_analysis"]
    response_macro = summary["response_macro_metrics"]

    lines = [
        "# Oracle-Evidence Verifier Report",
        "",
        f"- Setting: `{summary['setting']}`",
        f"- Scope: `{cohort['scope']}`",
        f"- Model: `{summary['run_config']['model']}`",
        f"- Model digest: `{summary['run_config']['model_digest']}`",
        f"- Completion: **{summary['completion_status']}**",
        f"- Prompt version: `{summary['run_config']['prompt_version']}`",
        f"- Evidence normalization: `{summary['run_config']['evidence_normalization_version']}`",
        f"- Run fingerprint: `{summary['run_config']['run_fingerprint']}`",
        "",
        "## Input and automatically derived cohort",
        "",
        "| Count | Value |",
        "|---|---:|",
        f"| Total input claims | {cohort['total_input_claims']} |",
        f"| Total source responses | {cohort['total_input_responses']} |",
        f"| Binary human-labelled claims | {cohort['binary_claims']} |",
        f"| Human UNKNOWN claims | {cohort['human_unknown_claims']} |",
        f"| Binary claims with `evidence_available=true` | {cohort['evidence_available_binary_claims']} |",
        f"| Binary claims with non-empty evidence bundle | {cohort['nonempty_evidence_bundle_binary_claims']} |",
        f"| Oracle-eligible claims after text normalization | {cohort['oracle_eligible_claims']} |",
        f"| Oracle-eligible responses | {cohort['oracle_eligible_responses']} |",
        f"| Claims selected for this run | {cohort['selected_claim_count']} |",
        f"| Result rows for this run | {cohort['actual_result_count']} |",
        "",
        "### Exclusions",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in sorted(cohort["exclusion_reason_counts"].items()):
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(
        [
            "",
            f"The {cohort['exclusion_reason_counts'].get('no_valid_evidence_text', 0)} "
            "structurally non-empty but unusable bundles contain only a URL, "
            "`URL:`/`Link:`, or punctuation. They are excluded rather than treated "
            "as oracle passages.",
            "",
            "## Technical status",
            "",
            "| Status | Count |",
            "|---|---:|",
        ]
    )
    for status in ("ok", "request_error", "parse_error", "missing"):
        lines.append(f"| `{status}` | {technical['status_counts'].get(status, 0)} |")
    lines.extend(
        [
            f"| Technical success rate | {format_metric(technical['technical_success_rate'])} |",
            "",
            "## Oracle primary binary metrics",
            "",
            "Model `UNKNOWN` and technical failures remain in the overall denominator.",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Matched gold claims | {metrics['gold_claim_count']} |",
            f"| Correct | {metrics['correct_count']} |",
            f"| Accuracy | {format_metric(metrics['accuracy_including_abstentions_and_errors'])} |",
            f"| Balanced accuracy | {format_metric(metrics['balanced_accuracy'])} |",
            f"| Macro-F1 | {format_metric(metrics['macro_f1'])} |",
            f"| Coverage | {format_metric(metrics['coverage'])} |",
            f"| Selective accuracy | {format_metric(metrics['selective_accuracy'])} |",
            f"| Model UNKNOWN | {metrics['model_unknown_count']} |",
            f"| Technical failures | {metrics['technical_failure_count']} |",
            f"| FACTUAL precision | {format_metric(metrics['FACTUAL']['precision'])} |",
            f"| FACTUAL recall | {format_metric(metrics['FACTUAL']['recall'])} |",
            f"| FACTUAL F1 | {format_metric(metrics['FACTUAL']['f1'])} |",
            f"| NON_FACTUAL precision | {format_metric(metrics['NON_FACTUAL']['precision'])} |",
            f"| NON_FACTUAL recall | {format_metric(metrics['NON_FACTUAL']['recall'])} |",
            f"| NON_FACTUAL F1 | {format_metric(metrics['NON_FACTUAL']['f1'])} |",
            "",
            "### Confusion matrix",
            "",
            "| Human label \\ prediction | FACTUAL | NON_FACTUAL | UNKNOWN | Error/missing |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in PRIMARY_LABELS:
        row = metrics["confusion_matrix"][label]
        lines.append(
            f"| {label} | {row['FACTUAL']} | {row['NON_FACTUAL']} | "
            f"{row['UNKNOWN']} | {row['ERROR_OR_MISSING']} |"
        )

    comparison_rows = (
        ("Accuracy", "accuracy_including_abstentions_and_errors", "accuracy_oracle_minus_no_evidence"),
        ("Balanced accuracy", "balanced_accuracy", "balanced_accuracy_oracle_minus_no_evidence"),
        ("Macro-F1", "macro_f1", "macro_f1_oracle_minus_no_evidence"),
        ("Coverage", "coverage", "coverage_oracle_minus_no_evidence"),
        ("Selective accuracy", "selective_accuracy", "selective_accuracy_oracle_minus_no_evidence"),
    )
    lines.extend(
        [
            "",
            "## Paired no-evidence vs oracle comparison",
            "",
            f"All values below use the same **{paired['matched_claim_count']} claim IDs**. "
            f"The broader {cohort['binary_claims']}-claim no-evidence headline is not used.",
            "",
            "| Metric | No evidence | Oracle evidence | Oracle − no evidence |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, metric_key, difference_key in comparison_rows:
        lines.append(
            f"| {label} | {format_metric(no_metrics[metric_key])} | "
            f"{format_metric(matched_oracle[metric_key])} | "
            f"{format_metric(differences[difference_key])} |"
        )
    lines.extend(
        [
            "",
            "### Paired transitions",
            "",
            "`wrong` means the opposite definitive binary label; it does not include UNKNOWN.",
            "",
            "| Transition | Count |",
            "|---|---:|",
        ]
    )
    transition_labels = (
        ("wrong→correct", "wrong_to_correct"),
        ("correct→wrong", "correct_to_wrong"),
        ("UNKNOWN→correct decision", "unknown_to_correct_decision"),
        ("decision→UNKNOWN", "decision_to_unknown"),
        ("wrong→wrong", "wrong_to_wrong"),
        ("correct→correct", "correct_to_correct"),
        ("UNKNOWN→wrong decision", "unknown_to_wrong_decision"),
        ("UNKNOWN→UNKNOWN", "unknown_to_unknown"),
    )
    for label, key in transition_labels:
        lines.append(f"| {label} | {transitions[key]['count']} |")
    lines.extend(
        [
            "",
            "> Paired response-cluster bootstrap is intentionally not implemented here: "
            "the repository contains no reusable bootstrap module. The summary records a "
            "TODO to use identical response-cluster resamples for both settings.",
            "",
            "## Response-level aggregation",
            "",
            f"Rows retain all {cohort['total_input_responses']} claim-bearing "
            "response IDs present in the claim input, including responses with "
            "zero eligible claims.",
            "",
            f"Response-macro no-evidence accuracy: {format_metric(response_macro['no_evidence_accuracy'])}; "
            f"oracle accuracy: {format_metric(response_macro['oracle_accuracy'])}; "
            f"delta: {format_metric(response_macro['oracle_minus_no_evidence_accuracy'])}.",
            "",
            "| response_id | Binary | Non-empty bundle | Eligible | Selected | No-evidence acc. | Oracle acc. | Delta | Coverage | Macro-F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["response_aggregation"]:
        lines.append(
            f"| `{row['response_id']}` | {row['binary_claims']} | "
            f"{row['nonempty_evidence_bundle_binary_claims']} | "
            f"{row['oracle_eligible_claims']} | {row['selected_claims_in_this_run']} | "
            f"{format_metric(row['no_evidence_accuracy'])} | "
            f"{format_metric(row['accuracy'])} | "
            f"{format_metric(row['oracle_minus_no_evidence_accuracy'])} | "
            f"{format_metric(row['coverage'])} | "
            f"{format_metric(row['macro_f1'])} |"
        )

    lines.extend(
        [
            "",
            "## Confidence distribution",
            "",
            "| Self-reported score | Count |",
            "|---:|---:|",
        ]
    )
    for score, count in confidence["exact_score_counts"].items():
        lines.append(f"| {score} | {count} |")
    lines.extend(
        [
            "",
            f"- Mean confidence, correct: {format_metric(confidence['mean_confidence_correct'])}",
            "- Mean confidence, incorrect/abstained: "
            f"{format_metric(confidence['mean_confidence_incorrect_or_abstained'])}",
            f"- High-confidence errors: {confidence['high_confidence_error_count']}",
            "",
            "> Confidence is a scalar self-report for the selected label, not a full "
            "three-class probability distribution.",
            "",
            "## Typical errors and abstentions",
            "",
            "Evidence is shown only as a short excerpt; complete normalized evidence remains "
            "traceable in the JSONL.",
            "",
            "| claim_id | Gold | Prediction | Confidence | Rationale | Evidence excerpt |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for row in summary["typical_errors"]:
        lines.append(
            f"| `{row['claim_id']}` | `{row['human_label']}` | `{row['prediction']}` | "
            f"{format_metric(row['confidence'])} | {markdown_escape(row['rationale'])} | "
            f"{markdown_escape(row['evidence_excerpt'])} |"
        )
    if not summary["typical_errors"]:
        lines.append("| — | — | — | — | No errors/abstentions in this selection. | — |")

    lines.extend(
        [
            "",
            "## Leakage and reproducibility audit",
            "",
            "- Every request contained one JSON-encoded claim and normalized evidence text.",
            "- Human labels are persisted for evaluation but are never included in "
            "the constructed model prompt.",
            "- Evidence `stance`, annotation `source`, rank, URL, and item-level `raw` were "
            "not sent to the model; `raw` was not persisted in the normalized bundle.",
            "- The observed model digest was required to equal the historical 08b digest.",
            "- Input, prompt, schema, normalized cohort, and 08b results hashes are included "
            "in the run fingerprint.",
            "- Results are paired to 08b only by exact `claim_id`.",
            "",
            "## Interpretation cautions",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in summary["interpretation_notes"])
    lines.append("")
    return "\n".join(lines)


def write_reports(summary: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    atomic_write_text(
        json_path,
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    atomic_write_text(markdown_path, build_markdown_report(summary))


def print_dry_run(
    selected_records: list[dict[str, Any]],
    cohort_audit: dict[str, Any],
    prompt_template: str,
    run_config: dict[str, Any],
) -> None:
    print("=" * 80)
    print("ORACLE-EVIDENCE VERIFIER DRY RUN")
    print("=" * 80)
    print(f"Model: {run_config['model']}")
    print(f"Expected historical digest: {run_config['expected_no_evidence_model_digest']}")
    print("Local digest preflight: not performed in dry-run")
    print(f"Structurally evidence-available binary claims: {cohort_audit['evidence_available_binary_claims']}")
    print(f"Oracle-eligible claims after normalization: {cohort_audit['oracle_eligible_claims']}")
    print(f"Selected claims: {len(selected_records)}")
    print(f"Run fingerprint: {run_config['run_fingerprint']}")
    print("Model input fields: ['gold_claim', 'oracle_evidence_text']")
    for index, record in enumerate(selected_records, start=1):
        prompt = build_prompt(
            prompt_template, record["gold_claim"], record["_oracle_evidence"]
        )
        print()
        print(f"--- Preview {index}/{len(selected_records)}: {record['claim_id']} ---")
        print("The claim_id header is local audit output, not part of the model message.")
        print("MODEL MESSAGE START")
        print(prompt)
        print("MODEL MESSAGE END")
    print()
    print("No Ollama calls were made. No output or report files were written.")


def summary_exit_code(summary: dict[str, Any]) -> int:
    return 0 if summary.get("completion_status") == "complete" else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_records = load_input_records(args.input)
    eligible_records, cohort_audit = build_oracle_cohort(input_records)
    selected_records = select_records(
        eligible_records,
        cohort_audit,
        limit=args.limit,
        claim_ids=args.claim_id,
        smoke_test=args.smoke_test,
    )
    args.selection_source = (
        "explicit_cli_selection" if args.selection_explicit else "default_full_cohort"
    )
    if (
        args.report_only
        and args.scope == "pilot"
        and not args.selection_explicit
        and args.output.exists()
    ):
        unchecked_rows = load_jsonl_objects(args.output)
        stored_ids = [row.get("claim_id") for row in unchecked_rows]
        if any(not isinstance(claim_id, str) for claim_id in stored_ids):
            raise ValueError("Cannot recover report-only selection from invalid claim IDs.")
        if len(stored_ids) != len(set(stored_ids)):
            raise ValueError("Cannot recover report-only selection from duplicate claim IDs.")
        eligible_by_id_for_recovery = {
            record["claim_id"]: record for record in eligible_records
        }
        unknown_ids = sorted(set(stored_ids) - set(eligible_by_id_for_recovery))
        if unknown_ids:
            raise ValueError(
                "Cannot recover report-only selection; output contains ineligible IDs: "
                f"{unknown_ids}"
            )
        stored_id_set = set(stored_ids)
        selected_records = [
            record
            for record in eligible_records
            if record["claim_id"] in stored_id_set
        ]
        args.selection_source = "recovered_from_report_only_output_ids"
    partial_selection = len(selected_records) != len(eligible_records)
    if not args.dry_run:
        validate_write_path_safety(args, partial_selection)

    prompt_template = load_prompt_template(args.prompt)
    input_by_id = {record["claim_id"]: record for record in input_records}
    eligible_by_id = {record["claim_id"]: record for record in eligible_records}
    input_sha256 = sha256_file(args.input)
    no_evidence_results, baseline_profile = load_no_evidence_baseline(
        args.no_evidence_results,
        input_by_id,
        expected_input_sha256=input_sha256,
        required_claim_ids={record["claim_id"] for record in eligible_records},
    )
    validate_model_config_against_baseline(args, baseline_profile)
    expected_digest = baseline_profile["model_digest"]

    prompt_sha256 = sha256_text(prompt_template)
    no_evidence_results_sha256 = sha256_file(args.no_evidence_results)
    cohort_sha256 = cohort_manifest_hash(eligible_records)

    if args.dry_run:
        run_config = build_run_config(
            args,
            input_sha256,
            prompt_sha256,
            no_evidence_results_sha256,
            cohort_sha256,
            model_digest=expected_digest,
            expected_model_digest=expected_digest,
        )
        print_dry_run(selected_records, cohort_audit, prompt_template, run_config)
        return 0

    if args.report_only:
        if not args.output.exists():
            raise FileNotFoundError(
                f"Cannot use --report-only; output does not exist: {args.output}"
            )
        unchecked = load_jsonl_objects(args.output)
        stored_digests = {
            row.get("model_digest")
            for row in unchecked
            if isinstance(row.get("model_digest"), str) and row.get("model_digest")
        }
        if len(stored_digests) != 1:
            raise ValueError("Cannot recover one model digest from oracle output.")
        stored_digest = next(iter(stored_digests))
        if stored_digest != expected_digest:
            raise ValueError(
                "Stored oracle digest does not match historical 08b digest: "
                f"oracle={stored_digest}, 08b={expected_digest}"
            )
        run_config = build_run_config(
            args,
            input_sha256,
            prompt_sha256,
            no_evidence_results_sha256,
            cohort_sha256,
            model_digest=stored_digest,
            expected_model_digest=expected_digest,
        )
        all_results = load_existing_results(
            args.output, eligible_by_id, run_config
        )
        finished_at = utc_now()
        summary = build_summary(
            input_records,
            eligible_records,
            selected_records,
            cohort_audit,
            all_results,
            no_evidence_results,
            run_config,
            args,
            started_at=None,
            finished_at=finished_at,
        )
        write_reports(summary, args.report, args.markdown_report)
        print(f"JSON report: {args.report}")
        print(f"Markdown report: {args.markdown_report}")
        return summary_exit_code(summary)

    if args.output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --resume, --overwrite, "
            "or a new output path."
        )

    client = _base.make_client(args)
    observed_digest = preflight_ollama(client, args.model)
    if observed_digest != expected_digest:
        raise ValueError(
            "Installed model digest does not match the historical 08b run; no "
            f"oracle output was changed. observed={observed_digest}, 08b={expected_digest}"
        )
    run_config = build_run_config(
        args,
        input_sha256,
        prompt_sha256,
        no_evidence_results_sha256,
        cohort_sha256,
        model_digest=observed_digest,
        expected_model_digest=expected_digest,
    )

    existing_results: list[dict[str, Any]] = []
    rewrite_resume_output = False
    if args.output.exists():
        if args.overwrite:
            output_mode = "w"
        else:
            existing_results = load_existing_results(
                args.output, eligible_by_id, run_config
            )
            output_mode = "a"
    else:
        output_mode = "w"

    if args.resume and existing_results:
        selected_id_set = {record["claim_id"] for record in selected_records}
        retained = [
            row
            for row in existing_results
            if row["claim_id"] not in selected_id_set or row.get("status") == "ok"
        ]
        rewrite_resume_output = len(retained) != len(existing_results)
        existing_results = retained

    existing_ids = {row["claim_id"] for row in existing_results}
    pending_records = [
        record for record in selected_records if record["claim_id"] not in existing_ids
    ]
    started_at = utc_now()

    if rewrite_resume_output:
        atomic_write_results(args.output, existing_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ORACLE-EVIDENCE VERIFIER")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Scope: {args.scope}")
    print(f"Observed/expected digest: {observed_digest}")
    print(f"Input claims: {cohort_audit['total_input_claims']}")
    print(f"Binary claims: {cohort_audit['binary_claims']}")
    print(
        "Structurally evidence-available binary claims: "
        f"{cohort_audit['evidence_available_binary_claims']}"
    )
    print(f"Oracle eligible after normalization: {len(eligible_records)}")
    print(f"Selected: {len(selected_records)}")
    print(f"Existing compatible results: {len(existing_results)}")
    print(f"Pending: {len(pending_records)}")

    new_results: list[dict[str, Any]] = []
    consecutive_request_errors = 0
    if pending_records:
        with args.output.open(output_mode, encoding="utf-8") as output_file:
            for index, record in enumerate(pending_records, start=1):
                print(
                    f"[{index}/{len(pending_records)}] Verifying {record['claim_id']}",
                    flush=True,
                )
                result = process_record(
                    record, prompt_template, run_config, client
                )
                write_result(output_file, result)
                new_results.append(result)
                print(
                    f"  status={result['status']} prediction={result.get('prediction')}",
                    flush=True,
                )
                if result["status"] == "request_error":
                    consecutive_request_errors += 1
                else:
                    consecutive_request_errors = 0
                if (
                    consecutive_request_errors
                    >= args.max_consecutive_request_errors
                ):
                    print(
                        "Stopping early after consecutive request failures; remaining "
                        "claims can be retried with --resume.",
                        flush=True,
                    )
                    break

    all_results = (
        load_existing_results(args.output, eligible_by_id, run_config)
        if args.output.exists()
        else new_results
    )
    finished_at = utc_now()
    summary = build_summary(
        input_records,
        eligible_records,
        selected_records,
        cohort_audit,
        all_results,
        no_evidence_results,
        run_config,
        args,
        started_at=started_at,
        finished_at=finished_at,
    )
    write_reports(summary, args.report, args.markdown_report)

    print("=" * 80)
    print("ORACLE-EVIDENCE VERIFIER FINISHED")
    print("=" * 80)
    print(f"Output: {args.output}")
    print(f"JSON report: {args.report}")
    print(f"Markdown report: {args.markdown_report}")
    print(f"Status counts: {summary['technical_status']['status_counts']}")
    print(
        "Primary accuracy: "
        f"{summary['oracle_primary_binary_metrics']['accuracy_including_abstentions_and_errors']}"
    )
    print(f"Coverage: {summary['oracle_primary_binary_metrics']['coverage']}")
    return summary_exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
