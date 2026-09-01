#!/usr/bin/env python3
"""Run a leakage-safe, closed-book factuality verifier with local Ollama.

The model receives one fresh request per claim. The only dynamic benchmark
value inserted into the request is ``gold_claim``. Gold labels and evidence
are joined by ``claim_id`` only after inference when reports are built.

Recommended sequence:

1. Dry-run three representative claims (zero model calls).
2. Run the same three claims into dedicated smoke-test outputs.
3. Inspect the smoke reports.
4. Run or resume the selected pilot/full cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

from dotenv import load_dotenv
from ollama import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factcheck_bench_analysis import (  # noqa: E402
    build_confidence_distribution,
    build_response_aggregation,
    compute_binary_metrics,
)
from factcheck_bench_pipeline import (  # noqa: E402
    FROZEN_QWEN3_8B_DIGEST,
    SCOPES,
    build_oracle_cohort,
    paths_for_scope,
)

load_dotenv(PROJECT_ROOT / ".env")


def report_path(path: Path) -> str:
    """Use repository-relative paths in portable generated reports."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "factcheck_bench"
    / "processed"
    / "fcb_gold_claims_pilot_20.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_pilot"
    / "jsonl"
    / "08b_no_evidence_verifier_results.jsonl"
)
DEFAULT_JSON_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_pilot"
    / "reports"
    / "08b_no_evidence_verifier_summary.json"
)
DEFAULT_MARKDOWN_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_pilot"
    / "reports"
    / "08b_no_evidence_verifier_report.md"
)
DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "no_evidence_verifier.txt"
FULL_PATHS = paths_for_scope(PROJECT_ROOT, "full")

LABELS = ("FACTUAL", "NON_FACTUAL", "UNKNOWN")
PRIMARY_LABELS = ("FACTUAL", "NON_FACTUAL")
PROMPT_PLACEHOLDER = "{gold_claim_json}"
PROMPT_VERSION = "no_evidence_v1"
OUTPUT_SCHEMA_VERSION = "no_evidence_output_v1"
RESULT_SCHEMA_VERSION = "no_evidence_result_v1"
REPORT_VERSION = "no_evidence_report_v1"
SETTING = "no_evidence"

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prediction", "confidence", "rationale"],
    "properties": {
        "prediction": {
            "type": "string",
            "enum": list(LABELS),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "rationale": {
            "type": "string",
            "minLength": 1,
            "maxLength": 240,
        },
    },
}

FORBIDDEN_RESULT_FIELDS = {
    "human_label",
    "human_label_bool",
    "human_label_raw",
    "is_binary_evaluable",
    "gold_evidence",
    "gold_evidence_texts",
    "raw_auto_evidence",
    "raw_auto_evidence_urls",
    "raw_auto_evidence_stances",
    "raw_human_evidence",
    "claim_needs_edit",
    "revised_claim",
    "revision_evidence_index",
}

RESULT_ALLOWED_FIELDS = {
    "result_schema_version",
    "claim_id",
    "response_id",
    "gold_claim",
    "claim_sha256",
    "setting",
    "model_input_fields",
    "model",
    "model_digest",
    "temperature",
    "seed",
    "num_predict",
    "think",
    "timeout_seconds",
    "max_retries",
    "max_consecutive_request_errors",
    "prompt_version",
    "prompt_sha256",
    "output_schema_version",
    "output_schema_sha256",
    "input_sha256",
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FactCheck-Bench no-evidence verifier with Ollama."
    )
    parser.add_argument("--scope", choices=SCOPES, default="pilot")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=None,
    )
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--model",
        default=os.getenv("CHAT_MODEL", "qwen3:8b"),
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--cohort",
        choices=("all", "binary", "matched", "human-unknown"),
        default=None,
        help=(
            "Claims to run before any --limit/--claim-id selection. Defaults "
            "to all for pilot and binary for full."
        ),
    )
    parser.add_argument(
        "--expected-model-digest",
        default=None,
        help=(
            "Require this installed Ollama digest before writing. Full scope "
            "defaults to the digest frozen by the completed pilot."
        ),
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Select the first N claims.",
    )
    selection.add_argument(
        "--claim-id",
        action="append",
        default=None,
        help="Select an exact claim ID; repeat this option for multiple IDs.",
    )
    selection.add_argument(
        "--smoke-test",
        action="store_true",
        help="Select one deterministic claim for each human gold label.",
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Number of retries after transport/request failures.",
    )
    parser.add_argument(
        "--max-consecutive-request-errors",
        type=int,
        default=3,
        help=(
            "Stop the run after this many consecutive request failures; "
            "remaining claims stay pending for --resume."
        ),
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.8,
    )

    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Append only claim IDs not already present in a compatible output.",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing output file.",
    )

    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--dry-run",
        "--preview",
        action="store_true",
        dest="dry_run",
        help="Print model messages and exit without calling Ollama or writing files.",
    )
    run_mode.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate reports from an existing result JSONL without model calls.",
    )

    args = parser.parse_args(argv)
    defaults = paths_for_scope(PROJECT_ROOT, args.scope)
    args.input = args.input or defaults.gold_claims
    args.output = args.output or defaults.no_evidence_output
    args.report = args.report or defaults.no_evidence_report
    args.markdown_report = (
        args.markdown_report or defaults.no_evidence_markdown
    )
    args.cohort = args.cohort or (
        "all" if args.scope == "pilot" else "binary"
    )
    if args.expected_model_digest is None and args.scope == "full":
        args.expected_model_digest = FROZEN_QWEN3_8B_DIGEST
    validate_args(parser, args)
    return args


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
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
    if args.expected_model_digest is not None and not re.fullmatch(
        r"[0-9a-fA-F]{64}", args.expected_model_digest
    ):
        parser.error("--expected-model-digest must be a 64-character hex digest")
    if args.report_only and (args.resume or args.overwrite):
        parser.error("--report-only cannot be combined with --resume/--overwrite")


def canonical_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def validate_write_path_safety(
    args: argparse.Namespace,
    partial_selection: bool,
) -> None:
    """Keep inputs, prompts, pilot history, and formal paths distinct."""
    destinations = {
        "json_report": canonical_path(args.report),
        "markdown_report": canonical_path(args.markdown_report),
    }
    if not args.report_only:
        destinations["output"] = canonical_path(args.output)
    if len(set(destinations.values())) != len(destinations):
        raise ValueError(f"Output/report paths must be distinct: {destinations}")

    protected = {
        canonical_path(args.input),
        canonical_path(args.prompt),
        canonical_path(Path(__file__)),
    }
    if args.report_only:
        protected.add(canonical_path(args.output))
    collisions = {
        name: str(path)
        for name, path in destinations.items()
        if path in protected
    }
    if collisions:
        raise ValueError(
            "Refusing to overwrite an input, prompt, or script: "
            f"{collisions}"
        )

    scope_paths = paths_for_scope(PROJECT_ROOT, args.scope)
    if partial_selection:
        formal = {
            canonical_path(scope_paths.no_evidence_output),
            canonical_path(scope_paths.no_evidence_report),
            canonical_path(scope_paths.no_evidence_markdown),
        }
        formal_collisions = {
            name: str(path)
            for name, path in destinations.items()
            if path in formal
        }
        if formal_collisions:
            raise ValueError(
                "A partial/smoke run may not use formal scope output paths: "
                f"{formal_collisions}"
            )

    if args.scope == "full":
        pilot = paths_for_scope(PROJECT_ROOT, "pilot")
        pilot_root = canonical_path(pilot.output_root)
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


def load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(
                    line,
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(
                    f"Line {line_number} of {path} must be a JSON object."
                )
            records.append(record)

    if not records:
        raise ValueError(f"JSONL file contains no records: {path}")
    return records


def load_input_records(path: Path) -> list[dict[str, Any]]:
    records = load_jsonl_objects(path)
    seen_ids: set[str] = set()

    for line_number, record in enumerate(records, start=1):
        claim_id = record.get("claim_id")
        response_id = record.get("response_id")
        gold_claim = record.get("gold_claim")
        human_label = record.get("human_label")
        human_label_bool = record.get("human_label_bool")
        is_binary = record.get("is_binary_evaluable")

        if not isinstance(claim_id, str) or not claim_id.strip():
            raise ValueError(f"Input line {line_number}: invalid claim_id.")
        if claim_id in seen_ids:
            raise ValueError(f"Duplicate input claim_id: {claim_id}")
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError(f"Input line {line_number}: invalid response_id.")
        if not isinstance(gold_claim, str) or not gold_claim.strip():
            raise ValueError(f"Input line {line_number}: invalid gold_claim.")
        if human_label not in LABELS:
            raise ValueError(
                f"Input line {line_number}: invalid human_label {human_label!r}."
            )
        if not isinstance(is_binary, bool):
            raise TypeError(
                f"Input line {line_number}: is_binary_evaluable must be boolean."
            )

        expected_bool: bool | None
        if human_label == "FACTUAL":
            expected_bool = True
        elif human_label == "NON_FACTUAL":
            expected_bool = False
        else:
            expected_bool = None

        if human_label_bool is not expected_bool:
            raise ValueError(
                f"Input line {line_number}: human_label_bool is inconsistent "
                f"with {human_label}."
            )
        if is_binary != (human_label in PRIMARY_LABELS):
            raise ValueError(
                f"Input line {line_number}: is_binary_evaluable is inconsistent "
                f"with {human_label}."
            )

        seen_ids.add(claim_id)

    return records


def filter_records_by_cohort(
    records: list[dict[str, Any]],
    cohort: str,
) -> list[dict[str, Any]]:
    """Apply a named cohort before any debugging/claim-ID selection."""
    if cohort == "all":
        return records
    if cohort == "binary":
        return [record for record in records if record["is_binary_evaluable"]]
    if cohort == "human-unknown":
        return [record for record in records if record["human_label"] == "UNKNOWN"]
    if cohort == "matched":
        eligible, _ = build_oracle_cohort(records)
        eligible_ids = {record["claim_id"] for record in eligible}
        return [record for record in records if record["claim_id"] in eligible_ids]
    raise ValueError(f"Unknown cohort: {cohort}")


def select_records(
    records: list[dict[str, Any]],
    limit: int | None = None,
    claim_ids: list[str] | None = None,
    smoke_test: bool = False,
) -> list[dict[str, Any]]:
    if claim_ids is not None:
        by_id = {record["claim_id"]: record for record in records}
        missing = [claim_id for claim_id in claim_ids if claim_id not in by_id]
        if missing:
            raise ValueError(f"Requested claim IDs were not found: {missing}")
        return [by_id[claim_id] for claim_id in claim_ids]

    if smoke_test:
        selected: list[dict[str, Any]] = []
        smoke_labels = (
            LABELS
            if any(record["human_label"] == "UNKNOWN" for record in records)
            else PRIMARY_LABELS
        )
        for label in smoke_labels:
            match = next(
                (record for record in records if record["human_label"] == label),
                None,
            )
            if match is None:
                raise ValueError(f"No {label} claim is available for smoke test.")
            selected.append(match)
        return selected

    if limit is not None:
        return records[:limit]
    return records


def load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"Prompt file is empty: {path}")
    if template.count(PROMPT_PLACEHOLDER) != 1:
        raise ValueError(
            f"Prompt must contain exactly one {PROMPT_PLACEHOLDER!r}."
        )
    placeholders = set(re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", template))
    if placeholders != {PROMPT_PLACEHOLDER}:
        raise ValueError(f"Unexpected prompt placeholders: {sorted(placeholders)}")
    return template


def build_prompt(template: str, gold_claim: str) -> str:
    if not isinstance(gold_claim, str) or not gold_claim.strip():
        raise ValueError("gold_claim must be a non-empty string.")
    claim_json = json.dumps(gold_claim.strip(), ensure_ascii=False)
    return template.replace(PROMPT_PLACEHOLDER, claim_json)


def build_run_config(
    args: argparse.Namespace,
    input_sha256: str,
    prompt_sha256: str,
    model_digest: str,
) -> dict[str, Any]:
    schema_sha256 = canonical_json_hash(OUTPUT_SCHEMA)
    fingerprint_payload = {
        "setting": SETTING,
        "model": args.model,
        "model_digest": model_digest,
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
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "output_schema_sha256": schema_sha256,
        "input_sha256": input_sha256,
    }
    return {
        **fingerprint_payload,
        "run_fingerprint": canonical_json_hash(fingerprint_payload),
    }


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def parse_model_output(raw_output: str) -> dict[str, Any]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("Model output is empty.")
    try:
        parsed = json.loads(
            raw_output.strip(),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Model output is not strict JSON: {error}") from error

    if not isinstance(parsed, dict):
        raise TypeError("Model output must be one JSON object.")
    expected_keys = {"prediction", "confidence", "rationale"}
    actual_keys = set(parsed)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"Model output keys do not match schema; missing={missing}, extra={extra}."
        )

    prediction = parsed["prediction"]
    confidence = parsed["confidence"]
    rationale = parsed["rationale"]

    if not isinstance(prediction, str) or prediction not in LABELS:
        raise ValueError(f"Invalid prediction label: {prediction!r}")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be a JSON number, not boolean/string.")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and between 0.0 and 1.0.")
    if not isinstance(rationale, str):
        raise TypeError("rationale must be a string.")
    rationale = rationale.strip()
    if not rationale:
        raise ValueError("rationale cannot be empty.")
    if len(rationale) > 240:
        raise ValueError("rationale cannot exceed 240 characters.")
    if len(rationale.split()) > 35:
        raise ValueError("rationale cannot exceed 35 words.")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "rationale": rationale,
    }


def response_value(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


def extract_response(response: Any) -> tuple[str, dict[str, Any]]:
    message = response_value(response, "message")
    if message is None:
        raise ValueError("Ollama response has no message.")
    content = response_value(message, "content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama response message is empty.")

    metadata_keys = (
        "model",
        "created_at",
        "done",
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    )
    metadata = {
        key: response_value(response, key)
        for key in metadata_keys
        if response_value(response, key) is not None
    }
    return content.strip(), metadata


def call_ollama(
    client: Any,
    run_config: dict[str, Any],
    prompt: str,
) -> tuple[str, dict[str, Any]]:
    response = client.chat(
        model=run_config["model"],
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        think=False,
        format=OUTPUT_SCHEMA,
        options={
            "temperature": run_config["temperature"],
            "seed": run_config["seed"],
            "num_predict": run_config["num_predict"],
        },
    )
    return extract_response(response)


def create_result_base(
    record: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "claim_id": record["claim_id"],
        "response_id": record["response_id"],
        "gold_claim": record["gold_claim"],
        "claim_sha256": sha256_text(record["gold_claim"]),
        "setting": SETTING,
        "model_input_fields": ["gold_claim"],
        "model": run_config["model"],
        "model_digest": run_config["model_digest"],
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
        "output_schema_version": run_config["output_schema_version"],
        "output_schema_sha256": run_config["output_schema_sha256"],
        "input_sha256": run_config["input_sha256"],
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
    prompt = build_prompt(prompt_template, record["gold_claim"])
    started = time.perf_counter()
    raw_output: str | None = None
    metadata: dict[str, Any] = {}
    request_error: Exception | None = None
    attempts = 0

    for attempt in range(run_config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw_output, metadata = call_ollama(client, run_config, prompt)
            request_error = None
            break
        except Exception as error:  # transport and malformed Ollama envelopes
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
    record: dict[str, Any],
    input_by_id: dict[str, dict[str, Any]],
    expected_run_config: dict[str, Any],
) -> None:
    claim_id = record.get("claim_id")
    if not isinstance(claim_id, str) or claim_id not in input_by_id:
        raise ValueError(f"Output contains unknown claim_id: {claim_id!r}")
    unexpected_fields = sorted(set(record) - RESULT_ALLOWED_FIELDS)
    missing_fields = sorted(RESULT_ALLOWED_FIELDS - set(record))
    if unexpected_fields or missing_fields:
        raise ValueError(
            f"Output fields do not match {RESULT_SCHEMA_VERSION}; "
            f"missing={missing_fields}, extra={unexpected_fields}."
        )
    if record.get("run_fingerprint") != expected_run_config["run_fingerprint"]:
        raise ValueError(
            "Existing output is incompatible with this input/prompt/model/config. "
            "Use a new output path or --overwrite."
        )
    config_fields = (
        "setting",
        "model",
        "model_digest",
        "temperature",
        "seed",
        "num_predict",
        "think",
        "timeout_seconds",
        "max_retries",
        "max_consecutive_request_errors",
        "prompt_version",
        "prompt_sha256",
        "output_schema_version",
        "output_schema_sha256",
        "input_sha256",
        "run_fingerprint",
    )
    mismatched_config = [
        field
        for field in config_fields
        if record.get(field) != expected_run_config.get(field)
    ]
    if mismatched_config:
        raise ValueError(
            f"Stored run metadata mismatch for {claim_id}: {mismatched_config}"
        )
    leaked_fields = sorted(FORBIDDEN_RESULT_FIELDS.intersection(record))
    if leaked_fields:
        raise ValueError(f"Prediction output contains forbidden gold fields: {leaked_fields}")
    if record.get("model_input_fields") != ["gold_claim"]:
        raise ValueError(f"Unexpected model_input_fields for {claim_id}.")
    if record.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unexpected result_schema_version for {claim_id}.")
    if record.get("setting") != SETTING:
        raise ValueError(f"Unexpected setting for {claim_id}.")

    input_record = input_by_id[claim_id]
    expected_claim = input_record["gold_claim"]
    if record.get("response_id") != input_record["response_id"]:
        raise ValueError(f"response_id mismatch for {claim_id}.")
    if record.get("gold_claim") != expected_claim:
        raise ValueError(f"gold_claim mismatch for {claim_id}.")
    if record.get("claim_sha256") != sha256_text(expected_claim):
        raise ValueError(f"claim_sha256 mismatch for {claim_id}.")

    status = record.get("status")
    if status not in {"ok", "request_error", "parse_error"}:
        raise ValueError(f"Invalid output status for {claim_id}.")
    if status == "ok":
        try:
            parsed_again = parse_model_output(
                json.dumps(
                    {
                        "prediction": record.get("prediction"),
                        "confidence": record.get("confidence"),
                        "rationale": record.get("rationale"),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid parsed prediction for {claim_id}: {error}") from error
        if record.get("error") is not None:
            raise ValueError(f"Successful output has a non-null error for {claim_id}.")
        for key, value in parsed_again.items():
            if record.get(key) != value:
                raise ValueError(f"Normalised {key} mismatch for {claim_id}.")
    else:
        if any(record.get(key) is not None for key in ("prediction", "confidence", "rationale")):
            raise ValueError(f"Technical failure contains a prediction for {claim_id}.")
        error_text = record.get("error")
        if not isinstance(error_text, str) or not error_text.strip():
            raise ValueError(f"Technical failure has no error message for {claim_id}.")


def load_existing_results(
    path: Path,
    input_by_id: dict[str, dict[str, Any]],
    expected_run_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = load_jsonl_objects(path)
    seen_ids: set[str] = set()
    for record in records:
        validate_result_record(record, input_by_id, expected_run_config)
        claim_id = record["claim_id"]
        if claim_id in seen_ids:
            raise ValueError(f"Duplicate claim_id in output: {claim_id}")
        seen_ids.add(claim_id)
    return records


def write_result(file: TextIO, result: dict[str, Any]) -> None:
    leaked_fields = sorted(FORBIDDEN_RESULT_FIELDS.intersection(result))
    if leaked_fields:
        raise ValueError(f"Refusing to write leaked gold fields: {leaked_fields}")
    file.write(json.dumps(result, ensure_ascii=False) + "\n")
    file.flush()


def atomic_write_results(path: Path, results: list[dict[str, Any]]) -> None:
    """Replace a result JSONL atomically after validating every written row."""
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
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for result in results:
                write_result(temporary_file, result)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def safe_mean(values: Iterable[float]) -> float | None:
    values_list = list(values)
    if not values_list:
        return None
    return statistics.fmean(values_list)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def f1_from_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    predicted_count = tp + fp
    gold_support = tp + fn

    # Use the conventional zero_division=0 behaviour when the class exists in
    # the gold cohort but the verifier never predicts it. Returning ``None`` in
    # that case would incorrectly omit a failed class from macro F1.
    if gold_support == 0:
        precision = None
        recall = None
        f1 = None
    else:
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / gold_support
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
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
    return prediction if prediction in LABELS else "ERROR_OR_MISSING"


def build_summary(
    selected_records: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
    run_config: dict[str, Any],
    args: argparse.Namespace,
    started_at: str | None,
    finished_at: str,
) -> dict[str, Any]:
    selected_ids = [record["claim_id"] for record in selected_records]
    selected_id_set = set(selected_ids)
    result_by_id = {
        result["claim_id"]: result
        for result in all_results
        if result["claim_id"] in selected_id_set
    }

    status_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    gold_counts = Counter(record["human_label"] for record in selected_records)
    confusion_rows = (*LABELS,)
    confusion_columns = (*LABELS, "ERROR_OR_MISSING")
    confusion = {
        gold: {prediction: 0 for prediction in confusion_columns}
        for gold in confusion_rows
    }

    for record in selected_records:
        result = result_by_id.get(record["claim_id"])
        if result is None:
            status_counts["missing"] += 1
        else:
            status_counts[str(result.get("status"))] += 1
            if result.get("status") == "ok":
                prediction_counts[str(result["prediction"])] += 1
        confusion[record["human_label"]][prediction_bucket(result)] += 1

    primary_records = [
        record for record in selected_records if record["is_binary_evaluable"]
    ]
    primary_total = len(primary_records)
    primary_pairs = [
        (record, result_by_id.get(record["claim_id"]))
        for record in primary_records
    ]
    correct = sum(
        result is not None
        and result.get("status") == "ok"
        and result.get("prediction") == record["human_label"]
        for record, result in primary_pairs
    )
    answered_pairs = [
        (record, result)
        for record, result in primary_pairs
        if result is not None
        and result.get("status") == "ok"
        and result.get("prediction") in PRIMARY_LABELS
    ]
    abstained_count = sum(
        result is not None
        and result.get("status") == "ok"
        and result.get("prediction") == "UNKNOWN"
        for _, result in primary_pairs
    )
    technical_failure_count = sum(
        result is None or result.get("status") != "ok"
        for _, result in primary_pairs
    )

    non_factual_tp = sum(
        record["human_label"] == "NON_FACTUAL"
        and result is not None
        and result.get("status") == "ok"
        and result.get("prediction") == "NON_FACTUAL"
        for record, result in primary_pairs
    )
    non_factual_fp = sum(
        record["human_label"] == "FACTUAL"
        and result is not None
        and result.get("status") == "ok"
        and result.get("prediction") == "NON_FACTUAL"
        for record, result in primary_pairs
    )
    non_factual_fn = sum(
        record["human_label"] == "NON_FACTUAL"
        and not (
            result is not None
            and result.get("status") == "ok"
            and result.get("prediction") == "NON_FACTUAL"
        )
        for record, result in primary_pairs
    )
    factual_tp = sum(
        record["human_label"] == "FACTUAL"
        and result is not None
        and result.get("status") == "ok"
        and result.get("prediction") == "FACTUAL"
        for record, result in primary_pairs
    )
    factual_fp = sum(
        record["human_label"] == "NON_FACTUAL"
        and result is not None
        and result.get("status") == "ok"
        and result.get("prediction") == "FACTUAL"
        for record, result in primary_pairs
    )
    factual_fn = sum(
        record["human_label"] == "FACTUAL"
        and not (
            result is not None
            and result.get("status") == "ok"
            and result.get("prediction") == "FACTUAL"
        )
        for record, result in primary_pairs
    )
    factual_metrics = f1_from_counts(factual_tp, factual_fp, factual_fn)
    non_factual_metrics = f1_from_counts(
        non_factual_tp,
        non_factual_fp,
        non_factual_fn,
    )
    f1_values = [
        value
        for value in (factual_metrics["f1"], non_factual_metrics["f1"])
        if value is not None
    ]

    confidence_pairs = [
        (record, result)
        for record, result in primary_pairs
        if result is not None and result.get("status") == "ok"
    ]
    correct_confidences = [
        float(result["confidence"])
        for record, result in confidence_pairs
        if result["prediction"] == record["human_label"]
    ]
    incorrect_confidences = [
        float(result["confidence"])
        for record, result in confidence_pairs
        if result["prediction"] != record["human_label"]
    ]
    high_confidence_errors = sum(
        result["prediction"] != record["human_label"]
        and float(result["confidence"]) >= args.high_confidence_threshold
        for record, result in confidence_pairs
    )
    high_confidence_abstentions = sum(
        result["prediction"] == "UNKNOWN"
        and float(result["confidence"]) >= args.high_confidence_threshold
        for _, result in confidence_pairs
    )
    high_confidence_wrong_decisions = sum(
        result["prediction"] in PRIMARY_LABELS
        and result["prediction"] != record["human_label"]
        and float(result["confidence"]) >= args.high_confidence_threshold
        for record, result in confidence_pairs
    )

    human_unknown_records = [
        record for record in selected_records if record["human_label"] == "UNKNOWN"
    ]
    human_unknown_predictions: Counter[str] = Counter()
    for record in human_unknown_records:
        result = result_by_id.get(record["claim_id"])
        human_unknown_predictions[prediction_bucket(result)] += 1

    latencies = [
        float(result["latency_seconds"])
        for result in result_by_id.values()
        if isinstance(result.get("latency_seconds"), (int, float))
    ]
    output_ids = {result["claim_id"] for result in all_results}
    selected_result_count = len(result_by_id)
    ok_count = status_counts["ok"]
    shared_metrics = compute_binary_metrics(primary_records, result_by_id)
    response_rows, response_macro = build_response_aggregation(
        primary_records, result_by_id
    )
    confidence_distribution = build_confidence_distribution(
        primary_records,
        result_by_id,
        args.high_confidence_threshold,
    )

    return {
        "report_version": REPORT_VERSION,
        "generated_at": finished_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "setting": SETTING,
        "completion_status": (
            "complete"
            if selected_result_count == len(selected_records)
            and ok_count == len(selected_records)
            else "incomplete"
        ),
        "row_completion_status": (
            "complete" if selected_result_count == len(selected_records) else "incomplete"
        ),
        "prediction_status": (
            "all_parsed" if ok_count == len(selected_records) else "has_failures"
        ),
        "files": {
            "input": report_path(args.input),
            "output": report_path(args.output),
            "prompt": report_path(args.prompt),
            "json_report": report_path(args.report),
            "markdown_report": report_path(args.markdown_report),
        },
        "run_config": run_config,
        "selection": {
            "scope": getattr(args, "scope", "pilot"),
            "cohort": getattr(args, "cohort", "all"),
            "selected_claim_count": len(selected_records),
            "selected_claim_ids": selected_ids,
            "result_records_for_selection": selected_result_count,
            "result_records_outside_selection": len(output_ids - selected_id_set),
        },
        "counts": {
            "gold_label_counts": dict(gold_counts),
            "status_counts": dict(status_counts),
            "prediction_counts": dict(prediction_counts),
            "technical_success_rate": safe_ratio(ok_count, len(selected_records)),
            "unique_result_claim_ids": len(output_ids),
            "duplicate_result_claim_ids": 0,
            "duplicate_validation": "validated_by_loader",
        },
        "primary_binary_metrics": {
            "cohort_definition": "human FACTUAL/NON_FACTUAL claims",
            "gold_claim_count": primary_total,
            "correct_count": correct,
            "accuracy_including_abstentions_and_errors": safe_ratio(
                correct,
                primary_total,
            ),
            "balanced_accuracy": shared_metrics["balanced_accuracy"],
            "answered_count": len(answered_pairs),
            "coverage": safe_ratio(len(answered_pairs), primary_total),
            "selective_accuracy": safe_ratio(correct, len(answered_pairs)),
            "model_unknown_count": abstained_count,
            "abstention_rate": safe_ratio(abstained_count, primary_total),
            "technical_failure_count": technical_failure_count,
            "FACTUAL": factual_metrics,
            "NON_FACTUAL": non_factual_metrics,
            "macro_f1": safe_mean(f1_values),
            "confusion_matrix": shared_metrics["confusion_matrix"],
        },
        "human_unknown_analysis": {
            "claim_count": len(human_unknown_records),
            "prediction_distribution": dict(human_unknown_predictions),
            "note": (
                "Human UNKNOWN is descriptive only and is excluded from primary "
                "binary accuracy/F1."
            ),
        },
        "confusion_matrix": confusion,
        "confidence_analysis": {
            "semantics": "self-reported confidence that the selected label is appropriate",
            "valid_binary_prediction_count": len(confidence_pairs),
            "mean_confidence_correct": safe_mean(correct_confidences),
            "mean_confidence_incorrect_or_abstained": safe_mean(
                incorrect_confidences
            ),
            "high_confidence_threshold": args.high_confidence_threshold,
            "high_confidence_error_count": high_confidence_errors,
            "high_confidence_wrong_decision_count": (
                high_confidence_wrong_decisions
            ),
            "high_confidence_abstention_count": high_confidence_abstentions,
            "exact_score_counts": confidence_distribution[
                "exact_score_counts"
            ],
            "by_prediction": confidence_distribution["by_prediction"],
            "high_confidence_error_claim_ids": confidence_distribution[
                "high_confidence_error_claim_ids"
            ],
            "calibration_warning": (
                "This scalar is not a full class-probability distribution; do not "
                "report multiclass Brier score, log loss, or ECE from it."
            ),
        },
        "response_aggregation": response_rows,
        "response_macro_metrics": response_macro,
        "paired_response_cluster_bootstrap": {
            "status": "not_implemented",
            "todo": (
                "Resample response_id clusters and recompute metrics; the "
                "repository has no reusable bootstrap implementation."
            ),
        },
        "latency_seconds": {
            "count": len(latencies),
            "mean": safe_mean(latencies),
            "median": statistics.median(latencies) if latencies else None,
            "p95_nearest_rank": percentile(latencies, 0.95),
        },
        "leakage_audit": {
            "model_input_fields": ["gold_claim"],
            "prediction_output_contains_gold_labels": False,
            "prediction_output_contains_evidence": False,
            "evaluation_join_key": "claim_id",
        },
        "interpretation_notes": [
            "Model UNKNOWN counts as incorrect in primary overall accuracy.",
            "Coverage is the fraction of binary gold claims receiving a valid "
            "FACTUAL/NON_FACTUAL decision.",
            "Selective accuracy is reported only on those non-abstained decisions.",
            "Request/parse failures remain in primary denominators and are never "
            "converted to UNKNOWN.",
            f"Claims are clustered within {len(response_rows)} selected source "
            "responses; confidence intervals should use response-level cluster "
            "bootstrap.",
        ],
    }


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_escape(value: Any, max_chars: int = 180) -> str:
    text = " ".join(str(value).split())
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text.replace("|", "\\|")


def build_markdown_report(
    summary: dict[str, Any],
    selected_records: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
) -> str:
    selected_ids = {record["claim_id"] for record in selected_records}
    result_by_id = {
        result["claim_id"]: result
        for result in all_results
        if result["claim_id"] in selected_ids
    }
    counts = summary["counts"]
    metrics = summary["primary_binary_metrics"]
    confidence = summary["confidence_analysis"]
    response_macro = summary["response_macro_metrics"]

    lines = [
        "# No-Evidence Verifier Report",
        "",
        f"- Setting: `{summary['setting']}`",
        f"- Scope/cohort: `{summary['selection']['scope']}` / "
        f"`{summary['selection']['cohort']}`",
        f"- Model: `{summary['run_config']['model']}`",
        f"- Completion: **{summary['completion_status']}**",
        f"- Row completion: **{summary['row_completion_status']}**",
        f"- Prediction status: **{summary['prediction_status']}**",
        f"- Selected claims: **{summary['selection']['selected_claim_count']}**",
        f"- Prompt version: `{summary['run_config']['prompt_version']}`",
        f"- Run fingerprint: `{summary['run_config']['run_fingerprint']}`",
        "",
        "## Technical status",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status in ("ok", "request_error", "parse_error", "missing"):
        lines.append(f"| `{status}` | {counts['status_counts'].get(status, 0)} |")

    lines.extend(
        [
            "",
            "## Prediction distribution",
            "",
            "| Prediction | Count |",
            "|---|---:|",
        ]
    )
    for label in LABELS:
        lines.append(f"| `{label}` | {counts['prediction_counts'].get(label, 0)} |")

    lines.extend(
        [
            "",
            "## Primary binary evaluation",
            "",
            "Human `UNKNOWN` claims are excluded from this primary cohort. Model "
            "`UNKNOWN`, request errors, and parse errors remain in the denominator.",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Gold binary claims | {metrics['gold_claim_count']} |",
            f"| Correct | {metrics['correct_count']} |",
            "| Accuracy including abstentions/errors | "
            f"{format_metric(metrics['accuracy_including_abstentions_and_errors'])} |",
            f"| Balanced accuracy | {format_metric(metrics['balanced_accuracy'])} |",
            f"| Answered FACTUAL/NON_FACTUAL | {metrics['answered_count']} |",
            f"| Coverage | {format_metric(metrics['coverage'])} |",
            f"| Selective accuracy | {format_metric(metrics['selective_accuracy'])} |",
            f"| Model UNKNOWN | {metrics['model_unknown_count']} |",
            f"| Abstention rate | {format_metric(metrics['abstention_rate'])} |",
            f"| Technical failures | {metrics['technical_failure_count']} |",
            f"| Macro F1 | {format_metric(metrics['macro_f1'])} |",
            f"| FACTUAL precision | {format_metric(metrics['FACTUAL']['precision'])} |",
            f"| FACTUAL recall | {format_metric(metrics['FACTUAL']['recall'])} |",
            f"| FACTUAL F1 | {format_metric(metrics['FACTUAL']['f1'])} |",
            f"| NON_FACTUAL precision | {format_metric(metrics['NON_FACTUAL']['precision'])} |",
            f"| NON_FACTUAL recall | {format_metric(metrics['NON_FACTUAL']['recall'])} |",
            f"| NON_FACTUAL F1 | {format_metric(metrics['NON_FACTUAL']['f1'])} |",
            "",
            "### Binary confusion matrix",
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

    lines.extend(
        [
            "",
            "### Response-level aggregation",
            "",
            "| Measure | Value |",
            "|---|---:|",
            f"| Responses | {response_macro['response_count']} |",
            f"| Response-macro accuracy | {format_metric(response_macro['accuracy'])} |",
            "| Response-macro balanced accuracy | "
            f"{format_metric(response_macro['balanced_accuracy'])} |",
            f"| Response-macro F1 | {format_metric(response_macro['macro_f1'])} |",
            f"| Response-macro coverage | {format_metric(response_macro['coverage'])} |",
            "",
            "## Confidence diagnostics",
            "",
            "| Measure | Value |",
            "|---|---:|",
            "| Mean confidence, correct | "
            f"{format_metric(confidence['mean_confidence_correct'])} |",
            "| Mean confidence, incorrect/abstained | "
            f"{format_metric(confidence['mean_confidence_incorrect_or_abstained'])} |",
            "| High-confidence error threshold | "
            f"{format_metric(confidence['high_confidence_threshold'])} |",
            "| High-confidence errors | "
            f"{confidence['high_confidence_error_count']} |",
            "| High-confidence wrong decisions | "
            f"{confidence['high_confidence_wrong_decision_count']} |",
            "| High-confidence abstentions | "
            f"{confidence['high_confidence_abstention_count']} |",
            "",
            "### Exact confidence scores",
            "",
            "| Score | Count |",
            "|---:|---:|",
        ]
    )
    for score, count in confidence["exact_score_counts"].items():
        lines.append(f"| {score} | {count} |")
    lines.extend(
        [
            "",
            "> Confidence is a model self-report for the selected label, not a "
            "three-class probability distribution. This report does not compute "
            "Brier score, log loss, or ECE.",
            "",
            "## Claim-level comparison",
            "",
            "Gold fields are joined here by `claim_id` after inference. They were "
            "not written to the prediction JSONL or sent to the model.",
            "",
            "| claim_id | Gold | Prediction | Confidence | Status | Rationale |",
            "|---|---|---|---:|---|---|",
        ]
    )

    for record in selected_records:
        result = result_by_id.get(record["claim_id"])
        if result is None:
            prediction = "—"
            confidence_value = "—"
            status = "missing"
            rationale = "—"
        else:
            prediction = result.get("prediction") or "—"
            confidence_value = format_metric(result.get("confidence"))
            status = result.get("status", "unknown")
            rationale = result.get("rationale") or result.get("error") or "—"
        lines.append(
            f"| `{record['claim_id']}` | `{record['human_label']}` | "
            f"`{prediction}` | {confidence_value} | `{status}` | "
            f"{markdown_escape(rationale)} |"
        )

    lines.extend(
        [
            "",
            "## Leakage audit",
            "",
            "- Each model request contained fixed task instructions plus one "
            "JSON-encoded `gold_claim`.",
            "- Prediction JSONL contains no human label, evidence, stance, or "
            "revision metadata.",
            "- Evaluation joined predictions to gold data only by `claim_id`.",
            "- Every claim used a fresh one-message Ollama request with no prior "
            "claim history.",
            "",
            "## Interpretation cautions",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in summary["interpretation_notes"])
    lines.append("")
    return "\n".join(lines)


def write_reports(
    summary: dict[str, Any],
    selected_records: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        build_markdown_report(summary, selected_records, all_results),
        encoding="utf-8",
    )


def print_dry_run(
    selected_records: list[dict[str, Any]],
    prompt_template: str,
    run_config: dict[str, Any],
) -> None:
    print("=" * 80)
    print("NO-EVIDENCE VERIFIER DRY RUN")
    print("=" * 80)
    print(f"Model: {run_config['model']}")
    print(f"Model digest: {run_config['model_digest']}")
    print(f"Selected claims: {len(selected_records)}")
    print(f"Run fingerprint: {run_config['run_fingerprint']}")
    print("Model input fields: ['gold_claim']")

    for index, record in enumerate(selected_records, start=1):
        prompt = build_prompt(prompt_template, record["gold_claim"])
        print()
        print(f"--- Preview {index}/{len(selected_records)}: {record['claim_id']} ---")
        print("The claim_id header is local audit output, not part of the model message.")
        print("MODEL MESSAGE START")
        print(prompt)
        print("MODEL MESSAGE END")

    print()
    print("No Ollama calls were made. No output or report files were written.")


def make_client(args: argparse.Namespace) -> Client:
    return Client(host=args.ollama_host, timeout=args.timeout)


def preflight_ollama(client: Any, model: str) -> str:
    """Verify the local Ollama service and requested model before writing output."""
    try:
        response = client.list()
    except Exception as error:
        raise ConnectionError(
            "Ollama preflight failed before any result file was changed. "
            "Confirm that the local Ollama service is running."
        ) from error

    models = response_value(response, "models")
    if not isinstance(models, (list, tuple)):
        raise ValueError("Ollama preflight returned no model list.")

    available: dict[str, str | None] = {}
    for item in models:
        name = response_value(item, "model") or response_value(item, "name")
        if isinstance(name, str) and name:
            digest = response_value(item, "digest")
            available[name] = digest if isinstance(digest, str) else None

    if model not in available:
        available_names = ", ".join(sorted(available)) or "none"
        raise ValueError(
            f"Ollama model {model!r} is not installed. Available models: "
            f"{available_names}"
        )
    digest = available[model]
    if not digest:
        raise ValueError(
            f"Ollama did not provide a digest for installed model {model!r}."
        )
    return digest


def summary_exit_code(summary: dict[str, Any]) -> int:
    return 0 if summary.get("completion_status") == "complete" else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_records = load_input_records(args.input)
    cohort_records = filter_records_by_cohort(input_records, args.cohort)
    if not cohort_records:
        raise ValueError(f"Selected cohort is empty: {args.cohort}")
    selected_records = select_records(
        cohort_records,
        limit=args.limit,
        claim_ids=args.claim_id,
        smoke_test=args.smoke_test,
    )
    formal_cohort = "all" if args.scope == "pilot" else "binary"
    partial_selection = (
        args.cohort != formal_cohort
        or len(selected_records) != len(cohort_records)
    )
    if not args.dry_run:
        validate_write_path_safety(args, partial_selection)
    prompt_template = load_prompt_template(args.prompt)
    input_sha256 = sha256_file(args.input)
    prompt_sha256 = sha256_text(prompt_template)
    input_by_id = {record["claim_id"]: record for record in input_records}

    if args.dry_run:
        run_config = build_run_config(
            args,
            input_sha256,
            prompt_sha256,
            model_digest="DRY_RUN_NOT_CHECKED",
        )
        print_dry_run(selected_records, prompt_template, run_config)
        return 0

    if args.report_only:
        if not args.output.exists():
            raise FileNotFoundError(
                f"Cannot use --report-only; output does not exist: {args.output}"
            )
        unchecked_results = load_jsonl_objects(args.output)
        stored_digests = {
            result.get("model_digest")
            for result in unchecked_results
            if isinstance(result.get("model_digest"), str)
            and result.get("model_digest")
        }
        if len(stored_digests) != 1:
            raise ValueError(
                "Cannot recover one model digest from the existing output."
            )
        stored_digest = next(iter(stored_digests))
        if (
            args.expected_model_digest is not None
            and stored_digest != args.expected_model_digest
        ):
            raise ValueError(
                "Stored output model digest does not match the required digest: "
                f"stored={stored_digest}, required={args.expected_model_digest}"
            )
        run_config = build_run_config(
            args,
            input_sha256,
            prompt_sha256,
            model_digest=stored_digest,
        )
        all_results = load_existing_results(
            args.output,
            input_by_id,
            run_config,
        )
        finished_at = utc_now()
        summary = build_summary(
            selected_records,
            all_results,
            run_config,
            args,
            started_at=None,
            finished_at=finished_at,
        )
        write_reports(
            summary,
            selected_records,
            all_results,
            args.report,
            args.markdown_report,
        )
        print(f"JSON report: {args.report}")
        print(f"Markdown report: {args.markdown_report}")
        return summary_exit_code(summary)

    if args.output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --resume, "
            "--overwrite, or a new output path."
        )

    client = make_client(args)
    observed_model_digest = preflight_ollama(client, args.model)
    if (
        args.expected_model_digest is not None
        and observed_model_digest != args.expected_model_digest
    ):
        raise ValueError(
            "Installed model digest does not match the frozen requirement; no "
            "result file was changed. "
            f"observed={observed_model_digest}, "
            f"required={args.expected_model_digest}"
        )
    run_config = build_run_config(
        args,
        input_sha256,
        prompt_sha256,
        model_digest=observed_model_digest,
    )

    existing_results: list[dict[str, Any]] = []
    rewrite_resume_output = False
    if args.output.exists():
        if args.overwrite:
            output_mode = "w"
        elif args.resume:
            existing_results = load_existing_results(
                args.output,
                input_by_id,
                run_config,
            )
            output_mode = "a"
    else:
        output_mode = "w"

    if args.resume and existing_results:
        selected_id_set = {record["claim_id"] for record in selected_records}
        retained_results = [
            result
            for result in existing_results
            if result["claim_id"] not in selected_id_set
            or result.get("status") == "ok"
        ]
        rewrite_resume_output = len(retained_results) != len(existing_results)
        existing_results = retained_results

    existing_ids = {result["claim_id"] for result in existing_results}
    pending_records = [
        record
        for record in selected_records
        if record["claim_id"] not in existing_ids
    ]
    started_at = utc_now()

    if rewrite_resume_output:
        atomic_write_results(args.output, existing_results)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NO-EVIDENCE VERIFIER")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Scope/cohort: {args.scope}/{args.cohort}")
    print(f"Selected: {len(selected_records)}")
    print(f"Existing compatible results: {len(existing_results)}")
    print(f"Pending: {len(pending_records)}")
    print(f"Observed local model digest: {observed_model_digest}")

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
                    record,
                    prompt_template,
                    run_config,
                    client,
                )
                write_result(output_file, result)
                new_results.append(result)
                print(
                    f"  status={result['status']} "
                    f"prediction={result.get('prediction')}",
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
                        "Stopping early after consecutive request failures; "
                        "remaining claims can be retried with --resume.",
                        flush=True,
                    )
                    break

    all_results = (
        load_existing_results(
            args.output,
            input_by_id,
            run_config,
        )
        if args.output.exists()
        else new_results
    )
    finished_at = utc_now()
    summary = build_summary(
        selected_records,
        all_results,
        run_config,
        args,
        started_at=started_at,
        finished_at=finished_at,
    )
    write_reports(
        summary,
        selected_records,
        all_results,
        args.report,
        args.markdown_report,
    )

    print("=" * 80)
    print("NO-EVIDENCE VERIFIER FINISHED")
    print("=" * 80)
    print(f"Output: {args.output}")
    print(f"JSON report: {args.report}")
    print(f"Markdown report: {args.markdown_report}")
    print(f"Status counts: {summary['counts']['status_counts']}")
    print(
        "Primary accuracy: "
        f"{summary['primary_binary_metrics']['accuracy_including_abstentions_and_errors']}"
    )
    print(f"Coverage: {summary['primary_binary_metrics']['coverage']}")
    return summary_exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
