#!/usr/bin/env python3
"""Run staged Experiment B CoVe mechanism evaluation.

Current stages:

* ``prepare-inputs``: data-only response manifest and frozen split validation.
* ``run-questions``: model-backed CoVe verification-question planning.
* ``analyze-questions``: data-only technical summary of question generation.
* ``run-alignment``: model-backed silver question-to-gold-claim alignment.
* ``analyze-alignment``: data-only coverage, atomicity, and redundancy metrics.
* ``run-answers``: one contextually independent model call per B1 question.
* ``analyze-answers``: data-only B3 technical and self-report summary.
* ``run-answer-evaluation``: evidence-grounded B4 question–claim pair judging.
* ``analyze-answer-evaluation``: data-only B4 correctness/funnel reporting.
* ``run-revision``: one standard CoVe final-revision call per response.
* ``analyze-revision``: data-only B5 technical/change-size reporting.
* ``run-revised-claim-extraction``: one B6a decomposition call per revision.
* ``analyze-revised-claim-extraction``: data-only B6a extraction reporting.
* ``run-revised-claim-alignment``: one B6b gold-to-revised mapping per response.
* ``analyze-revised-claim-alignment``: data-only B6b transition candidates.
* ``prepare-revised-claim-evidence``: frozen Hybrid top-5 for all B6a claims.
* ``run-revised-claim-factuality``: B6c evidence-grounded revised-claim labels.
* ``analyze-revised-claim-factuality``: data-only B6c outcome candidates.
* ``prepare-factuality-audit``: deterministic, non-relabeling B6c diagnostics.
* ``run-independent-factuality``: blind cross-family passage adjudication.
* ``recover-independent-factuality-format``: data-only B6d format recovery.
* ``analyze-factuality-consensus``: conservative B6d agreement gate.
* ``prepare-factuality-calibration``: frozen Hybrid top-5 for 121 gold dev claims.
* ``run-factuality-calibration-primary``: Qwen calibration predictions.
* ``run-factuality-calibration-independent``: Llama passage adjudications.
* ``analyze-factuality-calibration``: gold metrics and frozen policy selection.

Gold claims and labels are never inserted into the question-planning prompt.
B2 receives claim text as an evaluation anchor but withholds labels and
evidence; labels are joined only after alignment for stratified metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from ollama import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factcheck_bench_cove import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json_hash,
    cove_paths,
    load_config,
    load_jsonl,
    prepare_cove_inputs,
    sha256_file,
    sha256_text,
    validate_response_manifest,
)
from factcheck_bench_analysis import (  # noqa: E402
    build_response_aggregation,
    compute_binary_metrics,
    paired_response_cluster_bootstrap,
)
from factcheck_bench_pipeline import (  # noqa: E402
    normalize_oracle_evidence,
    retrieval_paths,
)
from factcheck_bench_retrieval_eval import (  # noqa: E402
    evaluation_paths,
    load_evaluation_config,
    rank_frozen_hybrid_queries,
)


load_dotenv(PROJECT_ROOT / ".env")

QUESTION_PLACEHOLDERS = {
    "{original_question_json}",
    "{initial_response_json}",
}
ALIGNMENT_PLACEHOLDERS = {
    "{original_question_json}",
    "{initial_response_json}",
    "{verification_questions_json}",
    "{gold_claims_json}",
}
ALIGNMENT_RELATIONS = {
    "DIRECT",
    "PARTIAL",
    "RELATED_NOT_VERIFYING",
    "NONE",
}
MATCH_RELATIONS = ALIGNMENT_RELATIONS - {"NONE"}
ATOMICITY_LABELS = {
    "ATOMIC",
    "MULTI_TARGET",
    "NOT_FACT_CHECKABLE",
}
ANSWER_PLACEHOLDERS = {"{verification_question_json}"}
ANSWER_STATUS_LABELS = {
    "ANSWERED",
    "UNCERTAIN",
    "INVALID_PREMISE",
}
ANSWER_CLAIM_PLACEHOLDERS = {
    "{verification_question_json}",
    "{verification_answer_json}",
    "{candidate_gold_claim_json}",
    "{oracle_evidence_text}",
}
ALIGNMENT_VALIDITY_LABELS = {
    "VALID_DIRECT",
    "VALID_PARTIAL",
    "INVALID",
}
EVIDENCE_SUFFICIENCY_LABELS = {
    "SUFFICIENT",
    "PARTIAL",
    "INSUFFICIENT",
}
ANSWER_CORRECTNESS_LABELS = {
    "CORRECT",
    "PARTIALLY_CORRECT",
    "INCORRECT",
    "INSUFFICIENT",
    "UNVERIFIABLE",
}
ANSWER_STANCE_LABELS = {
    "SUPPORTS_CLAIM",
    "CHALLENGES_CLAIM",
    "MIXED",
    "NO_POSITION",
}
REVISION_PLACEHOLDERS = {
    "{original_question_json}",
    "{initial_response_json}",
    "{verification_results_json}",
}
REVISED_CLAIM_PLACEHOLDERS = {
    "{original_question_json}",
    "{revised_response_json}",
}
REVISED_ALIGNMENT_PLACEHOLDERS = {
    "{original_question_json}",
    "{initial_claims_json}",
    "{revised_response_json}",
    "{revised_claims_json}",
}
REVISED_ALIGNMENT_RELATIONS = {
    "EQUIVALENT",
    "MODIFIED",
    "PARTIAL",
    "ABSENT",
    "PRESENT_UNEXTRACTED",
}
REVISED_FACTUALITY_PLACEHOLDERS = {
    "{revised_claim_json}",
    "{retrieved_evidence_text}",
}
REVISED_FACTUALITY_LABELS = {
    "FACTUAL",
    "NON_FACTUAL",
    "UNKNOWN",
}
INDEPENDENT_ADJUDICATION_PLACEHOLDERS = {
    "{revised_claim_json}",
    "{retrieved_evidence_text}",
}
PASSAGE_RELATION_LABELS = {
    "SUPPORTS",
    "REFUTES",
    "INSUFFICIENT",
}
INDEPENDENT_FORMAT_WARNINGS = {
    "PASSAGE_ASSESSMENTS_REORDERED",
    "RATIONALE_WORD_LIMIT_EXCEEDED",
    "RATIONALE_AT_SCHEMA_MAX_LENGTH",
}
EXPLICIT_INSUFFICIENCY_PATTERNS = (
    "do not mention",
    "does not mention",
    "do not directly",
    "does not directly",
    "do not confirm",
    "does not confirm",
    "not confirm",
    "no passage directly",
    "insufficient",
)
EXPLICIT_FACTUAL_RATIONALE_PATTERNS = (
    "confirming the claim's factual correctness",
    "confirms the claim's factual correctness",
    "directly supports the claim",
    "supporting the claim's factual assertions",
)
RELATION_PRIORITY = {
    "NONE": 0,
    "RELATED_NOT_VERIFYING": 1,
    "PARTIAL": 2,
    "DIRECT": 3,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def paths_for_args(args: argparse.Namespace) -> Any:
    """Resolve canonical Branch A or an isolated intervention-branch root."""

    return cove_paths(
        PROJECT_ROOT,
        args.scope,
        getattr(args, "branch", "a"),
    )


def response_value(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


def make_question_output_schema(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["question_planning"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verification_questions"],
        "properties": {
            "verification_questions": {
                "type": "array",
                "minItems": int(settings["minimum_questions"]),
                "maxItems": int(settings["maximum_questions"]),
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "minLength": 5,
                    "maxLength": 320,
                },
            }
        },
    }


def parse_question_output(
    raw_output: str,
    config: dict[str, Any],
) -> list[str]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("Model output is empty")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError(f"Model output is not strict JSON: {error}") from error
    if not isinstance(parsed, dict) or set(parsed) != {"verification_questions"}:
        raise ValueError(
            "Model output must be one object containing only verification_questions"
        )
    questions = parsed["verification_questions"]
    if not isinstance(questions, list):
        raise TypeError("verification_questions must be a JSON array")
    normalized: list[str] = []
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, str):
            raise TypeError(f"Question {index} is not a string")
        question = " ".join(question.split())
        if len(question) < 5:
            raise ValueError(f"Question {index} is too short")
        if len(question) > 320:
            raise ValueError(f"Question {index} exceeds 320 characters")
        normalized.append(question)
    minimum = int(config["question_planning"]["minimum_questions"])
    maximum = int(config["question_planning"]["maximum_questions"])
    if not minimum <= len(normalized) <= maximum:
        raise ValueError(
            f"Expected {minimum}-{maximum} questions, received {len(normalized)}"
        )
    keys = [" ".join(question.casefold().split()) for question in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError("Question plan contains exact normalized duplicates")
    return normalized


def load_question_prompt(config: dict[str, Any]) -> tuple[Path, str]:
    path = PROJECT_ROOT / config["question_planning"]["prompt_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"Question prompt is empty: {path}")
    found = {
        placeholder
        for placeholder in QUESTION_PLACEHOLDERS
        if placeholder in template
    }
    if found != QUESTION_PLACEHOLDERS:
        raise ValueError(
            f"Question prompt placeholders are incomplete: {sorted(found)}"
        )
    for placeholder in QUESTION_PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ValueError(
                f"Question prompt must contain {placeholder} exactly once"
            )
    return path, template


def build_question_prompt(template: str, row: dict[str, Any]) -> str:
    return (
        template.replace(
            "{original_question_json}",
            json.dumps(row["original_question"], ensure_ascii=False),
        ).replace(
            "{initial_response_json}",
            json.dumps(row["initial_response"], ensure_ascii=False),
        )
    )


def extract_response(response: Any) -> tuple[str, dict[str, Any]]:
    message = response_value(response, "message")
    content = response_value(message, "content") if message is not None else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama response contains no message content")
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


def preflight_ollama(client: Any, config: dict[str, Any]) -> str:
    settings = config["question_planning"]
    model = settings["model"]
    try:
        response = client.list()
    except Exception as error:
        raise ConnectionError(
            "Ollama preflight failed before any CoVe output was changed"
        ) from error
    models = response_value(response, "models")
    if not isinstance(models, (list, tuple)):
        raise ValueError("Ollama preflight returned no model list")
    available: dict[str, str | None] = {}
    for item in models:
        name = response_value(item, "model") or response_value(item, "name")
        if isinstance(name, str) and name:
            digest = response_value(item, "digest")
            available[name] = digest if isinstance(digest, str) else None
    if model not in available:
        raise ValueError(
            f"Configured model {model!r} is not installed; "
            f"available={sorted(available)}"
        )
    digest = available[model]
    if not digest:
        raise ValueError(f"Ollama provided no digest for {model!r}")
    expected = settings["expected_model_digest"]
    if digest != expected:
        raise ValueError(
            "Installed model digest differs from the frozen Experiment B "
            f"configuration: expected={expected}, actual={digest}"
        )
    return digest


def build_run_config(
    config: dict[str, Any],
    *,
    split: str,
    input_manifest: Path,
    prompt_path: Path,
    model_digest: str,
) -> dict[str, Any]:
    settings = config["question_planning"]
    schema = make_question_output_schema(config)
    payload = {
        "experiment": config["experiment"],
        "stage": "B1_question_planning",
        "split": split,
        "model": settings["model"],
        "model_digest": model_digest,
        "temperature": float(settings["temperature"]),
        "seed": int(settings["seed"]),
        "num_predict": int(settings["num_predict"]),
        "think": bool(settings["think"]),
        "timeout_seconds": float(settings["timeout_seconds"]),
        "max_retries": int(settings["max_retries"]),
        "max_consecutive_request_errors": int(
            settings["max_consecutive_request_errors"]
        ),
        "prompt_version": settings["prompt_version"],
        "prompt_sha256": sha256_file(prompt_path),
        "result_schema_version": settings["result_schema_version"],
        "output_schema_version": settings["output_schema_version"],
        "output_schema_sha256": canonical_json_hash(schema),
        "input_manifest_sha256": sha256_file(input_manifest),
        "model_input_fields": config["leakage_policy"]["model_input_fields"],
    }
    return {**payload, "run_fingerprint": canonical_json_hash(payload)}


def call_ollama(
    client: Any,
    run_config: dict[str, Any],
    output_schema: dict[str, Any],
    prompt: str,
) -> tuple[str, dict[str, Any]]:
    response = client.chat(
        model=run_config["model"],
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        think=False,
        format=output_schema,
        options={
            "temperature": run_config["temperature"],
            "seed": run_config["seed"],
            "num_predict": run_config["num_predict"],
        },
    )
    return extract_response(response)


def create_result_base(
    row: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "result_schema_version": run_config["result_schema_version"],
        "response_id": row["response_id"],
        "source_record_index": row["source_record_index"],
        "split": row["split"],
        "original_question": row["original_question"],
        "initial_response": row["initial_response"],
        "original_question_sha256": row["original_question_sha256"],
        "initial_response_sha256": row["initial_response_sha256"],
        "stage": run_config["stage"],
        "model_input_fields": run_config["model_input_fields"],
        "gold_fields_included": [],
        "model": run_config["model"],
        "model_digest": run_config["model_digest"],
        "temperature": run_config["temperature"],
        "seed": run_config["seed"],
        "num_predict": run_config["num_predict"],
        "think": run_config["think"],
        "timeout_seconds": run_config["timeout_seconds"],
        "max_retries": run_config["max_retries"],
        "max_consecutive_request_errors": run_config[
            "max_consecutive_request_errors"
        ],
        "prompt_version": run_config["prompt_version"],
        "prompt_sha256": run_config["prompt_sha256"],
        "output_schema_version": run_config["output_schema_version"],
        "output_schema_sha256": run_config["output_schema_sha256"],
        "input_manifest_sha256": run_config["input_manifest_sha256"],
        "run_fingerprint": run_config["run_fingerprint"],
    }


def process_response(
    row: dict[str, Any],
    template: str,
    config: dict[str, Any],
    run_config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    result = create_result_base(row, run_config)
    prompt = build_question_prompt(template, row)
    raw_output: str | None = None
    metadata: dict[str, Any] = {}
    request_error: Exception | None = None
    attempts = 0
    started = time.perf_counter()
    for attempt in range(run_config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw_output, metadata = call_ollama(
                client,
                run_config,
                make_question_output_schema(config),
                prompt,
            )
            request_error = None
            break
        except Exception as error:
            request_error = error
            if attempt < run_config["max_retries"]:
                time.sleep(1.0)
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
                "verification_questions": None,
                "error": f"{type(request_error).__name__}: {request_error}",
            }
        )
        return result
    try:
        questions = parse_question_output(raw_output or "", config)
    except Exception as error:
        result.update(
            {
                "status": "parse_error",
                "verification_questions": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return result
    result.update(
        {
            "status": "ok",
            "verification_questions": questions,
            "error": None,
        }
    )
    return result


def validate_existing_results(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    run_config: dict[str, Any],
    config: dict[str, Any],
) -> None:
    input_by_id = {row["response_id"]: row for row in selected}
    seen: set[str] = set()
    for row in rows:
        response_id = row.get("response_id")
        if response_id not in input_by_id:
            raise ValueError(f"Output contains unexpected response_id: {response_id}")
        if response_id in seen:
            raise ValueError(f"Duplicate response_id in output: {response_id}")
        seen.add(response_id)
        source = input_by_id[response_id]
        if row.get("run_fingerprint") != run_config["run_fingerprint"]:
            raise ValueError(
                "Existing CoVe question output is incompatible with the frozen "
                "input/prompt/model configuration"
            )
        if row.get("split") != source["split"]:
            raise ValueError(f"Split mismatch for {response_id}")
        if row.get("original_question_sha256") != source[
            "original_question_sha256"
        ]:
            raise ValueError(f"Question hash mismatch for {response_id}")
        if row.get("initial_response_sha256") != source[
            "initial_response_sha256"
        ]:
            raise ValueError(f"Initial response hash mismatch for {response_id}")
        if row.get("model_input_fields") != [
            "original_question",
            "initial_response",
        ]:
            raise ValueError(f"Unexpected model_input_fields for {response_id}")
        if row.get("gold_fields_included") != []:
            raise ValueError(f"Gold field leakage in output for {response_id}")
        status = row.get("status")
        if status not in {"ok", "request_error", "parse_error"}:
            raise ValueError(f"Invalid status for {response_id}: {status}")
        if status == "ok":
            questions = row.get("verification_questions")
            parse_question_output(
                json.dumps({"verification_questions": questions}),
                config,
            )
            if row.get("error") is not None:
                raise ValueError(f"Successful row has an error for {response_id}")
        else:
            if row.get("verification_questions") is not None:
                raise ValueError(
                    f"Technical failure contains questions for {response_id}"
                )


def question_summary(
    selected: list[dict[str, Any]],
    results: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    by_id = {row["response_id"]: row for row in results}
    status_counts = Counter(
        by_id.get(row["response_id"], {}).get("status", "missing")
        for row in selected
    )
    successful = [
        by_id[row["response_id"]]
        for row in selected
        if by_id.get(row["response_id"], {}).get("status") == "ok"
    ]
    question_counts = [
        len(row["verification_questions"]) for row in successful
    ]
    duplicate_responses = 0
    no_question_mark = 0
    total_questions = 0
    rows_report: list[dict[str, Any]] = []
    for source in selected:
        result = by_id.get(source["response_id"])
        questions = (
            result.get("verification_questions")
            if isinstance(result, dict)
            else None
        )
        if isinstance(questions, list):
            total_questions += len(questions)
            keys = [" ".join(item.casefold().split()) for item in questions]
            duplicate = len(keys) != len(set(keys))
            duplicate_responses += int(duplicate)
            no_question_mark += sum(not item.rstrip().endswith("?") for item in questions)
            count: int | None = len(questions)
        else:
            duplicate = False
            count = None
        rows_report.append(
            {
                "response_id": source["response_id"],
                "status": None if result is None else result.get("status"),
                "question_count": count,
                "normalized_duplicate_questions": duplicate,
                "error": None if result is None else result.get("error"),
            }
        )
    complete = len(successful) == len(selected)
    return {
        "schema_version": "cove_question_planning_summary_v1",
        "experiment": "Experiment B: CoVe mechanism evaluation",
        "stage": "B1_question_planning",
        "split": split,
        "completion_status": "complete" if complete else "incomplete",
        "selected_responses": len(selected),
        "result_rows": len(results),
        "status_counts": dict(status_counts),
        "successful_responses": len(successful),
        "total_verification_questions": total_questions,
        "question_count": {
            "minimum": min(question_counts) if question_counts else None,
            "maximum": max(question_counts) if question_counts else None,
            "mean": (
                round(statistics.mean(question_counts), 4)
                if question_counts
                else None
            ),
            "median": (
                statistics.median(question_counts) if question_counts else None
            ),
        },
        "responses_with_normalized_exact_duplicates": duplicate_responses,
        "questions_without_terminal_question_mark": no_question_mark,
        "gold_claims_exposed_to_planner": False,
        "coverage_evaluation_status": "pending_question_claim_alignment",
        "interpretation_notes": [
            "This report checks technical completion and surface structure only.",
            "Question usefulness, atomicity, redundancy, and claim coverage are not inferred from punctuation or question count.",
            "Gold claims and labels are joined only in the next evaluation stage.",
        ],
        "responses": rows_report,
    }


def build_question_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Experiment B — B1 CoVe Question Planning",
        "",
        f"- Split: `{summary['split']}`",
        f"- Completion: **{summary['completion_status']}**",
        f"- Responses: {summary['successful_responses']}/{summary['selected_responses']}",
        f"- Verification questions: {summary['total_verification_questions']}",
        "- Gold claims exposed to planner: **no**",
        "",
        "## Technical summary",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Minimum questions per successful response | {summary['question_count']['minimum']} |",
        f"| Median questions per successful response | {summary['question_count']['median']} |",
        f"| Mean questions per successful response | {summary['question_count']['mean']} |",
        f"| Maximum questions per successful response | {summary['question_count']['maximum']} |",
        f"| Responses with normalized exact duplicates | {summary['responses_with_normalized_exact_duplicates']} |",
        f"| Questions without terminal question mark | {summary['questions_without_terminal_question_mark']} |",
        "",
        "## Per-response status",
        "",
        "| response_id | status | questions | duplicate | error |",
        "|---|---|---:|---|---|",
    ]
    for row in summary["responses"]:
        error = str(row["error"] or "—").replace("|", "\\|")
        lines.append(
            f"| `{row['response_id']}` | {row['status'] or 'missing'} | "
            f"{row['question_count'] if row['question_count'] is not None else '—'} | "
            f"{row['normalized_duplicate_questions']} | {error} |"
        )
    lines.extend(
        [
            "",
            "## Boundary of this stage",
            "",
            "This report does not yet measure whether questions cover gold claims. "
            "Question-to-claim alignment is the next Experiment B stage and will "
            "use gold claims only as hidden evaluation anchors.",
            "",
        ]
    )
    return "\n".join(lines)


def write_question_reports(
    paths: Any,
    split: str,
    summary: dict[str, Any],
) -> None:
    atomic_write_json(paths.question_summary_json(split), summary)
    atomic_write_text(
        paths.question_summary_markdown(split),
        build_question_markdown(summary),
    )


def run_questions(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    if not paths.response_manifest.exists():
        raise FileNotFoundError(
            "CoVe response manifest is missing. Run prepare-cove-inputs first."
        )
    manifest = load_jsonl(paths.response_manifest)
    validate_response_manifest(manifest, config)
    selected = [row for row in manifest if row["split"] == args.split]
    if not selected:
        raise ValueError(f"No CoVe input rows for split {args.split}")
    prompt_path, template = load_question_prompt(config)

    if args.dry_run:
        run_config = build_run_config(
            config,
            split=args.split,
            input_manifest=paths.response_manifest,
            prompt_path=prompt_path,
            model_digest="DRY_RUN_NOT_CHECKED",
        )
        print("EXPERIMENT B B1 QUESTION-PLANNING DRY RUN")
        print(f"Split: {args.split}; responses: {len(selected)}")
        print(f"Run fingerprint: {run_config['run_fingerprint']}")
        print("Model input fields: original_question, initial_response")
        for index, row in enumerate(selected[:3], start=1):
            print(f"\n--- Preview {index}/3: {row['response_id']} ---")
            print(build_question_prompt(template, row))
        print("\nNo Ollama calls were made and no files were written.")
        return 0

    client = Client(
        host=args.ollama_host,
        timeout=float(config["question_planning"]["timeout_seconds"]),
    )
    model_digest = preflight_ollama(client, config)
    run_config = build_run_config(
        config,
        split=args.split,
        input_manifest=paths.response_manifest,
        prompt_path=prompt_path,
        model_digest=model_digest,
    )
    output_path = paths.question_results(args.split)
    existing = load_jsonl(output_path) if output_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Output already exists; rerun with --resume: {output_path}"
        )
    validate_existing_results(existing, selected, run_config, config)
    result_by_id = {row["response_id"]: row for row in existing}
    pending = [
        row
        for row in selected
        if result_by_id.get(row["response_id"], {}).get("status") != "ok"
    ]
    print(
        f"Experiment B B1: split={args.split}, total={len(selected)}, "
        f"retained_ok={len(selected) - len(pending)}, pending={len(pending)}",
        flush=True,
    )
    consecutive_request_errors = 0
    max_consecutive = int(
        config["question_planning"]["max_consecutive_request_errors"]
    )
    manifest_order = {row["response_id"]: index for index, row in enumerate(selected)}
    for row in pending:
        overall_index = manifest_order[row["response_id"]] + 1
        print(
            f"[{overall_index}/{len(selected)}] {row['response_id']} "
            "generating verification questions ...",
            flush=True,
        )
        result = process_response(row, template, config, run_config, client)
        result_by_id[row["response_id"]] = result
        ordered = [
            result_by_id[item["response_id"]]
            for item in selected
            if item["response_id"] in result_by_id
        ]
        atomic_write_jsonl(output_path, ordered)
        if result["status"] == "ok":
            consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(selected)}] success, "
                f"{len(result['verification_questions'])} questions, "
                f"{result['latency_seconds']:.1f}s",
                flush=True,
            )
        else:
            if result["status"] == "request_error":
                consecutive_request_errors += 1
            else:
                consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(selected)}] {result['status']}: "
                f"{result['error']}",
                flush=True,
            )
            if consecutive_request_errors >= max_consecutive:
                print(
                    "Stopping after the configured consecutive request-error "
                    "limit. The same command will resume safely.",
                    file=sys.stderr,
                    flush=True,
                )
                break

    all_results = [
        result_by_id[item["response_id"]]
        for item in selected
        if item["response_id"] in result_by_id
    ]
    validate_existing_results(all_results, selected, run_config, config)
    summary = question_summary(selected, all_results, args.split)
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    write_question_reports(paths, args.split, summary)
    print(f"Results: {output_path}", flush=True)
    print(
        f"Report: {paths.question_summary_markdown(args.split)}",
        flush=True,
    )
    return 0 if summary["completion_status"] == "complete" else 2


def analyze_questions(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    manifest = load_jsonl(paths.response_manifest)
    validate_response_manifest(manifest, config)
    selected = [row for row in manifest if row["split"] == args.split]
    output_path = paths.question_results(args.split)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("Question results do not have one valid model digest")
    prompt_path, _ = load_question_prompt(config)
    run_config = build_run_config(
        config,
        split=args.split,
        input_manifest=paths.response_manifest,
        prompt_path=prompt_path,
        model_digest=next(iter(model_digests)),
    )
    validate_existing_results(results, selected, run_config, config)
    summary = question_summary(selected, results, args.split)
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    write_question_reports(paths, args.split, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["completion_status"] == "complete" else 2


def make_alignment_output_schema(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["question_claim_alignment"]
    relation_labels = list(settings["relation_labels"])
    atomicity_labels = list(settings["atomicity_labels"])
    if set(relation_labels) != ALIGNMENT_RELATIONS:
        raise ValueError("Alignment relation labels differ from the frozen taxonomy")
    if set(atomicity_labels) != ATOMICITY_LABELS:
        raise ValueError("Alignment atomicity labels differ from the frozen taxonomy")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["question_alignments"],
        "properties": {
            "question_alignments": {
                "type": "array",
                "minItems": 1,
                "maxItems": int(config["question_planning"]["maximum_questions"]),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "question_id",
                        "overall_relation",
                        "atomicity",
                        "matches",
                        "rationale",
                    ],
                    "properties": {
                        "question_id": {
                            "type": "string",
                            "minLength": 5,
                            "maxLength": 80,
                        },
                        "overall_relation": {
                            "type": "string",
                            "enum": relation_labels,
                        },
                        "atomicity": {
                            "type": "string",
                            "enum": atomicity_labels,
                        },
                        "matches": {
                            "type": "array",
                            "maxItems": 64,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["claim_id", "relation"],
                                "properties": {
                                    "claim_id": {
                                        "type": "string",
                                        "minLength": 5,
                                        "maxLength": 160,
                                    },
                                    "relation": {
                                        "type": "string",
                                        "enum": sorted(MATCH_RELATIONS),
                                    },
                                },
                            },
                        },
                        "rationale": {
                            "type": "string",
                            "minLength": 3,
                            "maxLength": 360,
                        },
                    },
                },
            }
        },
    }


def load_alignment_prompt(config: dict[str, Any]) -> tuple[Path, str]:
    path = PROJECT_ROOT / config["question_claim_alignment"]["prompt_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"Alignment prompt is empty: {path}")
    found = {
        placeholder
        for placeholder in ALIGNMENT_PLACEHOLDERS
        if placeholder in template
    }
    if found != ALIGNMENT_PLACEHOLDERS:
        raise ValueError(
            f"Alignment prompt placeholders are incomplete: {sorted(found)}"
        )
    for placeholder in ALIGNMENT_PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ValueError(
                f"Alignment prompt must contain {placeholder} exactly once"
            )
    return path, template


def load_gold_claims_by_response(
    config: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    path = (
        PROJECT_ROOT
        / "data"
        / "factcheck_bench"
        / "processed"
        / "fcb_gold_claims_full.jsonl"
    )
    rows = load_jsonl(path)
    by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        claim_id = row.get("claim_id")
        response_id = row.get("response_id")
        claim = row.get("gold_claim")
        label = row.get("human_label")
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError("Gold claim row has an invalid claim_id")
        if claim_id in seen:
            raise ValueError(f"Duplicate gold claim_id: {claim_id}")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError(f"Gold claim {claim_id} has an invalid response_id")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"Gold claim {claim_id} has empty text")
        if label not in {"FACTUAL", "NON_FACTUAL", "UNKNOWN"}:
            raise ValueError(f"Gold claim {claim_id} has invalid label: {label}")
        seen.add(claim_id)
        by_response[response_id].append(row)
    expected = config["split_policy"]["expected"]
    if len(rows) != int(expected["all_claims"]):
        raise ValueError("Gold claim file does not contain the canonical claim count")
    return path, rows, by_response


def question_items(question_result: dict[str, Any]) -> list[dict[str, str]]:
    response_id = question_result["response_id"]
    questions = question_result.get("verification_questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"B1 result has no questions for {response_id}")
    return [
        {
            "question_id": f"{response_id}_q{index:02d}",
            "question": question,
        }
        for index, question in enumerate(questions, start=1)
    ]


def claim_items(gold_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "claim_id": row["claim_id"],
            "claim": row["gold_claim"],
        }
        for row in gold_rows
    ]


def build_alignment_prompt(
    template: str,
    question_result: dict[str, Any],
    gold_rows: list[dict[str, Any]],
) -> str:
    replacements = {
        "{original_question_json}": json.dumps(
            question_result["original_question"],
            ensure_ascii=False,
        ),
        "{initial_response_json}": json.dumps(
            question_result["initial_response"],
            ensure_ascii=False,
        ),
        "{verification_questions_json}": json.dumps(
            question_items(question_result),
            ensure_ascii=False,
            indent=2,
        ),
        "{gold_claims_json}": json.dumps(
            claim_items(gold_rows),
            ensure_ascii=False,
            indent=2,
        ),
    }
    prompt = template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def parse_alignment_output(
    raw_output: str,
    question_result: dict[str, Any],
    gold_rows: list[dict[str, Any]],
    *,
    derived_field_repairs: list[dict[str, str]] | None = None,
    invalid_reference_repairs: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("Alignment model output is empty")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Alignment model output is not strict JSON: {error}"
        ) from error
    if not isinstance(parsed, dict) or set(parsed) != {"question_alignments"}:
        raise ValueError(
            "Alignment output must contain only question_alignments"
        )
    alignments = parsed["question_alignments"]
    if not isinstance(alignments, list):
        raise TypeError("question_alignments must be an array")
    expected_questions = question_items(question_result)
    expected_ids = [item["question_id"] for item in expected_questions]
    actual_ids: list[str] = []
    valid_claim_ids = {row["claim_id"] for row in gold_rows}
    normalized: list[dict[str, Any]] = []
    exact_fields = {
        "question_id",
        "overall_relation",
        "atomicity",
        "matches",
        "rationale",
    }
    for index, alignment in enumerate(alignments, start=1):
        if not isinstance(alignment, dict) or set(alignment) != exact_fields:
            raise ValueError(f"Alignment item {index} has unexpected fields")
        question_id = alignment["question_id"]
        if not isinstance(question_id, str):
            raise TypeError(f"Alignment item {index} question_id is not a string")
        actual_ids.append(question_id)
        overall = alignment["overall_relation"]
        atomicity = alignment["atomicity"]
        if overall not in ALIGNMENT_RELATIONS:
            raise ValueError(f"Invalid relation for {question_id}: {overall}")
        if atomicity not in ATOMICITY_LABELS:
            raise ValueError(f"Invalid atomicity for {question_id}: {atomicity}")
        matches = alignment["matches"]
        if not isinstance(matches, list):
            raise TypeError(f"Matches for {question_id} must be an array")
        normalized_matches: list[dict[str, str]] = []
        matched_ids: set[str] = set()
        for match in matches:
            if not isinstance(match, dict) or set(match) != {
                "claim_id",
                "relation",
            }:
                raise ValueError(f"Invalid match structure for {question_id}")
            claim_id = match["claim_id"]
            relation = match["relation"]
            if claim_id not in valid_claim_ids:
                if invalid_reference_repairs is None:
                    raise ValueError(
                        f"Unknown or cross-response claim_id for {question_id}: "
                        f"{claim_id}"
                    )
                invalid_reference_repairs.append(
                    {
                        "question_id": question_id,
                        "field": "matches",
                        "original_value": claim_id,
                        "canonical_value": "<dropped>",
                        "reason": (
                            "the model-provided claim_id does not exist in the "
                            "canonical claim set for this response; recovery "
                            "drops the invalid reference without guessing a "
                            "replacement claim"
                        ),
                    }
                )
                continue
            if claim_id in matched_ids:
                raise ValueError(
                    f"Duplicate claim match for {question_id}: {claim_id}"
                )
            if relation not in MATCH_RELATIONS:
                raise ValueError(
                    f"Invalid match relation for {question_id}: {relation}"
                )
            matched_ids.add(claim_id)
            normalized_matches.append(
                {"claim_id": claim_id, "relation": relation}
            )
        expected_overall = (
            max(
                (match["relation"] for match in normalized_matches),
                key=RELATION_PRIORITY.__getitem__,
            )
            if normalized_matches
            else "NONE"
        )
        if overall != expected_overall:
            if derived_field_repairs is None:
                raise ValueError(
                    f"overall_relation for {question_id} must be "
                    f"{expected_overall}, not {overall}"
                )
            derived_field_repairs.append(
                {
                    "question_id": question_id,
                    "field": "overall_relation",
                    "original_value": overall,
                    "canonical_value": expected_overall,
                    "reason": (
                        "overall_relation is deterministically derived from "
                        "the validated match relations"
                    ),
                }
            )
            overall = expected_overall
        rationale = alignment["rationale"]
        if not isinstance(rationale, str):
            raise TypeError(f"Rationale for {question_id} is not a string")
        rationale = " ".join(rationale.split())
        if not 3 <= len(rationale) <= 360:
            raise ValueError(
                f"Rationale for {question_id} must contain 3-360 characters"
            )
        normalized.append(
            {
                "question_id": question_id,
                "overall_relation": overall,
                "atomicity": atomicity,
                "matches": normalized_matches,
                "rationale": rationale,
            }
        )
    if actual_ids != expected_ids:
        raise ValueError(
            "Alignment output must contain each B1 question exactly once and "
            "in its original order"
        )
    return normalized


def preflight_alignment_ollama(client: Any, config: dict[str, Any]) -> str:
    settings = config["question_claim_alignment"]
    model = settings["model"]
    try:
        response = client.list()
    except Exception as error:
        raise ConnectionError(
            "Ollama preflight failed before any B2 output was changed"
        ) from error
    models = response_value(response, "models")
    if not isinstance(models, (list, tuple)):
        raise ValueError("Ollama preflight returned no model list")
    available: dict[str, str | None] = {}
    for item in models:
        name = response_value(item, "model") or response_value(item, "name")
        if isinstance(name, str) and name:
            digest = response_value(item, "digest")
            available[name] = digest if isinstance(digest, str) else None
    if model not in available:
        raise ValueError(
            f"Configured alignment model {model!r} is not installed; "
            f"available={sorted(available)}"
        )
    digest = available[model]
    if not digest:
        raise ValueError(f"Ollama provided no digest for {model!r}")
    expected = settings["expected_model_digest"]
    if digest != expected:
        raise ValueError(
            "Installed model digest differs from the frozen B2 configuration: "
            f"expected={expected}, actual={digest}"
        )
    return digest


def build_alignment_run_config(
    config: dict[str, Any],
    *,
    split: str,
    prompt_path: Path,
    question_results_path: Path,
    gold_claims_path: Path,
    response_manifest_path: Path,
    model_digest: str,
) -> dict[str, Any]:
    settings = config["question_claim_alignment"]
    schema = make_alignment_output_schema(config)
    payload = {
        "experiment": config["experiment"],
        "stage": "B2_question_claim_alignment",
        "split": split,
        "annotation_status": "llm_assisted_silver_not_human_gold",
        "model": settings["model"],
        "model_digest": model_digest,
        "temperature": float(settings["temperature"]),
        "seed": int(settings["seed"]),
        "num_predict": int(settings["num_predict"]),
        "think": bool(settings["think"]),
        "timeout_seconds": float(settings["timeout_seconds"]),
        "max_retries": int(settings["max_retries"]),
        "max_consecutive_request_errors": int(
            settings["max_consecutive_request_errors"]
        ),
        "prompt_version": settings["prompt_version"],
        "prompt_sha256": sha256_file(prompt_path),
        "result_schema_version": settings["result_schema_version"],
        "output_schema_version": settings["output_schema_version"],
        "output_schema_sha256": canonical_json_hash(schema),
        "question_results_sha256": sha256_file(question_results_path),
        "gold_claims_sha256": sha256_file(gold_claims_path),
        "response_manifest_sha256": sha256_file(response_manifest_path),
        "model_input_fields": config["leakage_policy"][
            "alignment_model_input_fields"
        ],
        "withheld_fields": config["leakage_policy"][
            "alignment_withheld_fields"
        ],
    }
    return {**payload, "run_fingerprint": canonical_json_hash(payload)}


def create_alignment_result_base(
    question_result: dict[str, Any],
    gold_rows: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    alignment_input = {
        "original_question": question_result["original_question"],
        "initial_response": question_result["initial_response"],
        "verification_questions": question_items(question_result),
        "gold_claims": claim_items(gold_rows),
    }
    return {
        "result_schema_version": run_config["result_schema_version"],
        "response_id": question_result["response_id"],
        "source_record_index": question_result["source_record_index"],
        "split": question_result["split"],
        "stage": run_config["stage"],
        "annotation_status": run_config["annotation_status"],
        "b1_run_fingerprint": question_result["run_fingerprint"],
        "b1_result_sha256": canonical_json_hash(question_result),
        "alignment_input_sha256": canonical_json_hash(alignment_input),
        "model_input_fields": run_config["model_input_fields"],
        "gold_fields_included": ["gold_claim_id", "gold_claim_text"],
        "withheld_fields": run_config["withheld_fields"],
        "model": run_config["model"],
        "model_digest": run_config["model_digest"],
        "temperature": run_config["temperature"],
        "seed": run_config["seed"],
        "num_predict": run_config["num_predict"],
        "think": run_config["think"],
        "timeout_seconds": run_config["timeout_seconds"],
        "max_retries": run_config["max_retries"],
        "max_consecutive_request_errors": run_config[
            "max_consecutive_request_errors"
        ],
        "prompt_version": run_config["prompt_version"],
        "prompt_sha256": run_config["prompt_sha256"],
        "output_schema_version": run_config["output_schema_version"],
        "output_schema_sha256": run_config["output_schema_sha256"],
        "question_results_sha256": run_config["question_results_sha256"],
        "gold_claims_sha256": run_config["gold_claims_sha256"],
        "response_manifest_sha256": run_config["response_manifest_sha256"],
        "run_fingerprint": run_config["run_fingerprint"],
    }


def process_alignment_response(
    question_result: dict[str, Any],
    gold_rows: list[dict[str, Any]],
    template: str,
    config: dict[str, Any],
    run_config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    result = create_alignment_result_base(
        question_result,
        gold_rows,
        run_config,
    )
    prompt = build_alignment_prompt(template, question_result, gold_rows)
    raw_output: str | None = None
    metadata: dict[str, Any] = {}
    request_error: Exception | None = None
    attempts = 0
    started = time.perf_counter()
    for attempt in range(run_config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw_output, metadata = call_ollama(
                client,
                run_config,
                make_alignment_output_schema(config),
                prompt,
            )
            request_error = None
            break
        except Exception as error:
            request_error = error
            if attempt < run_config["max_retries"]:
                time.sleep(1.0)
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
                "question_alignments": None,
                "error": f"{type(request_error).__name__}: {request_error}",
            }
        )
        return result
    try:
        format_repairs: list[dict[str, str]] = []
        alignments = parse_alignment_output(
            raw_output or "",
            question_result,
            gold_rows,
            derived_field_repairs=format_repairs,
        )
    except Exception as error:
        result.update(
            {
                "status": "parse_error",
                "question_alignments": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return result
    result.update(
        {
            "status": "ok",
            "question_alignments": alignments,
            "format_repairs": format_repairs,
            "error": None,
        }
    )
    return result


def validate_existing_alignment_results(
    rows: list[dict[str, Any]],
    question_results: list[dict[str, Any]],
    gold_by_response: dict[str, list[dict[str, Any]]],
    run_config: dict[str, Any],
) -> None:
    question_by_id = {row["response_id"]: row for row in question_results}
    seen: set[str] = set()
    for row in rows:
        response_id = row.get("response_id")
        if response_id not in question_by_id:
            raise ValueError(
                f"B2 output contains unexpected response_id: {response_id}"
            )
        if response_id in seen:
            raise ValueError(f"Duplicate response_id in B2 output: {response_id}")
        seen.add(response_id)
        question_result = question_by_id[response_id]
        gold_rows = gold_by_response[response_id]
        expected_base = create_alignment_result_base(
            question_result,
            gold_rows,
            run_config,
        )
        for key, expected in expected_base.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"Existing B2 output is incompatible for {response_id}: {key}"
                )
        status = row.get("status")
        if status not in {"ok", "request_error", "parse_error"}:
            raise ValueError(f"Invalid B2 status for {response_id}: {status}")
        if status == "ok":
            parse_alignment_output(
                json.dumps(
                    {"question_alignments": row.get("question_alignments")},
                    ensure_ascii=False,
                ),
                question_result,
                gold_rows,
            )
            if row.get("error") is not None:
                raise ValueError(f"Successful B2 row has an error for {response_id}")
            repairs = row.get("format_repairs", [])
            if not isinstance(repairs, list):
                raise ValueError(
                    f"B2 format_repairs must be an array for {response_id}"
                )
        elif row.get("question_alignments") is not None:
            raise ValueError(
                f"Technical B2 failure contains alignments for {response_id}"
            )


def flatten_alignment_pairs(
    question_results: list[dict[str, Any]],
    alignment_results: list[dict[str, Any]],
    gold_by_response: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    question_by_response = {
        row["response_id"]: {
            item["question_id"]: item["question"]
            for item in question_items(row)
        }
        for row in question_results
    }
    alignment_by_response = {
        row["response_id"]: row
        for row in alignment_results
        if row.get("status") == "ok"
    }
    rows: list[dict[str, Any]] = []
    for response_id, result in alignment_by_response.items():
        gold_lookup = {
            row["claim_id"]: row for row in gold_by_response[response_id]
        }
        for alignment in result["question_alignments"]:
            base = {
                "schema_version": "cove_question_claim_pair_v1",
                "response_id": response_id,
                "split": result["split"],
                "question_id": alignment["question_id"],
                "question": question_by_response[response_id][
                    alignment["question_id"]
                ],
                "overall_relation": alignment["overall_relation"],
                "atomicity": alignment["atomicity"],
                "rationale": alignment["rationale"],
                "annotation_status": result["annotation_status"],
                "alignment_run_fingerprint": result["run_fingerprint"],
                "audit_status": "not_human_audited",
            }
            if not alignment["matches"]:
                rows.append(
                    {
                        **base,
                        "claim_id": None,
                        "gold_claim": None,
                        "human_label": None,
                        "relation": "NONE",
                    }
                )
                continue
            for match in alignment["matches"]:
                gold = gold_lookup[match["claim_id"]]
                rows.append(
                    {
                        **base,
                        "claim_id": gold["claim_id"],
                        "gold_claim": gold["gold_claim"],
                        "human_label": gold["human_label"],
                        "relation": match["relation"],
                    }
                )
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def alignment_summary(
    selected_questions: list[dict[str, Any]],
    results: list[dict[str, Any]],
    gold_by_response: dict[str, list[dict[str, Any]]],
    split: str,
) -> dict[str, Any]:
    result_by_id = {row["response_id"]: row for row in results}
    status_counts = Counter(
        result_by_id.get(row["response_id"], {}).get("status", "missing")
        for row in selected_questions
    )
    successful = [
        result_by_id[row["response_id"]]
        for row in selected_questions
        if result_by_id.get(row["response_id"], {}).get("status") == "ok"
    ]
    all_gold = [
        claim
        for source in selected_questions
        for claim in gold_by_response[source["response_id"]]
    ]
    all_claim_ids = {row["claim_id"] for row in all_gold}
    claim_labels = {row["claim_id"]: row["human_label"] for row in all_gold}
    useful_relations = {"DIRECT", "PARTIAL"}
    direct_claim_ids: set[str] = set()
    covered_claim_ids: set[str] = set()
    useful_question_counts_by_claim: Counter[str] = Counter()
    overall_counts: Counter[str] = Counter()
    atomicity_counts: Counter[str] = Counter()
    question_total = 0
    useful_question_total = 0
    questions_with_multiple_useful_claims = 0
    per_response: list[dict[str, Any]] = []

    for source in selected_questions:
        response_id = source["response_id"]
        result = result_by_id.get(response_id)
        gold_rows = gold_by_response[response_id]
        response_claim_ids = {row["claim_id"] for row in gold_rows}
        response_covered: set[str] = set()
        response_direct: set[str] = set()
        response_question_count = 0
        response_useful_questions = 0
        if isinstance(result, dict) and result.get("status") == "ok":
            for alignment in result["question_alignments"]:
                response_question_count += 1
                question_total += 1
                overall_counts[alignment["overall_relation"]] += 1
                atomicity_counts[alignment["atomicity"]] += 1
                useful_matches = [
                    match
                    for match in alignment["matches"]
                    if match["relation"] in useful_relations
                ]
                if useful_matches:
                    response_useful_questions += 1
                    useful_question_total += 1
                useful_ids = {match["claim_id"] for match in useful_matches}
                if len(useful_ids) > 1:
                    questions_with_multiple_useful_claims += 1
                for match in useful_matches:
                    claim_id = match["claim_id"]
                    covered_claim_ids.add(claim_id)
                    response_covered.add(claim_id)
                    useful_question_counts_by_claim[claim_id] += 1
                    if match["relation"] == "DIRECT":
                        direct_claim_ids.add(claim_id)
                        response_direct.add(claim_id)
        per_response.append(
            {
                "response_id": response_id,
                "status": None if result is None else result.get("status"),
                "questions": response_question_count,
                "useful_questions": response_useful_questions,
                "claims": len(response_claim_ids),
                "covered_claims": len(response_covered),
                "directly_covered_claims": len(response_direct),
                "claim_coverage": _rate(
                    len(response_covered),
                    len(response_claim_ids),
                ),
            }
        )

    if not covered_claim_ids <= all_claim_ids:
        raise ValueError("Alignment summary found a cross-split claim match")
    label_metrics: dict[str, Any] = {}
    for label in ("FACTUAL", "NON_FACTUAL", "UNKNOWN"):
        label_ids = {
            claim_id
            for claim_id, item_label in claim_labels.items()
            if item_label == label
        }
        label_covered = label_ids & covered_claim_ids
        label_direct = label_ids & direct_claim_ids
        label_metrics[label] = {
            "claims": len(label_ids),
            "covered_direct_or_partial": len(label_covered),
            "coverage": _rate(len(label_covered), len(label_ids)),
            "covered_direct": len(label_direct),
            "direct_coverage": _rate(len(label_direct), len(label_ids)),
        }

    revisited_claims = {
        claim_id
        for claim_id, count in useful_question_counts_by_claim.items()
        if count > 1
    }
    total_useful_mappings = sum(useful_question_counts_by_claim.values())
    atomic_denominator = (
        atomicity_counts["ATOMIC"] + atomicity_counts["MULTI_TARGET"]
    )
    complete = len(successful) == len(selected_questions)
    uncovered_non_factual = [
        {
            "claim_id": row["claim_id"],
            "response_id": row["response_id"],
            "gold_claim": row["gold_claim"],
        }
        for row in all_gold
        if row["human_label"] == "NON_FACTUAL"
        and row["claim_id"] not in covered_claim_ids
    ]
    return {
        "schema_version": "cove_question_claim_alignment_summary_v1",
        "experiment": "Experiment B: CoVe mechanism evaluation",
        "stage": "B2_question_claim_alignment",
        "split": split,
        "annotation_status": "llm_assisted_silver_not_human_gold",
        "completion_status": "complete" if complete else "incomplete",
        "selected_responses": len(selected_questions),
        "successful_responses": len(successful),
        "result_rows": len(results),
        "status_counts": dict(status_counts),
        "questions": {
            "total": question_total,
            "relation_counts": {
                label: overall_counts[label]
                for label in ("DIRECT", "PARTIAL", "RELATED_NOT_VERIFYING", "NONE")
            },
            "useful_direct_or_partial": useful_question_total,
            "useful_question_precision": _rate(
                useful_question_total,
                question_total,
            ),
            "atomicity_counts": {
                label: atomicity_counts[label]
                for label in (
                    "ATOMIC",
                    "MULTI_TARGET",
                    "NOT_FACT_CHECKABLE",
                )
            },
            "atomic_question_rate_among_fact_checkable": _rate(
                atomicity_counts["ATOMIC"],
                atomic_denominator,
            ),
            "questions_with_multiple_useful_claims": (
                questions_with_multiple_useful_claims
            ),
        },
        "claims": {
            "total": len(all_claim_ids),
            "covered_direct_or_partial": len(covered_claim_ids),
            "coverage": _rate(len(covered_claim_ids), len(all_claim_ids)),
            "covered_direct": len(direct_claim_ids),
            "direct_coverage": _rate(len(direct_claim_ids), len(all_claim_ids)),
            "by_human_label": label_metrics,
        },
        "redundancy_proxies": {
            "total_useful_question_claim_mappings": total_useful_mappings,
            "claims_with_multiple_useful_questions": len(revisited_claims),
            "claim_revisit_rate_among_covered": _rate(
                len(revisited_claims),
                len(covered_claim_ids),
            ),
            "extra_useful_mappings_beyond_first": (
                total_useful_mappings - len(covered_claim_ids)
            ),
        },
        "uncovered_non_factual_claims": uncovered_non_factual,
        "responses": per_response,
        "interpretation_notes": [
            "DIRECT and PARTIAL count as useful claim coverage; RELATED_NOT_VERIFYING does not.",
            "Human labels were withheld from the alignment model and joined only for data-only stratified metrics.",
            "The alignment is LLM-assisted silver annotation, not human-gold annotation.",
            "The alignment model uses the same frozen Qwen weights as B1, so evaluator dependence must be acknowledged.",
            "Coverage measures whether a question targets a claim, not whether the question premise or later verification answer is correct.",
        ],
    }


def build_alignment_markdown(summary: dict[str, Any]) -> str:
    questions = summary["questions"]
    claims = summary["claims"]
    label_metrics = claims["by_human_label"]
    lines = [
        "# Experiment B — B2 Question-to-Claim Alignment",
        "",
        f"- Split: `{summary['split']}`",
        f"- Completion: **{summary['completion_status']}**",
        f"- Responses: {summary['successful_responses']}/{summary['selected_responses']}",
        f"- Annotation status: `{summary['annotation_status']}`",
        "",
        "## Main metrics",
        "",
        "| Measure | Count | Rate |",
        "|---|---:|---:|",
        f"| Useful questions (DIRECT/PARTIAL) | {questions['useful_direct_or_partial']}/{questions['total']} | {questions['useful_question_precision']} |",
        f"| Gold claims covered (DIRECT/PARTIAL) | {claims['covered_direct_or_partial']}/{claims['total']} | {claims['coverage']} |",
        f"| Gold claims covered directly | {claims['covered_direct']}/{claims['total']} | {claims['direct_coverage']} |",
        f"| Atomic questions among fact-checkable questions | {questions['atomicity_counts']['ATOMIC']} | {questions['atomic_question_rate_among_fact_checkable']} |",
        f"| Questions covering multiple useful claims | {questions['questions_with_multiple_useful_claims']} | — |",
        "",
        "## Coverage by hidden human label",
        "",
        "| Label | Claims | DIRECT/PARTIAL covered | Coverage | Direct coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("FACTUAL", "NON_FACTUAL", "UNKNOWN"):
        metric = label_metrics[label]
        lines.append(
            f"| {label} | {metric['claims']} | "
            f"{metric['covered_direct_or_partial']} | {metric['coverage']} | "
            f"{metric['direct_coverage']} |"
        )
    lines.extend(
        [
            "",
            "Human labels were not shown to the alignment model. They were joined "
            "only after semantic alignment for these stratified metrics.",
            "",
            "## Question relations and atomicity",
            "",
            "| Relation | Questions |",
            "|---|---:|",
        ]
    )
    for label in ("DIRECT", "PARTIAL", "RELATED_NOT_VERIFYING", "NONE"):
        lines.append(
            f"| {label} | {questions['relation_counts'][label]} |"
        )
    lines.extend(["", "| Atomicity | Questions |", "|---|---:|"])
    for label in ("ATOMIC", "MULTI_TARGET", "NOT_FACT_CHECKABLE"):
        lines.append(
            f"| {label} | {questions['atomicity_counts'][label]} |"
        )
    redundancy = summary["redundancy_proxies"]
    lines.extend(
        [
            "",
            "## Redundancy proxies",
            "",
            f"- Useful question–claim mappings: {redundancy['total_useful_question_claim_mappings']}",
            f"- Covered claims revisited by more than one useful question: {redundancy['claims_with_multiple_useful_questions']}",
            f"- Claim-revisit rate among covered claims: {redundancy['claim_revisit_rate_among_covered']}",
            f"- Extra useful mappings beyond the first per claim: {redundancy['extra_useful_mappings_beyond_first']}",
            "",
            "## Per-response coverage",
            "",
            "| response_id | status | questions | useful questions | claims | covered | coverage |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["responses"]:
        lines.append(
            f"| `{row['response_id']}` | {row['status'] or 'missing'} | "
            f"{row['questions']} | {row['useful_questions']} | "
            f"{row['claims']} | {row['covered_claims']} | "
            f"{row['claim_coverage']} |"
        )
    lines.extend(
        [
            "",
            "## Uncovered NON_FACTUAL claims",
            "",
            "| response_id | claim_id | claim |",
            "|---|---|---|",
        ]
    )
    if summary["uncovered_non_factual_claims"]:
        for row in summary["uncovered_non_factual_claims"]:
            claim = row["gold_claim"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{row['response_id']}` | `{row['claim_id']}` | {claim} |"
            )
    else:
        lines.append("| — | — | None |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is LLM-assisted silver semantic alignment, not human-gold "
            "annotation. It measures whether B1 questions target canonical "
            "claims. It does not measure whether a question has a false premise "
            "or whether its future independent answer is correct; those are "
            "evaluated in B3/B4.",
            "",
        ]
    )
    return "\n".join(lines)


def write_alignment_reports(
    paths: Any,
    split: str,
    summary: dict[str, Any],
    pairs: list[dict[str, Any]],
) -> None:
    atomic_write_jsonl(paths.alignment_pairs(split), pairs)
    atomic_write_json(paths.alignment_summary_json(split), summary)
    atomic_write_text(
        paths.alignment_summary_markdown(split),
        build_alignment_markdown(summary),
    )


def load_validated_b1_results(
    paths: Any,
    config: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    manifest = load_jsonl(paths.response_manifest)
    validate_response_manifest(manifest, config)
    selected_manifest = [row for row in manifest if row["split"] == split]
    question_path = paths.question_results(split)
    if not question_path.exists():
        raise FileNotFoundError(
            f"B1 output is missing for {split}: {question_path}"
        )
    results = load_jsonl(question_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B1 results do not have one valid model digest")
    prompt_path, _ = load_question_prompt(config)
    b1_run_config = build_run_config(
        config,
        split=split,
        input_manifest=paths.response_manifest,
        prompt_path=prompt_path,
        model_digest=next(iter(model_digests)),
    )
    validate_existing_results(
        results,
        selected_manifest,
        b1_run_config,
        config,
    )
    by_id = {row["response_id"]: row for row in results}
    selected_results = [
        by_id[row["response_id"]]
        for row in selected_manifest
        if by_id.get(row["response_id"], {}).get("status") == "ok"
    ]
    if len(selected_results) != len(selected_manifest):
        raise ValueError(
            f"B1 {split} is incomplete; finish question planning before B2"
        )
    return selected_results


def run_alignment(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    selected_questions = load_validated_b1_results(
        paths,
        config,
        args.split,
    )
    gold_path, _, gold_by_response = load_gold_claims_by_response(config)
    expected_claims = int(
        config["split_policy"]["expected"][f"{args.split}_all_claims"]
    )
    selected_claim_count = sum(
        len(gold_by_response[row["response_id"]])
        for row in selected_questions
    )
    if selected_claim_count != expected_claims:
        raise ValueError(
            f"B2 {args.split} expected {expected_claims} gold claims, "
            f"found {selected_claim_count}"
        )
    prompt_path, template = load_alignment_prompt(config)
    question_path = paths.question_results(args.split)

    if args.dry_run:
        run_config = build_alignment_run_config(
            config,
            split=args.split,
            prompt_path=prompt_path,
            question_results_path=question_path,
            gold_claims_path=gold_path,
            response_manifest_path=paths.response_manifest,
            model_digest="DRY_RUN_NOT_CHECKED",
        )
        print("EXPERIMENT B B2 QUESTION–CLAIM ALIGNMENT DRY RUN")
        print(
            f"Split: {args.split}; responses: {len(selected_questions)}; "
            f"gold claims: {selected_claim_count}"
        )
        print(f"Run fingerprint: {run_config['run_fingerprint']}")
        print(
            "Human labels/evidence/stance/URLs/qrels/revisions are withheld "
            "from the alignment model."
        )
        preview = selected_questions[0]
        print(f"\n--- Preview: {preview['response_id']} ---")
        print(
            build_alignment_prompt(
                template,
                preview,
                gold_by_response[preview["response_id"]],
            )
        )
        print("\nNo Ollama calls were made and no files were written.")
        return 0

    client = Client(
        host=args.ollama_host,
        timeout=float(
            config["question_claim_alignment"]["timeout_seconds"]
        ),
    )
    model_digest = preflight_alignment_ollama(client, config)
    run_config = build_alignment_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        question_results_path=question_path,
        gold_claims_path=gold_path,
        response_manifest_path=paths.response_manifest,
        model_digest=model_digest,
    )
    output_path = paths.alignment_results(args.split)
    existing = load_jsonl(output_path) if output_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Output already exists; rerun with --resume: {output_path}"
        )
    validate_existing_alignment_results(
        existing,
        selected_questions,
        gold_by_response,
        run_config,
    )
    result_by_id = {row["response_id"]: row for row in existing}
    pending = [
        row
        for row in selected_questions
        if result_by_id.get(row["response_id"], {}).get("status") != "ok"
    ]
    print(
        f"Experiment B B2: split={args.split}, total={len(selected_questions)}, "
        f"retained_ok={len(selected_questions) - len(pending)}, "
        f"pending={len(pending)}",
        flush=True,
    )
    consecutive_request_errors = 0
    max_consecutive = int(
        config["question_claim_alignment"]["max_consecutive_request_errors"]
    )
    order = {
        row["response_id"]: index
        for index, row in enumerate(selected_questions)
    }
    for question_result in pending:
        response_id = question_result["response_id"]
        overall_index = order[response_id] + 1
        gold_rows = gold_by_response[response_id]
        print(
            f"[{overall_index}/{len(selected_questions)}] {response_id} "
            f"aligning {len(question_result['verification_questions'])} "
            f"questions to {len(gold_rows)} claims ...",
            flush=True,
        )
        result = process_alignment_response(
            question_result,
            gold_rows,
            template,
            config,
            run_config,
            client,
        )
        result_by_id[response_id] = result
        ordered = [
            result_by_id[row["response_id"]]
            for row in selected_questions
            if row["response_id"] in result_by_id
        ]
        atomic_write_jsonl(output_path, ordered)
        if result["status"] == "ok":
            consecutive_request_errors = 0
            useful = sum(
                alignment["overall_relation"] in {"DIRECT", "PARTIAL"}
                for alignment in result["question_alignments"]
            )
            print(
                f"[{overall_index}/{len(selected_questions)}] success, "
                f"{useful}/{len(result['question_alignments'])} useful "
                f"questions, {result['latency_seconds']:.1f}s",
                flush=True,
            )
        else:
            if result["status"] == "request_error":
                consecutive_request_errors += 1
            else:
                consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(selected_questions)}] "
                f"{result['status']}: {result['error']}",
                flush=True,
            )
            if consecutive_request_errors >= max_consecutive:
                print(
                    "Stopping after the configured consecutive request-error "
                    "limit. The same command will resume safely.",
                    file=sys.stderr,
                    flush=True,
                )
                break

    all_results = [
        result_by_id[row["response_id"]]
        for row in selected_questions
        if row["response_id"] in result_by_id
    ]
    validate_existing_alignment_results(
        all_results,
        selected_questions,
        gold_by_response,
        run_config,
    )
    summary = alignment_summary(
        selected_questions,
        all_results,
        gold_by_response,
        args.split,
    )
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    pairs = flatten_alignment_pairs(
        selected_questions,
        all_results,
        gold_by_response,
    )
    summary["pair_file"] = str(
        paths.alignment_pairs(args.split).relative_to(PROJECT_ROOT)
    )
    write_alignment_reports(paths, args.split, summary, pairs)
    print(f"Results: {output_path}", flush=True)
    print(
        f"Report: {paths.alignment_summary_markdown(args.split)}",
        flush=True,
    )
    return 0 if summary["completion_status"] == "complete" else 2


def analyze_alignment(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    selected_questions = load_validated_b1_results(
        paths,
        config,
        args.split,
    )
    gold_path, _, gold_by_response = load_gold_claims_by_response(config)
    output_path = paths.alignment_results(args.split)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B2 results do not have one valid model digest")
    prompt_path, _ = load_alignment_prompt(config)
    run_config = build_alignment_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        question_results_path=paths.question_results(args.split),
        gold_claims_path=gold_path,
        response_manifest_path=paths.response_manifest,
        model_digest=next(iter(model_digests)),
    )
    validate_existing_alignment_results(
        results,
        selected_questions,
        gold_by_response,
        run_config,
    )
    summary = alignment_summary(
        selected_questions,
        results,
        gold_by_response,
        args.split,
    )
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    pairs = flatten_alignment_pairs(
        selected_questions,
        results,
        gold_by_response,
    )
    summary["pair_file"] = str(
        paths.alignment_pairs(args.split).relative_to(PROJECT_ROOT)
    )
    write_alignment_reports(paths, args.split, summary, pairs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["completion_status"] == "complete" else 2


def recover_alignment_format(args: argparse.Namespace) -> int:
    """Recover deterministic B2 structure without inferring new alignments."""

    paths = paths_for_args(args)
    config = load_config(paths)
    selected_questions = load_validated_b1_results(
        paths,
        config,
        args.split,
    )
    gold_path, _, gold_by_response = load_gold_claims_by_response(config)
    output_path = paths.alignment_results(args.split)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B2 results do not have one valid model digest")
    prompt_path, _ = load_alignment_prompt(config)
    run_config = build_alignment_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        question_results_path=paths.question_results(args.split),
        gold_claims_path=gold_path,
        response_manifest_path=paths.response_manifest,
        model_digest=next(iter(model_digests)),
    )
    validate_existing_alignment_results(
        results,
        selected_questions,
        gold_by_response,
        run_config,
    )
    question_by_id = {
        row["response_id"]: row for row in selected_questions
    }
    recovered_rows = 0
    repairs_total = 0
    derived_fields_repaired = 0
    invalid_references_dropped = 0
    for row in results:
        if row.get("status") != "parse_error":
            continue
        response_id = row["response_id"]
        raw_output = row.get("raw_model_output")
        if not isinstance(raw_output, str) or not raw_output.strip():
            continue
        derived_repairs: list[dict[str, str]] = []
        invalid_repairs: list[dict[str, str]] = []
        try:
            alignments = parse_alignment_output(
                raw_output,
                question_by_id[response_id],
                gold_by_response[response_id],
                derived_field_repairs=derived_repairs,
                invalid_reference_repairs=invalid_repairs,
            )
        except Exception:
            continue
        repairs = [*invalid_repairs, *derived_repairs]
        if not repairs:
            continue
        original_error = row.get("error")
        row.update(
            {
                "status": "ok",
                "question_alignments": alignments,
                "format_repairs": repairs,
                "error": None,
                "format_recovery": {
                    "method": (
                        "deterministic_invalid_reference_and_derived_field_"
                        "recovery_v1"
                    ),
                    "recovered_at": utc_now(),
                    "original_status": "parse_error",
                    "original_error": original_error,
                    "model_recalled": False,
                    "semantic_fields_changed": bool(invalid_repairs),
                    "invalid_references_dropped": len(invalid_repairs),
                    "replacement_claims_inferred": 0,
                    "audit_required": bool(invalid_repairs),
                },
            }
        )
        recovered_rows += 1
        repairs_total += len(repairs)
        derived_fields_repaired += len(derived_repairs)
        invalid_references_dropped += len(invalid_repairs)
        print(
            f"[recovered] {response_id}: "
            f"{len(invalid_repairs)} invalid reference(s) dropped, "
            f"{len(derived_repairs)} derived overall_relation field(s)",
            flush=True,
        )
    if recovered_rows:
        atomic_write_jsonl(output_path, results)
    validate_existing_alignment_results(
        results,
        selected_questions,
        gold_by_response,
        run_config,
    )
    summary = alignment_summary(
        selected_questions,
        results,
        gold_by_response,
        args.split,
    )
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    summary["format_recovery"] = {
        "recovered_rows_this_run": recovered_rows,
        "repairs_total_this_run": repairs_total,
        "derived_fields_repaired_this_run": derived_fields_repaired,
        "invalid_references_dropped_this_run": invalid_references_dropped,
        "replacement_claims_inferred": 0,
        "remaining_parse_errors": sum(
            row.get("status") == "parse_error" for row in results
        ),
        "model_recalled": False,
    }
    pairs = flatten_alignment_pairs(
        selected_questions,
        results,
        gold_by_response,
    )
    summary["pair_file"] = str(
        paths.alignment_pairs(args.split).relative_to(PROJECT_ROOT)
    )
    write_alignment_reports(paths, args.split, summary, pairs)
    print(
        f"Recovered rows: {recovered_rows}; repairs: {repairs_total}; "
        f"invalid references dropped: {invalid_references_dropped}; "
        f"remaining parse errors: "
        f"{summary['format_recovery']['remaining_parse_errors']}",
        flush=True,
    )
    return 0 if summary["completion_status"] == "complete" else 2


def make_verification_answer_output_schema(
    config: dict[str, Any],
) -> dict[str, Any]:
    labels = list(
        config["independent_verification_answering"]["answer_status_labels"]
    )
    if set(labels) != ANSWER_STATUS_LABELS:
        raise ValueError(
            "Verification-answer status labels differ from the frozen taxonomy"
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer_status", "verification_answer"],
        "properties": {
            "answer_status": {
                "type": "string",
                "enum": labels,
            },
            "verification_answer": {
                "type": "string",
                "minLength": 2,
                "maxLength": 1200,
            },
        },
    }


def load_verification_answer_prompt(
    config: dict[str, Any],
) -> tuple[Path, str]:
    path = (
        PROJECT_ROOT
        / config["independent_verification_answering"]["prompt_path"]
    )
    if not path.exists():
        raise FileNotFoundError(path)
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"Verification-answer prompt is empty: {path}")
    found = {
        placeholder
        for placeholder in ANSWER_PLACEHOLDERS
        if placeholder in template
    }
    if found != ANSWER_PLACEHOLDERS:
        raise ValueError(
            f"Verification-answer placeholders are incomplete: {sorted(found)}"
        )
    for placeholder in ANSWER_PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ValueError(
                f"Verification-answer prompt must contain {placeholder} once"
            )
    return path, template


def build_verification_answer_prompt(
    template: str,
    verification_question: str,
) -> str:
    return template.replace(
        "{verification_question_json}",
        json.dumps(verification_question, ensure_ascii=False),
    )


def parse_verification_answer_output(
    raw_output: str,
) -> dict[str, str]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("Verification-answer model output is empty")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Verification-answer output is not strict JSON: {error}"
        ) from error
    if not isinstance(parsed, dict) or set(parsed) != {
        "answer_status",
        "verification_answer",
    }:
        raise ValueError(
            "Verification-answer output must contain only answer_status and "
            "verification_answer"
        )
    answer_status = parsed["answer_status"]
    if answer_status not in ANSWER_STATUS_LABELS:
        raise ValueError(f"Invalid answer_status: {answer_status}")
    answer = parsed["verification_answer"]
    if not isinstance(answer, str):
        raise TypeError("verification_answer must be a string")
    answer = " ".join(answer.split())
    if not 2 <= len(answer) <= 1200:
        raise ValueError("verification_answer must contain 2-1200 characters")
    return {
        "answer_status": answer_status,
        "verification_answer": answer,
    }


def preflight_verification_answer_ollama(
    client: Any,
    config: dict[str, Any],
) -> str:
    settings = config["independent_verification_answering"]
    model = settings["model"]
    try:
        response = client.list()
    except Exception as error:
        raise ConnectionError(
            "Ollama preflight failed before any B3 output was changed"
        ) from error
    models = response_value(response, "models")
    if not isinstance(models, (list, tuple)):
        raise ValueError("Ollama preflight returned no model list")
    available: dict[str, str | None] = {}
    for item in models:
        name = response_value(item, "model") or response_value(item, "name")
        if isinstance(name, str) and name:
            digest = response_value(item, "digest")
            available[name] = digest if isinstance(digest, str) else None
    if model not in available:
        raise ValueError(
            f"Configured B3 model {model!r} is not installed; "
            f"available={sorted(available)}"
        )
    digest = available[model]
    if not digest:
        raise ValueError(f"Ollama provided no digest for {model!r}")
    expected = settings["expected_model_digest"]
    if digest != expected:
        raise ValueError(
            "Installed model digest differs from the frozen B3 configuration: "
            f"expected={expected}, actual={digest}"
        )
    return digest


def load_validated_b2_gate(
    paths: Any,
    config: dict[str, Any],
    split: str,
    selected_questions: list[dict[str, Any]],
) -> str:
    gold_path, _, gold_by_response = load_gold_claims_by_response(config)
    output_path = paths.alignment_results(split)
    if not output_path.exists():
        raise FileNotFoundError(
            f"B2 output is missing for {split}: {output_path}"
        )
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B2 results do not have one valid model digest")
    prompt_path, _ = load_alignment_prompt(config)
    run_config = build_alignment_run_config(
        config,
        split=split,
        prompt_path=prompt_path,
        question_results_path=paths.question_results(split),
        gold_claims_path=gold_path,
        response_manifest_path=paths.response_manifest,
        model_digest=next(iter(model_digests)),
    )
    validate_existing_alignment_results(
        results,
        selected_questions,
        gold_by_response,
        run_config,
    )
    expected_ids = {row["response_id"] for row in selected_questions}
    ok_ids = {
        row["response_id"]
        for row in results
        if row.get("status") == "ok"
    }
    if ok_ids != expected_ids:
        raise ValueError(
            f"B2 {split} is incomplete; finish/recover alignment before B3"
        )
    return run_config["run_fingerprint"]


def verification_answer_units(
    question_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for question_result in question_results:
        for question_index, item in enumerate(
            question_items(question_result),
            start=1,
        ):
            question_id = item["question_id"]
            if question_id in seen:
                raise ValueError(f"Duplicate verification question ID: {question_id}")
            seen.add(question_id)
            units.append(
                {
                    "response_id": question_result["response_id"],
                    "source_record_index": question_result[
                        "source_record_index"
                    ],
                    "split": question_result["split"],
                    "question_id": question_id,
                    "question_index": question_index,
                    "verification_question": item["question"],
                    "b1_run_fingerprint": question_result["run_fingerprint"],
                    "b1_result_sha256": canonical_json_hash(question_result),
                }
            )
    return units


def build_verification_answer_run_config(
    config: dict[str, Any],
    *,
    split: str,
    prompt_path: Path,
    question_results_path: Path,
    response_manifest_path: Path,
    b2_gate_run_fingerprint: str,
    model_digest: str,
) -> dict[str, Any]:
    settings = config["independent_verification_answering"]
    schema = make_verification_answer_output_schema(config)
    payload = {
        "experiment": config["experiment"],
        "stage": "B3_independent_verification_answers",
        "split": split,
        "independence_unit": "one_model_call_per_verification_question",
        "model": settings["model"],
        "model_digest": model_digest,
        "temperature": float(settings["temperature"]),
        "seed": int(settings["seed"]),
        "num_predict": int(settings["num_predict"]),
        "think": bool(settings["think"]),
        "timeout_seconds": float(settings["timeout_seconds"]),
        "max_retries": int(settings["max_retries"]),
        "max_consecutive_request_errors": int(
            settings["max_consecutive_request_errors"]
        ),
        "prompt_version": settings["prompt_version"],
        "prompt_sha256": sha256_file(prompt_path),
        "result_schema_version": settings["result_schema_version"],
        "output_schema_version": settings["output_schema_version"],
        "output_schema_sha256": canonical_json_hash(schema),
        "question_results_sha256": sha256_file(question_results_path),
        "response_manifest_sha256": sha256_file(response_manifest_path),
        "b2_gate_run_fingerprint": b2_gate_run_fingerprint,
        "model_input_fields": config["leakage_policy"][
            "verification_answer_model_input_fields"
        ],
        "withheld_fields": config["leakage_policy"][
            "verification_answer_withheld_fields"
        ],
    }
    return {**payload, "run_fingerprint": canonical_json_hash(payload)}


def create_verification_answer_result_base(
    unit: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "result_schema_version": run_config["result_schema_version"],
        "response_id": unit["response_id"],
        "source_record_index": unit["source_record_index"],
        "split": unit["split"],
        "question_id": unit["question_id"],
        "question_index": unit["question_index"],
        "verification_question": unit["verification_question"],
        "verification_question_sha256": sha256_text(
            unit["verification_question"]
        ),
        "stage": run_config["stage"],
        "independence_unit": run_config["independence_unit"],
        "contextually_independent_call": True,
        "model_input_fields": run_config["model_input_fields"],
        "withheld_fields": run_config["withheld_fields"],
        "b1_run_fingerprint": unit["b1_run_fingerprint"],
        "b1_result_sha256": unit["b1_result_sha256"],
        "b2_gate_run_fingerprint": run_config["b2_gate_run_fingerprint"],
        "model": run_config["model"],
        "model_digest": run_config["model_digest"],
        "temperature": run_config["temperature"],
        "seed": run_config["seed"],
        "num_predict": run_config["num_predict"],
        "think": run_config["think"],
        "timeout_seconds": run_config["timeout_seconds"],
        "max_retries": run_config["max_retries"],
        "max_consecutive_request_errors": run_config[
            "max_consecutive_request_errors"
        ],
        "prompt_version": run_config["prompt_version"],
        "prompt_sha256": run_config["prompt_sha256"],
        "output_schema_version": run_config["output_schema_version"],
        "output_schema_sha256": run_config["output_schema_sha256"],
        "question_results_sha256": run_config["question_results_sha256"],
        "response_manifest_sha256": run_config["response_manifest_sha256"],
        "run_fingerprint": run_config["run_fingerprint"],
    }


def process_verification_answer_unit(
    unit: dict[str, Any],
    template: str,
    config: dict[str, Any],
    run_config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    result = create_verification_answer_result_base(unit, run_config)
    prompt = build_verification_answer_prompt(
        template,
        unit["verification_question"],
    )
    raw_output: str | None = None
    metadata: dict[str, Any] = {}
    request_error: Exception | None = None
    attempts = 0
    started = time.perf_counter()
    for attempt in range(run_config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw_output, metadata = call_ollama(
                client,
                run_config,
                make_verification_answer_output_schema(config),
                prompt,
            )
            request_error = None
            break
        except Exception as error:
            request_error = error
            if attempt < run_config["max_retries"]:
                time.sleep(1.0)
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
                "answer_status": None,
                "verification_answer": None,
                "error": f"{type(request_error).__name__}: {request_error}",
            }
        )
        return result
    try:
        parsed = parse_verification_answer_output(raw_output or "")
    except Exception as error:
        result.update(
            {
                "status": "parse_error",
                "answer_status": None,
                "verification_answer": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return result
    result.update(
        {
            "status": "ok",
            **parsed,
            "error": None,
        }
    )
    return result


def validate_existing_verification_answers(
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    unit_by_id = {unit["question_id"]: unit for unit in units}
    seen: set[str] = set()
    for row in rows:
        question_id = row.get("question_id")
        if question_id not in unit_by_id:
            raise ValueError(
                f"B3 output contains unexpected question_id: {question_id}"
            )
        if question_id in seen:
            raise ValueError(f"Duplicate question_id in B3 output: {question_id}")
        seen.add(question_id)
        expected_base = create_verification_answer_result_base(
            unit_by_id[question_id],
            run_config,
        )
        for key, expected in expected_base.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"Existing B3 output is incompatible for {question_id}: {key}"
                )
        status = row.get("status")
        if status not in {"ok", "request_error", "parse_error"}:
            raise ValueError(f"Invalid B3 status for {question_id}: {status}")
        if status == "ok":
            parse_verification_answer_output(
                json.dumps(
                    {
                        "answer_status": row.get("answer_status"),
                        "verification_answer": row.get(
                            "verification_answer"
                        ),
                    },
                    ensure_ascii=False,
                )
            )
            if row.get("error") is not None:
                raise ValueError(
                    f"Successful B3 row has an error for {question_id}"
                )
        else:
            if row.get("answer_status") is not None:
                raise ValueError(
                    f"Technical B3 failure has answer_status for {question_id}"
                )
            if row.get("verification_answer") is not None:
                raise ValueError(
                    f"Technical B3 failure has an answer for {question_id}"
                )


def verification_answer_summary(
    units: list[dict[str, Any]],
    results: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    result_by_id = {row["question_id"]: row for row in results}
    status_counts = Counter(
        result_by_id.get(unit["question_id"], {}).get("status", "missing")
        for unit in units
    )
    successful = [
        result_by_id[unit["question_id"]]
        for unit in units
        if result_by_id.get(unit["question_id"], {}).get("status") == "ok"
    ]
    answer_status_counts = Counter(
        row["answer_status"] for row in successful
    )
    latencies = [
        float(row["latency_seconds"])
        for row in successful
        if isinstance(row.get("latency_seconds"), (int, float))
    ]
    per_response: list[dict[str, Any]] = []
    response_ids = list(dict.fromkeys(unit["response_id"] for unit in units))
    for response_id in response_ids:
        response_units = [
            unit for unit in units if unit["response_id"] == response_id
        ]
        response_results = [
            result_by_id.get(unit["question_id"]) for unit in response_units
        ]
        response_success = [
            row
            for row in response_results
            if isinstance(row, dict) and row.get("status") == "ok"
        ]
        per_response.append(
            {
                "response_id": response_id,
                "questions": len(response_units),
                "successful_answers": len(response_success),
                "answer_status_counts": dict(
                    Counter(row["answer_status"] for row in response_success)
                ),
            }
        )
    complete = len(successful) == len(units)
    return {
        "schema_version": "cove_independent_verification_answer_summary_v1",
        "experiment": "Experiment B: CoVe mechanism evaluation",
        "stage": "B3_independent_verification_answers",
        "split": split,
        "completion_status": "complete" if complete else "incomplete",
        "selected_responses": len(response_ids),
        "selected_questions": len(units),
        "result_rows": len(results),
        "successful_answers": len(successful),
        "status_counts": dict(status_counts),
        "self_reported_answer_status_counts": {
            label: answer_status_counts[label]
            for label in ("ANSWERED", "UNCERTAIN", "INVALID_PREMISE")
        },
        "latency_seconds": {
            "total": round(sum(latencies), 4) if latencies else None,
            "mean": (
                round(statistics.mean(latencies), 4) if latencies else None
            ),
            "median": (
                round(statistics.median(latencies), 4) if latencies else None
            ),
        },
        "contextual_independence": {
            "unit": "one_model_call_per_verification_question",
            "model_input_fields": ["verification_question"],
            "initial_response_exposed": False,
            "original_question_exposed": False,
            "other_verification_questions_exposed": False,
            "gold_fields_exposed": False,
        },
        "correctness_evaluation_status": "pending_B4",
        "responses": per_response,
        "interpretation_notes": [
            "ANSWERED/UNCERTAIN/INVALID_PREMISE are model self-reports, not correctness labels.",
            "Every question is answered in a separate stateless request containing only that question.",
            "B4 must evaluate answer correctness against hidden claims/evidence and detect agreement or conflict with the initial response.",
        ],
    }


def build_verification_answer_markdown(summary: dict[str, Any]) -> str:
    latency = summary["latency_seconds"]
    counts = summary["self_reported_answer_status_counts"]
    lines = [
        "# Experiment B — B3 Independent Verification Answers",
        "",
        f"- Split: `{summary['split']}`",
        f"- Completion: **{summary['completion_status']}**",
        f"- Questions answered: {summary['successful_answers']}/{summary['selected_questions']}",
        "- Initial response exposed: **no**",
        "- Original user question exposed: **no**",
        "- Other verification questions exposed: **no**",
        "- Gold fields exposed: **no**",
        "",
        "## Technical summary",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| ANSWERED (self-reported) | {counts['ANSWERED']} |",
        f"| UNCERTAIN (self-reported) | {counts['UNCERTAIN']} |",
        f"| INVALID_PREMISE (self-reported) | {counts['INVALID_PREMISE']} |",
        f"| Mean latency, seconds | {latency['mean']} |",
        f"| Median latency, seconds | {latency['median']} |",
        "",
        "## Per-response completion",
        "",
        "| response_id | questions | successful | answer status counts |",
        "|---|---:|---:|---|",
    ]
    for row in summary["responses"]:
        counts_text = json.dumps(
            row["answer_status_counts"],
            sort_keys=True,
        ).replace("|", "\\|")
        lines.append(
            f"| `{row['response_id']}` | {row['questions']} | "
            f"{row['successful_answers']} | `{counts_text}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary of this stage",
            "",
            "The answer-status field is a model self-report and is not evidence "
            "of correctness. B4 will compare each answer with hidden claims and "
            "evidence, then determine whether the answer correctly confirms, "
            "correctly challenges, is insufficient, or is irrelevant.",
            "",
        ]
    )
    return "\n".join(lines)


def write_verification_answer_reports(
    paths: Any,
    split: str,
    summary: dict[str, Any],
) -> None:
    atomic_write_json(
        paths.verification_answer_summary_json(split),
        summary,
    )
    atomic_write_text(
        paths.verification_answer_summary_markdown(split),
        build_verification_answer_markdown(summary),
    )


def run_verification_answers(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    selected_questions = load_validated_b1_results(
        paths,
        config,
        args.split,
    )
    b2_gate_run_fingerprint = load_validated_b2_gate(
        paths,
        config,
        args.split,
        selected_questions,
    )
    units = verification_answer_units(selected_questions)
    expected_questions = sum(
        len(row["verification_questions"]) for row in selected_questions
    )
    if len(units) != expected_questions:
        raise ValueError("B3 question unit construction lost or duplicated rows")
    prompt_path, template = load_verification_answer_prompt(config)

    if args.dry_run:
        run_config = build_verification_answer_run_config(
            config,
            split=args.split,
            prompt_path=prompt_path,
            question_results_path=paths.question_results(args.split),
            response_manifest_path=paths.response_manifest,
            b2_gate_run_fingerprint=b2_gate_run_fingerprint,
            model_digest="DRY_RUN_NOT_CHECKED",
        )
        print("EXPERIMENT B B3 INDEPENDENT-ANSWER DRY RUN")
        print(
            f"Split: {args.split}; responses: {len(selected_questions)}; "
            f"independent question calls: {len(units)}"
        )
        print(f"Run fingerprint: {run_config['run_fingerprint']}")
        print(
            "Model input fields: verification_question only. No initial "
            "response, original question, other questions, gold, or B2 mapping."
        )
        for unit in units[:3]:
            print(f"\n--- Preview: {unit['question_id']} ---")
            print(
                build_verification_answer_prompt(
                    template,
                    unit["verification_question"],
                )
            )
        print("\nNo Ollama calls were made and no files were written.")
        return 0

    client = Client(
        host=args.ollama_host,
        timeout=float(
            config["independent_verification_answering"]["timeout_seconds"]
        ),
    )
    model_digest = preflight_verification_answer_ollama(client, config)
    run_config = build_verification_answer_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        question_results_path=paths.question_results(args.split),
        response_manifest_path=paths.response_manifest,
        b2_gate_run_fingerprint=b2_gate_run_fingerprint,
        model_digest=model_digest,
    )
    output_path = paths.verification_answer_results(args.split)
    existing = load_jsonl(output_path) if output_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Output already exists; rerun with --resume: {output_path}"
        )
    validate_existing_verification_answers(existing, units, run_config)
    result_by_id = {row["question_id"]: row for row in existing}
    pending = [
        unit
        for unit in units
        if result_by_id.get(unit["question_id"], {}).get("status") != "ok"
    ]
    print(
        f"Experiment B B3: split={args.split}, total={len(units)}, "
        f"retained_ok={len(units) - len(pending)}, pending={len(pending)}",
        flush=True,
    )
    consecutive_request_errors = 0
    max_consecutive = int(
        config["independent_verification_answering"][
            "max_consecutive_request_errors"
        ]
    )
    order = {unit["question_id"]: index for index, unit in enumerate(units)}
    for unit in pending:
        overall_index = order[unit["question_id"]] + 1
        print(
            f"[{overall_index}/{len(units)}] {unit['question_id']} "
            "answering independently ...",
            flush=True,
        )
        result = process_verification_answer_unit(
            unit,
            template,
            config,
            run_config,
            client,
        )
        result_by_id[unit["question_id"]] = result
        ordered = [
            result_by_id[item["question_id"]]
            for item in units
            if item["question_id"] in result_by_id
        ]
        atomic_write_jsonl(output_path, ordered)
        if result["status"] == "ok":
            consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(units)}] success, "
                f"{result['answer_status']}, "
                f"{result['latency_seconds']:.1f}s",
                flush=True,
            )
        else:
            if result["status"] == "request_error":
                consecutive_request_errors += 1
            else:
                consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(units)}] {result['status']}: "
                f"{result['error']}",
                flush=True,
            )
            if consecutive_request_errors >= max_consecutive:
                print(
                    "Stopping after the configured consecutive request-error "
                    "limit. The same command will resume safely.",
                    file=sys.stderr,
                    flush=True,
                )
                break
    all_results = [
        result_by_id[unit["question_id"]]
        for unit in units
        if unit["question_id"] in result_by_id
    ]
    validate_existing_verification_answers(all_results, units, run_config)
    summary = verification_answer_summary(units, all_results, args.split)
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    write_verification_answer_reports(paths, args.split, summary)
    print(f"Results: {output_path}", flush=True)
    print(
        f"Report: {paths.verification_answer_summary_markdown(args.split)}",
        flush=True,
    )
    return 0 if summary["completion_status"] == "complete" else 2


def analyze_verification_answers(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    selected_questions = load_validated_b1_results(
        paths,
        config,
        args.split,
    )
    b2_gate_run_fingerprint = load_validated_b2_gate(
        paths,
        config,
        args.split,
        selected_questions,
    )
    units = verification_answer_units(selected_questions)
    output_path = paths.verification_answer_results(args.split)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B3 results do not have one valid model digest")
    prompt_path, _ = load_verification_answer_prompt(config)
    run_config = build_verification_answer_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        question_results_path=paths.question_results(args.split),
        response_manifest_path=paths.response_manifest,
        b2_gate_run_fingerprint=b2_gate_run_fingerprint,
        model_digest=next(iter(model_digests)),
    )
    validate_existing_verification_answers(results, units, run_config)
    summary = verification_answer_summary(units, results, args.split)
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    write_verification_answer_reports(paths, args.split, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["completion_status"] == "complete" else 2


def make_answer_claim_evaluation_output_schema(
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = config["answer_claim_evaluation"]
    configured = {
        "alignment_validity": set(settings["alignment_validity_labels"]),
        "evidence_sufficiency": set(settings["evidence_sufficiency_labels"]),
        "answer_correctness": set(settings["answer_correctness_labels"]),
        "answer_stance": set(settings["answer_stance_labels"]),
    }
    expected = {
        "alignment_validity": ALIGNMENT_VALIDITY_LABELS,
        "evidence_sufficiency": EVIDENCE_SUFFICIENCY_LABELS,
        "answer_correctness": ANSWER_CORRECTNESS_LABELS,
        "answer_stance": ANSWER_STANCE_LABELS,
    }
    if configured != expected:
        raise ValueError("B4 label taxonomies differ from the frozen design")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "alignment_validity",
            "evidence_sufficiency",
            "answer_correctness",
            "answer_stance",
            "rationale",
        ],
        "properties": {
            "alignment_validity": {
                "type": "string",
                "enum": list(settings["alignment_validity_labels"]),
            },
            "evidence_sufficiency": {
                "type": "string",
                "enum": list(settings["evidence_sufficiency_labels"]),
            },
            "answer_correctness": {
                "type": "string",
                "enum": list(settings["answer_correctness_labels"]),
            },
            "answer_stance": {
                "type": "string",
                "enum": list(settings["answer_stance_labels"]),
            },
            "rationale": {
                "type": "string",
                "minLength": 3,
                "maxLength": 480,
            },
        },
    }


def load_answer_claim_evaluation_prompt(
    config: dict[str, Any],
) -> tuple[Path, str]:
    path = PROJECT_ROOT / config["answer_claim_evaluation"]["prompt_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"B4 prompt is empty: {path}")
    found = {
        placeholder
        for placeholder in ANSWER_CLAIM_PLACEHOLDERS
        if placeholder in template
    }
    if found != ANSWER_CLAIM_PLACEHOLDERS:
        raise ValueError(f"B4 prompt placeholders are incomplete: {sorted(found)}")
    for placeholder in ANSWER_CLAIM_PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ValueError(f"B4 prompt must contain {placeholder} exactly once")
    return path, template


def build_answer_claim_evaluation_prompt(
    template: str,
    unit: dict[str, Any],
) -> str:
    replacements = {
        "{verification_question_json}": json.dumps(
            unit["verification_question"],
            ensure_ascii=False,
        ),
        "{verification_answer_json}": json.dumps(
            unit["verification_answer"],
            ensure_ascii=False,
        ),
        "{candidate_gold_claim_json}": json.dumps(
            unit["candidate_gold_claim"],
            ensure_ascii=False,
        ),
        "{oracle_evidence_text}": unit["oracle_evidence_text"],
    }
    prompt = template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def parse_answer_claim_evaluation_output(
    raw_output: str,
) -> dict[str, str]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("B4 model output is empty")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError(f"B4 output is not strict JSON: {error}") from error
    expected_fields = {
        "alignment_validity",
        "evidence_sufficiency",
        "answer_correctness",
        "answer_stance",
        "rationale",
    }
    if not isinstance(parsed, dict) or set(parsed) != expected_fields:
        raise ValueError(f"B4 output fields must be exactly {sorted(expected_fields)}")
    label_sets = {
        "alignment_validity": ALIGNMENT_VALIDITY_LABELS,
        "evidence_sufficiency": EVIDENCE_SUFFICIENCY_LABELS,
        "answer_correctness": ANSWER_CORRECTNESS_LABELS,
        "answer_stance": ANSWER_STANCE_LABELS,
    }
    normalized: dict[str, str] = {}
    for field, labels in label_sets.items():
        value = parsed[field]
        if value not in labels:
            raise ValueError(f"Invalid B4 {field}: {value}")
        normalized[field] = value
    rationale = parsed["rationale"]
    if not isinstance(rationale, str):
        raise TypeError("B4 rationale must be a string")
    rationale = " ".join(rationale.split())
    if not 3 <= len(rationale) <= 480:
        raise ValueError("B4 rationale must contain 3-480 characters")
    normalized["rationale"] = rationale
    return normalized


def preflight_answer_claim_evaluation_ollama(
    client: Any,
    config: dict[str, Any],
) -> str:
    settings = config["answer_claim_evaluation"]
    model = settings["model"]
    try:
        response = client.list()
    except Exception as error:
        raise ConnectionError(
            "Ollama preflight failed before any B4 output was changed"
        ) from error
    models = response_value(response, "models")
    if not isinstance(models, (list, tuple)):
        raise ValueError("Ollama preflight returned no model list")
    available: dict[str, str | None] = {}
    for item in models:
        name = response_value(item, "model") or response_value(item, "name")
        if isinstance(name, str) and name:
            digest = response_value(item, "digest")
            available[name] = digest if isinstance(digest, str) else None
    if model not in available:
        raise ValueError(
            f"Configured B4 model {model!r} is not installed; "
            f"available={sorted(available)}"
        )
    digest = available[model]
    if not digest:
        raise ValueError(f"Ollama provided no digest for {model!r}")
    expected = settings["expected_model_digest"]
    if digest != expected:
        raise ValueError(
            "Installed model digest differs from the frozen B4 configuration: "
            f"expected={expected}, actual={digest}"
        )
    return digest


def load_validated_b3_results(
    paths: Any,
    config: dict[str, Any],
    split: str,
    selected_questions: list[dict[str, Any]],
    b2_gate_run_fingerprint: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    units = verification_answer_units(selected_questions)
    output_path = paths.verification_answer_results(split)
    if not output_path.exists():
        raise FileNotFoundError(f"B3 output is missing for {split}: {output_path}")
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B3 results do not have one valid model digest")
    prompt_path, _ = load_verification_answer_prompt(config)
    run_config = build_verification_answer_run_config(
        config,
        split=split,
        prompt_path=prompt_path,
        question_results_path=paths.question_results(split),
        response_manifest_path=paths.response_manifest,
        b2_gate_run_fingerprint=b2_gate_run_fingerprint,
        model_digest=next(iter(model_digests)),
    )
    validate_existing_verification_answers(results, units, run_config)
    ok_ids = {
        row["question_id"] for row in results if row.get("status") == "ok"
    }
    expected_ids = {unit["question_id"] for unit in units}
    if ok_ids != expected_ids:
        raise ValueError(f"B3 {split} is incomplete; finish answers before B4")
    return results, run_config


def answer_claim_evaluation_units(
    paths: Any,
    split: str,
    b3_results: list[dict[str, Any]],
    gold_by_response: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    pair_path = paths.alignment_pairs(split)
    if not pair_path.exists():
        raise FileNotFoundError(pair_path)
    pair_rows = load_jsonl(pair_path)
    useful_pairs = [
        row
        for row in pair_rows
        if row.get("relation") in {"DIRECT", "PARTIAL"}
    ]
    b3_by_question = {row["question_id"]: row for row in b3_results}
    gold_by_id = {
        row["claim_id"]: row
        for rows in gold_by_response.values()
        for row in rows
    }
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in useful_pairs:
        question_id = pair.get("question_id")
        claim_id = pair.get("claim_id")
        if question_id not in b3_by_question:
            raise ValueError(f"B2 pair has no B3 answer: {question_id}")
        if claim_id not in gold_by_id:
            raise ValueError(f"B2 pair has unknown claim: {claim_id}")
        b3 = b3_by_question[question_id]
        gold = gold_by_id[claim_id]
        if b3["response_id"] != gold["response_id"]:
            raise ValueError(f"Cross-response B4 pair: {question_id}/{claim_id}")
        evaluation_id = f"{question_id}__{claim_id}"
        if evaluation_id in seen:
            raise ValueError(f"Duplicate B4 evaluation ID: {evaluation_id}")
        seen.add(evaluation_id)
        evidence = normalize_oracle_evidence(gold.get("gold_evidence"))
        evidence_usable = evidence.get("status") == "ok"
        evidence_text = evidence.get("normalized_text")
        if not evidence_usable:
            evidence_text = "[No usable oracle evidence was supplied.]"
        units.append(
            {
                "evaluation_id": evaluation_id,
                "response_id": b3["response_id"],
                "split": split,
                "question_id": question_id,
                "verification_question": b3["verification_question"],
                "verification_answer": b3["verification_answer"],
                "b3_self_reported_answer_status": b3["answer_status"],
                "b3_run_fingerprint": b3["run_fingerprint"],
                "claim_id": claim_id,
                "candidate_gold_claim": gold["gold_claim"],
                "b2_relation": pair["relation"],
                "human_label": gold["human_label"],
                "oracle_evidence_usable": evidence_usable,
                "oracle_evidence_status": evidence["status"],
                "oracle_evidence_text": evidence_text,
                "oracle_evidence_sha256": evidence.get("normalized_sha256"),
                "oracle_evidence_valid_item_count": evidence.get(
                    "valid_item_count"
                ),
            }
        )
    return units


def build_answer_claim_evaluation_run_config(
    config: dict[str, Any],
    *,
    split: str,
    prompt_path: Path,
    alignment_pairs_path: Path,
    b3_results_path: Path,
    gold_claims_path: Path,
    model_digest: str,
) -> dict[str, Any]:
    settings = config["answer_claim_evaluation"]
    schema = make_answer_claim_evaluation_output_schema(config)
    payload = {
        "experiment": config["experiment"],
        "stage": "B4_answer_correctness_and_inconsistency",
        "split": split,
        "annotation_status": "evidence_grounded_llm_silver_not_human_gold",
        "evaluation_unit": "one_question_claim_pair_per_model_call",
        "model": settings["model"],
        "model_digest": model_digest,
        "temperature": float(settings["temperature"]),
        "seed": int(settings["seed"]),
        "num_predict": int(settings["num_predict"]),
        "think": bool(settings["think"]),
        "timeout_seconds": float(settings["timeout_seconds"]),
        "max_retries": int(settings["max_retries"]),
        "max_consecutive_request_errors": int(
            settings["max_consecutive_request_errors"]
        ),
        "prompt_version": settings["prompt_version"],
        "prompt_sha256": sha256_file(prompt_path),
        "result_schema_version": settings["result_schema_version"],
        "output_schema_version": settings["output_schema_version"],
        "output_schema_sha256": canonical_json_hash(schema),
        "alignment_pairs_sha256": sha256_file(alignment_pairs_path),
        "b3_results_sha256": sha256_file(b3_results_path),
        "gold_claims_sha256": sha256_file(gold_claims_path),
        "model_input_fields": config["leakage_policy"][
            "answer_claim_evaluation_model_input_fields"
        ],
        "withheld_fields": config["leakage_policy"][
            "answer_claim_evaluation_withheld_fields"
        ],
    }
    return {**payload, "run_fingerprint": canonical_json_hash(payload)}


def create_answer_claim_evaluation_result_base(
    unit: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "result_schema_version": run_config["result_schema_version"],
        "evaluation_id": unit["evaluation_id"],
        "response_id": unit["response_id"],
        "split": unit["split"],
        "question_id": unit["question_id"],
        "verification_question": unit["verification_question"],
        "verification_answer": unit["verification_answer"],
        "b3_self_reported_answer_status": unit[
            "b3_self_reported_answer_status"
        ],
        "claim_id": unit["claim_id"],
        "candidate_gold_claim": unit["candidate_gold_claim"],
        "b2_relation": unit["b2_relation"],
        "oracle_evidence_usable": unit["oracle_evidence_usable"],
        "oracle_evidence_status": unit["oracle_evidence_status"],
        "oracle_evidence_sha256": unit["oracle_evidence_sha256"],
        "oracle_evidence_valid_item_count": unit[
            "oracle_evidence_valid_item_count"
        ],
        "stage": run_config["stage"],
        "annotation_status": run_config["annotation_status"],
        "evaluation_unit": run_config["evaluation_unit"],
        "model_input_fields": run_config["model_input_fields"],
        "withheld_fields": run_config["withheld_fields"],
        "human_label_exposed": False,
        "b3_run_fingerprint": unit["b3_run_fingerprint"],
        "model": run_config["model"],
        "model_digest": run_config["model_digest"],
        "temperature": run_config["temperature"],
        "seed": run_config["seed"],
        "num_predict": run_config["num_predict"],
        "think": run_config["think"],
        "timeout_seconds": run_config["timeout_seconds"],
        "max_retries": run_config["max_retries"],
        "max_consecutive_request_errors": run_config[
            "max_consecutive_request_errors"
        ],
        "prompt_version": run_config["prompt_version"],
        "prompt_sha256": run_config["prompt_sha256"],
        "output_schema_version": run_config["output_schema_version"],
        "output_schema_sha256": run_config["output_schema_sha256"],
        "alignment_pairs_sha256": run_config["alignment_pairs_sha256"],
        "b3_results_sha256": run_config["b3_results_sha256"],
        "gold_claims_sha256": run_config["gold_claims_sha256"],
        "run_fingerprint": run_config["run_fingerprint"],
    }


def process_answer_claim_evaluation_unit(
    unit: dict[str, Any],
    template: str,
    config: dict[str, Any],
    run_config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    result = create_answer_claim_evaluation_result_base(unit, run_config)
    prompt = build_answer_claim_evaluation_prompt(template, unit)
    raw_output: str | None = None
    metadata: dict[str, Any] = {}
    request_error: Exception | None = None
    attempts = 0
    started = time.perf_counter()
    for attempt in range(run_config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw_output, metadata = call_ollama(
                client,
                run_config,
                make_answer_claim_evaluation_output_schema(config),
                prompt,
            )
            request_error = None
            break
        except Exception as error:
            request_error = error
            if attempt < run_config["max_retries"]:
                time.sleep(1.0)
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
                "alignment_validity": None,
                "evidence_sufficiency": None,
                "answer_correctness": None,
                "answer_stance": None,
                "rationale": None,
                "error": f"{type(request_error).__name__}: {request_error}",
            }
        )
        return result
    try:
        parsed = parse_answer_claim_evaluation_output(raw_output or "")
    except Exception as error:
        result.update(
            {
                "status": "parse_error",
                "alignment_validity": None,
                "evidence_sufficiency": None,
                "answer_correctness": None,
                "answer_stance": None,
                "rationale": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return result
    result.update({"status": "ok", **parsed, "error": None})
    return result


def validate_existing_answer_claim_evaluations(
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    unit_by_id = {unit["evaluation_id"]: unit for unit in units}
    seen: set[str] = set()
    for row in rows:
        evaluation_id = row.get("evaluation_id")
        if evaluation_id not in unit_by_id:
            raise ValueError(f"B4 output has unexpected ID: {evaluation_id}")
        if evaluation_id in seen:
            raise ValueError(f"Duplicate B4 evaluation ID: {evaluation_id}")
        seen.add(evaluation_id)
        expected_base = create_answer_claim_evaluation_result_base(
            unit_by_id[evaluation_id],
            run_config,
        )
        for key, expected in expected_base.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"Existing B4 output is incompatible for "
                    f"{evaluation_id}: {key}"
                )
        status = row.get("status")
        if status not in {"ok", "request_error", "parse_error"}:
            raise ValueError(f"Invalid B4 status for {evaluation_id}: {status}")
        if status == "ok":
            parse_answer_claim_evaluation_output(
                json.dumps(
                    {
                        "alignment_validity": row.get("alignment_validity"),
                        "evidence_sufficiency": row.get(
                            "evidence_sufficiency"
                        ),
                        "answer_correctness": row.get("answer_correctness"),
                        "answer_stance": row.get("answer_stance"),
                        "rationale": row.get("rationale"),
                    },
                    ensure_ascii=False,
                )
            )
            if row.get("error") is not None:
                raise ValueError(
                    f"Successful B4 row has error for {evaluation_id}"
                )
        else:
            for field in (
                "alignment_validity",
                "evidence_sufficiency",
                "answer_correctness",
                "answer_stance",
                "rationale",
            ):
                if row.get(field) is not None:
                    raise ValueError(
                        f"Technical B4 failure has {field}: {evaluation_id}"
                    )


def _b4_pair_outcome(row: dict[str, Any], human_label: str) -> str:
    if row["alignment_validity"] == "INVALID":
        return "INVALID_ALIGNMENT"
    if row["answer_correctness"] == "UNVERIFIABLE":
        return "UNVERIFIABLE"
    if row["answer_correctness"] == "INSUFFICIENT":
        return "INSUFFICIENT_ANSWER"
    correctish = row["answer_correctness"] in {
        "CORRECT",
        "PARTIALLY_CORRECT",
    }
    partial_prefix = (
        "PARTIAL_" if row["answer_correctness"] == "PARTIALLY_CORRECT" else ""
    )
    stance = row["answer_stance"]
    if human_label == "FACTUAL":
        if stance == "SUPPORTS_CLAIM" and correctish:
            return f"{partial_prefix}CORRECT_CONFIRMATION"
        if stance == "CHALLENGES_CLAIM":
            return "FALSE_CHALLENGE"
    if human_label == "NON_FACTUAL":
        if stance == "CHALLENGES_CLAIM" and correctish:
            return f"{partial_prefix}CORRECT_CHALLENGE"
        if stance == "SUPPORTS_CLAIM":
            return "FALSE_CONFIRMATION"
    if stance == "MIXED":
        return "MIXED_STANCE"
    if stance == "NO_POSITION":
        return "NO_POSITION"
    return "INCORRECT_OR_INCONSISTENT"


def answer_claim_evaluation_summary(
    units: list[dict[str, Any]],
    results: list[dict[str, Any]],
    split: str,
    all_gold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result_by_id = {row["evaluation_id"]: row for row in results}
    unit_by_id = {unit["evaluation_id"]: unit for unit in units}
    status_counts = Counter(
        result_by_id.get(unit["evaluation_id"], {}).get("status", "missing")
        for unit in units
    )
    successful = [
        result_by_id[unit["evaluation_id"]]
        for unit in units
        if result_by_id.get(unit["evaluation_id"], {}).get("status") == "ok"
    ]
    alignment_counts = Counter(row["alignment_validity"] for row in successful)
    evidence_counts = Counter(row["evidence_sufficiency"] for row in successful)
    correctness_counts = Counter(row["answer_correctness"] for row in successful)
    stance_counts = Counter(row["answer_stance"] for row in successful)
    primary = [
        row
        for row in successful
        if row["oracle_evidence_usable"]
        and row["alignment_validity"] != "INVALID"
    ]
    primary_correctness = Counter(row["answer_correctness"] for row in primary)
    pair_outcomes: Counter[str] = Counter()
    evaluated_rows: list[dict[str, Any]] = []
    for row in successful:
        human_label = unit_by_id[row["evaluation_id"]]["human_label"]
        outcome = _b4_pair_outcome(row, human_label)
        pair_outcomes[outcome] += 1
        evaluated_rows.append(
            {
                **row,
                "human_label": human_label,
                "derived_pair_outcome": outcome,
            }
        )

    claim_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated_rows:
        claim_groups[row["claim_id"]].append(row)
    claim_metrics: dict[str, Any] = {}
    for label in ("FACTUAL", "NON_FACTUAL"):
        label_groups = {
            claim_id: rows
            for claim_id, rows in claim_groups.items()
            if rows[0]["human_label"] == label
        }
        evidence_evaluable = {
            claim_id
            for claim_id, rows in label_groups.items()
            if any(
                row["oracle_evidence_usable"]
                and row["alignment_validity"] != "INVALID"
                for row in rows
            )
        }
        success_names = (
            {"CORRECT_CONFIRMATION", "PARTIAL_CORRECT_CONFIRMATION"}
            if label == "FACTUAL"
            else {"CORRECT_CHALLENGE", "PARTIAL_CORRECT_CHALLENGE"}
        )
        harm_name = "FALSE_CHALLENGE" if label == "FACTUAL" else "FALSE_CONFIRMATION"
        successful_claims = {
            claim_id
            for claim_id, rows in label_groups.items()
            if any(row["derived_pair_outcome"] in success_names for row in rows)
        }
        harmed_claims = {
            claim_id
            for claim_id, rows in label_groups.items()
            if any(row["derived_pair_outcome"] == harm_name for row in rows)
        }
        primary_successful_claims = successful_claims & evidence_evaluable
        primary_harmed_claims = harmed_claims & evidence_evaluable
        claim_metrics[label] = {
            "silver_covered_claims": len(label_groups),
            "evidence_evaluable_valid_anchor_claims": len(evidence_evaluable),
            "successful_claims": len(primary_successful_claims),
            "success_rate_among_evidence_evaluable": _rate(
                len(primary_successful_claims),
                len(evidence_evaluable),
            ),
            "harmed_or_reinforced_claims": len(primary_harmed_claims),
            "harm_rate_among_evidence_evaluable": _rate(
                len(primary_harmed_claims),
                len(evidence_evaluable),
            ),
        }

    split_gold = [
        row
        for row in all_gold_rows
        if (
            (split == "dev" and int(row["source_record_index"]) <= 20)
            or (split == "heldout" and int(row["source_record_index"]) >= 21)
        )
    ]
    all_label_counts = Counter(row["human_label"] for row in split_gold)
    funnel = {}
    for label in ("FACTUAL", "NON_FACTUAL"):
        metric = claim_metrics[label]
        funnel[label] = {
            "all_split_claims": all_label_counts[label],
            "silver_covered_claims": metric["silver_covered_claims"],
            "evidence_evaluable_valid_anchor_claims": metric[
                "evidence_evaluable_valid_anchor_claims"
            ],
            "successful_claims": metric["successful_claims"],
        }

    invalid_alignment_examples = [
        {
            "evaluation_id": row["evaluation_id"],
            "question_id": row["question_id"],
            "claim_id": row["claim_id"],
            "verification_question": row["verification_question"],
            "candidate_gold_claim": row["candidate_gold_claim"],
            "rationale": row["rationale"],
        }
        for row in successful
        if row["alignment_validity"] == "INVALID"
    ]
    complete = len(successful) == len(units)
    return {
        "schema_version": "cove_answer_claim_evaluation_summary_v1",
        "experiment": "Experiment B: CoVe mechanism evaluation",
        "stage": "B4_answer_correctness_and_inconsistency",
        "split": split,
        "annotation_status": "evidence_grounded_llm_silver_not_human_gold",
        "completion_status": "complete" if complete else "incomplete",
        "selected_pairs": len(units),
        "selected_questions": len({unit["question_id"] for unit in units}),
        "selected_claims": len({unit["claim_id"] for unit in units}),
        "successful_pairs": len(successful),
        "status_counts": dict(status_counts),
        "oracle_evidence_usable_pairs": sum(
            unit["oracle_evidence_usable"] for unit in units
        ),
        "alignment_validity_counts": {
            label: alignment_counts[label]
            for label in ("VALID_DIRECT", "VALID_PARTIAL", "INVALID")
        },
        "evidence_sufficiency_counts": {
            label: evidence_counts[label]
            for label in ("SUFFICIENT", "PARTIAL", "INSUFFICIENT")
        },
        "answer_correctness_counts_all_pairs": {
            label: correctness_counts[label]
            for label in (
                "CORRECT",
                "PARTIALLY_CORRECT",
                "INCORRECT",
                "INSUFFICIENT",
                "UNVERIFIABLE",
            )
        },
        "primary_valid_evidence_pair_count": len(primary),
        "primary_answer_correctness_counts": {
            label: primary_correctness[label]
            for label in (
                "CORRECT",
                "PARTIALLY_CORRECT",
                "INCORRECT",
                "INSUFFICIENT",
                "UNVERIFIABLE",
            )
        },
        "answer_stance_counts": {
            label: stance_counts[label]
            for label in (
                "SUPPORTS_CLAIM",
                "CHALLENGES_CLAIM",
                "MIXED",
                "NO_POSITION",
            )
        },
        "derived_pair_outcome_counts": dict(sorted(pair_outcomes.items())),
        "claim_metrics": claim_metrics,
        "claim_funnel": funnel,
        "invalid_alignment_examples": invalid_alignment_examples,
        "interpretation_notes": [
            "Human labels are withheld from the B4 model and joined only to derive pair and claim outcomes.",
            "All silver DIRECT/PARTIAL B2 pairs are audited for alignment validity.",
            "Answer correctness is primary only when oracle evidence is usable and the B4 anchor is valid.",
            "The same frozen Qwen weights generate and evaluate answers, so B4 remains evidence-grounded silver evaluation.",
        ],
    }


def build_answer_claim_evaluation_markdown(summary: dict[str, Any]) -> str:
    claim_metrics = summary["claim_metrics"]
    lines = [
        "# Experiment B — B4 Answer Correctness and Inconsistency",
        "",
        f"- Split: `{summary['split']}`",
        f"- Completion: **{summary['completion_status']}**",
        f"- Evaluated pairs: {summary['successful_pairs']}/{summary['selected_pairs']}",
        f"- Distinct questions/claims: {summary['selected_questions']}/{summary['selected_claims']}",
        f"- Oracle-evidence-usable pairs: {summary['oracle_evidence_usable_pairs']}",
        f"- Annotation: `{summary['annotation_status']}`",
        "",
        "## Alignment audit",
        "",
        "| Validity | Pairs |",
        "|---|---:|",
    ]
    for label in ("VALID_DIRECT", "VALID_PARTIAL", "INVALID"):
        lines.append(
            f"| {label} | {summary['alignment_validity_counts'][label]} |"
        )
    lines.extend(
        [
            "",
            "## Primary answer correctness",
            "",
            "Primary rows have usable oracle evidence and a B4-valid "
            "question–claim anchor.",
            "",
            "| Correctness | Pairs |",
            "|---|---:|",
        ]
    )
    for label in (
        "CORRECT",
        "PARTIALLY_CORRECT",
        "INCORRECT",
        "INSUFFICIENT",
        "UNVERIFIABLE",
    ):
        lines.append(
            f"| {label} | "
            f"{summary['primary_answer_correctness_counts'][label]} |"
        )
    lines.extend(
        [
            "",
            "## Claim-level mechanism outcomes",
            "",
            "| Label | Silver covered | Evidence-evaluable valid anchors | Successful confirmation/challenge | Success rate | Harm/reinforcement | Harm rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("FACTUAL", "NON_FACTUAL"):
        metric = claim_metrics[label]
        lines.append(
            f"| {label} | {metric['silver_covered_claims']} | "
            f"{metric['evidence_evaluable_valid_anchor_claims']} | "
            f"{metric['successful_claims']} | "
            f"{metric['success_rate_among_evidence_evaluable']} | "
            f"{metric['harmed_or_reinforced_claims']} | "
            f"{metric['harm_rate_among_evidence_evaluable']} |"
        )
    lines.extend(
        [
            "",
            "## Claim funnel",
            "",
            "| Label | All dev claims | B2 silver covered | B4 evidence-evaluable valid anchors | Successful B3 outcome |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in ("FACTUAL", "NON_FACTUAL"):
        item = summary["claim_funnel"][label]
        lines.append(
            f"| {label} | {item['all_split_claims']} | "
            f"{item['silver_covered_claims']} | "
            f"{item['evidence_evaluable_valid_anchor_claims']} | "
            f"{item['successful_claims']} |"
        )
    lines.extend(
        [
            "",
            "## Invalid silver-alignment examples",
            "",
            "| question_id | claim_id | question | claim |",
            "|---|---|---|---|",
        ]
    )
    if summary["invalid_alignment_examples"]:
        for row in summary["invalid_alignment_examples"]:
            question = row["verification_question"].replace("|", "\\|")
            claim = row["candidate_gold_claim"].replace("|", "\\|")
            lines.append(
                f"| `{row['question_id']}` | `{row['claim_id']}` | "
                f"{question} | {claim} |"
            )
    else:
        lines.append("| — | — | None | — |")
    lines.extend(
        [
            "",
            "## Reliability warning",
            "",
            "These are provisional silver labels, not authoritative accuracy "
            "estimates. Development inspection found both over-strict "
            "`INVALID` decisions for useful false-premise questions and "
            "multiple cases where the categorical correctness/stance labels "
            "contradict the model's own rationale. Do not use B4 labels as "
            "revision-control signals. See "
            "`B4_answer_claim_evaluation_dev_diagnostic.md` for the audited "
            "examples.",
            "",
            "## Interpretation boundary",
            "",
            "This is evidence-grounded LLM silver evaluation. Human labels were "
            "not shown to the evaluator. The same model family generated and "
            "evaluated answers, so final claims require this dependency to be "
            "reported and preferably checked by a small independent audit or "
            "sensitivity analysis.",
            "",
        ]
    )
    return "\n".join(lines)


def write_answer_claim_evaluation_reports(
    paths: Any,
    split: str,
    summary: dict[str, Any],
) -> None:
    atomic_write_json(paths.answer_claim_evaluation_summary_json(split), summary)
    atomic_write_text(
        paths.answer_claim_evaluation_summary_markdown(split),
        build_answer_claim_evaluation_markdown(summary),
    )


def run_answer_claim_evaluation(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    selected_questions = load_validated_b1_results(paths, config, args.split)
    b2_gate = load_validated_b2_gate(
        paths,
        config,
        args.split,
        selected_questions,
    )
    b3_results, _ = load_validated_b3_results(
        paths,
        config,
        args.split,
        selected_questions,
        b2_gate,
    )
    gold_path, all_gold, gold_by_response = load_gold_claims_by_response(config)
    units = answer_claim_evaluation_units(
        paths,
        args.split,
        b3_results,
        gold_by_response,
    )
    if not units:
        raise ValueError(f"No useful B2 pairs are available for B4 {args.split}")
    prompt_path, template = load_answer_claim_evaluation_prompt(config)

    if args.dry_run:
        run_config = build_answer_claim_evaluation_run_config(
            config,
            split=args.split,
            prompt_path=prompt_path,
            alignment_pairs_path=paths.alignment_pairs(args.split),
            b3_results_path=paths.verification_answer_results(args.split),
            gold_claims_path=gold_path,
            model_digest="DRY_RUN_NOT_CHECKED",
        )
        print("EXPERIMENT B B4 ANSWER–CLAIM EVALUATION DRY RUN")
        print(
            f"Split: {args.split}; pairs: {len(units)}; "
            f"questions: {len({unit['question_id'] for unit in units})}; "
            f"claims: {len({unit['claim_id'] for unit in units})}"
        )
        print(
            "Oracle-evidence-usable pairs: "
            f"{sum(unit['oracle_evidence_usable'] for unit in units)}"
        )
        print(f"Run fingerprint: {run_config['run_fingerprint']}")
        print(
            "Human labels, evidence stance/URLs, and B3 self-reported status "
            "are withheld from the B4 model."
        )
        for unit in units[:2]:
            print(f"\n--- Preview: {unit['evaluation_id']} ---")
            print(build_answer_claim_evaluation_prompt(template, unit))
        print("\nNo Ollama calls were made and no files were written.")
        return 0

    client = Client(
        host=args.ollama_host,
        timeout=float(config["answer_claim_evaluation"]["timeout_seconds"]),
    )
    model_digest = preflight_answer_claim_evaluation_ollama(client, config)
    run_config = build_answer_claim_evaluation_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        alignment_pairs_path=paths.alignment_pairs(args.split),
        b3_results_path=paths.verification_answer_results(args.split),
        gold_claims_path=gold_path,
        model_digest=model_digest,
    )
    output_path = paths.answer_claim_evaluation_results(args.split)
    existing = load_jsonl(output_path) if output_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Output already exists; rerun with --resume: {output_path}"
        )
    validate_existing_answer_claim_evaluations(existing, units, run_config)
    result_by_id = {row["evaluation_id"]: row for row in existing}
    pending = [
        unit
        for unit in units
        if result_by_id.get(unit["evaluation_id"], {}).get("status") != "ok"
    ]
    print(
        f"Experiment B B4: split={args.split}, total={len(units)}, "
        f"retained_ok={len(units) - len(pending)}, pending={len(pending)}",
        flush=True,
    )
    consecutive_request_errors = 0
    max_consecutive = int(
        config["answer_claim_evaluation"]["max_consecutive_request_errors"]
    )
    order = {unit["evaluation_id"]: index for index, unit in enumerate(units)}
    for unit in pending:
        overall_index = order[unit["evaluation_id"]] + 1
        print(
            f"[{overall_index}/{len(units)}] {unit['question_id']} → "
            f"{unit['claim_id']} evaluating answer ...",
            flush=True,
        )
        result = process_answer_claim_evaluation_unit(
            unit,
            template,
            config,
            run_config,
            client,
        )
        result_by_id[unit["evaluation_id"]] = result
        ordered = [
            result_by_id[item["evaluation_id"]]
            for item in units
            if item["evaluation_id"] in result_by_id
        ]
        atomic_write_jsonl(output_path, ordered)
        if result["status"] == "ok":
            consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(units)}] success, "
                f"{result['alignment_validity']}, "
                f"{result['answer_correctness']}, "
                f"{result['answer_stance']}, "
                f"{result['latency_seconds']:.1f}s",
                flush=True,
            )
        else:
            if result["status"] == "request_error":
                consecutive_request_errors += 1
            else:
                consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(units)}] {result['status']}: "
                f"{result['error']}",
                flush=True,
            )
            if consecutive_request_errors >= max_consecutive:
                print(
                    "Stopping after the configured consecutive request-error "
                    "limit. The same command will resume safely.",
                    file=sys.stderr,
                    flush=True,
                )
                break
    all_results = [
        result_by_id[unit["evaluation_id"]]
        for unit in units
        if unit["evaluation_id"] in result_by_id
    ]
    validate_existing_answer_claim_evaluations(all_results, units, run_config)
    summary = answer_claim_evaluation_summary(
        units,
        all_results,
        args.split,
        all_gold,
    )
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    write_answer_claim_evaluation_reports(paths, args.split, summary)
    print(f"Results: {output_path}", flush=True)
    print(
        f"Report: {paths.answer_claim_evaluation_summary_markdown(args.split)}",
        flush=True,
    )
    return 0 if summary["completion_status"] == "complete" else 2


def analyze_answer_claim_evaluation(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    selected_questions = load_validated_b1_results(paths, config, args.split)
    b2_gate = load_validated_b2_gate(
        paths,
        config,
        args.split,
        selected_questions,
    )
    b3_results, _ = load_validated_b3_results(
        paths,
        config,
        args.split,
        selected_questions,
        b2_gate,
    )
    gold_path, all_gold, gold_by_response = load_gold_claims_by_response(config)
    units = answer_claim_evaluation_units(
        paths,
        args.split,
        b3_results,
        gold_by_response,
    )
    output_path = paths.answer_claim_evaluation_results(args.split)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B4 results do not have one valid model digest")
    prompt_path, _ = load_answer_claim_evaluation_prompt(config)
    run_config = build_answer_claim_evaluation_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        alignment_pairs_path=paths.alignment_pairs(args.split),
        b3_results_path=paths.verification_answer_results(args.split),
        gold_claims_path=gold_path,
        model_digest=next(iter(model_digests)),
    )
    validate_existing_answer_claim_evaluations(results, units, run_config)
    summary = answer_claim_evaluation_summary(
        units,
        results,
        args.split,
        all_gold,
    )
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    write_answer_claim_evaluation_reports(paths, args.split, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["completion_status"] == "complete" else 2


def make_revision_output_schema(config: dict[str, Any]) -> dict[str, Any]:
    _ = config["response_revision"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["revised_response"],
        "properties": {
            "revised_response": {
                "type": "string",
                "minLength": 20,
                "maxLength": 30000,
            }
        },
    }


def load_revision_prompt(
    config: dict[str, Any],
) -> tuple[Path, str]:
    path = PROJECT_ROOT / config["response_revision"]["prompt_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"Revision prompt is empty: {path}")
    found = {
        placeholder
        for placeholder in REVISION_PLACEHOLDERS
        if placeholder in template
    }
    if found != REVISION_PLACEHOLDERS:
        raise ValueError(
            f"Revision prompt placeholders are incomplete: {sorted(found)}"
        )
    for placeholder in REVISION_PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ValueError(
                f"Revision prompt must contain {placeholder} exactly once"
            )
    return path, template


def build_revision_prompt(
    template: str,
    unit: dict[str, Any],
) -> str:
    replacements = {
        "{original_question_json}": json.dumps(
            unit["original_question"],
            ensure_ascii=False,
        ),
        "{initial_response_json}": json.dumps(
            unit["initial_response"],
            ensure_ascii=False,
        ),
        "{verification_results_json}": json.dumps(
            unit["verification_results"],
            ensure_ascii=False,
            indent=2,
        ),
    }
    output = template
    for placeholder, value in replacements.items():
        output = output.replace(placeholder, value)
    return output


def parse_revision_output(raw_output: str) -> str:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("Revision model output is empty")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError(f"Revision output is not strict JSON: {error}") from error
    if not isinstance(parsed, dict) or set(parsed) != {"revised_response"}:
        raise ValueError(
            "Revision output must contain only the revised_response field"
        )
    revised = parsed["revised_response"]
    if not isinstance(revised, str):
        raise TypeError("revised_response must be a string")
    revised = revised.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not 20 <= len(revised) <= 30000:
        raise ValueError("revised_response must contain 20-30000 characters")
    return revised


def recover_revision_output_format(
    raw_output: str,
) -> tuple[str, str]:
    """Recover known format-only B5 deviations without changing prose."""

    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("Revision model output is empty")
    raw_output = raw_output.strip()
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and len(parsed) == 1:
        key, value = next(iter(parsed.items()))
        if key in {"answer", "response"} and isinstance(value, str):
            revised = parse_revision_output(
                json.dumps(
                    {"revised_response": value},
                    ensure_ascii=False,
                )
            )
            return revised, f"alias_key_{key}"
        if isinstance(key, str) and isinstance(value, str):
            revised = parse_revision_output(
                json.dumps(
                    {"revised_response": value},
                    ensure_ascii=False,
                )
            )
            return revised, "singleton_arbitrary_key_string_value"

    if (
        isinstance(parsed, dict)
        and len(parsed) > 1
        and all(isinstance(key, str) for key in parsed)
        and all(isinstance(value, str) for value in parsed.values())
    ):
        rendered = "\n".join(
            f"{key}: {value}" for key, value in parsed.items()
        )
        revised = parse_revision_output(
            json.dumps(
                {"revised_response": rendered},
                ensure_ascii=False,
            )
        )
        return revised, "flat_string_object_rendered_as_text"

    if raw_output.startswith("{") and raw_output.endswith("}"):
        inner = raw_output[1:-1].strip()
        try:
            singleton = json.loads(inner)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Revision output does not match a recoverable format-only "
                f"pattern: {error}"
            ) from error
        if isinstance(singleton, str):
            revised = parse_revision_output(
                json.dumps(
                    {"revised_response": singleton},
                    ensure_ascii=False,
                )
            )
            return revised, "singleton_string_key_without_colon"
    raise ValueError(
        "Revision output does not match a recoverable format-only pattern"
    )


def preflight_revision_ollama(
    client: Any,
    config: dict[str, Any],
) -> str:
    settings = config["response_revision"]
    model = settings["model"]
    try:
        response = client.list()
    except Exception as error:
        raise ConnectionError(
            "Ollama preflight failed before any B5 output was changed"
        ) from error
    models = response_value(response, "models")
    if not isinstance(models, (list, tuple)):
        raise ValueError("Ollama preflight returned no model list")
    available: dict[str, str | None] = {}
    for item in models:
        name = response_value(item, "model") or response_value(item, "name")
        if isinstance(name, str) and name:
            digest = response_value(item, "digest")
            available[name] = digest if isinstance(digest, str) else None
    if model not in available:
        raise ValueError(
            f"Configured B5 model {model!r} is not installed; "
            f"available={sorted(available)}"
        )
    digest = available[model]
    if not digest:
        raise ValueError(f"Ollama provided no digest for {model!r}")
    expected = settings["expected_model_digest"]
    if digest != expected:
        raise ValueError(
            "Installed model digest differs from the frozen B5 configuration: "
            f"expected={expected}, actual={digest}"
        )
    return digest


def load_b4_completion_gate(paths: Any, split: str) -> str:
    summary_path = paths.answer_claim_evaluation_summary_json(split)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"B4 summary is missing for {split}: {summary_path}"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid B4 summary JSON: {error}") from error
    if not isinstance(summary, dict):
        raise TypeError("B4 summary must be a JSON object")
    if summary.get("split") != split:
        raise ValueError("B4 summary split does not match the B5 split")
    if summary.get("completion_status") != "complete":
        raise ValueError(f"B4 {split} is incomplete; do not start B5")
    if summary.get("successful_pairs") != summary.get("selected_pairs"):
        raise ValueError("B4 completion summary has inconsistent pair counts")
    return sha256_file(summary_path)


def revision_units(
    selected_responses: list[dict[str, Any]],
    b3_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    b3_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in b3_results:
        if row.get("status") != "ok":
            raise ValueError(
                f"B5 cannot use non-ok B3 row: {row.get('question_id')}"
            )
        b3_by_response[row["response_id"]].append(row)
    units: list[dict[str, Any]] = []
    for response in selected_responses:
        response_id = response["response_id"]
        answer_rows = sorted(
            b3_by_response.get(response_id, []),
            key=lambda row: (int(row["question_index"]), row["question_id"]),
        )
        if not answer_rows:
            raise ValueError(f"No B3 answers available for {response_id}")
        question_indices = [int(row["question_index"]) for row in answer_rows]
        if question_indices != list(range(1, len(answer_rows) + 1)):
            raise ValueError(f"Non-contiguous B3 question order for {response_id}")
        verification_results = [
            {
                "question_id": row["question_id"],
                "verification_question": row["verification_question"],
                "verification_answer": row["verification_answer"],
            }
            for row in answer_rows
        ]
        units.append(
            {
                "response_id": response_id,
                "source_record_index": response["source_record_index"],
                "split": response["split"],
                "original_question": response["original_question"],
                "initial_response": response["initial_response"],
                "input_sha256": canonical_json_hash(
                    {
                        "original_question": response["original_question"],
                        "initial_response": response["initial_response"],
                    }
                ),
                "verification_results": verification_results,
                "verification_results_sha256": canonical_json_hash(
                    verification_results
                ),
                "b3_run_fingerprint": answer_rows[0]["run_fingerprint"],
            }
        )
    extra_response_ids = set(b3_by_response) - {
        row["response_id"] for row in selected_responses
    }
    if extra_response_ids:
        raise ValueError(
            f"B3 contains responses outside B5 split: {sorted(extra_response_ids)}"
        )
    return units


def build_revision_run_config(
    config: dict[str, Any],
    *,
    split: str,
    prompt_path: Path,
    response_manifest_path: Path,
    b3_results_path: Path,
    b4_gate_sha256: str,
    model_digest: str,
) -> dict[str, Any]:
    settings = config["response_revision"]
    schema = make_revision_output_schema(config)
    payload = {
        "experiment": config["experiment"],
        "stage": "B5_cove_response_revision",
        "split": split,
        "revision_unit": "one_model_call_per_response",
        "revision_policy": "standard_cove_qa_only",
        "model": settings["model"],
        "model_digest": model_digest,
        "temperature": float(settings["temperature"]),
        "seed": int(settings["seed"]),
        "num_predict": int(settings["num_predict"]),
        "think": bool(settings["think"]),
        "timeout_seconds": float(settings["timeout_seconds"]),
        "max_retries": int(settings["max_retries"]),
        "max_consecutive_request_errors": int(
            settings["max_consecutive_request_errors"]
        ),
        "prompt_version": settings["prompt_version"],
        "prompt_sha256": sha256_file(prompt_path),
        "result_schema_version": settings["result_schema_version"],
        "output_schema_version": settings["output_schema_version"],
        "output_schema_sha256": canonical_json_hash(schema),
        "response_manifest_sha256": sha256_file(response_manifest_path),
        "b3_results_sha256": sha256_file(b3_results_path),
        "b4_completion_gate_sha256": b4_gate_sha256,
        "model_input_fields": config["leakage_policy"][
            "response_revision_model_input_fields"
        ],
        "withheld_fields": config["leakage_policy"][
            "response_revision_withheld_fields"
        ],
    }
    return {**payload, "run_fingerprint": canonical_json_hash(payload)}


def create_revision_result_base(
    unit: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "result_schema_version": run_config["result_schema_version"],
        "response_id": unit["response_id"],
        "source_record_index": unit["source_record_index"],
        "split": unit["split"],
        "original_question": unit["original_question"],
        "original_question_sha256": sha256_text(unit["original_question"]),
        "initial_response_sha256": sha256_text(unit["initial_response"]),
        "input_sha256": unit["input_sha256"],
        "verification_question_count": len(unit["verification_results"]),
        "verification_results_sha256": unit[
            "verification_results_sha256"
        ],
        "stage": run_config["stage"],
        "revision_unit": run_config["revision_unit"],
        "revision_policy": run_config["revision_policy"],
        "model_input_fields": run_config["model_input_fields"],
        "withheld_fields": run_config["withheld_fields"],
        "b3_run_fingerprint": unit["b3_run_fingerprint"],
        "b3_results_sha256": run_config["b3_results_sha256"],
        "b4_completion_gate_sha256": run_config[
            "b4_completion_gate_sha256"
        ],
        "model": run_config["model"],
        "model_digest": run_config["model_digest"],
        "temperature": run_config["temperature"],
        "seed": run_config["seed"],
        "num_predict": run_config["num_predict"],
        "think": run_config["think"],
        "timeout_seconds": run_config["timeout_seconds"],
        "max_retries": run_config["max_retries"],
        "max_consecutive_request_errors": run_config[
            "max_consecutive_request_errors"
        ],
        "prompt_version": run_config["prompt_version"],
        "prompt_sha256": run_config["prompt_sha256"],
        "output_schema_version": run_config["output_schema_version"],
        "output_schema_sha256": run_config["output_schema_sha256"],
        "response_manifest_sha256": run_config["response_manifest_sha256"],
        "run_fingerprint": run_config["run_fingerprint"],
    }


def process_revision_unit(
    unit: dict[str, Any],
    template: str,
    config: dict[str, Any],
    run_config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    result = create_revision_result_base(unit, run_config)
    prompt = build_revision_prompt(template, unit)
    raw_output: str | None = None
    metadata: dict[str, Any] = {}
    request_error: Exception | None = None
    attempts = 0
    started = time.perf_counter()
    for attempt in range(run_config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw_output, metadata = call_ollama(
                client,
                run_config,
                make_revision_output_schema(config),
                prompt,
            )
            request_error = None
            break
        except Exception as error:
            request_error = error
            if attempt < run_config["max_retries"]:
                time.sleep(1.0)
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
                "revised_response": None,
                "error": f"{type(request_error).__name__}: {request_error}",
            }
        )
        return result
    try:
        revised = parse_revision_output(raw_output or "")
        format_recovery = None
    except Exception as strict_error:
        try:
            revised, recovery_method = recover_revision_output_format(
                raw_output or ""
            )
            format_recovery = {
                "applied": True,
                "method": recovery_method,
                "strict_parse_error": (
                    f"{type(strict_error).__name__}: {strict_error}"
                ),
                "model_recalled": False,
                "prose_changed": False,
            }
        except Exception as recovery_error:
            result.update(
                {
                    "status": "parse_error",
                    "revised_response": None,
                    "format_recovery": None,
                    "error": (
                        f"{type(strict_error).__name__}: {strict_error}; "
                        "format recovery failed: "
                        f"{type(recovery_error).__name__}: {recovery_error}"
                    ),
                }
            )
            return result
    result.update(
        {
            "status": "ok",
            "revised_response": revised,
            "revised_response_sha256": sha256_text(revised),
            "format_recovery": format_recovery,
            "error": None,
        }
    )
    return result


def validate_existing_revisions(
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    unit_by_id = {unit["response_id"]: unit for unit in units}
    seen: set[str] = set()
    for row in rows:
        response_id = row.get("response_id")
        if response_id not in unit_by_id:
            raise ValueError(
                f"B5 output contains unexpected response_id: {response_id}"
            )
        if response_id in seen:
            raise ValueError(f"Duplicate response_id in B5 output: {response_id}")
        seen.add(response_id)
        expected_base = create_revision_result_base(
            unit_by_id[response_id],
            run_config,
        )
        for key, expected in expected_base.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"Existing B5 output is incompatible for {response_id}: {key}"
                )
        status = row.get("status")
        if status not in {"ok", "request_error", "parse_error"}:
            raise ValueError(f"Invalid B5 status for {response_id}: {status}")
        if status == "ok":
            revised = parse_revision_output(
                json.dumps(
                    {"revised_response": row.get("revised_response")},
                    ensure_ascii=False,
                )
            )
            if row.get("revised_response_sha256") != sha256_text(revised):
                raise ValueError(
                    f"B5 revised-response hash mismatch for {response_id}"
                )
            if row.get("error") is not None:
                raise ValueError(
                    f"Successful B5 row has an error for {response_id}"
                )
        else:
            if row.get("revised_response") is not None:
                raise ValueError(
                    f"Technical B5 failure has revised text for {response_id}"
                )


def _word_count(text: str) -> int:
    return len(text.split())


def revision_summary(
    units: list[dict[str, Any]],
    results: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    result_by_id = {row["response_id"]: row for row in results}
    status_counts = Counter(
        result_by_id.get(unit["response_id"], {}).get("status", "missing")
        for unit in units
    )
    successful = [
        result_by_id[unit["response_id"]]
        for unit in units
        if result_by_id.get(unit["response_id"], {}).get("status") == "ok"
    ]
    unit_by_id = {unit["response_id"]: unit for unit in units}
    per_response: list[dict[str, Any]] = []
    initial_words: list[int] = []
    revised_words: list[int] = []
    word_deltas: list[int] = []
    char_deltas: list[int] = []
    unchanged = 0
    latencies: list[float] = []
    recovery_methods: Counter[str] = Counter()
    for row in successful:
        unit = unit_by_id[row["response_id"]]
        initial = unit["initial_response"]
        revised = row["revised_response"]
        initial_word_count = _word_count(initial)
        revised_word_count = _word_count(revised)
        word_delta = revised_word_count - initial_word_count
        char_delta = len(revised) - len(initial)
        is_unchanged = " ".join(initial.split()) == " ".join(revised.split())
        unchanged += int(is_unchanged)
        initial_words.append(initial_word_count)
        revised_words.append(revised_word_count)
        word_deltas.append(word_delta)
        char_deltas.append(char_delta)
        latency = row.get("latency_seconds")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        recovery = row.get("format_recovery")
        if isinstance(recovery, dict) and recovery.get("applied") is True:
            recovery_methods[str(recovery.get("method"))] += 1
        per_response.append(
            {
                "response_id": row["response_id"],
                "verification_questions": row["verification_question_count"],
                "initial_words": initial_word_count,
                "revised_words": revised_word_count,
                "word_delta": word_delta,
                "char_delta": char_delta,
                "text_unchanged": is_unchanged,
                "latency_seconds": row.get("latency_seconds"),
            }
        )
    complete = len(successful) == len(units)
    return {
        "schema_version": "cove_response_revision_summary_v1",
        "experiment": "Experiment B: CoVe mechanism evaluation",
        "stage": "B5_cove_response_revision",
        "split": split,
        "completion_status": "complete" if complete else "incomplete",
        "revision_policy": "standard_cove_qa_only",
        "selected_responses": len(units),
        "successful_revisions": len(successful),
        "status_counts": dict(status_counts),
        "verification_questions_supplied": sum(
            len(unit["verification_results"]) for unit in units
        ),
        "unchanged_responses": unchanged,
        "format_only_recovery": {
            "recovered_responses": sum(recovery_methods.values()),
            "method_counts": dict(sorted(recovery_methods.items())),
            "model_recalled": False,
            "prose_changed": False,
        },
        "length_summary": {
            "initial_total_words": sum(initial_words),
            "revised_total_words": sum(revised_words),
            "mean_initial_words": (
                round(statistics.mean(initial_words), 4)
                if initial_words
                else None
            ),
            "mean_revised_words": (
                round(statistics.mean(revised_words), 4)
                if revised_words
                else None
            ),
            "mean_word_delta": (
                round(statistics.mean(word_deltas), 4)
                if word_deltas
                else None
            ),
            "median_word_delta": (
                statistics.median(word_deltas) if word_deltas else None
            ),
            "mean_char_delta": (
                round(statistics.mean(char_deltas), 4)
                if char_deltas
                else None
            ),
        },
        "latency_seconds": {
            "total": round(sum(latencies), 4) if latencies else None,
            "mean": (
                round(statistics.mean(latencies), 4) if latencies else None
            ),
            "median": (
                round(statistics.median(latencies), 4)
                if latencies
                else None
            ),
        },
        "leakage_boundary": {
            "model_input_fields": [
                "original_question",
                "initial_response",
                "verification_questions",
                "verification_answers",
            ],
            "gold_claims_or_labels_exposed": False,
            "gold_or_retrieved_evidence_exposed": False,
            "b2_or_b4_evaluation_exposed": False,
            "b3_self_reported_status_exposed": False,
        },
        "factuality_evaluation_status": "pending_B6_claim_extraction_and_alignment",
        "responses": per_response,
        "interpretation_notes": [
            "B5 is standard CoVe revision, not verifier-guided revision.",
            "B4 is a completion gate only and no B4 labels enter the revision prompt.",
            "Length change and unchanged text are technical descriptors, not factuality metrics.",
            "B6 must evaluate correction, retention, deletion, and newly added claims.",
        ],
    }


def build_revision_markdown(summary: dict[str, Any]) -> str:
    length = summary["length_summary"]
    latency = summary["latency_seconds"]
    lines = [
        "# Experiment B — B5 Standard CoVe Revision",
        "",
        f"- Split: `{summary['split']}`",
        f"- Completion: **{summary['completion_status']}**",
        f"- Revisions: {summary['successful_revisions']}/{summary['selected_responses']}",
        f"- Verification questions supplied: {summary['verification_questions_supplied']}",
        f"- Revision policy: `{summary['revision_policy']}`",
        "- Gold claims/labels/evidence exposed: **no**",
        "- B2/B4 evaluation labels exposed: **no**",
        "",
        "## Technical change summary",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Initial total words | {length['initial_total_words']} |",
        f"| Revised total words | {length['revised_total_words']} |",
        f"| Mean initial words | {length['mean_initial_words']} |",
        f"| Mean revised words | {length['mean_revised_words']} |",
        f"| Mean word delta | {length['mean_word_delta']} |",
        f"| Median word delta | {length['median_word_delta']} |",
        f"| Unchanged responses | {summary['unchanged_responses']} |",
        f"| Format-only recovered responses | {summary['format_only_recovery']['recovered_responses']} |",
        f"| Mean latency, seconds | {latency['mean']} |",
        "",
        "## Per-response change size",
        "",
        "| response_id | Q&A pairs | initial words | revised words | word delta | unchanged |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["responses"]:
        lines.append(
            f"| `{row['response_id']}` | {row['verification_questions']} | "
            f"{row['initial_words']} | {row['revised_words']} | "
            f"{row['word_delta']} | {str(row['text_unchanged']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This report measures technical completion and response-size change "
            "only. A shorter or substantially changed response is not "
            "necessarily more factual. B6 must extract revised claims and "
            "measure correction, factual retention, deletion, and additions.",
            "",
        ]
    )
    return "\n".join(lines)


def write_revision_reports(
    paths: Any,
    split: str,
    summary: dict[str, Any],
) -> None:
    atomic_write_json(paths.revision_summary_json(split), summary)
    atomic_write_text(
        paths.revision_summary_markdown(split),
        build_revision_markdown(summary),
    )


def _load_revision_dependencies(
    paths: Any,
    config: dict[str, Any],
    split: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    manifest = load_jsonl(paths.response_manifest)
    validate_response_manifest(manifest, config)
    selected = [row for row in manifest if row["split"] == split]
    selected_questions = load_validated_b1_results(paths, config, split)
    b2_gate = load_validated_b2_gate(
        paths,
        config,
        split,
        selected_questions,
    )
    b3_results, _ = load_validated_b3_results(
        paths,
        config,
        split,
        selected_questions,
        b2_gate,
    )
    b4_gate_sha256 = load_b4_completion_gate(paths, split)
    units = revision_units(selected, b3_results)
    if len(units) != len(selected):
        raise ValueError("B5 response-unit construction lost or duplicated rows")
    return selected, b3_results, units, b4_gate_sha256


def run_revision(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    _, _, units, b4_gate_sha256 = _load_revision_dependencies(
        paths,
        config,
        args.split,
    )
    prompt_path, template = load_revision_prompt(config)
    if args.dry_run:
        run_config = build_revision_run_config(
            config,
            split=args.split,
            prompt_path=prompt_path,
            response_manifest_path=paths.response_manifest,
            b3_results_path=paths.verification_answer_results(args.split),
            b4_gate_sha256=b4_gate_sha256,
            model_digest="DRY_RUN_NOT_CHECKED",
        )
        print("EXPERIMENT B B5 STANDARD COVE REVISION DRY RUN")
        print(
            f"Split: {args.split}; response calls: {len(units)}; "
            f"question-answer pairs: "
            f"{sum(len(unit['verification_results']) for unit in units)}"
        )
        print(f"Run fingerprint: {run_config['run_fingerprint']}")
        print(
            "Model input fields: original question, initial response, and B3 "
            "question-answer pairs only. B2/B4 labels and gold/evidence are "
            "withheld."
        )
        print(f"\n--- Preview: {units[0]['response_id']} ---")
        print(build_revision_prompt(template, units[0]))
        print("\nNo Ollama calls were made and no files were written.")
        return 0

    client = Client(
        host=args.ollama_host,
        timeout=float(config["response_revision"]["timeout_seconds"]),
    )
    model_digest = preflight_revision_ollama(client, config)
    run_config = build_revision_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        response_manifest_path=paths.response_manifest,
        b3_results_path=paths.verification_answer_results(args.split),
        b4_gate_sha256=b4_gate_sha256,
        model_digest=model_digest,
    )
    output_path = paths.revision_results(args.split)
    existing = load_jsonl(output_path) if output_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Output already exists; rerun with --resume: {output_path}"
        )
    validate_existing_revisions(existing, units, run_config)
    result_by_id = {row["response_id"]: row for row in existing}
    pending = [
        unit
        for unit in units
        if result_by_id.get(unit["response_id"], {}).get("status") != "ok"
    ]
    print(
        f"Experiment B B5: split={args.split}, total={len(units)}, "
        f"retained_ok={len(units) - len(pending)}, pending={len(pending)}",
        flush=True,
    )
    consecutive_request_errors = 0
    max_consecutive = int(
        config["response_revision"]["max_consecutive_request_errors"]
    )
    order = {unit["response_id"]: index for index, unit in enumerate(units)}
    for unit in pending:
        overall_index = order[unit["response_id"]] + 1
        print(
            f"[{overall_index}/{len(units)}] {unit['response_id']} revising "
            f"with {len(unit['verification_results'])} Q&A pairs ...",
            flush=True,
        )
        result = process_revision_unit(
            unit,
            template,
            config,
            run_config,
            client,
        )
        result_by_id[unit["response_id"]] = result
        ordered = [
            result_by_id[item["response_id"]]
            for item in units
            if item["response_id"] in result_by_id
        ]
        atomic_write_jsonl(output_path, ordered)
        if result["status"] == "ok":
            consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(units)}] success, "
                f"{_word_count(result['revised_response'])} words, "
                f"{result['latency_seconds']:.1f}s",
                flush=True,
            )
        else:
            if result["status"] == "request_error":
                consecutive_request_errors += 1
            else:
                consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(units)}] {result['status']}: "
                f"{result['error']}",
                flush=True,
            )
            if consecutive_request_errors >= max_consecutive:
                print(
                    "Stopping after the configured consecutive request-error "
                    "limit. The same command will resume safely.",
                    file=sys.stderr,
                    flush=True,
                )
                break
    all_results = [
        result_by_id[unit["response_id"]]
        for unit in units
        if unit["response_id"] in result_by_id
    ]
    validate_existing_revisions(all_results, units, run_config)
    summary = revision_summary(units, all_results, args.split)
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    write_revision_reports(paths, args.split, summary)
    print(f"Results: {output_path}", flush=True)
    print(
        f"Report: {paths.revision_summary_markdown(args.split)}",
        flush=True,
    )
    return 0 if summary["completion_status"] == "complete" else 2


def analyze_revision(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    _, _, units, b4_gate_sha256 = _load_revision_dependencies(
        paths,
        config,
        args.split,
    )
    output_path = paths.revision_results(args.split)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B5 results do not have one valid model digest")
    prompt_path, _ = load_revision_prompt(config)
    run_config = build_revision_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        response_manifest_path=paths.response_manifest,
        b3_results_path=paths.verification_answer_results(args.split),
        b4_gate_sha256=b4_gate_sha256,
        model_digest=next(iter(model_digests)),
    )
    validate_existing_revisions(results, units, run_config)
    summary = revision_summary(units, results, args.split)
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    write_revision_reports(paths, args.split, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["completion_status"] == "complete" else 2


def recover_revision_format(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    _, _, units, b4_gate_sha256 = _load_revision_dependencies(
        paths,
        config,
        args.split,
    )
    output_path = paths.revision_results(args.split)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B5 results do not have one valid model digest")
    prompt_path, _ = load_revision_prompt(config)
    run_config = build_revision_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        response_manifest_path=paths.response_manifest,
        b3_results_path=paths.verification_answer_results(args.split),
        b4_gate_sha256=b4_gate_sha256,
        model_digest=next(iter(model_digests)),
    )
    validate_existing_revisions(results, units, run_config)
    recovered = 0
    method_counts: Counter[str] = Counter()
    unrecoverable: list[str] = []
    for row in results:
        if row.get("status") != "parse_error":
            continue
        try:
            revised, method = recover_revision_output_format(
                row.get("raw_model_output") or ""
            )
        except Exception:
            unrecoverable.append(row["response_id"])
            continue
        previous_error = row.get("error")
        row.update(
            {
                "status": "ok",
                "revised_response": revised,
                "revised_response_sha256": sha256_text(revised),
                "format_recovery": {
                    "applied": True,
                    "method": method,
                    "strict_parse_error": previous_error,
                    "model_recalled": False,
                    "prose_changed": False,
                    "recovered_at": utc_now(),
                },
                "error": None,
            }
        )
        recovered += 1
        method_counts[method] += 1
    atomic_write_jsonl(output_path, results)
    validate_existing_revisions(results, units, run_config)
    summary = revision_summary(units, results, args.split)
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    write_revision_reports(paths, args.split, summary)
    report = {
        "stage": "B5_format_only_recovery",
        "split": args.split,
        "recovered_rows": recovered,
        "method_counts": dict(sorted(method_counts.items())),
        "unrecoverable_response_ids": unrecoverable,
        "model_recalled": False,
        "prose_changed": False,
        "completion_status": summary["completion_status"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if summary["completion_status"] == "complete" else 2


def make_revised_claim_output_schema(
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = config["revised_claim_extraction"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "minItems": int(settings["minimum_claims"]),
                "maxItems": int(settings["maximum_claims"]),
                "items": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 600,
                },
            }
        },
    }


def load_revised_claim_prompt(
    config: dict[str, Any],
) -> tuple[Path, str]:
    path = PROJECT_ROOT / config["revised_claim_extraction"]["prompt_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"Revised-claim prompt is empty: {path}")
    found = {
        placeholder
        for placeholder in REVISED_CLAIM_PLACEHOLDERS
        if placeholder in template
    }
    if found != REVISED_CLAIM_PLACEHOLDERS:
        raise ValueError(
            "Revised-claim prompt placeholders are incomplete: "
            f"{sorted(found)}"
        )
    for placeholder in REVISED_CLAIM_PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ValueError(
                f"Revised-claim prompt must contain {placeholder} exactly once"
            )
    return path, template


def build_revised_claim_prompt(
    template: str,
    unit: dict[str, Any],
) -> str:
    return template.replace(
        "{original_question_json}",
        json.dumps(unit["original_question"], ensure_ascii=False),
    ).replace(
        "{revised_response_json}",
        json.dumps(unit["revised_response"], ensure_ascii=False),
    )


def parse_revised_claim_output(
    raw_output: str,
    config: dict[str, Any],
) -> list[str]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("Revised-claim model output is empty")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Revised-claim output is not strict JSON: {error}"
        ) from error
    if not isinstance(parsed, dict) or set(parsed) != {"claims"}:
        raise ValueError(
            "Revised-claim output must contain only the claims field"
        )
    claims = parsed["claims"]
    if not isinstance(claims, list):
        raise TypeError("claims must be a JSON array")
    minimum = int(config["revised_claim_extraction"]["minimum_claims"])
    maximum = int(config["revised_claim_extraction"]["maximum_claims"])
    if not minimum <= len(claims) <= maximum:
        raise ValueError(
            f"Expected {minimum}-{maximum} revised claims, got {len(claims)}"
        )
    normalized: list[str] = []
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, str):
            raise TypeError(f"Revised claim {index} is not a string")
        claim = " ".join(claim.split())
        if not 3 <= len(claim) <= 600:
            raise ValueError(
                f"Revised claim {index} must contain 3-600 characters"
            )
        normalized.append(claim)
    duplicate_keys = [
        " ".join(claim.casefold().split()) for claim in normalized
    ]
    if len(duplicate_keys) != len(set(duplicate_keys)):
        raise ValueError("Revised claims contain normalized exact duplicates")
    return normalized


def preflight_revised_claim_ollama(
    client: Any,
    config: dict[str, Any],
) -> str:
    settings = config["revised_claim_extraction"]
    model = settings["model"]
    try:
        response = client.list()
    except Exception as error:
        raise ConnectionError(
            "Ollama preflight failed before any B6a output was changed"
        ) from error
    models = response_value(response, "models")
    if not isinstance(models, (list, tuple)):
        raise ValueError("Ollama preflight returned no model list")
    available: dict[str, str | None] = {}
    for item in models:
        name = response_value(item, "model") or response_value(item, "name")
        if isinstance(name, str) and name:
            digest = response_value(item, "digest")
            available[name] = digest if isinstance(digest, str) else None
    if model not in available:
        raise ValueError(
            f"Configured B6a model {model!r} is not installed; "
            f"available={sorted(available)}"
        )
    digest = available[model]
    if not digest:
        raise ValueError(f"Ollama provided no digest for {model!r}")
    expected = settings["expected_model_digest"]
    if digest != expected:
        raise ValueError(
            "Installed model digest differs from the frozen B6a "
            f"configuration: expected={expected}, actual={digest}"
        )
    return digest


def load_validated_b5_results(
    paths: Any,
    config: dict[str, Any],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if getattr(paths, "branch", "a") != "a":
        output_path = paths.revision_results(split)
        if not output_path.exists():
            raise FileNotFoundError(
                f"Branch {paths.branch.upper()} revision is missing for "
                f"{split}: {output_path}"
            )
        results = load_jsonl(output_path)
        manifest = load_jsonl(paths.response_manifest)
        validate_response_manifest(manifest, config)
        expected = {
            row["response_id"]: row
            for row in manifest
            if row["split"] == split
        }
        seen: set[str] = set()
        fingerprints: set[str] = set()
        for row in results:
            response_id = row.get("response_id")
            if response_id not in expected:
                raise ValueError(
                    f"Branch {paths.branch.upper()} has unexpected response: "
                    f"{response_id}"
                )
            if response_id in seen:
                raise ValueError(
                    f"Duplicate Branch {paths.branch.upper()} response: "
                    f"{response_id}"
                )
            seen.add(response_id)
            source = expected[response_id]
            if row.get("branch_id") != paths.branch:
                raise ValueError(
                    f"Branch identity mismatch for {response_id}"
                )
            if row.get("split") != split:
                raise ValueError(f"Branch split mismatch for {response_id}")
            if row.get("status") != "ok":
                raise ValueError(
                    f"Branch {paths.branch.upper()} revision is incomplete: "
                    f"{response_id}"
                )
            if row.get("source_record_index") != source["source_record_index"]:
                raise ValueError(
                    f"Branch source-record mismatch for {response_id}"
                )
            if row.get("original_question") != source["original_question"]:
                raise ValueError(
                    f"Branch original-question mismatch for {response_id}"
                )
            revised = row.get("revised_response")
            if (
                not isinstance(revised, str)
                or not revised.strip()
                or row.get("revised_response_sha256") != sha256_text(revised)
            ):
                raise ValueError(
                    f"Branch revised-response hash mismatch for {response_id}"
                )
            fingerprint = row.get("run_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError(
                    f"Branch run fingerprint missing for {response_id}"
                )
            fingerprints.add(fingerprint)
        if seen != set(expected):
            raise ValueError(
                f"Branch {paths.branch.upper()} {split} revision is "
                f"incomplete: expected {len(expected)}, found {len(seen)}"
            )
        if len(fingerprints) != 1:
            raise ValueError(
                f"Branch {paths.branch.upper()} has mixed run fingerprints"
            )
        return results, {
            "branch": paths.branch,
            "run_fingerprint": next(iter(fingerprints)),
            "validation": "isolated_branch_revision_v1",
        }

    _, _, units, b4_gate_sha256 = _load_revision_dependencies(
        paths,
        config,
        split,
    )
    output_path = paths.revision_results(split)
    if not output_path.exists():
        raise FileNotFoundError(f"B5 output is missing for {split}: {output_path}")
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B5 results do not have one valid model digest")
    prompt_path, _ = load_revision_prompt(config)
    run_config = build_revision_run_config(
        config,
        split=split,
        prompt_path=prompt_path,
        response_manifest_path=paths.response_manifest,
        b3_results_path=paths.verification_answer_results(split),
        b4_gate_sha256=b4_gate_sha256,
        model_digest=next(iter(model_digests)),
    )
    validate_existing_revisions(results, units, run_config)
    expected_ids = {unit["response_id"] for unit in units}
    ok_ids = {
        row["response_id"] for row in results if row.get("status") == "ok"
    }
    if ok_ids != expected_ids:
        raise ValueError(f"B5 {split} is incomplete; finish revision before B6a")
    return results, run_config


def revised_claim_units(
    b5_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(
        b5_results,
        key=lambda item: (
            int(item["source_record_index"]),
            item["response_id"],
        ),
    ):
        response_id = row["response_id"]
        if response_id in seen:
            raise ValueError(f"Duplicate B5 response_id: {response_id}")
        seen.add(response_id)
        units.append(
            {
                "response_id": response_id,
                "source_record_index": row["source_record_index"],
                "split": row["split"],
                "original_question": row["original_question"],
                "revised_response": row["revised_response"],
                "revised_response_sha256": row["revised_response_sha256"],
                "b5_run_fingerprint": row["run_fingerprint"],
            }
        )
    return units


def build_revised_claim_run_config(
    config: dict[str, Any],
    *,
    split: str,
    prompt_path: Path,
    b5_results_path: Path,
    model_digest: str,
) -> dict[str, Any]:
    settings = config["revised_claim_extraction"]
    schema = make_revised_claim_output_schema(config)
    payload = {
        "experiment": config["experiment"],
        "stage": "B6a_revised_claim_extraction",
        "split": split,
        "extraction_unit": "one_model_call_per_revised_response",
        "model": settings["model"],
        "model_digest": model_digest,
        "temperature": float(settings["temperature"]),
        "seed": int(settings["seed"]),
        "num_predict": int(settings["num_predict"]),
        "think": bool(settings["think"]),
        "timeout_seconds": float(settings["timeout_seconds"]),
        "max_retries": int(settings["max_retries"]),
        "max_consecutive_request_errors": int(
            settings["max_consecutive_request_errors"]
        ),
        "minimum_claims": int(settings["minimum_claims"]),
        "maximum_claims": int(settings["maximum_claims"]),
        "prompt_version": settings["prompt_version"],
        "prompt_sha256": sha256_file(prompt_path),
        "result_schema_version": settings["result_schema_version"],
        "output_schema_version": settings["output_schema_version"],
        "output_schema_sha256": canonical_json_hash(schema),
        "b5_results_sha256": sha256_file(b5_results_path),
        "model_input_fields": config["leakage_policy"][
            "revised_claim_extraction_model_input_fields"
        ],
        "withheld_fields": config["leakage_policy"][
            "revised_claim_extraction_withheld_fields"
        ],
    }
    return {**payload, "run_fingerprint": canonical_json_hash(payload)}


def create_revised_claim_result_base(
    unit: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "result_schema_version": run_config["result_schema_version"],
        "response_id": unit["response_id"],
        "source_record_index": unit["source_record_index"],
        "split": unit["split"],
        "original_question": unit["original_question"],
        "original_question_sha256": sha256_text(unit["original_question"]),
        "revised_response": unit["revised_response"],
        "revised_response_sha256": unit["revised_response_sha256"],
        "stage": run_config["stage"],
        "extraction_unit": run_config["extraction_unit"],
        "model_input_fields": run_config["model_input_fields"],
        "withheld_fields": run_config["withheld_fields"],
        "b5_run_fingerprint": unit["b5_run_fingerprint"],
        "b5_results_sha256": run_config["b5_results_sha256"],
        "model": run_config["model"],
        "model_digest": run_config["model_digest"],
        "temperature": run_config["temperature"],
        "seed": run_config["seed"],
        "num_predict": run_config["num_predict"],
        "think": run_config["think"],
        "timeout_seconds": run_config["timeout_seconds"],
        "max_retries": run_config["max_retries"],
        "max_consecutive_request_errors": run_config[
            "max_consecutive_request_errors"
        ],
        "minimum_claims": run_config["minimum_claims"],
        "maximum_claims": run_config["maximum_claims"],
        "prompt_version": run_config["prompt_version"],
        "prompt_sha256": run_config["prompt_sha256"],
        "output_schema_version": run_config["output_schema_version"],
        "output_schema_sha256": run_config["output_schema_sha256"],
        "run_fingerprint": run_config["run_fingerprint"],
    }


def process_revised_claim_unit(
    unit: dict[str, Any],
    template: str,
    config: dict[str, Any],
    run_config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    result = create_revised_claim_result_base(unit, run_config)
    prompt = build_revised_claim_prompt(template, unit)
    raw_output: str | None = None
    metadata: dict[str, Any] = {}
    request_error: Exception | None = None
    attempts = 0
    started = time.perf_counter()
    for attempt in range(run_config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw_output, metadata = call_ollama(
                client,
                run_config,
                make_revised_claim_output_schema(config),
                prompt,
            )
            request_error = None
            break
        except Exception as error:
            request_error = error
            if attempt < run_config["max_retries"]:
                time.sleep(1.0)
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
                "revised_claims": None,
                "error": f"{type(request_error).__name__}: {request_error}",
            }
        )
        return result
    try:
        claim_texts = parse_revised_claim_output(raw_output or "", config)
    except Exception as error:
        result.update(
            {
                "status": "parse_error",
                "revised_claims": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return result
    revised_claims = [
        {
            "claim_id": f"{unit['response_id']}_revised_c{index:03d}",
            "claim_index": index,
            "claim": claim,
            "claim_sha256": sha256_text(claim),
        }
        for index, claim in enumerate(claim_texts, start=1)
    ]
    result.update(
        {
            "status": "ok",
            "revised_claims": revised_claims,
            "claim_count": len(revised_claims),
            "error": None,
        }
    )
    return result


def validate_existing_revised_claims(
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    run_config: dict[str, Any],
    config: dict[str, Any],
) -> None:
    unit_by_id = {unit["response_id"]: unit for unit in units}
    seen: set[str] = set()
    for row in rows:
        response_id = row.get("response_id")
        if response_id not in unit_by_id:
            raise ValueError(
                f"B6a output contains unexpected response_id: {response_id}"
            )
        if response_id in seen:
            raise ValueError(
                f"Duplicate response_id in B6a output: {response_id}"
            )
        seen.add(response_id)
        expected_base = create_revised_claim_result_base(
            unit_by_id[response_id],
            run_config,
        )
        for key, expected in expected_base.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"Existing B6a output is incompatible for "
                    f"{response_id}: {key}"
                )
        status = row.get("status")
        if status not in {"ok", "request_error", "parse_error"}:
            raise ValueError(f"Invalid B6a status for {response_id}: {status}")
        if status == "ok":
            claims = row.get("revised_claims")
            if not isinstance(claims, list):
                raise TypeError(
                    f"Successful B6a row has no claims for {response_id}"
                )
            parsed = parse_revised_claim_output(
                json.dumps(
                    {
                        "claims": [
                            item.get("claim")
                            for item in claims
                            if isinstance(item, dict)
                        ]
                    },
                    ensure_ascii=False,
                ),
                config,
            )
            if len(parsed) != len(claims):
                raise ValueError(f"Malformed B6a claims for {response_id}")
            for index, (item, claim) in enumerate(
                zip(claims, parsed),
                start=1,
            ):
                expected_id = f"{response_id}_revised_c{index:03d}"
                if item.get("claim_id") != expected_id:
                    raise ValueError(
                        f"Unstable B6a claim ID for {response_id}: "
                        f"{item.get('claim_id')}"
                    )
                if item.get("claim_index") != index:
                    raise ValueError(
                        f"Invalid B6a claim index for {response_id}"
                    )
                if item.get("claim_sha256") != sha256_text(claim):
                    raise ValueError(
                        f"B6a claim hash mismatch for {expected_id}"
                    )
            if row.get("claim_count") != len(claims):
                raise ValueError(f"B6a claim_count mismatch for {response_id}")
            if row.get("error") is not None:
                raise ValueError(
                    f"Successful B6a row has an error for {response_id}"
                )
        else:
            if row.get("revised_claims") is not None:
                raise ValueError(
                    f"Technical B6a failure has claims for {response_id}"
                )


def revised_claim_summary(
    units: list[dict[str, Any]],
    results: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    result_by_id = {row["response_id"]: row for row in results}
    status_counts = Counter(
        result_by_id.get(unit["response_id"], {}).get("status", "missing")
        for unit in units
    )
    successful = [
        result_by_id[unit["response_id"]]
        for unit in units
        if result_by_id.get(unit["response_id"], {}).get("status") == "ok"
    ]
    counts = [int(row["claim_count"]) for row in successful]
    zero_claim_ids = [
        row["response_id"] for row in successful if row["claim_count"] == 0
    ]
    latencies = [
        float(row["latency_seconds"])
        for row in successful
        if isinstance(row.get("latency_seconds"), (int, float))
    ]
    per_response = [
        {
            "response_id": row["response_id"],
            "claim_count": row["claim_count"],
            "latency_seconds": row.get("latency_seconds"),
        }
        for row in successful
    ]
    recovered_rows = [
        row
        for row in successful
        if isinstance(row.get("format_recovery"), dict)
        and row["format_recovery"].get("applied") is True
    ]
    complete = len(successful) == len(units)
    return {
        "schema_version": "cove_revised_claim_extraction_summary_v1",
        "experiment": "Experiment B: CoVe mechanism evaluation",
        "stage": "B6a_revised_claim_extraction",
        "split": split,
        "completion_status": "complete" if complete else "incomplete",
        "selected_responses": len(units),
        "successful_responses": len(successful),
        "status_counts": dict(status_counts),
        "total_revised_claims": sum(counts),
        "claim_count": {
            "minimum": min(counts) if counts else None,
            "maximum": max(counts) if counts else None,
            "mean": round(statistics.mean(counts), 4) if counts else None,
            "median": statistics.median(counts) if counts else None,
        },
        "zero_claim_response_ids": zero_claim_ids,
        "conservative_format_recovery": {
            "recovered_responses": len(recovered_rows),
            "removed_normalized_exact_duplicates": sum(
                int(row["format_recovery"].get("removed_duplicate_count", 0))
                for row in recovered_rows
            ),
            "model_recalled": False,
            "audit_required_responses": sum(
                row["format_recovery"].get("audit_required") is True
                for row in recovered_rows
            ),
        },
        "latency_seconds": {
            "total": round(sum(latencies), 4) if latencies else None,
            "mean": (
                round(statistics.mean(latencies), 4) if latencies else None
            ),
            "median": (
                round(statistics.median(latencies), 4)
                if latencies
                else None
            ),
        },
        "leakage_boundary": {
            "model_input_fields": [
                "original_question",
                "revised_response",
            ],
            "initial_response_exposed": False,
            "gold_claims_or_labels_exposed": False,
            "gold_or_retrieved_evidence_exposed": False,
            "b2_b3_b4_outputs_exposed": False,
        },
        "factuality_evaluation_status": "pending_B6b_gold_revised_alignment",
        "responses": per_response,
        "interpretation_notes": [
            "B6a decomposes revised responses and does not judge factuality.",
            "Stable revised claim IDs are assigned by code in response order.",
            "B6b must align these claims to canonical initial gold claims and identify additions.",
        ],
    }


def build_revised_claim_markdown(summary: dict[str, Any]) -> str:
    count = summary["claim_count"]
    latency = summary["latency_seconds"]
    lines = [
        "# Experiment B — B6a Revised Claim Extraction",
        "",
        f"- Split: `{summary['split']}`",
        f"- Completion: **{summary['completion_status']}**",
        f"- Responses: {summary['successful_responses']}/{summary['selected_responses']}",
        f"- Revised claims: {summary['total_revised_claims']}",
        "- Initial response exposed: **no**",
        "- Gold claims/labels/evidence exposed: **no**",
        "",
        "## Extraction summary",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Minimum claims per response | {count['minimum']} |",
        f"| Median claims per response | {count['median']} |",
        f"| Mean claims per response | {count['mean']} |",
        f"| Maximum claims per response | {count['maximum']} |",
        f"| Zero-claim responses | {len(summary['zero_claim_response_ids'])} |",
        f"| Mean latency, seconds | {latency['mean']} |",
        "",
        "## Per-response claim counts",
        "",
        "| response_id | revised claims | latency, seconds |",
        "|---|---:|---:|",
    ]
    for row in summary["responses"]:
        lines.append(
            f"| `{row['response_id']}` | {row['claim_count']} | "
            f"{row['latency_seconds']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Claim count is not factuality. B6a does not decide whether a "
            "claim is retained, corrected, deleted, added, true, or false. "
            "Those decisions begin only after B6b gold-to-revised alignment.",
            "",
        ]
    )
    return "\n".join(lines)


def write_revised_claim_reports(
    paths: Any,
    split: str,
    summary: dict[str, Any],
) -> None:
    atomic_write_json(
        paths.revised_claim_extraction_summary_json(split),
        summary,
    )
    atomic_write_text(
        paths.revised_claim_extraction_summary_markdown(split),
        build_revised_claim_markdown(summary),
    )


def run_revised_claim_extraction(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    b5_results, _ = load_validated_b5_results(paths, config, args.split)
    units = revised_claim_units(b5_results)
    prompt_path, template = load_revised_claim_prompt(config)
    if args.dry_run:
        run_config = build_revised_claim_run_config(
            config,
            split=args.split,
            prompt_path=prompt_path,
            b5_results_path=paths.revision_results(args.split),
            model_digest="DRY_RUN_NOT_CHECKED",
        )
        print("EXPERIMENT B B6A REVISED-CLAIM EXTRACTION DRY RUN")
        print(f"Split: {args.split}; response calls: {len(units)}")
        print(f"Run fingerprint: {run_config['run_fingerprint']}")
        print(
            "Model input fields: original_question and revised_response only. "
            "Initial response, gold/evidence, and B2-B4 outputs are withheld."
        )
        print(f"\n--- Preview: {units[0]['response_id']} ---")
        print(build_revised_claim_prompt(template, units[0]))
        print("\nNo Ollama calls were made and no files were written.")
        return 0

    client = Client(
        host=args.ollama_host,
        timeout=float(config["revised_claim_extraction"]["timeout_seconds"]),
    )
    model_digest = preflight_revised_claim_ollama(client, config)
    run_config = build_revised_claim_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        b5_results_path=paths.revision_results(args.split),
        model_digest=model_digest,
    )
    output_path = paths.revised_claim_extraction_results(args.split)
    existing = load_jsonl(output_path) if output_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Output already exists; rerun with --resume: {output_path}"
        )
    validate_existing_revised_claims(
        existing,
        units,
        run_config,
        config,
    )
    result_by_id = {row["response_id"]: row for row in existing}
    pending = [
        unit
        for unit in units
        if result_by_id.get(unit["response_id"], {}).get("status") != "ok"
    ]
    print(
        f"Experiment B B6a: split={args.split}, total={len(units)}, "
        f"retained_ok={len(units) - len(pending)}, pending={len(pending)}",
        flush=True,
    )
    consecutive_request_errors = 0
    max_consecutive = int(
        config["revised_claim_extraction"][
            "max_consecutive_request_errors"
        ]
    )
    order = {unit["response_id"]: index for index, unit in enumerate(units)}
    for unit in pending:
        overall_index = order[unit["response_id"]] + 1
        print(
            f"[{overall_index}/{len(units)}] {unit['response_id']} "
            "extracting revised claims ...",
            flush=True,
        )
        result = process_revised_claim_unit(
            unit,
            template,
            config,
            run_config,
            client,
        )
        result_by_id[unit["response_id"]] = result
        ordered = [
            result_by_id[item["response_id"]]
            for item in units
            if item["response_id"] in result_by_id
        ]
        atomic_write_jsonl(output_path, ordered)
        if result["status"] == "ok":
            consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(units)}] success, "
                f"{result['claim_count']} claims, "
                f"{result['latency_seconds']:.1f}s",
                flush=True,
            )
        else:
            if result["status"] == "request_error":
                consecutive_request_errors += 1
            else:
                consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(units)}] {result['status']}: "
                f"{result['error']}",
                flush=True,
            )
            if consecutive_request_errors >= max_consecutive:
                print(
                    "Stopping after the configured consecutive request-error "
                    "limit. The same command will resume safely.",
                    file=sys.stderr,
                    flush=True,
                )
                break
    all_results = [
        result_by_id[unit["response_id"]]
        for unit in units
        if unit["response_id"] in result_by_id
    ]
    validate_existing_revised_claims(
        all_results,
        units,
        run_config,
        config,
    )
    summary = revised_claim_summary(units, all_results, args.split)
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    write_revised_claim_reports(paths, args.split, summary)
    print(f"Results: {output_path}", flush=True)
    print(
        "Report: "
        f"{paths.revised_claim_extraction_summary_markdown(args.split)}",
        flush=True,
    )
    return 0 if summary["completion_status"] == "complete" else 2


def analyze_revised_claim_extraction(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    b5_results, _ = load_validated_b5_results(paths, config, args.split)
    units = revised_claim_units(b5_results)
    output_path = paths.revised_claim_extraction_results(args.split)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B6a results do not have one valid model digest")
    prompt_path, _ = load_revised_claim_prompt(config)
    run_config = build_revised_claim_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        b5_results_path=paths.revision_results(args.split),
        model_digest=next(iter(model_digests)),
    )
    validate_existing_revised_claims(
        results,
        units,
        run_config,
        config,
    )
    summary = revised_claim_summary(units, results, args.split)
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    write_revised_claim_reports(paths, args.split, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["completion_status"] == "complete" else 2


def recover_revised_claim_extraction_format(
    args: argparse.Namespace,
) -> int:
    """Remove only normalized exact duplicate B6a claims without model calls."""
    paths = paths_for_args(args)
    config = load_config(paths)
    b5_results, _ = load_validated_b5_results(paths, config, args.split)
    units = revised_claim_units(b5_results)
    output_path = paths.revised_claim_extraction_results(args.split)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B6a results do not have one valid model digest")
    prompt_path, _ = load_revised_claim_prompt(config)
    run_config = build_revised_claim_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        b5_results_path=paths.revision_results(args.split),
        model_digest=next(iter(model_digests)),
    )
    validate_existing_revised_claims(
        results,
        units,
        run_config,
        config,
    )

    recovered_count = 0
    unrecovered: list[dict[str, str]] = []
    for row in results:
        if row.get("status") != "parse_error":
            continue
        response_id = row["response_id"]
        original_error = row.get("error")
        if original_error != (
            "ValueError: Revised claims contain normalized exact duplicates"
        ):
            unrecovered.append(
                {
                    "response_id": response_id,
                    "error": str(original_error),
                }
            )
            continue
        try:
            parsed = json.loads(row.get("raw_model_output", ""))
            if not isinstance(parsed, dict) or set(parsed) != {"claims"}:
                raise ValueError(
                    "B6a duplicate recovery requires only the claims field"
                )
            raw_claims = parsed["claims"]
            if not isinstance(raw_claims, list):
                raise TypeError("claims must be a JSON array")
            deduplicated: list[str] = []
            seen_keys: set[str] = set()
            removed: list[dict[str, Any]] = []
            for original_index, claim in enumerate(raw_claims, start=1):
                if not isinstance(claim, str):
                    raise TypeError(
                        f"Revised claim {original_index} is not a string"
                    )
                normalized_claim = " ".join(claim.split())
                duplicate_key = " ".join(
                    normalized_claim.casefold().split()
                )
                if duplicate_key in seen_keys:
                    removed.append(
                        {
                            "original_index": original_index,
                            "claim_sha256": sha256_text(normalized_claim),
                        }
                    )
                    continue
                seen_keys.add(duplicate_key)
                deduplicated.append(normalized_claim)
            if not removed:
                raise ValueError(
                    "B6a duplicate recovery found no exact duplicate"
                )
            claim_texts = parse_revised_claim_output(
                json.dumps(
                    {"claims": deduplicated},
                    ensure_ascii=False,
                ),
                config,
            )
        except Exception as error:
            unrecovered.append(
                {
                    "response_id": response_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        revised_claims = [
            {
                "claim_id": f"{response_id}_revised_c{index:03d}",
                "claim_index": index,
                "claim": claim,
                "claim_sha256": sha256_text(claim),
            }
            for index, claim in enumerate(claim_texts, start=1)
        ]
        row.update(
            {
                "status": "ok",
                "revised_claims": revised_claims,
                "claim_count": len(revised_claims),
                "error": None,
                "format_recovery": {
                    "applied": True,
                    "method": (
                        "preserve_first_normalized_exact_duplicate_v1"
                    ),
                    "original_status": "parse_error",
                    "original_error": original_error,
                    "raw_model_output_preserved": True,
                    "model_recalled": False,
                    "removed_duplicate_count": len(removed),
                    "removed_duplicates": removed,
                    "audit_required": True,
                    "recovered_at": utc_now(),
                },
            }
        )
        recovered_count += 1
        print(
            f"[recovered] {response_id}: removed {len(removed)} "
            "normalized exact duplicate(s), audit required",
            flush=True,
        )

    atomic_write_jsonl(output_path, results)
    validate_existing_revised_claims(
        results,
        units,
        run_config,
        config,
    )
    summary = revised_claim_summary(units, results, args.split)
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    summary["conservative_format_recovery"]["unrecovered"] = unrecovered
    write_revised_claim_reports(paths, args.split, summary)
    print(
        "B6a format recovery: "
        f"recovered={recovered_count}, unrecovered={len(unrecovered)}, "
        f"completion={summary['completion_status']}",
        flush=True,
    )
    print(f"Results: {output_path}", flush=True)
    print(
        "Report: "
        f"{paths.revised_claim_extraction_summary_markdown(args.split)}",
        flush=True,
    )
    return 0 if summary["completion_status"] == "complete" else 2


def make_revised_alignment_output_schema(
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = config["revised_claim_alignment"]
    relations = list(settings["relation_labels"])
    if set(relations) != REVISED_ALIGNMENT_RELATIONS:
        raise ValueError(
            "B6b relation labels differ from the frozen taxonomy"
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["initial_claim_alignments"],
        "properties": {
            "initial_claim_alignments": {
                "type": "array",
                "minItems": 1,
                "maxItems": int(settings["maximum_initial_claims"]),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "initial_claim_id",
                        "relation",
                        "revised_claim_ids",
                        "rationale",
                    ],
                    "properties": {
                        "initial_claim_id": {
                            "type": "string",
                            "minLength": 5,
                            "maxLength": 160,
                        },
                        "relation": {
                            "type": "string",
                            "enum": relations,
                        },
                        "revised_claim_ids": {
                            "type": "array",
                            "maxItems": int(
                                settings["maximum_revised_claims"]
                            ),
                            "uniqueItems": True,
                            "items": {
                                "type": "string",
                                "minLength": 5,
                                "maxLength": 160,
                            },
                        },
                        "rationale": {
                            "type": "string",
                            "minLength": 3,
                            "maxLength": 360,
                        },
                    },
                },
            }
        },
    }


def load_revised_alignment_prompt(
    config: dict[str, Any],
) -> tuple[Path, str]:
    path = PROJECT_ROOT / config["revised_claim_alignment"]["prompt_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"B6b alignment prompt is empty: {path}")
    found = {
        placeholder
        for placeholder in REVISED_ALIGNMENT_PLACEHOLDERS
        if placeholder in template
    }
    if found != REVISED_ALIGNMENT_PLACEHOLDERS:
        raise ValueError(
            f"B6b prompt placeholders are incomplete: {sorted(found)}"
        )
    for placeholder in REVISED_ALIGNMENT_PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ValueError(
                f"B6b prompt must contain {placeholder} exactly once"
            )
    return path, template


def build_revised_alignment_prompt(
    template: str,
    unit: dict[str, Any],
) -> str:
    initial_claims = [
        {
            "initial_claim_id": row["claim_id"],
            "claim": row["gold_claim"],
        }
        for row in unit["gold_rows"]
    ]
    revised_claims = [
        {
            "revised_claim_id": row["claim_id"],
            "claim": row["claim"],
        }
        for row in unit["revised_claims"]
    ]
    replacements = {
        "{original_question_json}": json.dumps(
            unit["original_question"],
            ensure_ascii=False,
        ),
        "{initial_claims_json}": json.dumps(
            initial_claims,
            ensure_ascii=False,
            indent=2,
        ),
        "{revised_response_json}": json.dumps(
            unit["revised_response"],
            ensure_ascii=False,
        ),
        "{revised_claims_json}": json.dumps(
            revised_claims,
            ensure_ascii=False,
            indent=2,
        ),
    }
    prompt = template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def parse_revised_alignment_output(
    raw_output: str,
    unit: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("B6b model output is empty")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"B6b model output is not strict JSON: {error}"
        ) from error
    if not isinstance(parsed, dict) or set(parsed) != {
        "initial_claim_alignments"
    }:
        raise ValueError(
            "B6b output must contain only initial_claim_alignments"
        )
    alignments = parsed["initial_claim_alignments"]
    if not isinstance(alignments, list):
        raise TypeError("initial_claim_alignments must be an array")
    expected_ids = [row["claim_id"] for row in unit["gold_rows"]]
    valid_revised_ids = {
        row["claim_id"] for row in unit["revised_claims"]
    }
    exact_fields = {
        "initial_claim_id",
        "relation",
        "revised_claim_ids",
        "rationale",
    }
    normalized: list[dict[str, Any]] = []
    actual_ids: list[str] = []
    for index, item in enumerate(alignments, start=1):
        if not isinstance(item, dict) or set(item) != exact_fields:
            raise ValueError(
                f"B6b alignment item {index} has unexpected fields"
            )
        initial_claim_id = item["initial_claim_id"]
        if not isinstance(initial_claim_id, str):
            raise TypeError(
                f"B6b item {index} initial_claim_id is not a string"
            )
        actual_ids.append(initial_claim_id)
        relation = item["relation"]
        if relation not in REVISED_ALIGNMENT_RELATIONS:
            raise ValueError(
                f"Invalid B6b relation for {initial_claim_id}: {relation}"
            )
        revised_ids = item["revised_claim_ids"]
        if not isinstance(revised_ids, list) or any(
            not isinstance(value, str) for value in revised_ids
        ):
            raise TypeError(
                f"revised_claim_ids for {initial_claim_id} must be strings"
            )
        if len(revised_ids) != len(set(revised_ids)):
            raise ValueError(
                f"Duplicate revised claim ID for {initial_claim_id}"
            )
        unknown = set(revised_ids) - valid_revised_ids
        if unknown:
            raise ValueError(
                f"Unknown revised claim IDs for {initial_claim_id}: "
                f"{sorted(unknown)}"
            )
        if relation in {"ABSENT", "PRESENT_UNEXTRACTED"} and revised_ids:
            raise ValueError(
                f"{relation} must not list revised claims for "
                f"{initial_claim_id}"
            )
        if relation not in {"ABSENT", "PRESENT_UNEXTRACTED"} and not revised_ids:
            raise ValueError(
                f"{relation} requires a revised claim for {initial_claim_id}"
            )
        rationale = item["rationale"]
        if not isinstance(rationale, str):
            raise TypeError(
                f"Rationale for {initial_claim_id} is not a string"
            )
        rationale = " ".join(rationale.split())
        if not 3 <= len(rationale) <= 360:
            raise ValueError(
                f"Rationale for {initial_claim_id} must contain "
                "3-360 characters"
            )
        normalized.append(
            {
                "initial_claim_id": initial_claim_id,
                "relation": relation,
                "revised_claim_ids": revised_ids,
                "rationale": rationale,
            }
        )
    if actual_ids != expected_ids:
        raise ValueError(
            "B6b output must contain every canonical initial claim exactly "
            "once and in its supplied order"
        )
    return normalized


def recover_revised_alignment_structure(
    raw_output: str,
    unit: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Conservatively recover a complete B6b alignment from its raw JSON.

    This recovery normally avoids revised-claim-ID inference. For an empty
    PARTIAL or MODIFIED relation, it may attach a single revised claim when
    either all non-function-word tokens from the initial claim occur in exactly
    one supplied revised claim, or the rationale explicitly says that the
    revised answer contains the claim and exactly one supplied revised claim
    covers at least 75% of those tokens. It may also normalize an empty PARTIAL to
    PRESENT_UNEXTRACTED when the rationale explicitly says that the full
    revised answer contains an aggregate list-level statement but no
    individual extracted claim captures it. These are deterministic structural
    repairs, not factuality judgments. An empty EQUIVALENT may likewise become
    PRESENT_UNEXTRACTED only when its rationale explicitly says the revised
    answer mentions content matching the initial claim. Other empty relations
    still require explicit absence/extraction wording, and omitted canonical
    claims receive visibly audited ABSENT placeholders.
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("B6b recovery requires non-empty raw model output")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"B6b recovery requires strict JSON: {error}"
        ) from error
    if not isinstance(parsed, dict) or set(parsed) != {
        "initial_claim_alignments"
    }:
        raise ValueError(
            "B6b recovery requires only initial_claim_alignments"
        )
    alignments = parsed["initial_claim_alignments"]
    if not isinstance(alignments, list):
        raise TypeError("initial_claim_alignments must be an array")

    expected_ids = [row["claim_id"] for row in unit["gold_rows"]]
    expected_set = set(expected_ids)
    valid_revised_ids = {
        row["claim_id"] for row in unit["revised_claims"]
    }
    initial_claim_text_by_id = {
        row["claim_id"]: row["gold_claim"] for row in unit["gold_rows"]
    }
    revised_claim_text_by_id = {
        row["claim_id"]: row["claim"] for row in unit["revised_claims"]
    }
    lexical_stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "to",
        "was",
        "were",
        "with",
    }

    def content_tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in lexical_stopwords
        }
    exact_fields = {
        "initial_claim_id",
        "relation",
        "revised_claim_ids",
        "rationale",
    }
    absence_markers = (
        "does not mention",
        "does not explicitly state",
        "doesn't mention",
        "not mentioned",
        "not specifically",
        "is absent",
        "was omitted",
        "omits this",
    )
    extraction_markers = (
        "not directly extracted",
        "not extracted",
        "extraction omission",
    )
    aggregate_extraction_marker_pairs = (
        ("revised answer mentions", "but not specifically by"),
    )
    equivalent_extraction_marker_pairs = (
        ("revised answer mentions", "matches the initial claim"),
    )
    explicit_presence_markers = (
        "revised answer includes",
        "revised answer mentions",
        "revised claim includes",
    )
    normalized_by_id: dict[str, dict[str, Any]] = {}
    actual_ids: list[str] = []
    repairs: list[dict[str, Any]] = []

    for index, item in enumerate(alignments, start=1):
        if not isinstance(item, dict) or set(item) != exact_fields:
            raise ValueError(
                f"B6b recovery item {index} has unexpected fields"
            )
        initial_claim_id = item["initial_claim_id"]
        if (
            not isinstance(initial_claim_id, str)
            or initial_claim_id not in expected_set
        ):
            raise ValueError(
                f"B6b recovery found unknown initial claim at item {index}: "
                f"{initial_claim_id!r}"
            )
        if initial_claim_id in normalized_by_id:
            raise ValueError(
                f"B6b recovery found duplicate initial claim: "
                f"{initial_claim_id}"
            )
        actual_ids.append(initial_claim_id)

        relation = item["relation"]
        if relation not in REVISED_ALIGNMENT_RELATIONS:
            raise ValueError(
                f"B6b recovery found invalid relation for "
                f"{initial_claim_id}: {relation}"
            )
        revised_ids = item["revised_claim_ids"]
        if not isinstance(revised_ids, list) or any(
            not isinstance(value, str) for value in revised_ids
        ):
            raise TypeError(
                f"B6b recovery requires string revised IDs for "
                f"{initial_claim_id}"
            )
        if len(revised_ids) != len(set(revised_ids)):
            raise ValueError(
                f"B6b recovery found duplicate revised IDs for "
                f"{initial_claim_id}"
            )
        unknown_revised = set(revised_ids) - valid_revised_ids
        if unknown_revised:
            raise ValueError(
                f"B6b recovery will not infer replacements for unknown "
                f"revised IDs: {sorted(unknown_revised)}"
            )
        rationale = item["rationale"]
        if not isinstance(rationale, str):
            raise TypeError(
                f"B6b recovery rationale is not a string for "
                f"{initial_claim_id}"
            )
        rationale = " ".join(rationale.split())
        rationale_lower = rationale.lower()

        if relation in {"PARTIAL", "MODIFIED"} and not revised_ids:
            initial_tokens = content_tokens(
                initial_claim_text_by_id[initial_claim_id]
            )
            lexical_candidates = [
                revised_id
                for revised_id, revised_text in revised_claim_text_by_id.items()
                if len(initial_tokens) >= 3
                and initial_tokens.issubset(content_tokens(revised_text))
            ]
            if len(lexical_candidates) == 1:
                revised_ids = lexical_candidates
                repairs.append(
                    {
                        "initial_claim_id": initial_claim_id,
                        "method": (
                            f"empty_{relation.lower()}_to_unique_lexical_"
                            "containment"
                        ),
                        "original_relation": relation,
                        "recovered_relation": relation,
                        "revised_claim_ids_inferred": True,
                        "inferred_revised_claim_ids": list(revised_ids),
                        "audit_required": True,
                    }
                )
            elif (
                len(initial_tokens) >= 3
                and any(
                    marker in rationale_lower
                    for marker in explicit_presence_markers
                )
            ):
                coverage_by_id = {
                    revised_id: (
                        len(
                            initial_tokens.intersection(
                                content_tokens(revised_text)
                            )
                        )
                        / len(initial_tokens)
                    )
                    for revised_id, revised_text
                    in revised_claim_text_by_id.items()
                }
                high_coverage_candidates = [
                    revised_id
                    for revised_id, coverage in coverage_by_id.items()
                    if coverage >= 0.75
                ]
                if len(high_coverage_candidates) == 1:
                    revised_ids = high_coverage_candidates
                    inferred_id = revised_ids[0]
                    repairs.append(
                        {
                            "initial_claim_id": initial_claim_id,
                            "method": (
                                f"empty_{relation.lower()}_to_unique_high_"
                                "coverage_lexical_match"
                            ),
                            "original_relation": relation,
                            "recovered_relation": relation,
                            "revised_claim_ids_inferred": True,
                            "inferred_revised_claim_ids": list(revised_ids),
                            "lexical_coverage": round(
                                coverage_by_id[inferred_id],
                                4,
                            ),
                            "audit_required": True,
                        }
                    )

        if (
            relation == "PARTIAL"
            and not revised_ids
            and any(
                first in rationale_lower and second in rationale_lower
                for first, second in aggregate_extraction_marker_pairs
            )
        ):
            relation = "PRESENT_UNEXTRACTED"
            repairs.append(
                {
                    "initial_claim_id": initial_claim_id,
                    "method": (
                        "empty_partial_to_present_unextracted_from_explicit_"
                        "aggregate_rationale"
                    ),
                    "original_relation": "PARTIAL",
                    "recovered_relation": "PRESENT_UNEXTRACTED",
                    "revised_claim_ids_inferred": False,
                    "audit_required": True,
                }
            )

        if (
            relation == "EQUIVALENT"
            and not revised_ids
            and any(
                first in rationale_lower and second in rationale_lower
                for first, second in equivalent_extraction_marker_pairs
            )
        ):
            relation = "PRESENT_UNEXTRACTED"
            repairs.append(
                {
                    "initial_claim_id": initial_claim_id,
                    "method": (
                        "empty_equivalent_to_present_unextracted_from_"
                        "explicit_matching_rationale"
                    ),
                    "original_relation": "EQUIVALENT",
                    "recovered_relation": "PRESENT_UNEXTRACTED",
                    "revised_claim_ids_inferred": False,
                    "audit_required": True,
                }
            )

        if relation not in {"ABSENT", "PRESENT_UNEXTRACTED"} and not revised_ids:
            original_relation = relation
            if any(marker in rationale_lower for marker in extraction_markers):
                relation = "PRESENT_UNEXTRACTED"
                method = (
                    "empty_relation_to_present_unextracted_from_explicit_"
                    "rationale"
                )
            elif any(marker in rationale_lower for marker in absence_markers):
                relation = "ABSENT"
                method = "empty_relation_to_absent_from_explicit_rationale"
            else:
                raise ValueError(
                    "B6b recovery refuses an empty semantic relation without "
                    f"an explicit non-mention/extraction rationale: "
                    f"{initial_claim_id}"
                )
            repairs.append(
                {
                    "initial_claim_id": initial_claim_id,
                    "method": method,
                    "original_relation": original_relation,
                    "recovered_relation": relation,
                    "revised_claim_ids_inferred": False,
                    "audit_required": True,
                }
            )

        normalized_by_id[initial_claim_id] = {
            "initial_claim_id": initial_claim_id,
            "relation": relation,
            "revised_claim_ids": revised_ids,
            "rationale": rationale,
        }

    if actual_ids != [
        claim_id for claim_id in expected_ids if claim_id in normalized_by_id
    ]:
        raise ValueError(
            "B6b recovery only supports model rows that remain in canonical "
            "initial-claim order"
        )

    for initial_claim_id in expected_ids:
        if initial_claim_id in normalized_by_id:
            continue
        normalized_by_id[initial_claim_id] = {
            "initial_claim_id": initial_claim_id,
            "relation": "ABSENT",
            "revised_claim_ids": [],
            "rationale": (
                "The model omitted this canonical initial claim from its "
                "alignment output; a conservative unmatched placeholder was "
                "inserted and requires audit."
            ),
        }
        repairs.append(
            {
                "initial_claim_id": initial_claim_id,
                "method": "missing_initial_claim_to_absent_placeholder",
                "original_relation": None,
                "recovered_relation": "ABSENT",
                "revised_claim_ids_inferred": False,
                "audit_required": True,
            }
        )

    recovered = [normalized_by_id[claim_id] for claim_id in expected_ids]
    validated = parse_revised_alignment_output(
        json.dumps(
            {"initial_claim_alignments": recovered},
            ensure_ascii=False,
        ),
        unit,
    )
    if not repairs:
        raise ValueError("B6b recovery found no supported structural repair")
    return validated, repairs


def preflight_revised_alignment_ollama(
    client: Any,
    config: dict[str, Any],
) -> str:
    settings = config["revised_claim_alignment"]
    model = settings["model"]
    try:
        response = client.list()
    except Exception as error:
        raise ConnectionError(
            "Ollama preflight failed before any B6b output was changed"
        ) from error
    models = response_value(response, "models")
    if not isinstance(models, (list, tuple)):
        raise ValueError("Ollama preflight returned no model list")
    available: dict[str, str | None] = {}
    for item in models:
        name = response_value(item, "model") or response_value(item, "name")
        if isinstance(name, str) and name:
            digest = response_value(item, "digest")
            available[name] = digest if isinstance(digest, str) else None
    if model not in available:
        raise ValueError(
            f"Configured B6b model {model!r} is not installed; "
            f"available={sorted(available)}"
        )
    digest = available[model]
    if not digest:
        raise ValueError(f"Ollama provided no digest for {model!r}")
    expected = settings["expected_model_digest"]
    if digest != expected:
        raise ValueError(
            "Installed model digest differs from the frozen B6b "
            f"configuration: expected={expected}, actual={digest}"
        )
    return digest


def load_validated_b6a_results(
    paths: Any,
    config: dict[str, Any],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    b5_results, _ = load_validated_b5_results(paths, config, split)
    units = revised_claim_units(b5_results)
    output_path = paths.revised_claim_extraction_results(split)
    if not output_path.exists():
        raise FileNotFoundError(
            f"B6a output is missing for {split}: {output_path}"
        )
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B6a results do not have one valid model digest")
    prompt_path, _ = load_revised_claim_prompt(config)
    run_config = build_revised_claim_run_config(
        config,
        split=split,
        prompt_path=prompt_path,
        b5_results_path=paths.revision_results(split),
        model_digest=next(iter(model_digests)),
    )
    validate_existing_revised_claims(results, units, run_config, config)
    expected_ids = {unit["response_id"] for unit in units}
    ok_ids = {
        row["response_id"] for row in results if row.get("status") == "ok"
    }
    if ok_ids != expected_ids:
        raise ValueError(
            f"B6a {split} is incomplete; finish extraction before B6b"
        )
    return results, run_config


def revised_alignment_units(
    paths: Any,
    config: dict[str, Any],
    split: str,
) -> tuple[Path, list[dict[str, Any]]]:
    b6a_results, _ = load_validated_b6a_results(paths, config, split)
    gold_path, _, gold_by_response = load_gold_claims_by_response(config)
    units: list[dict[str, Any]] = []
    for row in sorted(
        b6a_results,
        key=lambda item: (
            int(item["source_record_index"]),
            item["response_id"],
        ),
    ):
        response_id = row["response_id"]
        gold_rows = gold_by_response.get(response_id, [])
        if not gold_rows:
            raise ValueError(f"No canonical gold claims for {response_id}")
        if any(item["prompt"] != row["original_question"] for item in gold_rows):
            raise ValueError(f"Question mismatch in B6b unit {response_id}")
        units.append(
            {
                "response_id": response_id,
                "source_record_index": row["source_record_index"],
                "split": row["split"],
                "original_question": row["original_question"],
                "revised_response": row["revised_response"],
                "revised_response_sha256": row["revised_response_sha256"],
                "revised_claims": row["revised_claims"],
                "b6a_run_fingerprint": row["run_fingerprint"],
                "gold_rows": gold_rows,
            }
        )
    expected_responses = int(
        config["split_policy"]["expected"][f"{split}_responses"]
    )
    expected_claims = int(
        config["split_policy"]["expected"][f"{split}_all_claims"]
    )
    if len(units) != expected_responses:
        raise ValueError(
            f"B6b {split} expected {expected_responses} responses, "
            f"found {len(units)}"
        )
    actual_claims = sum(len(unit["gold_rows"]) for unit in units)
    if actual_claims != expected_claims:
        raise ValueError(
            f"B6b {split} expected {expected_claims} initial claims, "
            f"found {actual_claims}"
        )
    settings = config["revised_claim_alignment"]
    if any(
        len(unit["gold_rows"]) > int(settings["maximum_initial_claims"])
        for unit in units
    ):
        raise ValueError("A B6b unit exceeds maximum_initial_claims")
    if any(
        len(unit["revised_claims"]) > int(settings["maximum_revised_claims"])
        for unit in units
    ):
        raise ValueError("A B6b unit exceeds maximum_revised_claims")
    return gold_path, units


def build_revised_alignment_run_config(
    config: dict[str, Any],
    *,
    split: str,
    prompt_path: Path,
    gold_path: Path,
    b6a_results_path: Path,
    model_digest: str,
) -> dict[str, Any]:
    settings = config["revised_claim_alignment"]
    schema = make_revised_alignment_output_schema(config)
    payload = {
        "experiment": config["experiment"],
        "stage": "B6b_gold_revised_claim_alignment",
        "split": split,
        "alignment_unit": "one_model_call_per_response",
        "model": settings["model"],
        "model_digest": model_digest,
        "temperature": float(settings["temperature"]),
        "seed": int(settings["seed"]),
        "num_predict": int(settings["num_predict"]),
        "think": bool(settings["think"]),
        "timeout_seconds": float(settings["timeout_seconds"]),
        "max_retries": int(settings["max_retries"]),
        "max_consecutive_request_errors": int(
            settings["max_consecutive_request_errors"]
        ),
        "relation_labels": settings["relation_labels"],
        "prompt_version": settings["prompt_version"],
        "prompt_sha256": sha256_file(prompt_path),
        "result_schema_version": settings["result_schema_version"],
        "output_schema_version": settings["output_schema_version"],
        "output_schema_sha256": canonical_json_hash(schema),
        "gold_claims_sha256": sha256_file(gold_path),
        "b6a_results_sha256": sha256_file(b6a_results_path),
        "model_input_fields": config["leakage_policy"][
            "revised_claim_alignment_model_input_fields"
        ],
        "withheld_fields": config["leakage_policy"][
            "revised_claim_alignment_withheld_fields"
        ],
    }
    return {**payload, "run_fingerprint": canonical_json_hash(payload)}


def create_revised_alignment_result_base(
    unit: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "result_schema_version": run_config["result_schema_version"],
        "response_id": unit["response_id"],
        "source_record_index": unit["source_record_index"],
        "split": unit["split"],
        "original_question": unit["original_question"],
        "original_question_sha256": sha256_text(unit["original_question"]),
        "revised_response": unit["revised_response"],
        "revised_response_sha256": unit["revised_response_sha256"],
        "canonical_initial_claims": [
            {
                "claim_id": row["claim_id"],
                "claim": row["gold_claim"],
            }
            for row in unit["gold_rows"]
        ],
        "extracted_revised_claims": unit["revised_claims"],
        "stage": run_config["stage"],
        "alignment_unit": run_config["alignment_unit"],
        "model_input_fields": run_config["model_input_fields"],
        "withheld_fields": run_config["withheld_fields"],
        "b6a_run_fingerprint": unit["b6a_run_fingerprint"],
        "b6a_results_sha256": run_config["b6a_results_sha256"],
        "gold_claims_sha256": run_config["gold_claims_sha256"],
        "model": run_config["model"],
        "model_digest": run_config["model_digest"],
        "temperature": run_config["temperature"],
        "seed": run_config["seed"],
        "num_predict": run_config["num_predict"],
        "think": run_config["think"],
        "timeout_seconds": run_config["timeout_seconds"],
        "max_retries": run_config["max_retries"],
        "max_consecutive_request_errors": run_config[
            "max_consecutive_request_errors"
        ],
        "relation_labels": run_config["relation_labels"],
        "prompt_version": run_config["prompt_version"],
        "prompt_sha256": run_config["prompt_sha256"],
        "output_schema_version": run_config["output_schema_version"],
        "output_schema_sha256": run_config["output_schema_sha256"],
        "run_fingerprint": run_config["run_fingerprint"],
    }


def process_revised_alignment_unit(
    unit: dict[str, Any],
    template: str,
    config: dict[str, Any],
    run_config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    result = create_revised_alignment_result_base(unit, run_config)
    prompt = build_revised_alignment_prompt(template, unit)
    raw_output: str | None = None
    metadata: dict[str, Any] = {}
    request_error: Exception | None = None
    attempts = 0
    started = time.perf_counter()
    for attempt in range(run_config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw_output, metadata = call_ollama(
                client,
                run_config,
                make_revised_alignment_output_schema(config),
                prompt,
            )
            request_error = None
            break
        except Exception as error:
            request_error = error
            if attempt < run_config["max_retries"]:
                time.sleep(1.0)
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
                "initial_claim_alignments": None,
                "error": f"{type(request_error).__name__}: {request_error}",
            }
        )
        return result
    try:
        alignments = parse_revised_alignment_output(raw_output or "", unit)
    except Exception as error:
        result.update(
            {
                "status": "parse_error",
                "initial_claim_alignments": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return result
    matched_revised_ids = {
        claim_id
        for alignment in alignments
        for claim_id in alignment["revised_claim_ids"]
    }
    result.update(
        {
            "status": "ok",
            "initial_claim_alignments": alignments,
            "matched_revised_claim_ids": sorted(matched_revised_ids),
            "added_revised_claim_ids": [
                row["claim_id"]
                for row in unit["revised_claims"]
                if row["claim_id"] not in matched_revised_ids
            ],
            "error": None,
        }
    )
    return result


def validate_existing_revised_alignments(
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    unit_by_id = {unit["response_id"]: unit for unit in units}
    seen: set[str] = set()
    for row in rows:
        response_id = row.get("response_id")
        if response_id not in unit_by_id:
            raise ValueError(
                f"B6b output contains unexpected response_id: {response_id}"
            )
        if response_id in seen:
            raise ValueError(f"Duplicate response_id in B6b: {response_id}")
        seen.add(response_id)
        unit = unit_by_id[response_id]
        expected_base = create_revised_alignment_result_base(unit, run_config)
        for key, expected in expected_base.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"Existing B6b output is incompatible for "
                    f"{response_id}: {key}"
                )
        status = row.get("status")
        if status not in {"ok", "request_error", "parse_error"}:
            raise ValueError(f"Invalid B6b status for {response_id}: {status}")
        if status == "ok":
            alignments = row.get("initial_claim_alignments")
            parsed = parse_revised_alignment_output(
                json.dumps(
                    {"initial_claim_alignments": alignments},
                    ensure_ascii=False,
                ),
                unit,
            )
            if parsed != alignments:
                raise ValueError(
                    f"Non-canonical B6b alignments for {response_id}"
                )
            matched = {
                claim_id
                for alignment in alignments
                for claim_id in alignment["revised_claim_ids"]
            }
            expected_matched = sorted(matched)
            expected_added = [
                item["claim_id"]
                for item in unit["revised_claims"]
                if item["claim_id"] not in matched
            ]
            if row.get("matched_revised_claim_ids") != expected_matched:
                raise ValueError(
                    f"B6b matched revised IDs mismatch for {response_id}"
                )
            if row.get("added_revised_claim_ids") != expected_added:
                raise ValueError(
                    f"B6b added revised IDs mismatch for {response_id}"
                )
            if row.get("error") is not None:
                raise ValueError(
                    f"Successful B6b row has an error for {response_id}"
                )
        elif row.get("initial_claim_alignments") is not None:
            raise ValueError(
                f"Technical B6b failure has alignments for {response_id}"
            )


def provisional_transition(
    human_label: str,
    relation: str,
) -> str:
    if relation == "PRESENT_UNEXTRACTED":
        return "EXTRACTION_OMISSION_REQUIRES_REVIEW"
    if human_label == "UNKNOWN":
        return "UNKNOWN_ANCHOR_REQUIRES_REVIEW"
    if relation == "EQUIVALENT":
        return (
            "FACTUAL_RETAINED_CANDIDATE"
            if human_label == "FACTUAL"
            else "ERROR_RETAINED_CANDIDATE"
        )
    if relation == "ABSENT":
        return (
            "FACTUAL_DELETED_CANDIDATE"
            if human_label == "FACTUAL"
            else "ERROR_REMOVED_BY_DELETION_CANDIDATE"
        )
    return "REVISED_FACTUALITY_REQUIRED"


def flatten_revised_alignment_results(
    units: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unit_by_id = {unit["response_id"]: unit for unit in units}
    initial_rows: list[dict[str, Any]] = []
    added_rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") != "ok":
            continue
        unit = unit_by_id[result["response_id"]]
        recovery = result.get("structure_recovery")
        repair_by_initial_id = {
            repair["initial_claim_id"]: repair
            for repair in (
                recovery.get("repairs", [])
                if isinstance(recovery, dict)
                else []
            )
        }
        gold_by_id = {
            row["claim_id"]: row for row in unit["gold_rows"]
        }
        revised_by_id = {
            row["claim_id"]: row for row in unit["revised_claims"]
        }
        for alignment in result["initial_claim_alignments"]:
            gold = gold_by_id[alignment["initial_claim_id"]]
            alignment_repair = repair_by_initial_id.get(gold["claim_id"])
            matched = [
                revised_by_id[claim_id]
                for claim_id in alignment["revised_claim_ids"]
            ]
            initial_rows.append(
                {
                    "schema_version": (
                        "cove_initial_transition_candidate_v1"
                    ),
                    "response_id": result["response_id"],
                    "source_record_index": result["source_record_index"],
                    "split": result["split"],
                    "initial_claim_id": gold["claim_id"],
                    "initial_claim": gold["gold_claim"],
                    "human_label": gold["human_label"],
                    "human_label_bool": gold["human_label_bool"],
                    "human_label_evaluation_only": True,
                    "relation": alignment["relation"],
                    "revised_claim_ids": alignment[
                        "revised_claim_ids"
                    ],
                    "revised_claims": matched,
                    "alignment_rationale": alignment["rationale"],
                    "alignment_audit_required": (
                        alignment_repair is not None
                    ),
                    "alignment_structure_recovery": alignment_repair,
                    "provisional_transition": provisional_transition(
                        gold["human_label"],
                        alignment["relation"],
                    ),
                    "revised_factuality_status": (
                        "pending_B6c_for_modified_partial_and_additions"
                    ),
                    "b6b_run_fingerprint": result["run_fingerprint"],
                }
            )
        for revised_id in result["added_revised_claim_ids"]:
            revised = revised_by_id[revised_id]
            added_rows.append(
                {
                    "schema_version": "cove_added_claim_candidate_v1",
                    "response_id": result["response_id"],
                    "source_record_index": result["source_record_index"],
                    "split": result["split"],
                    "revised_claim_id": revised_id,
                    "revised_claim": revised["claim"],
                    "revised_claim_sha256": revised["claim_sha256"],
                    "alignment_status": "ADDED_UNMATCHED",
                    "factuality_status": "pending_B6c",
                    "b6b_run_fingerprint": result["run_fingerprint"],
                }
            )
    return initial_rows, added_rows


def revised_alignment_summary(
    units: list[dict[str, Any]],
    results: list[dict[str, Any]],
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    result_by_id = {row["response_id"]: row for row in results}
    status_counts = Counter(
        result_by_id.get(unit["response_id"], {}).get("status", "missing")
        for unit in units
    )
    successful = [
        result_by_id[unit["response_id"]]
        for unit in units
        if result_by_id.get(unit["response_id"], {}).get("status") == "ok"
    ]
    initial_rows, added_rows = flatten_revised_alignment_results(
        units,
        successful,
    )
    relation_counts = Counter(row["relation"] for row in initial_rows)
    relation_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    for row in initial_rows:
        relation_by_label[row["human_label"]][row["relation"]] += 1
    transition_counts = Counter(
        row["provisional_transition"] for row in initial_rows
    )
    total_revised = sum(
        len(unit["revised_claims"])
        for unit in units
        if result_by_id.get(unit["response_id"], {}).get("status") == "ok"
    )
    matched_revised = total_revised - len(added_rows)
    complete = len(successful) == len(units)
    recovered_rows = [
        row
        for row in successful
        if isinstance(row.get("structure_recovery"), dict)
        and row["structure_recovery"].get("applied") is True
    ]
    recovery_methods = Counter(
        repair["method"]
        for row in recovered_rows
        for repair in row["structure_recovery"].get("repairs", [])
    )
    any_revised_claim_ids_inferred = any(
        repair.get("revised_claim_ids_inferred") is True
        for row in recovered_rows
        for repair in row["structure_recovery"].get("repairs", [])
    )
    summary = {
        "schema_version": "cove_revised_claim_alignment_summary_v1",
        "experiment": "Experiment B: CoVe mechanism evaluation",
        "stage": "B6b_gold_revised_claim_alignment",
        "split": split,
        "completion_status": "complete" if complete else "incomplete",
        "selected_responses": len(units),
        "successful_responses": len(successful),
        "status_counts": dict(status_counts),
        "initial_claims_aligned": len(initial_rows),
        "initial_relation_counts": dict(sorted(relation_counts.items())),
        "initial_relation_counts_by_human_label": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(relation_by_label.items())
        },
        "provisional_transition_counts": dict(
            sorted(transition_counts.items())
        ),
        "revised_claims": {
            "total": total_revised,
            "matched_to_initial": matched_revised,
            "added_unmatched": len(added_rows),
            "matched_rate": (
                round(matched_revised / total_revised, 4)
                if total_revised
                else None
            ),
        },
        "present_unextracted_initial_claims": relation_counts[
            "PRESENT_UNEXTRACTED"
        ],
        "conservative_structure_recovery": {
            "recovered_responses": len(recovered_rows),
            "audit_required_responses": sum(
                row["structure_recovery"].get("audit_required") is True
                for row in recovered_rows
            ),
            "repair_counts": dict(sorted(recovery_methods.items())),
            "revised_claim_ids_inferred": any_revised_claim_ids_inferred,
        },
        "leakage_boundary": {
            "human_labels_exposed_to_model": False,
            "gold_or_retrieved_evidence_exposed": False,
            "initial_response_exposed": False,
            "full_revised_response_exposed": True,
            "full_revised_response_purpose": (
                "distinguish deletion from B6a extraction omission"
            ),
        },
        "factuality_evaluation_status": (
            "pending_B6c_for_modified_partial_and_added_claims"
        ),
        "interpretation_notes": [
            "B6b is semantic alignment, not revised-claim factuality evaluation.",
            "Human labels are joined only after model output for stratified candidate transitions.",
            "PRESENT_UNEXTRACTED identifies a B6a omission and must not be counted as deletion.",
            "Modified, partial, and added claims require independent factuality evaluation before net gain is computed.",
            "B6b is LLM-assisted silver alignment produced by the same model family and requires development audit.",
        ],
    }
    return summary, initial_rows, added_rows


def build_revised_alignment_markdown(summary: dict[str, Any]) -> str:
    revised = summary["revised_claims"]
    lines = [
        "# Experiment B — B6b Gold-to-Revised Claim Alignment",
        "",
        f"- Split: `{summary['split']}`",
        f"- Completion: **{summary['completion_status']}**",
        f"- Responses: {summary['successful_responses']}/{summary['selected_responses']}",
        f"- Canonical initial claims aligned: {summary['initial_claims_aligned']}",
        f"- Revised claims: {revised['total']}",
        "- Human labels/evidence exposed to model: **no**",
        "",
        "## Initial-claim relations",
        "",
        "| Relation | Count |",
        "|---|---:|",
    ]
    for relation in sorted(REVISED_ALIGNMENT_RELATIONS):
        lines.append(
            f"| `{relation}` | "
            f"{summary['initial_relation_counts'].get(relation, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Revised-claim coverage",
            "",
            "| Measure | Value |",
            "|---|---:|",
            f"| Matched to at least one initial claim | {revised['matched_to_initial']} |",
            f"| Added/unmatched claims | {revised['added_unmatched']} |",
            f"| Matched rate | {revised['matched_rate']} |",
            f"| Initial claims present but unextracted in B6a | "
            f"{summary['present_unextracted_initial_claims']} |",
            "",
            "## Conservative structural recovery",
            "",
            f"- Recovered responses: "
            f"{summary['conservative_structure_recovery']['recovered_responses']}",
            f"- Audit-required responses: "
            f"{summary['conservative_structure_recovery']['audit_required_responses']}",
            "- Revised-claim IDs inferred: **no**",
            "",
            "## Relations by hidden human label",
            "",
        ]
    )
    for label, counts in summary[
        "initial_relation_counts_by_human_label"
    ].items():
        lines.extend([f"### {label}", "", "| Relation | Count |", "|---|---:|"])
        for relation in sorted(REVISED_ALIGNMENT_RELATIONS):
            lines.append(f"| `{relation}` | {counts.get(relation, 0)} |")
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "B6b measures semantic transitions only. `EQUIVALENT` and `ABSENT` "
            "support provisional retention/deletion accounting. `MODIFIED`, "
            "`PARTIAL`, and all added claims require a later independent "
            "factuality decision before correction, harm, new-error, or net "
            "factual gain can be reported.",
            "",
        ]
    )
    return "\n".join(lines)


def write_revised_alignment_artifacts(
    paths: Any,
    split: str,
    summary: dict[str, Any],
    initial_rows: list[dict[str, Any]],
    added_rows: list[dict[str, Any]],
) -> None:
    atomic_write_jsonl(paths.initial_transition_candidates(split), initial_rows)
    if added_rows:
        atomic_write_jsonl(paths.added_claim_candidates(split), added_rows)
    else:
        atomic_write_text(paths.added_claim_candidates(split), "")
    atomic_write_json(
        paths.revised_claim_alignment_summary_json(split),
        summary,
    )
    atomic_write_text(
        paths.revised_claim_alignment_summary_markdown(split),
        build_revised_alignment_markdown(summary),
    )


def run_revised_claim_alignment(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    gold_path, units = revised_alignment_units(paths, config, args.split)
    prompt_path, template = load_revised_alignment_prompt(config)
    if args.dry_run:
        run_config = build_revised_alignment_run_config(
            config,
            split=args.split,
            prompt_path=prompt_path,
            gold_path=gold_path,
            b6a_results_path=paths.revised_claim_extraction_results(
                args.split
            ),
            model_digest="DRY_RUN_NOT_CHECKED",
        )
        print("EXPERIMENT B B6B GOLD-TO-REVISED ALIGNMENT DRY RUN")
        print(f"Split: {args.split}; response calls: {len(units)}")
        print(
            "Initial claims: "
            f"{sum(len(unit['gold_rows']) for unit in units)}; "
            "revised claims: "
            f"{sum(len(unit['revised_claims']) for unit in units)}"
        )
        print(f"Run fingerprint: {run_config['run_fingerprint']}")
        print(
            "Human labels and evidence are withheld. The full revised "
            "response is included only to detect B6a extraction omissions."
        )
        print(f"\n--- Preview: {units[0]['response_id']} ---")
        print(build_revised_alignment_prompt(template, units[0]))
        print("\nNo Ollama calls were made and no files were written.")
        return 0
    client = Client(
        host=args.ollama_host,
        timeout=float(config["revised_claim_alignment"]["timeout_seconds"]),
    )
    model_digest = preflight_revised_alignment_ollama(client, config)
    run_config = build_revised_alignment_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        gold_path=gold_path,
        b6a_results_path=paths.revised_claim_extraction_results(args.split),
        model_digest=model_digest,
    )
    output_path = paths.revised_claim_alignment_results(args.split)
    existing = load_jsonl(output_path) if output_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Output already exists; rerun with --resume: {output_path}"
        )
    validate_existing_revised_alignments(existing, units, run_config)
    result_by_id = {row["response_id"]: row for row in existing}
    pending = [
        unit
        for unit in units
        if result_by_id.get(unit["response_id"], {}).get("status") != "ok"
    ]
    print(
        f"Experiment B B6b: split={args.split}, total={len(units)}, "
        f"retained_ok={len(units) - len(pending)}, pending={len(pending)}",
        flush=True,
    )
    consecutive_request_errors = 0
    max_consecutive = int(
        config["revised_claim_alignment"][
            "max_consecutive_request_errors"
        ]
    )
    order = {unit["response_id"]: index for index, unit in enumerate(units)}
    for unit in pending:
        overall_index = order[unit["response_id"]] + 1
        print(
            f"[{overall_index}/{len(units)}] {unit['response_id']} aligning "
            f"{len(unit['gold_rows'])} initial ↔ "
            f"{len(unit['revised_claims'])} revised claims ...",
            flush=True,
        )
        result = process_revised_alignment_unit(
            unit,
            template,
            config,
            run_config,
            client,
        )
        result_by_id[unit["response_id"]] = result
        ordered = [
            result_by_id[item["response_id"]]
            for item in units
            if item["response_id"] in result_by_id
        ]
        atomic_write_jsonl(output_path, ordered)
        if result["status"] == "ok":
            consecutive_request_errors = 0
            relation_counts = Counter(
                item["relation"]
                for item in result["initial_claim_alignments"]
            )
            print(
                f"[{overall_index}/{len(units)}] success, "
                f"added={len(result['added_revised_claim_ids'])}, "
                f"relations={dict(relation_counts)}, "
                f"{result['latency_seconds']:.1f}s",
                flush=True,
            )
        else:
            if result["status"] == "request_error":
                consecutive_request_errors += 1
            else:
                consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(units)}] {result['status']}: "
                f"{result['error']}",
                flush=True,
            )
            if consecutive_request_errors >= max_consecutive:
                print(
                    "Stopping after the configured consecutive request-error "
                    "limit. The same command will resume safely.",
                    file=sys.stderr,
                    flush=True,
                )
                break
    all_results = [
        result_by_id[unit["response_id"]]
        for unit in units
        if unit["response_id"] in result_by_id
    ]
    validate_existing_revised_alignments(all_results, units, run_config)
    summary, initial_rows, added_rows = revised_alignment_summary(
        units,
        all_results,
        args.split,
    )
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    summary["initial_transition_file"] = str(
        paths.initial_transition_candidates(args.split).relative_to(
            PROJECT_ROOT
        )
    )
    summary["added_claim_file"] = str(
        paths.added_claim_candidates(args.split).relative_to(PROJECT_ROOT)
    )
    write_revised_alignment_artifacts(
        paths,
        args.split,
        summary,
        initial_rows,
        added_rows,
    )
    print(f"Results: {output_path}", flush=True)
    print(
        "Report: "
        f"{paths.revised_claim_alignment_summary_markdown(args.split)}",
        flush=True,
    )
    return 0 if summary["completion_status"] == "complete" else 2


def analyze_revised_claim_alignment(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    gold_path, units = revised_alignment_units(paths, config, args.split)
    output_path = paths.revised_claim_alignment_results(args.split)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B6b results do not have one valid model digest")
    prompt_path, _ = load_revised_alignment_prompt(config)
    run_config = build_revised_alignment_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        gold_path=gold_path,
        b6a_results_path=paths.revised_claim_extraction_results(args.split),
        model_digest=next(iter(model_digests)),
    )
    validate_existing_revised_alignments(results, units, run_config)
    summary, initial_rows, added_rows = revised_alignment_summary(
        units,
        results,
        args.split,
    )
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    summary["initial_transition_file"] = str(
        paths.initial_transition_candidates(args.split).relative_to(
            PROJECT_ROOT
        )
    )
    summary["added_claim_file"] = str(
        paths.added_claim_candidates(args.split).relative_to(PROJECT_ROOT)
    )
    write_revised_alignment_artifacts(
        paths,
        args.split,
        summary,
        initial_rows,
        added_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["completion_status"] == "complete" else 2


def recover_revised_claim_alignment_structure(
    args: argparse.Namespace,
) -> int:
    """Recover supported B6b structural contradictions without model calls."""
    paths = paths_for_args(args)
    config = load_config(paths)
    gold_path, units = revised_alignment_units(paths, config, args.split)
    output_path = paths.revised_claim_alignment_results(args.split)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B6b results do not have one valid model digest")
    prompt_path, _ = load_revised_alignment_prompt(config)
    run_config = build_revised_alignment_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        gold_path=gold_path,
        b6a_results_path=paths.revised_claim_extraction_results(args.split),
        model_digest=next(iter(model_digests)),
    )
    validate_existing_revised_alignments(results, units, run_config)
    unit_by_id = {unit["response_id"]: unit for unit in units}
    recovered_count = 0
    unrecovered: list[dict[str, str]] = []

    for row in results:
        if row.get("status") != "parse_error":
            continue
        response_id = row["response_id"]
        original_error = row.get("error")
        try:
            alignments, repairs = recover_revised_alignment_structure(
                row.get("raw_model_output", ""),
                unit_by_id[response_id],
            )
        except Exception as error:
            unrecovered.append(
                {
                    "response_id": response_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        matched_revised_ids = {
            claim_id
            for alignment in alignments
            for claim_id in alignment["revised_claim_ids"]
        }
        unit = unit_by_id[response_id]
        row.update(
            {
                "status": "ok",
                "initial_claim_alignments": alignments,
                "matched_revised_claim_ids": sorted(matched_revised_ids),
                "added_revised_claim_ids": [
                    item["claim_id"]
                    for item in unit["revised_claims"]
                    if item["claim_id"] not in matched_revised_ids
                ],
                "error": None,
                "structure_recovery": {
                    "applied": True,
                    "method": (
                        "conservative_b6b_structure_recovery_v1"
                    ),
                    "original_status": "parse_error",
                    "original_error": original_error,
                    "raw_model_output_preserved": True,
                    "model_recalled": False,
                    "revised_claim_ids_inferred": any(
                        repair.get("revised_claim_ids_inferred") is True
                        for repair in repairs
                    ),
                    "audit_required": True,
                    "repairs": repairs,
                    "recovered_at": utc_now(),
                },
            }
        )
        recovered_count += 1
        print(
            f"[recovered] {response_id}: "
            f"{len(repairs)} conservative repair(s), audit required",
            flush=True,
        )

    atomic_write_jsonl(output_path, results)
    validate_existing_revised_alignments(results, units, run_config)
    summary, initial_rows, added_rows = revised_alignment_summary(
        units,
        results,
        args.split,
    )
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    summary["initial_transition_file"] = str(
        paths.initial_transition_candidates(args.split).relative_to(
            PROJECT_ROOT
        )
    )
    summary["added_claim_file"] = str(
        paths.added_claim_candidates(args.split).relative_to(PROJECT_ROOT)
    )
    summary["conservative_structure_recovery"]["unrecovered"] = unrecovered
    write_revised_alignment_artifacts(
        paths,
        args.split,
        summary,
        initial_rows,
        added_rows,
    )
    print(
        f"B6b structural recovery: recovered={recovered_count}, "
        f"unrecovered={len(unrecovered)}, "
        f"completion={summary['completion_status']}",
        flush=True,
    )
    print(f"Results: {output_path}", flush=True)
    print(
        "Report: "
        f"{paths.revised_claim_alignment_summary_markdown(args.split)}",
        flush=True,
    )
    return 0 if summary["completion_status"] == "complete" else 2


def make_revised_factuality_output_schema(
    config: dict[str, Any],
) -> dict[str, Any]:
    labels = list(
        config["revised_claim_factuality"]["prediction_labels"]
    )
    if set(labels) != REVISED_FACTUALITY_LABELS:
        raise ValueError("B6c prediction labels differ from the frozen taxonomy")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["prediction", "confidence", "rationale"],
        "properties": {
            "prediction": {"type": "string", "enum": labels},
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "rationale": {
                "type": "string",
                "minLength": 3,
                "maxLength": 240,
            },
        },
    }


def load_revised_factuality_prompt(
    config: dict[str, Any],
) -> tuple[Path, str]:
    path = PROJECT_ROOT / config["revised_claim_factuality"]["prompt_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"B6c factuality prompt is empty: {path}")
    found = {
        placeholder
        for placeholder in REVISED_FACTUALITY_PLACEHOLDERS
        if placeholder in template
    }
    if found != REVISED_FACTUALITY_PLACEHOLDERS:
        raise ValueError(
            f"B6c prompt placeholders are incomplete: {sorted(found)}"
        )
    for placeholder in REVISED_FACTUALITY_PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ValueError(
                f"B6c prompt must contain {placeholder} exactly once"
            )
    return path, template


def parse_revised_factuality_output(
    raw_output: str,
    *,
    allow_word_limit_warning: bool = False,
) -> dict[str, Any]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("B6c model output is empty")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"B6c model output is not strict JSON: {error}"
        ) from error
    if not isinstance(parsed, dict) or set(parsed) != {
        "prediction",
        "confidence",
        "rationale",
    }:
        raise ValueError(
            "B6c output must contain exactly prediction, confidence, rationale"
        )
    prediction = parsed["prediction"]
    if prediction not in REVISED_FACTUALITY_LABELS:
        raise ValueError(f"Invalid B6c prediction: {prediction}")
    confidence = parsed["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("B6c confidence must be between 0 and 1")
    rationale = parsed["rationale"]
    if not isinstance(rationale, str):
        raise TypeError("B6c rationale must be a string")
    rationale = " ".join(rationale.split())
    if not 3 <= len(rationale) <= 240:
        raise ValueError(
            "B6c rationale must contain 3-240 characters"
        )
    word_limit_exceeded = len(rationale.split()) > 35
    if word_limit_exceeded and not allow_word_limit_warning:
        raise ValueError(
            "B6c rationale must contain at most 35 words and 240 characters"
        )
    normalized = {
        "prediction": prediction,
        "confidence": float(confidence),
        "rationale": rationale,
    }
    if word_limit_exceeded:
        normalized["format_warnings"] = [
            "RATIONALE_WORD_LIMIT_EXCEEDED"
        ]
    return normalized


def preflight_revised_factuality_ollama(
    client: Any,
    config: dict[str, Any],
) -> str:
    settings = config["revised_claim_factuality"]
    model = settings["model"]
    try:
        response = client.list()
    except Exception as error:
        raise ConnectionError(
            "Ollama preflight failed before any B6c output was changed"
        ) from error
    models = response_value(response, "models")
    if not isinstance(models, (list, tuple)):
        raise ValueError("Ollama preflight returned no model list")
    available: dict[str, str | None] = {}
    for item in models:
        name = response_value(item, "model") or response_value(item, "name")
        if isinstance(name, str) and name:
            digest = response_value(item, "digest")
            available[name] = digest if isinstance(digest, str) else None
    if model not in available:
        raise ValueError(
            f"Configured B6c model {model!r} is not installed; "
            f"available={sorted(available)}"
        )
    digest = available[model]
    if not digest:
        raise ValueError(f"Ollama provided no digest for {model!r}")
    expected = settings["expected_model_digest"]
    if digest != expected:
        raise ValueError(
            "Installed model digest differs from the frozen B6c "
            f"configuration: expected={expected}, actual={digest}"
        )
    return digest


def load_validated_b6b_results(
    paths: Any,
    config: dict[str, Any],
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gold_path, units = revised_alignment_units(paths, config, split)
    output_path = paths.revised_claim_alignment_results(split)
    if not output_path.exists():
        raise FileNotFoundError(
            f"B6b output is missing for {split}: {output_path}"
        )
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B6b results do not have one valid model digest")
    prompt_path, _ = load_revised_alignment_prompt(config)
    run_config = build_revised_alignment_run_config(
        config,
        split=split,
        prompt_path=prompt_path,
        gold_path=gold_path,
        b6a_results_path=paths.revised_claim_extraction_results(split),
        model_digest=next(iter(model_digests)),
    )
    validate_existing_revised_alignments(results, units, run_config)
    expected_ids = {unit["response_id"] for unit in units}
    ok_ids = {
        row["response_id"] for row in results if row.get("status") == "ok"
    }
    if ok_ids != expected_ids:
        raise ValueError(
            f"B6b {split} is incomplete; finish alignment before B6c"
        )
    return results, units, run_config


def revised_factuality_units(
    paths: Any,
    config: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    load_validated_b6b_results(paths, config, split)
    b6a_results, _ = load_validated_b6a_results(paths, config, split)
    expected_response_ids = {
        row["response_id"] for row in b6a_results
    }
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for response in sorted(
        b6a_results,
        key=lambda row: (
            int(row["source_record_index"]),
            row["response_id"],
        ),
    ):
        for claim in response["revised_claims"]:
            claim_id = claim["claim_id"]
            if claim_id in seen:
                raise ValueError(f"Duplicate B6c revised claim ID: {claim_id}")
            seen.add(claim_id)
            units.append(
                {
                    "revised_claim_id": claim_id,
                    "response_id": response["response_id"],
                    "source_record_index": response["source_record_index"],
                    "split": split,
                    "revised_claim": claim["claim"],
                    "revised_claim_sha256": claim["claim_sha256"],
                    "b6a_run_fingerprint": response["run_fingerprint"],
                }
            )
    unit_response_ids = {unit["response_id"] for unit in units}
    zero_claim_response_ids = {
        row["response_id"]
        for row in b6a_results
        if not row["revised_claims"]
    }
    if unit_response_ids | zero_claim_response_ids != expected_response_ids:
        raise ValueError(
            "B6c units do not preserve the complete branch-local B6a "
            "response set"
        )
    branch = getattr(paths, "branch", "a")
    if branch == "a" and split == "dev" and len(units) != 130:
        raise ValueError(
            "Canonical Branch A expected 130 B6c dev claims, "
            f"found {len(units)}"
        )
    if not units:
        raise ValueError(
            f"Branch {branch.upper()} has no B6c {split} claim units"
        )
    return units


def validate_revised_claim_evidence(
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    expected_by_id = {
        unit["revised_claim_id"]: unit for unit in units
    }
    seen: set[str] = set()
    top_k = int(config["revised_claim_factuality"]["evidence_top_k"])
    for row in rows:
        claim_id = row.get("revised_claim_id")
        if claim_id not in expected_by_id:
            raise ValueError(
                f"B6c evidence has unexpected revised claim: {claim_id}"
            )
        if claim_id in seen:
            raise ValueError(f"Duplicate B6c evidence claim: {claim_id}")
        seen.add(claim_id)
        unit = expected_by_id[claim_id]
        expected_fields = {
            "response_id": unit["response_id"],
            "source_record_index": unit["source_record_index"],
            "split": unit["split"],
            "revised_claim": unit["revised_claim"],
            "revised_claim_sha256": unit["revised_claim_sha256"],
            "retriever": config["revised_claim_factuality"]["retriever"],
            "top_k": top_k,
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"B6c evidence mismatch for {claim_id}: {field}"
                )
        items = row.get("items")
        if not isinstance(items, list) or len(items) != top_k:
            raise ValueError(
                f"B6c evidence for {claim_id} must contain top-{top_k}"
            )
        if [item.get("rank") for item in items] != list(
            range(1, top_k + 1)
        ):
            raise ValueError(f"B6c evidence ranks are invalid for {claim_id}")
        passage_ids: set[str] = set()
        for item in items:
            passage_id = item.get("passage_id")
            text = item.get("text")
            if (
                not isinstance(passage_id, str)
                or not passage_id
                or passage_id in passage_ids
            ):
                raise ValueError(
                    f"B6c evidence passage IDs invalid for {claim_id}"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"B6c evidence contains empty text for {claim_id}"
                )
            if item.get("text_sha256") != sha256_text(text):
                raise ValueError(
                    f"B6c evidence text hash mismatch for {claim_id}"
                )
            passage_ids.add(passage_id)
        normalized = row.get("normalized_text")
        if not isinstance(normalized, str) or not normalized:
            raise ValueError(f"B6c normalized evidence missing for {claim_id}")
        if row.get("normalized_sha256") != sha256_text(normalized):
            raise ValueError(
                f"B6c normalized evidence hash mismatch for {claim_id}"
            )
    if seen != set(expected_by_id):
        missing = sorted(set(expected_by_id) - seen)
        raise ValueError(
            f"B6c evidence is incomplete; missing {len(missing)} claims"
        )


def prepare_revised_claim_evidence(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    units = revised_factuality_units(paths, config, args.split)
    settings = config["revised_claim_factuality"]
    retrieval_config_path = PROJECT_ROOT / settings[
        "retrieval_config_path"
    ]
    corpus_paths = retrieval_paths(PROJECT_ROOT, args.scope)
    eval_paths = evaluation_paths(corpus_paths)
    if retrieval_config_path.resolve() != eval_paths.config.resolve():
        raise ValueError("B6c must use the canonical frozen retrieval config")
    retrieval_config = load_evaluation_config(eval_paths)
    queries = [
        {
            "query_id": unit["revised_claim_id"],
            "response_id": unit["response_id"],
            "text": unit["revised_claim"],
        }
        for unit in units
    ]
    if args.dry_run:
        report = {
            "stage": "B6c_prepare_revised_claim_evidence",
            "split": args.split,
            "query_count": len(queries),
            "retriever": settings["retriever"],
            "top_k": int(settings["evidence_top_k"]),
            "ranking_fields": retrieval_config["ranking_passage_fields"],
            "labels_or_qrels_used_for_ranking": False,
            "output": str(
                paths.revised_claim_evidence(args.split).relative_to(
                    PROJECT_ROOT
                )
            ),
            "dry_run": True,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    passages, _, _, hybrid_rows, embedding_digest = (
        rank_frozen_hybrid_queries(
            PROJECT_ROOT,
            corpus_paths,
            eval_paths,
            retrieval_config,
            queries,
            split_label=f"cove_{args.split}",
        )
    )
    passage_by_id = {
        row["passage_id"]: row for row in passages
    }
    ranked: dict[str, list[dict[str, Any]]] = defaultdict(list)
    top_k = int(settings["evidence_top_k"])
    for row in hybrid_rows:
        if int(row["rank"]) <= top_k:
            ranked[row["query_id"]].append(row)
    evidence_rows: list[dict[str, Any]] = []
    for unit in units:
        claim_id = unit["revised_claim_id"]
        run_rows = sorted(
            ranked.get(claim_id, []),
            key=lambda row: int(row["rank"]),
        )
        if [int(row["rank"]) for row in run_rows] != list(
            range(1, top_k + 1)
        ):
            raise ValueError(
                f"Frozen Hybrid lacks top-{top_k} for {claim_id}"
            )
        items: list[dict[str, Any]] = []
        visible: list[str] = []
        for run_row in run_rows:
            passage = passage_by_id[run_row["passage_id"]]
            text = str(passage["text"]).strip()
            rank = int(run_row["rank"])
            visible.append(
                f"Passage {rank} text (JSON-encoded): "
                f"{json.dumps(text, ensure_ascii=False)}"
            )
            items.append(
                {
                    "rank": rank,
                    "passage_id": run_row["passage_id"],
                    "doc_id": run_row["doc_id"],
                    "retrieval_score": float(run_row["score"]),
                    "text": text,
                    "text_sha256": sha256_text(text),
                }
            )
        normalized = "\n\n".join(visible)
        evidence_rows.append(
            {
                "schema_version": "cove_revised_claim_evidence_v1",
                **unit,
                "retriever": settings["retriever"],
                "top_k": top_k,
                "items": items,
                "normalized_text": normalized,
                "normalized_sha256": sha256_text(normalized),
                "embedding_model": retrieval_config["dense"]["model"],
                "embedding_model_digest": embedding_digest,
                "retrieval_config_sha256": canonical_json_hash(
                    retrieval_config
                ),
                "passages_sha256": sha256_file(corpus_paths.passages),
                "b6a_results_sha256": sha256_file(
                    paths.revised_claim_extraction_results(args.split)
                ),
                "b6b_completion_gate_sha256": sha256_file(
                    paths.revised_claim_alignment_results(args.split)
                ),
                "model_visible_fields": [
                    "revised_claim",
                    "passage_text",
                ],
                "ranking_excluded_fields": [
                    "human_label",
                    "gold_evidence",
                    "qrels",
                    "b6b_alignment_relation",
                ],
                "created_at": utc_now(),
            }
        )
    validate_revised_claim_evidence(
        evidence_rows,
        units,
        config,
    )
    output_path = paths.revised_claim_evidence(args.split)
    atomic_write_jsonl(output_path, evidence_rows)
    report = {
        "stage": "B6c_prepare_revised_claim_evidence",
        "status": "complete",
        "split": args.split,
        "query_count": len(evidence_rows),
        "retriever": settings["retriever"],
        "top_k": top_k,
        "embedding_model": retrieval_config["dense"]["model"],
        "embedding_model_digest": embedding_digest,
        "labels_or_qrels_used_for_ranking": False,
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "output_sha256": sha256_file(output_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_revised_factuality_prompt(
    template: str,
    unit: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    return template.replace(
        "{revised_claim_json}",
        json.dumps(unit["revised_claim"], ensure_ascii=False),
    ).replace(
        "{retrieved_evidence_text}",
        evidence["normalized_text"],
    )


def build_revised_factuality_run_config(
    config: dict[str, Any],
    *,
    split: str,
    prompt_path: Path,
    evidence_path: Path,
    b6a_path: Path,
    b6b_path: Path,
    model_digest: str,
) -> dict[str, Any]:
    settings = config["revised_claim_factuality"]
    schema = make_revised_factuality_output_schema(config)
    payload = {
        "experiment": config["experiment"],
        "stage": "B6c_revised_claim_factuality",
        "split": split,
        "evaluation_unit": "one_model_call_per_revised_claim",
        "retriever": settings["retriever"],
        "evidence_top_k": int(settings["evidence_top_k"]),
        "model": settings["model"],
        "model_digest": model_digest,
        "temperature": float(settings["temperature"]),
        "seed": int(settings["seed"]),
        "num_predict": int(settings["num_predict"]),
        "think": bool(settings["think"]),
        "timeout_seconds": float(settings["timeout_seconds"]),
        "max_retries": int(settings["max_retries"]),
        "max_consecutive_request_errors": int(
            settings["max_consecutive_request_errors"]
        ),
        "prediction_labels": settings["prediction_labels"],
        "prompt_version": settings["prompt_version"],
        "prompt_sha256": sha256_file(prompt_path),
        "result_schema_version": settings["result_schema_version"],
        "output_schema_version": settings["output_schema_version"],
        "output_schema_sha256": canonical_json_hash(schema),
        "evidence_sha256": sha256_file(evidence_path),
        "b6a_results_sha256": sha256_file(b6a_path),
        "b6b_completion_gate_sha256": sha256_file(b6b_path),
        "model_input_fields": config["leakage_policy"][
            "revised_claim_factuality_model_input_fields"
        ],
        "withheld_fields": config["leakage_policy"][
            "revised_claim_factuality_withheld_fields"
        ],
    }
    return {**payload, "run_fingerprint": canonical_json_hash(payload)}


def create_revised_factuality_result_base(
    unit: dict[str, Any],
    evidence: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "result_schema_version": run_config["result_schema_version"],
        "revised_claim_id": unit["revised_claim_id"],
        "response_id": unit["response_id"],
        "source_record_index": unit["source_record_index"],
        "split": unit["split"],
        "revised_claim": unit["revised_claim"],
        "revised_claim_sha256": unit["revised_claim_sha256"],
        "stage": run_config["stage"],
        "evaluation_unit": run_config["evaluation_unit"],
        "retriever": run_config["retriever"],
        "evidence_top_k": run_config["evidence_top_k"],
        "evidence_items": evidence["items"],
        "evidence_normalized_sha256": evidence["normalized_sha256"],
        "model_input_fields": run_config["model_input_fields"],
        "withheld_fields": run_config["withheld_fields"],
        "b6a_run_fingerprint": unit["b6a_run_fingerprint"],
        "b6a_results_sha256": run_config["b6a_results_sha256"],
        "b6b_completion_gate_sha256": run_config[
            "b6b_completion_gate_sha256"
        ],
        "evidence_sha256": run_config["evidence_sha256"],
        "model": run_config["model"],
        "model_digest": run_config["model_digest"],
        "temperature": run_config["temperature"],
        "seed": run_config["seed"],
        "num_predict": run_config["num_predict"],
        "think": run_config["think"],
        "timeout_seconds": run_config["timeout_seconds"],
        "max_retries": run_config["max_retries"],
        "max_consecutive_request_errors": run_config[
            "max_consecutive_request_errors"
        ],
        "prompt_version": run_config["prompt_version"],
        "prompt_sha256": run_config["prompt_sha256"],
        "output_schema_version": run_config["output_schema_version"],
        "output_schema_sha256": run_config["output_schema_sha256"],
        "run_fingerprint": run_config["run_fingerprint"],
    }


def process_revised_factuality_unit(
    unit: dict[str, Any],
    evidence: dict[str, Any],
    template: str,
    config: dict[str, Any],
    run_config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    result = create_revised_factuality_result_base(
        unit,
        evidence,
        run_config,
    )
    prompt = build_revised_factuality_prompt(template, unit, evidence)
    raw_output: str | None = None
    metadata: dict[str, Any] = {}
    request_error: Exception | None = None
    attempts = 0
    started = time.perf_counter()
    for attempt in range(run_config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw_output, metadata = call_ollama(
                client,
                run_config,
                make_revised_factuality_output_schema(config),
                prompt,
            )
            request_error = None
            break
        except Exception as error:
            request_error = error
            if attempt < run_config["max_retries"]:
                time.sleep(1.0)
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
        parsed = parse_revised_factuality_output(raw_output or "")
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
    result.update({"status": "ok", **parsed, "error": None})
    return result


def validate_existing_revised_factuality(
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    unit_by_id = {unit["revised_claim_id"]: unit for unit in units}
    seen: set[str] = set()
    for row in rows:
        claim_id = row.get("revised_claim_id")
        if claim_id not in unit_by_id:
            raise ValueError(
                f"B6c output contains unexpected revised_claim_id: {claim_id}"
            )
        if claim_id in seen:
            raise ValueError(f"Duplicate revised_claim_id in B6c: {claim_id}")
        seen.add(claim_id)
        expected_base = create_revised_factuality_result_base(
            unit_by_id[claim_id],
            evidence_by_id[claim_id],
            run_config,
        )
        for key, expected in expected_base.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"Existing B6c output is incompatible for "
                    f"{claim_id}: {key}"
                )
        status = row.get("status")
        if status not in {"ok", "request_error", "parse_error"}:
            raise ValueError(f"Invalid B6c status for {claim_id}: {status}")
        if status == "ok":
            recovery = row.get("format_recovery")
            allow_word_limit_warning = (
                isinstance(recovery, dict)
                and recovery.get("applied") is True
                and recovery.get("method")
                == "b6c_rationale_word_limit_warning_v1"
            )
            parsed = parse_revised_factuality_output(
                json.dumps(
                    {
                        "prediction": row.get("prediction"),
                        "confidence": row.get("confidence"),
                        "rationale": row.get("rationale"),
                    },
                    ensure_ascii=False,
                ),
                allow_word_limit_warning=allow_word_limit_warning,
            )
            for key, expected in parsed.items():
                if row.get(key) != expected:
                    raise ValueError(
                        f"Non-canonical B6c value for {claim_id}: {key}"
                    )
            if row.get("error") is not None:
                raise ValueError(
                    f"Successful B6c row has an error for {claim_id}"
                )
        else:
            for key in ("prediction", "confidence", "rationale"):
                if row.get(key) is not None:
                    raise ValueError(
                        f"Technical B6c failure has {key} for {claim_id}"
                    )


def revised_prediction_state(
    revised_claim_ids: list[str],
    prediction_by_id: dict[str, dict[str, Any]],
) -> str:
    if not revised_claim_ids:
        return "NO_REVISED_CLAIM"
    rows = [prediction_by_id.get(claim_id) for claim_id in revised_claim_ids]
    if any(row is None or row.get("status") != "ok" for row in rows):
        return "INCOMPLETE"
    labels = {str(row["prediction"]) for row in rows if row is not None}
    if "NON_FACTUAL" in labels:
        return "CONTAINS_NON_FACTUAL"
    if labels == {"FACTUAL"}:
        return "ALL_FACTUAL"
    return "UNKNOWN_OR_MIXED"


def classify_initial_outcome(
    human_label: str,
    relation: str,
    revised_state: str,
) -> str:
    if relation == "PRESENT_UNEXTRACTED":
        return "UNRESOLVED_EXTRACTION_OMISSION"
    if human_label == "UNKNOWN":
        return "UNRESOLVED_UNKNOWN_INITIAL_ANCHOR"
    if relation == "ABSENT":
        return (
            "FACTUAL_DELETED_CANDIDATE"
            if human_label == "FACTUAL"
            else "ERROR_REMOVED_BY_DELETION_CANDIDATE"
        )
    if revised_state in {"INCOMPLETE", "UNKNOWN_OR_MIXED"}:
        return "UNRESOLVED_REVISED_FACTUALITY"
    if revised_state == "ALL_FACTUAL":
        if human_label == "FACTUAL":
            return (
                "FACTUAL_RETAINED_CANDIDATE"
                if relation == "EQUIVALENT"
                else "FACTUAL_PRESERVED_AFTER_CHANGE_CANDIDATE"
            )
        return (
            "SILVER_LABEL_DISAGREEMENT_CANDIDATE"
            if relation == "EQUIVALENT"
            else "ERROR_CORRECTED_CANDIDATE"
        )
    if revised_state == "CONTAINS_NON_FACTUAL":
        if human_label == "FACTUAL":
            return "FACTUAL_DAMAGED_CANDIDATE"
        return (
            "ERROR_RETAINED_CANDIDATE"
            if relation == "EQUIVALENT"
            else "ERROR_STILL_PRESENT_AFTER_CHANGE_CANDIDATE"
        )
    return "UNRESOLVED_ALIGNMENT_OR_FACTUALITY"


def build_b6c_outcomes(
    paths: Any,
    split: str,
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_by_id = {
        row["revised_claim_id"]: row for row in results
    }
    initial_candidates = load_jsonl(
        paths.initial_transition_candidates(split)
    )
    added_candidates = load_jsonl(paths.added_claim_candidates(split))
    initial_outcomes: list[dict[str, Any]] = []
    for candidate in initial_candidates:
        revised_ids = list(candidate["revised_claim_ids"])
        revised_predictions = []
        for claim_id in revised_ids:
            result = prediction_by_id.get(claim_id)
            revised_predictions.append(
                {
                    "revised_claim_id": claim_id,
                    "status": (
                        None if result is None else result.get("status")
                    ),
                    "prediction": (
                        None if result is None else result.get("prediction")
                    ),
                    "confidence": (
                        None if result is None else result.get("confidence")
                    ),
                    "rationale": (
                        None if result is None else result.get("rationale")
                    ),
                    "format_warnings": (
                        None
                        if result is None
                        else result.get("format_warnings")
                    ),
                    "format_recovery": (
                        None
                        if result is None
                        else result.get("format_recovery")
                    ),
                }
            )
        state = revised_prediction_state(revised_ids, prediction_by_id)
        outcome = classify_initial_outcome(
            candidate["human_label"],
            candidate["relation"],
            state,
        )
        initial_outcomes.append(
            {
                "schema_version": "cove_initial_claim_outcome_v1",
                **candidate,
                "revised_prediction_state": state,
                "revised_predictions": revised_predictions,
                "provisional_outcome": outcome,
                "evaluation_tier": "silver_llm_assisted",
                "audit_status": "development_audit_required",
                "net_gain_eligible": False,
                "interpretation": (
                    "Candidate outcome only: B6b semantic alignment and B6c "
                    "factuality were produced by the same model family."
                ),
            }
        )
    added_outcomes: list[dict[str, Any]] = []
    for candidate in added_candidates:
        result = prediction_by_id.get(candidate["revised_claim_id"])
        prediction = None if result is None else result.get("prediction")
        status = None if result is None else result.get("status")
        if status != "ok" or prediction == "UNKNOWN":
            outcome = "ADDED_CLAIM_UNRESOLVED_CANDIDATE"
        elif prediction == "FACTUAL":
            outcome = "ADDED_FACTUAL_CANDIDATE"
        else:
            outcome = "NEW_ERROR_CANDIDATE"
        added_outcomes.append(
            {
                "schema_version": "cove_added_claim_outcome_v1",
                **candidate,
                "b6c_status": status,
                "b6c_prediction": prediction,
                "b6c_confidence": (
                    None if result is None else result.get("confidence")
                ),
                "b6c_rationale": (
                    None if result is None else result.get("rationale")
                ),
                "b6c_format_warnings": (
                    None
                    if result is None
                    else result.get("format_warnings")
                ),
                "b6c_format_recovery": (
                    None
                    if result is None
                    else result.get("format_recovery")
                ),
                "provisional_outcome": outcome,
                "evaluation_tier": "silver_llm_assisted",
                "audit_status": "development_audit_required",
                "net_gain_eligible": False,
            }
        )
    return initial_outcomes, added_outcomes


def revised_factuality_summary(
    units: list[dict[str, Any]],
    results: list[dict[str, Any]],
    initial_outcomes: list[dict[str, Any]],
    added_outcomes: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    result_by_id = {row["revised_claim_id"]: row for row in results}
    status_counts = Counter(
        result_by_id.get(
            unit["revised_claim_id"], {}
        ).get("status", "missing")
        for unit in units
    )
    successful = [
        result_by_id[unit["revised_claim_id"]]
        for unit in units
        if result_by_id.get(
            unit["revised_claim_id"], {}
        ).get("status") == "ok"
    ]
    prediction_counts = Counter(
        row["prediction"] for row in successful
    )
    confidences = [
        float(row["confidence"])
        for row in successful
        if isinstance(row.get("confidence"), (int, float))
    ]
    latencies = [
        float(row["latency_seconds"])
        for row in successful
        if isinstance(row.get("latency_seconds"), (int, float))
    ]
    complete = len(successful) == len(units)
    recovered = [
        row
        for row in successful
        if isinstance(row.get("format_recovery"), dict)
        and row["format_recovery"].get("applied") is True
    ]
    format_warning_counts = Counter(
        warning
        for row in successful
        for warning in row.get("format_warnings", [])
    )
    return {
        "schema_version": "cove_revised_claim_factuality_summary_v1",
        "experiment": "Experiment B: CoVe mechanism evaluation",
        "stage": "B6c_revised_claim_factuality",
        "split": split,
        "completion_status": "complete" if complete else "incomplete",
        "selected_responses": len(
            {unit["response_id"] for unit in units}
        ),
        "selected_revised_claims": len(units),
        "successful_revised_claims": len(successful),
        "status_counts": dict(sorted(status_counts.items())),
        "format_only_recovery": {
            "recovered_revised_claims": len(recovered),
            "model_recalled": False,
            "prediction_or_confidence_changed": False,
            "warning_counts": dict(sorted(format_warning_counts.items())),
        },
        "prediction_counts": {
            label: prediction_counts[label]
            for label in ("FACTUAL", "NON_FACTUAL", "UNKNOWN")
        },
        "self_reported_confidence": {
            "mean": (
                round(statistics.mean(confidences), 4)
                if confidences
                else None
            ),
            "median": (
                round(statistics.median(confidences), 4)
                if confidences
                else None
            ),
        },
        "latency_seconds": {
            "total": round(sum(latencies), 4) if latencies else None,
            "mean": (
                round(statistics.mean(latencies), 4)
                if latencies
                else None
            ),
            "median": (
                round(statistics.median(latencies), 4)
                if latencies
                else None
            ),
        },
        "provisional_initial_outcome_counts": dict(
            sorted(
                Counter(
                    row["provisional_outcome"]
                    for row in initial_outcomes
                ).items()
            )
        ),
        "provisional_added_outcome_counts": dict(
            sorted(
                Counter(
                    row["provisional_outcome"]
                    for row in added_outcomes
                ).items()
            )
        ),
        "evaluation_design": {
            "revised_claim_scope": "all_B6a_revised_claims",
            "evidence": "frozen_hybrid_rrf_top5",
            "query_fields": ["revised_claim_text"],
            "gold_or_qrels_used_for_ranking": False,
            "verifier_context": [
                "revised_claim_text",
                "retrieved_passage_text",
            ],
            "same_model_family_as_cove": True,
            "evaluation_tier": "silver_llm_assisted",
        },
        "net_factual_gain_status": (
            "not_reported_pending_alignment_and_verifier_audit"
        ),
        "interpretation_notes": [
            "B6c evaluates all revised claims, not only B6b MODIFIED/PARTIAL/ADDED candidates.",
            "B6b alignment errors can still change the provisional initial-claim outcome.",
            "B6c uses the same Qwen model family as CoVe; contextual independence is not model independence.",
            "Prediction counts and candidate outcomes are diagnostic silver results, not human-gold factuality.",
            "No net factual gain is reported from unaudited silver alignment and factuality labels.",
        ],
    }


def build_revised_factuality_markdown(summary: dict[str, Any]) -> str:
    confidence = summary["self_reported_confidence"]
    latency = summary["latency_seconds"]
    lines = [
        "# Experiment B — B6c Revised-Claim Factuality",
        "",
        f"- Split: `{summary['split']}`",
        f"- Completion: **{summary['completion_status']}**",
        (
            "- Revised claims verified: "
            f"{summary['successful_revised_claims']}/"
            f"{summary['selected_revised_claims']}"
        ),
        "- Evidence: frozen Hybrid RRF top-5 passages",
        "- Evaluation tier: **LLM-assisted silver**",
        "- Net factual gain: **not reported pending audit**",
        "",
        "## Revised-claim predictions",
        "",
        "| Prediction | Count |",
        "|---|---:|",
    ]
    for label in ("FACTUAL", "NON_FACTUAL", "UNKNOWN"):
        lines.append(
            f"| `{label}` | {summary['prediction_counts'][label]} |"
        )
    lines.extend(
        [
            "",
            "## Technical summary",
            "",
            "| Measure | Value |",
            "|---|---:|",
            f"| Mean self-reported confidence | {confidence['mean']} |",
            f"| Median self-reported confidence | {confidence['median']} |",
            f"| Mean latency, seconds | {latency['mean']} |",
            f"| Median latency, seconds | {latency['median']} |",
            f"| Format-only recovered claims | "
            f"{summary['format_only_recovery']['recovered_revised_claims']} |",
            "",
            "## Provisional initial-claim outcomes",
            "",
            "| Outcome candidate | Count |",
            "|---|---:|",
        ]
    )
    for outcome, count in summary[
        "provisional_initial_outcome_counts"
    ].items():
        lines.append(f"| `{outcome}` | {count} |")
    lines.extend(
        [
            "",
            "## Provisional added-claim outcomes",
            "",
            "| Outcome candidate | Count |",
            "|---|---:|",
        ]
    )
    for outcome, count in summary[
        "provisional_added_outcome_counts"
    ].items():
        lines.append(f"| `{outcome}` | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Every B6a revised claim is checked against evidence retrieved by "
            "the frozen Experiment A Hybrid RRF configuration. Query ranking "
            "uses revised-claim text only and does not use qrels, human labels, "
            "gold evidence, or B6b relation labels.",
            "",
            "The outputs remain silver labels: the same Qwen model family "
            "generated CoVe outputs, performed B6b alignment, and judges B6c "
            "factuality in separate stateless calls. Candidate corrections, "
            "harms, deletions, and new errors therefore require audit before "
            "they support a formal net factual gain.",
            "",
        ]
    )
    return "\n".join(lines)


def write_revised_factuality_artifacts(
    paths: Any,
    split: str,
    summary: dict[str, Any],
    initial_outcomes: list[dict[str, Any]],
    added_outcomes: list[dict[str, Any]],
) -> None:
    atomic_write_jsonl(paths.initial_claim_outcomes(split), initial_outcomes)
    if added_outcomes:
        atomic_write_jsonl(paths.added_claim_outcomes(split), added_outcomes)
    else:
        atomic_write_text(paths.added_claim_outcomes(split), "")
    atomic_write_json(
        paths.revised_claim_factuality_summary_json(split),
        summary,
    )
    atomic_write_text(
        paths.revised_claim_factuality_summary_markdown(split),
        build_revised_factuality_markdown(summary),
    )


def run_revised_claim_factuality(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    units = revised_factuality_units(paths, config, args.split)
    evidence_path = paths.revised_claim_evidence(args.split)
    if not evidence_path.exists():
        raise FileNotFoundError(
            "Prepare frozen revised-claim evidence before B6c: "
            f"{evidence_path}"
        )
    evidence_rows = load_jsonl(evidence_path)
    validate_revised_claim_evidence(evidence_rows, units, config)
    evidence_by_id = {
        row["revised_claim_id"]: row for row in evidence_rows
    }
    prompt_path, template = load_revised_factuality_prompt(config)
    if args.dry_run:
        run_config = build_revised_factuality_run_config(
            config,
            split=args.split,
            prompt_path=prompt_path,
            evidence_path=evidence_path,
            b6a_path=paths.revised_claim_extraction_results(args.split),
            b6b_path=paths.revised_claim_alignment_results(args.split),
            model_digest="DRY_RUN_NOT_CHECKED",
        )
        print("EXPERIMENT B B6C REVISED-CLAIM FACTUALITY DRY RUN")
        print(
            f"Split: {args.split}; independent claim calls: {len(units)}; "
            f"retriever: {run_config['retriever']}; "
            f"top-k: {run_config['evidence_top_k']}"
        )
        print(f"Run fingerprint: {run_config['run_fingerprint']}")
        print(
            "Model input fields: revised claim and frozen Hybrid top-5 "
            "passage text only. No gold fields, qrels, or B6b relation."
        )
        print(f"\n--- Preview: {units[0]['revised_claim_id']} ---")
        print(
            build_revised_factuality_prompt(
                template,
                units[0],
                evidence_by_id[units[0]["revised_claim_id"]],
            )
        )
        print("\nNo Ollama calls were made and no files were written.")
        return 0
    client = Client(
        host=args.ollama_host,
        timeout=float(
            config["revised_claim_factuality"]["timeout_seconds"]
        ),
    )
    model_digest = preflight_revised_factuality_ollama(client, config)
    run_config = build_revised_factuality_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        evidence_path=evidence_path,
        b6a_path=paths.revised_claim_extraction_results(args.split),
        b6b_path=paths.revised_claim_alignment_results(args.split),
        model_digest=model_digest,
    )
    output_path = paths.revised_claim_factuality_results(args.split)
    existing = load_jsonl(output_path) if output_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Output already exists; rerun with --resume: {output_path}"
        )
    validate_existing_revised_factuality(
        existing,
        units,
        evidence_by_id,
        run_config,
    )
    result_by_id = {
        row["revised_claim_id"]: row for row in existing
    }
    pending = [
        unit
        for unit in units
        if result_by_id.get(
            unit["revised_claim_id"], {}
        ).get("status") != "ok"
    ]
    print(
        f"Experiment B B6c: split={args.split}, total={len(units)}, "
        f"retained_ok={len(units) - len(pending)}, pending={len(pending)}",
        flush=True,
    )
    consecutive_request_errors = 0
    max_consecutive = int(
        config["revised_claim_factuality"][
            "max_consecutive_request_errors"
        ]
    )
    order = {
        unit["revised_claim_id"]: index
        for index, unit in enumerate(units)
    }
    for unit in pending:
        claim_id = unit["revised_claim_id"]
        overall_index = order[claim_id] + 1
        print(
            f"[{overall_index}/{len(units)}] {claim_id} verifying ...",
            flush=True,
        )
        result = process_revised_factuality_unit(
            unit,
            evidence_by_id[claim_id],
            template,
            config,
            run_config,
            client,
        )
        result_by_id[claim_id] = result
        ordered = [
            result_by_id[item["revised_claim_id"]]
            for item in units
            if item["revised_claim_id"] in result_by_id
        ]
        atomic_write_jsonl(output_path, ordered)
        if result["status"] == "ok":
            consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(units)}] success, "
                f"{result['prediction']} @ {result['confidence']:.2f}, "
                f"{result['latency_seconds']:.1f}s",
                flush=True,
            )
        else:
            if result["status"] == "request_error":
                consecutive_request_errors += 1
            else:
                consecutive_request_errors = 0
            print(
                f"[{overall_index}/{len(units)}] {result['status']}: "
                f"{result['error']}",
                flush=True,
            )
            if consecutive_request_errors >= max_consecutive:
                print(
                    "Stopping after the configured consecutive request-error "
                    "limit. The same command will resume safely.",
                    file=sys.stderr,
                    flush=True,
                )
                break
    all_results = [
        result_by_id[unit["revised_claim_id"]]
        for unit in units
        if unit["revised_claim_id"] in result_by_id
    ]
    validate_existing_revised_factuality(
        all_results,
        units,
        evidence_by_id,
        run_config,
    )
    initial_outcomes, added_outcomes = build_b6c_outcomes(
        paths,
        args.split,
        all_results,
    )
    summary = revised_factuality_summary(
        units,
        all_results,
        initial_outcomes,
        added_outcomes,
        args.split,
    )
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    summary["evidence_file"] = str(
        evidence_path.relative_to(PROJECT_ROOT)
    )
    write_revised_factuality_artifacts(
        paths,
        args.split,
        summary,
        initial_outcomes,
        added_outcomes,
    )
    print(f"Results: {output_path}", flush=True)
    print(
        "Report: "
        f"{paths.revised_claim_factuality_summary_markdown(args.split)}",
        flush=True,
    )
    return 0 if summary["completion_status"] == "complete" else 2


def analyze_revised_claim_factuality(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    units = revised_factuality_units(paths, config, args.split)
    evidence_path = paths.revised_claim_evidence(args.split)
    output_path = paths.revised_claim_factuality_results(args.split)
    if not evidence_path.exists():
        raise FileNotFoundError(evidence_path)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    evidence_rows = load_jsonl(evidence_path)
    validate_revised_claim_evidence(evidence_rows, units, config)
    evidence_by_id = {
        row["revised_claim_id"]: row for row in evidence_rows
    }
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B6c results do not have one valid model digest")
    prompt_path, _ = load_revised_factuality_prompt(config)
    run_config = build_revised_factuality_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        evidence_path=evidence_path,
        b6a_path=paths.revised_claim_extraction_results(args.split),
        b6b_path=paths.revised_claim_alignment_results(args.split),
        model_digest=next(iter(model_digests)),
    )
    validate_existing_revised_factuality(
        results,
        units,
        evidence_by_id,
        run_config,
    )
    initial_outcomes, added_outcomes = build_b6c_outcomes(
        paths,
        args.split,
        results,
    )
    summary = revised_factuality_summary(
        units,
        results,
        initial_outcomes,
        added_outcomes,
        args.split,
    )
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    summary["evidence_file"] = str(
        evidence_path.relative_to(PROJECT_ROOT)
    )
    write_revised_factuality_artifacts(
        paths,
        args.split,
        summary,
        initial_outcomes,
        added_outcomes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["completion_status"] == "complete" else 2


def recover_revised_claim_factuality_format(
    args: argparse.Namespace,
) -> int:
    """Accept valid B6c JSON whose rationale only exceeds the word limit."""
    paths = paths_for_args(args)
    config = load_config(paths)
    units = revised_factuality_units(paths, config, args.split)
    evidence_path = paths.revised_claim_evidence(args.split)
    output_path = paths.revised_claim_factuality_results(args.split)
    if not evidence_path.exists():
        raise FileNotFoundError(evidence_path)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    evidence_rows = load_jsonl(evidence_path)
    validate_revised_claim_evidence(evidence_rows, units, config)
    evidence_by_id = {
        row["revised_claim_id"]: row for row in evidence_rows
    }
    results = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B6c results do not have one valid model digest")
    prompt_path, _ = load_revised_factuality_prompt(config)
    run_config = build_revised_factuality_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        evidence_path=evidence_path,
        b6a_path=paths.revised_claim_extraction_results(args.split),
        b6b_path=paths.revised_claim_alignment_results(args.split),
        model_digest=next(iter(model_digests)),
    )
    validate_existing_revised_factuality(
        results,
        units,
        evidence_by_id,
        run_config,
    )

    recovered_count = 0
    unrecovered: list[dict[str, str]] = []
    for row in results:
        if row.get("status") != "parse_error":
            continue
        claim_id = str(row["revised_claim_id"])
        original_error = str(row.get("error"))
        try:
            parsed = parse_revised_factuality_output(
                row.get("raw_model_output") or "",
                allow_word_limit_warning=True,
            )
            if parsed.get("format_warnings") != [
                "RATIONALE_WORD_LIMIT_EXCEEDED"
            ]:
                raise ValueError(
                    "Raw output is not the supported rationale-word-limit "
                    "case"
                )
        except Exception as error:
            unrecovered.append(
                {
                    "revised_claim_id": claim_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        row.update(
            {
                "status": "ok",
                **parsed,
                "error": None,
                "format_recovery": {
                    "applied": True,
                    "method": "b6c_rationale_word_limit_warning_v1",
                    "original_status": "parse_error",
                    "original_error": original_error,
                    "raw_model_output_preserved": True,
                    "model_recalled": False,
                    "prediction_changed": False,
                    "confidence_changed": False,
                    "rationale_changed": False,
                    "recovered_at": utc_now(),
                },
            }
        )
        recovered_count += 1
        print(
            f"[recovered] {claim_id}: retained prediction={row['prediction']} "
            "and full rationale; recorded word-limit warning",
            flush=True,
        )

    atomic_write_jsonl(output_path, results)
    validate_existing_revised_factuality(
        results,
        units,
        evidence_by_id,
        run_config,
    )
    initial_outcomes, added_outcomes = build_b6c_outcomes(
        paths,
        args.split,
        results,
    )
    summary = revised_factuality_summary(
        units,
        results,
        initial_outcomes,
        added_outcomes,
        args.split,
    )
    summary["run_fingerprint"] = run_config["run_fingerprint"]
    summary["result_file"] = str(output_path.relative_to(PROJECT_ROOT))
    summary["evidence_file"] = str(
        evidence_path.relative_to(PROJECT_ROOT)
    )
    summary["format_only_recovery"]["unrecovered"] = unrecovered
    write_revised_factuality_artifacts(
        paths,
        args.split,
        summary,
        initial_outcomes,
        added_outcomes,
    )
    print(
        f"B6c format recovery: recovered={recovered_count}, "
        f"unrecovered={len(unrecovered)}, "
        f"completion={summary['completion_status']}",
        flush=True,
    )
    print(f"Results: {output_path}", flush=True)
    print(
        "Report: "
        f"{paths.revised_claim_factuality_summary_markdown(args.split)}",
        flush=True,
    )
    return 0 if summary["completion_status"] == "complete" else 2


def load_validated_b6c_context(
    paths: Any,
    config: dict[str, Any],
    split: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    units = revised_factuality_units(paths, config, split)
    evidence_path = paths.revised_claim_evidence(split)
    result_path = paths.revised_claim_factuality_results(split)
    if not evidence_path.exists():
        raise FileNotFoundError(evidence_path)
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    evidence_rows = load_jsonl(evidence_path)
    validate_revised_claim_evidence(evidence_rows, units, config)
    evidence_by_id = {
        row["revised_claim_id"]: row for row in evidence_rows
    }
    results = load_jsonl(result_path)
    model_digests = {row.get("model_digest") for row in results}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError("B6c results do not have one valid model digest")
    prompt_path, _ = load_revised_factuality_prompt(config)
    run_config = build_revised_factuality_run_config(
        config,
        split=split,
        prompt_path=prompt_path,
        evidence_path=evidence_path,
        b6a_path=paths.revised_claim_extraction_results(split),
        b6b_path=paths.revised_claim_alignment_results(split),
        model_digest=next(iter(model_digests)),
    )
    validate_existing_revised_factuality(
        results,
        units,
        evidence_by_id,
        run_config,
    )
    successful_ids = {
        row["revised_claim_id"]
        for row in results
        if row.get("status") == "ok"
    }
    expected_ids = {unit["revised_claim_id"] for unit in units}
    if successful_ids != expected_ids:
        raise ValueError(
            f"B6c {split} must be technically complete before B6d"
        )
    return units, evidence_rows, results, run_config


def primary_factuality_audit_flags(
    result: dict[str, Any],
) -> list[str]:
    rationale = str(result.get("rationale") or "").casefold()
    prediction = result.get("prediction")
    flags: list[str] = []
    if prediction in {"FACTUAL", "NON_FACTUAL"} and any(
        pattern in rationale
        for pattern in EXPLICIT_INSUFFICIENCY_PATTERNS
    ):
        flags.append("DECISIVE_LABEL_WITH_EXPLICIT_INSUFFICIENCY")
    if prediction == "NON_FACTUAL" and any(
        pattern in rationale
        for pattern in EXPLICIT_FACTUAL_RATIONALE_PATTERNS
    ):
        flags.append("NON_FACTUAL_LABEL_WITH_EXPLICIT_SUPPORT_LANGUAGE")
    return flags


def prepare_primary_factuality_audit(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    units, evidence_rows, results, run_config = load_validated_b6c_context(
        paths,
        config,
        args.split,
    )
    result_by_id = {
        row["revised_claim_id"]: row for row in results
    }
    evidence_by_id = {
        row["revised_claim_id"]: row for row in evidence_rows
    }
    initial_outcomes = load_jsonl(paths.initial_claim_outcomes(args.split))
    added_outcomes = load_jsonl(paths.added_claim_outcomes(args.split))
    consensus_settings = config["revised_factuality_consensus"]
    high_initial = set(
        consensus_settings["high_impact_initial_outcomes"]
    )
    high_added = set(
        consensus_settings["high_impact_added_outcomes"]
    )
    impact_roles: defaultdict[str, set[str]] = defaultdict(set)
    for row in initial_outcomes:
        if row["provisional_outcome"] in high_initial:
            for claim_id in row["revised_claim_ids"]:
                impact_roles[claim_id].add(row["provisional_outcome"])
    for row in added_outcomes:
        if row["provisional_outcome"] in high_added:
            impact_roles[row["revised_claim_id"]].add(
                row["provisional_outcome"]
            )
    audit_rows: list[dict[str, Any]] = []
    for unit in units:
        claim_id = unit["revised_claim_id"]
        result = result_by_id[claim_id]
        evidence = evidence_by_id[claim_id]
        flags = primary_factuality_audit_flags(result)
        document_count = len(
            {item["doc_id"] for item in evidence["items"]}
        )
        audit_rows.append(
            {
                "schema_version": "cove_primary_factuality_audit_v1",
                **unit,
                "primary_prediction": result["prediction"],
                "primary_confidence": result["confidence"],
                "primary_rationale": result["rationale"],
                "deterministic_flags": flags,
                "deterministic_flag_count": len(flags),
                "requires_independent_adjudication": True,
                "high_impact": bool(impact_roles[claim_id]),
                "high_impact_roles": sorted(impact_roles[claim_id]),
                "evidence_passage_count": len(evidence["items"]),
                "evidence_document_count": document_count,
                "evidence_normalized_sha256": evidence[
                    "normalized_sha256"
                ],
                "primary_run_fingerprint": result["run_fingerprint"],
                "audit_method": (
                    "deterministic_rationale_policy_flags_without_relabeling"
                ),
            }
        )
    flag_counts = Counter(
        flag for row in audit_rows for flag in row["deterministic_flags"]
    )
    prediction_counts = Counter(
        row["primary_prediction"] for row in audit_rows
    )
    confidence_counts = Counter(
        str(row["primary_confidence"]) for row in audit_rows
    )
    document_counts = [
        row["evidence_document_count"] for row in audit_rows
    ]
    summary = {
        "schema_version": "cove_primary_factuality_audit_summary_v1",
        "experiment": config["experiment"],
        "stage": "B6d_primary_factuality_audit",
        "split": args.split,
        "status": "complete",
        "claim_count": len(audit_rows),
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "deterministic_flagged_claims": sum(
            bool(row["deterministic_flags"]) for row in audit_rows
        ),
        "deterministic_flag_counts": dict(sorted(flag_counts.items())),
        "high_impact_claims": sum(
            bool(row["high_impact"]) for row in audit_rows
        ),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "evidence_document_diversity": {
            "minimum": min(document_counts),
            "median": statistics.median(document_counts),
            "mean": round(statistics.mean(document_counts), 4),
            "all_five_from_one_document": sum(
                value == 1 for value in document_counts
            ),
        },
        "primary_run_fingerprint": run_config["run_fingerprint"],
        "raw_labels_changed": False,
        "next_stage": "B6d_independent_revised_claim_adjudication",
        "interpretation_notes": [
            "Flags identify explicit policy inconsistencies; they do not relabel the primary B6c result.",
            "All revised claims, not only flagged or high-impact claims, require independent adjudication.",
            "The independent model must not see primary predictions, confidence, rationale, initial labels, B6b relations, or qrels.",
        ],
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("Dry-run only; no audit artifacts were written.")
        return 0
    audit_path = paths.factuality_audit_manifest(args.split)
    atomic_write_jsonl(audit_path, audit_rows)
    summary["audit_manifest"] = str(
        audit_path.relative_to(PROJECT_ROOT)
    )
    summary["audit_manifest_sha256"] = sha256_file(audit_path)
    atomic_write_json(
        paths.factuality_audit_summary_json(args.split),
        summary,
    )
    lines = [
        "# Experiment B — B6d Primary Factuality Audit",
        "",
        f"- Split: `{args.split}`",
        f"- Claims audited: {len(audit_rows)}",
        (
            "- Claims with deterministic policy flags: "
            f"{summary['deterministic_flagged_claims']}"
        ),
        f"- High-impact claims: {summary['high_impact_claims']}",
        "- Raw B6c labels changed: **no**",
        "",
        "## Deterministic flags",
        "",
        "| Flag | Count |",
        "|---|---:|",
    ]
    for flag, count in summary["deterministic_flag_counts"].items():
        lines.append(f"| `{flag}` | {count} |")
    lines.extend(
        [
            "",
            "These flags are conservative diagnostics, not corrected labels. "
            "Every claim proceeds to a blind adjudicator from a different "
            "model family.",
            "",
        ]
    )
    atomic_write_text(
        paths.factuality_audit_summary_markdown(args.split),
        "\n".join(lines),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def make_independent_adjudication_output_schema(
    config: dict[str, Any],
) -> dict[str, Any]:
    labels = list(
        config["independent_revised_claim_adjudication"][
            "passage_relation_labels"
        ]
    )
    if set(labels) != PASSAGE_RELATION_LABELS:
        raise ValueError(
            "Independent adjudication relation labels differ from taxonomy"
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["passage_assessments"],
        "properties": {
            "passage_assessments": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "passage_rank",
                        "relation",
                        "rationale",
                    ],
                    "properties": {
                        "passage_rank": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                        },
                        "relation": {
                            "type": "string",
                            "enum": labels,
                        },
                        "rationale": {
                            "type": "string",
                            "minLength": 3,
                            "maxLength": 220,
                        },
                    },
                },
            }
        },
    }


def load_independent_adjudication_prompt(
    config: dict[str, Any],
) -> tuple[Path, str]:
    settings = config["independent_revised_claim_adjudication"]
    path = PROJECT_ROOT / settings["prompt_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    template = path.read_text(encoding="utf-8")
    found = {
        placeholder
        for placeholder in INDEPENDENT_ADJUDICATION_PLACEHOLDERS
        if placeholder in template
    }
    if found != INDEPENDENT_ADJUDICATION_PLACEHOLDERS:
        raise ValueError(
            "Independent adjudication prompt placeholders are incomplete"
        )
    for placeholder in INDEPENDENT_ADJUDICATION_PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ValueError(
                f"Independent prompt must contain {placeholder} once"
            )
    return path, template


def derive_independent_prediction(
    assessments: list[dict[str, Any]],
) -> tuple[str, str]:
    relations = {row["relation"] for row in assessments}
    has_support = "SUPPORTS" in relations
    has_refute = "REFUTES" in relations
    if has_support and not has_refute:
        return "FACTUAL", "DIRECT_SUPPORT"
    if has_refute and not has_support:
        return "NON_FACTUAL", "DIRECT_REFUTATION"
    if has_support and has_refute:
        return "UNKNOWN", "CONFLICTING_DIRECT_EVIDENCE"
    return "UNKNOWN", "INSUFFICIENT_DIRECT_EVIDENCE"


def parse_independent_adjudication_output(
    raw_output: str,
) -> dict[str, Any]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("Independent adjudication output is empty")
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Independent output is not strict JSON: {error}"
        ) from error
    if not isinstance(parsed, dict) or set(parsed) != {
        "passage_assessments"
    }:
        raise ValueError(
            "Independent output must contain exactly passage_assessments"
        )
    assessments = parsed["passage_assessments"]
    if not isinstance(assessments, list) or len(assessments) != 5:
        raise ValueError("Independent output must contain five assessments")
    ranks = [
        row.get("passage_rank") if isinstance(row, dict) else None
        for row in assessments
    ]
    if (
        any(type(rank) is not int for rank in ranks)
        or set(ranks) != {1, 2, 3, 4, 5}
    ):
        raise ValueError(
            "Independent passage ranks must be unique integers 1 through 5"
        )
    format_warnings: set[str] = set()
    if ranks != [1, 2, 3, 4, 5]:
        format_warnings.add("PASSAGE_ASSESSMENTS_REORDERED")
    assessments = sorted(
        assessments,
        key=lambda row: row["passage_rank"],
    )
    normalized: list[dict[str, Any]] = []
    for expected_rank, row in enumerate(assessments, start=1):
        if not isinstance(row, dict) or set(row) != {
            "passage_rank",
            "relation",
            "rationale",
        }:
            raise ValueError("Invalid independent passage assessment object")
        if row["passage_rank"] != expected_rank:
            raise ValueError(
                "Independent passage assessments must be ordered 1 through 5"
            )
        if row["relation"] not in PASSAGE_RELATION_LABELS:
            raise ValueError(
                f"Invalid passage relation: {row['relation']}"
            )
        if not isinstance(row["rationale"], str):
            raise ValueError("Passage rationale must be a string")
        rationale = " ".join(row["rationale"].split())
        if not 3 <= len(rationale) <= 220:
            raise ValueError(
                "Passage rationale must contain 3 to 220 characters"
            )
        if len(rationale.split()) > 30:
            format_warnings.add("RATIONALE_WORD_LIMIT_EXCEEDED")
        if len(rationale) == 220:
            format_warnings.add("RATIONALE_AT_SCHEMA_MAX_LENGTH")
        normalized.append(
            {
                "passage_rank": expected_rank,
                "relation": row["relation"],
                "rationale": rationale,
            }
        )
    prediction, evidence_status = derive_independent_prediction(normalized)
    output = {
        "passage_assessments": normalized,
        "prediction": prediction,
        "evidence_status": evidence_status,
    }
    if format_warnings:
        output["format_warnings"] = sorted(format_warnings)
    return output


def preflight_independent_adjudicator(
    client: Any,
    config: dict[str, Any],
    *,
    split: str,
    paths: Any,
) -> str:
    settings = config["independent_revised_claim_adjudication"]
    model = settings["model"]
    if any(
        fragment.casefold() in model.casefold()
        for fragment in settings["disallowed_model_name_fragments"]
    ):
        raise ValueError(
            f"Independent adjudicator model is disallowed: {model}"
        )
    try:
        response = client.list()
    except Exception as error:
        raise ConnectionError(
            "Ollama preflight failed before independent outputs changed"
        ) from error
    available: dict[str, str | None] = {}
    for item in response_value(response, "models") or []:
        name = response_value(item, "model") or response_value(item, "name")
        digest = response_value(item, "digest")
        if isinstance(name, str):
            available[name] = digest if isinstance(digest, str) else None
    if model not in available:
        raise ValueError(
            f"Independent model {model!r} is not installed. "
            f"Run: ollama pull {model}"
        )
    digest = available[model]
    if not digest:
        raise ValueError(f"Ollama provided no digest for {model!r}")
    primary_digest = config["revised_claim_factuality"][
        "expected_model_digest"
    ]
    if digest == primary_digest:
        raise ValueError(
            "Independent adjudicator digest matches the primary Qwen model"
        )
    if split == "heldout":
        dev_path = paths.independent_factuality_results("dev")
        if not dev_path.exists():
            raise ValueError(
                "Held-out adjudication requires completed development output"
            )
        dev_rows = load_jsonl(dev_path)
        dev_models = {row.get("model") for row in dev_rows}
        dev_digests = {row.get("model_digest") for row in dev_rows}
        if dev_models != {model} or dev_digests != {digest}:
            raise ValueError(
                "Held-out adjudicator must match the development model/digest"
            )
    return digest


def build_independent_adjudication_prompt(
    template: str,
    unit: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    return template.replace(
        "{revised_claim_json}",
        json.dumps(unit["revised_claim"], ensure_ascii=False),
    ).replace(
        "{retrieved_evidence_text}",
        evidence["normalized_text"],
    )


def build_independent_adjudication_run_config(
    config: dict[str, Any],
    *,
    split: str,
    prompt_path: Path,
    evidence_path: Path,
    audit_path: Path,
    model_digest: str,
) -> dict[str, Any]:
    settings = config["independent_revised_claim_adjudication"]
    schema = make_independent_adjudication_output_schema(config)
    payload = {
        "experiment": config["experiment"],
        "stage": "B6d_independent_revised_claim_adjudication",
        "split": split,
        "evaluation_unit": "one_blind_model_call_per_revised_claim",
        "model": settings["model"],
        "model_digest": model_digest,
        "temperature": float(settings["temperature"]),
        "seed": int(settings["seed"]),
        "num_predict": int(settings["num_predict"]),
        "think": bool(settings["think"]),
        "timeout_seconds": float(settings["timeout_seconds"]),
        "max_retries": int(settings["max_retries"]),
        "max_consecutive_request_errors": int(
            settings["max_consecutive_request_errors"]
        ),
        "passage_relation_labels": settings["passage_relation_labels"],
        "prompt_version": settings["prompt_version"],
        "prompt_sha256": sha256_file(prompt_path),
        "result_schema_version": settings["result_schema_version"],
        "output_schema_version": settings["output_schema_version"],
        "output_schema_sha256": canonical_json_hash(schema),
        "evidence_sha256": sha256_file(evidence_path),
        "audit_manifest_sha256": sha256_file(audit_path),
        "model_input_fields": config["leakage_policy"][
            "independent_adjudication_model_input_fields"
        ],
        "withheld_fields": config["leakage_policy"][
            "independent_adjudication_withheld_fields"
        ],
        "overall_prediction_derivation": (
            "support_only=FACTUAL;refute_only=NON_FACTUAL;"
            "both_or_neither=UNKNOWN"
        ),
    }
    return {**payload, "run_fingerprint": canonical_json_hash(payload)}


def create_independent_adjudication_result_base(
    unit: dict[str, Any],
    evidence: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "result_schema_version": run_config["result_schema_version"],
        "revised_claim_id": unit["revised_claim_id"],
        "response_id": unit["response_id"],
        "source_record_index": unit["source_record_index"],
        "split": unit["split"],
        "revised_claim": unit["revised_claim"],
        "revised_claim_sha256": unit["revised_claim_sha256"],
        "stage": run_config["stage"],
        "evaluation_unit": run_config["evaluation_unit"],
        "evidence_items": evidence["items"],
        "evidence_normalized_sha256": evidence["normalized_sha256"],
        "model_input_fields": run_config["model_input_fields"],
        "withheld_fields": run_config["withheld_fields"],
        "model": run_config["model"],
        "model_digest": run_config["model_digest"],
        "temperature": run_config["temperature"],
        "seed": run_config["seed"],
        "num_predict": run_config["num_predict"],
        "think": run_config["think"],
        "timeout_seconds": run_config["timeout_seconds"],
        "max_retries": run_config["max_retries"],
        "max_consecutive_request_errors": run_config[
            "max_consecutive_request_errors"
        ],
        "prompt_version": run_config["prompt_version"],
        "prompt_sha256": run_config["prompt_sha256"],
        "output_schema_version": run_config["output_schema_version"],
        "output_schema_sha256": run_config["output_schema_sha256"],
        "evidence_sha256": run_config["evidence_sha256"],
        "audit_manifest_sha256": run_config["audit_manifest_sha256"],
        "overall_prediction_derivation": run_config[
            "overall_prediction_derivation"
        ],
        "run_fingerprint": run_config["run_fingerprint"],
    }


def process_independent_adjudication_unit(
    unit: dict[str, Any],
    evidence: dict[str, Any],
    template: str,
    config: dict[str, Any],
    run_config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    result = create_independent_adjudication_result_base(
        unit,
        evidence,
        run_config,
    )
    prompt = build_independent_adjudication_prompt(
        template,
        unit,
        evidence,
    )
    raw_output: str | None = None
    metadata: dict[str, Any] = {}
    request_error: Exception | None = None
    attempts = 0
    started = time.perf_counter()
    for attempt in range(run_config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw_output, metadata = call_ollama(
                client,
                run_config,
                make_independent_adjudication_output_schema(config),
                prompt,
            )
            request_error = None
            break
        except Exception as error:
            request_error = error
            if attempt < run_config["max_retries"]:
                time.sleep(1.0)
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
                "passage_assessments": None,
                "prediction": None,
                "evidence_status": None,
                "error": f"{type(request_error).__name__}: {request_error}",
            }
        )
        return result
    try:
        parsed = parse_independent_adjudication_output(raw_output or "")
    except Exception as error:
        result.update(
            {
                "status": "parse_error",
                "passage_assessments": None,
                "prediction": None,
                "evidence_status": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return result
    result.update({"status": "ok", **parsed, "error": None})
    return result


def validate_existing_independent_adjudications(
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    unit_by_id = {unit["revised_claim_id"]: unit for unit in units}
    seen: set[str] = set()
    for row in rows:
        claim_id = row.get("revised_claim_id")
        if claim_id not in unit_by_id:
            raise ValueError(
                f"Independent output has unexpected claim: {claim_id}"
            )
        if claim_id in seen:
            raise ValueError(
                f"Duplicate independent adjudication: {claim_id}"
            )
        seen.add(claim_id)
        expected_base = create_independent_adjudication_result_base(
            unit_by_id[claim_id],
            evidence_by_id[claim_id],
            run_config,
        )
        for key, expected in expected_base.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"Independent output incompatible for {claim_id}: {key}"
                )
        status = row.get("status")
        if status not in {"ok", "request_error", "parse_error"}:
            raise ValueError(
                f"Invalid independent status for {claim_id}: {status}"
            )
        if status == "ok":
            parsed = parse_independent_adjudication_output(
                json.dumps(
                    {
                        "passage_assessments": row.get(
                            "passage_assessments"
                        )
                    },
                    ensure_ascii=False,
                )
            )
            for key in (
                "passage_assessments",
                "prediction",
                "evidence_status",
            ):
                expected = parsed[key]
                if row.get(key) != expected:
                    raise ValueError(
                        f"Non-canonical independent {key}: {claim_id}"
                    )
            actual_warnings = row.get("format_warnings", [])
            if (
                not isinstance(actual_warnings, list)
                or any(
                    warning not in INDEPENDENT_FORMAT_WARNINGS
                    for warning in actual_warnings
                )
            ):
                raise ValueError(
                    f"Invalid independent format warnings: {claim_id}"
                )
            expected_stored_warnings = {
                warning
                for warning in parsed.get("format_warnings", [])
                if warning != "PASSAGE_ASSESSMENTS_REORDERED"
            }
            if not expected_stored_warnings.issubset(
                set(actual_warnings)
            ):
                raise ValueError(
                    f"Missing independent format warning: {claim_id}"
                )
            if row.get("error") is not None:
                raise ValueError(
                    f"Successful independent row has error: {claim_id}"
                )


def recover_independent_factuality_format(
    args: argparse.Namespace,
) -> int:
    """Recover structurally complete B6d outputs without another model call."""

    paths = paths_for_args(args)
    config = load_config(paths)
    units, evidence_rows, _, _ = load_validated_b6c_context(
        paths,
        config,
        args.split,
    )
    evidence_by_id = {
        row["revised_claim_id"]: row for row in evidence_rows
    }
    audit_path = paths.factuality_audit_manifest(args.split)
    output_path = paths.independent_factuality_results(args.split)
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    rows = load_jsonl(output_path)
    model_digests = {row.get("model_digest") for row in rows}
    if len(model_digests) != 1 or None in model_digests:
        raise ValueError(
            "Independent output does not have one valid model digest"
        )
    prompt_path, _ = load_independent_adjudication_prompt(config)
    run_config = build_independent_adjudication_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        evidence_path=paths.revised_claim_evidence(args.split),
        audit_path=audit_path,
        model_digest=next(iter(model_digests)),
    )
    validate_existing_independent_adjudications(
        rows,
        units,
        evidence_by_id,
        run_config,
    )
    recovered_this_run = 0
    for row in rows:
        if row.get("status") != "parse_error":
            continue
        raw_output = row.get("raw_model_output")
        if not isinstance(raw_output, str) or not raw_output.strip():
            continue
        try:
            parsed = parse_independent_adjudication_output(raw_output)
        except ValueError:
            continue
        warnings = parsed.pop("format_warnings", [])
        row.update(
            {
                **parsed,
                "status": "ok",
                "error": None,
                "format_warnings": warnings,
                "format_recovery": {
                    "method": (
                        "deterministic_rank_sort_and_rationale_limit_warning"
                    ),
                    "model_recalled": False,
                    "raw_model_output_changed": False,
                    "relation_labels_changed": False,
                },
            }
        )
        recovered_this_run += 1
    atomic_write_jsonl(output_path, rows)
    validate_existing_independent_adjudications(
        rows,
        units,
        evidence_by_id,
        run_config,
    )
    status_counts = Counter(row.get("status") for row in rows)
    recovered_rows = [
        row for row in rows if row.get("format_recovery") is not None
    ]
    recovery_warning_counts = Counter(
        warning
        for row in recovered_rows
        for warning in row.get("format_warnings", [])
    )
    summary = {
        "stage": "B6d_independent_factuality_format_recovery",
        "split": args.split,
        "rows": len(rows),
        "recovered_rows_this_run": recovered_this_run,
        "recovered_rows_total": len(recovered_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "recovery_warning_counts": dict(
            sorted(recovery_warning_counts.items())
        ),
        "model_recalled": False,
        "raw_model_outputs_changed": False,
        "relation_labels_changed": False,
        "results": str(output_path.relative_to(PROJECT_ROOT)),
        "next_stage": "analyze-factuality-consensus",
    }
    atomic_write_json(
        paths.independent_factuality_recovery_summary_json(args.split),
        summary,
    )
    lines = [
        "# Experiment B — B6d Independent Factuality Format Recovery",
        "",
        f"- Split: `{args.split}`",
        f"- Result rows: {len(rows)}",
        f"- Recovered rows: {len(recovered_rows)}",
        f"- Recovered on this invocation: {recovered_this_run}",
        f"- Final statuses: `{dict(sorted(status_counts.items()))}`",
        "- Model recalled: **no**",
        "- Raw model output changed: **no**",
        "- Passage relation labels changed: **no**",
        "",
        "## Recovery warnings",
        "",
        "| Warning | Rows |",
        "|---|---:|",
    ]
    for warning, count in sorted(recovery_warning_counts.items()):
        lines.append(f"| `{warning}` | {count} |")
    lines.extend(
        [
            "",
            "Recovery sorts a complete unique rank set and treats the "
            "30-word instruction as an auditable warning. The schema's "
            "3–220 character constraint remains mandatory.",
            "",
        ]
    )
    atomic_write_text(
        paths.independent_factuality_recovery_summary_markdown(args.split),
        "\n".join(lines),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    complete = (
        status_counts.get("ok") == len(units)
        and len(status_counts) == 1
    )
    return 0 if complete else 2


def run_independent_factuality(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    units, evidence_rows, _, _ = load_validated_b6c_context(
        paths,
        config,
        args.split,
    )
    evidence_by_id = {
        row["revised_claim_id"]: row for row in evidence_rows
    }
    audit_path = paths.factuality_audit_manifest(args.split)
    if not audit_path.exists():
        raise FileNotFoundError(
            "Run prepare-factuality-audit before independent adjudication"
        )
    audit_rows = load_jsonl(audit_path)
    if {row["revised_claim_id"] for row in audit_rows} != {
        unit["revised_claim_id"] for unit in units
    }:
        raise ValueError("B6d audit manifest does not cover exact claim set")
    prompt_path, template = load_independent_adjudication_prompt(config)
    settings = config["independent_revised_claim_adjudication"]
    if args.dry_run:
        run_config = build_independent_adjudication_run_config(
            config,
            split=args.split,
            prompt_path=prompt_path,
            evidence_path=paths.revised_claim_evidence(args.split),
            audit_path=audit_path,
            model_digest="DRY_RUN_NOT_CHECKED",
        )
        print("EXPERIMENT B B6D INDEPENDENT ADJUDICATION DRY RUN")
        print(
            f"Split: {args.split}; blind claim calls: {len(units)}; "
            f"model: {settings['model']}"
        )
        print(f"Run fingerprint: {run_config['run_fingerprint']}")
        print(
            "Primary B6c labels/rationales, initial labels, B6b relations, "
            "gold evidence, and qrels are withheld."
        )
        print(f"\n--- Preview: {units[0]['revised_claim_id']} ---")
        print(
            build_independent_adjudication_prompt(
                template,
                units[0],
                evidence_by_id[units[0]["revised_claim_id"]],
            )
        )
        print("\nNo Ollama calls were made and no files were written.")
        return 0
    client = Client(
        host=args.ollama_host,
        timeout=float(settings["timeout_seconds"]),
    )
    model_digest = preflight_independent_adjudicator(
        client,
        config,
        split=args.split,
        paths=paths,
    )
    run_config = build_independent_adjudication_run_config(
        config,
        split=args.split,
        prompt_path=prompt_path,
        evidence_path=paths.revised_claim_evidence(args.split),
        audit_path=audit_path,
        model_digest=model_digest,
    )
    output_path = paths.independent_factuality_results(args.split)
    existing = load_jsonl(output_path) if output_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError(
            f"Output already exists; rerun with --resume: {output_path}"
        )
    validate_existing_independent_adjudications(
        existing,
        units,
        evidence_by_id,
        run_config,
    )
    result_by_id = {
        row["revised_claim_id"]: row for row in existing
    }
    pending = [
        unit
        for unit in units
        if result_by_id.get(
            unit["revised_claim_id"], {}
        ).get("status") != "ok"
    ]
    print(
        f"Experiment B B6d: split={args.split}, total={len(units)}, "
        f"retained_ok={len(units) - len(pending)}, pending={len(pending)}, "
        f"model={settings['model']}",
        flush=True,
    )
    order = {
        unit["revised_claim_id"]: index
        for index, unit in enumerate(units)
    }
    consecutive_request_errors = 0
    max_consecutive = int(settings["max_consecutive_request_errors"])
    for unit in pending:
        claim_id = unit["revised_claim_id"]
        position = order[claim_id] + 1
        print(
            f"[{position}/{len(units)}] {claim_id} independently "
            "adjudicating five passages ...",
            flush=True,
        )
        result = process_independent_adjudication_unit(
            unit,
            evidence_by_id[claim_id],
            template,
            config,
            run_config,
            client,
        )
        result_by_id[claim_id] = result
        ordered = [
            result_by_id[item["revised_claim_id"]]
            for item in units
            if item["revised_claim_id"] in result_by_id
        ]
        atomic_write_jsonl(output_path, ordered)
        if result["status"] == "ok":
            consecutive_request_errors = 0
            relation_counts = Counter(
                item["relation"]
                for item in result["passage_assessments"]
            )
            print(
                f"[{position}/{len(units)}] success, "
                f"{result['prediction']} "
                f"({result['evidence_status']}), "
                f"relations={dict(relation_counts)}, "
                f"{result['latency_seconds']:.1f}s",
                flush=True,
            )
        else:
            if result["status"] == "request_error":
                consecutive_request_errors += 1
            else:
                consecutive_request_errors = 0
            print(
                f"[{position}/{len(units)}] {result['status']}: "
                f"{result['error']}",
                flush=True,
            )
            if consecutive_request_errors >= max_consecutive:
                print(
                    "Stopping after the configured consecutive request-error "
                    "limit. The same command will resume safely.",
                    file=sys.stderr,
                    flush=True,
                )
                break
    all_results = [
        result_by_id[unit["revised_claim_id"]]
        for unit in units
        if unit["revised_claim_id"] in result_by_id
    ]
    validate_existing_independent_adjudications(
        all_results,
        units,
        evidence_by_id,
        run_config,
    )
    status_counts = Counter(
        result_by_id.get(
            unit["revised_claim_id"], {}
        ).get("status", "missing")
        for unit in units
    )
    prediction_counts = Counter(
        row["prediction"]
        for row in all_results
        if row.get("status") == "ok"
    )
    print(
        json.dumps(
            {
                "stage": "B6d_independent_revised_claim_adjudication",
                "split": args.split,
                "status_counts": dict(status_counts),
                "prediction_counts": dict(prediction_counts),
                "results": str(output_path.relative_to(PROJECT_ROOT)),
                "next_stage": "analyze-factuality-consensus",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    complete = (
        status_counts.get("ok") == len(units)
        and len(status_counts) == 1
    )
    return 0 if complete else 2


def cohen_kappa_from_pairs(
    pairs: list[tuple[str, str]],
    labels: tuple[str, ...],
) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(
        left_counts[label] / len(pairs)
        * right_counts[label] / len(pairs)
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def analyze_factuality_consensus(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    units, evidence_rows, primary_results, _ = load_validated_b6c_context(
        paths,
        config,
        args.split,
    )
    audit_path = paths.factuality_audit_manifest(args.split)
    independent_path = paths.independent_factuality_results(args.split)
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    if not independent_path.exists():
        raise FileNotFoundError(independent_path)
    audit_rows = load_jsonl(audit_path)
    independent_rows = load_jsonl(independent_path)
    evidence_by_id = {
        row["revised_claim_id"]: row for row in evidence_rows
    }
    unit_ids = {unit["revised_claim_id"] for unit in units}
    if {row["revised_claim_id"] for row in audit_rows} != unit_ids:
        raise ValueError("Audit manifest claim set mismatch")
    if {row["revised_claim_id"] for row in independent_rows} != unit_ids:
        raise ValueError("Independent result claim set mismatch")
    if any(row.get("status") != "ok" for row in independent_rows):
        raise ValueError(
            "Independent adjudication must be complete before consensus"
        )
    independent_model_digests = {
        row.get("model_digest") for row in independent_rows
    }
    if (
        len(independent_model_digests) != 1
        or None in independent_model_digests
    ):
        raise ValueError(
            "Independent adjudication does not have one valid model digest"
        )
    independent_prompt_path, _ = load_independent_adjudication_prompt(
        config
    )
    independent_run_config = build_independent_adjudication_run_config(
        config,
        split=args.split,
        prompt_path=independent_prompt_path,
        evidence_path=paths.revised_claim_evidence(args.split),
        audit_path=audit_path,
        model_digest=next(iter(independent_model_digests)),
    )
    validate_existing_independent_adjudications(
        independent_rows,
        units,
        evidence_by_id,
        independent_run_config,
    )
    primary_by_id = {
        row["revised_claim_id"]: row for row in primary_results
    }
    audit_by_id = {
        row["revised_claim_id"]: row for row in audit_rows
    }
    independent_by_id = {
        row["revised_claim_id"]: row for row in independent_rows
    }
    settings = config["revised_factuality_consensus"]
    accepted_labels = set(settings["accepted_prediction_labels"])
    consensus_rows: list[dict[str, Any]] = []
    for unit in units:
        claim_id = unit["revised_claim_id"]
        primary = primary_by_id[claim_id]
        audit = audit_by_id[claim_id]
        independent = independent_by_id[claim_id]
        reasons: list[str] = []
        if audit["deterministic_flags"]:
            reasons.append("PRIMARY_POLICY_FLAG")
        if primary["prediction"] != independent["prediction"]:
            reasons.append("MODEL_DISAGREEMENT")
        if independent["prediction"] not in accepted_labels:
            reasons.append("NO_UNCONFLICTED_DIRECT_EVIDENCE")
        accepted = not reasons
        consensus_rows.append(
            {
                "schema_version": (
                    "cove_revised_claim_factuality_consensus_v1"
                ),
                **unit,
                "primary_model": primary["model"],
                "primary_model_digest": primary["model_digest"],
                "primary_prediction": primary["prediction"],
                "primary_confidence": primary["confidence"],
                "primary_rationale": primary["rationale"],
                "primary_policy_flags": audit["deterministic_flags"],
                "independent_model": independent["model"],
                "independent_model_digest": independent["model_digest"],
                "independent_prediction": independent["prediction"],
                "independent_evidence_status": independent[
                    "evidence_status"
                ],
                "independent_passage_assessments": independent[
                    "passage_assessments"
                ],
                "model_agreement": (
                    primary["prediction"] == independent["prediction"]
                ),
                "consensus_status": (
                    "ACCEPTED_DIRECT_AGREEMENT"
                    if accepted
                    else "UNRESOLVED"
                ),
                "consensus_prediction": (
                    primary["prediction"] if accepted else "UNKNOWN"
                ),
                "unresolved_reasons": reasons,
                "high_impact": audit["high_impact"],
                "high_impact_roles": audit["high_impact_roles"],
                "evaluation_tier": (
                    "blind_cross_family_direct_evidence_consensus"
                ),
            }
        )
    pseudo_results = [
        {
            "revised_claim_id": row["revised_claim_id"],
            "status": "ok",
            "prediction": row["consensus_prediction"],
            "confidence": None,
            "rationale": row["consensus_status"],
        }
        for row in consensus_rows
    ]
    initial_outcomes, added_outcomes = build_b6c_outcomes(
        paths,
        args.split,
        pseudo_results,
    )
    for row in initial_outcomes:
        row["schema_version"] = "cove_consensus_initial_claim_outcome_v1"
        row["evaluation_tier"] = (
            "blind_cross_family_direct_evidence_consensus"
        )
        row["audit_status"] = "b6b_alignment_still_requires_gate"
    for row in added_outcomes:
        row["schema_version"] = "cove_consensus_added_claim_outcome_v1"
        row["evaluation_tier"] = (
            "blind_cross_family_direct_evidence_consensus"
        )
        row["audit_status"] = "b6b_alignment_still_requires_gate"
    pairs = [
        (row["primary_prediction"], row["independent_prediction"])
        for row in consensus_rows
    ]
    cross_tab: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for left, right in pairs:
        cross_tab[left][right] += 1
    accepted = [
        row
        for row in consensus_rows
        if row["consensus_status"] == "ACCEPTED_DIRECT_AGREEMENT"
    ]
    high_impact = [row for row in consensus_rows if row["high_impact"]]
    accepted_high = [
        row
        for row in high_impact
        if row["consensus_status"] == "ACCEPTED_DIRECT_AGREEMENT"
    ]
    consensus_coverage = len(accepted) / len(consensus_rows)
    high_impact_coverage = (
        len(accepted_high) / len(high_impact) if high_impact else 1.0
    )
    factuality_gate_passed = (
        consensus_coverage
        >= float(
            settings["minimum_consensus_coverage_for_factuality_gate"]
        )
        and high_impact_coverage
        >= float(
            settings[
                "minimum_high_impact_coverage_for_factuality_gate"
            ]
        )
    )
    unresolved_counts = Counter(
        reason
        for row in consensus_rows
        for reason in row["unresolved_reasons"]
    )
    kappa = cohen_kappa_from_pairs(
        pairs,
        ("FACTUAL", "NON_FACTUAL", "UNKNOWN"),
    )
    summary = {
        "schema_version": "cove_factuality_consensus_summary_v1",
        "experiment": config["experiment"],
        "stage": "B6d_revised_claim_factuality_consensus",
        "split": args.split,
        "completion_status": "complete",
        "claim_count": len(consensus_rows),
        "primary_model": primary_results[0]["model"],
        "primary_model_digest": primary_results[0]["model_digest"],
        "independent_model": independent_rows[0]["model"],
        "independent_model_digest": independent_rows[0]["model_digest"],
        "cross_family_model_requirement_satisfied": (
            primary_results[0]["model_digest"]
            != independent_rows[0]["model_digest"]
        ),
        "raw_agreement_count": sum(
            left == right for left, right in pairs
        ),
        "raw_agreement_rate": round(
            sum(left == right for left, right in pairs) / len(pairs),
            4,
        ),
        "cohen_kappa_without_gold": (
            None if kappa is None else round(kappa, 4)
        ),
        "cross_tab_primary_by_independent": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(cross_tab.items())
        },
        "accepted_consensus_claims": len(accepted),
        "consensus_prediction_counts": dict(
            Counter(
                row["consensus_prediction"] for row in consensus_rows
            )
        ),
        "consensus_coverage": round(consensus_coverage, 4),
        "high_impact_claims": len(high_impact),
        "accepted_high_impact_claims": len(accepted_high),
        "high_impact_consensus_coverage": round(
            high_impact_coverage,
            4,
        ),
        "unresolved_reason_counts": dict(
            sorted(unresolved_counts.items())
        ),
        "provisional_initial_outcome_counts": dict(
            sorted(
                Counter(
                    row["provisional_outcome"]
                    for row in initial_outcomes
                ).items()
            )
        ),
        "provisional_added_outcome_counts": dict(
            sorted(
                Counter(
                    row["provisional_outcome"]
                    for row in added_outcomes
                ).items()
            )
        ),
        "factuality_gate": {
            "minimum_consensus_coverage": settings[
                "minimum_consensus_coverage_for_factuality_gate"
            ],
            "minimum_high_impact_coverage": settings[
                "minimum_high_impact_coverage_for_factuality_gate"
            ],
            "passed": factuality_gate_passed,
        },
        "alignment_gate": {
            "passed": False,
            "status": (
                "pending_independent_B6b_alignment_reliability_gate"
            ),
        },
        "heldout_ready": False,
        "heldout_readiness_reason": (
            "factuality gate result is reported here; B6b alignment "
            "reliability remains a separate required gate"
        ),
        "interpretation_notes": [
            "Cross-model agreement is not accuracy because revised claims lack human-gold labels.",
            "Only direct, unconflicted passage evidence with exact cross-family label agreement is accepted.",
            "Primary policy flags force UNKNOWN even when the models agree.",
            "B6b alignment remains unaudited, so consensus transition outcomes are still provisional.",
        ],
    }
    atomic_write_jsonl(
        paths.factuality_consensus_results(args.split),
        consensus_rows,
    )
    atomic_write_jsonl(
        paths.consensus_initial_outcomes(args.split),
        initial_outcomes,
    )
    if added_outcomes:
        atomic_write_jsonl(
            paths.consensus_added_outcomes(args.split),
            added_outcomes,
        )
    else:
        atomic_write_text(paths.consensus_added_outcomes(args.split), "")
    atomic_write_json(
        paths.factuality_consensus_summary_json(args.split),
        summary,
    )
    lines = [
        "# Experiment B — B6d Cross-Family Factuality Consensus",
        "",
        f"- Split: `{args.split}`",
        f"- Claims: {len(consensus_rows)}",
        f"- Raw model agreement: {summary['raw_agreement_rate']}",
        f"- Cohen's kappa (agreement only): {summary['cohen_kappa_without_gold']}",
        (
            "- Accepted direct consensus: "
            f"{len(accepted)}/{len(consensus_rows)} "
            f"({summary['consensus_coverage']})"
        ),
        (
            "- Accepted high-impact consensus: "
            f"{len(accepted_high)}/{len(high_impact)} "
            f"({summary['high_impact_consensus_coverage']})"
        ),
        (
            "- Development factuality gate: **"
            f"{'PASS' if factuality_gate_passed else 'FAIL'}**"
        ),
        "- Held-out ready: **no — independent alignment gate pending**",
        "",
        "## Primary × independent labels",
        "",
        "| Primary | Independent counts |",
        "|---|---|",
    ]
    for label, counts in summary[
        "cross_tab_primary_by_independent"
    ].items():
        text = json.dumps(counts, sort_keys=True).replace("|", "\\|")
        lines.append(f"| `{label}` | `{text}` |")
    lines.extend(
        [
            "",
            "Agreement is not accuracy. This gate only removes explicit "
            "primary-policy violations, cross-family disagreements, and "
            "claims without unconflicted direct passage evidence. B6b "
            "alignment remains a separate source of transition error.",
            "",
        ]
    )
    atomic_write_text(
        paths.factuality_consensus_summary_markdown(args.split),
        "\n".join(lines),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if factuality_gate_passed else 2


def factuality_calibration_units(
    paths: Any,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the exact 121 matched development claims without model labels."""

    settings = config["factuality_calibration"]
    if settings.get("split") != "dev":
        raise ValueError("B6e calibration must remain development-only")
    query_path = PROJECT_ROOT / settings["frozen_queries_path"]
    gold_path = PROJECT_ROOT / settings["gold_claims_path"]
    split_path = PROJECT_ROOT / settings["split_manifest_path"]
    queries = load_jsonl(query_path)
    gold_rows = load_jsonl(gold_path)
    split_rows = load_jsonl(split_path)
    expected_count = int(settings["expected_claims"])
    expected_responses = int(settings["expected_matched_responses"])
    dev_split = [
        row
        for row in split_rows
        if row.get("split") == "dev"
        and row.get("in_primary_matched_cohort") is True
    ]
    dev_ids = {str(row["claim_id"]) for row in dev_split}
    if len(dev_split) != expected_count or len(dev_ids) != expected_count:
        raise ValueError(
            f"B6e expected {expected_count} unique dev matched claims, "
            f"found rows={len(dev_split)}, unique={len(dev_ids)}"
        )
    if len({str(row["response_id"]) for row in dev_split}) != expected_responses:
        raise ValueError(
            f"B6e expected {expected_responses} matched development responses"
        )
    query_ids = [str(row["query_id"]) for row in queries]
    if len(query_ids) != expected_count or set(query_ids) != dev_ids:
        raise ValueError("Frozen dev query IDs differ from the matched dev set")
    if any(row.get("split") != "dev" for row in queries):
        raise ValueError("B6e frozen query file contains a non-dev query")
    gold_by_id = {
        str(row["claim_id"]): row
        for row in gold_rows
        if str(row.get("claim_id")) in dev_ids
    }
    if set(gold_by_id) != dev_ids:
        raise ValueError("Canonical gold file does not cover exact B6e IDs")
    units: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []
    for query in queries:
        claim_id = str(query["query_id"])
        gold = gold_by_id[claim_id]
        claim_text = str(query["text"]).strip()
        if claim_text != str(gold["gold_claim"]).strip():
            raise ValueError(f"Frozen query/gold claim mismatch: {claim_id}")
        if query.get("ranking_field_policy") != "gold_claim_text_only":
            raise ValueError(f"Unexpected ranking field policy: {claim_id}")
        if gold.get("human_label") not in {"FACTUAL", "NON_FACTUAL"}:
            raise ValueError(f"Non-binary B6e gold label: {claim_id}")
        unit = {
            "claim_id": claim_id,
            "response_id": str(gold["response_id"]),
            "source_record_index": int(gold["source_record_index"]),
            "split": "dev",
            "claim_text": claim_text,
            "claim_sha256": sha256_text(claim_text),
        }
        units.append(unit)
        evaluation_records.append(
            {
                **unit,
                "human_label": gold["human_label"],
            }
        )
    return units, evaluation_records


def validate_factuality_calibration_evidence(
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    expected = {row["claim_id"]: row for row in units}
    top_k = int(config["factuality_calibration"]["evidence_top_k"])
    seen: set[str] = set()
    for row in rows:
        claim_id = row.get("claim_id")
        if claim_id not in expected or claim_id in seen:
            raise ValueError(f"Unexpected/duplicate B6e evidence ID: {claim_id}")
        seen.add(str(claim_id))
        unit = expected[str(claim_id)]
        for field in (
            "response_id",
            "source_record_index",
            "split",
            "claim_text",
            "claim_sha256",
        ):
            if row.get(field) != unit[field]:
                raise ValueError(
                    f"B6e evidence mismatch for {claim_id}: {field}"
                )
        items = row.get("items")
        if not isinstance(items, list) or len(items) != top_k:
            raise ValueError(f"B6e evidence must contain top-{top_k}: {claim_id}")
        if [item.get("rank") for item in items] != list(range(1, top_k + 1)):
            raise ValueError(f"B6e evidence ranks are invalid: {claim_id}")
        for item in items:
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"B6e evidence contains empty text: {claim_id}")
            if item.get("text_sha256") != sha256_text(text):
                raise ValueError(f"B6e evidence hash mismatch: {claim_id}")
        normalized = row.get("normalized_text")
        if (
            not isinstance(normalized, str)
            or row.get("normalized_sha256") != sha256_text(normalized)
        ):
            raise ValueError(f"B6e normalized evidence invalid: {claim_id}")
        forbidden = {
            "human_label",
            "gold_evidence",
            "gold_evidence_text",
            "gold_evidence_stance",
            "qrels",
            "canonical_url",
            "raw_url",
        }
        if forbidden.intersection(row):
            raise ValueError(f"B6e evidence leaks evaluation fields: {claim_id}")
    if seen != set(expected):
        raise ValueError("B6e evidence does not cover the exact dev claim set")


def prepare_factuality_calibration(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    settings = config["factuality_calibration"]
    units, _ = factuality_calibration_units(paths, config)
    corpus_paths = retrieval_paths(PROJECT_ROOT, args.scope)
    run_path = PROJECT_ROOT / settings["frozen_hybrid_run_path"]
    queries_path = PROJECT_ROOT / settings["frozen_queries_path"]
    hybrid_rows = load_jsonl(run_path)
    passages = load_jsonl(corpus_paths.passages)
    passage_by_id = {row["passage_id"]: row for row in passages}
    top_k = int(settings["evidence_top_k"])
    ranked: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unit_ids = {unit["claim_id"] for unit in units}
    for row in hybrid_rows:
        if row.get("query_id") in unit_ids and int(row["rank"]) <= top_k:
            ranked[str(row["query_id"])].append(row)
    evidence_rows: list[dict[str, Any]] = []
    for unit in units:
        claim_id = unit["claim_id"]
        run_rows = sorted(
            ranked.get(claim_id, []),
            key=lambda row: int(row["rank"]),
        )
        if [int(row["rank"]) for row in run_rows] != list(
            range(1, top_k + 1)
        ):
            raise ValueError(f"Frozen Hybrid top-{top_k} incomplete: {claim_id}")
        items: list[dict[str, Any]] = []
        visible: list[str] = []
        for run_row in run_rows:
            passage_id = str(run_row["passage_id"])
            if passage_id not in passage_by_id:
                raise ValueError(f"Unknown frozen passage ID: {passage_id}")
            passage = passage_by_id[passage_id]
            text = str(passage["text"]).strip()
            rank = int(run_row["rank"])
            visible.append(
                f"Passage {rank} text (JSON-encoded): "
                f"{json.dumps(text, ensure_ascii=False)}"
            )
            items.append(
                {
                    "rank": rank,
                    "passage_id": passage_id,
                    "doc_id": str(run_row["doc_id"]),
                    "retrieval_score": float(run_row["score"]),
                    "text": text,
                    "text_sha256": sha256_text(text),
                }
            )
        normalized = "\n\n".join(visible)
        evidence_rows.append(
            {
                "schema_version": "cove_factuality_calibration_evidence_v1",
                **unit,
                "retriever": settings["retriever"],
                "top_k": top_k,
                "items": items,
                "normalized_text": normalized,
                "normalized_sha256": sha256_text(normalized),
                "queries_sha256": sha256_file(queries_path),
                "hybrid_run_sha256": sha256_file(run_path),
                "passages_sha256": sha256_file(corpus_paths.passages),
                "model_visible_fields": config["leakage_policy"][
                    "factuality_calibration_model_input_fields"
                ],
                "withheld_fields": config["leakage_policy"][
                    "factuality_calibration_withheld_fields"
                ],
                "created_at": utc_now(),
            }
        )
    validate_factuality_calibration_evidence(evidence_rows, units, config)
    report = {
        "stage": "B6e_prepare_factuality_calibration",
        "split": "dev",
        "claim_count": len(evidence_rows),
        "response_count": len({row["response_id"] for row in evidence_rows}),
        "retriever": settings["retriever"],
        "top_k": top_k,
        "ranking_reused_without_reranking": True,
        "human_labels_or_qrels_used_for_ranking": False,
        "heldout_touched": False,
        "output": str(
            paths.factuality_calibration_evidence.relative_to(PROJECT_ROOT)
        ),
    }
    if args.dry_run:
        report["dry_run"] = True
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    atomic_write_jsonl(paths.factuality_calibration_evidence, evidence_rows)
    report["output_sha256"] = sha256_file(
        paths.factuality_calibration_evidence
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def load_factuality_calibration_context(
    paths: Any,
    config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    units, evaluation_records = factuality_calibration_units(paths, config)
    evidence_path = paths.factuality_calibration_evidence
    if not evidence_path.exists():
        raise FileNotFoundError(
            "Run prepare-factuality-calibration before model calibration"
        )
    evidence_rows = load_jsonl(evidence_path)
    validate_factuality_calibration_evidence(evidence_rows, units, config)
    return units, evaluation_records, evidence_rows


def build_factuality_calibration_run_config(
    paths: Any,
    config: dict[str, Any],
    *,
    evaluator: str,
    model_digest: str,
) -> dict[str, Any]:
    settings = (
        config["revised_claim_factuality"]
        if evaluator == "qwen"
        else config["independent_revised_claim_adjudication"]
    )
    prompt_path = PROJECT_ROOT / settings["prompt_path"]
    schema = (
        make_revised_factuality_output_schema(config)
        if evaluator == "qwen"
        else make_independent_adjudication_output_schema(config)
    )
    payload = {
        "experiment": config["experiment"],
        "stage": f"B6e_factuality_calibration_{evaluator}",
        "split": "dev",
        "evaluation_unit": "one_blind_model_call_per_gold_dev_claim",
        "evaluator": evaluator,
        "model": settings["model"],
        "model_digest": model_digest,
        "temperature": float(settings["temperature"]),
        "seed": int(settings["seed"]),
        "num_predict": int(settings["num_predict"]),
        "think": bool(settings["think"]),
        "timeout_seconds": float(settings["timeout_seconds"]),
        "max_retries": int(settings["max_retries"]),
        "max_consecutive_request_errors": int(
            settings["max_consecutive_request_errors"]
        ),
        "prompt_version": settings["prompt_version"],
        "prompt_sha256": sha256_file(prompt_path),
        "result_schema_version": f"cove_factuality_calibration_{evaluator}_v1",
        "output_schema_sha256": canonical_json_hash(schema),
        "evidence_sha256": sha256_file(
            paths.factuality_calibration_evidence
        ),
        "model_input_fields": config["leakage_policy"][
            "factuality_calibration_model_input_fields"
        ],
        "withheld_fields": config["leakage_policy"][
            "factuality_calibration_withheld_fields"
        ],
    }
    return {**payload, "run_fingerprint": canonical_json_hash(payload)}


def factuality_calibration_result_base(
    unit: dict[str, Any],
    evidence: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "result_schema_version": run_config["result_schema_version"],
        **unit,
        "stage": run_config["stage"],
        "evaluation_unit": run_config["evaluation_unit"],
        "evaluator": run_config["evaluator"],
        "evidence_items": evidence["items"],
        "evidence_normalized_sha256": evidence["normalized_sha256"],
        "model_input_fields": run_config["model_input_fields"],
        "withheld_fields": run_config["withheld_fields"],
        "model": run_config["model"],
        "model_digest": run_config["model_digest"],
        "temperature": run_config["temperature"],
        "seed": run_config["seed"],
        "num_predict": run_config["num_predict"],
        "think": run_config["think"],
        "timeout_seconds": run_config["timeout_seconds"],
        "max_retries": run_config["max_retries"],
        "prompt_version": run_config["prompt_version"],
        "prompt_sha256": run_config["prompt_sha256"],
        "output_schema_sha256": run_config["output_schema_sha256"],
        "evidence_sha256": run_config["evidence_sha256"],
        "run_fingerprint": run_config["run_fingerprint"],
    }


def validate_factuality_calibration_results(
    rows: list[dict[str, Any]],
    units: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    expected = {unit["claim_id"]: unit for unit in units}
    seen: set[str] = set()
    evaluator = run_config["evaluator"]
    for row in rows:
        claim_id = row.get("claim_id")
        if claim_id not in expected or claim_id in seen:
            raise ValueError(
                f"Unexpected/duplicate B6e {evaluator} result: {claim_id}"
            )
        seen.add(str(claim_id))
        base = factuality_calibration_result_base(
            expected[str(claim_id)],
            evidence_by_id[str(claim_id)],
            run_config,
        )
        for key, value in base.items():
            if row.get(key) != value:
                raise ValueError(
                    f"Incompatible B6e {evaluator} result "
                    f"{claim_id}: {key}"
                )
        status = row.get("status")
        if status not in {"ok", "request_error", "parse_error"}:
            raise ValueError(f"Invalid B6e result status: {claim_id}")
        if status == "ok":
            if evaluator == "qwen":
                parsed = parse_revised_factuality_output(
                    json.dumps(
                        {
                            "prediction": row.get("prediction"),
                            "confidence": row.get("confidence"),
                            "rationale": row.get("rationale"),
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                parsed = parse_independent_adjudication_output(
                    json.dumps(
                        {
                            "passage_assessments": row.get(
                                "passage_assessments"
                            )
                        },
                        ensure_ascii=False,
                    )
                )
            for key in ("prediction",):
                if row.get(key) != parsed.get(key):
                    raise ValueError(
                        f"Non-canonical B6e prediction: {claim_id}"
                    )


def process_factuality_calibration_unit(
    unit: dict[str, Any],
    evidence: dict[str, Any],
    template: str,
    config: dict[str, Any],
    run_config: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    evaluator = run_config["evaluator"]
    result = factuality_calibration_result_base(unit, evidence, run_config)
    prompt = template.replace(
        "{revised_claim_json}",
        json.dumps(unit["claim_text"], ensure_ascii=False),
    ).replace("{retrieved_evidence_text}", evidence["normalized_text"])
    schema = (
        make_revised_factuality_output_schema(config)
        if evaluator == "qwen"
        else make_independent_adjudication_output_schema(config)
    )
    raw_output: str | None = None
    metadata: dict[str, Any] = {}
    request_error: Exception | None = None
    attempts = 0
    started = time.perf_counter()
    for attempt in range(run_config["max_retries"] + 1):
        attempts = attempt + 1
        try:
            raw_output, metadata = call_ollama(
                client,
                run_config,
                schema,
                prompt,
            )
            request_error = None
            break
        except Exception as error:
            request_error = error
            if attempt < run_config["max_retries"]:
                time.sleep(1.0)
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
                "error": f"{type(request_error).__name__}: {request_error}",
            }
        )
        return result
    try:
        parsed = (
            parse_revised_factuality_output(raw_output or "")
            if evaluator == "qwen"
            else parse_independent_adjudication_output(raw_output or "")
        )
    except Exception as error:
        result.update(
            {
                "status": "parse_error",
                "prediction": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return result
    result.update({"status": "ok", **parsed, "error": None})
    return result


def run_factuality_calibration(
    args: argparse.Namespace,
    *,
    evaluator: str,
) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    units, _, evidence_rows = load_factuality_calibration_context(
        paths,
        config,
    )
    evidence_by_id = {row["claim_id"]: row for row in evidence_rows}
    if evaluator == "qwen":
        prompt_path, template = load_revised_factuality_prompt(config)
        settings = config["revised_claim_factuality"]
        output_path = paths.factuality_calibration_primary_results
    else:
        prompt_path, template = load_independent_adjudication_prompt(config)
        settings = config["independent_revised_claim_adjudication"]
        output_path = paths.factuality_calibration_independent_results
    if args.dry_run:
        run_config = build_factuality_calibration_run_config(
            paths,
            config,
            evaluator=evaluator,
            model_digest="DRY_RUN_NOT_CHECKED",
        )
        first = units[0]
        prompt = template.replace(
            "{revised_claim_json}",
            json.dumps(first["claim_text"], ensure_ascii=False),
        ).replace(
            "{retrieved_evidence_text}",
            evidence_by_id[first["claim_id"]]["normalized_text"],
        )
        print(
            f"B6e {evaluator} dry run: 121 dev claims; "
            f"model={settings['model']}; heldout untouched"
        )
        print(f"Prompt: {prompt_path.relative_to(PROJECT_ROOT)}")
        print(f"Run fingerprint: {run_config['run_fingerprint']}")
        print(f"\n--- Preview: {first['claim_id']} ---\n{prompt}")
        print("\nNo Ollama calls were made and no files were written.")
        return 0
    client = Client(
        host=args.ollama_host,
        timeout=float(settings["timeout_seconds"]),
    )
    if evaluator == "qwen":
        model_digest = preflight_revised_factuality_ollama(client, config)
    else:
        model_digest = preflight_independent_adjudicator(
            client,
            config,
            split="dev",
            paths=paths,
        )
        prior_path = paths.independent_factuality_results("dev")
        if prior_path.exists():
            prior = load_jsonl(prior_path)
            prior_models = {row.get("model") for row in prior}
            prior_digests = {row.get("model_digest") for row in prior}
            if prior_models != {settings["model"]} or prior_digests != {
                model_digest
            }:
                raise ValueError(
                    "B6e Llama model/digest differs from completed B6d dev"
                )
    run_config = build_factuality_calibration_run_config(
        paths,
        config,
        evaluator=evaluator,
        model_digest=model_digest,
    )
    existing = load_jsonl(output_path) if output_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError(f"Use --resume for existing output: {output_path}")
    validate_factuality_calibration_results(
        existing,
        units,
        evidence_by_id,
        run_config,
    )
    result_by_id = {row["claim_id"]: row for row in existing}
    pending = [
        unit
        for unit in units
        if result_by_id.get(unit["claim_id"], {}).get("status") != "ok"
    ]
    print(
        f"Experiment B B6e {evaluator}: total={len(units)}, "
        f"retained_ok={len(units) - len(pending)}, pending={len(pending)}, "
        f"model={settings['model']}",
        flush=True,
    )
    order = {unit["claim_id"]: index for index, unit in enumerate(units)}
    consecutive_errors = 0
    max_consecutive = int(settings["max_consecutive_request_errors"])
    for unit in pending:
        claim_id = unit["claim_id"]
        position = order[claim_id] + 1
        action = (
            "classifying claim"
            if evaluator == "qwen"
            else "adjudicating five passages"
        )
        print(
            f"[{position}/{len(units)}] {claim_id} {action} ...",
            flush=True,
        )
        result = process_factuality_calibration_unit(
            unit,
            evidence_by_id[claim_id],
            template,
            config,
            run_config,
            client,
        )
        result_by_id[claim_id] = result
        atomic_write_jsonl(
            output_path,
            [
                result_by_id[item["claim_id"]]
                for item in units
                if item["claim_id"] in result_by_id
            ],
        )
        if result["status"] == "ok":
            consecutive_errors = 0
            detail = result["prediction"]
            if evaluator == "llama":
                counts = Counter(
                    row["relation"]
                    for row in result["passage_assessments"]
                )
                detail += f", relations={dict(counts)}"
            print(
                f"[{position}/{len(units)}] success, {detail}, "
                f"{result['latency_seconds']:.1f}s",
                flush=True,
            )
        else:
            consecutive_errors = (
                consecutive_errors + 1
                if result["status"] == "request_error"
                else 0
            )
            print(
                f"[{position}/{len(units)}] {result['status']}: "
                f"{result['error']}",
                flush=True,
            )
            if consecutive_errors >= max_consecutive:
                print(
                    "Stopping at the configured request-error limit; rerun "
                    "the same command to resume.",
                    file=sys.stderr,
                    flush=True,
                )
                break
    rows = [
        result_by_id[unit["claim_id"]]
        for unit in units
        if unit["claim_id"] in result_by_id
    ]
    validate_factuality_calibration_results(
        rows,
        units,
        evidence_by_id,
        run_config,
    )
    statuses = Counter(
        result_by_id.get(unit["claim_id"], {}).get("status", "missing")
        for unit in units
    )
    print(
        json.dumps(
            {
                "stage": run_config["stage"],
                "claim_count": len(units),
                "status_counts": dict(statuses),
                "prediction_counts": dict(
                    Counter(
                        row["prediction"]
                        for row in rows
                        if row.get("status") == "ok"
                    )
                ),
                "heldout_touched": False,
                "output": str(output_path.relative_to(PROJECT_ROOT)),
                "next_stage": (
                    "run-factuality-calibration-independent"
                    if evaluator == "qwen"
                    else "analyze-factuality-calibration"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    complete = statuses == Counter({"ok": len(units)})
    return 0 if complete else 2


def derive_calibrated_independent_prediction(
    result: dict[str, Any],
    *,
    policy: str,
) -> tuple[str, dict[str, int]]:
    assessments = result["passage_assessments"]
    item_by_rank = {
        int(item["rank"]): item for item in result["evidence_items"]
    }
    supports = [
        row for row in assessments if row["relation"] == "SUPPORTS"
    ]
    refutes = [
        row for row in assessments if row["relation"] == "REFUTES"
    ]
    support_docs = {
        item_by_rank[int(row["passage_rank"])]["doc_id"]
        for row in supports
    }
    refute_docs = {
        item_by_rank[int(row["passage_rank"])]["doc_id"]
        for row in refutes
    }
    counts = {
        "support_count": len(supports),
        "refute_count": len(refutes),
        "support_doc_count": len(support_docs),
        "refute_doc_count": len(refute_docs),
    }
    if policy == "any_direct":
        factual = counts["support_count"] >= 1 and not refutes
    elif policy == "corroborated_support":
        factual = counts["support_count"] >= 2 and not refutes
    elif policy == "multi_document_support":
        factual = counts["support_doc_count"] >= 2 and not refutes
    else:
        raise ValueError(f"Unknown B6e aggregation policy: {policy}")
    non_factual = counts["refute_count"] >= 1 and not supports
    if factual:
        return "FACTUAL", counts
    if non_factual:
        return "NON_FACTUAL", counts
    return "UNKNOWN", counts


def analyze_factuality_calibration(args: argparse.Namespace) -> int:
    paths = paths_for_args(args)
    config = load_config(paths)
    settings = config["factuality_calibration"]
    units, records, evidence_rows = load_factuality_calibration_context(
        paths,
        config,
    )
    evidence_by_id = {row["claim_id"]: row for row in evidence_rows}
    qwen_rows = load_jsonl(paths.factuality_calibration_primary_results)
    llama_rows = load_jsonl(paths.factuality_calibration_independent_results)
    if not qwen_rows or not llama_rows:
        raise FileNotFoundError(
            "Both B6e Qwen and Llama outputs are required for analysis"
        )
    qwen_digest = {row.get("model_digest") for row in qwen_rows}
    llama_digest = {row.get("model_digest") for row in llama_rows}
    if len(qwen_digest) != 1 or len(llama_digest) != 1:
        raise ValueError("B6e output model digests are inconsistent")
    qwen_config = build_factuality_calibration_run_config(
        paths,
        config,
        evaluator="qwen",
        model_digest=next(iter(qwen_digest)),
    )
    llama_config = build_factuality_calibration_run_config(
        paths,
        config,
        evaluator="llama",
        model_digest=next(iter(llama_digest)),
    )
    validate_factuality_calibration_results(
        qwen_rows,
        units,
        evidence_by_id,
        qwen_config,
    )
    validate_factuality_calibration_results(
        llama_rows,
        units,
        evidence_by_id,
        llama_config,
    )
    if (
        {row["claim_id"] for row in qwen_rows}
        != {unit["claim_id"] for unit in units}
        or {row["claim_id"] for row in llama_rows}
        != {unit["claim_id"] for unit in units}
        or any(row.get("status") != "ok" for row in qwen_rows + llama_rows)
    ):
        raise ValueError("B6e model outputs must be technically complete")
    qwen_by_id = {row["claim_id"]: row for row in qwen_rows}
    llama_by_id = {row["claim_id"]: row for row in llama_rows}
    policy_names = [
        str(row["name"]) for row in settings["aggregation_policies"]
    ]
    protocol_results: dict[str, dict[str, dict[str, Any]]] = {
        "qwen_primary": {
            claim_id: {
                "status": "ok",
                "prediction": row["prediction"],
            }
            for claim_id, row in qwen_by_id.items()
        }
    }
    prediction_rows: list[dict[str, Any]] = []
    for record in records:
        claim_id = record["claim_id"]
        qwen_prediction = qwen_by_id[claim_id]["prediction"]
        row: dict[str, Any] = {
            "claim_id": claim_id,
            "response_id": record["response_id"],
            "human_label": record["human_label"],
            "qwen_primary": qwen_prediction,
        }
        for policy in policy_names:
            llama_prediction, counts = derive_calibrated_independent_prediction(
                llama_by_id[claim_id],
                policy=policy,
            )
            llama_name = f"llama_{policy}"
            consensus_name = f"consensus_{policy}"
            protocol_results.setdefault(llama_name, {})[claim_id] = {
                "status": "ok",
                "prediction": llama_prediction,
            }
            consensus_prediction = (
                qwen_prediction
                if qwen_prediction == llama_prediction
                and qwen_prediction in {"FACTUAL", "NON_FACTUAL"}
                else "UNKNOWN"
            )
            protocol_results.setdefault(consensus_name, {})[claim_id] = {
                "status": "ok",
                "prediction": consensus_prediction,
            }
            row[llama_name] = llama_prediction
            row[consensus_name] = consensus_prediction
            row[f"{policy}_passage_counts"] = counts
        prediction_rows.append(row)
    metrics: dict[str, dict[str, Any]] = {}
    response_macro: dict[str, dict[str, Any]] = {}
    for name, results in protocol_results.items():
        metrics[name] = compute_binary_metrics(records, results)
        _, response_macro[name] = build_response_aggregation(records, results)
    comparisons = [
        (f"{name}_minus_qwen_primary", name, "qwen_primary")
        for name in protocol_results
        if name != "qwen_primary"
    ]
    bootstrap = paired_response_cluster_bootstrap(
        records,
        protocol_results,
        comparisons,
        samples=int(settings["bootstrap_samples"]),
        seed=int(settings["bootstrap_seed"]),
    )
    eligible_prefixes = tuple(settings["eligible_protocol_prefixes"])
    candidates: list[dict[str, Any]] = []
    for name, result in metrics.items():
        if not name.startswith(eligible_prefixes):
            continue
        nf_precision = result["NON_FACTUAL"]["precision"]
        passed = (
            result["balanced_accuracy"] is not None
            and result["balanced_accuracy"]
            >= float(settings["minimum_balanced_accuracy"])
            and result["coverage"] is not None
            and result["coverage"] >= float(settings["minimum_coverage"])
            and nf_precision is not None
            and nf_precision
            >= float(settings["minimum_non_factual_precision"])
        )
        candidates.append(
            {
                "protocol": name,
                "passed_gate": passed,
                "balanced_accuracy": result["balanced_accuracy"],
                "macro_f1": result["macro_f1"],
                "NON_FACTUAL_precision": nf_precision,
                "coverage": result["coverage"],
            }
        )
    passing = [row for row in candidates if row["passed_gate"]]
    passing.sort(
        key=lambda row: (
            row["balanced_accuracy"],
            row["macro_f1"],
            row["NON_FACTUAL_precision"],
            row["coverage"],
        ),
        reverse=True,
    )
    selected = passing[0]["protocol"] if passing else None
    summary = {
        "stage": "B6e_factuality_protocol_calibration",
        "status": "complete",
        "split": "dev",
        "claim_count": len(records),
        "response_count": len({row["response_id"] for row in records}),
        "gold_label_counts": dict(
            Counter(row["human_label"] for row in records)
        ),
        "protocol_metrics": metrics,
        "equal_response_macro_metrics": response_macro,
        "selection_policy": {
            "eligible_protocol_prefixes": list(eligible_prefixes),
            "qwen_primary_is_diagnostic_only": settings[
                "qwen_primary_is_diagnostic_only"
            ],
            "primary_metric": settings["primary_selection_metric"],
            "tie_break_metrics": settings["tie_break_metrics"],
            "gates": {
                "minimum_balanced_accuracy": settings[
                    "minimum_balanced_accuracy"
                ],
                "minimum_coverage": settings["minimum_coverage"],
                "minimum_non_factual_precision": settings[
                    "minimum_non_factual_precision"
                ],
            },
            "candidate_results": candidates,
            "selected_protocol": selected,
            "gate_passed": selected is not None,
        },
        "paired_response_cluster_bootstrap": bootstrap,
        "heldout_touched": False,
        "heldout_status": (
            "sealed_pending_researcher_review_even_if_gate_passes"
        ),
        "limitations": [
            "The 121 claims are a development calibration set, not a final test.",
            "Gold labels are joined only during this data-only analysis.",
            "The same frozen Hybrid top-5 evidence is used by both evaluators.",
            "A selected aggregation policy must not be changed after held-out is opened.",
        ],
    }
    atomic_write_jsonl(
        paths.factuality_calibration_predictions,
        prediction_rows,
    )
    atomic_write_json(paths.factuality_calibration_summary_json, summary)
    lines = [
        "# Experiment B — B6e Factuality-Protocol Calibration",
        "",
        "- Cohort: **121 matched claims from 19 matched responses within the "
        "first-20-raw-response development boundary**",
        "- Retrieval: frozen **Hybrid RRF top-5**; no reranking",
        "- Held-out claims touched: **no**",
        f"- Selected eligible protocol: **{selected or 'none'}**",
        f"- Calibration gate passed: **{'yes' if selected else 'no'}**",
        "",
        "## Claim-weighted gold metrics",
        "",
        "| Protocol | Accuracy | Balanced accuracy | Macro-F1 | Coverage | "
        "NON_FACTUAL precision | NON_FACTUAL recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in metrics.items():
        def pct(value: float | None) -> str:
            return "n/a" if value is None else f"{100 * value:.2f}%"

        lines.append(
            f"| `{name}` | "
            f"{pct(result['accuracy_including_abstentions_and_errors'])} | "
            f"{pct(result['balanced_accuracy'])} | "
            f"{pct(result['macro_f1'])} | "
            f"{pct(result['coverage'])} | "
            f"{pct(result['NON_FACTUAL']['precision'])} | "
            f"{pct(result['NON_FACTUAL']['recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Qwen is reported as a diagnostic because it belongs to the same "
            "model family used throughout the main CoVe pipeline. Eligible "
            "selection is restricted to the independent Llama protocols and "
            "their exact-label consensus with Qwen. UNKNOWN counts as "
            "incorrect for accuracy, balanced accuracy, and macro-F1, while "
            "coverage and selective accuracy describe abstention separately.",
            "",
            "Even if a protocol passes the preregistered development gate, "
            "held-out remains sealed until the researcher reviews this report "
            "and explicitly freezes the selected protocol.",
            "",
        ]
    )
    atomic_write_text(
        paths.factuality_calibration_summary_markdown,
        "\n".join(lines),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if selected is not None else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged FactCheck-Bench Experiment B CoVe evaluation."
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    prepare = subparsers.add_parser(
        "prepare-inputs",
        help="Build the leakage-safe response-level CoVe input manifest.",
    )
    prepare.add_argument("--scope", choices=("full",), default="full")
    prepare.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser(
        "run-questions",
        help="Generate standard CoVe verification questions.",
    )
    run.add_argument("--scope", choices=("full",), default="full")
    run.add_argument("--split", choices=("dev", "heldout"), default="dev")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help="Required before the held-out question-planning run.",
    )
    run.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    analyze = subparsers.add_parser(
        "analyze-questions",
        help="Regenerate the data-only B1 technical report.",
    )
    analyze.add_argument("--scope", choices=("full",), default="full")
    analyze.add_argument("--split", choices=("dev", "heldout"), default="dev")

    align = subparsers.add_parser(
        "run-alignment",
        help=(
            "Align B1 verification questions to hidden gold claims using the "
            "frozen B2 silver-evaluation protocol."
        ),
    )
    align.add_argument("--scope", choices=("full",), default="full")
    align.add_argument("--split", choices=("dev", "heldout"), default="dev")
    align.add_argument("--resume", action="store_true")
    align.add_argument("--dry-run", action="store_true")
    align.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help="Required before held-out B2 alignment.",
    )
    align.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    analyze_alignment_parser = subparsers.add_parser(
        "analyze-alignment",
        help="Regenerate the data-only B2 metrics and reports.",
    )
    analyze_alignment_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    analyze_alignment_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    recover_alignment_parser = subparsers.add_parser(
        "recover-alignment-format",
        help=(
            "Recover deterministic redundant overall_relation "
            "inconsistencies and drop impossible claim-ID references without "
            "guessing replacements or making model calls."
        ),
    )
    recover_alignment_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    recover_alignment_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    run_answers_parser = subparsers.add_parser(
        "run-answers",
        help=(
            "Answer every B1 verification question in a separate context that "
            "contains only that question."
        ),
    )
    run_answers_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    run_answers_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )
    run_answers_parser.add_argument("--resume", action="store_true")
    run_answers_parser.add_argument("--dry-run", action="store_true")
    run_answers_parser.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help="Required before held-out B3 answering.",
    )
    run_answers_parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    analyze_answers_parser = subparsers.add_parser(
        "analyze-answers",
        help="Regenerate the data-only B3 technical report.",
    )
    analyze_answers_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    analyze_answers_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    run_answer_evaluation_parser = subparsers.add_parser(
        "run-answer-evaluation",
        help=(
            "Evaluate B3 answers against B2 candidate claims and oracle "
            "evidence, with human labels withheld."
        ),
    )
    run_answer_evaluation_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    run_answer_evaluation_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )
    run_answer_evaluation_parser.add_argument("--resume", action="store_true")
    run_answer_evaluation_parser.add_argument("--dry-run", action="store_true")
    run_answer_evaluation_parser.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help="Required before held-out B4 evaluation.",
    )
    run_answer_evaluation_parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    analyze_answer_evaluation_parser = subparsers.add_parser(
        "analyze-answer-evaluation",
        help="Regenerate the data-only B4 correctness/funnel report.",
    )
    analyze_answer_evaluation_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    analyze_answer_evaluation_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    run_revision_parser = subparsers.add_parser(
        "run-revision",
        help=(
            "Revise each initial response using only its B3 independent "
            "question-answer pairs."
        ),
    )
    run_revision_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    run_revision_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )
    run_revision_parser.add_argument("--resume", action="store_true")
    run_revision_parser.add_argument("--dry-run", action="store_true")
    run_revision_parser.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help="Required before held-out B5 revision.",
    )
    run_revision_parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    analyze_revision_parser = subparsers.add_parser(
        "analyze-revision",
        help="Regenerate the data-only B5 technical/change-size report.",
    )
    analyze_revision_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    analyze_revision_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    recover_revision_parser = subparsers.add_parser(
        "recover-revision-format",
        help=(
            "Recover known B5 JSON container deviations without model calls "
            "or prose changes."
        ),
    )
    recover_revision_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    recover_revision_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    run_revised_claim_parser = subparsers.add_parser(
        "run-revised-claim-extraction",
        help=(
            "Extract exhaustive atomic claims from each B5 revised response "
            "without exposing the initial response or gold annotations."
        ),
    )
    run_revised_claim_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    run_revised_claim_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )
    run_revised_claim_parser.add_argument("--resume", action="store_true")
    run_revised_claim_parser.add_argument("--dry-run", action="store_true")
    run_revised_claim_parser.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help="Required before held-out B6a revised-claim extraction.",
    )
    run_revised_claim_parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    analyze_revised_claim_parser = subparsers.add_parser(
        "analyze-revised-claim-extraction",
        help="Regenerate the data-only B6a extraction report.",
    )
    analyze_revised_claim_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    analyze_revised_claim_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    recover_revised_claim_parser = subparsers.add_parser(
        "recover-revised-claim-extraction-format",
        help=(
            "Remove normalized exact duplicate B6a claims while preserving "
            "first occurrence, order, raw output, and an audit trail; no "
            "model calls."
        ),
    )
    recover_revised_claim_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    recover_revised_claim_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    run_revised_alignment_parser = subparsers.add_parser(
        "run-revised-claim-alignment",
        help=(
            "Align every canonical initial claim to B6a revised claims while "
            "withholding human labels and factuality evidence."
        ),
    )
    run_revised_alignment_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    run_revised_alignment_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )
    run_revised_alignment_parser.add_argument("--resume", action="store_true")
    run_revised_alignment_parser.add_argument("--dry-run", action="store_true")
    run_revised_alignment_parser.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help="Required before held-out B6b revised-claim alignment.",
    )
    run_revised_alignment_parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    analyze_revised_alignment_parser = subparsers.add_parser(
        "analyze-revised-claim-alignment",
        help="Regenerate data-only B6b transition-candidate artifacts.",
    )
    analyze_revised_alignment_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    analyze_revised_alignment_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    recover_revised_alignment_parser = subparsers.add_parser(
        "recover-revised-claim-alignment-structure",
        help=(
            "Conservatively recover supported B6b empty-relation and "
            "missing-initial-claim contradictions without model calls or "
            "invented revised-claim IDs; all repaired rows require audit."
        ),
    )
    recover_revised_alignment_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    recover_revised_alignment_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    prepare_revised_evidence_parser = subparsers.add_parser(
        "prepare-revised-claim-evidence",
        help=(
            "Retrieve frozen Experiment A Hybrid RRF top-5 passages for "
            "every B6a revised claim; no qrels or labels enter ranking."
        ),
    )
    prepare_revised_evidence_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    prepare_revised_evidence_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )
    prepare_revised_evidence_parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    prepare_revised_evidence_parser.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help="Required before retrieving evidence for held-out B6c claims.",
    )

    run_revised_factuality_parser = subparsers.add_parser(
        "run-revised-claim-factuality",
        help=(
            "Verify every B6a revised claim in an independent call using "
            "only its frozen Hybrid top-5 evidence."
        ),
    )
    run_revised_factuality_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    run_revised_factuality_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )
    run_revised_factuality_parser.add_argument(
        "--resume",
        action="store_true",
    )
    run_revised_factuality_parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    run_revised_factuality_parser.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help="Required before held-out B6c factuality verification.",
    )
    run_revised_factuality_parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    analyze_revised_factuality_parser = subparsers.add_parser(
        "analyze-revised-claim-factuality",
        help="Regenerate data-only B6c factuality and outcome reports.",
    )
    analyze_revised_factuality_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    analyze_revised_factuality_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    recover_revised_factuality_parser = subparsers.add_parser(
        "recover-revised-claim-factuality-format",
        help=(
            "Recover otherwise valid B6c JSON when only the 35-word "
            "rationale instruction was exceeded; preserve the full raw "
            "rationale, prediction, and confidence without a model call."
        ),
    )
    recover_revised_factuality_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    recover_revised_factuality_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    prepare_factuality_audit_parser = subparsers.add_parser(
        "prepare-factuality-audit",
        help=(
            "Audit completed B6c labels and rationales with deterministic "
            "policy checks; do not change any B6c label."
        ),
    )
    prepare_factuality_audit_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    prepare_factuality_audit_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )
    prepare_factuality_audit_parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    run_independent_factuality_parser = subparsers.add_parser(
        "run-independent-factuality",
        help=(
            "Use a blind model from a different family to assess each of "
            "five frozen passages per revised claim."
        ),
    )
    run_independent_factuality_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    run_independent_factuality_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )
    run_independent_factuality_parser.add_argument(
        "--resume",
        action="store_true",
    )
    run_independent_factuality_parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    run_independent_factuality_parser.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help="Required before held-out B6d independent adjudication.",
    )
    run_independent_factuality_parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    recover_independent_factuality_parser = subparsers.add_parser(
        "recover-independent-factuality-format",
        help=(
            "Recover complete B6d JSON with unordered ranks or overlong "
            "rationale wording without another model call."
        ),
    )
    recover_independent_factuality_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    recover_independent_factuality_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    analyze_factuality_consensus_parser = subparsers.add_parser(
        "analyze-factuality-consensus",
        help=(
            "Combine B6c and blind B6d results using the frozen conservative "
            "direct-evidence consensus rule."
        ),
    )
    analyze_factuality_consensus_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    analyze_factuality_consensus_parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
    )

    prepare_factuality_calibration_parser = subparsers.add_parser(
        "prepare-factuality-calibration",
        help=(
            "Build leakage-safe frozen Hybrid top-5 evidence for the exact "
            "121 matched development claims."
        ),
    )
    prepare_factuality_calibration_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    prepare_factuality_calibration_parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    run_factuality_calibration_primary_parser = subparsers.add_parser(
        "run-factuality-calibration-primary",
        help=(
            "Run the frozen Qwen B6c-style verifier on 121 unlabelled "
            "development claim/evidence inputs."
        ),
    )
    run_factuality_calibration_primary_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    run_factuality_calibration_primary_parser.add_argument(
        "--resume",
        action="store_true",
    )
    run_factuality_calibration_primary_parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    run_factuality_calibration_primary_parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    run_factuality_calibration_independent_parser = subparsers.add_parser(
        "run-factuality-calibration-independent",
        help=(
            "Run the frozen cross-family Llama passage adjudicator on the "
            "same 121 development claim/evidence inputs."
        ),
    )
    run_factuality_calibration_independent_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )
    run_factuality_calibration_independent_parser.add_argument(
        "--resume",
        action="store_true",
    )
    run_factuality_calibration_independent_parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    run_factuality_calibration_independent_parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )

    analyze_factuality_calibration_parser = subparsers.add_parser(
        "analyze-factuality-calibration",
        help=(
            "Join hidden dev gold labels, compare frozen aggregation "
            "policies, bootstrap response clusters, and apply the gate."
        ),
    )
    analyze_factuality_calibration_parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
    )

    branch_evaluation_parsers = (
        run_revised_claim_parser,
        analyze_revised_claim_parser,
        recover_revised_claim_parser,
        run_revised_alignment_parser,
        analyze_revised_alignment_parser,
        recover_revised_alignment_parser,
        prepare_revised_evidence_parser,
        run_revised_factuality_parser,
        analyze_revised_factuality_parser,
        recover_revised_factuality_parser,
        prepare_factuality_audit_parser,
    )
    for branch_parser in branch_evaluation_parsers:
        branch_parser.add_argument(
            "--branch",
        choices=("a", "b", "c", "d2"),
            default="a",
            help=(
                "Evaluate the canonical standard-CoVe branch (a) or an "
                "isolated intervention branch (b/c), or the frozen active "
                "Branch D implementation (d2)."
            ),
        )

    args = parser.parse_args(argv)
    branch = getattr(args, "branch", "a")
    branch_capable_stages = {
        "run-revised-claim-extraction",
        "analyze-revised-claim-extraction",
        "recover-revised-claim-extraction-format",
        "run-revised-claim-alignment",
        "analyze-revised-claim-alignment",
        "recover-revised-claim-alignment-structure",
        "prepare-revised-claim-evidence",
        "run-revised-claim-factuality",
        "analyze-revised-claim-factuality",
        "recover-revised-claim-factuality-format",
        "prepare-factuality-audit",
    }
    if branch != "a" and args.stage not in branch_capable_stages:
        parser.error(
            "Intervention branches b/c/d2 are valid only for the shared "
            "B6a/B6b/B6c evaluation protocol"
        )
    if (
        args.stage
        in {
            "run-questions",
            "run-alignment",
            "run-answers",
            "run-answer-evaluation",
            "run-revision",
            "run-revised-claim-extraction",
            "run-revised-claim-alignment",
            "prepare-revised-claim-evidence",
            "run-revised-claim-factuality",
            "run-independent-factuality",
        }
        and args.split == "heldout"
        and not args.confirm_config_frozen
    ):
        parser.error(
            f"heldout {args.stage} requires --confirm-config-frozen after "
            "development configuration freeze"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "prepare-inputs":
        paths = paths_for_args(args)
        config = load_config(paths)
        report = prepare_cove_inputs(
            PROJECT_ROOT,
            paths,
            config,
            dry_run=args.dry_run,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "run-questions":
        return run_questions(args)
    if args.stage == "analyze-questions":
        return analyze_questions(args)
    if args.stage == "run-alignment":
        return run_alignment(args)
    if args.stage == "analyze-alignment":
        return analyze_alignment(args)
    if args.stage == "recover-alignment-format":
        return recover_alignment_format(args)
    if args.stage == "run-answers":
        return run_verification_answers(args)
    if args.stage == "analyze-answers":
        return analyze_verification_answers(args)
    if args.stage == "run-answer-evaluation":
        return run_answer_claim_evaluation(args)
    if args.stage == "analyze-answer-evaluation":
        return analyze_answer_claim_evaluation(args)
    if args.stage == "run-revision":
        return run_revision(args)
    if args.stage == "analyze-revision":
        return analyze_revision(args)
    if args.stage == "recover-revision-format":
        return recover_revision_format(args)
    if args.stage == "run-revised-claim-extraction":
        return run_revised_claim_extraction(args)
    if args.stage == "analyze-revised-claim-extraction":
        return analyze_revised_claim_extraction(args)
    if args.stage == "recover-revised-claim-extraction-format":
        return recover_revised_claim_extraction_format(args)
    if args.stage == "run-revised-claim-alignment":
        return run_revised_claim_alignment(args)
    if args.stage == "analyze-revised-claim-alignment":
        return analyze_revised_claim_alignment(args)
    if args.stage == "recover-revised-claim-alignment-structure":
        return recover_revised_claim_alignment_structure(args)
    if args.stage == "prepare-revised-claim-evidence":
        return prepare_revised_claim_evidence(args)
    if args.stage == "run-revised-claim-factuality":
        return run_revised_claim_factuality(args)
    if args.stage == "analyze-revised-claim-factuality":
        return analyze_revised_claim_factuality(args)
    if args.stage == "recover-revised-claim-factuality-format":
        return recover_revised_claim_factuality_format(args)
    if args.stage == "prepare-factuality-audit":
        return prepare_primary_factuality_audit(args)
    if args.stage == "run-independent-factuality":
        return run_independent_factuality(args)
    if args.stage == "recover-independent-factuality-format":
        return recover_independent_factuality_format(args)
    if args.stage == "analyze-factuality-consensus":
        return analyze_factuality_consensus(args)
    if args.stage == "prepare-factuality-calibration":
        return prepare_factuality_calibration(args)
    if args.stage == "run-factuality-calibration-primary":
        return run_factuality_calibration(args, evaluator="qwen")
    if args.stage == "run-factuality-calibration-independent":
        return run_factuality_calibration(args, evaluator="llama")
    if args.stage == "analyze-factuality-calibration":
        return analyze_factuality_calibration(args)
    raise ValueError(f"Unsupported Experiment B stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
