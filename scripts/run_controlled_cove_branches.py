#!/usr/bin/env python3
"""Controlled four-branch CoVe comparison used by the final methodology.

The script extends the completed standard-CoVe mechanism run without changing
its artifacts:

* Branch A reuses the frozen B1/B3/B5 standard-CoVe outputs.
* Branch B answers the same frozen B1 questions with frozen Hybrid evidence,
  then revises the original response with the same Q&A revision prompt.
* Branch C gives the frozen Branch A response one generic extra revision call.
* Branch D uses the bounded D2 implementation to give the frozen Branch A
  response one evidence-guided targeted revision call.

Every branch has a separate output directory and a branch-specific run
fingerprint. Gold labels, qrels, benchmark evidence, and gold URL mappings are
never read by branch generation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

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
    load_json,
    load_jsonl,
    sha256_file,
    sha256_text,
    validate_response_manifest,
)
from factcheck_bench_pipeline import retrieval_paths  # noqa: E402
from factcheck_bench_retrieval_eval import (  # noqa: E402
    evaluation_paths,
    load_evaluation_config,
    rank_frozen_hybrid_queries,
)


load_dotenv(PROJECT_ROOT / ".env")

BRANCH_CONFIG_SCHEMA = "fcb_cove_branch_experiment_config_v1"
BRANCH_D2_CONFIG_SCHEMA = "fcb_cove_branch_d2_config_v1"
BRANCHES = ("a", "b", "c", "d")
REVISION_FALLBACK_POLICY_VERSION = (
    "preserve_branch_base_on_unrecoverable_revision_v1"
)
MODEL_STAGES = {
    "run-grounded-answers",
    "run-grounded-revision",
    "run-extra-revision",
    "run-targeted-evidence-revision",
}
STAGES = (
    "audit",
    "prepare-grounded-evidence",
    "run-grounded-answers",
    "run-grounded-revision",
    "run-extra-revision",
    "prepare-targeted-feedback-candidates",
    "prepare-bounded-targeted-feedback",
    "run-targeted-evidence-revision",
    "summarize",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def project_relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def branch_config_path() -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "factcheck_bench"
        / "cove"
        / "config"
        / "cove_branch_experiment_config.json"
    )


def branch_d2_config_path() -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "factcheck_bench"
        / "cove"
        / "config"
        / "cove_branch_d2_config.json"
    )


def branch_manifest_path(split: str) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "factcheck_bench"
        / "cove"
        / "manifests"
        / f"cove_branch_manifest_{split}.jsonl"
    )


def branches_root() -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "factcheck_bench_full"
        / "cove"
        / "branches"
    )


def status_json_path(split: str) -> Path:
    return branches_root() / "reports" / f"branch_status_{split}.json"


def status_markdown_path(split: str) -> Path:
    return branches_root() / "reports" / f"branch_status_{split}.md"


def grounded_evidence_path(split: str) -> Path:
    return (
        cove_paths(PROJECT_ROOT, "full", "b").jsonl_dir
        / f"Q2_grounded_question_evidence_{split}.jsonl"
    )


def grounded_answers_path(split: str) -> Path:
    return (
        cove_paths(PROJECT_ROOT, "full", "b").jsonl_dir
        / f"Q3_grounded_verification_answers_{split}.jsonl"
    )


def targeted_feedback_candidates_path(split: str) -> Path:
    return (
        cove_paths(PROJECT_ROOT, "full", "d").jsonl_dir
        / f"D1_selective_verifier_feedback_{split}.jsonl"
    )


def bounded_targeted_feedback_path(split: str) -> Path:
    return (
        cove_paths(PROJECT_ROOT, "full", "d2").jsonl_dir
        / f"D2_compact_selective_feedback_{split}.jsonl"
    )


def branch_report_path(branch: str, split: str) -> Path:
    return (
        cove_paths(PROJECT_ROOT, "full", branch).reports_dir
        / f"branch_{branch}_generation_{split}_summary.json"
    )


def branch_settings(config: Mapping[str, Any], branch: str) -> dict[str, Any]:
    if branch == "d2":
        d2_config = config.get("_branch_d2_config")
        settings = (
            d2_config.get("branch")
            if isinstance(d2_config, dict)
            else None
        )
        if not isinstance(settings, dict):
            raise ValueError("Branch D v2 configuration is missing")
        return settings
    settings = config["branches"].get(branch)
    if not isinstance(settings, dict):
        raise ValueError(f"Unknown branch configuration: {branch}")
    return settings


def load_branch_config() -> tuple[Path, dict[str, Any]]:
    path = branch_config_path()
    config = load_json(path)
    if config.get("schema_version") != BRANCH_CONFIG_SCHEMA:
        raise ValueError(
            f"Unsupported branch config schema: {config.get('schema_version')}"
        )
    if config.get("scope") != "full":
        raise ValueError("The controlled CoVe branch experiment is full-scope")
    if set(config.get("branches", {})) != set(BRANCHES):
        raise ValueError("Branch config must define exactly branches a/b/c/d")
    d2_path = branch_d2_config_path()
    d2_config = load_json(d2_path)
    if d2_config.get("schema_version") != BRANCH_D2_CONFIG_SCHEMA:
        raise ValueError(
            "Unsupported Branch D v2 config schema: "
            f"{d2_config.get('schema_version')}"
        )
    if d2_config.get("scope") != "full":
        raise ValueError("The Branch D v2 experiment is full-scope")
    if d2_config.get("parent_branch_config_path") != project_relative(path):
        raise ValueError("Branch D v2 parent config path is inconsistent")
    d2 = d2_config.get("branch")
    if not isinstance(d2, dict):
        raise ValueError("Branch D v2 config must define branch")
    for positive_integer_field in (
        "maximum_feedback_claims_per_response",
        "maximum_passages_per_claim",
        "maximum_passage_characters",
        "maximum_rationale_characters",
    ):
        value = d2.get(positive_integer_field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"branch_d_v2.{positive_integer_field} must be positive"
            )
    policy = config.get("comparison_policy", {})
    required_true = (
        "branch_outputs_are_not_chained",
        "branch_b_never_reads_branch_a_revision",
        "branch_c_never_reads_evidence_or_verifier_feedback",
        "branch_d_never_reads_branch_b_or_branch_c_outputs",
        "gold_fields_are_evaluation_only",
        "same_final_evaluation_protocol_for_all_branches",
    )
    if any(policy.get(field) is not True for field in required_true):
        raise ValueError("Every branch isolation policy must remain true")
    config["_branch_d2_config"] = d2_config
    return path, config


def load_shared_responses(
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = cove_paths(PROJECT_ROOT, "full", "a")
    config = load_config(paths)
    manifest = load_jsonl(paths.response_manifest)
    validate_response_manifest(manifest, config)
    responses = sorted(
        (row for row in manifest if row["split"] == split),
        key=lambda row: int(row["source_record_index"]),
    )
    expected_responses = int(
        config["split_policy"]["expected"][f"{split}_responses"]
    )
    if len(responses) != expected_responses:
        raise ValueError(
            f"{split} response count mismatch: {len(responses)} "
            f"!= {expected_responses}"
        )
    return config, responses


def load_frozen_questions(
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config, responses = load_shared_responses(split)
    paths = cove_paths(PROJECT_ROOT, "full", "a")
    questions = load_jsonl(paths.question_results(split))
    response_ids = {row["response_id"] for row in responses}
    if any(row.get("status") != "ok" for row in questions):
        raise ValueError(f"Frozen B1 has non-ok rows for {split}")
    extra = {row.get("response_id") for row in questions} - response_ids
    if extra:
        raise ValueError(f"Frozen B1 has cross-split rows: {extra}")
    if {row["response_id"] for row in questions} != response_ids:
        raise ValueError(f"Branch A B1 is incomplete for {split}")
    return config, responses, questions


def load_branch_a_revisions(
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config, responses = load_shared_responses(split)
    paths = cove_paths(PROJECT_ROOT, "full", "a")
    revisions = load_jsonl(paths.revision_results(split))
    response_ids = {row["response_id"] for row in responses}
    if any(row.get("status") != "ok" for row in revisions):
        raise ValueError(f"Branch A B5 has non-ok rows for {split}")
    if {row.get("response_id") for row in revisions} != response_ids:
        raise ValueError(f"Branch A B5 is incomplete or cross-split for {split}")
    response_by_id = {row["response_id"]: row for row in responses}
    for row in revisions:
        source = response_by_id[row["response_id"]]
        if row["original_question"] != source["original_question"]:
            raise ValueError(f"Branch A question drift: {row['response_id']}")
        if row["initial_response_sha256"] != source["initial_response_sha256"]:
            raise ValueError(f"Branch A initial-response drift: {row['response_id']}")
        if row["revised_response_sha256"] != sha256_text(
            row["revised_response"]
        ):
            raise ValueError(f"Branch A revision hash mismatch: {row['response_id']}")
    return config, responses, revisions


def load_base_dependencies(
    split: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    config, responses, questions = load_frozen_questions(split)
    paths = cove_paths(PROJECT_ROOT, "full", "a")
    answers = load_jsonl(paths.verification_answer_results(split))
    _, _, revisions = load_branch_a_revisions(split)
    response_ids = {row["response_id"] for row in responses}
    if any(row.get("status") != "ok" for row in answers):
        raise ValueError(f"Branch A B3 has non-ok rows for {split}")
    extra = {row.get("response_id") for row in answers} - response_ids
    if extra:
        raise ValueError(f"Branch A B3 has cross-split rows: {extra}")
    question_by_id: dict[str, tuple[str, str]] = {}
    for row in questions:
        for index, text in enumerate(row["verification_questions"], start=1):
            question_id = f"{row['response_id']}_q{index:02d}"
            if question_id in question_by_id:
                raise ValueError(f"Duplicate frozen question ID: {question_id}")
            question_by_id[question_id] = (row["response_id"], text)
    if {row["question_id"] for row in answers} != set(question_by_id):
        raise ValueError(f"Branch A B3 question IDs differ from frozen B1 {split}")
    for row in answers:
        response_id, text = question_by_id[row["question_id"]]
        if row["response_id"] != response_id:
            raise ValueError(f"Cross-response B3 answer: {row['question_id']}")
        if row["verification_question"] != text:
            raise ValueError(f"B3 question text drift: {row['question_id']}")
    return config, responses, questions, answers, revisions


def _rows_or_empty(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path) if path.exists() else []


def _artifact_state(
    path: Path,
    *,
    expected_rows: int | None = None,
    ok_field: str | None = "status",
) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": project_relative(path),
            "exists": False,
            "row_count": 0,
            "ok_count": 0,
            "complete": False,
            "sha256": None,
        }
    rows = load_jsonl(path)
    ok_count = (
        sum(row.get(ok_field) == "ok" for row in rows)
        if ok_field is not None
        else len(rows)
    )
    complete = (
        expected_rows is not None
        and len(rows) == expected_rows
        and ok_count == expected_rows
    )
    return {
        "path": project_relative(path),
        "exists": True,
        "row_count": len(rows),
        "ok_count": ok_count,
        "complete": complete,
        "sha256": sha256_file(path),
    }


def audit_rows_and_summary(
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    branch_config_file, branch_config = load_branch_config()
    base_config, responses, questions, answers, revisions = (
        load_base_dependencies(split)
    )
    expected_responses = len(responses)
    expected_questions = sum(
        len(row["verification_questions"]) for row in questions
    )
    a_paths = cove_paths(PROJECT_ROOT, "full", "a")
    b_paths = cove_paths(PROJECT_ROOT, "full", "b")
    c_paths = cove_paths(PROJECT_ROOT, "full", "c")
    d_paths = cove_paths(PROJECT_ROOT, "full", "d")

    a_b6a = _rows_or_empty(a_paths.revised_claim_extraction_results(split))
    a_evidence = _rows_or_empty(a_paths.revised_claim_evidence(split))
    a_factuality = _rows_or_empty(a_paths.revised_claim_factuality_results(split))
    a_audit = _rows_or_empty(a_paths.factuality_audit_manifest(split))
    d_feedback_exists = targeted_feedback_candidates_path(split).exists()
    revised_claim_count = sum(
        len(row.get("revised_claims") or [])
        for row in a_b6a
        if row.get("status") == "ok"
    )
    a_by_response = {row["response_id"]: row for row in revisions}
    b1_by_response = {row["response_id"]: row for row in questions}
    b3_by_response: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answers:
        b3_by_response[row["response_id"]].append(row)
    d_claims_by_response: Counter[str] = Counter(
        row.get("response_id") for row in a_factuality
    )
    manifest_rows: list[dict[str, Any]] = []
    for source in responses:
        response_id = source["response_id"]
        a_revision = a_by_response[response_id]
        question_plan = b1_by_response[response_id]
        answer_rows = sorted(
            b3_by_response[response_id],
            key=lambda row: int(row["question_index"]),
        )
        manifest_rows.append(
            {
                "schema_version": "fcb_cove_branch_manifest_v1",
                "response_id": response_id,
                "source_record_index": source["source_record_index"],
                "split": split,
                "shared_initial_response_sha256": source[
                    "initial_response_sha256"
                ],
                "shared_question_plan_sha256": sha256_text(
                    json.dumps(
                        question_plan["verification_questions"],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
                "shared_question_count": len(
                    question_plan["verification_questions"]
                ),
                "branch_a": {
                    "status": "complete",
                    "standard_answer_count": len(answer_rows),
                    "standard_revision_sha256": a_revision[
                        "revised_response_sha256"
                    ],
                    "revision_artifact": project_relative(
                        a_paths.revision_results(split)
                    ),
                },
                "branch_b": {
                    "status": (
                        "complete"
                        if _response_revision_ok(
                            b_paths.revision_results(split), response_id
                        )
                        else "pending"
                    ),
                    "revision_artifact": project_relative(
                        b_paths.revision_results(split)
                    ),
                },
                "branch_c": {
                    "status": (
                        "complete"
                        if _response_revision_ok(
                            c_paths.revision_results(split), response_id
                        )
                        else "pending"
                    ),
                    "revision_base_sha256": a_revision[
                        "revised_response_sha256"
                    ],
                    "revision_artifact": project_relative(
                        c_paths.revision_results(split)
                    ),
                },
                "branch_d": {
                    "prepared_claim_count": d_claims_by_response[response_id],
                    "status": (
                        "complete"
                        if _response_revision_ok(
                            d_paths.revision_results(split), response_id
                        )
                        else (
                            "selective_feedback_prepared_revision_pending"
                            if d_feedback_exists
                            else "ingredients_prepared"
                        )
                    ),
                    "revision_base_sha256": a_revision[
                        "revised_response_sha256"
                    ],
                    "revision_artifact": project_relative(
                        d_paths.revision_results(split)
                    ),
                },
            }
        )

    artifacts = {
        "branch_a": {
            "question_planning": _artifact_state(
                a_paths.question_results(split),
                expected_rows=expected_responses,
            ),
            "parametric_answers": _artifact_state(
                a_paths.verification_answer_results(split),
                expected_rows=expected_questions,
            ),
            "standard_revision": _artifact_state(
                a_paths.revision_results(split),
                expected_rows=expected_responses,
            ),
            "revised_claim_extraction": _artifact_state(
                a_paths.revised_claim_extraction_results(split),
                expected_rows=expected_responses,
            ),
            "gold_revised_claim_alignment": _artifact_state(
                a_paths.revised_claim_alignment_results(split),
                expected_rows=expected_responses,
            ),
            "revised_claim_evidence": _artifact_state(
                a_paths.revised_claim_evidence(split),
                expected_rows=revised_claim_count,
                ok_field=None,
            ),
            "revised_claim_factuality": _artifact_state(
                a_paths.revised_claim_factuality_results(split),
                expected_rows=revised_claim_count,
            ),
            "deterministic_factuality_audit": _artifact_state(
                a_paths.factuality_audit_manifest(split),
                expected_rows=revised_claim_count,
                ok_field=None,
            ),
        },
        "branch_b": {
            "grounded_question_evidence": _artifact_state(
                grounded_evidence_path(split),
                expected_rows=expected_questions,
                ok_field=None,
            ),
            "grounded_answers": _artifact_state(
                grounded_answers_path(split),
                expected_rows=expected_questions,
            ),
            "revision": _artifact_state(
                b_paths.revision_results(split),
                expected_rows=expected_responses,
            ),
        },
        "branch_c": {
            "revision": _artifact_state(
                c_paths.revision_results(split),
                expected_rows=expected_responses,
            ),
        },
        "branch_d": {
            "branch_a_claims": _artifact_state(
                a_paths.revised_claim_extraction_results(split),
                expected_rows=expected_responses,
            ),
            "branch_a_claim_evidence": _artifact_state(
                a_paths.revised_claim_evidence(split),
                expected_rows=revised_claim_count,
                ok_field=None,
            ),
            "branch_a_verifier_predictions": _artifact_state(
                a_paths.revised_claim_factuality_results(split),
                expected_rows=revised_claim_count,
            ),
            "branch_a_deterministic_audit": _artifact_state(
                a_paths.factuality_audit_manifest(split),
                expected_rows=revised_claim_count,
                ok_field=None,
            ),
            "selective_feedback": _artifact_state(
                targeted_feedback_candidates_path(split),
                expected_rows=revised_claim_count,
                ok_field=None,
            ),
            "revision": _artifact_state(
                d_paths.revision_results(split),
                expected_rows=expected_responses,
            ),
        },
    }
    summary = {
        "schema_version": "fcb_cove_branch_status_v1",
        "status": "audited",
        "split": split,
        "response_count": expected_responses,
        "frozen_question_count": expected_questions,
        "branch_a_revised_claim_count": revised_claim_count,
        "branch_a_assessment": (
            "complete_standard_cove_generation_and_silver_evaluation"
        ),
        "branch_d_assessment": (
            "intervention_complete"
            if artifacts["branch_d"]["revision"]["complete"]
            else (
                "selective_feedback_prepared_revision_pending"
                if artifacts["branch_d"]["selective_feedback"]["complete"]
                else "prepared_verifier_ingredients_only"
            )
        ),
        "artifacts": artifacts,
        "isolation_assertions": {
            "same_response_ids_all_branches": True,
            "same_initial_response_branch_a_and_b": True,
            "same_frozen_question_plan_branch_a_and_b": True,
            "branch_b_does_not_read_branch_a_revision": True,
            "branch_c_reads_only_branch_a_revision": True,
            "branch_d_reads_branch_a_revision_and_selected_feedback_only": True,
            "branch_c_and_d_each_add_one_revision_call": True,
            "gold_fields_excluded_from_generation": True,
        },
        "input_hashes": {
            "branch_config_sha256": sha256_file(branch_config_file),
            "base_config_sha256": sha256_file(cove_paths(
                PROJECT_ROOT, "full", "a"
            ).config),
            "response_manifest_sha256": sha256_file(
                cove_paths(PROJECT_ROOT, "full", "a").response_manifest
            ),
            "B1_question_results_sha256": sha256_file(
                cove_paths(PROJECT_ROOT, "full", "a").question_results(split)
            ),
            "B3_answer_results_sha256": sha256_file(
                cove_paths(
                    PROJECT_ROOT, "full", "a"
                ).verification_answer_results(split)
            ),
            "B5_branch_a_revision_sha256": sha256_file(
                cove_paths(PROJECT_ROOT, "full", "a").revision_results(split)
            ),
        },
        "base_expected": base_config["split_policy"]["expected"],
        "branch_config_experiment": branch_config["experiment"],
        "generated_at": utc_now(),
    }
    return manifest_rows, summary


def _response_revision_ok(path: Path, response_id: str) -> bool:
    if not path.exists():
        return False
    return any(
        row.get("response_id") == response_id and row.get("status") == "ok"
        for row in load_jsonl(path)
    )


def build_status_markdown(summary: Mapping[str, Any]) -> str:
    split = summary["split"]
    artifacts = summary["artifacts"]
    lines = [
        f"# Controlled CoVe branch status — {split}",
        "",
        f"- Responses: **{summary['response_count']}**",
        f"- Frozen verification questions: **{summary['frozen_question_count']}**",
        f"- Branch A revised claims: **{summary['branch_a_revised_claim_count']}**",
        f"- Branch A: **{summary['branch_a_assessment']}**",
        f"- Branch D: **{summary['branch_d_assessment']}**",
        "",
        "| Branch | Condition | Generation status | Evaluation status |",
        "|---|---|---|---|",
    ]
    labels = {
        "a": "Standard CoVe",
        "b": "Evidence-grounded CoVe",
        "c": "Extra-revision control",
        "d": "Selective retrieval-verifier revision",
    }
    for branch in BRANCHES:
        group = artifacts[f"branch_{branch}"]
        revision = (
            group.get("standard_revision")
            if branch == "a"
            else group.get("revision")
        )
        generation = "complete" if revision and revision["complete"] else "pending"
        branch_paths = cove_paths(PROJECT_ROOT, "full", branch)
        evaluation = (
            "complete"
            if branch_paths.initial_claim_outcomes(split).exists()
            and branch_paths.added_claim_outcomes(split).exists()
            else "pending"
        )
        lines.append(
            f"| {branch.upper()} | {labels[branch]} | {generation} | "
            f"{evaluation} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Branch A is the frozen standard-CoVe baseline. Branch D is not "
            "complete merely because Branch A claims, retrieved evidence, and "
            "verifier judgments exist; it becomes complete only after the "
            "selective feedback artifact and its own revision output exist.",
            "",
            "Branches B, C, and D write to separate directories. Branch B starts "
            "from the shared initial response, while C and D start independently "
            "from the same frozen Branch A response. No intervention branch reads "
            "another intervention branch.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_stage(split: str, *, dry_run: bool) -> int:
    rows, summary = audit_rows_and_summary(split)
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("\nDry-run: no manifest or report was written.")
        return 0
    atomic_write_jsonl(branch_manifest_path(split), rows)
    atomic_write_json(status_json_path(split), summary)
    atomic_write_text(status_markdown_path(split), build_status_markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def frozen_question_units(
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, _, questions = load_frozen_questions(split)
    units: list[dict[str, Any]] = []
    for row in sorted(
        questions, key=lambda item: int(item["source_record_index"])
    ):
        for index, question in enumerate(row["verification_questions"], start=1):
            units.append(
                {
                    "query_id": f"{row['response_id']}_q{index:02d}",
                    "question_id": f"{row['response_id']}_q{index:02d}",
                    "response_id": row["response_id"],
                    "source_record_index": row["source_record_index"],
                    "split": split,
                    "question_index": index,
                    "verification_question": question,
                    "verification_question_sha256": sha256_text(question),
                    "text": question,
                    "b1_run_fingerprint": row["run_fingerprint"],
                }
            )
    return questions, units


def prepare_grounded_evidence(split: str, *, dry_run: bool) -> int:
    config_path, branch_config = load_branch_config()
    _, units = frozen_question_units(split)
    settings = branch_config["branches"]["b"]
    corpus_paths = retrieval_paths(PROJECT_ROOT, "full")
    eval_paths = evaluation_paths(corpus_paths)
    configured_path = PROJECT_ROOT / branch_config["retrieval_config_path"]
    if configured_path.resolve() != eval_paths.config.resolve():
        raise ValueError("Branch B must use the canonical retrieval config")
    retrieval_config = load_evaluation_config(eval_paths)
    if settings["retriever"] != "hybrid_rrf":
        raise ValueError("Branch B is frozen to Hybrid RRF")
    top_k = int(settings["evidence_top_k"])
    if dry_run:
        print(
            json.dumps(
                {
                    "stage": "branch_b_prepare_grounded_question_evidence",
                    "split": split,
                    "question_count": len(units),
                    "query_field": "verification_question",
                    "retriever": settings["retriever"],
                    "top_k": top_k,
                    "output": project_relative(grounded_evidence_path(split)),
                    "gold_fields_used": [],
                    "branch_a_revision_read": False,
                    "dry_run": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    passages, _, _, hybrid_rows, embedding_digest = rank_frozen_hybrid_queries(
        PROJECT_ROOT,
        corpus_paths,
        eval_paths,
        retrieval_config,
        [
            {
                "query_id": unit["question_id"],
                "response_id": unit["response_id"],
                "text": unit["verification_question"],
            }
            for unit in units
        ],
        split_label=f"cove_branch_b_questions_{split}",
    )
    passage_by_id = {row["passage_id"]: row for row in passages}
    ranked: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hybrid_rows:
        if int(row["rank"]) <= top_k:
            ranked[row["query_id"]].append(row)
    rows: list[dict[str, Any]] = []
    for unit in units:
        run_rows = sorted(
            ranked[unit["question_id"]], key=lambda row: int(row["rank"])
        )
        if [int(row["rank"]) for row in run_rows] != list(
            range(1, top_k + 1)
        ):
            raise ValueError(
                f"Hybrid top-{top_k} incomplete for {unit['question_id']}"
            )
        items = []
        visible = []
        for run_row in run_rows:
            passage = passage_by_id[run_row["passage_id"]]
            text = str(passage["text"]).strip()
            rank = int(run_row["rank"])
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
            visible.append(
                f"Passage {rank} text (JSON-encoded): "
                f"{json.dumps(text, ensure_ascii=False)}"
            )
        normalized = "\n\n".join(visible)
        rows.append(
            {
                "schema_version": "cove_branch_b_question_evidence_v1",
                **{key: value for key, value in unit.items() if key != "text"},
                "branch_id": "b",
                "retriever": "hybrid_rrf",
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
                "branch_config_sha256": sha256_file(config_path),
                "ranking_input_fields": ["verification_question"],
                "ranking_excluded_fields": [
                    "initial_response",
                    "branch_a_answer",
                    "branch_a_revised_response",
                    "gold_claim",
                    "human_label",
                    "gold_evidence",
                    "gold_url_mapping",
                    "qrels",
                ],
                "created_at": utc_now(),
            }
        )
    atomic_write_jsonl(grounded_evidence_path(split), rows)
    summary = {
        "stage": "branch_b_prepare_grounded_question_evidence",
        "status": "complete",
        "split": split,
        "question_count": len(rows),
        "top_k": top_k,
        "output": project_relative(grounded_evidence_path(split)),
        "output_sha256": sha256_file(grounded_evidence_path(split)),
        "gold_fields_used": [],
    }
    atomic_write_json(branch_report_path("b", split), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_prompt(
    relative_path: str,
    placeholders: Iterable[str],
) -> tuple[Path, str]:
    path = PROJECT_ROOT / relative_path
    text = path.read_text(encoding="utf-8")
    required = set(placeholders)
    found = {placeholder for placeholder in required if placeholder in text}
    if found != required:
        raise ValueError(f"Prompt placeholders missing in {path}: {required-found}")
    if any(text.count(placeholder) != 1 for placeholder in required):
        raise ValueError(f"Every prompt placeholder must occur once in {path}")
    return path, text


def answer_schema(top_k: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "answer_status",
            "verification_answer",
            "cited_passage_ranks",
        ],
        "properties": {
            "answer_status": {
                "type": "string",
                "enum": ["ANSWERED", "UNCERTAIN", "INVALID_PREMISE"],
            },
            "verification_answer": {
                "type": "string",
                "minLength": 2,
                "maxLength": 1200,
            },
            "cited_passage_ranks": {
                "type": "array",
                "uniqueItems": True,
                "items": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": top_k,
                },
            },
        },
    }


def revision_schema() -> dict[str, Any]:
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


def response_value(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


def preflight_model(client: Client, settings: Mapping[str, Any]) -> str:
    response = client.list()
    models = response_value(response, "models")
    available: dict[str, str | None] = {}
    for item in models or []:
        name = response_value(item, "model") or response_value(item, "name")
        if isinstance(name, str) and name:
            digest = response_value(item, "digest")
            available[name] = digest if isinstance(digest, str) else None
    model = str(settings["model"])
    if model not in available:
        raise ValueError(f"Configured model is not installed: {model}")
    digest = available[model]
    if digest != settings["expected_model_digest"]:
        raise ValueError(
            "Installed model digest differs from the frozen branch config: "
            f"expected={settings['expected_model_digest']}, actual={digest}"
        )
    return str(digest)


def call_model(
    client: Client,
    settings: Mapping[str, Any],
    prompt: str,
    schema: Mapping[str, Any],
    *,
    num_predict: int,
) -> tuple[str, dict[str, Any]]:
    response = client.chat(
        model=settings["model"],
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        format=dict(schema),
        options={
            "temperature": float(settings["temperature"]),
            "seed": int(settings["seed"]),
            "num_predict": int(num_predict),
        },
        think=bool(settings["think"]),
    )
    message = response_value(response, "message")
    content = response_value(message, "content") if message is not None else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama response contains no message content")
    metadata = {
        key: response_value(response, key)
        for key in (
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
        if response_value(response, key) is not None
    }
    return content.strip(), metadata


def parse_grounded_answer(raw: str, top_k: int) -> dict[str, Any]:
    parsed = json.loads(raw)
    expected = {
        "answer_status",
        "verification_answer",
        "cited_passage_ranks",
    }
    if not isinstance(parsed, dict) or set(parsed) != expected:
        raise ValueError(f"Grounded answer fields must be exactly {sorted(expected)}")
    if parsed["answer_status"] not in {
        "ANSWERED",
        "UNCERTAIN",
        "INVALID_PREMISE",
    }:
        raise ValueError("Invalid grounded answer status")
    answer = " ".join(str(parsed["verification_answer"]).split())
    if not 2 <= len(answer) <= 1200:
        raise ValueError("Grounded verification answer length is invalid")
    ranks = parsed["cited_passage_ranks"]
    if (
        not isinstance(ranks, list)
        or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks)
        or len(ranks) != len(set(ranks))
        or any(not 1 <= rank <= top_k for rank in ranks)
    ):
        raise ValueError("Grounded answer cited passage ranks are invalid")
    if parsed["answer_status"] == "UNCERTAIN" and ranks:
        # An uncertain answer may still cite why evidence was insufficient.
        pass
    return {
        "answer_status": parsed["answer_status"],
        "verification_answer": answer,
        "cited_passage_ranks": sorted(ranks),
    }


def parse_revision(raw: str) -> str:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or set(parsed) != {"revised_response"}:
        raise ValueError("Revision output must contain only revised_response")
    text = parsed["revised_response"]
    if not isinstance(text, str):
        raise TypeError("revised_response must be a string")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not 20 <= len(text) <= 30000:
        raise ValueError("revised_response length is invalid")
    return text


def recover_revision_output_format(raw: str) -> tuple[str, str]:
    """Recover format-only revision deviations without changing the prose.

    Qwen occasionally returns the requested response under a generic singleton
    key such as ``answer``, under a response-derived singleton key, or as the
    only JSON string inside braces without a colon, even when Ollama receives a
    stricter JSON schema. These are the same format-only patterns handled by
    the standard-CoVe B5 recovery. The recovery never rewrites, summarizes, or
    regenerates the response text.
    """

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Revision model output is empty")
    candidate = raw.strip()
    fence_method: str | None = None
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Revision output has an invalid code fence")
        opening = lines[0].strip().lower()
        if opening not in {"```", "```json"}:
            raise ValueError("Revision output uses an unsupported code fence")
        candidate = "\n".join(lines[1:-1]).strip()
        fence_method = "json_code_fence"
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as error:
        if candidate.startswith("{") and candidate.endswith("}"):
            inner = candidate[1:-1].strip()
            try:
                singleton = json.loads(inner)
            except json.JSONDecodeError:
                singleton = None
            if isinstance(singleton, str):
                revised = parse_revision(
                    json.dumps(
                        {"revised_response": singleton},
                        ensure_ascii=False,
                    )
                )
                method = "singleton_string_key_without_colon"
                if fence_method is not None:
                    method = f"{fence_method}+{method}"
                return revised, method
        raise ValueError(
            f"Revision output is not recoverable strict JSON: {error}"
        ) from error
    if not isinstance(parsed, dict) or len(parsed) != 1:
        raise ValueError(
            "Revision recovery requires a singleton JSON object"
        )
    key, value = next(iter(parsed.items()))
    aliases = {
        "answer",
        "response",
        "revised_answer",
        "final_answer",
        "output",
    }
    if not isinstance(key, str) or not isinstance(value, str):
        raise ValueError(
            "Revision singleton key and value must both be strings"
        )
    revised = parse_revision(
        json.dumps({"revised_response": value}, ensure_ascii=False)
    )
    method = (
        f"alias_key_{key}"
        if key in aliases
        else "singleton_arbitrary_key_string_value"
    )
    if fence_method is not None:
        method = f"{fence_method}+{method}"
    return revised, method


def recover_existing_revision_parse_errors(
    output_path: Path,
    run_fingerprint: str,
    units: Iterable[Mapping[str, Any]],
) -> int:
    """Normalize existing parse errors before a resume run.

    A format-only deviation is recovered from the raw output. If the raw
    output cannot unambiguously be interpreted as a revised response, the
    branch base is preserved as a fail-safe no-op. This avoids both a second
    content-generation call and contamination from claim-analysis prose.
    """

    if not output_path.exists():
        return 0
    unit_by_id = {str(unit["response_id"]): unit for unit in units}
    rows = load_jsonl(output_path)
    recovered_count = 0
    for row in rows:
        if (
            row.get("status") != "parse_error"
            or row.get("run_fingerprint") != run_fingerprint
        ):
            continue
        raw = row.get("raw_model_output")
        original_error = row.get("error")
        try:
            revised, method = recover_revision_output_format(raw)
        except Exception as recovery_error:
            unit = unit_by_id.get(str(row.get("response_id")))
            if unit is None:
                continue
            revised = str(unit["base_response"])
            if sha256_text(revised) != row.get("base_response_sha256"):
                raise ValueError(
                    "Cannot apply revision fallback because the branch-base "
                    f"hash differs for {row.get('response_id')}"
                )
            row.update(
                {
                    "status": "ok",
                    "revised_response": revised,
                    "revised_response_sha256": sha256_text(revised),
                    "error": None,
                    "revision_parse_mode": "base_response_fallback",
                    "format_recovery_method": None,
                    "original_parse_error": original_error,
                    "format_recovery_error": (
                        f"{type(recovery_error).__name__}: {recovery_error}"
                    ),
                    "fallback_applied": True,
                    "fallback_policy_version": (
                        REVISION_FALLBACK_POLICY_VERSION
                    ),
                    "fallback_reason": "unrecoverable_model_revision_output",
                    "model_output_usable_as_revision": False,
                    "format_recovered_at": utc_now(),
                }
            )
        else:
            row.update(
                {
                    "status": "ok",
                    "revised_response": revised,
                    "revised_response_sha256": sha256_text(revised),
                    "error": None,
                    "revision_parse_mode": "format_recovered",
                    "format_recovery_method": method,
                    "original_parse_error": original_error,
                    "format_recovery_error": None,
                    "fallback_applied": False,
                    "fallback_policy_version": None,
                    "fallback_reason": None,
                    "model_output_usable_as_revision": True,
                    "format_recovered_at": utc_now(),
                }
            )
        recovered_count += 1
    if recovered_count:
        atomic_write_jsonl(output_path, rows)
        print(
            f"[recovery] normalized {recovered_count} existing revision "
            f"parse_error row(s) without a model call"
        )
    return recovered_count


def _run_resumable(
    *,
    units: list[dict[str, Any]],
    output_path: Path,
    key: str,
    run_fingerprint: str,
    resume: bool,
    label: str,
    process: Callable[[dict[str, Any]], dict[str, Any]],
    max_consecutive_errors: int,
) -> list[dict[str, Any]]:
    existing = _rows_or_empty(output_path)
    if existing and not resume:
        raise FileExistsError(f"Output exists; use --resume: {output_path}")
    expected_ids = {unit[key] for unit in units}
    by_id: dict[str, dict[str, Any]] = {}
    for row in existing:
        row_id = row.get(key)
        if row_id not in expected_ids or row_id in by_id:
            raise ValueError(f"Incompatible existing {label} row: {row_id}")
        if row.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                f"Existing {label} fingerprint differs; refusing contamination"
            )
        by_id[str(row_id)] = row
    pending = [
        unit
        for unit in units
        if by_id.get(str(unit[key]), {}).get("status") != "ok"
    ]
    print(
        f"{label}: total={len(units)}, retained_ok={len(units)-len(pending)}, "
        f"pending={len(pending)}",
        flush=True,
    )
    order = {str(unit[key]): index for index, unit in enumerate(units)}
    consecutive = 0
    for unit in pending:
        row_id = str(unit[key])
        position = order[row_id] + 1
        print(f"[{position}/{len(units)}] {label} {row_id} ...", flush=True)
        result = process(unit)
        by_id[row_id] = result
        ordered = [
            by_id[str(item[key])]
            for item in units
            if str(item[key]) in by_id
        ]
        atomic_write_jsonl(output_path, ordered)
        if result["status"] == "ok":
            consecutive = 0
            print(
                f"[{position}/{len(units)}] success, "
                f"{result['latency_seconds']:.1f}s",
                flush=True,
            )
        else:
            consecutive = (
                consecutive + 1
                if result["status"] == "request_error"
                else 0
            )
            print(
                f"[{position}/{len(units)}] {result['status']}: "
                f"{result['error']}",
                flush=True,
            )
            if consecutive >= max_consecutive_errors:
                print(
                    "Stopping at the consecutive request-error limit; rerun "
                    "the same command with --resume.",
                    file=sys.stderr,
                    flush=True,
                )
                break
    return [
        by_id[str(unit[key])]
        for unit in units
        if str(unit[key]) in by_id
    ]


def run_grounded_answers(
    split: str,
    *,
    resume: bool,
    dry_run: bool,
    ollama_host: str,
) -> int:
    config_path, config = load_branch_config()
    _, question_units = frozen_question_units(split)
    evidence_path = grounded_evidence_path(split)
    evidence_rows = load_jsonl(evidence_path)
    evidence_by_id = {row["question_id"]: row for row in evidence_rows}
    if set(evidence_by_id) != {row["question_id"] for row in question_units}:
        raise ValueError("Branch B grounded evidence is incomplete")
    settings = config["branches"]["b"]
    generation = config["generation"]
    prompt_path, template = _load_prompt(
        settings["answer_prompt_path"],
        ("{verification_question_json}", "{retrieved_evidence_text}"),
    )
    top_k = int(settings["evidence_top_k"])
    payload = {
        "stage": "branch_b_grounded_verification_answers",
        "split": split,
        "model": generation["model"],
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "num_predict": generation["answer_num_predict"],
        "prompt_version": settings["answer_prompt_version"],
        "prompt_sha256": sha256_file(prompt_path),
        "schema_sha256": canonical_json_hash(answer_schema(top_k)),
        "branch_config_sha256": sha256_file(config_path),
        "question_results_sha256": sha256_file(
            cove_paths(PROJECT_ROOT, "full", "a").question_results(split)
        ),
        "grounded_evidence_sha256": sha256_file(evidence_path),
        "model_input_fields": [
            "frozen_verification_question",
            "retrieved_passage_text",
        ],
        "withheld_fields": settings["answer_withheld_fields"],
    }
    if dry_run:
        print(
            json.dumps(
                {
                    **payload,
                    "question_calls": len(question_units),
                    "run_fingerprint": canonical_json_hash(payload),
                    "output": project_relative(grounded_answers_path(split)),
                    "dry_run": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    client = Client(
        host=ollama_host,
        timeout=float(generation["answer_timeout_seconds"]),
    )
    digest = preflight_model(client, generation)
    payload["model_digest"] = digest
    run_fingerprint = canonical_json_hash(payload)

    def process(unit: dict[str, Any]) -> dict[str, Any]:
        evidence = evidence_by_id[unit["question_id"]]
        prompt = template.replace(
            "{verification_question_json}",
            json.dumps(unit["verification_question"], ensure_ascii=False),
        ).replace(
            "{retrieved_evidence_text}",
            evidence["normalized_text"],
        )
        raw: str | None = None
        metadata: dict[str, Any] = {}
        request_error: Exception | None = None
        attempts = 0
        started = time.perf_counter()
        for attempt in range(int(generation["max_retries"]) + 1):
            attempts = attempt + 1
            try:
                raw, metadata = call_model(
                    client,
                    generation,
                    prompt,
                    answer_schema(top_k),
                    num_predict=int(generation["answer_num_predict"]),
                )
                request_error = None
                break
            except Exception as error:
                request_error = error
                if attempt < int(generation["max_retries"]):
                    time.sleep(1)
        base = {
            "result_schema_version": settings["answer_result_schema_version"],
            **{key: value for key, value in unit.items() if key != "text"},
            "branch_id": "b",
            "stage": payload["stage"],
            "contextually_independent_call": True,
            "model_input_fields": payload["model_input_fields"],
            "withheld_fields": payload["withheld_fields"],
            "evidence_sha256": evidence["normalized_sha256"],
            "evidence_top_k": top_k,
            "retriever": "hybrid_rrf",
            "run_fingerprint": run_fingerprint,
            "model": generation["model"],
            "model_digest": digest,
            "attempts": attempts,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "raw_model_output": raw,
            "ollama_metadata": metadata,
            "created_at": utc_now(),
        }
        if request_error is not None:
            return {
                **base,
                "status": "request_error",
                "answer_status": None,
                "verification_answer": None,
                "cited_passage_ranks": None,
                "error": f"{type(request_error).__name__}: {request_error}",
            }
        try:
            parsed = parse_grounded_answer(raw or "", top_k)
        except Exception as error:
            return {
                **base,
                "status": "parse_error",
                "answer_status": None,
                "verification_answer": None,
                "cited_passage_ranks": None,
                "error": f"{type(error).__name__}: {error}",
            }
        return {**base, "status": "ok", **parsed, "error": None}

    results = _run_resumable(
        units=question_units,
        output_path=grounded_answers_path(split),
        key="question_id",
        run_fingerprint=run_fingerprint,
        resume=resume,
        label="Branch B grounded answer",
        process=process,
        max_consecutive_errors=int(
            generation["max_consecutive_request_errors"]
        ),
    )
    complete = len(results) == len(question_units) and all(
        row["status"] == "ok" for row in results
    )
    summary = {
        "stage": payload["stage"],
        "status": "complete" if complete else "incomplete",
        "split": split,
        "question_count": len(question_units),
        "ok_count": sum(row["status"] == "ok" for row in results),
        "answer_status_counts": Counter(
            row.get("answer_status")
            for row in results
            if row.get("status") == "ok"
        ),
        "run_fingerprint": run_fingerprint,
        "output": project_relative(grounded_answers_path(split)),
    }
    atomic_write_json(branch_report_path("b", split), summary)
    return 0 if complete else 2


def _revision_units(
    branch: str,
    split: str,
) -> tuple[list[dict[str, Any]], Path, str, list[str], list[str]]:
    _, config = load_branch_config()
    units: list[dict[str, Any]] = []
    if branch == "b":
        _, responses = load_shared_responses(split)
        answers_path = grounded_answers_path(split)
        answer_rows = load_jsonl(answers_path)
        if any(row.get("status") != "ok" for row in answer_rows):
            raise ValueError("Finish Branch B grounded answers before revision")
        answers_by_response: defaultdict[str, list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for row in answer_rows:
            answers_by_response[row["response_id"]].append(row)
        for source in responses:
            rows = sorted(
                answers_by_response[source["response_id"]],
                key=lambda row: int(row["question_index"]),
            )
            if not rows:
                raise ValueError(
                    f"No Branch B answers for {source['response_id']}"
                )
            verification_results = [
                {
                    "question_id": row["question_id"],
                    "verification_question": row["verification_question"],
                    "verification_answer": row["verification_answer"],
                }
                for row in rows
            ]
            units.append(
                {
                    "response_id": source["response_id"],
                    "source_record_index": source["source_record_index"],
                    "split": split,
                    "original_question": source["original_question"],
                    "base_response": source["initial_response"],
                    "base_response_sha256": source["initial_response_sha256"],
                    "verification_results": verification_results,
                    "intervention_sha256": sha256_text(
                        json.dumps(
                            verification_results,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
                }
            )
        settings = config["branches"]["b"]
        return (
            units,
            grounded_answers_path(split),
            settings["revision_prompt_path"],
            [
                "original_question",
                "initial_response",
                "frozen_verification_questions",
                "branch_b_grounded_answers",
            ],
            settings["revision_withheld_fields"],
        )
    if branch == "c":
        _, responses, a_revisions = load_branch_a_revisions(split)
        source_by_id = {row["response_id"]: row for row in responses}
        a_revision_by_id = {row["response_id"]: row for row in a_revisions}
        for response_id, source in source_by_id.items():
            a_row = a_revision_by_id[response_id]
            units.append(
                {
                    "response_id": response_id,
                    "source_record_index": source["source_record_index"],
                    "split": split,
                    "original_question": source["original_question"],
                    "base_response": a_row["revised_response"],
                    "base_response_sha256": a_row["revised_response_sha256"],
                    "intervention_sha256": sha256_text(
                        "generic_extra_revision_control_v1"
                    ),
                }
            )
        settings = config["branches"]["c"]
        return (
            units,
            cove_paths(PROJECT_ROOT, "full", "a").revision_results(split),
            settings["revision_prompt_path"],
            ["original_question", "frozen_branch_a_response"],
            settings["withheld_fields"],
        )
    if branch == "d2":
        _, responses, a_revisions = load_branch_a_revisions(split)
        source_by_id = {row["response_id"]: row for row in responses}
        a_revision_by_id = {row["response_id"]: row for row in a_revisions}
        feedback_path = bounded_targeted_feedback_path(split)
        feedback_rows = load_jsonl(feedback_path)
        feedback_by_response = {
            row["response_id"]: row for row in feedback_rows
        }
        if set(feedback_by_response) != set(source_by_id):
            raise ValueError(
                "Branch D v2 compact feedback must contain every response"
            )
        for response_id, source in source_by_id.items():
            a_row = a_revision_by_id[response_id]
            feedback_row = feedback_by_response[response_id]
            compact_feedback = feedback_row["compact_feedback"]
            units.append(
                {
                    "response_id": response_id,
                    "source_record_index": source["source_record_index"],
                    "split": split,
                    "original_question": source["original_question"],
                    "base_response": a_row["revised_response"],
                    "base_response_sha256": a_row["revised_response_sha256"],
                    "compact_feedback": compact_feedback,
                    "selected_feedback_count": feedback_row[
                        "selected_feedback_count"
                    ],
                    "omitted_feedback_count": feedback_row[
                        "omitted_feedback_count"
                    ],
                    "visible_evidence_character_count": feedback_row[
                        "visible_evidence_character_count"
                    ],
                    "intervention_sha256": feedback_row[
                        "compact_feedback_sha256"
                    ],
                }
            )
        settings = branch_settings(config, branch)
        return (
            units,
            feedback_path,
            settings["revision_prompt_path"],
            [
                "original_question",
                "frozen_branch_a_response",
                "bounded_revision_targets",
            ],
            settings["withheld_fields"],
        )
    raise ValueError("Revision generation is implemented for branches b/c/d2")


def run_branch_revision(
    branch: str,
    split: str,
    *,
    resume: bool,
    dry_run: bool,
    ollama_host: str,
) -> int:
    config_path, config = load_branch_config()
    units, intervention_path, prompt_relative, input_fields, withheld = (
        _revision_units(branch, split)
    )
    generation = config["generation"]
    settings = branch_settings(config, branch)
    if branch == "b":
        prompt_path, template = _load_prompt(
            prompt_relative,
            (
                "{original_question_json}",
                "{initial_response_json}",
                "{verification_results_json}",
            ),
        )
    elif branch == "c":
        prompt_path, template = _load_prompt(
            prompt_relative,
            ("{original_question_json}", "{branch_a_response_json}"),
        )
    else:
        prompt_path, template = _load_prompt(
            prompt_relative,
            (
                "{original_question_json}",
                "{branch_a_response_json}",
                "{compact_feedback_json}",
            ),
        )
    branch_a_revision_path = cove_paths(
        PROJECT_ROOT, "full", "a"
    ).revision_results(split)
    payload = {
        "stage": f"branch_{branch}_{settings['name']}_revision",
        "branch_id": branch,
        "split": split,
        "revision_base": settings["revision_base"],
        "model": generation["model"],
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "num_predict": generation["revision_num_predict"],
        "prompt_version": settings["revision_prompt_version"],
        "prompt_sha256": sha256_file(prompt_path),
        "schema_sha256": canonical_json_hash(revision_schema()),
        "branch_config_sha256": sha256_file(config_path),
        "branch_a_revision_sha256": (
            sha256_file(branch_a_revision_path)
            if branch in {"c", "d2"}
            else None
        ),
        "intervention_artifact_sha256": sha256_file(intervention_path),
        "model_input_fields": input_fields,
        "withheld_fields": withheld,
    }
    if branch == "d2":
        payload["branch_d2_config_sha256"] = sha256_file(
            branch_d2_config_path()
        )
    if dry_run:
        print(
            json.dumps(
                {
                    **payload,
                    "response_calls": len(units),
                    "run_fingerprint": canonical_json_hash(payload),
                    "output": project_relative(
                        cove_paths(
                            PROJECT_ROOT, "full", branch
                        ).revision_results(split)
                    ),
                    "dry_run": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    client = Client(
        host=ollama_host,
        timeout=float(generation["revision_timeout_seconds"]),
    )
    digest = preflight_model(client, generation)
    payload["model_digest"] = digest
    run_fingerprint = canonical_json_hash(payload)

    def process(unit: dict[str, Any]) -> dict[str, Any]:
        prompt = template.replace(
            "{original_question_json}",
            json.dumps(unit["original_question"], ensure_ascii=False),
        )
        if branch == "b":
            prompt = prompt.replace(
                "{initial_response_json}",
                json.dumps(unit["base_response"], ensure_ascii=False),
            ).replace(
                "{verification_results_json}",
                json.dumps(
                    unit["verification_results"],
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        elif branch == "c":
            prompt = prompt.replace(
                "{branch_a_response_json}",
                json.dumps(unit["base_response"], ensure_ascii=False),
            )
        elif branch == "d":
            prompt = prompt.replace(
                "{branch_a_response_json}",
                json.dumps(unit["base_response"], ensure_ascii=False),
            ).replace(
                "{selected_feedback_json}",
                json.dumps(
                    unit["selected_feedback"],
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        else:
            prompt = template.replace(
                "{original_question_json}",
                json.dumps(unit["original_question"], ensure_ascii=False),
            ).replace(
                "{branch_a_response_json}",
                json.dumps(unit["base_response"], ensure_ascii=False),
            ).replace(
                "{compact_feedback_json}",
                json.dumps(
                    unit["compact_feedback"],
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        raw: str | None = None
        metadata: dict[str, Any] = {}
        request_error: Exception | None = None
        attempts = 0
        started = time.perf_counter()
        for attempt in range(int(generation["max_retries"]) + 1):
            attempts = attempt + 1
            try:
                raw, metadata = call_model(
                    client,
                    generation,
                    prompt,
                    revision_schema(),
                    num_predict=int(generation["revision_num_predict"]),
                )
                request_error = None
                break
            except Exception as error:
                request_error = error
                if attempt < int(generation["max_retries"]):
                    time.sleep(1)
        base = {
            "result_schema_version": settings[
                "revision_result_schema_version"
            ],
            "branch_id": branch,
            "branch_name": settings["name"],
            "response_id": unit["response_id"],
            "source_record_index": unit["source_record_index"],
            "split": split,
            "original_question": unit["original_question"],
            "original_question_sha256": sha256_text(
                unit["original_question"]
            ),
            "revision_base": settings["revision_base"],
            "base_response_sha256": unit["base_response_sha256"],
            "intervention_sha256": unit["intervention_sha256"],
            "model_input_fields": input_fields,
            "withheld_fields": withheld,
            "stage": payload["stage"],
            "run_fingerprint": run_fingerprint,
            "model": generation["model"],
            "model_digest": digest,
            "temperature": generation["temperature"],
            "seed": generation["seed"],
            "num_predict": generation["revision_num_predict"],
            "think": generation["think"],
            "prompt_version": settings["revision_prompt_version"],
            "prompt_sha256": sha256_file(prompt_path),
            "attempts": attempts,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "raw_model_output": raw,
            "ollama_metadata": metadata,
            "created_at": utc_now(),
        }
        if branch == "d2":
            base["selected_feedback_count"] = len(
                unit.get("selected_feedback", unit.get("compact_feedback", []))
            )
        if branch == "d2":
            base["source_selected_feedback_count"] = unit[
                "selected_feedback_count"
            ]
            base["omitted_feedback_count"] = unit[
                "omitted_feedback_count"
            ]
            base["visible_evidence_character_count"] = unit[
                "visible_evidence_character_count"
            ]
        if branch == "b":
            base["verification_question_count"] = len(
                unit["verification_results"]
            )
        if request_error is not None:
            return {
                **base,
                "status": "request_error",
                "revised_response": None,
                "revised_response_sha256": None,
                "error": f"{type(request_error).__name__}: {request_error}",
            }
        revision_parse_mode = "strict"
        format_recovery_method: str | None = None
        original_parse_error: str | None = None
        format_recovery_error: str | None = None
        fallback_applied = False
        try:
            revised = parse_revision(raw or "")
        except Exception as strict_error:
            original_parse_error = (
                f"{type(strict_error).__name__}: {strict_error}"
            )
            try:
                revised, format_recovery_method = (
                    recover_revision_output_format(raw or "")
                )
                revision_parse_mode = "format_recovered"
            except Exception as recovery_error:
                format_recovery_error = (
                    f"{type(recovery_error).__name__}: {recovery_error}"
                )
                revised = unit["base_response"]
                revision_parse_mode = "base_response_fallback"
                fallback_applied = True
        return {
            **base,
            "status": "ok",
            "revised_response": revised,
            "revised_response_sha256": sha256_text(revised),
            "revision_parse_mode": revision_parse_mode,
            "format_recovery_method": format_recovery_method,
            "original_parse_error": original_parse_error,
            "format_recovery_error": format_recovery_error,
            "fallback_applied": fallback_applied,
            "fallback_policy_version": (
                REVISION_FALLBACK_POLICY_VERSION
                if fallback_applied
                else None
            ),
            "fallback_reason": (
                "unrecoverable_model_revision_output"
                if fallback_applied
                else None
            ),
            "model_output_usable_as_revision": not fallback_applied,
            "error": None,
        }

    output_path = cove_paths(
        PROJECT_ROOT, "full", branch
    ).revision_results(split)
    recovered_existing_count = (
        recover_existing_revision_parse_errors(
            output_path,
            run_fingerprint,
            units,
        )
        if resume
        else 0
    )
    results = _run_resumable(
        units=units,
        output_path=output_path,
        key="response_id",
        run_fingerprint=run_fingerprint,
        resume=resume,
        label=f"Branch {branch.upper()} revision",
        process=process,
        max_consecutive_errors=int(
            generation["max_consecutive_request_errors"]
        ),
    )
    complete = len(results) == len(units) and all(
        row["status"] == "ok" for row in results
    )
    usable_revision_count = sum(
        row.get("status") == "ok"
        and row.get("model_output_usable_as_revision", True) is True
        for row in results
    )
    intervention_valid = complete and usable_revision_count == len(units)
    unchanged = sum(
        row.get("status") == "ok"
        and row.get("revised_response_sha256") == row.get("base_response_sha256")
        for row in results
    )
    summary = {
        "stage": payload["stage"],
        "status": (
            "complete"
            if complete and (branch != "d2" or intervention_valid)
            else (
                "intervention_execution_failed"
                if branch == "d2" and complete
                else "incomplete"
            )
        ),
        "branch_id": branch,
        "branch_name": settings["name"],
        "split": split,
        "response_count": len(units),
        "ok_count": sum(row["status"] == "ok" for row in results),
        "unchanged_from_branch_base_count": unchanged,
        "strict_parse_count": sum(
            row.get("status") == "ok"
            and row.get("revision_parse_mode", "strict") == "strict"
            for row in results
        ),
        "format_recovered_count": sum(
            row.get("status") == "ok"
            and row.get("revision_parse_mode") == "format_recovered"
            for row in results
        ),
        "base_response_fallback_count": sum(
            row.get("status") == "ok"
            and row.get("revision_parse_mode") == "base_response_fallback"
            for row in results
        ),
        "model_output_usable_as_revision_count": usable_revision_count,
        "intervention_execution_valid": intervention_valid,
        "revision_fallback_policy_version": (
            REVISION_FALLBACK_POLICY_VERSION
        ),
        "recovered_existing_count": recovered_existing_count,
        "format_recovery_method_counts": Counter(
            row.get("format_recovery_method")
            for row in results
            if row.get("status") == "ok"
            and row.get("format_recovery_method") is not None
        ),
        "run_fingerprint": run_fingerprint,
        "output": project_relative(output_path),
    }
    atomic_write_json(branch_report_path(branch, split), summary)
    return 0 if complete and (branch != "d2" or intervention_valid) else 2


def prepare_targeted_feedback_candidates(split: str, *, dry_run: bool) -> int:
    config_path, config = load_branch_config()
    settings = config["branches"]["d"]
    a_paths = cove_paths(PROJECT_ROOT, "full", "a")
    factuality = load_jsonl(a_paths.revised_claim_factuality_results(split))
    evidence = load_jsonl(a_paths.revised_claim_evidence(split))
    audit = load_jsonl(a_paths.factuality_audit_manifest(split))
    evidence_by_id = {row["revised_claim_id"]: row for row in evidence}
    audit_by_id = {row["revised_claim_id"]: row for row in audit}
    factuality_ids = {row["revised_claim_id"] for row in factuality}
    if set(evidence_by_id) != factuality_ids or set(audit_by_id) != factuality_ids:
        raise ValueError("Branch D prepared ingredients have different claim sets")
    minimum_confidence = float(settings["minimum_confidence"])
    minimum_passages = int(settings["minimum_evidence_passages"])
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for prediction in factuality:
        claim_id = prediction["revised_claim_id"]
        evidence_row = evidence_by_id[claim_id]
        audit_row = audit_by_id[claim_id]
        rejection: list[str] = []
        if prediction.get("status") != "ok":
            rejection.append("verifier_not_ok")
        if prediction.get("prediction") != settings["eligible_prediction"]:
            rejection.append("prediction_not_non_factual")
        confidence = prediction.get("confidence")
        if not isinstance(confidence, (int, float)) or float(
            confidence
        ) < minimum_confidence:
            rejection.append("confidence_below_threshold")
        flags = audit_row.get("deterministic_flags")
        if settings["require_no_deterministic_policy_flags"] and flags:
            rejection.append("deterministic_policy_flag")
        items = evidence_row.get("items")
        if not isinstance(items, list) or len(items) < minimum_passages:
            rejection.append("insufficient_retrieved_passages")
        selected = not rejection
        if selected:
            reasons["selected"] += 1
        else:
            reasons.update(rejection)
        visible_passages = [
            {
                "rank": item["rank"],
                "passage_id": item["passage_id"],
                "doc_id": item["doc_id"],
                "text": item["text"],
            }
            for item in (items or [])
        ]
        rows.append(
            {
                "schema_version": "cove_branch_d_selective_feedback_v1",
                "branch_id": "d",
                "response_id": prediction["response_id"],
                "source_record_index": prediction["source_record_index"],
                "split": split,
                "revised_claim_id": claim_id,
                "revised_claim": prediction["revised_claim"],
                "revised_claim_sha256": prediction["revised_claim_sha256"],
                "selected_for_feedback": selected,
                "selection_policy_version": settings[
                    "feedback_policy_version"
                ],
                "selection_rejection_reasons": rejection,
                "verifier_prediction": prediction.get("prediction"),
                "verifier_confidence": prediction.get("confidence"),
                "verifier_rationale": prediction.get("rationale"),
                "retrieved_passages": visible_passages,
                "retrieved_evidence_sha256": evidence_row[
                    "normalized_sha256"
                ],
                "deterministic_policy_flags": flags,
                "model_visible_if_selected": [
                    "revised_claim_id",
                    "revised_claim",
                    "verifier_prediction",
                    "verifier_confidence",
                    "verifier_rationale",
                    "retrieved_passage_rank",
                    "retrieved_passage_text",
                ],
                "withheld_fields": settings["withheld_fields"],
                "source_hashes": {
                    "B6c_factuality_sha256": sha256_file(
                        a_paths.revised_claim_factuality_results(split)
                    ),
                    "B6c_evidence_sha256": sha256_file(
                        a_paths.revised_claim_evidence(split)
                    ),
                    "B6d_audit_sha256": sha256_file(
                        a_paths.factuality_audit_manifest(split)
                    ),
                    "branch_config_sha256": sha256_file(config_path),
                },
                "created_at": utc_now(),
            }
        )
    summary = {
        "stage": "branch_d_prepare_selective_verifier_feedback",
        "status": "validated" if dry_run else "complete",
        "split": split,
        "candidate_claim_count": len(rows),
        "selected_claim_count": sum(
            row["selected_for_feedback"] for row in rows
        ),
        "responses_with_selected_feedback": len(
            {
                row["response_id"]
                for row in rows
                if row["selected_for_feedback"]
            }
        ),
        "selection_reason_counts": dict(sorted(reasons.items())),
        "policy": {
            "eligible_prediction": settings["eligible_prediction"],
            "minimum_confidence": minimum_confidence,
            "require_no_deterministic_policy_flags": settings[
                "require_no_deterministic_policy_flags"
            ],
            "minimum_evidence_passages": minimum_passages,
            "unknown_action": settings["unknown_action"],
        },
        "output": project_relative(targeted_feedback_candidates_path(split)),
        "gold_fields_used": [],
        "dry_run": dry_run,
    }
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    atomic_write_jsonl(targeted_feedback_candidates_path(split), rows)
    atomic_write_json(branch_report_path("d", split), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _bounded_visible_text(value: Any, maximum_characters: int) -> tuple[str, bool]:
    """Normalize whitespace and truncate model-visible text deterministically."""

    normalized = " ".join(str(value or "").split())
    if len(normalized) <= maximum_characters:
        return normalized, False
    cut = normalized.rfind(" ", 0, maximum_characters - 1)
    if cut < maximum_characters // 2:
        cut = maximum_characters - 1
    return normalized[:cut].rstrip() + "…", True


def _passage_ranks_mentioned_in_rationale(rationale: str) -> list[int]:
    ranks: list[int] = []
    for match in re.finditer(
        r"\bpassages?\s*(?:number\s*)?#?\s*(\d+)\b",
        rationale,
        flags=re.IGNORECASE,
    ):
        rank = int(match.group(1))
        if rank not in ranks:
            ranks.append(rank)
    return ranks


def prepare_bounded_targeted_feedback(
    split: str,
    *,
    dry_run: bool,
) -> int:
    """Create a bounded, response-level Branch D v2 intervention artifact.

    This stage makes no model calls. It preserves the frozen v1 eligibility
    decisions, then deterministically limits how much of that feedback a
    revision call can see. The v1 artifact remains untouched.
    """

    config_path, config = load_branch_config()
    settings = branch_settings(config, "d2")
    source_path = targeted_feedback_candidates_path(split)
    if not source_path.exists():
        raise FileNotFoundError(
            "Prepare the frozen Branch D v1 feedback before Branch D v2: "
            f"{source_path}"
        )
    source_rows = load_jsonl(source_path)
    _, responses, _ = load_branch_a_revisions(split)
    expected_response_ids = {row["response_id"] for row in responses}
    unexpected = {
        row.get("response_id") for row in source_rows
    } - expected_response_ids
    if unexpected:
        raise ValueError(
            f"Legacy feedback contains unexpected responses: {sorted(unexpected)}"
        )

    selected_by_response: defaultdict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in source_rows:
        if row.get("selected_for_feedback") is True:
            selected_by_response[str(row["response_id"])].append(row)
    source_selected_rows = [
        row
        for rows in selected_by_response.values()
        for row in rows
    ]
    source_selected_passage_count = sum(
        len(row.get("retrieved_passages") or [])
        for row in source_selected_rows
    )
    source_selected_evidence_character_count = sum(
        len(str(passage.get("text") or ""))
        for row in source_selected_rows
        for passage in (row.get("retrieved_passages") or [])
        if isinstance(passage, dict)
    )

    max_claims = int(settings["maximum_feedback_claims_per_response"])
    max_passages = int(settings["maximum_passages_per_claim"])
    max_passage_chars = int(settings["maximum_passage_characters"])
    max_rationale_chars = int(settings["maximum_rationale_characters"])
    output_rows: list[dict[str, Any]] = []
    total_visible_targets = 0
    total_omitted_targets = 0
    total_visible_passages = 0
    total_visible_evidence_characters = 0

    for response in responses:
        response_id = response["response_id"]
        candidates = sorted(
            selected_by_response[response_id],
            key=lambda row: (
                -float(row.get("verifier_confidence") or 0.0),
                str(row["revised_claim_id"]),
            ),
        )
        visible_candidates = candidates[:max_claims]
        omitted_candidates = candidates[max_claims:]
        compact_feedback: list[dict[str, Any]] = []
        visible_evidence_characters = 0

        for target_index, row in enumerate(visible_candidates, start=1):
            rationale, rationale_truncated = _bounded_visible_text(
                row.get("verifier_rationale"),
                max_rationale_chars,
            )
            passages = row.get("retrieved_passages")
            if not isinstance(passages, list) or not passages:
                raise ValueError(
                    "Selected feedback candidate has no passages: "
                    f"{row['revised_claim_id']}"
                )
            by_rank = {
                int(item["rank"]): item
                for item in passages
                if isinstance(item, dict)
                and isinstance(item.get("rank"), int)
            }
            preferred_ranks = [
                rank
                for rank in _passage_ranks_mentioned_in_rationale(rationale)
                if rank in by_rank
            ]
            remaining_ranks = [
                rank for rank in sorted(by_rank) if rank not in preferred_ranks
            ]
            chosen_ranks = (preferred_ranks + remaining_ranks)[:max_passages]
            excerpts: list[dict[str, Any]] = []
            for rank in chosen_ranks:
                passage = by_rank[rank]
                text, text_truncated = _bounded_visible_text(
                    passage.get("text"),
                    max_passage_chars,
                )
                visible_evidence_characters += len(text)
                excerpts.append(
                    {
                        "passage_rank": rank,
                        "passage_id": passage["passage_id"],
                        "text": text,
                        "text_truncated": text_truncated,
                    }
                )
            compact_feedback.append(
                {
                    "target_index": target_index,
                    "target_claim_id": row["revised_claim_id"],
                    "target_claim": row["revised_claim"],
                    "requested_action": (
                        "make_the_smallest_supported_correction_or_delete"
                    ),
                    "warning_reason": rationale,
                    "warning_reason_truncated": rationale_truncated,
                    "evidence_excerpts": excerpts,
                }
            )

        compact_sha256 = sha256_text(
            json.dumps(
                compact_feedback,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        output_rows.append(
            {
                "schema_version": "cove_branch_d2_compact_feedback_v1",
                "branch_id": "d2",
                "response_id": response_id,
                "source_record_index": response["source_record_index"],
                "split": split,
                "source_selected_feedback_count": len(candidates),
                "selected_feedback_count": len(compact_feedback),
                "omitted_feedback_count": len(omitted_candidates),
                "omitted_feedback_claim_ids": [
                    row["revised_claim_id"] for row in omitted_candidates
                ],
                "compact_feedback": compact_feedback,
                "visible_passage_count": sum(
                    len(item["evidence_excerpts"])
                    for item in compact_feedback
                ),
                "visible_evidence_character_count": (
                    visible_evidence_characters
                ),
                "compact_feedback_sha256": compact_sha256,
                "source_feedback_sha256": sha256_file(source_path),
                "branch_config_sha256": sha256_file(config_path),
                "branch_d2_config_sha256": sha256_file(
                    branch_d2_config_path()
                ),
                "feedback_policy_version": settings[
                    "feedback_policy_version"
                ],
                "gold_fields_used": [],
            }
        )
        total_visible_targets += len(compact_feedback)
        total_omitted_targets += len(omitted_candidates)
        total_visible_passages += sum(
            len(item["evidence_excerpts"]) for item in compact_feedback
        )
        total_visible_evidence_characters += visible_evidence_characters

    output_path = bounded_targeted_feedback_path(split)
    summary = {
        "stage": "branch_d_prepare_bounded_targeted_feedback",
        "status": "validated" if dry_run else "complete",
        "split": split,
        "response_count": len(output_rows),
        "responses_with_visible_feedback": sum(
            bool(row["compact_feedback"]) for row in output_rows
        ),
        "source_selected_target_count": sum(
            row["source_selected_feedback_count"] for row in output_rows
        ),
        "source_selected_passage_count": source_selected_passage_count,
        "source_selected_evidence_character_count": (
            source_selected_evidence_character_count
        ),
        "visible_target_count": total_visible_targets,
        "omitted_target_count": total_omitted_targets,
        "visible_passage_count": total_visible_passages,
        "visible_evidence_character_count": total_visible_evidence_characters,
        "visible_evidence_character_reduction_fraction": (
            round(
                1.0
                - total_visible_evidence_characters
                / source_selected_evidence_character_count,
                6,
            )
            if source_selected_evidence_character_count
            else 0.0
        ),
        "policy": {
            field: settings[field]
            for field in (
                "feedback_policy_version",
                "maximum_feedback_claims_per_response",
                "maximum_passages_per_claim",
                "maximum_passage_characters",
                "maximum_rationale_characters",
                "passage_selection",
                "claim_priority",
            )
        },
        "source": project_relative(source_path),
        "source_sha256": sha256_file(source_path),
        "branch_config_sha256": sha256_file(config_path),
        "branch_d2_config_sha256": sha256_file(branch_d2_config_path()),
        "output": project_relative(output_path),
        "gold_fields_used": [],
        "dry_run": dry_run,
    }
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    atomic_write_jsonl(output_path, output_rows)
    report_path = (
        cove_paths(PROJECT_ROOT, "full", "d2").reports_dir
        / f"branch_d2_feedback_{split}_summary.json"
    )
    atomic_write_json(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _branch_outcome_summary(branch: str, split: str) -> dict[str, Any]:
    paths = cove_paths(PROJECT_ROOT, "full", branch)
    revision_state = _artifact_state(
        paths.revision_results(split),
        expected_rows=20 if split == "dev" else 72,
    )
    initial_path = paths.initial_claim_outcomes(split)
    added_path = paths.added_claim_outcomes(split)
    result: dict[str, Any] = {
        "branch_id": branch,
        "revision": revision_state,
        "evaluation_complete": initial_path.exists() and added_path.exists(),
    }
    if result["evaluation_complete"]:
        initial_rows = load_jsonl(initial_path)
        added_rows = load_jsonl(added_path)
        result["initial_claim_count"] = len(initial_rows)
        result["initial_outcome_counts"] = dict(
            sorted(
                Counter(
                    row.get("provisional_outcome") for row in initial_rows
                ).items()
            )
        )
        result["added_claim_count"] = len(added_rows)
        result["added_outcome_counts"] = dict(
            sorted(
                Counter(
                    row.get("provisional_outcome") for row in added_rows
                ).items()
            )
        )
        result["evaluation_tier"] = "same_protocol_silver_llm_assisted"
        result["formal_net_gain_eligible"] = False
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _preserved_outcome_count(counts: Mapping[str, int]) -> int:
    return counts.get("FACTUAL_RETAINED_CANDIDATE", 0) + counts.get(
        "FACTUAL_PRESERVED_AFTER_CHANGE_CANDIDATE",
        0,
    )


def _research_branch_metrics(branch: str, split: str) -> dict[str, Any]:
    paths = cove_paths(PROJECT_ROOT, "full", branch)
    predictions = load_jsonl(
        paths.revised_claim_factuality_results(split)
    )
    initial = load_jsonl(paths.initial_claim_outcomes(split))
    added = load_jsonl(paths.added_claim_outcomes(split))
    prediction_counts = Counter(
        row.get("prediction")
        for row in predictions
        if row.get("status") == "ok"
    )
    response_predictions: defaultdict[str, Counter[str]] = defaultdict(
        Counter
    )
    for row in predictions:
        if row.get("status") == "ok":
            response_predictions[row["response_id"]][
                str(row["prediction"])
            ] += 1
    response_resolved_proportions = []
    for counts in response_predictions.values():
        resolved = counts["FACTUAL"] + counts["NON_FACTUAL"]
        if resolved:
            response_resolved_proportions.append(
                counts["FACTUAL"] / resolved
            )
    initial_counts = Counter(
        row.get("provisional_outcome") for row in initial
    )
    added_counts = Counter(
        row.get("provisional_outcome") for row in added
    )
    human_label_counts = Counter(row.get("human_label") for row in initial)
    non_factual_total = human_label_counts["NON_FACTUAL"]
    factual_total = human_label_counts["FACTUAL"]
    corrected = initial_counts["ERROR_CORRECTED_CANDIDATE"]
    error_deleted = initial_counts[
        "ERROR_REMOVED_BY_DELETION_CANDIDATE"
    ]
    factual_preserved = (
        initial_counts["FACTUAL_RETAINED_CANDIDATE"]
        + initial_counts["FACTUAL_PRESERVED_AFTER_CHANGE_CANDIDATE"]
    )
    factual_damaged = initial_counts["FACTUAL_DAMAGED_CANDIDATE"]
    factual_deleted = initial_counts["FACTUAL_DELETED_CANDIDATE"]
    new_errors = added_counts["NEW_ERROR_CANDIDATE"]
    resolved_predictions = (
        prediction_counts["FACTUAL"] + prediction_counts["NON_FACTUAL"]
    )
    return {
        "branch_id": branch,
        "revised_claim_count": len(predictions),
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "resolved_factual_proportion_micro": _ratio(
            prediction_counts["FACTUAL"],
            resolved_predictions,
        ),
        "resolved_factual_proportion_response_macro": (
            round(
                sum(response_resolved_proportions)
                / len(response_resolved_proportions),
                6,
            )
            if response_resolved_proportions
            else None
        ),
        "strict_error_corrections": corrected,
        "initial_non_factual_count": non_factual_total,
        "strict_error_correction_rate": _ratio(
            corrected,
            non_factual_total,
        ),
        "error_deletions": error_deleted,
        "beneficial_error_dispositions": corrected + error_deleted,
        "beneficial_error_disposition_rate": _ratio(
            corrected + error_deleted,
            non_factual_total,
        ),
        "initial_factual_count": factual_total,
        "factual_preserved": factual_preserved,
        "factual_preservation_rate": _ratio(
            factual_preserved,
            factual_total,
        ),
        "factual_damaged": factual_damaged,
        "factual_damage_rate": _ratio(
            factual_damaged,
            factual_total,
        ),
        "factual_deleted": factual_deleted,
        "factual_deletion_rate": _ratio(
            factual_deleted,
            factual_total,
        ),
        "added_claim_count": len(added),
        "added_factual": added_counts["ADDED_FACTUAL_CANDIDATE"],
        "added_new_errors": new_errors,
        "added_unresolved": added_counts[
            "ADDED_CLAIM_UNRESOLVED_CANDIDATE"
        ],
        "added_new_error_rate": _ratio(new_errors, len(added)),
        "strict_diagnostic_candidate_balance": (
            corrected - factual_damaged - new_errors
        ),
        "formal_net_gain_eligible": False,
    }


def _paired_research_contrast(
    baseline: str,
    treatment: str,
    split: str,
) -> dict[str, Any]:
    baseline_rows = {
        row["initial_claim_id"]: row
        for row in load_jsonl(
            cove_paths(
                PROJECT_ROOT,
                "full",
                baseline,
            ).initial_claim_outcomes(split)
        )
    }
    treatment_rows = {
        row["initial_claim_id"]: row
        for row in load_jsonl(
            cove_paths(
                PROJECT_ROOT,
                "full",
                treatment,
            ).initial_claim_outcomes(split)
        )
    }
    if set(baseline_rows) != set(treatment_rows):
        raise ValueError(
            f"Paired contrast {baseline}->{treatment} has claim mismatch"
        )
    beneficial = {
        "ERROR_CORRECTED_CANDIDATE",
        "ERROR_REMOVED_BY_DELETION_CANDIDATE",
    }
    preserved = {
        "FACTUAL_RETAINED_CANDIDATE",
        "FACTUAL_PRESERVED_AFTER_CHANGE_CANDIDATE",
    }
    non_factual_ids = [
        claim_id
        for claim_id, row in baseline_rows.items()
        if row["human_label"] == "NON_FACTUAL"
    ]
    factual_ids = [
        claim_id
        for claim_id, row in baseline_rows.items()
        if row["human_label"] == "FACTUAL"
    ]
    error_gains = sum(
        baseline_rows[claim_id]["provisional_outcome"] not in beneficial
        and treatment_rows[claim_id]["provisional_outcome"] in beneficial
        for claim_id in non_factual_ids
    )
    error_regressions = sum(
        baseline_rows[claim_id]["provisional_outcome"] in beneficial
        and treatment_rows[claim_id]["provisional_outcome"] not in beneficial
        for claim_id in non_factual_ids
    )
    factual_rescues = sum(
        baseline_rows[claim_id]["provisional_outcome"] not in preserved
        and treatment_rows[claim_id]["provisional_outcome"] in preserved
        for claim_id in factual_ids
    )
    factual_harms = sum(
        baseline_rows[claim_id]["provisional_outcome"] in preserved
        and treatment_rows[claim_id]["provisional_outcome"] not in preserved
        for claim_id in factual_ids
    )
    same = sum(
        baseline_rows[claim_id]["provisional_outcome"]
        == treatment_rows[claim_id]["provisional_outcome"]
        for claim_id in baseline_rows
    )
    return {
        "baseline": baseline,
        "treatment": treatment,
        "paired_initial_claim_count": len(baseline_rows),
        "same_outcome": same,
        "different_outcome": len(baseline_rows) - same,
        "error_adverse_to_beneficial": error_gains,
        "error_beneficial_to_adverse": error_regressions,
        "net_beneficial_error_change": error_gains - error_regressions,
        "factual_harm_to_preserved": factual_rescues,
        "factual_preserved_to_harm": factual_harms,
        "net_factual_preservation_change": factual_rescues - factual_harms,
    }


def _branch_d_target_strata(split: str) -> dict[str, Any]:
    compact_path = bounded_targeted_feedback_path(split)
    if not compact_path.exists():
        return {"available": False}
    target_ids = {
        target["target_claim_id"]
        for row in load_jsonl(compact_path)
        for target in row["compact_feedback"]
    }
    a_paths = cove_paths(PROJECT_ROOT, "full", "a")
    initial_transitions = load_jsonl(
        a_paths.initial_transition_candidates(split)
    )
    target_initial_ids = {
        row["initial_claim_id"]
        for row in initial_transitions
        if target_ids.intersection(row.get("revised_claim_ids") or [])
    }
    branch_rows = {
        branch: {
            row["initial_claim_id"]: row
            for row in load_jsonl(
                cove_paths(
                    PROJECT_ROOT,
                    "full",
                    branch,
                ).initial_claim_outcomes(split)
            )
        }
        for branch in ("a", "c", "d2")
    }
    all_ids = set(branch_rows["a"])

    def cohort_counts(claim_ids: set[str]) -> dict[str, Any]:
        return {
            branch: {
                "count": len(claim_ids),
                "human_label_counts": dict(
                    sorted(
                        Counter(
                            rows[claim_id]["human_label"]
                            for claim_id in claim_ids
                        ).items()
                    )
                ),
                "outcome_counts": dict(
                    sorted(
                        Counter(
                            rows[claim_id]["provisional_outcome"]
                            for claim_id in claim_ids
                        ).items()
                    )
                ),
            }
            for branch, rows in branch_rows.items()
        }

    return {
        "available": True,
        "visible_branch_a_revised_targets": len(target_ids),
        "target_associated_initial_claims": len(target_initial_ids),
        "target_associated": cohort_counts(target_initial_ids),
        "non_target_associated": cohort_counts(all_ids - target_initial_ids),
        "interpretation": (
            "Target association is a semantic join through Branch A B6b; it "
            "is diagnostic and not a human-gold causal annotation."
        ),
    }


def _research_analysis_markdown(analysis: Mapping[str, Any]) -> str:
    d2 = analysis["branch_metrics"]["d2"]
    c = analysis["branch_metrics"]["c"]
    d2_minus_c = analysis["paired_contrasts"]["d2_minus_c"]
    target_strata = analysis["target_strata"]
    quality = analysis["quality_checks"]
    lines = [
        f"# Research Branch D evaluation — {analysis['split']}",
        "",
        (
            "Branch D is implemented in the frozen `d2` artifact namespace. "
            "This namespace is retained so that the code matches the final "
            "methodology artifacts exactly."
        ),
        "",
        "## Execution and evaluation checks",
        "",
        (
            f"- D-v2 produced {analysis['execution_gate']['usable_revisions']}"
            " usable revisions with "
            f"{analysis['execution_gate']['fallbacks']} fallbacks; the "
            "intervention-execution gate passed."
        ),
        (
            f"- B6a extracted {d2['revised_claim_count']} claims. B6c "
            f"successfully evaluated {quality['b6c_successful_claims']}/"
            f"{quality['b6c_selected_claims']} claims."
        ),
        (
            f"- B6b used deterministic structural recovery for "
            f"{quality['b6b_recovered_responses']} response and "
            f"{quality['b6b_repaired_relations']} relations. No model was "
            "recalled; this response remains audit-required."
        ),
        (
            f"- B6d flagged {quality['b6d_deterministic_flagged_claims']} "
            "label/rationale policy inconsistencies without changing raw "
            "labels."
        ),
        (
            "- The frozen version is identified by config, prompt, compact-"
            "feedback, generation-output, and run fingerprints in the JSON "
            "companion report."
        ),
        "",
        "## Same-protocol branch outcomes",
        "",
        "| Branch | Revised claims | Resolved factual (micro) | "
        "Resolved factual (response macro) | Strict corrections | "
        "Error deletions | Factual preserved | Factual damaged | "
        "Factual deleted | Added errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in ("a", "b", "c", "d2"):
        metric = analysis["branch_metrics"][branch]
        lines.append(
            "| "
            + " | ".join(
                [
                    "D" if branch == "d2" else branch.upper(),
                    str(metric["revised_claim_count"]),
                    f"{100 * metric['resolved_factual_proportion_micro']:.1f}%",
                    (
                        f"{100 * metric['resolved_factual_proportion_response_macro']:.1f}%"
                    ),
                    str(metric["strict_error_corrections"]),
                    str(metric["error_deletions"]),
                    str(metric["factual_preserved"]),
                    str(metric["factual_damaged"]),
                    str(metric["factual_deleted"]),
                    str(metric["added_new_errors"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
        ]
    )
    for key in ("d2_minus_a", "d2_minus_c", "d2_minus_b"):
        item = analysis["paired_contrasts"][key]
        lines.append(
            f"- {key}: error net **{item['net_beneficial_error_change']:+d}**, "
            "factual-preservation net "
            f"**{item['net_factual_preservation_change']:+d}**."
        )
    lines.extend(
        [
            "",
            "## Primary D minus C interpretation",
            "",
            (
                f"- Error correction changed only slightly: D corrected "
                f"{d2['strict_error_corrections']} initial errors versus "
                f"{c['strict_error_corrections']} for C; beneficial "
                f"correction-or-deletion outcomes were "
                f"{d2['beneficial_error_dispositions']} versus "
                f"{c['beneficial_error_dispositions']}."
            ),
            (
                f"- Preservation changed more substantially: D preserved "
                f"{d2['factual_preserved']}/"
                f"{d2['initial_factual_count']} initial factual claims versus "
                f"{c['factual_preserved']}/"
                f"{c['initial_factual_count']} for C, and damaged "
                f"{d2['factual_damaged']} versus {c['factual_damaged']}."
            ),
            (
                f"- The claim-paired transition view gives a net "
                f"{d2_minus_c['net_beneficial_error_change']:+d} beneficial "
                "error change and a net "
                f"{d2_minus_c['net_factual_preservation_change']:+d} factual-"
                "preservation change."
            ),
            (
                f"- D introduced {d2['added_new_errors']} candidate new "
                f"errors among {d2['added_claim_count']} added claims; C "
                f"introduced {c['added_new_errors']} among "
                f"{c['added_claim_count']}."
            ),
            (
                "- The revised-claim factual proportions are descriptive, "
                "not a direct paired effect, because each branch generated a "
                "different claim set."
            ),
            "",
        ]
    )
    if target_strata["available"]:
        d_target = target_strata["target_associated"]["d2"][
            "outcome_counts"
        ]
        c_target = target_strata["target_associated"]["c"][
            "outcome_counts"
        ]
        d_non_target = target_strata["non_target_associated"]["d2"][
            "outcome_counts"
        ]
        c_non_target = target_strata["non_target_associated"]["c"][
            "outcome_counts"
        ]
        target_labels = target_strata["target_associated"]["d2"][
            "human_label_counts"
        ]
        lines.extend(
            [
                "## Target-associated diagnostic",
                "",
                (
                    f"The {target_strata['visible_branch_a_revised_targets']} "
                    "visible feedback targets map through the Branch A B6b "
                    "alignment to "
                    f"{target_strata['target_associated_initial_claims']} "
                    "initial claims. Among those claims, D records "
                    f"{d_target.get('FACTUAL_DAMAGED_CANDIDATE', 0)} factual-"
                    "damage candidates versus "
                    f"{c_target.get('FACTUAL_DAMAGED_CANDIDATE', 0)} for C, "
                    "and "
                    f"{d_target.get('ERROR_CORRECTED_CANDIDATE', 0)} strict "
                    "correction candidates versus "
                    f"{c_target.get('ERROR_CORRECTED_CANDIDATE', 0)} for C."
                ),
                "",
                (
                    "The mapped target-associated anchors contain "
                    f"{target_labels.get('FACTUAL', 0)} human-labelled "
                    "FACTUAL, "
                    f"{target_labels.get('NON_FACTUAL', 0)} NON_FACTUAL, and "
                    f"{target_labels.get('UNKNOWN', 0)} UNKNOWN claims. "
                    "Verifier selection is therefore not an oracle error set; "
                    "false-positive feedback remains a central source of "
                    "intervention risk."
                ),
                "",
                (
                    "For target-associated factual anchors, D preserves "
                    f"{_preserved_outcome_count(d_target)}, damages "
                    f"{d_target.get('FACTUAL_DAMAGED_CANDIDATE', 0)}, and "
                    f"deletes {d_target.get('FACTUAL_DELETED_CANDIDATE', 0)}; "
                    "C preserves "
                    f"{_preserved_outcome_count(c_target)}, damages "
                    f"{c_target.get('FACTUAL_DAMAGED_CANDIDATE', 0)}, and "
                    f"deletes {c_target.get('FACTUAL_DELETED_CANDIDATE', 0)}."
                ),
                "",
                (
                    "For non-target-associated factual anchors, D preserves "
                    f"{_preserved_outcome_count(d_non_target)} versus "
                    f"{_preserved_outcome_count(c_non_target)} for C and "
                    "damages "
                    f"{d_non_target.get('FACTUAL_DAMAGED_CANDIDATE', 0)} "
                    "versus "
                    f"{c_non_target.get('FACTUAL_DAMAGED_CANDIDATE', 0)}. "
                    "This suggests that the bounded editing contract, not "
                    "only direct target correction, contributes to the "
                    "preservation signal."
                ),
                "",
                (
                    "This target association is a diagnostic semantic join, "
                    "not a human-gold causal annotation."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            (
                "All transition counts are same-protocol silver candidates. "
                "B6b alignment is LLM-assisted, and B6c uses the same Qwen "
                "model family as generation. The earlier independent "
                "factuality protocol did not pass its frozen development "
                "calibration gate. These results therefore support a "
                "development finding about relative preservation behaviour, "
                "but they are not eligible for a formal human-gold net-"
                "factual-gain claim. This is also a 20-response development "
                "comparison without a response-cluster confidence interval; "
                "the pattern must be tested once on frozen held-out data."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_branch_d_research_analysis(split: str) -> dict[str, Any]:
    generation_report = load_json(
        branch_report_path("d2", split)
    )
    d2_paths = cove_paths(PROJECT_ROOT, "full", "d2")
    extraction_report = load_json(
        d2_paths.revised_claim_extraction_summary_json(split)
    )
    alignment_report = load_json(
        d2_paths.revised_claim_alignment_summary_json(split)
    )
    factuality_report = load_json(
        d2_paths.revised_claim_factuality_summary_json(split)
    )
    audit_report = load_json(
        d2_paths.factuality_audit_summary_json(split)
    )
    recovery = alignment_report.get(
        "conservative_structure_recovery",
        {},
    )
    analysis = {
        "schema_version": "fcb_cove_branch_d_research_analysis_v1",
        "status": "complete",
        "split": split,
        "research_condition_label": "Branch D",
        "implementation_version": "D-v2",
        "artifact_namespace": "d2",
        "frozen_version_fingerprints": {
            "branch_config_sha256": sha256_file(branch_config_path()),
            "branch_d2_config_sha256": sha256_file(
                branch_d2_config_path()
            ),
            "revision_prompt_sha256": sha256_file(
                PROJECT_ROOT
                / "prompts"
                / "cove_selective_verifier_revision_v2.txt"
            ),
            "compact_feedback_sha256": sha256_file(
                bounded_targeted_feedback_path(split)
            ),
            "generation_output_sha256": sha256_file(
                d2_paths.revision_results(split)
            ),
            "generation_run_fingerprint": generation_report.get(
                "run_fingerprint"
            ),
        },
        "execution_gate": {
            "passed": generation_report.get(
                "intervention_execution_valid"
            )
            is True,
            "usable_revisions": generation_report.get(
                "model_output_usable_as_revision_count"
            ),
            "fallbacks": generation_report.get(
                "base_response_fallback_count"
            ),
        },
        "quality_checks": {
            "b6a_successful_responses": extraction_report.get(
                "successful_responses"
            ),
            "b6a_revised_claims": extraction_report.get(
                "total_revised_claims"
            ),
            "b6b_successful_responses": alignment_report.get(
                "successful_responses"
            ),
            "b6b_recovered_responses": recovery.get(
                "recovered_responses",
                0,
            ),
            "b6b_repaired_relations": sum(
                recovery.get("repair_counts", {}).values()
            ),
            "b6c_selected_claims": factuality_report.get(
                "selected_revised_claims"
            ),
            "b6c_successful_claims": factuality_report.get(
                "successful_revised_claims"
            ),
            "b6d_deterministic_flagged_claims": audit_report.get(
                "deterministic_flagged_claims"
            ),
            "raw_factuality_labels_changed": audit_report.get(
                "raw_labels_changed"
            ),
        },
        "branch_metrics": {
            branch: _research_branch_metrics(branch, split)
            for branch in ("a", "b", "c", "d2")
        },
        "paired_contrasts": {
            "d2_minus_a": _paired_research_contrast("a", "d2", split),
            "d2_minus_c": _paired_research_contrast("c", "d2", split),
            "d2_minus_b": _paired_research_contrast("b", "d2", split),
        },
        "target_strata": _branch_d_target_strata(split),
        "evaluation_tier": "same_protocol_silver_llm_assisted",
        "formal_net_gain_eligible": False,
        "generated_at": utc_now(),
    }
    return analysis


def summarize_active_branches(split: str, *, dry_run: bool) -> int:
    """Summarize the active A/B/C/D comparison."""

    active_branches = ("a", "b", "c", "d2")
    branch_results = {
        branch: _branch_outcome_summary(branch, split)
        for branch in active_branches
    }
    research_analysis = (
        build_branch_d_research_analysis(split)
        if all(
            result["evaluation_complete"]
            for result in branch_results.values()
        )
        else None
    )
    comparison = {
        "schema_version": "fcb_cove_four_branch_comparison_v2_status_v1",
        "status": (
            "complete"
            if all(
                result["evaluation_complete"]
                for result in branch_results.values()
            )
            else "incomplete"
        ),
        "split": split,
        "paired_response_count": len(
            load_jsonl(
                cove_paths(
                    PROJECT_ROOT,
                    "full",
                    "a",
                ).revision_results(split)
            )
        ),
        "active_branch_mapping": {
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d2",
        },
        "branches": branch_results,
        "research_branch_d_analysis": research_analysis,
        "interpretation_boundary": (
            "Active research Branch D maps to the frozen d2 artifact "
            "namespace. Outcome counts remain silver because the same-family "
            "evaluator and semantic alignment are imperfect. D-minus-C is "
            "interpretable only because the Branch D generation report passed "
            "its intervention-execution validity gate."
        ),
        "branch_config_sha256": sha256_file(branch_config_path()),
        "branch_d2_config_sha256": sha256_file(branch_d2_config_path()),
        "generated_at": utc_now(),
    }
    if dry_run:
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return 0
    output_path = (
        branches_root()
        / "reports"
        / f"four_branch_comparison_v2_{split}.json"
    )
    atomic_write_json(output_path, comparison)
    if research_analysis is not None:
        analysis_json_path = (
            branches_root()
            / "reports"
            / f"branch_d_research_evaluation_{split}.json"
        )
        analysis_markdown_path = (
            branches_root()
            / "reports"
            / f"branch_d_research_evaluation_{split}.md"
        )
        atomic_write_json(analysis_json_path, research_analysis)
        atomic_write_text(
            analysis_markdown_path,
            _research_analysis_markdown(research_analysis),
        )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one controlled four-branch CoVe stage."
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--scope", choices=("full",), default="full")
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-config-frozen", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
    )
    args = parser.parse_args(argv)
    if args.resume and args.stage not in MODEL_STAGES:
        parser.error("--resume is valid only for model-backed branch stages")
    if (
        args.split == "heldout"
        and args.stage in {
            "prepare-grounded-evidence",
            *MODEL_STAGES,
            "prepare-targeted-feedback-candidates",
            "prepare-bounded-targeted-feedback",
        }
        and not args.confirm_config_frozen
    ):
        parser.error(
            "Held-out branch intervention stages require "
            "--confirm-config-frozen"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "audit":
        return audit_stage(args.split, dry_run=args.dry_run)
    if args.stage == "prepare-grounded-evidence":
        return prepare_grounded_evidence(args.split, dry_run=args.dry_run)
    if args.stage == "run-grounded-answers":
        return run_grounded_answers(
            args.split,
            resume=args.resume,
            dry_run=args.dry_run,
            ollama_host=args.ollama_host,
        )
    if args.stage == "run-grounded-revision":
        return run_branch_revision(
            "b",
            args.split,
            resume=args.resume,
            dry_run=args.dry_run,
            ollama_host=args.ollama_host,
        )
    if args.stage == "run-extra-revision":
        return run_branch_revision(
            "c",
            args.split,
            resume=args.resume,
            dry_run=args.dry_run,
            ollama_host=args.ollama_host,
        )
    if args.stage == "prepare-targeted-feedback-candidates":
        return prepare_targeted_feedback_candidates(
            args.split,
            dry_run=args.dry_run,
        )
    if args.stage == "prepare-bounded-targeted-feedback":
        return prepare_bounded_targeted_feedback(
            args.split,
            dry_run=args.dry_run,
        )
    if args.stage == "run-targeted-evidence-revision":
        return run_branch_revision(
            "d2",
            args.split,
            resume=args.resume,
            dry_run=args.dry_run,
            ollama_host=args.ollama_host,
        )
    if args.stage == "summarize":
        return summarize_active_branches(args.split, dry_run=args.dry_run)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
