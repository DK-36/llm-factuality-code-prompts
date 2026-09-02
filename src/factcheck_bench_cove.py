"""Study II paths and leakage-safe CoVe input preparation.

This module contains no model client.  It builds one response-level input row
per eligible FactCheck-Bench source response and verifies that Study II uses the same response-level development boundary as Study I.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CONFIG_SCHEMA_VERSION = "fcb_cove_experiment_config_v1"
INPUT_SCHEMA_VERSION = "fcb_cove_response_input_v1"
SUMMARY_SCHEMA_VERSION = "fcb_cove_input_summary_v1"
PRIMARY_LABELS = {"FACTUAL", "NON_FACTUAL"}


@dataclass(frozen=True)
class CovePaths:
    """Canonical Study II paths."""

    scope: str
    branch: str
    root: Path
    config: Path
    manifests_dir: Path
    response_manifest: Path
    input_summary: Path
    output_root: Path
    jsonl_dir: Path
    reports_dir: Path

    def question_results(self, split: str) -> Path:
        return self.jsonl_dir / f"B1_question_planning_{split}.jsonl"

    def question_summary_json(self, split: str) -> Path:
        return self.reports_dir / f"B1_question_planning_{split}_summary.json"

    def question_summary_markdown(self, split: str) -> Path:
        return self.reports_dir / f"B1_question_planning_{split}_report.md"

    def alignment_results(self, split: str) -> Path:
        return self.jsonl_dir / f"B2_question_claim_alignment_{split}.jsonl"

    def alignment_pairs(self, split: str) -> Path:
        return self.jsonl_dir / f"B2_question_claim_pairs_{split}.jsonl"

    def alignment_summary_json(self, split: str) -> Path:
        return self.reports_dir / f"B2_question_claim_alignment_{split}_summary.json"

    def alignment_summary_markdown(self, split: str) -> Path:
        return self.reports_dir / f"B2_question_claim_alignment_{split}_report.md"

    def verification_answer_results(self, split: str) -> Path:
        return self.jsonl_dir / f"B3_independent_verification_answers_{split}.jsonl"

    def verification_answer_summary_json(self, split: str) -> Path:
        return (
            self.reports_dir
            / f"B3_independent_verification_answers_{split}_summary.json"
        )

    def verification_answer_summary_markdown(self, split: str) -> Path:
        return (
            self.reports_dir
            / f"B3_independent_verification_answers_{split}_report.md"
        )

    def answer_claim_evaluation_results(self, split: str) -> Path:
        return self.jsonl_dir / f"B4_answer_claim_evaluation_{split}.jsonl"

    def answer_claim_evaluation_summary_json(self, split: str) -> Path:
        return (
            self.reports_dir
            / f"B4_answer_claim_evaluation_{split}_summary.json"
        )

    def answer_claim_evaluation_summary_markdown(self, split: str) -> Path:
        return (
            self.reports_dir
            / f"B4_answer_claim_evaluation_{split}_report.md"
        )

    def revision_results(self, split: str) -> Path:
        return self.jsonl_dir / f"B5_cove_revision_{split}.jsonl"

    def revision_summary_json(self, split: str) -> Path:
        return self.reports_dir / f"B5_cove_revision_{split}_summary.json"

    def revision_summary_markdown(self, split: str) -> Path:
        return self.reports_dir / f"B5_cove_revision_{split}_report.md"

    def revised_claim_extraction_results(self, split: str) -> Path:
        return self.jsonl_dir / f"B6a_revised_claim_extraction_{split}.jsonl"

    def revised_claim_extraction_summary_json(self, split: str) -> Path:
        return (
            self.reports_dir
            / f"B6a_revised_claim_extraction_{split}_summary.json"
        )

    def revised_claim_extraction_summary_markdown(self, split: str) -> Path:
        return (
            self.reports_dir
            / f"B6a_revised_claim_extraction_{split}_report.md"
        )

    def revised_claim_alignment_results(self, split: str) -> Path:
        return (
            self.jsonl_dir
            / f"B6b_gold_revised_claim_alignment_{split}.jsonl"
        )

    def initial_transition_candidates(self, split: str) -> Path:
        return (
            self.jsonl_dir
            / f"B6b_initial_claim_transition_candidates_{split}.jsonl"
        )

    def added_claim_candidates(self, split: str) -> Path:
        return (
            self.jsonl_dir
            / f"B6b_added_claim_candidates_{split}.jsonl"
        )

    def revised_claim_alignment_summary_json(self, split: str) -> Path:
        return (
            self.reports_dir
            / f"B6b_gold_revised_claim_alignment_{split}_summary.json"
        )

    def revised_claim_alignment_summary_markdown(self, split: str) -> Path:
        return (
            self.reports_dir
            / f"B6b_gold_revised_claim_alignment_{split}_report.md"
        )

    def revised_claim_evidence(self, split: str) -> Path:
        return (
            self.jsonl_dir
            / f"B6c_revised_claim_hybrid_top5_{split}.jsonl"
        )

    def revised_claim_factuality_results(self, split: str) -> Path:
        return (
            self.jsonl_dir
            / f"B6c_revised_claim_factuality_{split}.jsonl"
        )

    def initial_claim_outcomes(self, split: str) -> Path:
        return (
            self.jsonl_dir
            / f"B6c_initial_claim_outcomes_{split}.jsonl"
        )

    def added_claim_outcomes(self, split: str) -> Path:
        return (
            self.jsonl_dir
            / f"B6c_added_claim_outcomes_{split}.jsonl"
        )

    def revised_claim_factuality_summary_json(self, split: str) -> Path:
        return (
            self.reports_dir
            / f"B6c_revised_claim_factuality_{split}_summary.json"
        )

    def revised_claim_factuality_summary_markdown(self, split: str) -> Path:
        return (
            self.reports_dir
            / f"B6c_revised_claim_factuality_{split}_report.md"
        )

    def factuality_audit_manifest(self, split: str) -> Path:
        return (
            self.jsonl_dir
            / f"post_revision_factuality_audit_{split}.jsonl"
        )

    def factuality_audit_summary_json(self, split: str) -> Path:
        return (
            self.reports_dir
            / f"post_revision_factuality_audit_{split}_summary.json"
        )

    def factuality_audit_summary_markdown(self, split: str) -> Path:
        return (
            self.reports_dir
            / f"post_revision_factuality_audit_{split}_report.md"
        )

def cove_paths(
    project_root: Path,
    scope: str = "full",
    branch: str = "a",
) -> CovePaths:
    if scope != "full":
        raise ValueError("Formal Study II currently supports full scope only")
    if branch not in {"a", "b", "c", "d", "d2"}:
        raise ValueError("CoVe branch must be one of: a, b, c, d, d2")
    root = project_root / "data" / "factcheck_bench" / "cove"
    canonical_output_root = (
        project_root / "outputs" / "factcheck_bench_full" / "cove"
    )
    output_root = (
        canonical_output_root
        if branch == "a"
        else canonical_output_root / "branches" / f"branch_{branch}"
    )
    return CovePaths(
        scope=scope,
        branch=branch,
        root=root,
        config=root / "config" / "cove_experiment_config.json",
        manifests_dir=root / "manifests",
        response_manifest=root / "manifests" / "cove_response_manifest.jsonl",
        input_summary=root / "manifests" / "cove_input_summary.json",
        output_root=output_root,
        jsonl_dir=output_root / "jsonl",
        reports_dir=output_root / "reports",
    )


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(f"Line {line_number} of {path} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    return rows


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(value)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )
    atomic_write_text(path, text)


def load_config(paths: CovePaths, override: Path | None = None) -> dict[str, Any]:
    config_path = override if override is not None else paths.config
    config = load_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported CoVe config schema: {config.get('schema_version')!r}"
        )
    if config.get("scope") != "full":
        raise ValueError("Study II config scope must be full")
    return config


def _split_for_index(index: int, config: dict[str, Any]) -> str:
    policy = config["split_policy"]
    if (
        int(policy["development_source_record_start"])
        <= index
        <= int(policy["development_source_record_end"])
    ):
        return "dev"
    if index >= int(policy["heldout_source_record_start"]):
        return "heldout"
    raise ValueError(f"Source record index is outside split policy: {index}")


def _assert_expected(actual: dict[str, int], expected: dict[str, Any]) -> None:
    mismatches = {
        key: {"expected": int(value), "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != int(value)
    }
    if mismatches:
        raise ValueError(
            "Study II cohort counts do not match the frozen design: "
            + json.dumps(mismatches, sort_keys=True)
        )


def prepare_cove_inputs(
    project_root: Path,
    paths: CovePaths,
    config: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build and validate the response-level CoVe input manifest."""

    raw_path = (
        project_root
        / "data"
        / "factcheck_bench"
        / "raw"
        / "factcheck-GPT-benchmark.jsonl"
    )
    gold_path = (
        project_root
        / "data"
        / "factcheck_bench"
        / "processed"
        / "fcb_gold_claims_full.jsonl"
    )
    retrieval_split_path = (
        project_root
        / "data"
        / "factcheck_bench"
        / "retrieval"
        / "manifests"
        / "retrieval_split_manifest.jsonl"
    )
    raw_rows = load_jsonl(raw_path)
    gold_rows = load_jsonl(gold_path)
    retrieval_rows = load_jsonl(retrieval_split_path)

    gold_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_claim_ids: set[str] = set()
    for row in gold_rows:
        claim_id = row.get("claim_id")
        response_id = row.get("response_id")
        index = row.get("source_record_index")
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError("Gold row has invalid claim_id")
        if claim_id in seen_claim_ids:
            raise ValueError(f"Duplicate gold claim_id: {claim_id}")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError(f"Gold row {claim_id} has invalid response_id")
        if not isinstance(index, int) or index <= 0:
            raise ValueError(f"Gold row {claim_id} has invalid source_record_index")
        if response_id != f"fcb_r{index:04d}":
            raise ValueError(f"Gold response/index mismatch for {claim_id}")
        seen_claim_ids.add(claim_id)
        gold_by_response[response_id].append(row)

    response_manifest: list[dict[str, Any]] = []
    excluded_no_claim_response_ids: list[str] = []
    for index, raw in enumerate(raw_rows, start=1):
        response_id = f"fcb_r{index:04d}"
        question = raw.get("prompt")
        initial_response = raw.get("response")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Raw response {response_id} has invalid prompt")
        if not isinstance(initial_response, str) or not initial_response.strip():
            raise ValueError(f"Raw response {response_id} has invalid response")

        anchors = gold_by_response.get(response_id, [])
        if not anchors:
            excluded_no_claim_response_ids.append(response_id)
            continue
        for anchor in anchors:
            if anchor.get("prompt") != question:
                raise ValueError(f"Prompt mismatch between raw/gold for {response_id}")
            if anchor.get("source_response") != initial_response:
                raise ValueError(f"Response mismatch between raw/gold for {response_id}")

        response_manifest.append(
            {
                "schema_version": INPUT_SCHEMA_VERSION,
                "response_id": response_id,
                "source_record_index": index,
                "split": _split_for_index(index, config),
                "original_question": question,
                "initial_response": initial_response,
                "original_question_sha256": sha256_text(question),
                "initial_response_sha256": sha256_text(initial_response),
                "model_input_fields": [
                    "original_question",
                    "initial_response",
                ],
                "gold_fields_included": [],
            }
        )

    matched_gold_ids = {
        row["claim_id"]
        for row in gold_rows
        if row.get("in_primary_matched_cohort") is True
    }
    retrieval_ids = {row.get("claim_id") for row in retrieval_rows}
    if None in retrieval_ids or len(retrieval_ids) != len(retrieval_rows):
        raise ValueError("Retrieval split manifest has invalid/duplicate claim IDs")
    if retrieval_ids != matched_gold_ids:
        raise ValueError(
            "Retrieval split manifest does not equal the canonical matched cohort"
        )
    for row in retrieval_rows:
        response_id = row.get("response_id")
        index_text = str(response_id).removeprefix("fcb_r")
        if not index_text.isdigit():
            raise ValueError(f"Invalid retrieval response_id: {response_id!r}")
        expected_split = _split_for_index(int(index_text), config)
        if row.get("split") != expected_split:
            raise ValueError(
                f"Retrieval/CoVe split mismatch for {row.get('claim_id')}"
            )

    claim_split_counts: Counter[tuple[str, str]] = Counter()
    label_split_counts: Counter[tuple[str, str]] = Counter()
    for row in gold_rows:
        split = _split_for_index(int(row["source_record_index"]), config)
        claim_split_counts[(split, "all")] += 1
        if row.get("is_binary_evaluable") is True:
            claim_split_counts[(split, "binary")] += 1
        if row.get("in_primary_matched_cohort") is True:
            claim_split_counts[(split, "matched")] += 1
        label_split_counts[(split, str(row.get("human_label")))] += 1

    actual = {
        "raw_responses": len(raw_rows),
        "eligible_responses": len(response_manifest),
        "excluded_no_claim_responses": len(excluded_no_claim_response_ids),
        "dev_responses": sum(row["split"] == "dev" for row in response_manifest),
        "heldout_responses": sum(
            row["split"] == "heldout" for row in response_manifest
        ),
        "all_claims": len(gold_rows),
        "binary_claims": sum(
            row.get("human_label") in PRIMARY_LABELS for row in gold_rows
        ),
        "matched_claims": len(matched_gold_ids),
        "dev_all_claims": claim_split_counts[("dev", "all")],
        "dev_binary_claims": claim_split_counts[("dev", "binary")],
        "dev_matched_claims": claim_split_counts[("dev", "matched")],
        "heldout_all_claims": claim_split_counts[("heldout", "all")],
        "heldout_binary_claims": claim_split_counts[("heldout", "binary")],
        "heldout_matched_claims": claim_split_counts[("heldout", "matched")],
    }
    _assert_expected(actual, config["split_policy"]["expected"])

    response_ids = [row["response_id"] for row in response_manifest]
    if len(set(response_ids)) != len(response_ids):
        raise ValueError("Duplicate response_id in CoVe response manifest")

    manifest_fingerprint = canonical_json_hash(response_manifest)
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "validated" if dry_run else "complete",
        "experiment": config["experiment"],
        "scope": paths.scope,
        "split_unit": "response_id",
        "actual_counts": actual,
        "label_counts": {
            split: {
                label: label_split_counts[(split, label)]
                for label in ("FACTUAL", "NON_FACTUAL", "UNKNOWN")
            }
            for split in ("dev", "heldout")
        },
        "excluded_no_claim_response_ids": excluded_no_claim_response_ids,
        "retrieval_split_consistency": "passed",
        "model_input_fields": config["leakage_policy"]["model_input_fields"],
        "evaluation_only_fields": config["leakage_policy"][
            "evaluation_only_fields"
        ],
        "source_files": {
            "raw_benchmark": str(raw_path.relative_to(project_root)),
            "raw_benchmark_sha256": sha256_file(raw_path),
            "gold_claims": str(gold_path.relative_to(project_root)),
            "gold_claims_sha256": sha256_file(gold_path),
            "retrieval_split_manifest": str(
                retrieval_split_path.relative_to(project_root)
            ),
            "retrieval_split_manifest_sha256": sha256_file(retrieval_split_path),
            "config": str(paths.config.relative_to(project_root)),
            "config_sha256": sha256_file(paths.config),
        },
        "response_manifest": str(paths.response_manifest.relative_to(project_root)),
        "response_manifest_fingerprint": manifest_fingerprint,
    }
    if not dry_run:
        atomic_write_jsonl(paths.response_manifest, response_manifest)
        atomic_write_json(paths.input_summary, summary)
    return summary


def validate_response_manifest(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    expected_fields = {
        "schema_version",
        "response_id",
        "source_record_index",
        "split",
        "original_question",
        "initial_response",
        "original_question_sha256",
        "initial_response_sha256",
        "model_input_fields",
        "gold_fields_included",
    }
    seen: set[str] = set()
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError(
                f"Unexpected response manifest fields for {row.get('response_id')}"
            )
        response_id = row["response_id"]
        if response_id in seen:
            raise ValueError(f"Duplicate response_id: {response_id}")
        seen.add(response_id)
        if row["schema_version"] != INPUT_SCHEMA_VERSION:
            raise ValueError(f"Unexpected input schema for {response_id}")
        if row["split"] not in {"dev", "heldout"}:
            raise ValueError(f"Invalid split for {response_id}")
        if row["split"] != _split_for_index(row["source_record_index"], config):
            raise ValueError(f"Split/index mismatch for {response_id}")
        if row["model_input_fields"] != [
            "original_question",
            "initial_response",
        ]:
            raise ValueError(f"Unexpected model inputs for {response_id}")
        if row["gold_fields_included"] != []:
            raise ValueError(f"Gold field leakage in manifest for {response_id}")
        if row["original_question_sha256"] != sha256_text(
            row["original_question"]
        ):
            raise ValueError(f"Question hash mismatch for {response_id}")
        if row["initial_response_sha256"] != sha256_text(
            row["initial_response"]
        ):
            raise ValueError(f"Response hash mismatch for {response_id}")

    expected = config["split_policy"]["expected"]
    if len(rows) != int(expected["eligible_responses"]):
        raise ValueError("CoVe response manifest row count is not canonical")
    if sum(row["split"] == "dev" for row in rows) != int(
        expected["dev_responses"]
    ):
        raise ValueError("CoVe development response count is not canonical")
    if sum(row["split"] == "heldout" for row in rows) != int(
        expected["heldout_responses"]
    ):
        raise ValueError("CoVe held-out response count is not canonical")
