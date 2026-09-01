#!/usr/bin/env python3
"""Run pinned VeriScore with the frozen provider adapter and import X2.

The official package remains in ``work/`` and is not vendored into this
project. This launcher performs strict preflight checks, streams the official
progress output to the terminal, records non-secret run metadata only after a
successful run, and invokes the canonical X2 importer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "factcheck_bench"
    / "cove"
    / "config"
    / "cove_external_veriscore_config.json"
)
EXTERNAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "factcheck_bench_full"
    / "cove"
    / "external_evaluation"
    / "veriscore"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def command_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_provider_adapter(checkout: Path, protocol: dict[str, Any]) -> tuple[Path, str]:
    adapter = protocol["provider_adapter"]
    patch_path = (PROJECT_ROOT / adapter["path"]).resolve()
    if not patch_path.is_file():
        raise FileNotFoundError(f"Missing frozen provider adapter: {patch_path}")
    actual_hash = sha256_file(patch_path)
    if actual_hash != adapter["sha256"]:
        raise ValueError(
            f"Provider adapter hash is {actual_hash}; expected {adapter['sha256']}"
        )

    changed = command_output(["git", "diff", "--name-only"], cwd=checkout)
    expected_changed = {
        "veriscore/get_response.py",
        "veriscore/utils.py",
    }
    changed_names = set(changed.splitlines()) if changed else set()
    if not changed_names:
        subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=checkout,
            check=True,
        )
    elif changed_names != expected_changed:
        raise ValueError(
            "Official checkout contains unexpected tracked changes: "
            f"{sorted(changed_names)}"
        )

    reverse_check = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch_path)],
        cwd=checkout,
        capture_output=True,
        text=True,
    )
    if reverse_check.returncode != 0:
        raise ValueError("Frozen DeepSeek provider adapter is not cleanly applied")
    actual_diff = command_output(
        ["git", "diff", "--", *sorted(expected_changed)], cwd=checkout
    )
    if actual_diff.strip() != patch_path.read_text(encoding="utf-8").strip():
        raise ValueError("Official checkout provider diff does not equal frozen adapter")
    return patch_path, actual_hash


def credential_environment(protocol: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    environment = dict(os.environ)
    if not environment.get("SERPER_KEY_PRIVATE") and environment.get("SERPER_API_KEY"):
        environment["SERPER_KEY_PRIVATE"] = environment["SERPER_API_KEY"]
    environment["DEEPSEEK_BASE_URL"] = protocol["provider_base_url"]
    missing = [
        name
        for name in ("DEEPSEEK_API_KEY", "SERPER_KEY_PRIVATE")
        if not environment.get(name)
    ]
    return environment, missing


def preflight(
    split: str,
    checkout: Path,
    vendor_python: Path,
) -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    protocol = config["official_protocol"]
    official_input = (
        EXTERNAL_ROOT / "input" / f"factcheck_bench_cove_{split}.jsonl"
    )
    required_paths = [
        official_input,
        checkout / "setup.py",
        checkout / "prompt",
        checkout / "data" / "demos" / "few_shot_examples.jsonl",
        vendor_python,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "VeriScore preflight is missing required paths:\n- "
            + "\n- ".join(missing_paths)
        )

    # VeriScore 2.0.2 hard-codes its Serper cache to data/cache but does not
    # create the parent directory before the tenth query triggers a save.
    # Creating it here prevents a deterministic mid-retrieval FileNotFoundError.
    (checkout / "data" / "cache").mkdir(parents=True, exist_ok=True)

    commit = command_output(["git", "rev-parse", "HEAD"], cwd=checkout)
    expected_commit = protocol["official_repository_commit"]
    if commit != expected_commit:
        raise ValueError(
            f"Official checkout commit is {commit}; frozen commit is {expected_commit}"
        )

    adapter_path, adapter_hash = ensure_provider_adapter(checkout, protocol)

    package_version = command_output([
        str(vendor_python),
        "-c",
        "import importlib.metadata; print(importlib.metadata.version('VeriScore'))",
    ])
    if package_version != protocol["expected_package_version"]:
        raise ValueError(
            f"Installed VeriScore is {package_version}; expected "
            f"{protocol['expected_package_version']}"
        )

    command_output([
        str(vendor_python),
        "-c",
        "import spacy; spacy.load('en_core_web_sm'); print('ok')",
    ])
    environment, missing_credentials = credential_environment(protocol)
    return {
        "config": config,
        "protocol": protocol,
        "official_input": official_input,
        "commit": commit,
        "package_version": package_version,
        "adapter_path": adapter_path,
        "adapter_hash": adapter_hash,
        "environment": environment,
        "missing_credentials": missing_credentials,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned official VeriScore evaluation, write run metadata, "
            "and import X2. Secrets are read only from environment variables."
        )
    )
    parser.add_argument("--split", choices=("heldout",), default="heldout")
    parser.add_argument(
        "--checkout",
        type=Path,
        default=PROJECT_ROOT / "work" / "veriscore_official",
    )
    parser.add_argument(
        "--vendor-python",
        type=Path,
        default=PROJECT_ROOT / "work" / "veriscore_env" / "bin" / "python",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate code, environment, inputs, and credentials without API calls.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one synthetic end-to-end DeepSeek/Serper example in ignored "
            "work directories; do not analyse project responses."
        ),
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Run official VeriScore but do not invoke the project X2 importer.",
    )
    args = parser.parse_args()
    if args.preflight_only and args.smoke_test:
        parser.error("--preflight-only and --smoke-test are mutually exclusive")
    if args.smoke_test and args.skip_analyze:
        parser.error("--skip-analyze is not applicable to --smoke-test")
    return args


def official_command(
    vendor_python: Path,
    vendor_data: Path,
    input_name: str,
    output_dir: Path,
    cache_dir: Path,
    protocol: dict[str, Any],
) -> list[str]:
    return [
        str(vendor_python),
        "-m",
        "veriscore.veriscore",
        "--data_dir",
        str(vendor_data),
        "--input_file",
        input_name,
        "--output_dir",
        str(output_dir),
        "--cache_dir",
        str(cache_dir),
        "--model_name_extraction",
        protocol["extraction_model"],
        "--model_name_verification",
        protocol["verification_model"],
        "--label_n",
        str(protocol["label_n"]),
        "--search_res_num",
        str(protocol["search_res_num"]),
    ]


def run_smoke_test(
    checkout: Path,
    vendor_python: Path,
    protocol: dict[str, Any],
    environment: dict[str, str],
) -> int:
    vendor_data = checkout / "data"
    (vendor_data / "cache").mkdir(parents=True, exist_ok=True)
    input_name = "fcb_veriscore_deepseek_smoke.jsonl"
    input_path = vendor_data / input_name
    smoke_row = {
        "question": "Who wrote Pride and Prejudice?",
        "response": "Jane Austen wrote Pride and Prejudice.",
        "model": "fcb_external_evaluator_smoke_only",
        "prompt_source": "synthetic_smoke_test",
    }
    input_path.write_text(
        json.dumps(smoke_row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_dir = PROJECT_ROOT / "work" / "veriscore_smoke_output"
    cache_dir = PROJECT_ROOT / "work" / "veriscore_smoke_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("[smoke] running one synthetic extraction/search/verification row", flush=True)
    subprocess.run(
        official_command(
            vendor_python,
            vendor_data,
            input_name,
            output_dir,
            cache_dir,
            protocol,
        ),
        cwd=checkout,
        env=environment,
        check=True,
    )
    result = (
        output_dir
        / "model_output"
        / "verification_fcb_veriscore_deepseek_smoke_2.jsonl"
    )
    rows = [
        json.loads(line)
        for line in result.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise ValueError(f"Smoke output expected one row, found {len(rows)}")
    claims = rows[0].get("all_claims")
    verifications = rows[0].get("claim_verification_result")
    if not isinstance(claims, list) or not claims:
        raise ValueError("Smoke output has no extracted claim")
    if not isinstance(verifications, list) or not verifications:
        raise ValueError("Smoke output has no verification result")
    labels = {
        str(item.get("verification_result", "")).strip().lower().rstrip(".")
        for item in verifications
        if isinstance(item, dict)
    }
    if not labels or not labels <= {"supported", "unsupported"}:
        raise ValueError(f"Smoke output contains unexpected labels: {sorted(labels)}")
    print(json.dumps({
        "status": "smoke_test_complete",
        "rows": 1,
        "extracted_claims": len(claims),
        "verification_labels": sorted(labels),
        "output": str(result.relative_to(PROJECT_ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    checkout = args.checkout.resolve()
    # Do not resolve the virtual-environment interpreter symlink: resolving it
    # would bypass the venv and execute the base interpreter instead.
    vendor_python = args.vendor_python.absolute()
    checked = preflight(args.split, checkout, vendor_python)
    protocol = checked["protocol"]

    summary = {
        "status": "ready" if not checked["missing_credentials"] else "blocked",
        "split": args.split,
        "official_repository_commit": checked["commit"],
        "package_version": checked["package_version"],
        "extraction_model": protocol["extraction_model"],
        "verification_model": protocol["verification_model"],
        "provider": protocol["provider"],
        "provider_base_url": protocol["provider_base_url"],
        "thinking_mode": protocol["thinking_mode"],
        "provider_adapter_sha256": checked["adapter_hash"],
        "label_n": protocol["label_n"],
        "search_res_num": protocol["search_res_num"],
        "missing_credentials": checked["missing_credentials"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if checked["missing_credentials"]:
        print(
            "Set the missing variables in the current shell and rerun. "
            "The launcher never writes credential values to disk.",
            file=sys.stderr,
        )
        return 2
    if args.preflight_only:
        return 0
    if args.smoke_test:
        return run_smoke_test(
            checkout,
            vendor_python,
            protocol,
            checked["environment"],
        )

    vendor_data = checkout / "data"
    vendor_input = vendor_data / checked["official_input"].name
    shutil.copy2(checked["official_input"], vendor_input)

    vendor_output = EXTERNAL_ROOT / "vendor_output"
    cache_dir = PROJECT_ROOT / "work" / "veriscore_cache"
    vendor_output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_path = (
        vendor_output
        / "model_output"
        / f"verification_factcheck_bench_cove_{args.split}_2.jsonl"
    )
    metadata_path = vendor_output / f"veriscore_run_metadata_{args.split}.json"

    command = official_command(
        vendor_python,
        vendor_data,
        vendor_input.name,
        vendor_output,
        cache_dir,
        protocol,
    )
    started_at = utc_now()
    print("[VeriScore] starting official held-out run", flush=True)
    print("[VeriScore] official tqdm progress will appear below", flush=True)
    subprocess.run(
        command,
        cwd=checkout,
        env=checked["environment"],
        check=True,
    )
    completed_at = utc_now()
    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Official command completed without a non-empty result: {result_path}"
        )

    metadata = {
        "schema_version": "fcb_external_veriscore_run_metadata_v1",
        "status": "complete",
        "split": args.split,
        "official_repository": protocol["official_repository"],
        "official_repository_commit": checked["commit"],
        "package_version": checked["package_version"],
        "python_version": command_output([str(vendor_python), "--version"]),
        "extraction_model": protocol["extraction_model"],
        "verification_model": protocol["verification_model"],
        "model_access_mode": protocol["model_access_mode"],
        "provider": protocol["provider"],
        "provider_base_url": protocol["provider_base_url"],
        "thinking_mode": protocol["thinking_mode"],
        "temperature": protocol["temperature"],
        "provider_adapter_path": protocol["provider_adapter"]["path"],
        "provider_adapter_sha256": checked["adapter_hash"],
        "label_n": protocol["label_n"],
        "search_provider": protocol["search_provider"],
        "search_res_num": protocol["search_res_num"],
        "run_started_at": started_at,
        "run_completed_at": completed_at,
        "notes": (
            "All five frozen conditions were evaluated in one official input "
            "batch using the frozen DeepSeek provider adapter. Thinking mode "
            "was disabled. Credential values were supplied via environment "
            "variables and were not persisted."
        ),
    }
    atomic_json(metadata_path, metadata)
    print(f"[VeriScore] run metadata: {metadata_path}", flush=True)

    if not args.skip_analyze:
        analyze_command = [
            sys.executable,
            str(
                PROJECT_ROOT
                / "scripts"
                / "prepare_and_analyze_veriscore.py"
            ),
            "analyze",
            "--scope",
            "full",
            "--split",
            args.split,
            "--results",
            str(result_path),
            "--run-metadata",
            str(metadata_path),
        ]
        print("[X2] validating and analysing official output", flush=True)
        subprocess.run(analyze_command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
