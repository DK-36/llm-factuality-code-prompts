#!/usr/bin/env python3
"""Build Experiment A's benchmark-grounded closed source-document corpus.

Every subcommand is data-only: this script never imports Ollama and never runs
the verifier.  Network access occurs only for the explicitly named
``fetch-corpus`` subcommand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factcheck_bench_pipeline import retrieval_paths  # noqa: E402
from factcheck_bench_retrieval import (  # noqa: E402
    build_evidence_and_url_manifests,
    build_passages,
    build_qrels,
    config_for_paths,
    load_json,
    prepare_retrieval_splits,
    summarize_corpus,
)


STAGES = (
    "prepare-retrieval-splits",
    "build-evidence-manifest",
    "fetch-corpus",
    "reprocess-frozen-corpus",
    "build-passages",
    "build-qrels",
    "summarize-corpus",
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
        help="Experiment A is defined only over canonical full scope.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Retrieval config JSON. Defaults to "
            "data/factcheck_bench/retrieval/config/retrieval_corpus_config.json."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and compute the stage without writing artifacts.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the benchmark-grounded closed source-document corpus for "
            "FactCheck-Bench Experiment A."
        )
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    for stage in ("prepare-retrieval-splits", "build-evidence-manifest"):
        child = subparsers.add_parser(stage)
        _add_common(child)

    fetch = subparsers.add_parser(
        "fetch-corpus",
        help="Fetch/freeze source documents. This is the only networked stage.",
    )
    _add_common(fetch)
    fetch.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N canonical URL candidates.",
    )
    fetch.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use the configured small URL limit unless --limit is supplied.",
    )
    fetch_mode = fetch.add_mutually_exclusive_group()
    fetch_mode.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help=(
            "Validate and skip both successful documents and completed fetch "
            "failures (default)."
        ),
    )
    fetch_mode.add_argument(
        "--refetch",
        action="store_true",
        help="Explicitly fetch candidates again instead of resuming prior outcomes.",
    )

    reprocess = subparsers.add_parser(
        "reprocess-frozen-corpus",
        help=(
            "Re-extract already frozen HTML with the current extractor. "
            "This stage is offline and snapshots prior canonical artifacts."
        ),
    )
    _add_common(reprocess)

    passages = subparsers.add_parser("build-passages")
    _add_common(passages)
    passages.add_argument("--chunk-size", type=int, default=None)
    passages.add_argument("--chunk-overlap", type=int, default=None)
    passages.add_argument(
        "--limit", type=int, default=None, help="Process at most N successful docs."
    )
    passages.add_argument("--smoke-test", action="store_true")

    qrels = subparsers.add_parser("build-qrels")
    _add_common(qrels)
    qrels.add_argument(
        "--split",
        choices=("dev", "heldout", "all"),
        default="dev",
        help=(
            "Build development qrels by default. Held-out/all require an "
            "explicit frozen-configuration confirmation."
        ),
    )
    qrels.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help=(
            "Required for heldout/all qrels. Records that chunk/retrieval "
            "configuration selection is complete before unsealing held-out QA."
        ),
    )
    qrels.add_argument(
        "--limit", type=int, default=None, help="Map at most N matched claims."
    )
    qrels.add_argument("--smoke-test", action="store_true")

    summary = subparsers.add_parser("summarize-corpus")
    _add_common(summary)

    args = parser.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if (
        args.stage == "build-qrels"
        and args.split in {"heldout", "all"}
        and not args.confirm_config_frozen
    ):
        parser.error(
            "heldout/all qrels require --confirm-config-frozen after dev selection"
        )
    return args


def _load_config(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    paths = retrieval_paths(PROJECT_ROOT, args.scope)
    if args.config is None:
        config = config_for_paths(paths)
    else:
        config = load_json(args.config)
        if config.get("schema_version") != "fcb_retrieval_config_v1":
            raise ValueError("Unsupported retrieval config schema_version")
    return paths, config


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    paths, config = _load_config(args)
    if args.stage == "prepare-retrieval-splits":
        return prepare_retrieval_splits(
            PROJECT_ROOT, paths, config, dry_run=args.dry_run
        )
    if args.stage == "build-evidence-manifest":
        return build_evidence_and_url_manifests(
            PROJECT_ROOT, paths, config, dry_run=args.dry_run
        )
    if args.stage == "fetch-corpus":
        from factcheck_bench_corpus_fetch import fetch_corpus

        limit = args.limit
        if args.smoke_test and limit is None:
            limit = int(config["fetch"]["smoke_test_url_limit"])
        return fetch_corpus(
            PROJECT_ROOT,
            paths,
            config,
            limit=limit,
            resume=not args.refetch,
            dry_run=args.dry_run,
        )
    if args.stage == "reprocess-frozen-corpus":
        from factcheck_bench_corpus_fetch import reprocess_frozen_documents

        return reprocess_frozen_documents(
            PROJECT_ROOT,
            paths,
            config,
            dry_run=args.dry_run,
        )
    if args.stage == "build-passages":
        limit = args.limit
        if args.smoke_test and limit is None:
            limit = int(config["fetch"]["smoke_test_url_limit"])
        return build_passages(
            PROJECT_ROOT,
            paths,
            config,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            limit=limit,
            artifact_namespace=("smoke" if limit is not None else "canonical"),
            dry_run=args.dry_run,
        )
    if args.stage == "build-qrels":
        limit = args.limit
        if args.smoke_test and limit is None:
            limit = int(config["fetch"]["smoke_test_url_limit"])
        smoke_passages = paths.root / "smoke" / "passages.jsonl"
        return build_qrels(
            PROJECT_ROOT,
            paths,
            config,
            split_scope=args.split,
            confirm_config_frozen=args.confirm_config_frozen,
            limit=limit,
            artifact_namespace=("smoke" if limit is not None else "canonical"),
            input_passages=(
                smoke_passages
                if args.smoke_test and smoke_passages.exists()
                else paths.passages
            ),
            dry_run=args.dry_run,
        )
    if args.stage == "summarize-corpus":
        return summarize_corpus(PROJECT_ROOT, paths, config, dry_run=args.dry_run)
    raise ValueError(f"Unsupported stage: {args.stage}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_stage(args)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "scope": args.scope,
                "dry_run": args.dry_run,
                "status": report.get("status", "validated"),
                "report": report,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
