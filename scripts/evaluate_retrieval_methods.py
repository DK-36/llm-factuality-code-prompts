#!/usr/bin/env python3
"""Run two-level retrieval evaluation on development or held-out claims.

Development supports configuration selection. Held-out requires an explicit
configuration-frozen acknowledgement and never performs reselection.
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
from factcheck_bench_retrieval_eval import (  # noqa: E402
    compare_retrievers,
    evaluation_paths,
    load_evaluation_config,
    prepare_two_level_qrels,
    run_retriever,
)


STAGES = (
    "prepare-two-level-qrels",
    "run-bm25-retrieval",
    "run-dense-retrieval",
    "run-hybrid-retrieval",
    "compare-retrieval",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BM25 and Dense retrieval on the frozen 121-claim dev "
            "split at source-document and strict-passage levels."
        )
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument(
        "--scope",
        choices=("full",),
        default="full",
        help="The canonical Experiment A corpus is full-scope only.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Evaluation config JSON. Defaults to data/factcheck_bench/"
            "retrieval/config/retrieval_evaluation_config.json."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("dev", "heldout"),
        default="dev",
        help="Evaluate dev by default; heldout requires frozen-config confirmation.",
    )
    parser.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help="Required before creating or evaluating held-out artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report planned counts without writing artifacts.",
    )
    args = parser.parse_args(argv)
    if args.split == "heldout" and not args.confirm_config_frozen:
        parser.error("heldout evaluation requires --confirm-config-frozen")
    return args


def run_stage(args: argparse.Namespace) -> dict[str, Any]:
    corpus_paths = retrieval_paths(PROJECT_ROOT, args.scope)
    eval_paths = evaluation_paths(corpus_paths)
    if args.config is not None:
        eval_paths = type(eval_paths)(
            **{**eval_paths.__dict__, "config": args.config.resolve()}
        )
    config = load_evaluation_config(eval_paths)
    if args.stage == "prepare-two-level-qrels":
        return prepare_two_level_qrels(
            PROJECT_ROOT,
            corpus_paths,
            eval_paths,
            config,
            split=args.split,
            confirm_config_frozen=args.confirm_config_frozen,
            dry_run=args.dry_run,
        )
    if args.stage == "run-bm25-retrieval":
        return run_retriever(
            PROJECT_ROOT,
            corpus_paths,
            eval_paths,
            config,
            retriever="bm25",
            split=args.split,
            dry_run=args.dry_run,
        )
    if args.stage == "run-dense-retrieval":
        return run_retriever(
            PROJECT_ROOT,
            corpus_paths,
            eval_paths,
            config,
            retriever="dense",
            split=args.split,
            dry_run=args.dry_run,
        )
    if args.stage == "run-hybrid-retrieval":
        return run_retriever(
            PROJECT_ROOT,
            corpus_paths,
            eval_paths,
            config,
            retriever="hybrid",
            split=args.split,
            dry_run=args.dry_run,
        )
    if args.stage == "compare-retrieval":
        return compare_retrievers(
            PROJECT_ROOT,
            corpus_paths,
            eval_paths,
            config,
            split=args.split,
            dry_run=args.dry_run,
        )
    raise ValueError(f"Unsupported stage: {args.stage}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_stage(args)
    if "per_query" in report:
        concise = {
            key: report[key]
            for key in (
                "schema_version",
                "status",
                "retriever",
                "split",
                "query_count",
                "source_document",
                "strict_passage",
                "model_digest",
                "run_artifact",
            )
            if key in report
        }
    else:
        concise = report
    print(json.dumps(concise, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
