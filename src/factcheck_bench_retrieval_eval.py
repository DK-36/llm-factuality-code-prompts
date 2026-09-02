"""Two-level retrieval evaluation for FactCheck-Bench Study I.

Ranking uses only canonical claim text and the frozen passage ``text`` field. Gold
URL mappings and strict passage qrels are loaded only after ranking to evaluate
benchmark-associated source-document retrieval and strict evidence-passage
retrieval. The module never calls a verifier or reads human factuality labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from factcheck_bench_pipeline import RetrievalPaths, canonical_json_hash, paths_for_scope
from factcheck_bench_retrieval import (
    RETRIEVAL_SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    load_json,
    load_jsonl,
    project_relative,
    sha256_file,
    utc_now,
)


EVALUATION_SCHEMA_VERSION = "fcb_two_level_retrieval_evaluation_v1"
WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class RetrievalEvaluationPaths:
    root: Path
    config: Path
    queries_dev: Path
    source_qrels_dev: Path
    strict_qrels_dev: Path
    qrels_report: Path
    dense_embeddings: Path
    dense_embeddings_partial: Path
    dense_embeddings_state: Path
    dense_embeddings_report: Path
    bm25_run: Path
    dense_run: Path
    hybrid_run: Path
    bm25_metrics: Path
    dense_metrics: Path
    hybrid_metrics: Path
    comparison_json: Path
    comparison_markdown: Path
    queries_heldout: Path
    source_qrels_heldout: Path
    strict_qrels_heldout: Path
    qrels_heldout_report: Path
    bm25_heldout_run: Path
    dense_heldout_run: Path
    hybrid_heldout_run: Path
    bm25_heldout_metrics: Path
    dense_heldout_metrics: Path
    hybrid_heldout_metrics: Path
    heldout_comparison_json: Path
    heldout_comparison_markdown: Path


def evaluation_paths(paths: RetrievalPaths) -> RetrievalEvaluationPaths:
    root = paths.root / "evaluation"
    reports = root / "reports"
    return RetrievalEvaluationPaths(
        root=root,
        config=paths.root / "config" / "retrieval_evaluation_config.json",
        queries_dev=root / "queries_dev.jsonl",
        source_qrels_dev=root / "qrels_source_dev.jsonl",
        strict_qrels_dev=root / "qrels_strict_passage_dev.jsonl",
        qrels_report=reports / "two_level_qrels_dev_report.json",
        dense_embeddings=root / "indexes" / "nomic_passage_embeddings.npy",
        dense_embeddings_partial=(
            root / "indexes" / "nomic_passage_embeddings.partial.npy"
        ),
        dense_embeddings_state=(
            root / "indexes" / "nomic_passage_embeddings_state.json"
        ),
        dense_embeddings_report=(
            reports / "nomic_passage_embeddings_report.json"
        ),
        bm25_run=root / "runs" / "bm25_dev.jsonl",
        dense_run=root / "runs" / "dense_nomic_dev.jsonl",
        hybrid_run=root / "runs" / "hybrid_rrf_dev.jsonl",
        bm25_metrics=reports / "bm25_two_level_dev_metrics.json",
        dense_metrics=reports / "dense_nomic_two_level_dev_metrics.json",
        hybrid_metrics=reports / "hybrid_rrf_two_level_dev_metrics.json",
        comparison_json=reports / "two_level_dev_comparison.json",
        comparison_markdown=reports / "two_level_dev_comparison.md",
        queries_heldout=root / "queries_heldout.jsonl",
        source_qrels_heldout=root / "qrels_source_heldout.jsonl",
        strict_qrels_heldout=root / "qrels_strict_passage_heldout.jsonl",
        qrels_heldout_report=reports / "two_level_qrels_heldout_report.json",
        bm25_heldout_run=root / "runs" / "bm25_heldout.jsonl",
        dense_heldout_run=root / "runs" / "dense_nomic_heldout.jsonl",
        hybrid_heldout_run=root / "runs" / "hybrid_rrf_heldout.jsonl",
        bm25_heldout_metrics=reports / "bm25_two_level_heldout_metrics.json",
        dense_heldout_metrics=reports / "dense_nomic_two_level_heldout_metrics.json",
        hybrid_heldout_metrics=reports / "hybrid_rrf_two_level_heldout_metrics.json",
        heldout_comparison_json=reports / "two_level_heldout_comparison.json",
        heldout_comparison_markdown=reports / "two_level_heldout_comparison.md",
    )


def _split_artifacts(
    eval_paths: RetrievalEvaluationPaths, split: str
) -> dict[str, Path]:
    if split == "dev":
        return {
            "queries": eval_paths.queries_dev,
            "source_qrels": eval_paths.source_qrels_dev,
            "strict_qrels": eval_paths.strict_qrels_dev,
            "qrels_report": eval_paths.qrels_report,
            "bm25_run": eval_paths.bm25_run,
            "dense_run": eval_paths.dense_run,
            "hybrid_run": eval_paths.hybrid_run,
            "bm25_metrics": eval_paths.bm25_metrics,
            "dense_metrics": eval_paths.dense_metrics,
            "hybrid_metrics": eval_paths.hybrid_metrics,
            "comparison_json": eval_paths.comparison_json,
            "comparison_markdown": eval_paths.comparison_markdown,
        }
    if split == "heldout":
        return {
            "queries": eval_paths.queries_heldout,
            "source_qrels": eval_paths.source_qrels_heldout,
            "strict_qrels": eval_paths.strict_qrels_heldout,
            "qrels_report": eval_paths.qrels_heldout_report,
            "bm25_run": eval_paths.bm25_heldout_run,
            "dense_run": eval_paths.dense_heldout_run,
            "hybrid_run": eval_paths.hybrid_heldout_run,
            "bm25_metrics": eval_paths.bm25_heldout_metrics,
            "dense_metrics": eval_paths.dense_heldout_metrics,
            "hybrid_metrics": eval_paths.hybrid_heldout_metrics,
            "comparison_json": eval_paths.heldout_comparison_json,
            "comparison_markdown": eval_paths.heldout_comparison_markdown,
        }
    raise ValueError("split must be 'dev' or 'heldout'")


def load_evaluation_config(paths: RetrievalEvaluationPaths) -> dict[str, Any]:
    config = load_json(paths.config)
    if config.get("schema_version") != "fcb_retrieval_evaluation_config_v1":
        raise ValueError("Unsupported retrieval evaluation config schema_version")
    if config.get("scope") != "dev":
        raise ValueError("Two-level configuration selection is dev-only")
    cutoffs = config.get("metric_cutoffs")
    if (
        not isinstance(cutoffs, list)
        or not cutoffs
        or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in cutoffs)
        or cutoffs != sorted(set(cutoffs))
    ):
        raise ValueError("metric_cutoffs must be sorted unique positive integers")
    maximum_rank = config.get("maximum_rank")
    if (
        isinstance(maximum_rank, bool)
        or not isinstance(maximum_rank, int)
        or maximum_rank < max(cutoffs)
    ):
        raise ValueError("maximum_rank must cover every metric cutoff")
    if config.get("evaluation_policy", {}).get("heldout_is_sealed") is not True:
        raise ValueError("Evaluation config must keep held-out sealed")
    hybrid = config.get("hybrid", {})
    if hybrid.get("method") != "reciprocal_rank_fusion":
        raise ValueError("Frozen hybrid method must be reciprocal_rank_fusion")
    if hybrid.get("components") != ["bm25", "dense_nomic"]:
        raise ValueError("Hybrid components must remain BM25 plus Dense Nomic")
    for field in ("rrf_constant", "fusion_depth"):
        value = hybrid.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"hybrid.{field} must be a positive integer")
    return config


def _index_unique(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Every row requires a non-empty {key}")
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    return result


def _passage_text(row: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    fields = config.get("ranking_passage_fields")
    if fields != ["text"]:
        raise ValueError("Frozen ranking_passage_fields must be ['text']")
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Passage has no rankable text: {row.get('passage_id')}")
    return text.strip()


def _validate_upstream(
    project_root: Path,
    corpus_paths: RetrievalPaths,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    passage_report = load_json(corpus_paths.passage_build_report)
    passages = load_jsonl(corpus_paths.passages)
    documents = load_jsonl(corpus_paths.documents)
    if passage_report.get("status") != "complete":
        raise ValueError("Canonical passage build is not complete")
    if passage_report.get("input_documents_sha256") != sha256_file(
        corpus_paths.documents
    ):
        raise ValueError("Passages are stale relative to documents.jsonl")
    if passage_report.get("passages_file_sha256") != sha256_file(
        corpus_paths.passages
    ):
        raise ValueError("Passages do not match passage_build_report.json")
    if len(passages) != passage_report.get("passage_count"):
        raise ValueError("Passage row count does not match build report")
    passage_ids = [row.get("passage_id") for row in passages]
    if any(not isinstance(value, str) for value in passage_ids) or len(
        passage_ids
    ) != len(set(passage_ids)):
        raise ValueError("Canonical passages require unique passage_id values")
    full_paths = paths_for_scope(project_root, "full")
    gold_claims = load_jsonl(full_paths.gold_claims)
    return passages, documents, gold_claims


def prepare_two_level_qrels(
    project_root: Path,
    corpus_paths: RetrievalPaths,
    eval_paths: RetrievalEvaluationPaths,
    config: Mapping[str, Any],
    *,
    split: str = "dev",
    confirm_config_frozen: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create split queries plus source-document and strict-passage judgments."""
    artifacts = _split_artifacts(eval_paths, split)
    if split == "heldout":
        if not confirm_config_frozen:
            raise ValueError("Held-out preparation requires frozen-config confirmation")
        dev_comparison = load_json(eval_paths.comparison_json)
        selection = dev_comparison.get("selection", {})
        if (
            dev_comparison.get("status") != "complete"
            or dev_comparison.get("split") != "dev"
            or selection.get("configuration_is_frozen") is not True
        ):
            raise ValueError("Held-out remains sealed until dev configuration is frozen")
    passages, documents, gold_claims = _validate_upstream(project_root, corpus_paths)
    split_rows = load_jsonl(corpus_paths.split_manifest)
    split_by_claim = _index_unique(split_rows, "claim_id")
    gold_by_claim = _index_unique(gold_claims, "claim_id")
    selected_claim_ids = sorted(
        row["claim_id"] for row in split_rows if row.get("split") == split
    )
    expected_count = 121 if split == "dev" else 468
    if len(selected_claim_ids) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {split} claims, found {len(selected_claim_ids)}"
        )
    queries = []
    for claim_id in selected_claim_ids:
        gold = gold_by_claim.get(claim_id)
        split_row = split_by_claim[claim_id]
        if gold is None:
            raise ValueError(f"Missing canonical gold claim: {claim_id}")
        text = gold.get("gold_claim")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Gold claim text missing: {claim_id}")
        queries.append(
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "query_id": claim_id,
                "response_id": split_row["response_id"],
                "split": split,
                "text": text.strip(),
                "ranking_field_policy": "gold_claim_text_only",
            }
        )

    document_by_id = _index_unique(documents, "doc_id")
    passage_ids = {str(row["passage_id"]) for row in passages}
    passage_doc_ids = {str(row["doc_id"]) for row in passages}
    mapping_audit_path = (
        corpus_paths.qrels_dev_mapping_audit
        if split == "dev"
        else corpus_paths.qrels_heldout_mapping_audit
    )
    automatic_qrels_path = (
        corpus_paths.qrels_dev_jsonl
        if split == "dev"
        else corpus_paths.qrels_heldout_jsonl
    )
    audit_rows = load_jsonl(mapping_audit_path)
    source_pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in audit_rows:
        query_id = row.get("query_id")
        if query_id not in split_by_claim or split_by_claim[query_id].get("split") != split:
            continue
        if row.get("fetch_status") != "success":
            continue
        doc_id = row.get("doc_id")
        if not isinstance(doc_id, str) or doc_id not in document_by_id:
            continue
        document = document_by_id[doc_id]
        primary_doc_id = document.get("duplicate_of_doc_id") or doc_id
        if primary_doc_id not in passage_doc_ids:
            continue
        evidence_id = row.get("evidence_id")
        if isinstance(evidence_id, str):
            source_pairs[(str(query_id), str(primary_doc_id))].add(evidence_id)
    source_qrels = [
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "query_id": query_id,
            "doc_id": doc_id,
            "relevance": 1,
            "evidence_ids": sorted(evidence_ids),
            "field_usage": "evaluation_only_source_qrel",
            "split": split,
        }
        for (query_id, doc_id), evidence_ids in sorted(source_pairs.items())
    ]

    strict_input = load_jsonl(automatic_qrels_path)
    strict_qrels = []
    for row in strict_input:
        relevance = row.get("relevance")
        if isinstance(relevance, bool) or not isinstance(relevance, int):
            raise ValueError("Strict passage qrel relevance must be an integer")
        if relevance <= 0:
            continue
        query_id = row.get("query_id")
        passage_id = row.get("passage_id")
        if query_id not in selected_claim_ids or passage_id not in passage_ids:
            raise ValueError(f"Strict passage qrel references unknown {split} input")
        strict_qrels.append(
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "query_id": query_id,
                "passage_id": passage_id,
                "relevance": relevance,
                "evidence_ids": row.get("evidence_ids", []),
                "alignment_methods": row.get("alignment_methods", []),
                "field_usage": "evaluation_only_strict_passage_qrel",
                "split": split,
            }
        )

    source_queries = {row["query_id"] for row in source_qrels}
    strict_queries = {row["query_id"] for row in strict_qrels}
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "validated" if dry_run else "complete",
        "generated_at": utc_now(),
        "split": split,
        "heldout_configuration_frozen_confirmation": bool(
            confirm_config_frozen if split == "heldout" else False
        ),
        "query_count": len(queries),
        "source_qrel_pair_count": len(source_qrels),
        "source_qrel_query_count": len(source_queries),
        "source_qrel_query_coverage": len(source_queries) / len(queries),
        "strict_passage_qrel_pair_count": len(strict_qrels),
        "strict_passage_qrel_query_count": len(strict_queries),
        "strict_passage_qrel_query_coverage": len(strict_queries) / len(queries),
        "queries_without_successful_source_document": len(queries)
        - len(source_queries),
        "queries_without_strict_passage_qrel": len(queries) - len(strict_queries),
        "ranking_inputs": {
            "query": "gold_claim_text_only",
            "passage_fields": list(config["ranking_passage_fields"]),
        },
        "input_hashes": {
            "evaluation_config_sha256": canonical_json_hash(config),
            "split_manifest_sha256": sha256_file(corpus_paths.split_manifest),
            "gold_claims_sha256": sha256_file(paths_for_scope(project_root, "full").gold_claims),
            "passages_sha256": sha256_file(corpus_paths.passages),
            "documents_sha256": sha256_file(corpus_paths.documents),
            "mapping_audit_sha256": sha256_file(mapping_audit_path),
            "automatic_qrels_sha256": sha256_file(automatic_qrels_path),
        },
        "artifacts": {
            "queries": project_relative(project_root, artifacts["queries"]),
            "source_qrels": project_relative(project_root, artifacts["source_qrels"]),
            "strict_passage_qrels": project_relative(
                project_root, artifacts["strict_qrels"]
            ),
        },
    }
    if not dry_run:
        atomic_write_jsonl(artifacts["queries"], queries)
        atomic_write_jsonl(artifacts["source_qrels"], source_qrels)
        atomic_write_jsonl(artifacts["strict_qrels"], strict_qrels)
        report["artifact_hashes"] = {
            "queries_sha256": sha256_file(artifacts["queries"]),
            "source_qrels_sha256": sha256_file(artifacts["source_qrels"]),
            "strict_passage_qrels_sha256": sha256_file(
                artifacts["strict_qrels"]
            ),
        }
        atomic_write_json(artifacts["qrels_report"], report)
    return report


def _validate_evaluation_inputs(
    project_root: Path,
    corpus_paths: RetrievalPaths,
    eval_paths: RetrievalEvaluationPaths,
    config: Mapping[str, Any],
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifacts = _split_artifacts(eval_paths, split)
    qrels_report = load_json(artifacts["qrels_report"])
    if qrels_report.get("status") != "complete":
        raise ValueError("Two-level qrels must be complete before retrieval")
    if qrels_report.get("split") != split:
        raise ValueError(f"Two-level qrels split mismatch: expected {split}")
    if split == "heldout" and qrels_report.get(
        "heldout_configuration_frozen_confirmation"
    ) is not True:
        raise ValueError("Held-out qrels were not explicitly unsealed")
    expected = {
        "evaluation_config_sha256": canonical_json_hash(config),
        "passages_sha256": sha256_file(corpus_paths.passages),
        "documents_sha256": sha256_file(corpus_paths.documents),
    }
    for key, value in expected.items():
        if qrels_report.get("input_hashes", {}).get(key) != value:
            raise ValueError(f"Two-level qrels are stale: {key}")
    artifact_paths = {
        "queries_sha256": artifacts["queries"],
        "source_qrels_sha256": artifacts["source_qrels"],
        "strict_passage_qrels_sha256": artifacts["strict_qrels"],
    }
    for key, path in artifact_paths.items():
        if qrels_report.get("artifact_hashes", {}).get(key) != sha256_file(path):
            raise ValueError(f"Two-level artifact hash mismatch: {path}")
    passages = load_jsonl(corpus_paths.passages)
    queries = load_jsonl(artifacts["queries"])
    expected_count = 121 if split == "dev" else 468
    if len(queries) != expected_count:
        raise ValueError(
            f"Retrieval evaluation requires all {expected_count} {split} queries"
        )
    return passages, queries


def _tokenize_bm25(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def _rank_bm25(
    passages: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    settings = config["bm25"]
    k1 = float(settings["k1"])
    b = float(settings["b"])
    passage_tokens = [_tokenize_bm25(_passage_text(row, config)) for row in passages]
    term_frequencies = [Counter(tokens) for tokens in passage_tokens]
    lengths = [len(tokens) for tokens in passage_tokens]
    average_length = sum(lengths) / len(lengths)
    document_frequency: Counter[str] = Counter()
    for frequencies in term_frequencies:
        document_frequency.update(frequencies.keys())
    total = len(passages)
    maximum_rank = int(config["maximum_rank"])
    run_rows: list[dict[str, Any]] = []
    for position, query in enumerate(queries, start=1):
        query_terms = Counter(_tokenize_bm25(str(query["text"])))
        scores = [0.0] * total
        for term, query_frequency in query_terms.items():
            df = document_frequency.get(term, 0)
            if not df:
                continue
            inverse_document_frequency = math.log(
                1.0 + (total - df + 0.5) / (df + 0.5)
            )
            for index, frequencies in enumerate(term_frequencies):
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + k1 * (
                    1.0 - b + b * lengths[index] / average_length
                )
                scores[index] += (
                    query_frequency
                    * inverse_document_frequency
                    * frequency
                    * (k1 + 1.0)
                    / denominator
                )
        ranked = sorted(
            range(total),
            key=lambda index: (-scores[index], str(passages[index]["passage_id"])),
        )[:maximum_rank]
        for rank, index in enumerate(ranked, start=1):
            passage = passages[index]
            run_rows.append(
                {
                    "schema_version": EVALUATION_SCHEMA_VERSION,
                    "retriever": "bm25",
                    "query_id": query["query_id"],
                    "response_id": query["response_id"],
                    "passage_id": passage["passage_id"],
                    "doc_id": passage["doc_id"],
                    "rank": rank,
                    "score": scores[index],
                    "split": split,
                }
            )
        print(f"[{position}/{len(queries)}] BM25 ranked {query['query_id']}", flush=True)
    return run_rows


def _model_digest(model_name: str) -> str:
    import ollama

    response = ollama.list()
    models = getattr(response, "models", None)
    if models is None and isinstance(response, Mapping):
        models = response.get("models", [])
    for model in models or []:
        name = getattr(model, "model", None) or getattr(model, "name", None)
        digest = getattr(model, "digest", None)
        if isinstance(model, Mapping):
            name = name or model.get("model") or model.get("name")
            digest = digest or model.get("digest")
        if name == model_name and isinstance(digest, str):
            return digest
    raise RuntimeError(
        f"Required embedding model is not installed in Ollama: {model_name}. "
        f"Run: ollama pull {model_name}"
    )


def _ollama_embed_batch(model: str, texts: Sequence[str], truncate: bool) -> list[list[float]]:
    import ollama

    response = ollama.embed(model=model, input=list(texts), truncate=truncate)
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, Mapping):
        embeddings = response.get("embeddings")
    if not isinstance(embeddings, Sequence) or len(embeddings) != len(texts):
        raise RuntimeError("Ollama returned an unexpected embedding batch")
    return [list(map(float, vector)) for vector in embeddings]


def _embedding_fingerprint(
    corpus_paths: RetrievalPaths,
    queries_path: Path,
    config: Mapping[str, Any],
    model_digest: str,
) -> str:
    return canonical_json_hash(
        {
            "passages_sha256": sha256_file(corpus_paths.passages),
            "queries_sha256": sha256_file(queries_path),
            "dense": config["dense"],
            "model_digest": model_digest,
            "ranking_passage_fields": config["ranking_passage_fields"],
        }
    )


def _build_or_resume_passage_embeddings(
    project_root: Path,
    corpus_paths: RetrievalPaths,
    eval_paths: RetrievalEvaluationPaths,
    passages: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_digest: str,
    embed_batch: Callable[[str, Sequence[str], bool], list[list[float]]],
) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Dense retrieval requires numpy; install requirements.txt") from exc

    dense = config["dense"]
    model = str(dense["model"])
    batch_size = int(dense["batch_size"])
    truncate = bool(dense["truncate"])
    prefix = str(dense["document_prefix"])
    fingerprint = _embedding_fingerprint(
        corpus_paths, eval_paths.queries_dev, config, model_digest
    )
    final_report = load_json(eval_paths.dense_embeddings_report, allow_missing=True)
    if (
        eval_paths.dense_embeddings.is_file()
        and final_report.get("status") == "complete"
        and final_report.get("embedding_fingerprint") == fingerprint
        and final_report.get("embeddings_sha256")
        == sha256_file(eval_paths.dense_embeddings)
    ):
        matrix = np.load(eval_paths.dense_embeddings, mmap_mode="r")
        if matrix.shape[0] != len(passages):
            raise ValueError("Cached embedding row count is stale")
        atomic_write_json(
            eval_paths.dense_embeddings_state,
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "status": "complete",
                "embedding_fingerprint": fingerprint,
                "completed_count": len(passages),
                "passage_count": len(passages),
                "dimension": int(matrix.shape[1]),
                "model": model,
                "model_digest": model_digest,
                "updated_at": utc_now(),
            },
        )
        print(f"Dense embeddings: verified cache with {matrix.shape[0]} rows", flush=True)
        return matrix

    state = load_json(eval_paths.dense_embeddings_state, allow_missing=True)
    completed = 0
    dimension = 0
    matrix = None
    if (
        eval_paths.dense_embeddings_partial.is_file()
        and state.get("embedding_fingerprint") == fingerprint
        and isinstance(state.get("completed_count"), int)
        and isinstance(state.get("dimension"), int)
    ):
        completed = int(state["completed_count"])
        dimension = int(state["dimension"])
        matrix = np.lib.format.open_memmap(
            eval_paths.dense_embeddings_partial,
            mode="r+",
            dtype="float32",
            shape=(len(passages), dimension),
        )
        print(f"Dense embeddings: resuming at {completed}/{len(passages)}", flush=True)

    while completed < len(passages):
        end = min(len(passages), completed + batch_size)
        texts = [
            prefix + _passage_text(passages[index], config)
            for index in range(completed, end)
        ]
        vectors = embed_batch(model, texts, truncate)
        batch = np.asarray(vectors, dtype=np.float32)
        if batch.ndim != 2 or batch.shape[0] != len(texts):
            raise RuntimeError("Dense embedding batch has invalid shape")
        norms = np.linalg.norm(batch, axis=1)
        if np.any(~np.isfinite(batch)) or np.any(norms <= 0):
            raise RuntimeError("Dense embedding batch contains invalid vectors")
        batch = batch / norms[:, None]
        if matrix is None:
            dimension = int(batch.shape[1])
            eval_paths.dense_embeddings_partial.parent.mkdir(parents=True, exist_ok=True)
            matrix = np.lib.format.open_memmap(
                eval_paths.dense_embeddings_partial,
                mode="w+",
                dtype="float32",
                shape=(len(passages), dimension),
            )
        elif batch.shape[1] != dimension:
            raise RuntimeError("Dense embedding dimension changed between batches")
        matrix[completed:end] = batch
        matrix.flush()
        completed = end
        atomic_write_json(
            eval_paths.dense_embeddings_state,
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "status": "partial",
                "embedding_fingerprint": fingerprint,
                "completed_count": completed,
                "passage_count": len(passages),
                "dimension": dimension,
                "model": model,
                "model_digest": model_digest,
                "updated_at": utc_now(),
            },
        )
        print(f"[{completed}/{len(passages)}] dense passages embedded", flush=True)

    if matrix is None:
        raise RuntimeError("No passage embeddings were generated")
    matrix.flush()
    del matrix
    os.replace(eval_paths.dense_embeddings_partial, eval_paths.dense_embeddings)
    embeddings_hash = sha256_file(eval_paths.dense_embeddings)
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "generated_at": utc_now(),
        "embedding_fingerprint": fingerprint,
        "passage_count": len(passages),
        "dimension": dimension,
        "model": model,
        "model_digest": model_digest,
        "document_prefix": prefix,
        "truncate": truncate,
        "embeddings_sha256": embeddings_hash,
        "input_passages_sha256": sha256_file(corpus_paths.passages),
        "artifact": project_relative(project_root, eval_paths.dense_embeddings),
    }
    atomic_write_json(eval_paths.dense_embeddings_report, report)
    atomic_write_json(
        eval_paths.dense_embeddings_state,
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "complete",
            "embedding_fingerprint": fingerprint,
            "completed_count": len(passages),
            "passage_count": len(passages),
            "dimension": dimension,
            "model": model,
            "model_digest": model_digest,
            "updated_at": utc_now(),
        },
    )
    return np.load(eval_paths.dense_embeddings, mmap_mode="r")


def _rank_dense(
    project_root: Path,
    corpus_paths: RetrievalPaths,
    eval_paths: RetrievalEvaluationPaths,
    passages: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    split: str,
) -> tuple[list[dict[str, Any]], str]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Dense retrieval requires numpy; install requirements.txt") from exc
    dense = config["dense"]
    model = str(dense["model"])
    model_digest = _model_digest(model)
    expected_prefix = dense.get("expected_digest_prefix")
    if (
        isinstance(expected_prefix, str)
        and not model_digest.startswith(expected_prefix)
    ):
        raise RuntimeError(
            f"Embedding model digest mismatch: expected {expected_prefix}, got {model_digest}"
        )
    passage_matrix = _build_or_resume_passage_embeddings(
        project_root,
        corpus_paths,
        eval_paths,
        passages,
        config,
        model_digest,
        _ollama_embed_batch,
    )
    query_texts = [str(dense["query_prefix"]) + str(row["text"]) for row in queries]
    vectors = _ollama_embed_batch(model, query_texts, bool(dense["truncate"]))
    query_matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(query_matrix, axis=1)
    if np.any(~np.isfinite(query_matrix)) or np.any(norms <= 0):
        raise RuntimeError("Dense query embeddings contain invalid vectors")
    query_matrix = query_matrix / norms[:, None]
    if query_matrix.shape[1] != passage_matrix.shape[1]:
        raise RuntimeError("Query and passage embedding dimensions differ")
    scores = query_matrix @ passage_matrix.T
    maximum_rank = int(config["maximum_rank"])
    passage_ids = [str(row["passage_id"]) for row in passages]
    run_rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        ranked = sorted(
            range(len(passages)),
            key=lambda index: (-float(scores[query_index, index]), passage_ids[index]),
        )[:maximum_rank]
        for rank, index in enumerate(ranked, start=1):
            passage = passages[index]
            run_rows.append(
                {
                    "schema_version": EVALUATION_SCHEMA_VERSION,
                    "retriever": "dense_nomic",
                    "query_id": query["query_id"],
                    "response_id": query["response_id"],
                    "passage_id": passage["passage_id"],
                    "doc_id": passage["doc_id"],
                    "rank": rank,
                    "score": float(scores[query_index, index]),
                    "split": split,
                }
            )
        print(
            f"[{query_index + 1}/{len(queries)}] Dense ranked {query['query_id']}",
            flush=True,
        )
    return run_rows, model_digest


def _group_run(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in rows:
        query_id = row.get("query_id")
        rank = row.get("rank")
        if not isinstance(query_id, str) or isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError("Run rows require query_id and integer rank")
        if (query_id, rank) in seen:
            raise ValueError("Duplicate query/rank pair in retrieval run")
        seen.add((query_id, rank))
        grouped[query_id].append(row)
    for query_rows in grouped.values():
        query_rows.sort(key=lambda row: int(row["rank"]))
        if [row["rank"] for row in query_rows] != list(range(1, len(query_rows) + 1)):
            raise ValueError("Retrieval run ranks must be contiguous from 1")
    return grouped


def _rank_hybrid(
    eval_paths: RetrievalEvaluationPaths,
    queries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    artifacts = _split_artifacts(eval_paths, split)
    return _fuse_hybrid_runs(
        load_jsonl(artifacts["bm25_run"]),
        load_jsonl(artifacts["dense_run"]),
        queries,
        config,
        split,
    )


def _fuse_hybrid_runs(
    bm25_rows: Sequence[Mapping[str, Any]],
    dense_rows: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    """Fuse two in-memory runs with the frozen RRF configuration."""

    bm25 = _group_run(bm25_rows)
    dense = _group_run(dense_rows)
    expected_queries = {str(row["query_id"]) for row in queries}
    if set(bm25) != expected_queries or set(dense) != expected_queries:
        raise ValueError("Hybrid inputs must cover the exact query set")
    settings = config["hybrid"]
    constant = int(settings["rrf_constant"])
    depth = int(settings["fusion_depth"])
    maximum_rank = int(config["maximum_rank"])
    output: list[dict[str, Any]] = []
    for position, query in enumerate(queries, start=1):
        query_id = str(query["query_id"])
        passage_meta: dict[str, Mapping[str, Any]] = {}
        scores: defaultdict[str, float] = defaultdict(float)
        for component in (bm25[query_id][:depth], dense[query_id][:depth]):
            for row in component:
                passage_id = str(row["passage_id"])
                passage_meta[passage_id] = row
                scores[passage_id] += 1.0 / (constant + int(row["rank"]))
        ranked = sorted(scores, key=lambda passage_id: (-scores[passage_id], passage_id))[
            :maximum_rank
        ]
        for rank, passage_id in enumerate(ranked, start=1):
            source = passage_meta[passage_id]
            output.append(
                {
                    "schema_version": EVALUATION_SCHEMA_VERSION,
                    "retriever": "hybrid_rrf",
                    "query_id": query_id,
                    "response_id": query["response_id"],
                    "passage_id": passage_id,
                    "doc_id": source["doc_id"],
                    "rank": rank,
                    "score": scores[passage_id],
                    "split": split,
                    "rrf_constant": constant,
                    "fusion_depth": depth,
                }
            )
        print(f"[{position}/{len(queries)}] Hybrid ranked {query_id}", flush=True)
    return output


def rank_frozen_hybrid_queries(
    project_root: Path,
    corpus_paths: RetrievalPaths,
    eval_paths: RetrievalEvaluationPaths,
    config: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    *,
    split_label: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    """Rank arbitrary label-free claims with the frozen BM25/Dense/Hybrid stack.

    This helper reuses the verified canonical passage embeddings but never
    reads qrels, labels, evidence stance, or claim-specific gold URL mappings.
    It returns passages plus in-memory BM25, Dense, and Hybrid runs without
    overwriting Study I run artifacts.
    """

    passages, _, _ = _validate_upstream(project_root, corpus_paths)
    if not queries:
        raise ValueError("At least one external retrieval query is required")
    seen: set[str] = set()
    normalized_queries: list[dict[str, Any]] = []
    for row in queries:
        query_id = row.get("query_id")
        response_id = row.get("response_id")
        text = row.get("text")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("External query has an invalid query_id")
        if query_id in seen:
            raise ValueError(f"Duplicate external query_id: {query_id}")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError(f"External query {query_id} has no response_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"External query {query_id} has no text")
        seen.add(query_id)
        normalized_queries.append(
            {
                "query_id": query_id,
                "response_id": response_id,
                "text": text.strip(),
            }
        )
    bm25_rows = _rank_bm25(
        passages,
        normalized_queries,
        config,
        split_label,
    )
    dense_rows, model_digest = _rank_dense(
        project_root,
        corpus_paths,
        eval_paths,
        passages,
        normalized_queries,
        config,
        split_label,
    )
    hybrid_rows = _fuse_hybrid_runs(
        bm25_rows,
        dense_rows,
        normalized_queries,
        config,
        split_label,
    )
    return passages, bm25_rows, dense_rows, hybrid_rows, model_digest


def _dcg(relevances: Sequence[int]) -> float:
    return sum((2**relevance - 1) / math.log2(rank + 1) for rank, relevance in enumerate(relevances, start=1))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot take a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_bootstrap_ci(
    values: Mapping[str, float],
    response_by_query: Mapping[str, str],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    by_response: dict[str, list[float]] = defaultdict(list)
    for query_id, value in values.items():
        by_response[response_by_query[query_id]].append(float(value))
    response_ids = sorted(by_response)
    if not response_ids:
        return {"lower": None, "upper": None, "cluster_count": 0}
    samples = int(settings["samples"])
    rng = random.Random(int(settings["seed"]))
    estimates = []
    for _ in range(samples):
        selected = [rng.choice(response_ids) for _ in response_ids]
        sample_values = [value for response_id in selected for value in by_response[response_id]]
        estimates.append(sum(sample_values) / len(sample_values))
    alpha = 1.0 - float(settings["confidence_level"])
    return {
        "lower": _percentile(estimates, alpha / 2.0),
        "upper": _percentile(estimates, 1.0 - alpha / 2.0),
        "cluster_count": len(response_ids),
        "samples": samples,
        "seed": int(settings["seed"]),
        "unit": "response_id",
    }


def _paired_cluster_bootstrap_difference_ci(
    dense_values: Mapping[str, float],
    bm25_values: Mapping[str, float],
    response_by_query: Mapping[str, str],
    settings: Mapping[str, Any],
    *,
    estimand: str = "dense_minus_bm25",
) -> dict[str, Any]:
    if set(dense_values) != set(bm25_values):
        raise ValueError("Paired bootstrap requires identical query IDs")
    differences = {
        query_id: float(dense_values[query_id]) - float(bm25_values[query_id])
        for query_id in dense_values
    }
    result = _cluster_bootstrap_ci(differences, response_by_query, settings)
    result["estimand"] = estimand
    return result


def evaluate_run(
    project_root: Path,
    corpus_paths: RetrievalPaths,
    eval_paths: RetrievalEvaluationPaths,
    config: Mapping[str, Any],
    run_path: Path,
    retriever: str,
    *,
    split: str = "dev",
    model_digest: str | None = None,
) -> dict[str, Any]:
    artifacts = _split_artifacts(eval_paths, split)
    queries = load_jsonl(artifacts["queries"])
    source_qrels = load_jsonl(artifacts["source_qrels"])
    strict_qrels = load_jsonl(artifacts["strict_qrels"])
    run_rows = load_jsonl(run_path)
    grouped = _group_run(run_rows)
    query_ids = [str(row["query_id"]) for row in queries]
    if set(grouped) != set(query_ids):
        raise ValueError(f"Retrieval run does not cover the exact {split} query set")
    response_by_query = {str(row["query_id"]): str(row["response_id"]) for row in queries}
    source_relevant: dict[str, set[str]] = defaultdict(set)
    for row in source_qrels:
        source_relevant[str(row["query_id"])].add(str(row["doc_id"]))
    strict_relevant: dict[str, dict[str, int]] = defaultdict(dict)
    for row in strict_qrels:
        strict_relevant[str(row["query_id"])][str(row["passage_id"])] = int(row["relevance"])
    cutoffs = [int(value) for value in config["metric_cutoffs"]]
    per_query: dict[str, dict[str, Any]] = {}
    for query_id in query_ids:
        ranking = grouped[query_id]
        source_first = next(
            (
                int(row["rank"])
                for row in ranking
                if str(row["doc_id"]) in source_relevant.get(query_id, set())
            ),
            None,
        )
        strict_first = next(
            (
                int(row["rank"])
                for row in ranking
                if str(row["passage_id"]) in strict_relevant.get(query_id, {})
            ),
            None,
        )
        strict_metrics: dict[str, float] = {}
        if query_id in strict_relevant:
            ideal = sorted(strict_relevant[query_id].values(), reverse=True)
            for cutoff in cutoffs:
                observed = [
                    strict_relevant[query_id].get(str(row["passage_id"]), 0)
                    for row in ranking[:cutoff]
                ]
                ideal_dcg = _dcg(ideal[:cutoff])
                strict_metrics[f"ndcg_at_{cutoff}"] = (
                    _dcg(observed) / ideal_dcg if ideal_dcg else 0.0
                )
        per_query[query_id] = {
            "response_id": response_by_query[query_id],
            "source_eligible": query_id in source_relevant,
            "strict_passage_eligible": query_id in strict_relevant,
            "source_first_relevant_rank": source_first,
            "strict_passage_first_relevant_rank": strict_first,
            "source": {
                **{
                    f"recall_at_{cutoff}": float(
                        source_first is not None and source_first <= cutoff
                    )
                    for cutoff in cutoffs
                },
                "mrr_at_10": (
                    1.0 / source_first
                    if source_first is not None and source_first <= 10
                    else 0.0
                ),
            },
            "strict_passage": {
                **{
                    f"recall_at_{cutoff}": float(
                        strict_first is not None and strict_first <= cutoff
                    )
                    for cutoff in cutoffs
                },
                "mrr_at_10": (
                    1.0 / strict_first
                    if strict_first is not None and strict_first <= 10
                    else 0.0
                ),
                **strict_metrics,
            },
        }

    source_queries = [query_id for query_id in query_ids if query_id in source_relevant]
    strict_queries = [query_id for query_id in query_ids if query_id in strict_relevant]

    def aggregate(
        eligible_queries: Sequence[str], level: str, metric: str
    ) -> dict[str, Any]:
        values = {
            query_id: float(per_query[query_id][level][metric])
            for query_id in eligible_queries
        }
        point = sum(values.values()) / len(values)
        return {
            "value": point,
            "query_count": len(values),
            "response_cluster_bootstrap_95_ci": _cluster_bootstrap_ci(
                values, response_by_query, config["bootstrap"]
            ),
        }

    source_metrics = {
        f"conditional_recall_at_{cutoff}": aggregate(
            source_queries, "source", f"recall_at_{cutoff}"
        )
        for cutoff in cutoffs
    }
    source_metrics["conditional_mrr_at_10"] = aggregate(
        source_queries, "source", "mrr_at_10"
    )
    for cutoff in cutoffs:
        source_metrics[f"all_{split}_end_to_end_recovery_at_{cutoff}"] = aggregate(
            query_ids, "source", f"recall_at_{cutoff}"
        )

    strict_metrics_report: dict[str, Any] = {}
    for cutoff in cutoffs:
        strict_metrics_report[f"recall_at_{cutoff}"] = aggregate(
            strict_queries, "strict_passage", f"recall_at_{cutoff}"
        )
        strict_metrics_report[f"ndcg_at_{cutoff}"] = aggregate(
            strict_queries, "strict_passage", f"ndcg_at_{cutoff}"
        )
    strict_metrics_report["mrr_at_10"] = aggregate(
        strict_queries, "strict_passage", "mrr_at_10"
    )

    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "generated_at": utc_now(),
        "retriever": retriever,
        "split": split,
        "query_count": len(query_ids),
        "source_document": {
            "eligible_query_count": len(source_queries),
            "corpus_coverage": len(source_queries) / len(query_ids),
            "metrics": source_metrics,
        },
        "strict_passage": {
            "eligible_query_count": len(strict_queries),
            "qrel_coverage": len(strict_queries) / len(query_ids),
            "metrics": strict_metrics_report,
        },
        "per_query": per_query,
        "model_digest": model_digest,
        "input_hashes": {
            "evaluation_config_sha256": canonical_json_hash(config),
            "queries_sha256": sha256_file(artifacts["queries"]),
            "passages_sha256": sha256_file(corpus_paths.passages),
            "source_qrels_sha256": sha256_file(artifacts["source_qrels"]),
            "strict_passage_qrels_sha256": sha256_file(artifacts["strict_qrels"]),
            "run_sha256": sha256_file(run_path),
        },
        "run_artifact": project_relative(project_root, run_path),
    }
    return report


def run_retriever(
    project_root: Path,
    corpus_paths: RetrievalPaths,
    eval_paths: RetrievalEvaluationPaths,
    config: Mapping[str, Any],
    *,
    retriever: str,
    split: str = "dev",
    dry_run: bool = False,
) -> dict[str, Any]:
    artifacts = _split_artifacts(eval_paths, split)
    passages, queries = _validate_evaluation_inputs(
        project_root, corpus_paths, eval_paths, config, split
    )
    if dry_run:
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "validated",
            "retriever": retriever,
            "split": split,
            "query_count": len(queries),
            "passage_count": len(passages),
            "maximum_rank": config["maximum_rank"],
        }
    if retriever == "bm25":
        run_rows = _rank_bm25(passages, queries, config, split)
        run_path = artifacts["bm25_run"]
        metrics_path = artifacts["bm25_metrics"]
        model_digest = None
    elif retriever == "dense":
        run_rows, model_digest = _rank_dense(
            project_root,
            corpus_paths,
            eval_paths,
            passages,
            queries,
            config,
            split,
        )
        run_path = artifacts["dense_run"]
        metrics_path = artifacts["dense_metrics"]
    elif retriever == "hybrid":
        run_rows = _rank_hybrid(eval_paths, queries, config, split)
        run_path = artifacts["hybrid_run"]
        metrics_path = artifacts["hybrid_metrics"]
        model_digest = None
    else:
        raise ValueError("retriever must be 'bm25', 'dense', or 'hybrid'")
    expected_rows = len(queries) * int(config["maximum_rank"])
    if len(run_rows) != expected_rows:
        raise ValueError("Retrieval run row count is incomplete")
    atomic_write_jsonl(run_path, run_rows)
    report = evaluate_run(
        project_root,
        corpus_paths,
        eval_paths,
        config,
        run_path,
        {
            "bm25": "bm25",
            "dense": "dense_nomic",
            "hybrid": "hybrid_rrf",
        }[retriever],
        split=split,
        model_digest=model_digest,
    )
    atomic_write_json(metrics_path, report)
    return report


def _comparison_markdown(report: Mapping[str, Any]) -> str:
    split = str(report["split"])
    split_label = "Dev" if split == "dev" else "Held-out"
    lines = [
        f"# Study I {split} two-level retrieval comparison",
        "",
        f"- Status: **{report['status']}**",
        f"- {split_label} queries: **{report['query_count']}**",
        f"- Source-covered queries: **{report['source_eligible_query_count']}**",
        f"- Strict-passage-qrel queries: **{report['strict_eligible_query_count']}**",
        "",
        "| Metric | BM25 | Dense | Hybrid RRF |",
        "|---|---:|---:|---:|",
    ]
    for row in report["metric_comparison"]:
        lines.append(
            f"| `{row['metric']}` | {row['bm25']:.4f} | {row['dense']:.4f} | {row['hybrid']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Primary metric: `{report['selection']['primary_metric']}`.",
            f"Decision: **{report['selection']['decision']}**.",
            "",
            "Source-document metrics evaluate whether a top-ranked passage comes",
            "from a benchmark-associated frozen source. Strict passage metrics use",
            "only automatically accepted positive passage qrels. Unjudged candidates",
            "are never treated as relevant.",
            "",
        ]
    )
    return "\n".join(lines)


def compare_retrievers(
    project_root: Path,
    corpus_paths: RetrievalPaths,
    eval_paths: RetrievalEvaluationPaths,
    config: Mapping[str, Any],
    *,
    split: str = "dev",
    dry_run: bool = False,
) -> dict[str, Any]:
    artifacts = _split_artifacts(eval_paths, split)
    bm25 = load_json(artifacts["bm25_metrics"])
    dense = load_json(artifacts["dense_metrics"])
    hybrid = load_json(artifacts["hybrid_metrics"])
    for report in (bm25, dense, hybrid):
        if report.get("status") != "complete" or report.get("split") != split:
            raise ValueError(f"All current {split} retrieval reports are required")
        expected = {
            "evaluation_config_sha256": canonical_json_hash(config),
            "queries_sha256": sha256_file(artifacts["queries"]),
            "passages_sha256": sha256_file(corpus_paths.passages),
            "source_qrels_sha256": sha256_file(artifacts["source_qrels"]),
            "strict_passage_qrels_sha256": sha256_file(artifacts["strict_qrels"]),
        }
        for key, value in expected.items():
            if report.get("input_hashes", {}).get(key) != value:
                raise ValueError(f"Stale retrieval report: {report.get('retriever')} {key}")

    metric_paths = [
        ("source_document.conditional_recall_at_1", "source_document", "conditional_recall_at_1"),
        ("source_document.conditional_recall_at_5", "source_document", "conditional_recall_at_5"),
        ("source_document.conditional_recall_at_10", "source_document", "conditional_recall_at_10"),
        ("source_document.conditional_mrr_at_10", "source_document", "conditional_mrr_at_10"),
        (
            f"source_document.all_{split}_end_to_end_recovery_at_5",
            "source_document",
            f"all_{split}_end_to_end_recovery_at_5",
        ),
        ("strict_passage.recall_at_1", "strict_passage", "recall_at_1"),
        ("strict_passage.recall_at_5", "strict_passage", "recall_at_5"),
        ("strict_passage.recall_at_10", "strict_passage", "recall_at_10"),
        ("strict_passage.mrr_at_10", "strict_passage", "mrr_at_10"),
        ("strict_passage.ndcg_at_10", "strict_passage", "ndcg_at_10"),
    ]
    comparison = []
    response_by_query = {
        query_id: str(row["response_id"])
        for query_id, row in bm25["per_query"].items()
    }
    for name, level, metric in metric_paths:
        bm25_value = float(bm25[level]["metrics"][metric]["value"])
        dense_value = float(dense[level]["metrics"][metric]["value"])
        hybrid_value = float(hybrid[level]["metrics"][metric]["value"])
        if name.startswith("source_document.conditional_"):
            eligible = [
                query_id
                for query_id, row in bm25["per_query"].items()
                if row["source_eligible"]
            ]
            per_query_level = "source"
            per_query_metric = metric.removeprefix("conditional_")
        elif name.startswith(f"source_document.all_{split}_"):
            eligible = list(bm25["per_query"])
            per_query_level = "source"
            per_query_metric = metric.replace(
                f"all_{split}_end_to_end_recovery_at_", "recall_at_", 1
            )
        else:
            eligible = [
                query_id
                for query_id, row in bm25["per_query"].items()
                if row["strict_passage_eligible"]
            ]
            per_query_level = "strict_passage"
            per_query_metric = metric
        bm25_values = {
            query_id: float(
                bm25["per_query"][query_id][per_query_level][per_query_metric]
            )
            for query_id in eligible
        }
        dense_values = {
            query_id: float(
                dense["per_query"][query_id][per_query_level][per_query_metric]
            )
            for query_id in eligible
        }
        hybrid_values = {
            query_id: float(
                hybrid["per_query"][query_id][per_query_level][per_query_metric]
            )
            for query_id in eligible
        }
        comparison.append(
            {
                "metric": name,
                "bm25": bm25_value,
                "dense": dense_value,
                "hybrid": hybrid_value,
                "difference": dense_value - bm25_value,
                "paired_response_cluster_bootstrap_95_ci": (
                    _paired_cluster_bootstrap_difference_ci(
                        dense_values,
                        bm25_values,
                        response_by_query,
                        config["bootstrap"],
                        estimand="dense_minus_bm25",
                    )
                ),
                "hybrid_minus_bm25": hybrid_value - bm25_value,
                "hybrid_minus_bm25_paired_response_cluster_bootstrap_95_ci": (
                    _paired_cluster_bootstrap_difference_ci(
                        hybrid_values,
                        bm25_values,
                        response_by_query,
                        config["bootstrap"],
                        estimand="hybrid_minus_bm25",
                    )
                ),
                "hybrid_minus_dense": hybrid_value - dense_value,
                "hybrid_minus_dense_paired_response_cluster_bootstrap_95_ci": (
                    _paired_cluster_bootstrap_difference_ci(
                        hybrid_values,
                        dense_values,
                        response_by_query,
                        config["bootstrap"],
                        estimand="hybrid_minus_dense",
                    )
                ),
            }
        )
    primary_name = str(config["primary_selection_metric"])
    secondary_name = str(config["secondary_confirmation_metric"])
    by_name = {row["metric"]: row for row in comparison}
    primary_hybrid = by_name[primary_name]["hybrid"]
    secondary_hybrid = by_name[secondary_name]["hybrid"]
    tolerance = 1e-12
    if split == "heldout":
        dev_comparison = load_json(eval_paths.comparison_json)
        dev_selection = dev_comparison.get("selection", {})
        if dev_selection.get("configuration_is_frozen") is not True:
            raise ValueError("Held-out comparison requires a frozen dev selection")
        decision = "frozen_hybrid_rrf_evaluated_without_reselection"
        configuration_is_frozen = True
        heldout_remains_sealed = False
        selected_configuration = dev_selection.get("selected_configuration")
    elif (
        primary_hybrid > max(by_name[primary_name]["bm25"], by_name[primary_name]["dense"]) + tolerance
        and secondary_hybrid
        >= max(by_name[secondary_name]["bm25"], by_name[secondary_name]["dense"])
        - tolerance
    ):
        decision = "hybrid_rrf_selected_on_primary_and_secondary"
        configuration_is_frozen = True
        heldout_remains_sealed = True
        selected_configuration = {
            "retriever": "hybrid_rrf",
            "rrf_constant": int(config["hybrid"]["rrf_constant"]),
            "fusion_depth": int(config["hybrid"]["fusion_depth"]),
            "maximum_rank": int(config["maximum_rank"]),
            "chunking": "384_tokens_64_overlap_from_canonical_passages",
            "dense_model": config["dense"]["model"],
            "evidence_top_k": 5,
        }
    else:
        decision = "no_single_configuration_dominates_preregistered_metrics"
        configuration_is_frozen = False
        heldout_remains_sealed = True
        selected_configuration = None
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "validated" if dry_run else "complete",
        "generated_at": utc_now(),
        "split": split,
        "query_count": int(bm25["query_count"]),
        "source_eligible_query_count": int(
            bm25["source_document"]["eligible_query_count"]
        ),
        "strict_eligible_query_count": int(
            bm25["strict_passage"]["eligible_query_count"]
        ),
        "metric_comparison": comparison,
        "selection": {
            "primary_metric": primary_name,
            "secondary_confirmation_metric": secondary_name,
            "decision": decision,
            "configuration_is_frozen": configuration_is_frozen,
            "heldout_remains_sealed": heldout_remains_sealed,
            "selected_configuration": selected_configuration,
        },
        "input_hashes": {
            "evaluation_config_sha256": canonical_json_hash(config),
            "bm25_metrics_sha256": sha256_file(artifacts["bm25_metrics"]),
            "dense_metrics_sha256": sha256_file(artifacts["dense_metrics"]),
            "hybrid_metrics_sha256": sha256_file(artifacts["hybrid_metrics"]),
        },
        "artifacts": {
            "bm25_metrics": project_relative(project_root, artifacts["bm25_metrics"]),
            "dense_metrics": project_relative(project_root, artifacts["dense_metrics"]),
            "hybrid_metrics": project_relative(project_root, artifacts["hybrid_metrics"]),
        },
    }
    if not dry_run:
        atomic_write_json(artifacts["comparison_json"], report)
        atomic_write_text(artifacts["comparison_markdown"], _comparison_markdown(report))
    return report
