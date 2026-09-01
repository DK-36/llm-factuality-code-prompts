#!/usr/bin/env python3
"""Analyse VeriScore F1@K sensitivity from frozen X2 response scores.

This script performs no model, extraction, verification, retrieval, or search
calls. It validates and reuses the held-out response-level counts written by X2,
recomputes F1 for the frozen K grid, and writes X3 results plus a PNG figure.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from factcheck_bench_cove import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    load_json,
    load_jsonl,
    sha256_file,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "data"
    / "factcheck_bench"
    / "cove"
    / "config"
    / "cove_external_veriscore_k_sensitivity_config.json"
)

CONDITIONS = (
    ("initial", "Initial"),
    ("a", "A: Standard CoVe"),
    ("b", "B: Evidence-grounded CoVe"),
    ("c", "C: Extra-revision control"),
    ("d2", "D: Selective verifier revision"),
)

COLORS = {
    "initial": "#4D4D4D",
    "a": "#0072B2",
    "b": "#009E73",
    "c": "#E69F00",
    "d2": "#CC79A7",
    "b_minus_a": "#009E73",
    "d_minus_c": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and calculate without writing X3 artifacts.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def project_path(value: str) -> Path:
    path = PROJECT_ROOT / value
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Configured path escapes project root: {value}") from exc
    return path


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def f1_at_k(row: dict[str, Any], k: int) -> float:
    precision = float(row["factual_precision"])
    supported = int(row["supported_claim_count"])
    recall = min(supported / k, 1.0)
    if recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def validate_inputs(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    primary: dict[str, Any],
    external_cfg: dict[str, Any],
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    if cfg.get("schema_version") != "fcb_cove_external_veriscore_k_sensitivity_config_v1":
        raise ValueError("Unexpected K-sensitivity config schema")
    if cfg.get("status") != "frozen":
        raise ValueError("K-sensitivity config must be frozen")
    if primary.get("status") != "complete" or primary.get("split") != "heldout":
        raise ValueError("Frozen held-out X2 analysis is not complete")
    protocol = cfg["protocol"]
    frozen_k = int(protocol["frozen_primary_k"])
    if frozen_k != int(primary.get("shared_median_k", -1)):
        raise ValueError("X3 frozen primary K does not match X2")
    if external_cfg.get("schema_version") != "fcb_cove_external_veriscore_config_v1":
        raise ValueError("Unexpected external VeriScore config schema")
    expected_responses = int(external_cfg["splits"]["heldout"]["expected_responses"])

    condition_ids = [item[0] for item in CONDITIONS]
    by_condition: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    required = {
        "response_id",
        "condition_id",
        "verifiable_claim_count",
        "supported_claim_count",
        "factual_precision",
        "shared_median_k",
        "veriscore_f1_at_shared_median_k",
    }
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"X2 score row {index} is missing {sorted(missing)}")
        condition = str(row["condition_id"])
        response_id = str(row["response_id"])
        if condition not in condition_ids:
            raise ValueError(f"Unexpected condition {condition!r} at X2 row {index}")
        if response_id in by_condition[condition]:
            raise ValueError(f"Duplicate X2 score for {condition}/{response_id}")
        verifiable = int(row["verifiable_claim_count"])
        supported = int(row["supported_claim_count"])
        if verifiable < 0 or supported < 0 or supported > verifiable:
            raise ValueError(f"Invalid claim counts at X2 row {index}")
        expected_precision = supported / verifiable if verifiable else 0.0
        if not math.isclose(
            float(row["factual_precision"]), expected_precision, abs_tol=1e-12
        ):
            raise ValueError(f"Precision mismatch at X2 row {index}")
        if int(row["shared_median_k"]) != frozen_k:
            raise ValueError(f"Shared K mismatch at X2 row {index}")
        if not math.isclose(
            f1_at_k(row, frozen_k),
            float(row["veriscore_f1_at_shared_median_k"]),
            abs_tol=1e-12,
        ):
            raise ValueError(f"Frozen F1@K reproduction failed at X2 row {index}")
        by_condition[condition][response_id] = row

    if set(by_condition) != set(condition_ids):
        raise ValueError("X2 score file does not contain all frozen conditions")
    response_ids = sorted(by_condition[condition_ids[0]])
    if len(response_ids) != expected_responses:
        raise ValueError(
            f"Expected {expected_responses} held-out responses, found {len(response_ids)}"
        )
    expected_ids = set(response_ids)
    for condition in condition_ids:
        if set(by_condition[condition]) != expected_ids:
            raise ValueError(f"Condition {condition} is not response-paired")
    if len(rows) != expected_responses * len(condition_ids):
        raise ValueError("Unexpected X2 evaluation-unit count")
    return response_ids, dict(by_condition)


def condition_means(
    by_condition: dict[str, dict[str, dict[str, Any]]],
    response_ids: list[str],
    k_grid: list[int],
) -> dict[str, dict[int, float]]:
    return {
        condition: {
            k: statistics.fmean(
                f1_at_k(by_condition[condition][response_id], k)
                for response_id in response_ids
            )
            for k in k_grid
        }
        for condition, _ in CONDITIONS
    }


def paired_contrasts(
    by_condition: dict[str, dict[str, dict[str, Any]]],
    response_ids: list[str],
    k_grid: list[int],
    comparisons: list[dict[str, str]],
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, dict[int, dict[str, Any]]]:
    differences: dict[str, dict[int, list[float]]] = {}
    for comparison in comparisons:
        name = comparison["name"]
        treatment = comparison["treatment"]
        baseline = comparison["baseline"]
        differences[name] = {
            k: [
                f1_at_k(by_condition[treatment][response_id], k)
                - f1_at_k(by_condition[baseline][response_id], k)
                for response_id in response_ids
            ]
            for k in k_grid
        }

    draws: dict[str, dict[int, list[float]]] = {
        comparison["name"]: {k: [] for k in k_grid}
        for comparison in comparisons
    }
    rng = random.Random(seed)
    response_count = len(response_ids)
    for _ in range(samples):
        selected = [rng.randrange(response_count) for _ in response_ids]
        for comparison in comparisons:
            name = comparison["name"]
            for k in k_grid:
                values = differences[name][k]
                draws[name][k].append(
                    statistics.fmean(values[index] for index in selected)
                )

    alpha = 1.0 - confidence_level
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for comparison in comparisons:
        name = comparison["name"]
        result[name] = {}
        for k in k_grid:
            lower = percentile(draws[name][k], alpha / 2.0)
            upper = percentile(draws[name][k], 1.0 - alpha / 2.0)
            result[name][k] = {
                "point_estimate": statistics.fmean(differences[name][k]),
                "lower": lower,
                "upper": upper,
                "includes_zero": lower <= 0.0 <= upper,
                "valid_replicates": len(draws[name][k]),
            }
    return result


def csv_text(rows: Iterable[Iterable[Any]]) -> str:
    stream = io.StringIO()
    import csv

    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue()


def load_font(size: int, bold: bool = False) -> Any:
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size, index=1 if bold else 0)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_dashed_line(
    draw: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str,
    width: int,
    dash: int = 10,
    gap: int = 7,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    position = 0.0
    while position < length:
        stop = min(position + dash, length)
        draw.line(
            (
                x1 + dx * position,
                y1 + dy * position,
                x1 + dx * stop,
                y1 + dy * stop,
            ),
            fill=fill,
            width=width,
        )
        position += dash + gap


def write_png(
    path: Path,
    k_grid: list[int],
    means: dict[str, dict[int, float]],
    contrasts: dict[str, dict[int, dict[str, Any]]],
    comparisons: list[dict[str, str]],
    frozen_k: int,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "PNG generation requires Pillow; install dependencies from requirements.txt"
        ) from exc

    width, height = 1800, 1500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(38, bold=True)
    subtitle_font = load_font(22)
    panel_font = load_font(27, bold=True)
    label_font = load_font(21)
    small_font = load_font(18)
    tick_font = load_font(18)

    draw.text((110, 52), "VeriScore F1@K sensitivity on 72 paired responses", fill="#111111", font=title_font)
    draw.text((110, 103), "Frozen X2 response scores; K=9 remains the primary external setting", fill="#555555", font=subtitle_font)

    left, right = 145, 1725

    def axes(
        top: int,
        bottom: int,
        y_min: float,
        y_max: float,
        y_ticks: list[float],
        panel_title: str,
        y_label: str,
    ) -> tuple[Callable[[int], float], Callable[[float], float]]:
        draw.text((left, top - 55), panel_title, fill="#222222", font=panel_font)
        plot_top = top
        plot_bottom = bottom
        x = lambda k: left + (k - k_grid[0]) / (k_grid[-1] - k_grid[0]) * (right - left)
        y = lambda value: plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
        for tick in y_ticks:
            y_pos = y(tick)
            draw.line((left, y_pos, right, y_pos), fill="#D9D9D9", width=2)
            text = f"{tick:.2f}"
            box = draw.textbbox((0, 0), text, font=tick_font)
            draw.text((left - 18 - (box[2] - box[0]), y_pos - 10), text, fill="#444444", font=tick_font)
        draw.line((left, plot_top, left, plot_bottom), fill="#333333", width=3)
        draw.line((left, plot_bottom, right, plot_bottom), fill="#333333", width=3)
        for tick in (1, 5, 9, 10, 15, 20):
            x_pos = x(tick)
            draw.line((x_pos, plot_bottom, x_pos, plot_bottom + 8), fill="#333333", width=2)
            text = str(tick)
            box = draw.textbbox((0, 0), text, font=tick_font)
            draw.text((x_pos - (box[2] - box[0]) / 2, plot_bottom + 11), text, fill="#333333", font=tick_font)
        frozen_x = x(frozen_k)
        draw_dashed_line(draw, (frozen_x, plot_top), (frozen_x, plot_bottom), "#B2182B", 3)
        draw.text((frozen_x + 10, plot_top + 8), "primary K=9", fill="#B2182B", font=small_font)
        x_label = "K"
        box = draw.textbbox((0, 0), x_label, font=label_font)
        draw.text(((left + right) / 2 - (box[2] - box[0]) / 2, plot_bottom + 47), x_label, fill="#222222", font=label_font)
        rotated = Image.new("RGBA", (260, 45), (255, 255, 255, 0))
        rotated_draw = ImageDraw.Draw(rotated)
        rotated_draw.text((0, 4), y_label, fill="#222222", font=label_font)
        rotated = rotated.rotate(90, expand=True)
        image.paste(rotated, (28, int((plot_top + plot_bottom) / 2 - rotated.height / 2)), rotated)
        return x, y

    legend_positions = [(150, 165), (585, 165), (1085, 165), (150, 205), (585, 205)]
    for (condition, label), (x_pos, y_pos) in zip(CONDITIONS, legend_positions, strict=True):
        draw.line((x_pos, y_pos + 10, x_pos + 55, y_pos + 10), fill=COLORS[condition], width=6)
        draw.text((x_pos + 68, y_pos), label, fill="#222222", font=small_font)

    x1, y1 = axes(
        295,
        700,
        0.30,
        0.80,
        [0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        "A. Mean response-level F1@K by condition",
        "Mean F1@K",
    )
    for condition, _ in CONDITIONS:
        points = [(x1(k), y1(means[condition][k])) for k in k_grid]
        draw.line(points, fill=COLORS[condition], width=6, joint="curve")
        x_pos, y_pos = x1(frozen_k), y1(means[condition][frozen_k])
        draw.ellipse((x_pos - 7, y_pos - 7, x_pos + 7, y_pos + 7), fill=COLORS[condition])

    contrast_values = [
        item[key]
        for comparison in comparisons
        for item in contrasts[comparison["name"]].values()
        for key in ("lower", "upper")
    ]
    contrast_min = min(-0.02, math.floor(min(contrast_values) * 20) / 20)
    contrast_max = max(0.10, math.ceil(max(contrast_values) * 20) / 20)
    ticks: list[float] = []
    tick = contrast_min
    while tick <= contrast_max + 1e-9:
        ticks.append(round(tick, 10))
        tick += 0.05

    for index, comparison in enumerate(comparisons):
        x_pos = 150 + index * 430
        y_pos = 825
        name = comparison["name"]
        draw.line((x_pos, y_pos + 10, x_pos + 55, y_pos + 10), fill=COLORS[name], width=6)
        draw.text((x_pos + 68, y_pos), comparison["display_name"], fill="#222222", font=small_font)

    x2, y2 = axes(
        905,
        1350,
        contrast_min,
        contrast_max,
        ticks,
        "B. Paired controlled contrasts with 95% bootstrap intervals",
        "F1@K difference",
    )
    if contrast_min <= 0.0 <= contrast_max:
        draw_dashed_line(draw, (left, y2(0.0)), (right, y2(0.0)), "#666666", 2)
    for comparison in comparisons:
        name = comparison["name"]
        for k in k_grid:
            item = contrasts[name][k]
            draw.line((x2(k), y2(item["lower"]), x2(k), y2(item["upper"])), fill=COLORS[name], width=3)
        points = [(x2(k), y2(contrasts[name][k]["point_estimate"])) for k in k_grid]
        draw.line(points, fill=COLORS[name], width=6, joint="curve")
        x_pos, y_pos = x2(frozen_k), y2(contrasts[name][frozen_k]["point_estimate"])
        draw.ellipse((x_pos - 7, y_pos - 7, x_pos + 7, y_pos + 7), fill=COLORS[name])

    footer = "External automatic response-level sensitivity analysis"
    box = draw.textbbox((0, 0), footer, font=small_font)
    draw.text((right - (box[2] - box[0]), 1460), footer, fill="#555555", font=small_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".png", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        image.save(stream, format="PNG", optimize=True)
    os.replace(temporary, path)


def markdown_report(
    means: dict[str, dict[int, float]],
    contrasts: dict[str, dict[int, dict[str, Any]]],
    comparisons: list[dict[str, str]],
    stability: dict[str, dict[str, Any]],
    frozen_k: int,
    k_grid: list[int],
    figure_path: str,
    fingerprints: dict[str, str],
) -> str:
    lines = [
        "# X3 - VeriScore K-Sensitivity Analysis (held-out)",
        "",
        "- Status: `complete`",
        "- Evidence strength: `EXTERNAL_AUTOMATIC_RESPONSE_LEVEL_SUPPLEMENT`",
        f"- K grid: `{k_grid[0]}--{k_grid[-1]}` (all integers)",
        f"- Frozen primary K: `{frozen_k}`",
        "- Paired held-out responses: `72`",
        f"- Figure: `{figure_path}`",
        "",
        "## Condition means at K=9",
        "",
        "| Condition | Mean F1@9 |",
        "|---|---:|",
    ]
    for condition, label in CONDITIONS:
        lines.append(f"| {label} | {means[condition][frozen_k]:.4f} |")
    lines.extend([
        "",
        "## Stability across the K grid",
        "",
        "| Contrast | Positive point estimate at every K | 95% interval excludes zero at every K | Point-estimate range |",
        "|---|---|---|---:|",
    ])
    for comparison in comparisons:
        item = stability[comparison["name"]]
        lines.append(
            f"| {comparison['display_name']} | "
            f"{'yes' if item['point_positive_all_k'] else 'no'} | "
            f"{'yes' if item['interval_positive_all_k'] else 'no'} | "
            f"{item['min_point_estimate']:+.4f} to {item['max_point_estimate']:+.4f} |"
        )
    lines.extend([
        "",
        "Point-estimate consistency and interval exclusion are reported separately. No K is selected post hoc, and K=9 remains the primary external setting.",
        "",
        "## Controlled contrasts at K=9",
        "",
        "| Contrast | Difference | 95% interval |",
        "|---|---:|---:|",
    ])
    for comparison in comparisons:
        item = contrasts[comparison["name"]][frozen_k]
        lines.append(
            f"| {comparison['display_name']} | {item['point_estimate']:+.4f} | "
            f"[{item['lower']:+.4f}, {item['upper']:+.4f}] |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "X3 reuses frozen X2 response-level counts and makes no new model, extraction, verification, retrieval, or search calls. VeriScore evaluates complete responses and is not human-gold validation of aligned initial-to-revised claim transitions.",
        "",
        "## Source fingerprints",
        "",
    ])
    for source, digest in fingerprints.items():
        lines.append(f"- `{source}`: `{digest}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    inputs = {name: project_path(path) for name, path in cfg["inputs"].items()}
    outputs = {name: project_path(path) for name, path in cfg["outputs"].items()}
    rows = load_jsonl(inputs["response_scores"])
    primary = load_json(inputs["primary_summary"])
    external_cfg = load_json(inputs["external_veriscore_config"])
    response_ids, by_condition = validate_inputs(cfg, rows, primary, external_cfg)

    protocol = cfg["protocol"]
    k_grid = list(
        range(
            int(protocol["k_min"]),
            int(protocol["k_max"]) + 1,
            int(protocol["k_step"]),
        )
    )
    frozen_k = int(protocol["frozen_primary_k"])
    if frozen_k not in k_grid:
        raise ValueError("K-sensitivity grid must include the frozen primary K")
    comparisons = protocol["paired_comparisons"]
    means = condition_means(by_condition, response_ids, k_grid)
    contrasts = paired_contrasts(
        by_condition,
        response_ids,
        k_grid,
        comparisons,
        int(protocol["bootstrap_samples"]),
        int(protocol["bootstrap_seed"]),
        float(protocol["confidence_level"]),
    )

    stability: dict[str, dict[str, Any]] = {}
    for comparison in comparisons:
        name = comparison["name"]
        points = [contrasts[name][k]["point_estimate"] for k in k_grid]
        stability[name] = {
            "point_positive_all_k": all(value > 0.0 for value in points),
            "interval_positive_all_k": all(contrasts[name][k]["lower"] > 0.0 for k in k_grid),
            "positive_point_k_values": [k for k in k_grid if contrasts[name][k]["point_estimate"] > 0.0],
            "interval_excludes_zero_k_values": [k for k in k_grid if contrasts[name][k]["lower"] > 0.0],
            "min_point_estimate": min(points),
            "max_point_estimate": max(points),
        }

    # X3 must reproduce all frozen K=9 condition and contrast point estimates.
    for condition, _ in CONDITIONS:
        expected = primary["condition_summary"][condition]["mean_veriscore_f1_at_shared_median_k"]
        if not math.isclose(means[condition][frozen_k], expected, abs_tol=1e-12):
            raise ValueError(f"K=9 condition reproduction failed for {condition}")
    for comparison in comparisons:
        name = comparison["name"]
        expected = primary["paired_response_bootstrap"]["comparisons"][name]["metrics"]["veriscore_f1_at_shared_median_k"]["point_estimate"]
        if not math.isclose(contrasts[name][frozen_k]["point_estimate"], expected, abs_tol=1e-12):
            raise ValueError(f"K=9 contrast reproduction failed for {name}")

    fingerprints = {
        relative(args.config): sha256_file(args.config),
        **{relative(path): sha256_file(path) for path in inputs.values()},
    }
    report = {
        "schema_version": "fcb_cove_external_veriscore_k_sensitivity_v1",
        "status": "complete",
        "evidence_strength": "EXTERNAL_AUTOMATIC_RESPONSE_LEVEL_SUPPLEMENT",
        "split": "heldout",
        "response_count": len(response_ids),
        "evaluation_unit_count": len(rows),
        "protocol": {
            "k_grid": k_grid,
            "frozen_primary_k": frozen_k,
            "bootstrap_samples": int(protocol["bootstrap_samples"]),
            "bootstrap_seed": int(protocol["bootstrap_seed"]),
            "confidence_level": float(protocol["confidence_level"]),
            "sampling_unit": protocol["sampling_unit"],
            "interval_method": protocol["interval_method"],
        },
        "condition_means": {
            condition: {str(k): means[condition][k] for k in k_grid}
            for condition, _ in CONDITIONS
        },
        "paired_contrasts": {
            comparison["name"]: {str(k): contrasts[comparison["name"]][k] for k in k_grid}
            for comparison in comparisons
        },
        "stability": stability,
        "primary_k_reproduction": "exact_point_estimates",
        "source_fingerprints": fingerprints,
        "figure": relative(outputs["figure_png"]),
        "interpretation_boundary": (
            "External automatic whole-response sensitivity analysis; not a "
            "human-gold initial-to-revised claim-transition evaluation."
        ),
        "generated_at": utc_now(),
    }

    condition_rows: list[list[Any]] = [["k", *(item[0] for item in CONDITIONS)]]
    for k in k_grid:
        condition_rows.append([k, *(f"{means[item[0]][k]:.12f}" for item in CONDITIONS)])
    contrast_rows: list[list[Any]] = [[
        "contrast", "k", "point_estimate", "lower_95", "upper_95", "includes_zero"
    ]]
    for comparison in comparisons:
        name = comparison["name"]
        for k in k_grid:
            item = contrasts[name][k]
            contrast_rows.append([
                name,
                k,
                f"{item['point_estimate']:.12f}",
                f"{item['lower']:.12f}",
                f"{item['upper']:.12f}",
                str(item["includes_zero"]).lower(),
            ])
    markdown = markdown_report(
        means,
        contrasts,
        comparisons,
        stability,
        frozen_k,
        k_grid,
        relative(outputs["figure_png"]),
        fingerprints,
    )

    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    atomic_write_json(outputs["summary_json"], report)
    atomic_write_text(outputs["report_markdown"], markdown)
    atomic_write_text(outputs["condition_csv"], csv_text(condition_rows))
    atomic_write_text(outputs["contrast_csv"], csv_text(contrast_rows))
    write_png(outputs["figure_png"], k_grid, means, contrasts, comparisons, frozen_k)
    print(json.dumps({
        "status": "complete",
        "split": "heldout",
        "response_count": len(response_ids),
        "k_grid": [k_grid[0], k_grid[-1]],
        "stability": stability,
        "summary": relative(outputs["summary_json"]),
        "figure": relative(outputs["figure_png"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
