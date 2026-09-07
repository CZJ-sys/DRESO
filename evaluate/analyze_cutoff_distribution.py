"""Audit learned spectral-filter cutoffs across every DRESO block."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


BLOCK_PATTERN = re.compile(
    r"(?P<branch>encoder|decoder)\.layers\.(?P<stage>\d+)\.blocks\.(?P<block>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="DRESO checkpoint directory.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output prefix; .json, .csv, and .png are written beside it.",
    )
    parser.add_argument(
        "--movement_tolerance",
        type=float,
        default=1e-4,
        help="Minimum absolute cutoff change counted as movement from initialization.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def output_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_suffix(suffix)


def block_location(name: str, fallback_index: int) -> dict[str, int | str]:
    match = BLOCK_PATTERN.search(name)
    if match is None:
        return {
            "branch": "other",
            "stage": -1,
            "block": fallback_index,
        }
    return {
        "branch": match.group("branch"),
        "stage": int(match.group("stage")),
        "block": int(match.group("block")),
    }


def collect_cutoffs(model: torch.nn.Module) -> list[dict]:
    initial_cutoff = float(getattr(model.config, "spectral_filter_cutoff", 0.25))
    rows = []
    for name, module in model.named_modules():
        if not hasattr(module, "spectral_filter_cutoff_logit"):
            continue
        raw = module.spectral_filter_cutoff_logit.detach().float().cpu()
        if raw.numel() != 1:
            raise ValueError(
                f"Expected one cutoff logit in {name}, found shape {tuple(raw.shape)}."
            )
        cutoff = float(torch.sigmoid(raw).item())
        location = block_location(name, len(rows))
        parameter = module._parameters.get("spectral_filter_cutoff_logit")
        learnable = bool(parameter is not None and parameter.requires_grad)
        rows.append(
            {
                "index": len(rows),
                "module": name,
                **location,
                "filter_type": str(
                    getattr(module.config, "spectral_filter_type", "unknown")
                ),
                "learnable": learnable,
                "raw_logit": float(raw.item()),
                "initial_cutoff": initial_cutoff,
                "learned_cutoff": cutoff,
                "delta": cutoff - initial_cutoff,
                "absolute_delta": abs(cutoff - initial_cutoff),
            }
        )
    if not rows:
        raise RuntimeError(
            "No spectral_filter_cutoff_logit tensors were found in the checkpoint."
        )
    return rows


def summarize_values(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "minimum": float(tensor.min()),
        "q25": float(torch.quantile(tensor, 0.25)),
        "median": float(torch.quantile(tensor, 0.50)),
        "q75": float(torch.quantile(tensor, 0.75)),
        "maximum": float(tensor.max()),
    }


def summarize_rows(rows: list[dict], tolerance: float) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[f"{row['branch']}.stage{row['stage']}"].append(row)

    learned = [row["learned_cutoff"] for row in rows]
    deltas = [row["delta"] for row in rows]
    absolute_deltas = [row["absolute_delta"] for row in rows]
    moved = [value > tolerance for value in absolute_deltas]
    return {
        "initial_cutoff": rows[0]["initial_cutoff"],
        "filter_types": sorted({row["filter_type"] for row in rows}),
        "num_blocks": len(rows),
        "num_learnable_blocks": sum(row["learnable"] for row in rows),
        "movement_tolerance": tolerance,
        "num_blocks_moved": sum(moved),
        "fraction_blocks_moved": sum(moved) / len(rows),
        "learned_cutoff": summarize_values(learned),
        "delta_from_initial": summarize_values(deltas),
        "absolute_delta_from_initial": summarize_values(absolute_deltas),
        "by_stage": {
            key: {
                "learned_cutoff": summarize_values(
                    [row["learned_cutoff"] for row in stage_rows]
                ),
                "delta_from_initial": summarize_values(
                    [row["delta"] for row in stage_rows]
                ),
                "num_learnable_blocks": sum(
                    row["learnable"] for row in stage_rows
                ),
                "num_blocks_moved": sum(
                    row["absolute_delta"] > tolerance for row in stage_rows
                ),
            }
            for key, stage_rows in grouped.items()
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "index",
        "module",
        "branch",
        "stage",
        "block",
        "filter_type",
        "learnable",
        "raw_logit",
        "initial_cutoff",
        "learned_cutoff",
        "delta",
        "absolute_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_distribution(path: Path, rows: list[dict], dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = [row["index"] for row in rows]
    cutoffs = [row["learned_cutoff"] for row in rows]
    deltas = [row["delta"] for row in rows]
    initial = rows[0]["initial_cutoff"]
    colors = ["#2676b8" if row["branch"] == "encoder" else "#d95f45" for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    axes[0].scatter(indices, cutoffs, c=colors, s=34, edgecolor="white", linewidth=0.4)
    axes[0].axhline(initial, color="black", linestyle="--", linewidth=1.2, label="initial")
    axes[0].set_xlabel("Spectral block index")
    axes[0].set_ylabel("Cutoff")
    axes[0].set_title("Learned cutoff by block")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    bins = min(12, max(4, int(math.sqrt(len(rows))) + 1))
    axes[1].hist(cutoffs, bins=bins, color="#4c9f70", edgecolor="white")
    axes[1].axvline(initial, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Cutoff")
    axes[1].set_ylabel("Number of blocks")
    axes[1].set_title("Cutoff distribution")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(indices, deltas, color=colors, width=0.8)
    axes[2].axhline(0.0, color="black", linewidth=1.0)
    axes[2].set_xlabel("Spectral block index")
    axes[2].set_ylabel("Learned - initial cutoff")
    axes[2].set_title("Movement from initialization")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "DRESO learnable-cutoff audit (blue: encoder, red: decoder)",
        fontsize=13,
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.movement_tolerance < 0:
        raise ValueError("--movement_tolerance must be non-negative.")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    prefix = Path(args.output).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    from model.model import ScOT

    model = ScOT.from_pretrained(str(checkpoint)).cpu().eval()
    rows = collect_cutoffs(model)
    summary = summarize_rows(rows, args.movement_tolerance)
    report = {
        "checkpoint": str(checkpoint),
        "interpretation": (
            "A nonzero delta proves that optimization changed the stored cutoff "
            "from its configured initialization; it does not by itself prove that "
            "the learned cutoff causally improves prediction quality."
        ),
        "summary": summary,
        "blocks": rows,
    }

    json_path = output_path(prefix, ".json")
    csv_path = output_path(prefix, ".csv")
    png_path = output_path(prefix, ".png")
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(csv_path, rows)
    plot_distribution(png_path, rows, args.dpi)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
