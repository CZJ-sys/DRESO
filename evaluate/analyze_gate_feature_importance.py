"""Analyze causal importance of the active DRESO gate descriptors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(ROOT))

from evaluate.evaluate_poseidon_fair import (  # noqa: E402
    POSEIDON_DATASETS,
    evaluate_dataset,
)
from model.evaluation_contract import evaluation_channels  # noqa: E402
from model.experiment import resolve_dataset_name  # noqa: E402
from model.model import (  # noqa: E402
    DualRegionEnergySpectralBlock,
    ScOT,
    ScOTConfig,
    SignalEvidenceRouter,
)
from model.problems.base import get_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure gate activation, gate sensitivity, and causal dt1/rollout "
            "importance for every DRESO signal descriptor."
        )
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--data_path")
    parser.add_argument("--datasets", nargs="+", default=list(POSEIDON_DATASETS))
    parser.add_argument("--rollout_steps", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--compute_local_sensitivity",
        action="store_true",
        help=(
            "Recompute each router after locally masking every descriptor. "
            "This adds router work inside the baseline forward pass but does "
            "not launch additional model rollouts."
        ),
    )
    parser.add_argument(
        "--causal_mode",
        choices=("none", "dt1", "rollout"),
        default="none",
        help=(
            "Prediction-level leave-one-feature-out audit. 'none' traces the "
            "baseline only, 'dt1' uses one prediction step, and 'rollout' "
            "reruns the requested full horizon for every selected feature."
        ),
    )
    parser.add_argument(
        "--causal_features",
        nargs="+",
        default=None,
        help="Optional descriptor names to causally ablate; defaults to all ten.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--save_vector",
        action="store_true",
        help="Also save editable SVG and PDF copies of the heatmap.",
    )
    parser.add_argument(
        "--heatmap_low_color",
        default="#d9f2ff",
        help="Low-value heatmap color (default: light blue).",
    )
    parser.add_argument(
        "--heatmap_high_color",
        default="#ff2d2d",
        help="High-value heatmap color (default: bright red).",
    )
    parser.add_argument(
        "--rollout_color_boundaries",
        nargs="+",
        type=float,
        default=[0.0, 5.0, 10.0, 20.0, 40.0, 80.0],
        metavar="B",
        help=(
            "Continuous piecewise-linear rollout-error color anchors. "
            "Adjacent numeric intervals receive equal visual width while "
            "colors still vary smoothly inside every interval."
        ),
    )
    parser.add_argument(
        "--replot_arrays",
        default=None,
        help=(
            "Re-render an existing gate_feature_arrays.npz without loading "
            "the checkpoint or repeating model evaluation."
        ),
    )
    return parser.parse_args()


def mean_rel_l1(result: dict, section: str, space: str) -> float:
    return float(
        result[section][space]["joint"]["mean_relative_l1_percent"]
    )


class GateTraceCollector:
    def __init__(self, model: ScOT, compute_local_sensitivity: bool = False):
        routed_modules = [
            (name, module)
            for name, module in model.named_modules()
            if (
                isinstance(module, DualRegionEnergySpectralBlock)
                and isinstance(
                    module.region_signal_router, SignalEvidenceRouter
                )
            )
        ]
        feature_name_sets = {
            tuple(module.region_signal_router.active_feature_names)
            for _, module in routed_modules
        }
        if len(feature_name_sets) != 1:
            raise ValueError(
                "Gate audit requires every signal router to use the same "
                f"active descriptors, got {sorted(feature_name_sets)}."
            )
        self.feature_names = list(next(iter(feature_name_sets)))
        self.compute_local_sensitivity = bool(compute_local_sensitivity)
        self.activation_sum = {}
        self.sensitivity_sum = {}
        self.activation_count = {}
        self.sensitivity_count = {}
        self.handles = []
        for name, module in routed_modules:
            self.handles.append(
                module.register_forward_hook(self._hook(name))
            )

    def _hook(self, name):
        def collect(module, _inputs, _output):
            stats = module.last_selection_stats
            if not stats or "region_effective_features" not in stats:
                return
            effective = stats["region_effective_features"].float()
            valid = stats["region_gate_valid"]
            weight = valid[..., None].to(effective.dtype)
            activation = (
                effective.abs() * weight
            ).sum(dim=(0, 1, 2)).double()
            count = weight.sum().double()
            if name not in self.activation_sum:
                self.activation_sum[name] = torch.zeros_like(activation)
                self.activation_count[name] = count.new_zeros(())
            self.activation_sum[name].add_(activation)
            self.activation_count[name].add_(count)

            if not self.compute_local_sensitivity:
                return

            features = stats["region_features"]
            base_gates = stats["region_gate"]
            router = module.region_signal_router
            original_mask = router.feature_mask.detach().clone()
            if name not in self.sensitivity_sum:
                self.sensitivity_sum[name] = effective.new_zeros(
                    len(self.feature_names), dtype=torch.float64
                )
                self.sensitivity_count[name] = effective.new_zeros(
                    (), dtype=torch.float64
                )
            for feature_index in range(len(self.feature_names)):
                router.feature_mask.copy_(original_mask)
                router.feature_mask[feature_index] = 0.0
                logits = router(features, valid)
                ablated_gates, _ = module._competitive_gains(
                    logits, valid, dim=2
                )
                delta = (ablated_gates - base_gates).abs()
                self.sensitivity_sum[name][feature_index].add_(
                    (delta * valid.to(delta.dtype)).sum().double()
                )
            router.feature_mask.copy_(original_mask)
            self.sensitivity_count[name].add_(valid.sum().double())

        return collect

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def arrays(self):
        names = list(self.activation_sum)
        activation = np.stack(
            [
                (
                    self.activation_sum[name]
                    / self.activation_count[name].clamp_min(1.0)
                ).cpu().numpy()
                for name in names
            ]
        )
        if self.compute_local_sensitivity:
            sensitivity = np.stack(
                [
                    (
                        self.sensitivity_sum[name]
                        / self.sensitivity_count[name].clamp_min(1.0)
                    ).cpu().numpy()
                    for name in names
                ]
            )
        else:
            sensitivity = np.full_like(activation, np.nan)
        return names, activation, sensitivity


def signal_routers(model: ScOT):
    return [
        module
        for module in model.modules()
        if isinstance(module, SignalEvidenceRouter)
    ]


def set_feature_mask(routers, feature_index: int | None, originals):
    with torch.no_grad():
        for router, original in zip(routers, originals):
            router.feature_mask.copy_(original)
            if feature_index is not None:
                router.feature_mask[feature_index] = 0.0


def make_dataset(args, dataset_name):
    return get_dataset(
        dataset=resolve_dataset_name(dataset_name),
        which="test",
        num_trajectories=1,
        data_path=args.data_path,
        max_num_time_steps=args.rollout_steps,
        time_step_size=1,
        allowed_time_transitions=[1],
    )


def evaluate(model, dataset, args, channels, device, rollout_steps):
    del channels
    result = evaluate_dataset(
        model,
        dataset,
        args.current_dataset_name,
        args.num_samples,
        rollout_steps,
        args.batch_size,
        device,
    )
    return {"dt1": result["dt1"], "rollout": result["rollout_mean"]}


def plot_heatmaps(
    output_path: Path,
    feature_names,
    block_names,
    activation,
    sensitivity,
    dataset_names,
    dt1_delta,
    rollout_delta,
    compute_local_sensitivity,
    causal_mode,
    dpi,
    save_vector,
    heatmap_low_color,
    heatmap_high_color,
    rollout_color_boundaries,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize
    except ImportError as error:
        raise ImportError(
            "Heatmap rendering requires matplotlib. Install it with "
            "`pip install matplotlib`."
        ) from error
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    boundaries = np.asarray(rollout_color_boundaries, dtype=np.float64)
    if boundaries.ndim != 1 or len(boundaries) < 2:
        raise ValueError("--rollout_color_boundaries needs at least two values.")
    if not np.all(np.isfinite(boundaries)) or not np.all(np.diff(boundaries) > 0):
        raise ValueError(
            "--rollout_color_boundaries must be finite and strictly increasing."
        )

    class PiecewiseLinearNorm(Normalize):
        """Continuous normalization with uniformly spaced numeric anchors."""

        def __init__(self, anchors, clip=False):
            self.anchors = np.asarray(anchors, dtype=np.float64)
            self.positions = np.linspace(0.0, 1.0, len(self.anchors))
            super().__init__(
                vmin=float(self.anchors[0]),
                vmax=float(self.anchors[-1]),
                clip=clip,
            )

        @staticmethod
        def _linear_extrapolate(values, source, target):
            mapped = np.interp(values, source, target)
            below = values < source[0]
            above = values > source[-1]
            if np.any(below):
                mapped[below] = target[0] + (
                    (values[below] - source[0])
                    * (target[1] - target[0])
                    / (source[1] - source[0])
                )
            if np.any(above):
                mapped[above] = target[-1] + (
                    (values[above] - source[-1])
                    * (target[-1] - target[-2])
                    / (source[-1] - source[-2])
                )
            return mapped

        def __call__(self, value, clip=None):
            values, is_scalar = self.process_value(value)
            data = np.asarray(values.data, dtype=np.float64)
            use_clip = self.clip if clip is None else clip
            if use_clip:
                data = np.clip(data, self.vmin, self.vmax)
            mapped = self._linear_extrapolate(
                data.copy(), self.anchors, self.positions
            )
            result = np.ma.array(mapped, mask=np.ma.getmask(values), copy=False)
            return result[0] if is_scalar else result

        def inverse(self, value):
            values = np.asarray(value, dtype=np.float64)
            return self._linear_extrapolate(
                values.copy(), self.positions, self.anchors
            )
    heatmap_cmap = LinearSegmentedColormap.from_list(
        "light_blue_to_bright_red",
        [
            heatmap_low_color,
            "#82c8f4",
            "#fff2ec",
            "#ff9878",
            heatmap_high_color,
        ],
        N=256,
    )
    heatmap_cmap.set_under(heatmap_low_color)
    heatmap_cmap.set_over(heatmap_high_color)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    panels = [
        (activation, block_names, "Mean |effective descriptor|"),
        (sensitivity, block_names, "Gate change after feature masking"),
        (dt1_delta, dataset_names, "Causal dt1 RelL1 increase (percentage points)"),
        (
            rollout_delta,
            dataset_names,
            "Causal rollout RelL1 increase (percentage points)",
        ),
    ]
    for panel_index, (axis, (values, row_labels, title)) in enumerate(
        zip(axes.flat, panels)
    ):
        unavailable = (
            (panel_index == 1 and not compute_local_sensitivity)
            or (panel_index == 2 and causal_mode == "none")
            or (panel_index == 3 and causal_mode != "rollout")
        )
        if unavailable:
            axis.set_axis_off()
            axis.set_title(title)
            axis.text(
                0.5,
                0.5,
                "Not requested",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=12,
            )
            continue
        colorbar_kwargs = {}
        if panel_index == 3:
            norm = PiecewiseLinearNorm(boundaries, clip=False)
            image = axis.imshow(
                values,
                aspect="auto",
                cmap=heatmap_cmap,
                norm=norm,
            )
            finite_values = np.asarray(values)[np.isfinite(values)]
            below = bool(
                finite_values.size and finite_values.min() < boundaries[0]
            )
            above = bool(
                finite_values.size and finite_values.max() > boundaries[-1]
            )
            colorbar_kwargs = {
                "ticks": boundaries,
                "extend": (
                    "both" if below and above else
                    "min" if below else
                    "max" if above else
                    "neither"
                ),
                "spacing": "uniform",
            }
        else:
            image = axis.imshow(
                values,
                aspect="auto",
                cmap=heatmap_cmap,
            )
        axis.set_title(title)
        axis.set_xticks(range(len(feature_names)))
        axis.set_xticklabels(
            feature_names,
            rotation=45,
            rotation_mode="anchor",
            ha="right",
            fontsize=8,
        )
        axis.set_yticks(range(len(row_labels)))
        axis.set_yticklabels(row_labels, fontsize=7)
        fig.colorbar(
            image,
            ax=axis,
            fraction=0.025,
            pad=0.02,
            **colorbar_kwargs,
        )
    fig.suptitle(
        "DRESO gate feature audit: activation is descriptive; error increase is causal",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=dpi)
    if save_vector:
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def replot_saved_arrays(args, output_dir: Path) -> None:
    arrays_path = Path(args.replot_arrays).expanduser().resolve()
    with np.load(arrays_path, allow_pickle=False) as arrays:
        required = {
            "feature_names",
            "block_names",
            "activation",
            "gate_sensitivity",
            "dataset_names",
            "dt1_delta",
            "rollout_delta",
        }
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(
                f"{arrays_path} is missing required arrays: {missing}."
            )
        feature_names = arrays["feature_names"].astype(str).tolist()
        block_names = arrays["block_names"].astype(str).tolist()
        activation = arrays["activation"]
        sensitivity = arrays["gate_sensitivity"]
        dataset_names = arrays["dataset_names"].astype(str).tolist()
        dt1_delta = arrays["dt1_delta"]
        rollout_delta = arrays["rollout_delta"]

    compute_local_sensitivity = bool(np.isfinite(sensitivity).any())
    if np.isfinite(rollout_delta).any():
        causal_mode = "rollout"
    elif np.isfinite(dt1_delta).any():
        causal_mode = "dt1"
    else:
        causal_mode = "none"
    plot_heatmaps(
        output_dir / "gate_feature_heatmaps.png",
        feature_names,
        block_names,
        activation,
        sensitivity,
        dataset_names,
        dt1_delta,
        rollout_delta,
        compute_local_sensitivity,
        causal_mode,
        args.dpi,
        args.save_vector,
        args.heatmap_low_color,
        args.heatmap_high_color,
        args.rollout_color_boundaries,
    )
    print(f"Replotted gate heatmap from {arrays_path} to {output_dir}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.replot_arrays is not None:
        replot_saved_arrays(args, output_dir)
        return
    missing_inputs = [
        name for name in ("checkpoint", "data_path") if getattr(args, name) is None
    ]
    if missing_inputs:
        raise ValueError(
            "Model evaluation requires "
            + ", ".join(f"--{name}" for name in missing_inputs)
            + "; alternatively use --replot_arrays."
        )
    dataset_names = args.datasets
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    checkpoint_config = ScOTConfig.from_pretrained(str(checkpoint))
    model = ScOT.from_pretrained(
        str(checkpoint), config=checkpoint_config
    ).to(device)
    model.eval()
    routers = signal_routers(model)
    if not routers:
        raise ValueError("Checkpoint contains no SignalEvidenceRouter modules.")
    originals = [router.feature_mask.detach().clone() for router in routers]

    collector = GateTraceCollector(
        model, compute_local_sensitivity=args.compute_local_sensitivity
    )
    baselines = {}
    datasets = {}
    feature_names = collector.feature_names
    selected_features = (
        feature_names if args.causal_features is None else args.causal_features
    )
    unknown_features = sorted(set(selected_features) - set(feature_names))
    if unknown_features:
        raise ValueError(
            f"Unknown --causal_features {unknown_features}; available "
            f"descriptors are {feature_names}."
        )
    if args.causal_mode == "none" and args.causal_features is not None:
        raise ValueError(
            "--causal_features requires --causal_mode dt1 or rollout."
        )
    causal_feature_count = (
        0 if args.causal_mode == "none" else len(selected_features)
    )
    evaluation_steps = 1 if args.causal_mode == "dt1" else args.rollout_steps
    total_evaluations = len(dataset_names) * (1 + causal_feature_count)
    print(
        "Gate audit plan: "
        f"causal_mode={args.causal_mode}, datasets={len(dataset_names)}, "
        f"causal_features={causal_feature_count}, "
        f"model_evaluations={total_evaluations}, "
        f"steps_per_evaluation={evaluation_steps}."
    )
    if args.causal_mode == "rollout":
        print(
            "Full causal rollout is intentionally expensive: one baseline plus "
            "one complete rollout per selected feature and dataset."
    )
    for dataset_name in dataset_names:
        args.current_dataset_name = dataset_name
        dataset = make_dataset(args, dataset_name)
        datasets[dataset_name] = dataset
        channels = evaluation_channels(
            dataset_name, model.config.num_out_channels
        )
        print(
            f"[baseline {len(baselines) + 1}/{len(dataset_names)}] "
            f"dataset={dataset_name}, steps={evaluation_steps}"
        )
        baselines[dataset_name] = evaluate(
            model, dataset, args, channels, device, evaluation_steps
        )
    collector.close()
    block_names, activation, sensitivity = collector.arrays()
    dt1_delta = np.full(
        (len(dataset_names), len(feature_names)), np.nan, dtype=np.float64
    )
    rollout_delta = np.full_like(dt1_delta, np.nan)
    masked_results = {}
    if args.causal_mode != "none":
        try:
            for selected_index, feature_name in enumerate(selected_features):
                feature_index = feature_names.index(feature_name)
                set_feature_mask(routers, feature_index, originals)
                masked_results[feature_name] = {}
                for dataset_index, dataset_name in enumerate(dataset_names):
                    args.current_dataset_name = dataset_name
                    print(
                        f"[mask feature={selected_index + 1}/"
                        f"{len(selected_features)}, dataset={dataset_index + 1}/"
                        f"{len(dataset_names)}] {feature_name} / {dataset_name}, "
                        f"steps={evaluation_steps}"
                    )
                    channels = evaluation_channels(
                        dataset_name, model.config.num_out_channels
                    )
                    result = evaluate(
                        model,
                        datasets[dataset_name],
                        args,
                        channels,
                        device,
                        evaluation_steps,
                    )
                    masked_results[feature_name][dataset_name] = result
                    dt1_delta[dataset_index, feature_index] = (
                        mean_rel_l1(result, "dt1", "native_normalized")
                        - mean_rel_l1(
                            baselines[dataset_name], "dt1", "native_normalized"
                        )
                    )
                    if args.causal_mode == "rollout":
                        rollout_delta[dataset_index, feature_index] = (
                            mean_rel_l1(
                                result, "rollout", "native_normalized"
                            )
                            - mean_rel_l1(
                                baselines[dataset_name],
                                "rollout",
                                "native_normalized",
                            )
                        )
        finally:
            set_feature_mask(routers, None, originals)

    plot_heatmaps(
        output_dir / "gate_feature_heatmaps.png",
        feature_names,
        block_names,
        activation,
        sensitivity,
        dataset_names,
        dt1_delta,
        rollout_delta,
        args.compute_local_sensitivity,
        args.causal_mode,
        args.dpi,
        args.save_vector,
        args.heatmap_low_color,
        args.heatmap_high_color,
        args.rollout_color_boundaries,
    )
    np.savez_compressed(
        output_dir / "gate_feature_arrays.npz",
        feature_names=np.asarray(feature_names),
        block_names=np.asarray(block_names),
        activation=activation,
        gate_sensitivity=sensitivity,
        dataset_names=np.asarray(dataset_names),
        dt1_delta=dt1_delta,
        rollout_delta=rollout_delta,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "protocol": {
            "first_target_time": 1,
            "rollout_steps": args.rollout_steps,
            "num_samples": args.num_samples,
            "compute_local_sensitivity": args.compute_local_sensitivity,
            "causal_mode": args.causal_mode,
            "causal_features": (
                selected_features if args.causal_mode != "none" else []
            ),
            "model_evaluations": total_evaluations,
            "metric": "native normalized joint mean relative L1 percent",
            "masking": "set one normalized router descriptor to zero in every spectral block",
            "heatmap_low_color": args.heatmap_low_color,
            "heatmap_high_color": args.heatmap_high_color,
            "rollout_color_boundaries": args.rollout_color_boundaries,
        },
        "feature_names": feature_names,
        "block_names": block_names,
        "baseline": baselines,
        "masked": masked_results,
        "dt1_increase_percentage_points": [
            [None if np.isnan(value) else float(value) for value in row]
            for row in dt1_delta
        ],
        "rollout_increase_percentage_points": [
            [None if np.isnan(value) else float(value) for value in row]
            for row in rollout_delta
        ],
    }
    with open(
        output_dir / "gate_feature_importance.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved gate audit to {output_dir}")


if __name__ == "__main__":
    main()
