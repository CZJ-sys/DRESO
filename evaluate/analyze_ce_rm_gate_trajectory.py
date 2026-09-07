#!/usr/bin/env python3
"""Audit DRESO's ten signal descriptors along one Poseidon trajectory.

This script analyzes the *physical-field proxy* of the router descriptors.  It
uses exactly the same diagonal-Nyquist radial normalization, rFFT
multiplicity, Gaussian low/high partition, and two-region normalization as the
DRESO signal-competitive router.  A model checkpoint is not required.

The router inside a trained model sees latent block features rather than these
primitive physical fields.  Consequently, the generated figures explain the
physical meaning and temporal behavior of the descriptors; they are not a
substitute for checkpoint-level hooks and causal feature-masking results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


FEATURE_NAMES = (
    "log_power",
    "power_share",
    "entropy",
    "flatness",
    "log_crest",
    "log_peak_to_floor",
    "signed_log_kurtosis",
    "centroid",
    "spread",
    "coherence",
)
REGION_NAMES = ("low", "high")

DATASET_SPECS = {
    "CE-RM": {
        "filename": "CE-RM.nc",
        "variable": "solution",
        "channels": ("density", "u", "v", "pressure", "passive_tracer"),
        "fallback_time_end": 2.0,
        "time_label": "Physical time",
    },
    "NS-Gauss": {
        "filename": "NS-Gauss.nc",
        "variable": "velocity",
        "channels": ("u", "v"),
        "fallback_time_end": 1.0,
        "time_label": "Normalized trajectory time",
    },
    "NS-Sines": {
        "filename": "NS-Sines.nc",
        "variable": "velocity",
        "channels": ("u", "v"),
        "fallback_time_end": 1.0,
        "time_label": "Normalized trajectory time",
    },
    "CE-RP": {
        "filename": "CE-RP.nc",
        "variable": "data",
        "channels": ("density", "u", "v", "pressure"),
        "fallback_time_end": 1.0,
        "time_label": "Normalized trajectory time",
    },
    "CE-CRP": {
        "filename": "CE-CRP.nc",
        "variable": "data",
        "channels": ("density", "u", "v", "pressure"),
        "fallback_time_end": 1.0,
        "time_label": "Normalized trajectory time",
    },
    "CE-Gauss": {
        "filename": "CE-Gauss.nc",
        "variable": "data",
        "channels": ("density", "u", "v", "pressure"),
        "fallback_time_end": 1.0,
        "time_label": "Normalized trajectory time",
    },
    "CE-KH": {
        "filename": "CE-KH.nc",
        "variable": "data",
        "channels": ("density", "u", "v", "pressure"),
        "fallback_time_end": 1.0,
        "time_label": "Normalized trajectory time",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_SPECS),
        default="CE-RM",
        help="Poseidon subset whose trajectory is analyzed.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Explicit .nc file. Otherwise --data_root/FILE is used.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Poseidon root containing the dataset .nc files.",
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--cutoff", type=float, default=0.25)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--normalized_colorbar_step",
        type=float,
        default=0.1,
        help=(
            "Tick interval for the low-region normalized-router heatmap "
            "colorbar. Use 0.1 for detailed inspection or 0.2 for a more "
            "compact publication figure."
        ),
    )
    parser.add_argument("--save_vector", action="store_true")
    return parser.parse_args()


def radial_frequency_grid(height: int, width: int) -> np.ndarray:
    """Match model.spectral_filters.radial_frequency_grid exactly."""
    fy = np.abs(np.fft.fftfreq(height)) / 0.5
    fx = np.fft.rfftfreq(width) / 0.5
    return np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / math.sqrt(2.0)


def rfft_multiplicity(width: int) -> np.ndarray:
    width_rfft = width // 2 + 1
    multiplicity = np.ones(width_rfft, dtype=np.float64)
    if width % 2 == 0:
        multiplicity[1:-1] = 2.0
    else:
        multiplicity[1:] = 2.0
    return multiplicity[None, :]


def signal_descriptors(
    spectrum: np.ndarray,
    radius: np.ndarray,
    multiplicity: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return C x 10 descriptors and validity for one spectral region."""
    channels = spectrum.shape[0]
    raw_power = np.abs(spectrum) ** 2
    weighted_power = raw_power * multiplicity[None]
    power = weighted_power.sum(axis=(-2, -1))
    count = float(np.sum(multiplicity) * spectrum.shape[-2])
    valid = power > eps
    mean_power = power / max(count, 1.0)

    probability = weighted_power / np.maximum(power[:, None, None], eps)
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, eps)), axis=(-2, -1)
    )
    if count > 1:
        entropy /= math.log(count)
    else:
        entropy.fill(0.0)

    mean_log_power = np.sum(
        np.log(np.maximum(raw_power, eps)) * multiplicity[None],
        axis=(-2, -1),
    ) / max(count, 1.0)
    flatness = np.clip(
        np.exp(mean_log_power) / np.maximum(mean_power, eps), 0.0, 1.0
    )

    peak_power = raw_power.max(axis=(-2, -1))
    crest = peak_power / np.maximum(mean_power, eps)
    background_floor = np.maximum(power - peak_power, 0.0) / max(
        count - 1.0, 1.0
    )
    peak_to_floor = peak_power / np.maximum(background_floor, eps)

    centered = raw_power - mean_power[:, None, None]
    variance = np.sum(
        centered**2 * multiplicity[None], axis=(-2, -1)
    ) / max(count, 1.0)
    fourth_moment = np.sum(
        centered**4 * multiplicity[None], axis=(-2, -1)
    ) / max(count, 1.0)
    excess_kurtosis = np.clip(
        fourth_moment / np.maximum(variance**2, eps) - 3.0, -2.0, 20.0
    )
    signed_log_kurtosis = np.sign(excess_kurtosis) * np.log1p(
        np.abs(excess_kurtosis)
    )

    centroid = np.sum(
        weighted_power * radius[None], axis=(-2, -1)
    ) / np.maximum(power, eps)
    radial_variance = np.sum(
        weighted_power * (radius[None] - centroid[:, None, None]) ** 2,
        axis=(-2, -1),
    ) / np.maximum(power, eps)
    spread = np.sqrt(np.maximum(radial_variance, 0.0) + eps) - math.sqrt(eps)

    if channels > 1:
        reference = (spectrum.sum(axis=0, keepdims=True) - spectrum) / float(
            channels - 1
        )
    else:
        reference = spectrum
    cross_power = np.sum(
        spectrum * np.conj(reference) * multiplicity[None], axis=(-2, -1)
    )
    reference_power = np.sum(
        np.abs(reference) ** 2 * multiplicity[None], axis=(-2, -1)
    )
    coherence = np.clip(
        np.abs(cross_power) ** 2
        / np.maximum(power * reference_power, eps),
        0.0,
        1.0,
    )

    features = np.stack(
        [
            np.log1p(power),
            np.zeros_like(power),  # filled jointly across low/high below
            entropy,
            flatness,
            np.log1p(np.minimum(crest, 1e4)),
            np.log1p(np.minimum(peak_to_floor, 1e4)),
            signed_log_kurtosis,
            centroid,
            spread,
            coherence,
        ],
        axis=-1,
    )
    features[~valid] = 0.0
    return features, valid


def trajectory_features(
    fields: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, _, height, width = fields.shape
    radius = radial_frequency_grid(height, width)
    multiplicity = rfft_multiplicity(width)
    mask = np.exp(-0.5 * (radius / cutoff) ** 2)
    raw = np.empty((times, fields.shape[1], 2, len(FEATURE_NAMES)))
    validity = np.empty((times, fields.shape[1], 2), dtype=bool)

    for time_index in range(times):
        spectrum = np.fft.rfft2(fields[time_index], axes=(-2, -1))
        low_features, low_valid = signal_descriptors(
            spectrum * mask[None], radius, multiplicity
        )
        high_features, high_valid = signal_descriptors(
            spectrum * (1.0 - mask)[None], radius, multiplicity
        )
        raw[time_index, :, 0] = low_features
        raw[time_index, :, 1] = high_features
        validity[time_index, :, 0] = low_valid
        validity[time_index, :, 1] = high_valid

    region_power = np.expm1(raw[..., 0])
    raw[..., 1] = region_power / np.maximum(
        region_power.sum(axis=2, keepdims=True), 1e-8
    )

    valid_float = validity[..., None].astype(np.float64)
    count = np.maximum(valid_float.sum(axis=2, keepdims=True), 1.0)
    mean = (raw * valid_float).sum(axis=2, keepdims=True) / count
    variance = (
        (raw - mean) ** 2 * valid_float
    ).sum(axis=2, keepdims=True) / count
    effective = (raw - mean) / np.sqrt(variance + 1e-5)
    effective *= valid_float
    return raw, effective, mask


def rank_correlation(values: np.ndarray) -> float:
    if np.allclose(values, values[0]):
        return 0.0
    ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    time_rank = np.arange(values.size)
    return float(np.corrcoef(time_rank, ranks)[0, 1])


def summarize(
    raw: np.ndarray,
    effective: np.ndarray,
    channel_names: tuple[str, ...],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if raw.shape[1] != len(channel_names):
        raise ValueError(
            f"Descriptor tensor has {raw.shape[1]} channels, but "
            f"{len(channel_names)} channel names were provided."
        )
    for channel_index, channel_name in enumerate(channel_names):
        for region_index, region_name in enumerate(REGION_NAMES):
            for feature_index, feature_name in enumerate(FEATURE_NAMES):
                values = raw[:, channel_index, region_index, feature_index]
                routed = effective[:, channel_index, region_index, feature_index]
                rms = float(np.sqrt(np.mean(values**2)))
                variation = float(np.std(values) / max(rms, 1e-8))
                rho = rank_correlation(values)
                if variation < 0.02:
                    behavior = "near_constant"
                elif abs(rho) >= 0.8:
                    behavior = "monotonic_increase" if rho > 0 else "monotonic_decrease"
                elif abs(rho) >= 0.5:
                    behavior = "trend_with_fluctuation"
                else:
                    behavior = "non_monotonic"
                records.append(
                    {
                        "channel": channel_name,
                        "region": region_name,
                        "feature": feature_name,
                        "start": float(values[0]),
                        "end": float(values[-1]),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "relative_variation": variation,
                        "spearman_time": rho,
                        "effective_std": float(np.std(routed)),
                        "behavior": behavior,
                    }
                )
    return records


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, path: Path, args: argparse.Namespace) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    if args.save_vector:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_physical_evolution(
    fields: np.ndarray, times: np.ndarray, output: Path, args: argparse.Namespace
) -> None:
    indices = np.unique(np.linspace(0, len(times) - 1, 5, dtype=int))
    channel_names = tuple(args.channel_names)
    rows = []
    for name, label in (
        ("density", "Density"),
        ("u", "Velocity u"),
        ("v", "Velocity v"),
        ("pressure", "Pressure"),
        ("passive_tracer", "Passive tracer"),
    ):
        if name in channel_names:
            rows.append((fields[:, channel_names.index(name)], label))
    if "u" in channel_names and "v" in channel_names:
        velocity = np.sqrt(
            fields[:, channel_names.index("u")] ** 2
            + fields[:, channel_names.index("v")] ** 2
        )
        velocity_position = 1 if "density" in channel_names else len(rows)
        rows.insert(velocity_position, (velocity, "Velocity magnitude"))
    fig, axes = plt.subplots(len(rows), len(indices), figsize=(10.0, 7.2))
    for row_index, (quantity, label) in enumerate(rows):
        vmin, vmax = np.nanpercentile(quantity, [1, 99])
        for column_index, time_index in enumerate(indices):
            ax = axes[row_index, column_index]
            ax.imshow(
                quantity[time_index], origin="lower", cmap="viridis",
                vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            if row_index == 0:
                ax.set_title(f"t={times[time_index]:.2f}")
            if column_index == 0:
                ax.set_ylabel(label)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{args.dataset} trajectory used for the gate-descriptor audit")
    fig.tight_layout()
    save_figure(fig, output / f"{args.output_prefix}_physical_trajectory", args)


def temporal_zscore(array: np.ndarray) -> np.ndarray:
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    return (array - mean) / np.maximum(std, 1e-8)


def plot_descriptor_heatmaps(
    raw: np.ndarray,
    effective: np.ndarray,
    times: np.ndarray,
    output: Path,
    args: argparse.Namespace,
) -> None:
    # Average only for visualization. JSON/NPZ retain every physical channel.
    raw_mean = raw.mean(axis=1)
    effective_mean = effective.mean(axis=1)
    raw_z = temporal_zscore(raw_mean).transpose(1, 2, 0).reshape(20, len(times))
    # With two valid regions, descriptor-wise standardization makes the high
    # row the sign-reversed counterpart of the low row. Keep only low in the
    # figure; the full low/high tensors remain in the NPZ and JSON summaries.
    routed_low = effective_mean[:, 0, :].T
    raw_labels = [
        f"{region}:{feature}"
        for region in REGION_NAMES
        for feature in FEATURE_NAMES
    ]
    routed_labels = list(FEATURE_NAMES)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 7.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.25)},
    )
    images = [
        axes[0].imshow(raw_z, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5),
        axes[1].imshow(
            routed_low,
            aspect="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        ),
    ]
    axes[0].set_title("Raw descriptor temporal z-score")
    axes[1].set_title("Actual normalized router input (low region)")
    axes[0].set_yticks(np.arange(len(raw_labels)), raw_labels)
    axes[1].set_yticks(np.arange(len(routed_labels)), routed_labels)
    for ax in axes:
        ax.set_xticks(np.arange(len(times)), [f"{value:.1f}" for value in times])
        ax.set_xlabel(args.time_axis_label)
    fig.colorbar(images[0], ax=axes[0], shrink=0.75)
    normalized_ticks = np.arange(
        -1.0,
        1.0 + 0.5 * args.normalized_colorbar_step,
        args.normalized_colorbar_step,
    )
    normalized_ticks = normalized_ticks[
        (normalized_ticks >= -1.0 - 1e-9)
        & (normalized_ticks <= 1.0 + 1e-9)
    ]
    if not np.isclose(normalized_ticks[-1], 1.0):
        normalized_ticks = np.append(normalized_ticks, 1.0)
    normalized_ticks[np.isclose(normalized_ticks, 0.0)] = 0.0
    normalized_colorbar = fig.colorbar(
        images[1],
        ax=axes[1],
        shrink=0.82,
        ticks=normalized_ticks,
    )
    normalized_colorbar.ax.set_yticklabels(
        [f"{value:.1f}" for value in normalized_ticks]
    )
    fig.suptitle(
        f"{args.dataset} evolution of raw low/high descriptors and low-region router input"
    )
    save_figure(fig, output / f"{args.output_prefix}_gate_descriptor_heatmaps", args)


def plot_feature_curves(
    raw: np.ndarray, times: np.ndarray, output: Path, args: argparse.Namespace
) -> None:
    mean = raw.mean(axis=1)
    fig, axes = plt.subplots(5, 2, figsize=(10.0, 12.0), sharex=True)
    for feature_index, ax in enumerate(axes.flat):
        ax.plot(times, mean[:, 0, feature_index], label="low", color="#2171b5")
        ax.plot(times, mean[:, 1, feature_index], label="high", color="#cb181d")
        ax.set_title(FEATURE_NAMES[feature_index])
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel(args.time_axis_label)
    fig.suptitle(
        f"Physical-channel mean descriptors along one {args.dataset} trajectory"
    )
    fig.tight_layout()
    save_figure(fig, output / f"{args.output_prefix}_gate_descriptor_curves", args)


def resolve_data_path(args: argparse.Namespace) -> Path:
    if args.data is not None:
        path = args.data.expanduser().resolve()
    elif args.data_root is not None:
        path = (
            args.data_root.expanduser().resolve()
            / str(DATASET_SPECS[args.dataset]["filename"])
        )
    else:
        raise ValueError("Provide either --data or --data_root.")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_time_axis(
    dataset: Dataset, length: int, fallback_end: float
) -> tuple[np.ndarray, bool]:
    for name, variable in dataset.variables.items():
        normalized_name = name.lower().replace("-", "_")
        if "time" not in normalized_name and normalized_name not in {"t", "times"}:
            continue
        if variable.ndim == 1 and variable.shape[0] == length:
            values = np.asarray(variable[:], dtype=np.float64)
            if np.isfinite(values).all():
                return values, True
    return np.linspace(0.0, fallback_end, length), False


def main() -> None:
    args = parse_args()
    if not 0.0 < args.cutoff <= 1.0:
        raise ValueError("--cutoff must be in (0, 1]")
    if not 0.0 < args.normalized_colorbar_step <= 1.0:
        raise ValueError("--normalized_colorbar_step must be in (0, 1]")
    spec = DATASET_SPECS[args.dataset]
    data_path = resolve_data_path(args)
    args.channel_names = tuple(spec["channels"])
    args.output_prefix = args.dataset.lower().replace("-", "_")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Dataset(data_path, "r") as dataset:
        variable_name = str(spec["variable"])
        if variable_name not in dataset.variables:
            raise KeyError(
                f"{data_path} has no variable {variable_name!r}; available: "
                f"{tuple(dataset.variables)}"
            )
        variable = dataset.variables[variable_name]
        if not 0 <= args.sample_index < variable.shape[0]:
            raise IndexError(
                f"sample {args.sample_index} outside [0, {variable.shape[0]})"
            )
        fields = np.asarray(variable[args.sample_index], dtype=np.float64)
        times, has_file_time = read_time_axis(
            dataset, fields.shape[0], float(spec["fallback_time_end"])
        )
    expected_channels = len(args.channel_names)
    if fields.ndim != 4 or fields.shape[1] < expected_channels:
        raise ValueError(
            f"Expected [time, channel, height, width] with at least "
            f"{expected_channels} channels for {args.dataset}, found {fields.shape}."
        )
    fields = fields[:, :expected_channels]

    args.time_axis_label = (
        "Physical time" if has_file_time else str(spec["time_label"])
    )
    raw, effective, mask = trajectory_features(fields, args.cutoff)
    records = summarize(raw, effective, args.channel_names)
    summary = {
        "dataset": args.dataset,
        "source": str(data_path),
        "sample_index": args.sample_index,
        "cutoff": args.cutoff,
        "radial_normalization": "diagonal Nyquist = 1",
        "channels": args.channel_names,
        "features": FEATURE_NAMES,
        "regions": REGION_NAMES,
        "normalized_heatmap": {
            "displayed_region": "low",
            "colorbar_step": args.normalized_colorbar_step,
            "high_region_omitted_from_figure": (
                "For two-region descriptor-wise normalization, the valid "
                "high-region value is the sign-reversed low-region value. "
                "Both regions remain available in records and NPZ arrays."
            ),
        },
        "warning": (
            "Physical-field proxy only. A trained DRESO router consumes "
            f"latent block spectra, not primitive {args.dataset} channels."
        ),
        "records": records,
    }
    with (args.output_dir / f"{args.output_prefix}_gate_trajectory.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(
        args.output_dir / f"{args.output_prefix}_gate_trajectory_arrays.npz",
        fields=fields,
        times=times,
        raw_features=raw,
        effective_features=effective,
        gaussian_mask=mask,
        channel_names=np.asarray(args.channel_names),
        feature_names=np.asarray(FEATURE_NAMES),
        region_names=np.asarray(REGION_NAMES),
    )

    configure_style()
    plot_physical_evolution(fields, times, args.output_dir, args)
    plot_descriptor_heatmaps(raw, effective, times, args.output_dir, args)
    plot_feature_curves(raw, times, args.output_dir, args)

    counts: dict[str, int] = {}
    for record in records:
        behavior = str(record["behavior"])
        counts[behavior] = counts.get(behavior, 0) + 1
    print(json.dumps({"output_dir": str(args.output_dir), "behavior_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
