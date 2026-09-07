"""Evaluate DRESO checkpoints on the selected two-dimensional Well datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running this file directly puts ``evaluate/`` on sys.path, not the repository
# root that owns the ``model`` package. Resolve it from __file__ so evaluation
# does not depend on the caller exporting PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.model import ScOT
from model.problems.the_well import (
    THE_WELL_NATIVE_RESOLUTIONS,
    THE_WELL_SUBSETS,
    _resolve_subset_path,
    _well_imports,
)


class WellRolloutDataset(Dataset):
    def __init__(
        self, root, subset, history, rollout_steps, samples,
        resolution_scale=1.0, first_target_time=1,
    ):
        WellDataset, ZScoreNormalization = _well_imports()
        self.source = WellDataset(
            path=str(_resolve_subset_path(root, subset)),
            well_split_name="test",
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            n_steps_input=first_target_time,
            n_steps_output=rollout_steps,
            max_rollout_steps=rollout_steps,
            min_dt_stride=1,
            max_dt_stride=1,
            return_grid=False,
            boundary_return_type=None,
            full_trajectory_mode=True,
        )
        self.history = int(history)
        self.first_target_time = int(first_target_time)
        if self.first_target_time < self.history:
            raise ValueError("first_target_time must be at least the checkpoint history")
        self.rollout_steps = int(rollout_steps)
        self.samples = min(int(samples), len(self.source))
        self.subset = subset
        self.native_resolution = tuple(THE_WELL_NATIVE_RESOLUTIONS[subset])
        self.resolution_scale = float(resolution_scale)
        self.resolution = tuple(
            max(1, int(round(value * self.resolution_scale)))
            for value in self.native_resolution
        )
        self.dynamic_channels = int(self.source.metadata.n_fields)
        self.constant_channels = int(self.source.metadata.n_constant_fields)

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        sample = self.source[index]
        history = torch.nan_to_num(sample["input_fields"].float()).permute(0, 3, 1, 2)
        targets = torch.nan_to_num(sample["output_fields"].float()).permute(0, 3, 1, 2)
        history = history[-self.history :]
        if history.shape[-2:] != self.resolution:
            history = F.interpolate(history, size=self.resolution, mode="bilinear", align_corners=False)
            targets = F.interpolate(targets, size=self.resolution, mode="bilinear", align_corners=False)
        constants = sample.get("constant_fields")
        if constants is not None and constants.numel() > 0:
            constants = torch.nan_to_num(constants.float()).permute(2, 0, 1).unsqueeze(0)
            constants = constants.squeeze(0)
            if constants.shape[-2:] != self.resolution:
                constants = F.interpolate(
                    constants.unsqueeze(0), size=self.resolution,
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
        else:
            constants = history.new_zeros((0, *self.resolution))
        if history.shape[-2:] != self.resolution or targets.shape[-2:] != self.resolution:
            raise ValueError(
                f"{self.subset} returned non-native shapes "
                f"{history.shape[-2:]} and {targets.shape[-2:]}; expected {self.resolution}."
            )
        return history, targets, constants


def pad_channels(value, channels, dim=1):
    dim = dim % value.ndim
    current_channels = value.shape[dim]
    if current_channels == channels:
        return value
    if current_channels > channels:
        raise ValueError(
            f"Cannot pad dimension {dim} from {current_channels} down to {channels}"
        )
    output_shape = list(value.shape)
    output_shape[dim] = channels
    output = value.new_zeros(output_shape)
    destination = [slice(None)] * value.ndim
    destination[dim] = slice(0, current_channels)
    output[tuple(destination)] = value
    return output


def relative_l1(prediction, target):
    numerator = (prediction.float() - target.float()).abs().flatten(1).sum(dim=1)
    denominator = target.float().abs().flatten(1).sum(dim=1).clamp_min(1e-8)
    return numerator / denominator


def denormalize(source, value):
    channel_last = value.permute(0, 2, 3, 1)
    physical = source.norm.denormalize_flattened(channel_last, "variable")
    return physical.permute(0, 3, 1, 2).contiguous()


def resolve_checkpoint_subsets(model, data_root, history):
    """Recover the training subset contract for old and new checkpoints."""

    declared = getattr(model.config, "training_datasets", None)
    if declared:
        return (declared,) if isinstance(declared, str) else tuple(declared)

    max_dynamic = int(model.config.num_out_channels)
    max_constant = int(model.config.num_channels) - history * max_dynamic
    exact_matches = []
    for subset in THE_WELL_SUBSETS:
        probe = WellRolloutDataset(data_root, subset, history, 1, 1)
        if (
            probe.dynamic_channels == max_dynamic
            and probe.constant_channels == max_constant
        ):
            exact_matches.append(subset)

    # Per-subset contracts are unique for the selected Well datasets. A padded
    # multi-dataset contract has no exact match and is treated as the legacy
    # all-subset pretraining contract.
    return tuple(exact_matches) if len(exact_matches) == 1 else tuple(THE_WELL_SUBSETS)


@torch.no_grad()
def evaluate_subset(model, dataset, batch_size, workers, device):
    model.eval()
    max_dynamic = int(model.config.num_out_channels)
    max_constant = int(model.config.num_channels) - max_dynamic
    if dataset.dynamic_channels > max_dynamic or dataset.constant_channels > max_constant:
        raise ValueError(
            f"Checkpoint contract supports {max_dynamic} dynamic and {max_constant} "
            f"constant channels, but the dataset needs {dataset.dynamic_channels} and "
            f"{dataset.constant_channels}."
        )
    normalized_dt1 = []
    normalized_rollout = []
    physical_dt1 = []
    physical_rollout = []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    for history, targets, constants in loader:
        history = history.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        constants = constants.to(device, non_blocking=True)
        actual_dynamic = dataset.dynamic_channels
        if history.ndim != 5 or constants.ndim != 4:
            raise ValueError(
                "Expected history [B,T,C,H,W] and constants [B,C,H,W], got "
                f"{tuple(history.shape)} and {tuple(constants.shape)}"
            )
        history = pad_channels(history, max_dynamic, dim=2)
        if constants.shape[1] < max_constant:
            constants = pad_channels(constants, max_constant, dim=1)
        predictions = []
        moving_history = history
        for step in range(dataset.rollout_steps):
            repeated_constants = constants[:, None].expand(
                -1, dataset.history, -1, -1, -1
            )
            model_input = torch.cat((moving_history, repeated_constants), dim=2)
            model_input = model_input.flatten(1, 2)
            prediction = model(
                pixel_values=model_input,
                time=1.0,
                labels=None,
            ).output
            predictions.append(prediction[:, :actual_dynamic])
            moving_history = torch.cat(
                (moving_history[:, 1:], prediction[:, None]), dim=1
            )
        predictions = torch.stack(predictions, dim=1)
        normalized_values = torch.stack(
            [
                relative_l1(predictions[:, step], targets[:, step])
                for step in range(dataset.rollout_steps)
            ],
            dim=1,
        )
        normalized_dt1.append(normalized_values[:, 0].cpu())
        normalized_rollout.append(normalized_values.flatten().cpu())
        physical_prediction = torch.stack(
            [
                denormalize(dataset.source, predictions[:, step])
                for step in range(dataset.rollout_steps)
            ],
            dim=1,
        )
        physical_target = torch.stack(
            [
                denormalize(dataset.source, targets[:, step])
                for step in range(dataset.rollout_steps)
            ],
            dim=1,
        )
        physical_values = torch.stack(
            [
                relative_l1(physical_prediction[:, step], physical_target[:, step])
                for step in range(dataset.rollout_steps)
            ],
            dim=1,
        )
        physical_dt1.append(physical_values[:, 0].cpu())
        physical_rollout.append(physical_values.flatten().cpu())

    def summarize(values):
        merged = torch.cat(values)
        return {
            "mean_relative_l1": float(merged.mean()),
            "mean_relative_l1_percent": float(100.0 * merged.mean()),
            "num_values": int(merged.numel()),
        }

    return {
        "dt1": {
            "normalized": summarize(normalized_dt1),
            "physical": summarize(physical_dt1),
        },
        "rollout_mean": {
            "normalized": summarize(normalized_rollout),
            "physical": summarize(physical_rollout),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--rollout_steps", type=int, default=19)
    parser.add_argument("--first_target_time", type=int, default=1)
    parser.add_argument("--samples_per_dataset", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resolution_scale", type=float, choices=(1.0, 0.5, 0.25), default=1.0
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.rollout_steps, args.samples_per_dataset, args.batch_size) < 1:
        raise ValueError("rollout steps, samples and batch size must be positive")
    device = torch.device(args.device)
    model = ScOT.from_pretrained(args.checkpoint).to(device)
    history = 1
    if args.first_target_time != 1:
        raise ValueError(
            "The Well protocol is single-frame t0->t1; "
            f"got first_target_time={args.first_target_time}."
        )
    trained_subsets = resolve_checkpoint_subsets(model, args.data_root, history)
    requested_subsets = tuple(args.datasets or trained_subsets)
    unknown = sorted(set(requested_subsets) - set(THE_WELL_SUBSETS))
    if unknown:
        raise ValueError(f"Unknown The Well datasets: {unknown}")
    unavailable = sorted(set(requested_subsets) - set(trained_subsets))
    if unavailable:
        raise ValueError(
            f"Checkpoint was trained for {list(trained_subsets)}, not {unavailable}."
        )
    results = {
        "protocol": {
            "history": history,
            "first_target_time": args.first_target_time,
            "rollout_steps": args.rollout_steps,
            "metric": "relative_l1",
            "normalization": "The Well field-wise z-score",
            "resolution": "scaled from each dataset's native grid",
            "resolution_scale": args.resolution_scale,
            "sampling": "full test trajectories starting at t=0",
            "static_fields_during_rollout": "fixed",
            "rollout_aggregation": "mean over every predicted step and trajectory",
            "rollout_includes_initial_state": False,
            "checkpoint_training_datasets": list(trained_subsets),
        },
        "datasets": {},
    }
    for subset in requested_subsets:
        native_resolution = tuple(THE_WELL_NATIVE_RESOLUTIONS[subset])
        evaluation_resolution = tuple(
            int(round(value * args.resolution_scale)) for value in native_resolution
        )
        dataset = WellRolloutDataset(
            args.data_root,
            subset,
            history,
            args.rollout_steps,
            args.samples_per_dataset,
            resolution_scale=args.resolution_scale,
            first_target_time=args.first_target_time,
        )
        results["datasets"][subset] = evaluate_subset(
            model, dataset, args.batch_size, args.num_workers, device
        )
        results["datasets"][subset]["native_resolution"] = list(native_resolution)
        results["datasets"][subset]["evaluation_resolution"] = list(evaluation_resolution)
        print(
            f"{subset}: dt1="
            f"{results['datasets'][subset]['dt1']['physical']['mean_relative_l1_percent']:.5f}% "
            f"rollout="
            f"{results['datasets'][subset]['rollout_mean']['physical']['mean_relative_l1_percent']:.5f}%"
        )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
