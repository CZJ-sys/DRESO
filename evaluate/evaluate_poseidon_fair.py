"""Evaluate a DRESO checkpoint on the six Poseidon pretraining datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.evaluation_contract import evaluation_channels  # noqa: E402
from model.evaluation_protocol import dual_space_metric_summary  # noqa: E402
from model.experiment import resolve_dataset_name  # noqa: E402
from model.model import ScOT, ScOTConfig  # noqa: E402
from model.problems.base import get_dataset  # noqa: E402


POSEIDON_DATASETS = (
    "NS-Gauss",
    "CE-RP",
    "CE-CRP",
    "CE-Gauss",
    "NS-Sines",
    "CE-KH",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DRESO single-frame Poseidon dt1 and rollout evaluation."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(POSEIDON_DATASETS))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--rollout_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def preserve_ns_constants(prediction: torch.Tensor, state: torch.Tensor, dataset):
    """Keep the synthetic density and pressure channels fixed during rollout."""

    if (
        getattr(dataset, "just_velocities", False)
        or not hasattr(dataset, "density")
        or prediction.shape[1] < 4
    ):
        return prediction
    prediction = prediction.clone()
    prediction[:, 0] = state[:, 0]
    prediction[:, 3] = state[:, 3]
    return prediction


def load_state_batch(dataset, indices: list[int], time_index: int, device):
    return torch.stack(
        [dataset._state_at(index, time_index) for index in indices], dim=0
    ).to(device, non_blocking=True)


@torch.no_grad()
def evaluate_dataset(
    model: ScOT,
    dataset,
    dataset_name: str,
    samples: int,
    rollout_steps: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()
    count = dataset.N_test if samples == -1 else min(samples, dataset.N_test)
    if count < 1:
        raise ValueError("--samples must be positive or -1 for the full test split.")

    channels = evaluation_channels(dataset_name, model.config.num_out_channels)
    time_constant = float(dataset.constants["time"])
    predictions, targets = [], []

    for start in tqdm(range(0, count, batch_size), desc=dataset_name):
        local_indices = range(start, min(start + batch_size, count))
        trajectory_indices = [dataset.start + index for index in local_indices]
        state = load_state_batch(dataset, trajectory_indices, 0, device)
        batch_predictions, batch_targets = [], []

        for time_index in range(rollout_steps):
            dt = torch.full(
                (state.shape[0],),
                1.0 / time_constant,
                dtype=state.dtype,
                device=device,
            )
            prediction = model(pixel_values=state, time=dt, labels=None).output
            prediction = preserve_ns_constants(prediction, state, dataset)
            target = load_state_batch(
                dataset, trajectory_indices, time_index + 1, device
            )
            batch_predictions.append(prediction.cpu())
            batch_targets.append(target.cpu())
            state = prediction

        predictions.append(torch.stack(batch_predictions, dim=1))
        targets.append(torch.stack(batch_targets, dim=1))

    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    return {
        "dt1": dual_space_metric_summary(
            prediction[:, 0], target[:, 0], channels, dataset
        ),
        "rollout_mean": dual_space_metric_summary(
            prediction.flatten(0, 1), target.flatten(0, 1), channels, dataset
        ),
    }


def compact_metric(metric: dict) -> dict:
    return {
        space: {
            "mean_relative_l1_percent": values["joint"][
                "mean_relative_l1_percent"
            ],
            "num_values": values["num_samples"],
            "eval_channels": values["eval_channels"],
        }
        for space, values in metric.items()
    }


def main() -> None:
    args = parse_args()
    if args.rollout_steps < 1 or args.batch_size < 1:
        raise ValueError("rollout_steps and batch_size must be positive.")
    unknown = sorted(set(args.datasets) - set(POSEIDON_DATASETS))
    if unknown:
        raise ValueError(
            f"Clean evaluation supports the six pretraining datasets only: {unknown}"
        )

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable.")
    config = ScOTConfig.from_pretrained(checkpoint)
    model = ScOT.from_pretrained(checkpoint, config=config).to(device)

    results = {
        "checkpoint": str(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "protocol": {
            "input_time": 0,
            "first_target_time": 1,
            "rollout_steps": args.rollout_steps,
            "history": 1,
            "teacher_forcing": False,
            "metric": "mean_relative_l1_percent",
            "metric_spaces": ["native_normalized", "physical"],
            "rollout_aggregation": "mean over every trajectory and predicted step",
            "ns_eval_channels": [1, 2],
            "ns_constant_channels": [0, 3],
        },
        "datasets": {},
    }

    for dataset_name in args.datasets:
        dataset = get_dataset(
            dataset=resolve_dataset_name(dataset_name),
            which="test",
            num_trajectories=1,
            data_path=args.data_path,
            max_num_time_steps=args.rollout_steps,
            time_step_size=1,
            allowed_time_transitions=[1],
        )
        metrics = evaluate_dataset(
            model,
            dataset,
            dataset_name,
            args.samples,
            args.rollout_steps,
            args.batch_size,
            device,
        )
        results["datasets"][dataset_name] = {
            "dt1": compact_metric(metrics["dt1"]),
            "rollout_mean": compact_metric(metrics["rollout_mean"]),
        }
        normalized = results["datasets"][dataset_name]
        print(
            f"{dataset_name}: normalized dt1="
            f"{normalized['dt1']['native_normalized']['mean_relative_l1_percent']:.5f}% "
            "rollout="
            f"{normalized['rollout_mean']['native_normalized']['mean_relative_l1_percent']:.5f}%"
        )

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint / "eval_poseidon.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
