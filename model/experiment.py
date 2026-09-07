"""Shared experiment helpers for Poseidon-style training."""

from __future__ import annotations

import bisect

from torch.utils.data import ConcatDataset, Dataset


DATASET_ALIASES = {
    "NS-Gauss": "fluids.incompressible.Gaussians",
    "NS-Sines": "fluids.incompressible.Sines",
    "CE-RP": "fluids.compressible.Riemann",
    "CE-CRP": "fluids.compressible.RiemannCurved",
    "CE-Gauss": "fluids.compressible.Gaussians",
    "CE-KH": "fluids.compressible.KelvinHelmholtz",
}

def resolve_dataset_name(dataset):
    """Expand known short aliases and leave every other dataset name unchanged."""

    if isinstance(dataset, dict) and "value" in dataset:
        dataset = dataset["value"]

    if isinstance(dataset, (list, tuple)):
        return [resolve_dataset_name(item) for item in dataset]

    if isinstance(dataset, str):
        return DATASET_ALIASES.get(dataset, dataset)
    return dataset


def normalize_dataset_config(config: dict) -> dict:
    config = dict(config)
    if "dataset" in config:
        config["dataset"] = resolve_dataset_name(config["dataset"])
    return config


def build_train_eval_set_kwargs(config: dict, params) -> dict:
    kwargs = (
        {"just_velocities": True}
        if ("incompressible" in str(config["dataset"])) and getattr(params, "just_velocities", False)
        else {}
    )
    if getattr(params, "move_data", None) is not None:
        kwargs["move_to_local_scratch"] = params.move_data

    max_steps = getattr(params, "max_num_train_time_steps", None)
    if max_steps is None:
        max_steps = config.get("max_num_train_time_steps")
    if max_steps is not None:
        kwargs["max_num_time_steps"] = max_steps

    step_size = getattr(params, "train_time_step_size", None)
    if step_size is None:
        step_size = config.get("train_time_step_size")
    if step_size is not None:
        kwargs["time_step_size"] = step_size

    if getattr(params, "train_small_time_transition", False) or config.get("train_small_time_transition", False):
        kwargs["allowed_time_transitions"] = [1]
    return kwargs


class NonZeroTimeDataset(Dataset):
    """Filter out samples whose endpoint lead time is zero."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = [
            idx
            for idx in range(len(dataset))
            if self._lead_time(idx) > 0
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def _lead_time(self, idx: int) -> float:
        dataset, local_idx = self._resolve_dataset_index(idx)
        if hasattr(dataset, "fixed_lead_time"):
            return float(dataset.fixed_lead_time)
        if not hasattr(dataset, "_idx_map"):
            item = dataset[local_idx]
            return float(item.get("time", 1.0))
        _trajectory_index, lead_time, _t_start, _t_end = dataset._idx_map(local_idx)
        return float(lead_time)

    def _resolve_dataset_index(self, idx: int):
        if not isinstance(self.dataset, ConcatDataset):
            return self.dataset, idx
        dataset_idx = bisect.bisect_right(self.dataset.cumulative_sizes, idx)
        previous = 0 if dataset_idx == 0 else self.dataset.cumulative_sizes[dataset_idx - 1]
        return self.dataset.datasets[dataset_idx], idx - previous


def drop_zero_time_pairs(dataset):
    """Return a dataset view without t1 == t2 samples."""

    return NonZeroTimeDataset(dataset)


def unwrap_filtered_dataset(dataset):
    """Return the underlying dataset used for metadata inspection."""

    return dataset.dataset if isinstance(dataset, NonZeroTimeDataset) else dataset
