"""Shared metric utilities for Poseidon-style states."""

from __future__ import annotations

import torch


def _channel_axis(tensor: torch.Tensor) -> int:
    if tensor.ndim < 3:
        raise ValueError(f"Expected (..., C, H, W), got shape {tuple(tensor.shape)}.")
    return tensor.ndim - 3


def _channel_parameter(value, tensor: torch.Tensor, channels: int) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=tensor.dtype, device=tensor.device).flatten()
    if value.numel() == 1:
        value = value.expand(channels)
    if value.numel() < channels:
        raise ValueError(
            f"Normalization has {value.numel()} channels, but the state has {channels}."
        )
    shape = [1] * tensor.ndim
    shape[_channel_axis(tensor)] = channels
    return value[:channels].reshape(shape)


def to_physical_state(state: torch.Tensor, dataset) -> torch.Tensor:
    """Invert the affine normalization used by Poseidon dataset readers."""

    channels = state.shape[_channel_axis(state)]
    mean = _channel_parameter(dataset.constants["mean"], state, channels)
    std = _channel_parameter(dataset.constants["std"], state, channels)
    physical = state * std + mean
    if channels >= 4 and hasattr(dataset, "mean_pressure"):
        index = [slice(None)] * physical.ndim
        index[_channel_axis(physical)] = 3
        physical = physical.clone()
        physical[tuple(index)] += float(dataset.mean_pressure)
    return physical


def metric_summary(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_indices: list[int],
) -> dict:
    """Return joint-channel and per-QoI mean/median relative errors."""

    prediction = prediction[:, channel_indices].float()
    target = target[:, channel_indices].float()
    difference = prediction - target
    spatial_dims = tuple(range(2, prediction.ndim))
    joint_dims = tuple(range(1, prediction.ndim))

    def relative(p: int, dims: tuple[int, ...]) -> torch.Tensor:
        numerator = difference.abs().pow(p).sum(dim=dims)
        denominator = target.abs().pow(p).sum(dim=dims).clamp_min(1e-10)
        return (numerator / denominator).pow(1.0 / p) * 100.0

    joint_l1 = relative(1, joint_dims)
    qoi_l1 = relative(1, spatial_dims)
    return {
        "num_samples": int(prediction.shape[0]),
        "eval_channels": list(channel_indices),
        "joint": {
            "mean_relative_l1_percent": float(joint_l1.mean()),
            "median_relative_l1_percent": float(torch.quantile(joint_l1, 0.5)),
        },
        "per_qoi": {
            "mean_relative_l1_percent": [float(value) for value in qoi_l1.mean(dim=0)],
            "median_relative_l1_percent": [float(value) for value in torch.quantile(qoi_l1, 0.5, dim=0)],
            "macro_mean_relative_l1_percent": float(qoi_l1.mean(dim=0).mean()),
            "macro_median_relative_l1_percent": float(torch.quantile(qoi_l1, 0.5, dim=0).mean()),
        },
    }


def dual_space_metric_summary(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_indices: list[int],
    dataset,
) -> dict:
    return {
        "native_normalized": metric_summary(prediction, target, channel_indices),
        "physical": metric_summary(
            to_physical_state(prediction, dataset),
            to_physical_state(target, dataset),
            channel_indices,
        ),
    }
