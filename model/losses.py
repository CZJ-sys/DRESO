"""Loss helpers shared by the DRESO training objectives."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def expand_ignore_mask(
    ignore_mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Broadcast a channel or spatial ignore mask to ``reference``."""
    if ignore_mask is None:
        return torch.zeros_like(reference, dtype=torch.bool)

    mask = ignore_mask.to(device=reference.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    while mask.ndim < reference.ndim:
        mask = mask.unsqueeze(-1)
    try:
        return torch.broadcast_to(mask, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Cannot broadcast ignore mask {tuple(ignore_mask.shape)} to "
            f"prediction shape {tuple(reference.shape)}."
        ) from exc


def normalized_channel_group_loss(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    channel_slices: Sequence[int] | None,
    p: int,
    ignore_mask: torch.Tensor | None = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute the original normalized group loss over supervised values only."""
    if p not in {1, 2}:
        raise ValueError("p must be 1 or 2")

    supervised = ~expand_ignore_mask(ignore_mask, prediction)
    pointwise_error = (prediction - labels).abs().pow(p)

    if channel_slices is None:
        count = supervised.sum()
        if not bool(count):
            return pointwise_error.sum() * 0.0
        return pointwise_error.masked_select(supervised).mean()

    group_losses = []
    for start, end in zip(channel_slices[:-1], channel_slices[1:]):
        group_supervised = supervised[:, start:end]
        count = group_supervised.sum()
        if not bool(count):
            continue

        error_mean = pointwise_error[:, start:end].masked_select(
            group_supervised
        ).mean()
        target_mean = labels[:, start:end].abs().pow(p).masked_select(
            group_supervised
        ).mean()
        group_losses.append(error_mean / (target_mean + eps))

    if not group_losses:
        return pointwise_error.sum() * 0.0
    return torch.stack(group_losses).mean()


def relative_l2_channel_group_loss(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    channel_slices: Sequence[int] | None,
    ignore_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Average masked relative L2 over samples and physical channel groups."""

    supervised = ~expand_ignore_mask(ignore_mask, prediction)
    if not bool(supervised.any()):
        raise ValueError(
            "relative-L2 received no supervised values; check pixel_mask "
            "semantics and dataset output channels."
        )
    error = prediction - labels
    slices = (
        [(0, prediction.shape[1])]
        if channel_slices is None
        else list(zip(channel_slices[:-1], channel_slices[1:]))
    )

    relative_errors = []
    for start, end in slices:
        group_supervised = supervised[:, start:end]
        valid_samples = group_supervised.flatten(1).any(dim=1)
        if not bool(valid_samples.any()):
            continue

        mask = group_supervised.to(prediction.dtype)
        numerator = (error[:, start:end] * mask).flatten(1).norm(dim=1)
        denominator = (labels[:, start:end] * mask).flatten(1).norm(dim=1)
        relative = numerator / denominator.clamp_min(eps)
        relative_errors.append(relative[valid_samples])

    if not relative_errors:
        raise ValueError(
            "relative-L2 found no supervised channel group; check "
            f"channel_slices={channel_slices} for {prediction.shape[1]} outputs."
        )
    return torch.cat(relative_errors).mean()


def high_frequency_error_loss(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    fft_norm: str,
    ignore_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the weighted high-frequency error on supervised channels."""
    supervised = ~expand_ignore_mask(ignore_mask, prediction)
    error = (prediction - labels) * supervised.to(prediction.dtype)
    _, _, height, width = error.shape
    error_fft = torch.fft.rfft2(error.contiguous(), norm=fft_norm)
    spectrum = error_fft.real.square() + error_fft.imag.square()

    fy = torch.fft.fftfreq(height, d=1.0, device=error.device).abs()
    fx = torch.fft.rfftfreq(width, d=1.0, device=error.device).abs()
    ry = fy / (fy.max() + 1e-8)
    rx = fx / (fx.max() + 1e-8)
    radius = torch.sqrt(ry[:, None].square() + rx[None, :].square())
    weights = (radius / (radius.max() + 1e-8)).pow(alpha)

    # The Poseidon masks are channel-wise. The spatial mean also gives a
    # sensible effective count for datasets that mask fixed spatial regions.
    supervised_fraction = supervised.to(spectrum.dtype).mean(dim=(-2, -1))
    effective_channels = supervised_fraction.sum()
    if not bool(effective_channels):
        return spectrum.sum() * 0.0
    return (spectrum * weights).sum() / (
        effective_channels * spectrum.shape[-2] * spectrum.shape[-1]
    )
