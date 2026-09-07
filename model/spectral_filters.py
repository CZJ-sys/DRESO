"""Frequency-grid and Gaussian partition utilities used by DRESO."""

from __future__ import annotations

import math

import torch


def radial_frequency_grid(height, width_rfft, *, device, dtype):
    """Return the rFFT radial grid normalized by the diagonal Nyquist radius."""

    spatial_width = max(2 * (width_rfft - 1), 1)
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype).abs() / 0.5
    fx = torch.fft.rfftfreq(spatial_width, device=device, dtype=dtype) / 0.5
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    return radius / math.sqrt(2.0)


def gaussian_low_pass_response(radius, cutoff):
    """Build DRESO's differentiable zero-phase Gaussian low-pass response."""

    cutoff = torch.as_tensor(
        cutoff, dtype=radius.dtype, device=radius.device
    ).clamp(1e-4, 1.0 - 1e-4)
    return torch.exp(-0.5 * (radius / cutoff).square())
