"""
DRESO: Evidence-Guided Dual-Region Spectral Operators for long-horizon PDE forecasting.

The encoder-decoder scaffold is derived from ScOT/Poseidon. Every operator
block uses DRESO's learnable Gaussian split, dual-region channel transforms,
ten-dimensional signal-evidence router, and full-spectrum identity path.

A lot of this file is taken from the transformers library and changed to our purposes. Huggingface Transformers is licensed under
Apache 2.0 License, see trainer.py for details.

We follow https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/configuration_swinv2.py
and https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/modeling_swinv2.py#L1129

The class ConvNeXtBlock is taken from the facebookresearch/ConvNeXt repository and is licensed under the MIT License,

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from transformers import (
    Swinv2PreTrainedModel,
    PretrainedConfig,
)
from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2EncoderOutput,
    Swinv2DropPath,
    Swinv2Intermediate,
    Swinv2Output,
)
from transformers.utils import ModelOutput
from dataclasses import dataclass
import torch
from torch import nn
from typing import Optional, Union, Tuple, List
import math
import collections
import torch, torch.nn as nn
from torch import Tensor


from model.losses import (
    expand_ignore_mask,
    high_frequency_error_loss,
    normalized_channel_group_loss,
)

try:
    from .spectral_filters import (
        gaussian_low_pass_response,
        radial_frequency_grid,
    )
except ImportError:
    from spectral_filters import (
        gaussian_low_pass_response,
        radial_frequency_grid,
    )


class SignalEvidenceRouter(nn.Module):
    """Score one channel/region using signal descriptors plus a learned residual."""

    feature_names = (
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
    feature_groups = {
        "energy": (0, 1),
        "distribution": (2, 3, 4, 5, 6),
        "geometry": (7, 8),
        "coherence": (9,),
    }

    def __init__(
        self,
        hidden_dim,
        prior_weight=1.0,
    ):
        super().__init__()
        feature_dim = len(self.feature_names)
        self.residual_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Start from a deterministic DSP prior. The MLP initially contributes
        # no random preference and learns only the task-specific correction.
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)
        full_prior = torch.tensor(
            [
                0.00,   # absolute stage power is handled by normalization
                0.35,   # occupied energy share
                -0.50,  # low entropy indicates concentrated structure
                -0.50,  # low flatness indicates signal-like spectra
                0.20,   # peak-to-average ratio
                0.25,   # CA-CFAR-like peak-to-floor evidence
                0.15,   # impulsive/non-Gaussian spectral structure
                0.00,   # centroid is context, not an intrinsic preference
                0.00,   # spread is context, not an intrinsic preference
                0.40,   # cross-channel phase-consistent structure
            ],
            dtype=torch.float32,
        )
        self.evidence_weights = nn.Parameter(full_prior)
        self.prior_weight = float(prior_weight)
        # Kept as a buffer so the feature-audit visualizer can causally mask
        # one descriptor at inference time. Training always uses all ten.
        self.register_buffer(
            "feature_mask", torch.ones(feature_dim, dtype=torch.float32)
        )
        self.last_effective_features = None
        self.last_learned_score = None
        self.last_prior_contributions = None

    def forward(self, features, valid=None):
        if features.shape[-1] != len(self.feature_names):
            raise ValueError(
                "SignalEvidenceRouter expects the complete 10D descriptor "
                f"tensor, got {features.shape[-1]}."
            )
        if valid is None:
            valid = torch.ones_like(features[..., 0], dtype=torch.bool)
        valid_float = valid[..., None].to(features.dtype)
        count = valid_float.sum(dim=-2, keepdim=True).clamp_min(1.0)
        mean = (features * valid_float).sum(dim=-2, keepdim=True) / count
        variance = (
            (features - mean).square() * valid_float
        ).sum(dim=-2, keepdim=True) / count
        features = (features - mean) / torch.sqrt(variance + 1e-5)
        features = (
            features
            * valid_float
            * self.feature_mask.to(features.dtype)
        )
        learned_score = self.residual_mlp(features).squeeze(-1)
        self.last_effective_features = features.detach()
        self.last_learned_score = learned_score.detach()
        self.last_prior_contributions = (
            features
            * self.evidence_weights.to(features.dtype)
            * self.prior_weight
        ).detach()
        evidence_score = torch.einsum(
            "...f,f->...", features, self.evidence_weights
        )
        return learned_score + self.prior_weight * evidence_score


class DualRegionEnergySpectralBlock(nn.Module):
    """
    Complementary low/high, evidence-guided spectral residual operator.

    A learnable Gaussian response softly partitions all rFFT modes into low and
    high regions. Separate complex channel mixers transform both regions, the
    ten-dimensional signal router allocates channel-wise gains, and the full
    input spectrum remains on an identity path while transformed terms enter as
    residual corrections.
    """

    def __init__(self, config, dim, patch_grid, drop_path=0.0):
        super().__init__()
        self.config = config
        self.FFT_norm = config.fft_norm
        self.dim = dim
        self.patch_grid = tuple(int(value) for value in patch_grid)

        init_scale = config.spectral_mixer_init_scale

        def initialize_mixer():
            return (
                torch.eye(dim) + torch.randn(dim, dim) * init_scale,
                torch.randn(dim, dim) * init_scale,
            )

        low_real, low_imag = initialize_mixer()
        high_real, high_imag = initialize_mixer()
        self.low_weight_real = nn.Parameter(low_real)
        self.low_weight_imag = nn.Parameter(low_imag)
        self.high_weight_real = nn.Parameter(high_real)
        self.high_weight_imag = nn.Parameter(high_imag)

        self.gate_temperature = float(
            getattr(config, "spectral_gate_temperature", 1.0)
        )
        self.gate_floor = float(
            getattr(config, "spectral_gate_floor", 0.25)
        )
        self.gate_prior_weight = float(
            getattr(config, "spectral_gate_prior_weight", 1.0)
        )
        if self.gate_temperature <= 0.0:
            raise ValueError("spectral_gate_temperature must be positive.")
        if not 0.0 <= self.gate_floor < 1.0:
            raise ValueError("spectral_gate_floor must be in [0, 1).")
        if self.gate_prior_weight < 0.0:
            raise ValueError("spectral_gate_prior_weight must be non-negative.")
        self.region_signal_router = SignalEvidenceRouter(
            config.spectral_region_gate_hidden,
            prior_weight=self.gate_prior_weight,
        )

        self.conv_local = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=False
        )
        self.linear = nn.Linear(dim, dim)

        self.norm = build_operator_norm(config, dim)
        self.drop_path = (
            Swinv2DropPath(drop_path) if drop_path > 0 else nn.Identity()
        )

        cutoff = min(max(float(config.spectral_filter_cutoff), 1e-4), 1.0 - 1e-4)
        cutoff_logit = math.log(cutoff / (1.0 - cutoff))
        self.spectral_filter_cutoff_logit = nn.Parameter(
            torch.tensor(cutoff_logit, dtype=torch.float32)
        )

        self.last_selection_stats = None

    @staticmethod
    def _complex_residual(spectrum, weight_real, weight_imag):
        batch_size, channels, height, width = spectrum.shape
        weight = torch.complex(weight_real, weight_imag)
        flat = spectrum.reshape(batch_size, channels, -1)
        mapped = (weight @ flat).view(
            batch_size, channels, height, width
        )
        return mapped - spectrum

    @staticmethod
    def _signal_descriptors(spectrum, band_masks, radius, eps=1e-8):
        """Return channel-aware DSP descriptors for every masked band.

        Output shapes are B x C x S for statistics and B x C x S x F for
        features. All shape descriptors are dimensionless or normalized to the
        rFFT Nyquist radius so the same router can be used across stages.
        """

        batch_size, channels, height, width = spectrum.shape
        num_bands = band_masks.shape[0]
        raw_power = spectrum.abs().square()
        multiplicity = raw_power.new_ones(1, 1, height, width)
        if width > 2:
            multiplicity[..., 1:-1] = 2.0
        masks = band_masks.to(raw_power.dtype)
        weighted_masks = masks[None, None] * multiplicity[:, :, None]
        expanded_power = raw_power[:, :, None]
        masked_power = expanded_power * weighted_masks
        band_power = masked_power.sum(dim=(-2, -1))
        counts = weighted_masks.sum(dim=(-2, -1)).expand(
            batch_size, channels, num_bands
        )
        valid = (band_power > eps) & (counts > 0)
        mean_power = band_power / counts.clamp_min(1.0)

        probabilities = masked_power / band_power[..., None, None].clamp_min(eps)
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(eps))
        ).sum(dim=(-2, -1))
        entropy = torch.where(
            counts > 1,
            entropy / torch.log(counts.clamp_min(2.0)),
            torch.zeros_like(entropy),
        )

        mean_log_power = (
            torch.log(expanded_power.clamp_min(eps)) * weighted_masks
        ).sum(dim=(-2, -1)) / counts.clamp_min(1.0)
        flatness = (
            torch.exp(mean_log_power) / mean_power.clamp_min(eps)
        ).clamp(0.0, 1.0)

        inside = masks.bool()[None, None]
        peak_power = torch.where(
            inside,
            expanded_power,
            torch.zeros_like(expanded_power),
        ).amax(dim=(-2, -1))
        crest = peak_power / mean_power.clamp_min(eps)
        background_floor = (
            (band_power - peak_power).clamp_min(0.0)
            / (counts - 1.0).clamp_min(1.0)
        )
        peak_to_floor = peak_power / background_floor.clamp_min(eps)

        centered = expanded_power - mean_power[..., None, None]
        variance = (
            centered.square() * weighted_masks
        ).sum(dim=(-2, -1)) / counts.clamp_min(1.0)
        fourth_moment = (
            centered.pow(4) * weighted_masks
        ).sum(dim=(-2, -1)) / counts.clamp_min(1.0)
        excess_kurtosis = (
            fourth_moment / variance.square().clamp_min(eps) - 3.0
        ).clamp(-2.0, 20.0)
        signed_log_kurtosis = torch.sign(excess_kurtosis) * torch.log1p(
            excess_kurtosis.abs()
        )

        radius_grid = radius[None, None, None]
        centroid = (
            masked_power * radius_grid
        ).sum(dim=(-2, -1)) / band_power.clamp_min(eps)
        radial_variance = (
            masked_power
            * (radius_grid - centroid[..., None, None]).square()
        ).sum(dim=(-2, -1)) / band_power.clamp_min(eps)
        spread = torch.sqrt(radial_variance + eps) - math.sqrt(eps)

        if channels > 1:
            reference = (
                spectrum.sum(dim=1, keepdim=True) - spectrum
            ) / float(channels - 1)
        else:
            reference = spectrum
        cross_spectrum = spectrum[:, :, None] * reference.conj()[:, :, None]
        cross_power = (cross_spectrum * weighted_masks).sum(dim=(-2, -1))
        reference_power = (
            reference.abs().square()[:, :, None] * weighted_masks
        ).sum(dim=(-2, -1))
        coherence = (
            cross_power.abs().square()
            / (band_power * reference_power).clamp_min(eps)
        ).clamp(0.0, 1.0)

        power_share = band_power / band_power.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
        features = torch.stack(
            [
                torch.log1p(band_power),
                power_share,
                entropy,
                flatness,
                torch.log1p(crest.clamp_max(1e4)),
                torch.log1p(peak_to_floor.clamp_max(1e4)),
                signed_log_kurtosis,
                centroid,
                spread,
                coherence,
            ],
            dim=-1,
        )
        features = torch.where(
            valid[..., None], features, torch.zeros_like(features)
        )
        return features, valid, band_power

    def _competitive_gains(self, logits, valid, dim=-1):
        """Allocate a fixed mean-one gain budget over valid regions."""

        any_valid = valid.any(dim=dim, keepdim=True)
        allocation_valid = torch.where(any_valid, valid, torch.ones_like(valid))
        masked_logits = torch.where(
            allocation_valid,
            logits / self.gate_temperature,
            torch.full_like(logits, -1e4),
        )
        weights = torch.softmax(masked_logits, dim=dim)
        weights = weights * allocation_valid.to(weights.dtype)
        weights = weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)
        count = allocation_valid.sum(dim=dim, keepdim=True).to(logits.dtype)
        gains = self.gate_floor + count * (1.0 - self.gate_floor) * weights
        gains = torch.where(allocation_valid, gains, torch.ones_like(gains))
        return gains, weights

    @staticmethod
    def _normalized_allocation_entropy(weights, valid, dim=-1):
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=dim)
        count = valid.sum(dim=dim).to(weights.dtype)
        return torch.where(
            count > 1,
            entropy / torch.log(count.clamp_min(2.0)),
            torch.zeros_like(entropy),
        )

    @staticmethod
    def _all_mode_stats(spectrum, eps=1e-8):
        """Return coherent instance statistics without sorting frequency modes."""

        energy = spectrum.abs().square().mean(dim=1, keepdim=True)
        multiplicity = energy.new_ones(1, 1, 1, energy.shape[-1])
        if energy.shape[-1] > 2:
            multiplicity[..., 1:-1] = 2.0
        flat = (energy * multiplicity).flatten(2).float()
        total = flat.sum(dim=-1)
        valid = total > eps
        probabilities = flat / total[..., None].clamp_min(eps)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(eps))).sum(dim=-1)
        if flat.shape[-1] > 1:
            entropy = entropy / math.log(flat.shape[-1])
        ones = valid.to(flat.dtype)
        return {
            "energy": total * ones,
            "coverage": ones,
            "entropy": entropy * ones,
            "density": ones,
            "valid": valid,
        }

    def _spectral_forward(self, spectrum):
        _, _, height, width = spectrum.shape
        radius = radial_frequency_grid(
            height,
            width,
            device=spectrum.device,
            dtype=spectrum.real.dtype,
        )
        cutoff = torch.sigmoid(self.spectral_filter_cutoff_logit).to(
            spectrum.real.dtype
        )
        low_response = gaussian_low_pass_response(radius, cutoff)
        high_response = 1.0 - low_response
        low_spectrum = spectrum * low_response[None, None]
        high_spectrum = spectrum * high_response[None, None]
        low_stats = self._all_mode_stats(low_spectrum)
        high_stats = self._all_mode_stats(high_spectrum)

        def energy_radius(region_spectrum):
            region_energy = region_spectrum.abs().square().mean(dim=1, keepdim=True)
            multiplicity = region_energy.new_ones(
                1, 1, 1, region_energy.shape[-1]
            )
            if region_energy.shape[-1] > 2:
                multiplicity[..., 1:-1] = 2.0
            region_energy = region_energy * multiplicity
            numerator = (region_energy * radius[None, None]).sum(dim=(-2, -1))
            denominator = region_energy.sum(dim=(-2, -1))
            return numerator / denominator.clamp_min(1e-8)

        low_radius = energy_radius(low_spectrum)
        high_radius = energy_radius(high_spectrum)
        low_delta = self._complex_residual(
            low_spectrum, self.low_weight_real, self.low_weight_imag
        )
        high_delta = self._complex_residual(
            high_spectrum, self.high_weight_real, self.high_weight_imag
        )

        full_band = radius.new_ones(1, height, width)
        low_features, low_valid, low_power = self._signal_descriptors(
            low_spectrum, full_band, radius
        )
        high_features, high_valid, high_power = self._signal_descriptors(
            high_spectrum, full_band, radius
        )
        region_features = torch.stack(
            [low_features[:, :, 0], high_features[:, :, 0]], dim=2
        )
        region_valid = torch.stack(
            [low_valid[:, :, 0], high_valid[:, :, 0]], dim=2
        )
        region_power = torch.stack(
            [low_power[:, :, 0], high_power[:, :, 0]], dim=2
        )
        region_features = region_features.clone()
        region_features[..., 1] = region_power / region_power.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-8)
        region_logits = self.region_signal_router(region_features, region_valid)
        region_gates, region_allocation = self._competitive_gains(
            region_logits, region_valid, dim=2
        )
        low_gate = region_gates[:, :, 0, None, None]
        high_gate = region_gates[:, :, 1, None, None]

        self.last_selection_stats = {
            "cutoff": cutoff.detach(),
            "low_coverage": low_stats["coverage"].detach(),
            "high_coverage": high_stats["coverage"].detach(),
            "low_density": low_stats["density"].detach(),
            "high_density": high_stats["density"].detach(),
            "low_radius": low_radius.detach(),
            "high_radius": high_radius.detach(),
            "overlap_density": spectrum.real.new_ones(spectrum.shape[0], 1),
            "low_valid": low_stats["valid"].detach(),
            "high_valid": high_stats["valid"].detach(),
            "region_gate": region_gates.detach(),
            "region_features": region_features.detach(),
            "region_logits": region_logits.detach(),
            "region_effective_features": (
                self.region_signal_router.last_effective_features.detach()
            ),
            "region_prior_contributions": (
                self.region_signal_router.last_prior_contributions.detach()
            ),
            "region_learned_score": (
                self.region_signal_router.last_learned_score.detach()
            ),
            "region_allocation": region_allocation.detach(),
            "region_gate_valid": region_valid.detach(),
            "region_gate_entropy": self._normalized_allocation_entropy(
                region_allocation, region_valid, dim=2
            ).detach(),
        }
        return spectrum + low_gate * low_delta + high_gate * high_delta
    def forward(self, x, time, U_true=None, input_dimensions=None):
        del U_true
        batch_size, sequence_length, channels = x.shape
        height, width = input_dimensions or self.patch_grid
        if sequence_length != height * width:
            raise ValueError(
                f"Spectral block expected {height}x{width} tokens, got {sequence_length}"
            )
        spatial = x.view(
            batch_size, height, width, channels
        ).permute(0, 3, 1, 2).contiguous()
        spectrum = torch.fft.rfft2(spatial, norm=self.FFT_norm)
        fused = self._spectral_forward(spectrum)
        spectral_output = torch.fft.irfft2(
            fused - spectrum,
            s=(height, width),
            norm=self.FFT_norm,
        )

        local_output = self.conv_local(spatial)
        spatial_tokens = spatial.permute(0, 2, 3, 1).reshape(
            batch_size, sequence_length, channels
        )
        pointwise_output = self.linear(spatial_tokens)
        pointwise_output = pointwise_output.reshape(
            batch_size, height, width, channels
        ).permute(0, 3, 1, 2)

        output = (
            spectral_output + local_output + pointwise_output
        ).permute(0, 2, 3, 1).reshape(
            batch_size, sequence_length, channels
        )
        return self.drop_path(self.norm(output, time))


def collect_dual_energy_statistics(module):
    collected = {
        "cutoff": [],
        "low_coverage": [],
        "high_coverage": [],
        "low_density": [],
        "high_density": [],
        "low_radius": [],
        "high_radius": [],
        "overlap_density": [],
        "region_gate_mean": [],
        "region_gate_std": [],
        "region_gate_contrast": [],
        "region_gate_entropy": [],
        "region_gate_instance_std": [],
    }
    for child in module.modules():
        if not isinstance(child, DualRegionEnergySpectralBlock):
            continue
        stats = child.last_selection_stats
        if stats is None:
            continue
        collected["cutoff"].append(stats["cutoff"].float().mean())
        for region in ("low", "high"):
            valid = stats[f"{region}_valid"]
            if not valid.any():
                continue
            for metric in ("coverage", "density"):
                values = stats[f"{region}_{metric}"]
                collected[f"{region}_{metric}"].append(
                    values[valid].float().mean()
                )
            collected[f"{region}_radius"].append(
                stats[f"{region}_radius"][valid].float().mean()
            )
        collected["overlap_density"].append(
            stats["overlap_density"].float().mean()
        )
        region_gates = stats["region_gate"].float()
        collected["region_gate_mean"].append(region_gates.mean())
        collected["region_gate_std"].append(
            region_gates.std(dim=-1, unbiased=False).mean()
        )
        collected["region_gate_contrast"].append(
            (region_gates[..., 0] - region_gates[..., 1]).abs().mean()
        )
        collected["region_gate_instance_std"].append(
            region_gates.std(dim=0, unbiased=False).mean()
        )
        if "region_gate_entropy" in stats:
            region_entropy = stats["region_gate_entropy"].float().mean()
        else:
            allocation = region_gates / region_gates.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            region_entropy = -(
                allocation * torch.log(allocation.clamp_min(1e-8))
            ).sum(dim=-1).mean() / math.log(2.0)
        collected["region_gate_entropy"].append(region_entropy)
    if not collected["cutoff"]:
        return None
    return {
        name: (
            torch.stack(values).mean()
            if values
            else collected["cutoff"][0].new_tensor(0.0)
        )
        for name, values in collected.items()
    }


@dataclass
class ScOTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    base_loss: Optional[torch.FloatTensor] = None
    output: torch.FloatTensor = None
    spectral_cutoff: Optional[torch.FloatTensor] = None
    spectral_low_coverage: Optional[torch.FloatTensor] = None
    spectral_high_coverage: Optional[torch.FloatTensor] = None
    spectral_low_density: Optional[torch.FloatTensor] = None
    spectral_high_density: Optional[torch.FloatTensor] = None
    spectral_low_radius: Optional[torch.FloatTensor] = None
    spectral_high_radius: Optional[torch.FloatTensor] = None
    spectral_overlap_density: Optional[torch.FloatTensor] = None
    spectral_region_gate_mean: Optional[torch.FloatTensor] = None
    spectral_region_gate_std: Optional[torch.FloatTensor] = None
    spectral_region_gate_contrast: Optional[torch.FloatTensor] = None
    spectral_region_gate_entropy: Optional[torch.FloatTensor] = None
    spectral_region_gate_instance_std: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class ScOTConfig(PretrainedConfig):
    """Configuration for the final single-frame DRESO architecture."""

    model_type = "swinv2"
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        num_out_channels=1,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        skip_connections=(True, True, True),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=2,
        channel_slice_list_normalized_loss=None,
        fft_norm="forward",
        spectral_filter_cutoff=0.25,
        spectral_mixer_init_scale=0.02,
        spectral_region_gate_hidden=64,
        spectral_gate_temperature=1.0,
        spectral_gate_floor=0.25,
        spectral_gate_prior_weight=1.0,
        hf_loss_lambda=0.3,
        hf_loss_alpha=1.5,
        use_hf_loss=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = int(num_channels)
        self.num_out_channels = int(num_out_channels)
        self.embed_dim = int(embed_dim)
        self.depths = list(depths)
        self.num_layers = len(self.depths)
        self.num_heads = list(num_heads)
        self.skip_connections = list(skip_connections)
        self.window_size = window_size
        self.mlp_ratio = float(mlp_ratio)
        self.qkv_bias = bool(qkv_bias)
        self.hidden_dropout_prob = float(hidden_dropout_prob)
        self.attention_probs_dropout_prob = float(attention_probs_dropout_prob)
        self.drop_path_rate = float(drop_path_rate)
        self.hidden_act = hidden_act
        self.initializer_range = float(initializer_range)
        self.layer_norm_eps = float(layer_norm_eps)
        self.p = int(p)
        self.channel_slice_list_normalized_loss = channel_slice_list_normalized_loss
        self.hidden_size = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.pretrained_window_sizes = (0, 0, 0, 0)

        # Final DRESO contract. These values are metadata, not experiment
        # switches: clean releases expose one architecture only.
        self.fft_norm = str(fft_norm)
        self.use_spectral = True
        self.spectral_operator_mode = "dual_energy"
        self.spectral_filter_type = "gaussian"
        self.spectral_filter_cutoff = float(spectral_filter_cutoff)
        self.spectral_filter_learnable_cutoff = True
        self.spectral_mixer_init = "identity_perturbed"
        self.spectral_mixer_init_scale = float(spectral_mixer_init_scale)
        self.spectral_selection_mode = "all"
        self.spectral_use_instance_router = True
        self.spectral_router_normalize_features = True
        self.spectral_gate_design = "signal_competitive"
        self.spectral_gate_allocation = "softmax"
        self.spectral_full_spectrum_identity = True
        self.spectral_residual_contract = "outer_correction"
        self.spectral_region_gate_hidden = int(spectral_region_gate_hidden)
        self.spectral_gate_temperature = float(spectral_gate_temperature)
        self.spectral_gate_floor = float(spectral_gate_floor)
        self.spectral_gate_prior_weight = float(spectral_gate_prior_weight)
        self.hf_loss_lambda = float(hf_loss_lambda)
        self.hf_loss_alpha = float(hf_loss_alpha)
        self.use_hf_loss = bool(use_hf_loss)

        if not 0.0 < self.spectral_filter_cutoff < 1.0:
            raise ValueError("spectral_filter_cutoff must be in (0, 1).")
        if self.spectral_mixer_init_scale < 0.0:
            raise ValueError("spectral_mixer_init_scale must be non-negative.")
        if self.spectral_region_gate_hidden < 1:
            raise ValueError("spectral_region_gate_hidden must be positive.")
        if self.spectral_gate_temperature <= 0.0:
            raise ValueError("spectral_gate_temperature must be positive.")
        if not 0.0 <= self.spectral_gate_floor < 1.0:
            raise ValueError("spectral_gate_floor must be in [0, 1).")
        if self.spectral_gate_prior_weight < 0.0:
            raise ValueError("spectral_gate_prior_weight must be non-negative.")

        self.keys_to_ignore_at_inference = [
            "base_loss",
            "spectral_cutoff",
            "spectral_low_coverage",
            "spectral_high_coverage",
            "spectral_low_density",
            "spectral_high_density",
            "spectral_low_radius",
            "spectral_high_radius",
            "spectral_overlap_density",
            "spectral_region_gate_mean",
            "spectral_region_gate_std",
            "spectral_region_gate_contrast",
            "spectral_region_gate_entropy",
            "spectral_region_gate_instance_std",
        ]

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, time):
        return super().forward(x)


def build_operator_norm(config, dim, eps=None):
    """Build the LayerNorm used by every DRESO operator block."""
    eps = config.layer_norm_eps if eps is None else eps
    return LayerNorm(dim, eps=eps)


class ConvNeXtBlock(nn.Module):
    r"""Taken from: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, config, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  
        self.norm = build_operator_norm(config, dim)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.weight = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )  
        self.drop_path = Swinv2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, time, input_dimensions=None):
        batch_size, sequence_length, hidden_size = x.shape
        if input_dimensions is None:
            input_dim = math.isqrt(sequence_length)
            input_dimensions = (input_dim, input_dim)
        height, width = input_dimensions
        if height * width != sequence_length:
            raise ValueError(f"Invalid token grid {input_dimensions} for {sequence_length} tokens")

        input = x
        x = x.reshape(batch_size, height, width, hidden_size)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x, time)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.weight is not None:
            x = self.weight * x
        x = x.reshape(batch_size, sequence_length, hidden_size)

        x = input + self.drop_path(x)
        return x


class ScOTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )

        self.projection = nn.Conv2d(
            self.num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(
        self, pixel_values: Optional[torch.FloatTensor]
    ) -> Tuple[torch.Tensor, Tuple[int]]:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                f"DRESO expects {self.num_channels} input channels, got "
                f"{num_channels}. The clean model uses one input frame."
            )
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, output_dimensions


class ScOTEmbeddings(nn.Module):
    """Construct convolutional patch embeddings and an optional mask token."""

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = ScOTPatchEmbeddings(config)
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            if use_mask_token
            else None
        )

        self.norm = build_operator_norm(config, config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor],
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        time: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings, time)
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions



class ScOTLayer(nn.Module):
    """One DRESO spectral operator block followed by the channel MLP."""

    def __init__(
        self,
        config,
        dim,
        input_resolution,
        num_heads,
        drop_path=0.0,
        shift_size=0,
        pretrained_window_size=0,
    ):
        super().__init__()
        del num_heads, shift_size, pretrained_window_size
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.input_resolution = input_resolution
        self.attention = DualRegionEnergySpectralBlock(
            config=config,
            dim=dim,
            patch_grid=input_resolution,
            drop_path=drop_path,
        )
        self.attn_type = "spectral"
        self.drop_path = (
            Swinv2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.intermediate = Swinv2Intermediate(config, dim)
        self.output = Swinv2Output(config, dim)
        self.layernorm_after = build_operator_norm(config, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        time: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        U_true: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        del head_mask, always_partition
        shortcut = hidden_states
        hidden_states = shortcut + self.attention(
            shortcut, time, U_true=U_true, input_dimensions=input_dimensions
        )
        residual = hidden_states
        layer_output = self.output(self.intermediate(hidden_states))
        layer_output = residual + self.drop_path(
            self.layernorm_after(layer_output, time)
        )
        return (layer_output, None) if output_attentions else (layer_output,)

class ScOTPatchRecovery(nn.Module):
    """https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py"""

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_out_channels, hidden_size = (
            config.num_out_channels,
            config.embed_dim,  
        )
        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[0] // patch_size[0]) * (
            image_size[1] // patch_size[1]
        )
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_out_channels = num_out_channels
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )

        self.projection = nn.ConvTranspose2d(
            in_channels=hidden_size,
            out_channels=num_out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.mixup = nn.Conv2d(
            num_out_channels,
            num_out_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )

    def maybe_crop(self, pixel_values, height, width):
        if pixel_values.shape[2] > height:
            pixel_values = pixel_values[:, :, :height, :]
        if pixel_values.shape[3] > width:
            pixel_values = pixel_values[:, :, :, :width]
        return pixel_values

    def forward(self, hidden_states, grid_size=None, output_size=None):
        grid_size = tuple(grid_size or self.grid_size)
        output_size = tuple(output_size or self.image_size)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], hidden_states.shape[1], *grid_size
        )

        output = self.projection(hidden_states)
        output = self.maybe_crop(output, output_size[0], output_size[1])
        return self.mixup(output)


class ScOTPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(
        self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = LayerNorm
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(
        self,
        input_feature: torch.Tensor,
        input_dimensions: Tuple[int, int],
        time: torch.Tensor,
    ) -> torch.Tensor:
        height, width = input_dimensions
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        input_feature = self.maybe_pad(input_feature, height, width)
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        input_feature = torch.cat(
            [input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1
        )
        input_feature = input_feature.view(
            batch_size, -1, 4 * num_channels
        )  

        input_feature = self.reduction(input_feature)
        input_feature = self.norm(input_feature, time)

        return input_feature


class ScOTPatchUnmerging(nn.Module):
    def __init__(
        self,
        input_resolution: Tuple[int],
        dim: int,
        norm_layer: nn.Module = LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.upsample = nn.Linear(dim, 2 * dim, bias=False)
        self.mixup = nn.Linear(dim // 2, dim // 2, bias=False)
        self.norm = norm_layer(dim // 2)

    def maybe_crop(self, input_feature, height, width):
        height_in, width_in = input_feature.shape[1], input_feature.shape[2]
        if height_in > height:
            input_feature = input_feature[:, :height, :, :]
        if width_in > width:
            input_feature = input_feature[:, :, :width, :]
        return input_feature

    def forward(
        self,
        input_feature: torch.Tensor,
        output_dimensions: Tuple[int, int],
        time: torch.Tensor,
    ) -> torch.Tensor:
        output_height, output_width = output_dimensions
        batch_size, seq_len, hidden_size = input_feature.shape
        input_height = (output_height + 1) // 2
        input_width = (output_width + 1) // 2
        if input_height * input_width != seq_len:
            raise ValueError(
                f"Cannot unmerge {seq_len} tokens to {output_height}x{output_width}"
            )
        input_feature = self.upsample(input_feature)
        input_feature = input_feature.reshape(
            batch_size, input_height, input_width, 2, 2, hidden_size // 2
        )
        input_feature = input_feature.permute(0, 1, 3, 2, 4, 5)
        input_feature = input_feature.reshape(
            batch_size, 2 * input_height, 2 * input_width, hidden_size // 2
        )

        input_feature = self.maybe_crop(input_feature, output_height, output_width)
        input_feature = input_feature.reshape(batch_size, -1, hidden_size // 2)

        input_feature = self.norm(input_feature, time)
        return self.mixup(input_feature)


class ScOTEncodeStage(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        depth,
        num_heads,
        drop_path,
        downsample,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        window_size = (
            config.window_size
            if isinstance(config.window_size, collections.abc.Iterable)
            else (config.window_size, config.window_size)
        )
        self.blocks = nn.ModuleList(
            [
                ScOTLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=(
                        [0, 0]
                        if (i % 2 == 0)
                        else [window_size[0] // 2, window_size[1] // 2]
                    ),
                    drop_path=drop_path[i],
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            layer_norm = lambda width: build_operator_norm(config, width)
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=layer_norm
            )
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        time: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions

        inputs = hidden_states

        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                time,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
                always_partition=always_partition,
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(
                hidden_states_before_downsampling + inputs, input_dimensions, time
            )
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states,
            hidden_states_before_downsampling,
            output_dimensions,
        )

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class ScOTDecodeStage(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        depth,
        num_heads,
        drop_path,
        upsample,
        upsampled_size,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        window_size = (
            config.window_size
            if isinstance(config.window_size, collections.abc.Iterable)
            else (config.window_size, config.window_size)
        )
        self.blocks = nn.ModuleList(
            [
                ScOTLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=(
                        [0, 0]
                        if (i % 2 == 0)
                        else [window_size[0] // 2, window_size[1] // 2]
                    ),
                    drop_path=drop_path[depth - 1 - i],  
                    pretrained_window_size=pretrained_window_size,
                )
                for i in reversed(range(depth)) 
            ]
        )

        if upsample is not None:
            layer_norm = lambda width: build_operator_norm(config, width)
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=layer_norm)
            self.upsampled_size = upsampled_size
        else:
            self.upsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        time: torch.Tensor,
        U_true: Optional[Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions

        
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                time,
                U_true=None,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
                always_partition=always_partition,
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_upsampling = hidden_states
        if self.upsample is not None:
            height_upsampled, width_upsampled = 2 * height, 2 * width
            output_dimensions = (height, width, height_upsampled, width_upsampled)
            hidden_states = self.upsample(
                hidden_states_before_upsampling,
                (height_upsampled, width_upsampled),
                time,
            )
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states,
            hidden_states_before_upsampling,
            output_dimensions,
        )

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs



class ScOTEncoder(nn.Module):
    """
    This is just a Swinv2Encoder with changed dpr.
    We just have to change the drop path rate since we also have a decoder by default.
    """

    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        drop_rates_encode_decode = torch.linspace(
            0, config.drop_path_rate, 2 * sum(config.depths)
        )
        dpr = [
            x.item()
            for x in drop_rates_encode_decode[: drop_rates_encode_decode.shape[0] // 2]
        ]
        self.layers = nn.ModuleList(
            [
                ScOTEncodeStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer),
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[
                        sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])
                    ],
                    downsample=(
                        ScOTPatchMerging if (i_layer < self.num_layers - 1) else None
                    ),
                    pretrained_window_size=pretrained_window_sizes[i_layer],
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        time: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Swinv2EncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            
            reshaped_hidden_state = hidden_states.view(
                batch_size, *input_dimensions, hidden_size
            )
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    input_dimensions,
                    time,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    input_dimensions,
                    time,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                )

            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size,
                    *(output_dimensions[0], output_dimensions[1]),
                    hidden_size,
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                reshaped_hidden_state = hidden_states.view(
                    batch_size, *input_dimensions, hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return Swinv2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class ScOTDecoder(nn.Module):
    """Here we do reverse encoder."""

    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        drop_rates_encode_decode = torch.linspace(
            0, config.drop_path_rate, 2 * sum(config.depths)
        )
        dpr = [
            x.item()
            for x in drop_rates_encode_decode[drop_rates_encode_decode.shape[0] // 2 :]
        ]
        self.layers = nn.ModuleList(
            [
                ScOTDecodeStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer),
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[
                        sum(config.depths[i_layer + 1 :]) : sum(config.depths[i_layer:])
                    ],
                    upsample=ScOTPatchUnmerging if i_layer > 0 else None,
                    upsampled_size=(
                        grid_size[0] // (2 ** (i_layer - 1)),
                        grid_size[1] // (2 ** (i_layer - 1)),
                    ),
                    pretrained_window_size=pretrained_window_sizes[i_layer],
                )
                for i_layer in reversed(range(self.num_layers))
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        skip_states: List[torch.FloatTensor],
        time: torch.Tensor,
        U_true: Optional[Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_upsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Swinv2EncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        U_true = None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            reshaped_hidden_state = hidden_states.view(
                batch_size, *input_dimensions, hidden_size
            )
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if i != 0 and skip_states[len(skip_states) - i] is not None:
                hidden_states = hidden_states + skip_states[len(skip_states) - i]

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    input_dimensions,
                    time,
                    U_true=U_true,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                    always_partition=always_partition,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    input_dimensions,
                    time,
                    U_true=U_true,                     
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                    always_partition=always_partition,
                )

            hidden_states = layer_outputs[0]
            hidden_states_before_upsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_upsampling:
                batch_size, _, hidden_size = hidden_states_before_upsampling.shape
                reshaped_hidden_state = hidden_states_before_upsampling.view(
                    batch_size,
                    *(output_dimensions[0], output_dimensions[1]),
                    hidden_size,
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_upsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_upsampling:
                batch_size, _, hidden_size = hidden_states.shape
                reshaped_hidden_state = hidden_states.view(
                    batch_size, *input_dimensions, hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return Swinv2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )



class ScOT(Swinv2PreTrainedModel):
    """Inspired by https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/modeling_swinv2.py#L1129"""

    # Swinv2PreTrainedModel defaults to Swinv2Config. Without this override,
    # from_pretrained() silently drops DRESO defaults when an older
    # checkpoint config does not yet contain newly introduced spectral fields.
    config_class = ScOTConfig

    def __init__(self, config, use_mask_token=False):
        super().__init__(config)

        self.config = config
        self.num_layers_encoder = len(config.depths)
        self.num_layers_decoder = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers_encoder - 1))

        self.embeddings = ScOTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ScOTEncoder(config, self.embeddings.patch_grid)
        self.decoder = ScOTDecoder(config, self.embeddings.patch_grid)
        self.patch_recovery = ScOTPatchRecovery(config)

        self.residual_blocks = nn.ModuleList(
            [
                (
                    nn.ModuleList(
                        [
                            ConvNeXtBlock(config, config.embed_dim * 2**i)
                            for _ in range(depth)
                        ]
                    )
                    if depth > 0
                    else nn.ModuleList([nn.Identity()])
                )
                for i, depth in enumerate(config.skip_connections)
            ]
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layers[layer].attention.prune_heads(heads)
        for layer, heads in reversed(heads_to_prune.items()):
            self.decoder.layers[layer].attention.prune_heads(heads)

    def _downsample(self, image, target_size):
        image_size = image.shape[-2]
        freqs = torch.fft.fftfreq(image_size, d=1 / image_size)
        sel = torch.logical_and(freqs >= -target_size / 2, freqs <= target_size / 2 - 1)
        image_hat = torch.fft.fft2(image, norm="forward")
        image_hat = image_hat[:, :, sel, :][:, :, :, sel]
        image = torch.fft.ifft2(image_hat, norm="forward").real
        return image

    def _upsample(self, image, target_size):
        image_size = image.shape[-2]
        image_hat = torch.fft.fft2(image, norm="forward")
        image_hat = torch.fft.fftshift(image_hat)
        pad_size = (target_size - image_size) // 2
        real = nn.functional.pad(
            image_hat.real, (pad_size, pad_size, pad_size, pad_size), value=0.0
        )
        imag = nn.functional.pad(
            image_hat.imag, (pad_size, pad_size, pad_size, pad_size), value=0.0
        )
        image_hat = torch.fft.ifftshift(torch.complex(real, imag))
        image = torch.fft.ifft2(image_hat, norm="forward").real
        return image

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        time: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ScOTOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("pixel_values cannot be None")

        head_mask = self.get_head_mask(
            head_mask, self.num_layers_encoder + self.num_layers_decoder
        )

        if isinstance(head_mask, list):
            head_mask_encoder = head_mask[: self.num_layers_encoder]
            head_mask_decoder = head_mask[self.num_layers_encoder :]
        else:
            head_mask_encoder, head_mask_decoder = head_mask.split(
                [self.num_layers_encoder, self.num_layers_decoder]
            )

        input_spatial_size = tuple(int(v) for v in pixel_values.shape[-2:])

        embedding_output, input_dimensions = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, time=time
        )

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            time,
            head_mask=head_mask_encoder,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=return_dict,
        )

        if return_dict:
            skip_states = list(encoder_outputs.hidden_states[1:])
        else:
            skip_states = list(encoder_outputs[1][1:])

        for i in range(len(skip_states)):
            for block in self.residual_blocks[i]:
                if isinstance(block, nn.Identity):
                    skip_states[i] = block(skip_states[i])
                else:
                    stage_dimensions = tuple(value // (2**i) for value in input_dimensions)
                    skip_states[i] = block(skip_states[i], time, stage_dimensions)

        deepest_dimensions = tuple(
            value // (2 ** (len(skip_states) - 1)) for value in input_dimensions
        )

        decoder_output = self.decoder(
            skip_states[-1],
            deepest_dimensions,
            time=time,
            U_true=None,
            skip_states=skip_states[:-1],
            head_mask=head_mask_decoder,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_output[0]
        prediction = self.patch_recovery(
            sequence_output,
            grid_size=input_dimensions,
            output_size=input_spatial_size,
        )
        if pixel_mask is not None and labels is not None:
            ignore_mask = expand_ignore_mask(pixel_mask, prediction)
            prediction = torch.where(
                ignore_mask,
                labels.type_as(prediction),
                prediction,
            )

        loss = None
        base_loss = None
        if labels is not None:
            loss = normalized_channel_group_loss(
                prediction,
                labels,
                self.config.channel_slice_list_normalized_loss,
                p=1,
                ignore_mask=pixel_mask,
            )

            base_loss = loss
            if self.config.use_hf_loss and self.config.hf_loss_lambda > 0:
                loss = loss + self.config.hf_loss_lambda * high_frequency_error_loss(
                    prediction,
                    labels,
                    alpha=self.config.hf_loss_alpha,
                    fft_norm=self.config.fft_norm,
                    ignore_mask=pixel_mask,
                ).mean()

        spectral_stats = collect_dual_energy_statistics(self)
        if not return_dict:
            output = (prediction,) + decoder_output[1:] + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ScOTOutput(
            loss=loss,
            base_loss=base_loss,
            output=prediction,
            spectral_cutoff=(
                spectral_stats["cutoff"] if spectral_stats is not None else None
            ),
            spectral_low_coverage=(
                spectral_stats["low_coverage"]
                if spectral_stats is not None
                else None
            ),
            spectral_high_coverage=(
                spectral_stats["high_coverage"]
                if spectral_stats is not None
                else None
            ),
            spectral_low_density=(
                spectral_stats["low_density"]
                if spectral_stats is not None
                else None
            ),
            spectral_high_density=(
                spectral_stats["high_density"]
                if spectral_stats is not None
                else None
            ),
            spectral_low_radius=(
                spectral_stats["low_radius"]
                if spectral_stats is not None
                else None
            ),
            spectral_high_radius=(
                spectral_stats["high_radius"]
                if spectral_stats is not None
                else None
            ),
            spectral_overlap_density=(
                spectral_stats["overlap_density"]
                if spectral_stats is not None
                else None
            ),
            spectral_region_gate_mean=(
                spectral_stats["region_gate_mean"]
                if spectral_stats is not None
                else None
            ),
            spectral_region_gate_std=(
                spectral_stats["region_gate_std"]
                if spectral_stats is not None
                else None
            ),
            spectral_region_gate_contrast=(
                spectral_stats["region_gate_contrast"]
                if spectral_stats is not None
                else None
            ),
            spectral_region_gate_entropy=(
                spectral_stats["region_gate_entropy"]
                if spectral_stats is not None
                else None
            ),
            spectral_region_gate_instance_std=(
                spectral_stats["region_gate_instance_std"]
                if spectral_stats is not None
                else None
            ),
            hidden_states=(
                decoder_output.hidden_states + encoder_outputs.hidden_states
                if output_hidden_states is not None and output_hidden_states is True
                else None
            ),
            attentions=(
                decoder_output.attentions + encoder_outputs.attentions
                if output_attentions is not None and output_attentions is True
                else None
            ),
            reshaped_hidden_states=(
                decoder_output.reshaped_hidden_states
                + encoder_outputs.reshaped_hidden_states
                if output_hidden_states is not None and output_hidden_states is True
                else None
            ),
        )


# Public DRESO aliases. Keep ScOT names as the serialization contract so all
# previously trained checkpoints remain fully compatible with from_pretrained.
DRESOConfig = ScOTConfig
DRESOOutput = ScOTOutput
DRESOModel = ScOT
DRESO = ScOT


