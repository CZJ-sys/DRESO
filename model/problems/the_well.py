"""Single-frame The Well adapter for DRESO training and evaluation."""

from __future__ import annotations

from pathlib import Path

import torch
from .base import BaseDataset


THE_WELL_SUBSETS = (
    "well.acoustic_discontinuous",
    "well.active_matter",
    "well.gray_scott",
    "well.planetswe",
)

THE_WELL_NATIVE_RESOLUTIONS = {
    "well.acoustic_discontinuous": (256, 256),
    "well.active_matter": (256, 256),
    "well.gray_scott": (128, 128),
    "well.planetswe": (256, 512),
}

_DIRECTORY_ALIASES = {
    "well.acoustic_discontinuous": (
        "acoustic_scattering_discontinuous",
        "Acoustic Scattering - Single Discontinuity",
    ),
    "well.active_matter": ("active_matter", "active"),
    "well.gray_scott": ("gray_scott_reaction_diffusion", "gray_scott"),
    "well.planetswe": ("planetswe", "planetwe"),
}


def _well_imports():
    try:
        from the_well.data import WellDataset
        from the_well.data.normalization import ZScoreNormalization
    except ImportError as error:
        raise ModuleNotFoundError(
            "The Well datasets require the official package. Install it with "
            "`pip install 'the_well>=1.2,<2'`."
        ) from error
    return WellDataset, ZScoreNormalization


def _resolve_subset_path(root: str | Path, subset: str) -> Path:
    if subset not in THE_WELL_SUBSETS:
        raise ValueError(f"Unknown The Well subset {subset!r}")
    root = Path(root).expanduser().resolve()
    for directory in _DIRECTORY_ALIASES[subset]:
        candidate = root / directory
        if candidate.is_dir():
            return candidate
    expected = ", ".join(_DIRECTORY_ALIASES[subset])
    raise FileNotFoundError(f"Cannot find {subset} under {root}; expected one of: {expected}")


class TheWellDataset(BaseDataset):
    """Return normalized adjacent transitions in the ScOT sample contract."""

    fixed_lead_time = 1.0
    def __init__(self, *args, subset: str, **kwargs):
        kwargs.pop("max_num_time_steps", None)
        kwargs.pop("time_step_size", None)
        kwargs.pop("allowed_time_transitions", None)
        kwargs.pop("fix_input_to_time_step", None)
        super().__init__(*args, **kwargs)
        self.subset = subset
        self.resolution = THE_WELL_NATIVE_RESOLUTIONS[subset]
        self.path = _resolve_subset_path(self.data_path, subset)
        split = {"train": "train", "val": "valid", "test": "test"}[self.which]
        WellDataset, ZScoreNormalization = _well_imports()
        self.source = WellDataset(
            path=str(self.path),
            well_split_name=split,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            n_steps_input=1,
            n_steps_output=1,
            min_dt_stride=1,
            max_dt_stride=1,
            return_grid=False,
            boundary_return_type=None,
        )
        self.dynamic_channels = int(self.source.metadata.n_fields)
        self.constant_channels = int(self.source.metadata.n_constant_fields)
        if self.dynamic_channels < 1:
            raise ValueError(f"{subset} contains no time-varying fields")
        requested_length = int(self.num_trajectories)
        self.use_full_split = self.which == "train" and requested_length == -1
        if self.use_full_split:
            self.length = len(self.source)
        elif requested_length > 0:
            self.length = (
                requested_length
                if self.which == "train"
                else min(requested_length, 2000, len(self.source))
            )
        else:
            raise ValueError(
                "The Well supports a positive sample budget or -1 for the "
                "complete training split."
            )
        if self.length < 1:
            raise ValueError(f"{subset} {split} split contains no usable windows")
        self.sampling_mode = (
            "full_indexed"
            if self.use_full_split
            else ("uniform_with_replacement" if self.which == "train" else "indexed")
        )
        print(
            f"{subset} split={split}: source_windows={len(self.source)}, "
            f"selected_windows={self.length}, sampling={self.sampling_mode}",
            flush=True,
        )
        self._set_channel_contract(self.dynamic_channels, self.constant_channels)

    def _set_channel_contract(self, dynamic_channels: int, constant_channels: int):
        self.max_dynamic_channels = int(dynamic_channels)
        self.max_constant_channels = int(constant_channels)
        self.input_dim = self.max_dynamic_channels + self.max_constant_channels
        self.output_dim = self.max_dynamic_channels
        self.channel_slice_list = [0, self.output_dim]
        self.printable_channel_description = ["dynamic_fields"]
        self.label_description = "[dynamic_fields]"
        self.pixel_mask = torch.arange(self.output_dim) >= self.dynamic_channels

    def __len__(self):
        return self.length

    @staticmethod
    def _pad_channels(value: torch.Tensor, channels: int) -> torch.Tensor:
        if value.shape[1] == channels:
            return value
        padded = value.new_zeros((value.shape[0], channels, *value.shape[2:]))
        padded[:, : value.shape[1]] = value
        return padded

    def __getitem__(self, index):
        if self.which == "train" and not self.use_full_split:
            source_index = int(torch.randint(len(self.source), ()).item())
        else:
            source_index = int(index) % len(self.source)
        sample = self.source[source_index]
        inputs = torch.nan_to_num(sample["input_fields"].float()).permute(0, 3, 1, 2)
        labels = torch.nan_to_num(sample["output_fields"][0].float()).permute(2, 0, 1)
        if tuple(inputs.shape[-2:]) != self.resolution:
            raise ValueError(
                f"{self.subset} expected native resolution {self.resolution}, "
                f"got {tuple(inputs.shape[-2:])}"
            )
        inputs = self._pad_channels(inputs, self.max_dynamic_channels)
        labels = self._pad_channels(labels.unsqueeze(0), self.max_dynamic_channels).squeeze(0)

        constants = sample.get("constant_fields")
        if constants is None or constants.numel() == 0:
            constants = inputs.new_zeros((1, self.max_constant_channels, *self.resolution))
        else:
            constants = torch.nan_to_num(constants.float()).permute(2, 0, 1).unsqueeze(0)
            constants = self._pad_channels(constants, self.max_constant_channels)
        inputs = torch.cat((inputs, constants), dim=1).flatten(0, 1)
        return {
            "pixel_values": inputs,
            "labels": labels,
            "time": 1.0,
            "pixel_mask": self.pixel_mask,
        }


def make_well_dataset(subset: str, **kwargs):
    return TheWellDataset(subset=subset, **kwargs)
