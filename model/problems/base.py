"""
This file contains the dataset selector get_dataset, as well as the base 
classes for all datasets.
"""

from torch.utils.data import Dataset, ConcatDataset
from typing import Optional, List, Dict
from abc import ABC
import re
import os
import shutil
from accelerate.utils import broadcast_object_list


def get_dataset(dataset, **kwargs):
    """Build one of the six Poseidon datasets or one The Well subset."""

    if isinstance(dataset, list):
        if any(item.startswith("well.") for item in dataset):
            if not all(item.startswith("well.") for item in dataset):
                raise ValueError("The Well and Poseidon cannot be mixed in one run.")
            if len(dataset) != 1:
                raise ValueError("Train one The Well subset per checkpoint.")
            return get_dataset(dataset[0], **kwargs)
        return ConcatDataset([get_dataset(d, **kwargs) for d in dataset])

    if dataset.startswith("well."):
        from .the_well import make_well_dataset

        return make_well_dataset(dataset, **kwargs)

    defaults = {"max_num_time_steps": 7, "time_step_size": 2}
    if dataset == "fluids.incompressible.Gaussians":
        from .fluids.incompressible import Gaussians as dset
        kwargs = {"tracer": False, **defaults, **kwargs}
    elif dataset == "fluids.incompressible.Sines":
        from .fluids.incompressible import Sines as dset
        kwargs = {"tracer": False, **defaults, **kwargs}
    elif dataset == "fluids.compressible.Riemann":
        from .fluids.compressible import Riemann as dset
        kwargs = {**defaults, **kwargs}
    elif dataset == "fluids.compressible.RiemannCurved":
        from .fluids.compressible import RiemannCurved as dset
        kwargs = {**defaults, **kwargs}
    elif dataset == "fluids.compressible.Gaussians":
        from .fluids.compressible import Gaussians as dset
        kwargs = {**defaults, **kwargs}
    elif dataset == "fluids.compressible.KelvinHelmholtz":
        from .fluids.compressible import KelvinHelmholtz as dset
        kwargs = {**defaults, **kwargs}
    else:
        raise ValueError(f"Unknown clean-release dataset {dataset!r}.")
    return dset(**kwargs)


class BaseDataset(Dataset, ABC):
    """A base class for all datasets. Can be directly derived from if you have a steady/non-time dependent problem."""

    def __init__(
        self,
        which: Optional[str] = None,
        num_trajectories: Optional[int] = None,
        data_path: Optional[str] = "./data",
        move_to_local_scratch: Optional[str] = None,
    ) -> None:
        """
        Args:
            which: Which dataset to use, i.e. train, val, or test.
            resolution: The resolution of the dataset.
            num_trajectories: The number of trajectories to use for training.
            data_path: The path to the data files.
            move_to_local_scratch: If not None, move the data to this directory at dataset initialization and use it from there.
        """
        assert which in ["train", "val", "test"]
        assert num_trajectories is not None and (
            num_trajectories > 0 or num_trajectories == -1
        )

        self.num_trajectories = num_trajectories
        self.data_path = data_path
        self.which = which
        self.move_to_local_scratch = move_to_local_scratch

    def _move_to_local_scratch(self, file_path):
        if self.move_to_local_scratch is not None:
            data_dir = os.path.join(self.data_path, file_path)
            file = file_path.split("/")[-1]
            scratch_dir = self.move_to_local_scratch
            dest_dir = os.path.join(scratch_dir, file)
            RANK = int(os.environ.get("LOCAL_RANK", -1))
            if not os.path.exists(dest_dir) and (RANK == 0 or RANK == -1):
                print(f"Start copying {file} to {dest_dir}...")
                shutil.copy(data_dir, dest_dir)
                print("Finished data copy.")
            # idk how to do the barrier differently
            ls = broadcast_object_list([dest_dir], from_process=0)
            dest_dir = ls[0]
            return dest_dir
        else:
            return file_path

    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths and normalization constants are set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        if self.num_trajectories == -1:
            self.num_trajectories = self.N_max - self.N_val - self.N_test
        assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        if self.which == "train":
            self.length = self.num_trajectories
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test
            self.start = self.N_max - self.N_test

        self.output_dim = self.label_description.count(",") + 1
        descriptors, channel_slice_list = self.get_channel_lists(self.label_description)
        self.printable_channel_description = descriptors
        self.channel_slice_list = channel_slice_list

    def __len__(self) -> int:
        """
        Returns: overall length of dataset.
        """
        return self.length

    def __getitem__(self, idx) -> Dict:
        """
        Get an item. OVERWRITE!

        Args:
            idx: The index of the sample to get.

        Returns:
            A dict of key-value pairs of data.
        """
        pass

    @staticmethod
    def get_channel_lists(label_description):
        matches = re.findall(r"\[([^\[\]]+)\]", label_description)
        channel_slice_list = [0]  # use as channel_slice_list[i]:channel_slice_list[i+1]
        beautiful_descriptors = []
        for match in matches:
            channel_slice_list.append(channel_slice_list[-1] + 1 + match.count(","))
            splt = match.split(",")
            if len(splt) > 1:
                beautiful_descriptors.append("".join(splt))
            else:
                beautiful_descriptors.append(match)
        return beautiful_descriptors, channel_slice_list


class BaseTimeDataset(BaseDataset, ABC):
    """Base class for single-frame time-dependent PDE datasets."""

    def __init__(
        self,
        *args,
        max_num_time_steps: Optional[int] = None,
        time_step_size: Optional[int] = None,
        fix_input_to_time_step: Optional[int] = None,
        allowed_time_transitions: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            max_num_time_steps: The maximum number of time steps to use.
            time_step_size: The size of the time step.
            fix_input_to_time_step: If not None, fix the input to this time step.
            allowed_time_transitions: If not None, only allow these time transitions (time steps).
        """
        assert max_num_time_steps is not None and max_num_time_steps > 0
        assert time_step_size is not None and time_step_size > 0
        assert fix_input_to_time_step is None or fix_input_to_time_step >= 0

        super().__init__(*args, **kwargs)
        self.max_num_time_steps = max_num_time_steps
        self.time_step_size = time_step_size
        self.fix_input_to_time_step = fix_input_to_time_step
        self.allowed_time_transitions = allowed_time_transitions

    def input_time_indices(self, current_time: int) -> List[int]:
        """Return the one input frame used by DRESO."""

        return [int(current_time)]

    def _idx_map(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1) + self.fix_input_to_time_step
            t = t2 - t1
        return i, t, t1, t2

    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths and normalization constants are set.
        self.max_time_step must have already been set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        if self.num_trajectories == -1:
            self.num_trajectories = self.N_max - self.N_val - self.N_test
        elif self.num_trajectories == -2:
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 2
        elif self.num_trajectories == -8:
            self.num_trajectories = (self.N_max - self.N_val - self.N_test) // 8
        assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        assert self.max_num_time_steps is not None and self.max_num_time_steps > 0
        if self.fix_input_to_time_step is not None:
            self.multiplier = self.max_num_time_steps
        else:
            self.time_indices = []
            for i in range(self.max_num_time_steps + 1):
                for j in range(i, self.max_num_time_steps + 1):
                    if (
                        self.allowed_time_transitions is not None
                        and (j - i) not in self.allowed_time_transitions
                    ):
                        continue
                    self.time_indices.append(
                        (self.time_step_size * i, self.time_step_size * j)
                    )
            self.multiplier = len(self.time_indices)
            if self.multiplier == 0:
                raise ValueError(
                    "No valid temporal samples remain for max_num_time_steps="
                    f"{self.max_num_time_steps}, and time_step_size="
                    f"{self.time_step_size}."
                )

        if self.which == "train":
            self.length = self.num_trajectories * self.multiplier
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val * self.multiplier
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test * self.multiplier
            self.start = self.N_max - self.N_test

        self.output_dim = self.label_description.count(",") + 1
        self.input_dim_per_frame = self.input_dim
        self.stacked_input_dim = self.input_dim
        descriptors, channel_slice_list = self.get_channel_lists(self.label_description)
        self.printable_channel_description = descriptors
        self.channel_slice_list = channel_slice_list


