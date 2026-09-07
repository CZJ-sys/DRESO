"""Dataset-specific metric contracts used by fair Poseidon evaluation."""

from __future__ import annotations


INCOMPRESSIBLE_POSEIDON_DATASETS = frozenset(
    {
        "ns-gauss",
        "ns-sines",
    }
)


def evaluation_channels(dataset_name: str, num_channels: int) -> list[int]:
    """Return physical channels included in reported errors."""
    normalized_name = dataset_name.lower().replace("_", "-")
    if (
        "incompressible" in normalized_name
        or normalized_name in INCOMPRESSIBLE_POSEIDON_DATASETS
    ):
        if num_channels == 2:
            return [0, 1]
        if num_channels < 3:
            raise ValueError(
                "Incompressible NS evaluation needs two velocity channels, "
                f"got {num_channels}."
            )
        # Full Poseidon NS state: [synthetic density, u, v, synthetic pressure].
        return [1, 2]
    return list(range(num_channels))
