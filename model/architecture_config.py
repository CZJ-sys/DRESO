"""Persist the resolved DRESO architecture beside each checkpoint."""

import json
from pathlib import Path


def write_architecture_config(output_dir, architecture):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "architecture.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(architecture, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return config_path
