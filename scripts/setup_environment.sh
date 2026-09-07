#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-dreso}"

conda env create --name "${ENV_NAME}" --file "${REPO_ROOT}/environment.repro.yml"
conda run --name "${ENV_NAME}" python -m pip install --upgrade pip
conda run --name "${ENV_NAME}" python -m pip install \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
conda run --name "${ENV_NAME}" python -m pip install \
  --requirement "${REPO_ROOT}/requirements.repro.txt"

# Installing The Well without dependencies prevents it from replacing the
# verified NumPy/h5py/PyTorch stack above.
conda run --name "${ENV_NAME}" python -m pip install \
  --no-deps --force-reinstall the_well==1.2.0

conda run --name "${ENV_NAME}" python - <<'PY'
import h5py
import numpy
import torch
import the_well

print("numpy", numpy.__version__)
print("h5py", h5py.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("the_well", the_well.__version__ if hasattr(the_well, "__version__") else "import OK")
print("cuda_available", torch.cuda.is_available())
PY
