# DRESO

**DRESO: Evidence-Guided Dual-Region Spectral Operators for Long-Horizon PDE Forecasting**

This repository is the clean reproduction version of DRESO. It includes the
model, training and evaluation on Poseidon and The Well, and diagnostic
visualizations of gate features and learned filter cutoffs. All training uses
single-frame inputs.

## Environment

Linux, Python 3.10, and an NVIDIA GPU are recommended. Conda creates the base
Python environment; all remaining dependencies are installed with pip:

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
bash scripts/setup_environment.sh
conda activate dreso
```

The setup script installs PyTorch 2.0.1 with CUDA 11.8. The Well is installed
with `--no-deps` to prevent it from replacing the verified NumPy, h5py, and
PyTorch versions.

## Data

The Poseidon root directory should directly contain `NS-Gauss.nc`, `CE-RP.nc`,
`CE-CRP.nc`, `CE-Gauss.nc`, `NS-Sines.nc`, and `CE-KH.nc`.

The Well loader supports these four original Hugging Face dataset directories
without preprocessing:

```text
acoustic_scattering_discontinuous/
active_matter/
gray_scott/
planetswe/
```

Each dataset must contain the official `data/train`, `data/valid`, and
`data/test` splits. Train a separate model from scratch for each The Well
dataset because their channel semantics differ.

## Poseidon Training

The default configuration trains jointly on the six datasets, using 200
trajectories per dataset for 200 epochs. Choose `Tiny`, `Big`, or `L`:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_poseidon.sh /path/to/Poseiden L dreso_poseidon_l
```

Training settings are defined in `configs/train_poseidon.yaml`. The default
checkpoint directory for this command is:

```text
checkpoint/dreso_poseidon/dreso_poseidon_l/
```

## Poseidon Evaluation

Evaluation starts from a single state at `t=0`. It reports dt1 at `t=1` and
the fully autoregressive rollout mean over `t=1..20`, averaged across all
predicted steps and trajectories. JSON results include both normalized and
physical relative L1 errors:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/evaluate_poseidon.sh \
  checkpoint/dreso_poseidon/dreso_poseidon_l \
  /path/to/Poseiden
```

Set `SAMPLES=-1` to evaluate the complete test split.

## The Well Training

By default, training samples 2,000 adjacent-frame pairs uniformly with
replacement from the selected dataset's training windows and runs for 50
epochs. Training and evaluation retain the native spatial grid:

| Dataset | Resolution |
| --- | --- |
| Acoustic scattering | 256 x 256 |
| Active matter | 256 x 256 |
| Gray-Scott | 128 x 128 |
| PlanetsWE | 256 x 512 |

Validation runs every epoch. The best checkpoint is selected by minimum
`eval_loss`. When HF loss is enabled, this is the total validation loss,
including base normalized relative L1 and HF loss. The final model exported
to the run directory is the reloaded best checkpoint, not the last epoch.

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_the_well.sh \
  "/path/to/The Well" well.active_matter L
```

Supported dataset identifiers:

```text
well.acoustic_discontinuous
well.active_matter
well.gray_scott
well.planetswe
```

Settings are defined in `configs/train_the_well.yaml`. Override training
resources through environment variables, for example:

```bash
EPOCHS=50 NUM_SAMPLES=2000 BATCH_SIZE=8 GRAD_ACCUM=1 \
bash scripts/train_the_well.sh "/path/to/The Well" well.gray_scott Big
```

## The Well Evaluation

Evaluate a checkpoint on the dataset and native grid used for training:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/evaluate_the_well.sh \
  checkpoint/dreso_the_well/dreso_active_matter_L \
  "/path/to/The Well" well.active_matter
```

Results include normalized and physical relative L1 means for dt1 and
autoregressive rollout. No spatial resizing is performed.

## Visualizations

Inspect the learned cutoff distribution across spectral blocks:

```bash
python evaluate/analyze_cutoff_distribution.py \
  --checkpoint /path/to/checkpoint \
  --output diagnostics/cutoff_distribution
```

Inspect the ten gate descriptors. By default, this collects feature responses
without additional causal masking evaluations:

```bash
python evaluate/analyze_gate_feature_importance.py \
  --checkpoint /path/to/checkpoint \
  --data_path /path/to/Poseiden \
  --datasets NS-Gauss CE-RP CE-CRP CE-Gauss NS-Sines CE-KH \
  --num_samples 64 \
  --output_dir diagnostics/gate_features
```

Analyze low- and high-frequency descriptors along a physical PDE trajectory:

```bash
python evaluate/analyze_ce_rm_gate_trajectory.py \
  --dataset CE-RM \
  --data /path/to/Poseiden/CE-RM.nc \
  --sample_index 0 \
  --cutoff 0.25 \
  --output_dir diagnostics/ce_rm_gate_trajectory_sample0 \
  --save_vector
```

## Notes

- Training requires CUDA and never silently falls back to CPU.
- Poseidon incompressible NS tasks supervise and evaluate only velocity
  channels `[u, v]`. Artificial density and pressure channels remain constant
  during rollout.
- The Well uses official field-wise z-score normalization and keeps static
  fields fixed during rollout.
- The `ScOT` class name is retained for compatibility with existing Hugging
  Face checkpoint serialization.
