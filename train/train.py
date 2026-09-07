"""Train DRESO from scratch on Poseidon or one The Well subset."""

import argparse
import math
import os
import torch


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, got {value!r}."
    )


def _require_cuda_training():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    accelerate_use_cpu = os.environ.get("ACCELERATE_USE_CPU", "").strip().lower()
    details = (
        f"torch={torch.__version__}, torch_cuda={torch.version.cuda}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}, "
        f"LOCAL_RANK={local_rank}, "
        f"ACCELERATE_USE_CPU={accelerate_use_cpu or '<not set>'}"
    )
    try:
        if accelerate_use_cpu in {"1", "true", "yes", "on"}:
            raise RuntimeError("ACCELERATE_USE_CPU explicitly requests CPU execution")
        torch.cuda.init()
        visible_count = torch.cuda.device_count()
        if visible_count < 1:
            raise RuntimeError("PyTorch reports zero visible CUDA devices")
        if local_rank < 0 or local_rank >= visible_count:
            raise RuntimeError(
                f"LOCAL_RANK {local_rank} is outside {visible_count} visible GPUs"
            )
        torch.cuda.set_device(local_rank)
        probe = torch.ones(1, device=f"cuda:{local_rank}")
        probe.add_(1).item()
    except Exception as exc:
        raise RuntimeError(
            "GPU training is mandatory and the CUDA preflight failed before "
            f"loading Transformers/TensorBoard. {details}"
        ) from exc
    print(
        "CUDA preflight OK: "
        f"local_rank={local_rank}, device={torch.cuda.get_device_name(local_rank)}, "
        f"{details}"
    )


_require_cuda_training()

import numpy as np
import random
import psutil
import sys
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import transformers
from accelerate.utils import broadcast_object_list
from model.trainer import TrainingArguments, Trainer
from model.model import (
    DualRegionEnergySpectralBlock,
    ScOT,
    ScOTConfig,
)
from model.architecture_config import write_architecture_config
from model.problems.base import get_dataset
from model.utils import (
    get_num_parameters,
    read_cli,
    get_num_parameters_no_embed,
)
from model.metrics import relative_lp_error
from model.experiment import (
    build_train_eval_set_kwargs,
    drop_zero_time_pairs,
    normalize_dataset_config,
    unwrap_filtered_dataset,
)

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
MODEL_MAP = {
    "Tiny": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [1, 1, 1, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [2, 2, 2, 2],
        "embed_dim": 24,
    },
    "Big": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [1, 1, 1, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [2, 2, 2, 2],
        "embed_dim": 48,
    },
    "L": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [4, 4, 4, 4],
        "embed_dim": 48,
    },
}
MODEL_SIZE_ALIASES = {
    "tiny": "Tiny",
    "big": "Big",
    "l": "L",
    "large": "L",
}


def _load_config(params):
    with open(params.config, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def setup(params, model_map=True):
    RANK = int(os.environ.get("LOCAL_RANK", -1))
    world_size = max(int(os.environ.get("WORLD_SIZE", 1)), 1)
    available_cpu_cores = len(psutil.Process().cpu_affinity())
    total_worker_budget = min(available_cpu_cores, 16)
    automatic_workers = max(total_worker_budget // world_size, 1)
    CPU_CORES = (
        automatic_workers
        if params.num_workers is None
        else max(int(params.num_workers), 0)
    )
    if RANK in (-1, 0):
        print(
            f"Detected {available_cpu_cores} CPU cores; world_size={world_size}; "
            f"using {CPU_CORES} DataLoader workers per process "
            f"({CPU_CORES * world_size} total)."
        )
    if params.disable_tqdm:
        transformers.utils.logging.disable_progress_bar()

    config = _load_config(params)

    ckpt_dir = "./"
    if RANK == 0 or RANK == -1:
        ckpt_dir = os.path.join(
            params.checkpoint_path, params.project_name, params.run_name
        )
    if (RANK == 0 or RANK == -1) and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ls = broadcast_object_list([ckpt_dir], from_process=0)
    ckpt_dir = ls[0]

    if model_map:
        requested_size = params.model_size or config.get("model_name", "L")
        model_size = MODEL_SIZE_ALIASES.get(str(requested_size).lower())
        if model_size is None:
            raise ValueError(
                f"Unknown model size {requested_size!r}; choose Tiny, Big, or L."
            )
        config["model_name"] = model_size
        config = {**config, **MODEL_MAP[model_size]}
    return config, ckpt_dir, RANK, CPU_CORES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRESO from scratch.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for initialization, data loading, and training.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers per training process.",
    )
    parser.add_argument(
        "--model_size",
        choices=["Tiny", "Big", "L"],
        default=None,
        help="DRESO capacity preset; defaults to model_name in the config.",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument(
        "--use_absolute_embeddings",
        type=_str_to_bool,
        default=None,
        help=(
            "Enable a learned absolute position embedding on the patch grid. "
            "Defaults to use_absolute_embeddings in the YAML, or false."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Override config.dataset, including one The Well subset.",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=None,
        help="Override config.num_trajectories; The Well accepts -1 for all.",
    )
    params = read_cli(parser).parse_args()
    SEED = int(params.seed)
    if SEED < 0:
        raise ValueError("--seed must be non-negative.")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    config, ckpt_dir, RANK, CPU_CORES = setup(params)
    config["training_seed"] = SEED
    if RANK == 0 or RANK == -1:
        print(f"Training seed={SEED}")
    if params.num_trajectories is not None:
        if params.num_trajectories <= 0 and params.num_trajectories != -1:
            raise ValueError(
                "--num_trajectories must be positive, or -1 for the complete "
                "training split when the selected dataset supports it."
            )
        config["num_trajectories"] = int(params.num_trajectories)
    if params.lr is not None:
        if not math.isfinite(params.lr) or params.lr <= 0:
            raise ValueError("--lr must be finite and greater than zero.")
        config["lr"] = float(params.lr)
    if params.warmup_ratio is not None:
        if (
            not math.isfinite(params.warmup_ratio)
            or params.warmup_ratio < 0
            or params.warmup_ratio >= 1
        ):
            raise ValueError("--warmup_ratio must be finite and in [0, 1).")
        config["warmup_ratio"] = float(params.warmup_ratio)
    num_epochs_override = getattr(params, "num_epochs", None)
    if num_epochs_override is not None:
        if num_epochs_override <= 0:
            raise ValueError("--num_epochs must be greater than zero.")
        config["num_epochs"] = float(num_epochs_override)
    if params.batch_size is not None:
        if params.batch_size <= 0:
            raise ValueError("--batch_size must be greater than zero.")
        config["batch_size"] = int(params.batch_size)
    if params.gradient_accumulation_steps is not None:
        if params.gradient_accumulation_steps <= 0:
            raise ValueError(
                "--gradient_accumulation_steps must be greater than zero."
            )
        config["gradient_accumulation_steps"] = int(
            params.gradient_accumulation_steps
        )
    config = normalize_dataset_config(config)
    if params.datasets is not None:
        config["dataset"] = list(params.datasets)
        config = normalize_dataset_config(config)
    fft_norm = str(config.get("fft_norm", "forward"))
    spectral_filter_cutoff = float(config.get("spectral_filter_cutoff", 0.25))
    spectral_mixer_init_scale = float(
        config.get("spectral_mixer_init_scale", 0.02)
    )
    spectral_region_gate_hidden = int(
        config.get("spectral_region_gate_hidden", 64)
    )
    spectral_gate_temperature = float(
        config.get("spectral_gate_temperature", 1.0)
    )
    spectral_gate_floor = float(config.get("spectral_gate_floor", 0.25))
    spectral_gate_prior_weight = float(
        config.get("spectral_gate_prior_weight", 1.0)
    )
    use_absolute_embeddings = bool(
        config.get("use_absolute_embeddings", False)
        if params.use_absolute_embeddings is None
        else params.use_absolute_embeddings
    )
    config["use_absolute_embeddings"] = use_absolute_embeddings
    use_hf_loss = bool(config.get("use_hf_loss", True))
    hf_loss_lambda = float(config.get("hf_loss_lambda", 0.3))
    hf_loss_alpha = float(config.get("hf_loss_alpha", 1.5))
    train_eval_set_kwargs = build_train_eval_set_kwargs(config, params)
    train_dataset = get_dataset(
        dataset=config["dataset"],
        which="train",
        num_trajectories=config["num_trajectories"],
        data_path=params.data_path,
        **train_eval_set_kwargs,
    )
    eval_dataset = get_dataset(
        dataset=config["dataset"],
        which="val",
        num_trajectories=config.get(
            "num_validation_samples", config["num_trajectories"]
        ),
        data_path=params.data_path,
        **train_eval_set_kwargs,
    )
    train_dataset = drop_zero_time_pairs(train_dataset)
    eval_dataset = drop_zero_time_pairs(eval_dataset)

    config["effective_train_set_size"] = len(train_dataset)
    metadata_dataset = unwrap_filtered_dataset(train_dataset)
    if not isinstance(metadata_dataset, torch.utils.data.ConcatDataset):
        resolution = metadata_dataset.resolution
        input_dim = metadata_dataset.input_dim
        output_dim = metadata_dataset.output_dim
        channel_slice_list = metadata_dataset.channel_slice_list
        printable_channel_description = metadata_dataset.printable_channel_description
    else:
        resolution = metadata_dataset.datasets[0].resolution
        input_dim = metadata_dataset.datasets[0].input_dim
        output_dim = metadata_dataset.datasets[0].output_dim
        channel_slice_list = metadata_dataset.datasets[0].channel_slice_list
        printable_channel_description = metadata_dataset.datasets[
            0
        ].printable_channel_description

    model_config = ScOTConfig(
            image_size=resolution,
            patch_size=config["patch_size"],
            num_channels=input_dim,
            num_out_channels=output_dim,
            embed_dim=config["embed_dim"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            skip_connections=config["skip_connections"],
            window_size=config["window_size"],
            mlp_ratio=config["mlp_ratio"],
            qkv_bias=True,
            hidden_dropout_prob=0.0,  # default
            attention_probs_dropout_prob=0.0,  # default
            drop_path_rate=0.0,
            hidden_act="gelu",
            use_absolute_embeddings=use_absolute_embeddings,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            p=1,
            channel_slice_list_normalized_loss=channel_slice_list,
            fft_norm=fft_norm,
            spectral_filter_cutoff=spectral_filter_cutoff,
            spectral_mixer_init_scale=spectral_mixer_init_scale,
            spectral_region_gate_hidden=spectral_region_gate_hidden,
            spectral_gate_temperature=spectral_gate_temperature,
            spectral_gate_floor=spectral_gate_floor,
            spectral_gate_prior_weight=spectral_gate_prior_weight,
            use_hf_loss=use_hf_loss,
            hf_loss_lambda=hf_loss_lambda,
            hf_loss_alpha=hf_loss_alpha,
    )

    train_config = TrainingArguments(
        output_dir=ckpt_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(
            config.get("gradient_accumulation_steps", 1)
        ),
        eval_accumulation_steps=16,
        max_grad_norm=float(config["max_grad_norm"]),
        num_train_epochs=float(config["num_epochs"]),
        optim="adamw_torch",
        learning_rate=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        adam_beta1=0.9,  # default
        adam_beta2=0.999,  # default
        adam_epsilon=1e-8,  # default
        lr_scheduler_type=config["lr_scheduler"],
        warmup_ratio=float(config["warmup_ratio"]),
        log_level="passive",
        logging_strategy="steps",
        logging_steps=5,
        logging_dir=str(Path(ckpt_dir) / "runs"),
        logging_nan_inf_filter=False,
        save_strategy="epoch",
        save_total_limit=1,
        seed=SEED,
        fp16=False,
        dataloader_num_workers=CPU_CORES,
        dataloader_drop_last=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        auto_find_batch_size=False,
        full_determinism=False,
        torch_compile=False,
        no_cuda=False,
        report_to=params.report_to if params.report_to != "none" else [],
        run_name=params.run_name,
    )

    model = ScOT(model_config)
    model.config.training_datasets = list(config["dataset"])
    architecture = {
        "model_identity": {
            "short_name": "DRESO",
            "full_name": (
                "Evidence-Guided Dual-Region Spectral Operators for "
                "Long-Horizon PDE Forecasting"
            ),
        },
        "capacity": {
            "preset": config["model_name"],
            "embed_dim": config["embed_dim"],
            "depths": config["depths"],
            "skip_connections": config["skip_connections"],
        },
        "input": {
            "frames": 1,
            "absolute_position_embeddings": use_absolute_embeddings,
            "position_embedding_domain": "patch_grid",
            "cross_resolution_policy": (
                "bilinear_interpolation"
                if use_absolute_embeddings
                else "not_applicable"
            ),
        },
        "spectral_operator": {
            "partition": "learnable_gaussian_dual_region",
            "cutoff_initialization": spectral_filter_cutoff,
            "cutoff_learnable": True,
            "mode_support": "all",
            "mixer_initialization": "identity_perturbed",
            "mixer_initialization_scale": spectral_mixer_init_scale,
            "full_spectrum_identity": True,
            "residual_contract": "outer_correction",
            "router": "normalized_10d_signal_evidence",
            "router_hidden": spectral_region_gate_hidden,
            "gate_allocation": "softmax",
            "gate_temperature": spectral_gate_temperature,
            "gate_floor": spectral_gate_floor,
            "evidence_prior_weight": spectral_gate_prior_weight,
        },
        "high_frequency_loss": {
            "enabled": use_hf_loss,
            "weight": hf_loss_lambda,
            "alpha": hf_loss_alpha,
        },
        "training_objective": {
            "base": "masked_normalized_relative_l1_channel_group",
            "best_model_metric": "eval_loss",
            "best_model_metric_definition": "validation total loss",
            "best_model_includes_hf_loss": use_hf_loss and hf_loss_lambda > 0,
        },
    }
    if RANK == 0 or RANK == -1:
        architecture_path = write_architecture_config(ckpt_dir, architecture)
        print(f"Architecture config: {architecture_path}")
    num_params = get_num_parameters(model)
    config["num_params"] = num_params
    num_params_no_embed = get_num_parameters_no_embed(model)
    config["num_params_wout_embed"] = num_params_no_embed
    if RANK == 0 or RANK == -1:
        print(
            "Model identity: DRESO (Evidence-Guided Dual-Region Spectral "
            "Operators for Long-Horizon PDE Forecasting)"
        )
        print(f"Model size: {num_params:,} ({num_params / 1e6:.2f}M parameters)")
        print(
            "Model size without embeddings: "
            f"{num_params_no_embed:,} ({num_params_no_embed / 1e6:.2f}M parameters)"
        )
        spectral_blocks = [
            module
            for module in model.modules()
            if isinstance(module, DualRegionEnergySpectralBlock)
        ]
        spectral_params = sum(
            parameter.numel()
            for block in spectral_blocks
            for parameter in block.parameters()
        )
        spectral_mixer_params = sum(
            parameter.numel()
            for block in spectral_blocks
            for name, parameter in block.named_parameters(recurse=False)
            if name
            in {
                "low_weight_real",
                "low_weight_imag",
                "high_weight_real",
                "high_weight_imag",
            }
        )
        base_resolution = (
            tuple(int(value) for value in resolution)
            if isinstance(resolution, (tuple, list))
            else (int(resolution), int(resolution))
        )
        stage_shapes = [
            {
                "grid": tuple(value // (2**stage) for value in base_resolution),
                "channels": config["embed_dim"] * (2**stage),
                "blocks_per_encoder_or_decoder": depth,
            }
            for stage, depth in enumerate(config["depths"])
        ]
        print(
            "Architecture audit: "
            f"model={config['model_name']}, depths={config['depths']}, "
            "input_frames=1, operator_norm=layer, "
            f"absolute_position_embeddings={use_absolute_embeddings}, "
            f"spectral_blocks={len(spectral_blocks)}, "
            "partition=learnable_gaussian, mode_support=all, "
            "spectral_gate_design=normalized_10d_signal_evidence, "
            "spectral_gate_allocation=softmax, "
            f"spectral_block_type="
            f"{spectral_blocks[0].__class__.__name__ if spectral_blocks else 'none'}, "
            f"spectral_parameters={spectral_params}, "
            f"spectral_mixer_parameters={spectral_mixer_params}, "
            f"stages={stage_shapes}"
        )
        print(
            "Loss audit: incompressible Poseidon NS supervises channels "
            "[1, 2] only; compressible Euler supervises all physical channels. "
            f"HF loss enabled={use_hf_loss}, report_to={params.report_to}, "
            f"tensorboard_dir={Path(ckpt_dir) / 'runs'}"
        )

    def compute_metrics(eval_preds):
        channel_list = channel_slice_list
        predictions = eval_preds.predictions
        if isinstance(predictions, (tuple, list)):
            candidates = [value for value in predictions if isinstance(value, np.ndarray)]
            matching = [
                value for value in candidates if value.shape == eval_preds.label_ids.shape
            ]
            if not matching:
                raise ValueError(
                    "Could not identify the prediction tensor in Trainer tuple output."
                )
            predictions = matching[0]

        def get_statistics(errors):
            median_error = np.median(errors, axis=0)
            mean_error = np.mean(errors, axis=0)
            std_error = np.std(errors, axis=0)
            min_error = np.min(errors, axis=0)
            max_error = np.max(errors, axis=0)
            return {
                "median_relative_l1_error": median_error,
                "mean_relative_l1_error": mean_error,
                "std_relative_l1_error": std_error,
                "min_relative_l1_error": min_error,
                "max_relative_l1_error": max_error,
            }

        error_statistics = [
            get_statistics(
                relative_lp_error(
                    predictions[:, channel_list[i] : channel_list[i + 1]],
                    eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                    p=1,
                    return_percent=True,
                )
            )
            for i in range(len(channel_list) - 1)
        ]

        if output_dim == 1:
            error_statistics = error_statistics[0]
            return error_statistics
        else:
            mean_over_means = np.mean(
                np.array(
                    [stats["mean_relative_l1_error"] for stats in error_statistics]
                ),
                axis=0,
            )
            mean_over_medians = np.mean(
                np.array(
                    [stats["median_relative_l1_error"] for stats in error_statistics]
                ),
                axis=0,
            )
            error_statistics_ = {
                "mean_relative_l1_error": mean_over_means,
                "mean_over_median_relative_l1_error": mean_over_medians,
            }
            for i, stats in enumerate(error_statistics):
                for key, value in stats.items():
                    error_statistics_[printable_channel_description[i] + "/" + key] = (
                        value
                    )
            return error_statistics_

    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[],
    )

    trainer_device = torch.device(trainer.args.device)
    if RANK == 0 or RANK == -1:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
        cuda_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        distributed_type = (
            "torch.distributed"
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else "none"
        )
        print(
            "Runtime device audit: "
            f"trainer_device={trainer_device}, "
            f"distributed_type={distributed_type}, "
            f"num_processes={world_size}, "
            f"CUDA_VISIBLE_DEVICES={visible_devices}, "
            f"torch_cuda={torch.version.cuda}, "
            f"cuda_available={torch.cuda.is_available()}, "
            f"visible_cuda_count={torch.cuda.device_count()}, "
            f"visible_cuda_names={cuda_names}, "
            f"ACCELERATE_USE_CPU={os.environ.get('ACCELERATE_USE_CPU', '<not set>')}"
        )
    if trainer_device.type != "cuda":
        raise RuntimeError(
            "Training resolved to CPU. Check ACCELERATE_USE_CPU, "
            "CUDA_VISIBLE_DEVICES, the PyTorch CUDA build, and the NVIDIA "
            "driver/runtime compatibility."
        )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(train_config.output_dir)
