"""Utility functions."""


def read_cli(parser):
    """Reads command line arguments."""

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="run",
        help="Checkpoint run-directory name.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="dreso",
        help="Checkpoint project-directory name.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "none"],
        help="Training logging backend.",
    )
    parser.add_argument(
        "--max_num_train_time_steps",
        type=int,
        default=None,
        help="Maximum number of time steps to use for training and validation.",
    )
    parser.add_argument(
        "--train_time_step_size",
        type=int,
        default=None,
        help="Time step size to use for training and validation.",
    )
    parser.add_argument(
        "--train_small_time_transition",
        action="store_true",
        help="Whether to train only for next step prediction.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Base path to data.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Root directory for project/run checkpoints.",
    )
    parser.add_argument(
        "--num_epochs",
        type=float,
        default=None,
        help="Override num_epochs from the config file for this training stage.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override the per-GPU train/eval batch size from the config file.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Override gradient accumulation while keeping the physical batch size fixed.",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Whether to disable tqdm progress bar",
    )
    return parser


def get_num_parameters(model):
    """Returns the number of trainable parameters in a model."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_num_parameters_no_embed(model):
    """Return trainable DRESO parameters outside embedding and recovery layers."""
    out = 0
    for name, p in model.named_parameters():
        if not ("embeddings" in name or "patch_recovery" in name) and p.requires_grad:
            out += p.numel()
    return out
