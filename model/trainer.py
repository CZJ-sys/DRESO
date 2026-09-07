"""Minimal Hugging Face Trainer integration for DRESO."""

from transformers import Trainer as HuggingFaceTrainer
from transformers import TrainingArguments


SPECTRAL_TRAINING_METRICS = (
    ("spectral_cutoff", "spectral/cutoff"),
    ("spectral_low_coverage", "spectral/low_coverage"),
    ("spectral_high_coverage", "spectral/high_coverage"),
    ("spectral_low_density", "spectral/low_density"),
    ("spectral_high_density", "spectral/high_density"),
    ("spectral_low_radius", "spectral/low_radius"),
    ("spectral_high_radius", "spectral/high_radius"),
    ("spectral_overlap_density", "spectral/overlap_density"),
    ("spectral_region_gate_mean", "spectral/region_gate_mean"),
    ("spectral_region_gate_std", "spectral/region_gate_std"),
    ("spectral_region_gate_contrast", "spectral/region_gate_contrast"),
    ("spectral_region_gate_entropy", "spectral/region_gate_entropy"),
    (
        "spectral_region_gate_instance_std",
        "spectral/region_gate_instance_std",
    ),
)


class Trainer(HuggingFaceTrainer):
    """Trainer that adds DRESO component metrics to ordinary HF training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._component_metric_sums = {}
        self._component_metric_counts = {}

    def log(self, logs):
        if self._component_metric_sums:
            component_metrics = {
                name: total / self._component_metric_counts[name]
                for name, total in self._component_metric_sums.items()
            }
            logs = {**logs, **component_metrics}
            self._component_metric_sums.clear()
            self._component_metric_counts.clear()
        return super().log(logs)

    def _record_component_metrics(self, metrics):
        for name, value in metrics.items():
            self._component_metric_sums[name] = (
                self._component_metric_sums.get(name, 0.0) + value
            )
            self._component_metric_counts[name] = (
                self._component_metric_counts.get(name, 0) + 1
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs[0]
        if loss is None:
            keys = list(outputs.keys()) if isinstance(outputs, dict) else []
            raise ValueError(f"DRESO did not return a training loss; output keys={keys}.")

        if model.training:
            metrics = {"loss/total": float(loss.detach())}
            base_loss = outputs.get("base_loss") if isinstance(outputs, dict) else None
            if base_loss is not None:
                metrics["loss/base"] = float(base_loss.detach())
            for output_name, metric_name in SPECTRAL_TRAINING_METRICS:
                value = (
                    outputs.get(output_name)
                    if isinstance(outputs, dict)
                    else getattr(outputs, output_name, None)
                )
                if value is not None:
                    metrics[metric_name] = float(value.detach())
            self._record_component_metrics(metrics)

        return (loss, outputs) if return_outputs else loss
