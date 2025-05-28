# marble/tasks/EMO/probe.py
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import Metric, R2Score, MetricCollection

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config

class ProbeAudioTask(BaseTask):
    """
    GTZAN genre probe task.  Inherits training/val logic, multi‐head,
    losses, metrics and EMA support from BaseTask.
    """

    def __init__(
        self,
        sample_rate: int,
        use_ema: bool,
        encoder: dict,
        emb_transforms: list[dict],
        decoders: list[dict],
        losses: list[dict],
        metrics: dict[str, dict[str, dict]],
    ):
        # 1) build all submodules from your YAML configs
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]

        # metrics comes in as nested dict: { split: { name: cfg, … }, … }
        metric_maps = {
            split: {
                name: instantiate_from_config(cfg)
                for name, cfg in metrics[split].items()
            }
            for split in ("train", "val", "test")
        }

        # 2) hand everything off to BaseTask
        super().__init__(
            encoder=enc,
            emb_transforms=tfs,
            decoders=decs,
            losses=loss_fns,
            metrics=metric_maps,
            sample_rate=sample_rate,
            use_ema=use_ema,
        )

    def on_test_start(self) -> None:
        # Initialize storage for per-slice test outputs
        self._test_file_outputs: list[dict] = []

    def test_step(self, batch, batch_idx):
        x, labels, file_paths = batch
        logits = self(x)

        # Store per-slice probabilities and labels for file-level aggregation
        for fp, logit, lb in zip(file_paths, logits, labels):
            self._test_file_outputs.append({
                "file_path": fp,
                "logit": logit.detach().to(torch.float32), # (C,)
                "label": lb.to(torch.float32) # (C,)
            })
            # print(self._test_file_outputs[-1]['logit'].shape, self._test_file_outputs[-1]['label'])

    def on_test_epoch_end(self) -> None:
        # Aggregate per-file predictions
        file_dict: dict[str, dict] = {}
        for entry in self._test_file_outputs:
            fp = entry["file_path"]
            info = file_dict.setdefault(fp, {"logits": [], "label": entry["label"]})
            info["logits"].append(entry["logit"])

        # aggregate logits and compute file-level metrics
        print(f"Aggregating {len(file_dict)} files with per-slice outputs")
        batched_logits = []
        batched_labels = []
        for fp, info in file_dict.items():
            arr = torch.stack(info["logits"])      # (n_slices, C)
            mean_logit = arr.mean(dim=0)             # (C,)
            batched_logits.append(mean_logit)
            batched_labels.append(info["label"])
        batched_logits = torch.stack(batched_logits)
        batched_labels = torch.stack(batched_labels)
        # compute metrics
        mc: MetricCollection = getattr(self, "test_metrics", None)
        if mc is not None:
            metrics_out = mc(batched_logits, batched_labels)
            self.log_dict(metrics_out, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            


class SliceR2(Metric):
    """
    对 2 维回归输出 y_pred[...,dim] / y[...,dim] 做 R2 评估。
    """
    def __init__(self, dim: int, **r2_kwargs):
        super().__init__()
        self.dim = dim
        # 负责具体计算的 R2Score
        self.inner = R2Score(**r2_kwargs)

    def update(self, preds, targets):
        # preds: (B,2)，targets: (B,2)
        self.inner.update(preds[:, self.dim], targets[:, self.dim])

    def compute(self):
        return self.inner.compute()
