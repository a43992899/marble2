# marble/tasks/GS/probe.py
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
import mir_eval
import torchmetrics
from torchmetrics import Metric, MetricCollection

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config
from marble.tasks.GS.datamodule import _GSAudioBase


class ProbeAudioTask(BaseTask):
    """
    GS genre probe task. Inherits training/val logic, multi-head,
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

    def test_step(self, batch, batch_idx):
        x, labels, ori_uids = batch
        logits = self(x)

        # Update all test metrics: accuracy, weighted_score, etc.
        if hasattr(self, "test_metrics"):
            for metric_name, metric in self.test_metrics.items():
                # Update each metric with its respective data (logits, labels, UIDs)
                if isinstance(metric, torchmetrics.Metric):
                    if metric_name == "weighted_score":
                        # For KeyWeightedScore, we pass logits, labels, and UIDs
                        metric.update(logits, labels, ori_uids)
                    else:
                        metric.update(logits, labels)

    def on_test_epoch_end(self) -> None:
        # Instead of manually aggregating the logits, rely on the metric to compute it
        # Compute the final test metrics, including KeyWeightedScore
        mc: MetricCollection = getattr(self, "test_metrics", None)
        if mc is not None:
            metrics_out = mc.compute()  # Metrics will be computed at the UID level internally
            self.log_dict(metrics_out, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)



class KeyWeightedScore(Metric):
    """
    Key weighted score for key estimation using logits.
    Uses mir_eval to compute the weighted score for key estimation.
    """
    IDX2LABEL = _GSAudioBase.IDX2LABEL
    LABEL2IDX = _GSAudioBase.LABEL2IDX
    
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("uids", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, labels: torch.Tensor, uids: torch.Tensor = None):
        """
        Update the metric with predictions, ground truth labels, and file-level UID.
        If uids is None, no aggregation will be performed.
        Ensure all tensors are on the same device.
        """
        device = preds.device  # Use the device of preds to ensure consistency
        self.preds.append(preds.to(device))
        self.labels.append(labels.to(device))
        
        if uids is not None:
            self.uids.append(uids.to(device))
        else:
            # If uids is None, no aggregation will be done
            self.uids.append(None)

    def compute(self):
        """
        Compute the weighted score using mir_eval after aggregating predictions at the UID level.
        If uids is None, compute the score directly on the predictions and labels without aggregation.
        """
        preds = torch.cat(self.preds, dim=0)
        labels = torch.cat(self.labels, dim=0)
        uids = self.uids

        # Ensure everything is on the same device as the model (GPU or CPU)
        device = preds.device
        preds = preds.to(device)
        labels = labels.to(device)

        if uids[0] is not None:
            # If uids is not None, aggregate predictions and labels at the UID level
            uids = torch.cat(uids, dim=0)
            unique_uids = torch.unique(uids)
            aggregated_preds = []
            aggregated_labels = []

            for uid in unique_uids:
                # Get the indices of the current UID
                indices = (uids == uid).nonzero(as_tuple=True)[0]
                # Aggregate logits by averaging the slices for each UID
                uid_preds = preds[indices].mean(dim=0)
                uid_labels = labels[indices][0]  # Assuming labels are the same for all slices of a file

                aggregated_preds.append(uid_preds)
                aggregated_labels.append(uid_labels)

            # Convert lists to tensors
            aggregated_preds = torch.stack(aggregated_preds).to(device)
            aggregated_labels = torch.stack(aggregated_labels).to(device)

        else:
            # If uids is None, no aggregation; compute directly on the logits and labels
            aggregated_preds = preds
            aggregated_labels = labels

        # Convert logits to predicted labels
        preds_label = torch.argmax(aggregated_preds, dim=-1)

        # Convert to numpy for mir_eval
        preds_label = preds_label.cpu().numpy()
        aggregated_labels = aggregated_labels.cpu().numpy()

        # Calculate the weighted score
        scores = [
            mir_eval.key.weighted_score(self.IDX2LABEL[ref_key], self.IDX2LABEL[est_key])
            for ref_key, est_key in zip(aggregated_labels, preds_label)
        ]
        return torch.tensor(np.mean(scores), device=device)

