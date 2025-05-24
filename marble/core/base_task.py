# marble/core/base_task.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection

from marble.modules.ema import LitEma


class BaseTask(LightningModule, ABC):
    """
    Base Task class to encapsulate encoder-decoder models with:
      - support for multiple embedding transforms
      - multiple decoders (multi‐head)
      - multiple loss functions
      - split‐specific MetricCollections
      - optional EMA on encoder weights
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        emb_transforms: list[nn.Module] | None = None,
        decoders: list[nn.Module] | None = None,
        losses: list[nn.Module] | None = None,
        metrics: dict[str, dict[str, nn.Module]] | None = None,
        sample_rate: int | None = None,
        use_ema: bool = False,
        **kwargs,
    ):
        super().__init__()
        # save all args passed to init (for LightningCLI, checkpointing, etc.)
        self.save_hyperparameters(ignore=['encoder', 'emb_transforms', 'decoders', 'losses', 'metrics'])

        # core modules
        self.encoder = encoder
        self.emb_transforms = nn.ModuleList(emb_transforms or [])
        self.decoders = nn.ModuleList(decoders or [])
        self.loss_fns = nn.ModuleList(losses or [])

        # optional EMA on encoder parameters
        self.use_ema = use_ema
        if self.use_ema:
            self.ema = LitEma(self.encoder)

        # build and register metrics per split
        if metrics:
            for split in ('train', 'val', 'test'):
                split_cfg = metrics.get(split)
                if split_cfg:
                    mc = MetricCollection(
                        {name: m for name, m in split_cfg.items()},
                        prefix=f"{split}/"
                    )
                    setattr(self, f"{split}_metrics", mc)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """
        Default forward: encoder → transforms → each decoder head.
        Returns single Tensor if only one head, else list of Tensors.
        """
        h = self.encoder(x)
        for t in self.emb_transforms:
            h = t(h)
        outputs = [dec(h) for dec in self.decoders]
        return outputs[0] if len(outputs) == 1 else outputs

    def _shared_step(self, batch, batch_idx: int, split: str) -> torch.Tensor:
        """
        Common logic for train/val:
          - unpack batch
          - forward
          - sum all loss_fns
          - log loss and metrics
        """
        x, y = batch[:2]
        logits = self(x)

        # compute and log loss
        losses = [fn(logits, y) for fn in self.loss_fns]
        loss = sum(losses)
        self.log(f"{split}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # compute and log metrics
        mc: MetricCollection = getattr(self, f"{split}_metrics", None)
        if mc is not None:
            metrics_out = mc(logits, y)
            self.log_dict(metrics_out, prog_bar=(split == "val"), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, batch_idx, "val")

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0) -> None:
        if self.use_ema:
            self.ema.update()

    def test_step(self, batch, batch_idx: int):
        """
        Default test: returns raw logits and labels for aggregation.
        Override in subclass for custom behavior.
        """
        x, y = batch[:2]
        logits = self(x)
        return {"logits": logits, "labels": y}

    def configure_optimizers(self):
        # delegate to LightningCLI / super if using CLI
        return super().configure_optimizers()
