# marble/core/base_task.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection, Accuracy

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


class BaseFewShotTask(LightningModule, ABC):
    """
    Few-shot multiclass classification via nearest-centroid.

    Workflow in each epoch
    ──────────────────────
    1. `training_step` collects embeddings & labels for all train batches.
    2. `on_validation_epoch_start` computes one centroid per class.
    3. Validation/test steps assign each example to the nearest centroid.
    """

    # ──────────────────────────────────────────────────────────────
    # helpers
    @staticmethod
    def _to_label_indices(y: torch.Tensor) -> torch.Tensor:
        """
        Convert `y` to shape (N,) of dtype long.

        Accepts:
          • (N,)             already indices
          • (N, 1)           unsqueezed indices
          • (N, C, …)        one-hot (argmax over dim=1)
        """
        if y.ndim == 2 and y.size(1) == 1:        # (N,1)
            y = y.squeeze(1)
        elif y.ndim >= 2:                         # (N,C,…)
            y = y.argmax(dim=1)
        return y.long().view(-1)

    # ──────────────────────────────────────────────────────────────
    # init / forward
    def __init__(
        self,
        sample_rate: int,
        num_classes: int,
        encoder,
        emb_transforms,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "emb_transforms"])

        # build the encoder + any post-embedding transforms
        self.encoder = encoder
        self.emb_transforms = nn.ModuleList(emb_transforms or [])

        self.sample_rate = sample_rate
        self.num_classes = num_classes

        # metrics
        self.val_accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.test_accuracy = Accuracy(num_classes=num_classes, task="multiclass")

        # accumulators for one epoch of training
        self._train_embeddings: list[torch.Tensor] = []
        self._train_labels: list[torch.Tensor] = []

        # centroids buffer (will be filled each epoch)
        self.register_buffer("class_centroids", torch.empty(0))

        # no optimisation steps needed (nearest-centroid is non-parametric)
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for tf in self.emb_transforms:
            h = tf(h)
        h = h.view(h.size(0), -1)  # flatten if needed
        return h

    # ──────────────────────────────────────────────────────────────
    # training: collect embeddings & labels
    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        y = self._to_label_indices(y)

        emb = self(x)
        self._train_embeddings.append(emb.detach())
        self._train_labels.append(y.detach())

        return None  # no optimiser step

    # ──────────────────────────────────────────────────────────────
    # compute centroids right before validation starts
    def on_validation_epoch_start(self) -> None:
        # skip if nothing collected (can happen *before* first training batch)
        if not self._train_embeddings:
            return

        embs = torch.cat(self._train_embeddings, dim=0)   # (N, D)
        labels = torch.cat(self._train_labels, dim=0)     # (N,)

        classes = torch.unique(labels).sort()[0]
        centroids = torch.stack(
            [embs[labels == c].mean(dim=0) for c in classes], dim=0
        )  # (C, D)

        self.class_centroids = centroids.to(self.device)

        # clear for the next epoch
        self._train_embeddings.clear()
        self._train_labels.clear()

    # ──────────────────────────────────────────────────────────────
    # nearest-centroid classification
    @staticmethod
    def _nearest(emb: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Return index of closest centroid for each embedding.
        emb       : (B, D)
        centroids : (C, D)
        → (B,)
        """
        dists = torch.norm(
            emb.unsqueeze(1) - centroids.unsqueeze(0), dim=2
        )  # (B, C)
        return dists.argmin(dim=1)

    # ──────────────────────────────────────────────────────────────
    # validation / test
    def validation_step(self, batch, batch_idx):
        if self.class_centroids.numel() == 0:
            raise RuntimeError(
                "Centroids empty – ensure `on_validation_epoch_start` has run."
            )

        x, y = batch[:2]
        y = self._to_label_indices(y)  # Ensure y is of shape (B,) - class indices

        embs = self(x)  # Get embeddings from the model
        preds = self._nearest(embs, self.class_centroids)

        # Log accuracy using one-hot encoded preds and class index y
        self.log(
            "val/acc",
            self.val_accuracy(preds.float(), y),  # Use class index labels, not one-hot y
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        
        
    def test_step(self, batch, batch_idx):
        if self.class_centroids.numel() == 0:
            # Compute centroids using the training data loader (mimic valid's approach)
            train_loader = self.trainer.datamodule.train_dataloader()  # Assuming you have a DataModule
            self._train_embeddings.clear()
            self._train_labels.clear()
            
            # Collect embeddings and labels from the training data
            for batch in train_loader:
                x, y = batch[:2]
                y = self._to_label_indices(y)
                emb = self(x)
                self._train_embeddings.append(emb.detach())
                self._train_labels.append(y.detach())

            embs = torch.cat(self._train_embeddings, dim=0)  # (N, D)
            labels = torch.cat(self._train_labels, dim=0)  # (N,)

            classes = torch.unique(labels).sort()[0]
            centroids = torch.stack(
                [embs[labels == c].mean(dim=0) for c in classes], dim=0
            )  # (C, D)

            self.class_centroids = centroids.to(self.device)
            self._train_embeddings.clear()
            self._train_labels.clear()

        x, y = batch[:2]
        y = self._to_label_indices(y)
        preds = self._nearest(self(x), self.class_centroids)

        # Ensure the metric is on the correct device (same as the model output)
        self.test_accuracy = self.test_accuracy.to(self.device)

        # Log accuracy using class indices for both preds and y, ensuring both are on the same device
        self.log(
            "test/acc",
            self.test_accuracy(preds.to(self.device), y.to(self.device)),  # Move both to the same device
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
    
    def configure_optimizers(self):
        # No optimizers needed for nearest-centroid, but required by Lightning
        return []
        