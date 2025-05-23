# marble/core/base_task.py

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict
from lightning.pytorch import LightningModule
import torchmetrics

class BaseTask(LightningModule, ABC):
    """
    Base Task class to encapsulate model training, validation, and evaluation.
    It combines an encoder and a decoder and provides methods for training, validation, etc.
    
    Each specific task should extend this class and implement the necessary logic.
    """

    def __init__(self,
                 encoder,
                 decoders,
                 emb_transforms,
                 losses,
                 metrics,
                 sample_rate,
                 use_ema=False,
                 **kwargs):
        """
        Initializes the task with encoder, decoder, and metrics.

        Args:
            encoder: The encoder module (e.g., Wav2Vec2).
            decoder: The decoder module (e.g., MLP).
            metrics: Dictionary of metrics to evaluate the model (e.g., accuracy).
            kwargs: Additional arguments such as learning rate, optimizer, etc.
        """
        super().__init__()
        self.encoder = encoder
        self.emb_transforms = emb_transforms # list of modules
        self.emb_transforms = nn.ModuleList(emb_transforms)
        self.decoder = decoders
        self.metrics = torch.nn.ModuleDict(metrics)
        self.kwargs = kwargs

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the task. It combines the encoder and decoder.
        
        Args:
            x: The input tensor.
        
        Returns:
            torch.Tensor: The task output (e.g., logits, predictions).
        """
        # Pass through encoder
        x = self.encoder(x)
        
        # Apply transformations
        for transform in self.emb_transforms:
            x = transform(x)
        
        # Pass through decoder
        logits = self.decoder(x)
        
        return logits

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
            batch: A batch of input data.
            batch_idx: The batch index.
        
        Returns:
            torch.Tensor: The loss for this batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Args:
            batch: A batch of input data.
            batch_idx: The batch index.
        
        Returns:
            Dict[str, torch.Tensor]: The validation metrics.
        """
        x, y = batch
        logits = self(x)
        for name, metric in self.metrics.items():
            self.log(f"val/{name}", metric(logits, y), prog_bar=True)
        return {"val/loss": self.loss_fn(logits, y)}

    def loss_fn(self, logits, labels):
        """
        Defines the loss function.
        
        Args:
            logits: The model's output.
            labels: The ground truth labels.
        
        Returns:
            torch.Tensor: The computed loss.
        """
        return NotImplementedError("Loss function not implemented. Please implement in subclass.")
