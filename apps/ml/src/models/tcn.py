"""
Temporal Convolutional Network (TCN) — production candidate.

Reference
---------
Bai, S., Kolter, J. Z., & Koltun, V. (2018).  "An Empirical Evaluation of
Generic Convolutional and Recurrent Networks for Sequence Modeling."
arXiv:1803.01271.

Architectural notes
-------------------
* Six causal dilated 1D-convolutional blocks with dilation 1, 2, 4, 8, 16, 32.
  Receptive field = 1 + 2 × (k − 1) × Σdilations = 1 + 4 × 63 = 253 samples,
  well above the T = 60 input window (full history coverage).
* Each ``TemporalBlock`` is two Conv1D + BatchNorm + ReLU + Dropout layers
  with a residual connection.  Causality is enforced by trimming the right
  pad after each Conv1D.
* Global average pooling on the time axis collapses the convolutional
  features into a single vector — keeps the head agnostic to ``T``.
* Forward consumes ``(B, T, C)`` and transposes internally to ``(B, C, T)``
  for ``Conv1D``; this matches the contract expected by the DataModule.
* Output is raw logits (shape ``(B, 2)``) — softmax is applied at inference.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryF1Score


class TemporalBlock(nn.Module):
    """Two stacked causal dilated convolutions with a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Causal padding: pad only the left side after slicing the right tail.
        pad: int = (kernel_size - 1) * dilation
        self.conv1: nn.Conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
        )
        self.conv2: nn.Conv1d = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
        )
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(out_channels)
        self.bn2: nn.BatchNorm1d = nn.BatchNorm1d(out_channels)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        # 1×1 projection only when the channel count changes (first block).
        self.downsample: nn.Module = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.pad: int = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv1(x)[..., : -self.pad]
        out = F.relu(self.bn1(out))
        out = self.dropout(out)
        out = self.conv2(out)[..., : -self.pad]
        out = F.relu(self.bn2(out))
        out = self.dropout(out)
        return F.relu(out + self.downsample(x))


class TcnClassifier(pl.LightningModule):
    """
    Binary fault classifier built from stacked TemporalBlocks.

    Parameters
    ----------
    n_channels:
        Number of input sensor channels (12 for the ``raw`` feature set).
    hidden:
        Hidden channel width inside every TCN block.
    kernel_size:
        Conv1D kernel size.  Combined with the dilation schedule this
        determines the receptive field.
    n_blocks:
        Number of stacked TemporalBlocks.  Dilation doubles each block.
    dropout:
        Dropout probability inside each block.
    learning_rate:
        Initial AdamW learning rate; CosineAnnealingLR steps it down.
    pos_weight:
        Inverse-frequency weight for the positive class in CrossEntropyLoss.
        Mirrors the strategy used by ``train_mlp.py``.
    max_epochs:
        Required by the cosine scheduler — must equal the Trainer's value.
    """

    def __init__(
        self,
        n_channels: int,
        hidden: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 6,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        pos_weight: float = 1.0,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        blocks: list[nn.Module] = []
        in_ch: int = n_channels
        for i in range(n_blocks):
            blocks.append(TemporalBlock(in_ch, hidden, kernel_size, 2**i, dropout))
            in_ch = hidden
        self.tcn: nn.Sequential = nn.Sequential(*blocks)
        self.head: nn.Linear = nn.Linear(hidden, 2)

        self.register_buffer(
            "class_weights",
            torch.tensor([1.0, pos_weight], dtype=torch.float32),
        )

        self.val_f1: BinaryF1Score = BinaryF1Score(threshold=0.5)
        self.val_auc: BinaryAUROC = BinaryAUROC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, C, T) for Conv1D → (B, C') after GAP → (B, 2)
        h: torch.Tensor = self.tcn(x.transpose(1, 2))
        h = h.mean(dim=-1)
        return self.head(h)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits: torch.Tensor = self(x)
        loss: torch.Tensor = F.cross_entropy(logits, y, weight=self.class_weights)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits: torch.Tensor = self(x)
        loss: torch.Tensor = F.cross_entropy(logits, y, weight=self.class_weights)
        probs: torch.Tensor = torch.softmax(logits, dim=1)[:, 1]
        self.val_f1.update(probs, y)
        self.val_auc.update(probs, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_auc", self.val_auc.compute(), prog_bar=True)
        self.val_f1.reset()
        self.val_auc.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
        )
        scheduler: torch.optim.lr_scheduler.LRScheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=1e-6,
            )
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
