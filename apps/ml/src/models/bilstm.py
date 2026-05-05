"""
Bidirectional LSTM with additive attention — academic baseline.

Reference
---------
* Hochreiter, S., & Schmidhuber, J. (1997).  "Long Short-Term Memory."
  Neural Computation.
* Bahdanau, D. et al. (2015).  "Neural Machine Translation by Jointly
  Learning to Align and Translate." — additive attention pooling.

Architectural notes
-------------------
* 2-layer bidirectional LSTM with hidden size 64; output is ``(B, T, 2H)``.
* Additive attention computes a soft pooling weight per timestep, producing
  a context vector ``(B, 2H)``.  This is the interpretability hook for the
  TCC: the attention weights highlight which seconds of the window drove
  the classification decision.
* During training BiLSTM uses both past and future inside the window —
  acceptable because the window itself is past-only at inference (the
  buffer never feeds the model future samples).
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryF1Score


class BiLstmClassifier(pl.LightningModule):
    """Binary classifier: BiLSTM encoder + additive attention + linear head."""

    def __init__(
        self,
        n_channels: int,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        pos_weight: float = 1.0,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # PyTorch only applies dropout *between* LSTM layers, so it's a no-op
        # when n_layers == 1; we still pass it for n_layers >= 2 cases.
        self.lstm: nn.LSTM = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn: nn.Linear = nn.Linear(hidden * 2, 1)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.head: nn.Linear = nn.Linear(hidden * 2, 2)

        self.register_buffer(
            "class_weights",
            torch.tensor([1.0, pos_weight], dtype=torch.float32),
        )

        self.val_f1: BinaryF1Score = BinaryF1Score(threshold=0.5)
        self.val_auc: BinaryAUROC = BinaryAUROC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C); LSTM output: (B, T, 2H)
        h, _ = self.lstm(x)
        # Attention scores: (B, T)
        scores: torch.Tensor = self.attn(h).squeeze(-1)
        weights: torch.Tensor = torch.softmax(scores, dim=1)
        # Weighted context vector: (B, 2H)
        context: torch.Tensor = (h * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(self.dropout(context))

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
