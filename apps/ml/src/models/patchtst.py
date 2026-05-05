"""
PatchTST — Time-Series Transformer with channel-mixing patching.

Reference
---------
Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).
"A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers."
ICLR 2023.  arXiv:2211.14730.

Architectural variant — channel-mixing
--------------------------------------
The original paper introduces *channel-independent* PatchTST: each sensor
channel becomes its own sequence of patches, processed by the shared
Transformer, then the per-channel CLS embeddings are averaged.

We adopt the **channel-mixing** variant for this fault-classification task:
each patch token is the flattened ``(channel × patch_len)`` vector, so the
self-attention learns cross-channel interactions from the very first layer.

Justification (TCC defence point)
* Industrial fault signatures are inherently joint — an Air Leak manifests
  as TP2 falling *while* Motor_current rises.  Channel-independent
  processing would force the model to discover this signature only at the
  averaging stage; channel-mixing exposes it directly to attention.
* Channel-mixing also exports cleanly to ONNX (max abs diff ~2e-7),
  whereas the channel-independent ``(B*C, N, D)`` reshape triggers a
  PyTorch 2.3.x ONNX-tracing bug that yields ~0.4 divergence even with
  ``enable_nested_tensor=False`` and ``.contiguous()`` calls.

Architectural notes
-------------------
* Patching: unfold the time axis into overlapping patches of length
  ``patch_len`` with step ``stride``.  For T=60, patch_len=12, stride=6 →
  ``n_patches = 9``.
* Each patch (shape ``patch_len × n_channels``) is flattened and projected
  through a single ``nn.Linear`` to ``d_model``.
* A learnable [CLS] token is prepended; positional embedding added.
* TransformerEncoder over ``(n_patches + 1)`` tokens; the final CLS
  embedding feeds the binary classification head.

ONNX-export caveats
-------------------
* Requires opset 17+.
* ``enable_nested_tensor=False`` is REQUIRED on ``nn.TransformerEncoder``
  to avoid a PyTorch 2.3.x tracing bug (see comment at the constructor).
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryF1Score


class PatchTSTClassifier(pl.LightningModule):
    """Channel-mixing PatchTST classifier."""

    def __init__(
        self,
        n_channels: int,
        window_size: int = 60,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        pos_weight: float = 1.0,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.n_patches: int = (window_size - patch_len) // stride + 1

        # Channel-mixing projection: each patch token is the flattened
        # (n_channels × patch_len) vector, projected to d_model.
        self.proj: nn.Linear = nn.Linear(n_channels * patch_len, d_model)
        self.cls_token: nn.Parameter = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed: nn.Parameter = nn.Parameter(
            torch.randn(1, self.n_patches + 1, d_model) * 0.02
        )

        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # ``enable_nested_tensor=False`` is REQUIRED for ONNX export:
        # the default nested-tensor fast-path in PyTorch 2.3.x leaks an
        # ``is_causal`` Tensor into the trace, which silently produces a
        # broken ONNX graph (max abs diff ~0.4).  Disabling the fast-path
        # restores byte-equivalence (~1e-7).  Cost: a small inference-time
        # overhead that's irrelevant on our 60-step windows.
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.head: nn.Linear = nn.Linear(d_model, 2)

        self.register_buffer(
            "class_weights",
            torch.tensor([1.0, pos_weight], dtype=torch.float32),
        )

        self.val_f1: BinaryF1Score = BinaryF1Score(threshold=0.5)
        self.val_auc: BinaryAUROC = BinaryAUROC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) — unfold along the time axis directly.
        b: int = x.shape[0]

        # Patch the time dimension: (B, T, C) → (B, n_patches, C, patch_len)
        # ``unfold`` returns a view; the ``.contiguous()`` after reshape
        # avoids strided-tensor edge cases during ONNX tracing.
        patches: torch.Tensor = x.unfold(
            dimension=1,
            size=self.hparams.patch_len,
            step=self.hparams.stride,
        )
        b_, n_, c_, p_ = patches.shape
        # Flatten the (channel, patch_len) axes into a single feature vector
        # per token — this is the channel-mixing operation.
        patches = patches.contiguous().reshape(b_, n_, c_ * p_)

        tokens: torch.Tensor = self.proj(patches)  # (B, n_patches, d_model)

        cls: torch.Tensor = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        # ``self.pos_embed`` has shape (1, n_patches+1, d_model); broadcast
        # against (B, n_patches+1, d_model) avoids a dynamic Slice op.
        tokens = tokens + self.pos_embed

        encoded: torch.Tensor = self.encoder(tokens)  # (B, n_patches+1, d_model)
        # Explicit 3-axis slice ``[:, 0, :]`` forces the ONNX tracer to emit a
        # clean ``Gather`` op on dim 1.  The shorter ``[:, 0]`` form is parsed
        # via ``aten::select`` which can degrade into ``Squeeze`` + ``Reshape``
        # under PyTorch 2.3.x — that combination bakes in the dummy-input
        # batch size and crashes at inference with shape mismatches like
        # "Input shape:{10,8,64}, requested shape:{10,64}".
        cls_out: torch.Tensor = encoded[:, 0, :]  # (B, d_model)

        return self.head(cls_out)

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
