"""
Conv1D Autoencoder for unsupervised anomaly detection (MetroPT-3).

Principle
---------
The model is trained exclusively on *healthy* compressor windows (label=0).
It learns the manifold of normal operation.  At inference, a window from a
faulty period yields a high reconstruction error (MSE) because it falls
outside that manifold — the network cannot reconstruct signals it has never
been trained to reproduce.

Architecture
------------
Input  : (B, T, C) = (B, 60, 12) — transposed to (B, C, T) for Conv1D
Encoder:
  Block 1 → Conv1D(12,  32, k=5, stride=2, pad=2) + BN + GELU → (B, 32, 30)
  Block 2 → Conv1D(32,  64, k=5, stride=2, pad=2) + BN + GELU → (B, 64, 15)
  Block 3 → Conv1D(64, 128, k=5, stride=2, pad=2) + BN + GELU → (B, 128, 8)
Bottleneck: 128 channels × 8 timesteps (compression factor ≈ 9×)
Decoder (mirrored with ConvTranspose1D):
  Block 1 → ConvTranspose1D(128, 64, k=5, s=2, p=2, op=0) + BN + GELU → (B, 64, 15)
  Block 2 → ConvTranspose1D( 64, 32, k=5, s=2, p=2, op=1) + BN + GELU → (B, 32, 30)
  Block 3 → ConvTranspose1D( 32, 12, k=5, s=2, p=2, op=1)              → (B, 12, 60)
Output : transposed back to (B, T, C) = (B, 60, 12) — raw reconstruction, no clamp

Dimension derivation
--------------------
Encoder (L_out = floor((L_in + 2p - d*(k-1) - 1) / s + 1)):
  60 → 30 → 15 → 8  (all with k=5, s=2, p=2, d=1)
Decoder (L_out = (L_in-1)*s - 2p + d*(k-1) + op + 1):
  8→15 (op=0):  7*2 - 4 + 4 + 0 + 1 = 15  ✓
  15→30 (op=1): 14*2 - 4 + 4 + 1 + 1 = 30  ✓
  30→60 (op=1): 29*2 - 4 + 4 + 1 + 1 = 60  ✓

Training signal
---------------
MSE(reconstruction, input) on healthy-only windows from
MetroPTUnsupervisedDataModule (train split filtered to label=0).

Threshold calibration
---------------------
After training, `train_autoencoder.py` computes the 99th percentile of the
MSE distribution on *healthy* validation windows and stores it as
``mse_threshold`` in the model card.  The backend adapter applies a sigmoid
centred on that threshold so the ModelService sees a standard [0,1] score.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    """Strided Conv1D + BatchNorm + GELU — one encoder stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
    ) -> None:
        super().__init__()
        pad: int = kernel_size // 2
        self.net: nn.Sequential = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride=stride, padding=pad
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConvTransposeBlock(nn.Module):
    """ConvTranspose1D + BatchNorm + GELU — one decoder stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        output_padding: int = 0,
        activation: bool = True,
    ) -> None:
        super().__init__()
        pad: int = kernel_size // 2
        layers: list[nn.Module] = [
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=pad,
                output_padding=output_padding,
            ),
        ]
        if activation:
            layers += [nn.BatchNorm1d(out_channels), nn.GELU()]
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Autoencoder LightningModule
# ---------------------------------------------------------------------------


class Conv1DAutoencoder(pl.LightningModule):
    """
    Unsupervised 1D-CNN Autoencoder for compressor health monitoring.

    Parameters
    ----------
    n_channels:
        Number of sensor channels (C).  12 for the raw MetroPT-3 feature set.
    base_channels:
        Channel width of the first encoder block; doubles every subsequent block.
    kernel_size:
        Conv1D kernel size used by every encoder and decoder block.
    learning_rate:
        Initial AdamW learning rate — stepped down by CosineAnnealingLR.
    weight_decay:
        AdamW L2 regularisation coefficient.
    max_epochs:
        Required by the cosine scheduler; must equal the Trainer's value.
    """

    def __init__(
        self,
        n_channels: int = 12,
        base_channels: int = 32,
        kernel_size: int = 5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        c1: int = base_channels  # 32
        c2: int = base_channels * 2  # 64
        c3: int = base_channels * 4  # 128

        self.encoder: nn.Sequential = nn.Sequential(
            _ConvBlock(n_channels, c1, kernel_size),  # (B, 32, 30)
            _ConvBlock(c1, c2, kernel_size),  # (B, 64, 15)
            _ConvBlock(c2, c3, kernel_size),  # (B, 128, 8)
        )

        self.decoder: nn.Sequential = nn.Sequential(
            _ConvTransposeBlock(c3, c2, kernel_size, output_padding=0),  # (B, 64, 15)
            _ConvTransposeBlock(c2, c1, kernel_size, output_padding=1),  # (B, 32, 30)
            _ConvTransposeBlock(
                c1, n_channels, kernel_size, output_padding=1, activation=False
            ),  # (B, 12, 60)
        )

        # Buffers accumulate MSE values during each validation epoch.
        # Using lists on `self` (not registered buffers) is intentional —
        # they must be cleared each epoch and never serialised to state_dict.
        self._val_mse_healthy: list[float] = []
        self._val_mse_anomaly: list[float] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the input window.

        Parameters
        ----------
        x : torch.Tensor of shape (B, T, C)

        Returns
        -------
        torch.Tensor of shape (B, T, C) — the reconstruction.
        """
        # (B, T, C) → (B, C, T) for Conv1D
        h: torch.Tensor = x.transpose(1, 2)
        h = self.encoder(h)
        h = self.decoder(h)
        # (B, C, T) → (B, T, C)
        return h.transpose(1, 2)

    # ------------------------------------------------------------------
    # Lightning training hooks
    # ------------------------------------------------------------------

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch  # labels are all 0 (healthy-only training set)
        reconstruction: torch.Tensor = self(x)
        loss: torch.Tensor = F.mse_loss(reconstruction, x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_mse_healthy = []
        self._val_mse_anomaly = []

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        reconstruction: torch.Tensor = self(x)

        loss: torch.Tensor = F.mse_loss(reconstruction, x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Per-sample MSE: mean over T and C axes → shape (B,)
        per_sample_mse: torch.Tensor = F.mse_loss(
            reconstruction, x, reduction="none"
        ).mean(dim=(1, 2))

        healthy_mask: torch.Tensor = y == 0
        anomaly_mask: torch.Tensor = y == 1

        if healthy_mask.any():
            self._val_mse_healthy.extend(
                per_sample_mse[healthy_mask].detach().cpu().tolist()
            )
        if anomaly_mask.any():
            self._val_mse_anomaly.extend(
                per_sample_mse[anomaly_mask].detach().cpu().tolist()
            )

    def on_validation_epoch_end(self) -> None:
        if self._val_mse_healthy:
            import numpy as np

            healthy_arr = np.array(self._val_mse_healthy)
            self.log("val_mse_healthy_mean", float(healthy_arr.mean()), prog_bar=True)
            self.log("val_mse_healthy_p99", float(np.percentile(healthy_arr, 99)))

        if self._val_mse_anomaly:
            import numpy as np

            anomaly_arr = np.array(self._val_mse_anomaly)
            self.log("val_mse_anomaly_mean", float(anomaly_arr.mean()), prog_bar=False)

            if self._val_mse_healthy:
                # Separation ratio: how many σ above healthy mean is anomaly mean.
                healthy_arr = np.array(self._val_mse_healthy)
                sep = (anomaly_arr.mean() - healthy_arr.mean()) / (
                    healthy_arr.std() + 1e-8
                )
                self.log("val_mse_separation_sigma", float(sep), prog_bar=True)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
