"""Neutral forecasting components used by the final Stage 2A checkpoint.

The implementations are deliberately small and independent.  They contain no
online update, replay, gating, spectral neural layer, or manuscript-specific
architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
import torch
from torch import nn

from .model import diagonal_average, polynomial_vandermonde


@dataclass(frozen=True)
class CausalScaler:
    """Scalar z-score fitted only on values supplied by the caller."""

    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "CausalScaler":
        array = np.asarray(values, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("scaler input contains NaN or Inf")
        scale = float(array.std())
        return cls(float(array.mean()), scale if scale > 1e-12 else 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.std + self.mean


class MovingAverage(nn.Module):
    """Centered moving average with the edge padding used by LTSF-Linear."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = int(kernel_size)
        self.pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("DLinear input must have shape [batch, time, channel]")
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        return self.pool(torch.cat([front, x, end], dim=1).permute(0, 2, 1)).permute(0, 2, 1)


class DLinear(nn.Module):
    """Minimal univariate DLinear: decomposition, two projections, and sum."""

    def __init__(self, lookback: int, horizon: int, kernel_size: int) -> None:
        super().__init__()
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.moving_average = MovingAverage(kernel_size)
        self.seasonal_projection = nn.Linear(self.lookback, self.horizon)
        self.trend_projection = nn.Linear(self.lookback, self.horizon)

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(x)
        return x - trend, trend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decompose(x)
        seasonal_out = self.seasonal_projection(seasonal.permute(0, 2, 1))
        trend_out = self.trend_projection(trend.permute(0, 2, 1))
        return (seasonal_out + trend_out).permute(0, 2, 1)


class ResidualMLP(nn.Module):
    """One generic fixed residual learner with no specialist mechanisms."""

    def __init__(self, lookback: int, horizon: int, hidden: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(lookback), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(horizon)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass(frozen=True)
class FittedTorchModel:
    model: nn.Module
    best_epoch: int
    train_loss: float
    validation_loss: float


def fit_torch_regressor(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 2026,
    epochs: int = 40,
    patience: int = 6,
    learning_rate: float = 1e-3,
) -> FittedTorchModel:
    """Deterministic chronological early stopping followed by a full-data refit."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    features = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    targets = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    if len(features) < 3 or len(features) != len(targets):
        raise ValueError("training requires at least three aligned samples")
    split = max(2, int(np.floor(len(features) * 0.8)))
    split = min(split, len(features) - 1)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features[:split]), targets[:split])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(features[split:]), targets[split:]).item())
        if val_loss < best_loss - 1e-10:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    # Refit for the selected epoch on every causally legal training sample.
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(best_epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_loss = float(loss_fn(model(features), targets).item())
    return FittedTorchModel(model, best_epoch, train_loss, best_loss)


@dataclass(frozen=True)
class HankelPolynomialResult:
    forecast: np.ndarray
    reconstructed_history: np.ndarray


@dataclass(frozen=True)
class HankelPolynomialExpert:
    """Rank-one Hankel reconstruction plus quadratic time extrapolation."""

    lookback: int
    horizon: int = 144
    hankel_rows: int = 144
    polynomial_degree: int = 2
    polynomial_ridge: float = 5.0

    def _projection(self) -> np.ndarray:
        tin = np.linspace(-1.0, 1.0, self.lookback)
        tout = np.linspace(
            1.0 + 2.0 / self.lookback,
            1.0 + 2.0 * self.horizon / self.lookback,
            self.horizon,
        )
        vin = polynomial_vandermonde(tin, self.polynomial_degree)
        vout = polynomial_vandermonde(tout, self.polynomial_degree)
        gram = vin.T @ vin + self.polynomial_ridge * np.eye(self.polynomial_degree + 1)
        return vout @ np.linalg.solve(gram, vin.T)

    def forecast(self, history: np.ndarray) -> HankelPolynomialResult:
        values = np.asarray(history, dtype=float).reshape(-1)
        if values.size != self.lookback or not np.isfinite(values).all():
            raise ValueError("history does not match the declared lookback")
        if not 2 <= self.hankel_rows < self.lookback:
            raise ValueError("invalid Hankel row count")
        center = float(values.mean())
        centered = values - center
        trajectory = np.lib.stride_tricks.sliding_window_view(centered, self.hankel_rows).T
        if min(trajectory.shape) > 1:
            left, _, _ = svds(
                trajectory,
                k=1,
                which="LM",
                solver="arpack",
                v0=np.ones(min(trajectory.shape), dtype=float),
            )
        else:
            left, _, _ = svd(trajectory, full_matrices=False, check_finite=False)
            left = left[:, :1]
        reconstructed = diagonal_average(left @ (left.T @ trajectory))
        forecast = np.maximum(self._projection() @ reconstructed + center, 0.0)
        return HankelPolynomialResult(forecast, reconstructed + center)


@dataclass(frozen=True)
class StructuralResidualHybrid:
    """Fixed additive combination of an analytical expert and residual MLP."""

    expert: HankelPolynomialExpert
    residual_model: ResidualMLP
    residual_scaler: CausalScaler

    def predict(self, history: np.ndarray) -> np.ndarray:
        analytical = self.expert.forecast(history)
        residual_history = history - analytical.reconstructed_history
        x = self.residual_scaler.transform(residual_history)[None, :]
        with torch.no_grad():
            residual_z = self.residual_model(torch.as_tensor(x, dtype=torch.float32)).numpy()[0]
        residual_kw = self.residual_scaler.inverse(residual_z)
        return np.maximum(analytical.forecast + residual_kw, 0.0)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
