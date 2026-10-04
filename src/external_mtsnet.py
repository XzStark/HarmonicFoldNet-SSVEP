from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


MTSNET_UPSTREAM = "https://github.com/lanzhen19/MTSNet"
MTSNET_COMMIT = "890b0a4f93c3affd50741a1f1d036fb74a30a062"


def complex_spectrum_560(x: torch.Tensor, *, normalization: str = "none") -> torch.Tensor:
    """Build the 560-real-value representation implied by the author example.

    A 560-point real FFT contains 281 complex bins. Removing DC leaves 280
    positive-frequency complex bins; concatenating real and imaginary parts
    produces the 560 real values expected by the released model.
    """
    if x.ndim != 3:
        raise ValueError("MTSNet input must have shape [batch, channels, time]")
    spectrum = torch.fft.rfft(x, n=560, dim=-1)[..., 1:]
    features = torch.cat((spectrum.real, spectrum.imag), dim=-1)
    if features.shape[-1] != 560:
        raise RuntimeError(f"unexpected spectral feature length {features.shape[-1]}")
    if normalization == "channel_zscore":
        features = (features - features.mean(dim=-1, keepdim=True)) / features.std(
            dim=-1, keepdim=True, correction=0,
        ).clamp_min(1e-6)
    elif normalization != "none":
        raise ValueError(f"unknown spectral normalization: {normalization}")
    return features


def _load_author_class(source: Path):
    if not source.exists():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("external_mtsnet_author_code", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import external MTSNet source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "ViT_MTFNet"):
        raise RuntimeError(f"external source does not define ViT_MTFNet: {source}")
    return module.ViT_MTFNet


class ExternalMTSNetAdapter(nn.Module):
    """Research-only adapter around the authors' unmodified external source.

    The upstream repository currently contains no license. This adapter does
    not copy its implementation and must not be used to redistribute that
    source. It exists only for local scientific comparison with provenance.
    """

    def __init__(
        self,
        *,
        source: str | Path,
        time_points: int,
        channels: int,
        classes: int,
        depth_local: int = 2,
        depth_fusion: int = 2,
        kernel_length: int = 31,
        dropout: float = 0.5,
        spectral_normalization: str = "none",
    ) -> None:
        super().__init__()
        self.source = str(Path(source).resolve())
        self.time_points = int(time_points)
        self.spectral_normalization = spectral_normalization
        author_class = _load_author_class(Path(source))
        self.model = author_class(
            depth_L=int(depth_local),
            depth_M=int(depth_fusion),
            time_points=self.time_points,
            frequency_components=560,
            attention_kernal_length=int(kernel_length),
            chs_num=int(channels),
            class_num=int(classes),
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.time_points:
            raise ValueError(
                f"MTSNet was built for {self.time_points} samples, got {x.shape[-1]}"
            )
        spectrum = complex_spectrum_560(
            x, normalization=self.spectral_normalization,
        )
        return self.model(x, spectrum)
