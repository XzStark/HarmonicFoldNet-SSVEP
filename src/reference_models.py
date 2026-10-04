from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EEGNetAdaptive(nn.Module):
    """EEGNet-8,2 style reference with a fixed adaptive output head.

    The convolutional blocks follow EEGNet. Adaptive pooling is the disclosed
    modification that permits one trained model to accept every registered
    stimulation duration.
    """

    def __init__(
        self, channels: int, classes: int, *, temporal_kernel: int = 64,
        f1: int = 8, depth_multiplier: int = 2, f2: int = 16, dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(
            1, f1, (1, temporal_kernel), padding="same", bias=False,
        )
        self.temporal_norm = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(
            f1, f1 * depth_multiplier, (channels, 1),
            groups=f1, bias=False,
        )
        self.spatial_norm = nn.BatchNorm2d(f1 * depth_multiplier)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        self.separable_depth = nn.Conv2d(
            f1 * depth_multiplier, f1 * depth_multiplier, (1, 16),
            padding="same", groups=f1 * depth_multiplier, bias=False,
        )
        self.separable_point = nn.Conv2d(f1 * depth_multiplier, f2, 1, bias=False)
        self.separable_norm = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        self.output_pool = nn.AdaptiveAvgPool2d((1, 4))
        self.classifier = nn.Linear(f2 * 4, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal_norm(self.temporal(x))
        x = F.elu(self.spatial_norm(self.spatial(x)))
        x = self.dropout1(self.pool1(x))
        x = self.separable_depth(x)
        x = F.elu(self.separable_norm(self.separable_point(x)))
        x = self.dropout2(self.pool2(x))
        x = self.output_pool(x).flatten(1)
        return self.classifier(x)


class _SSVEPFormerEncoder(nn.Module):
    """One encoder from the public PyTorch SSVEPformer reproduction."""

    def __init__(self, channels: int, features: int, dropout: float) -> None:
        super().__init__()
        self.channels = channels
        self.norm_conv = nn.LayerNorm(features)
        self.conv = nn.Conv1d(channels, channels, 31, padding=15)
        self.norm_after_conv = nn.LayerNorm(features)
        self.norm_mlp = nn.LayerNorm(features)
        # The published reproduction applies a shared channel-to-spectrum
        # projection to one spectral coordinate at a time.
        self.mlp = nn.Linear(channels, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv(self.norm_conv(x))
        x = self.dropout(F.gelu(self.norm_after_conv(x))) + residual
        residual = x
        x = self.norm_mlp(x)
        x = torch.stack([self.mlp(x[:, :, index]) for index in range(self.channels)], dim=1)
        return self.dropout(x) + residual


class SSVEPFormerReference(nn.Module):
    """Protocol-adapted public PyTorch reproduction of SSVEPformer.

    The network structure follows okbalefthanded/ssvepformer. Only the input
    channel count, class count, sampling rate and shared evaluation band are
    parameterized so it can be evaluated under the same signal contract as the
    proposed model.
    """

    def __init__(
        self, channels: int, classes: int, *, sample_rate: int = 250,
        resolution_hz: float = 0.25, band_hz: tuple[float, float] = (6.0, 45.0),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_fft = int(round(sample_rate / resolution_hz))
        self.start = int(round(band_hz[0] / resolution_hz))
        self.end = int(round(band_hz[1] / resolution_hz)) + 1
        bins = self.end - self.start
        features = bins * 2
        hidden_channels = channels * 2
        self.channel_combination = nn.Conv1d(channels, hidden_channels, 1)
        self.channel_norm = nn.LayerNorm(features)
        self.channel_dropout = nn.Dropout(dropout)
        self.encoder1 = _SSVEPFormerEncoder(hidden_channels, features, dropout)
        self.encoder2 = _SSVEPFormerEncoder(hidden_channels, features, dropout)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(hidden_channels * features, 6 * classes),
            nn.LayerNorm(6 * classes), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(6 * classes, classes),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _complex_spectrum(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        samples = x.shape[-1]
        spectrum = torch.fft.fft(x, n=self.n_fft, dim=-1)
        real = spectrum.real[..., self.start:self.end] / samples
        imaginary = spectrum.imag[..., self.start:self.end] / samples
        return real, imaginary

    def forward_from_spectrum(
        self, real: torch.Tensor, imaginary: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([real, imaginary], dim=-1)
        x = self.channel_combination(x)
        x = self.channel_dropout(F.gelu(self.channel_norm(x)))
        x = self.encoder1(x)
        x = self.encoder2(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_spectrum(*self._complex_spectrum(x))


class FBSSVEPFormerReference(nn.Module):
    """Protocol-adapted three-subband FB-SSVEPformer reference.

    The original paper uses three independently parameterized SSVEPformer
    subnetworks and learns the subband fusion weights.  Here the registered
    common 6--45 Hz signal contract is retained, so the paper's lower cutoffs
    (with its disclosed 2 Hz margin) become 6, 14 and 22 Hz while all three
    branches share the registered 45 Hz upper bound.  Frequency masking is
    deterministic and differentiable; no target-participant data is used.

    This is deliberately a protocol adaptation, not a claim of bit-identical
    reproduction of the authors' 80 Hz IIR preprocessing pipeline.
    """

    def __init__(
        self, channels: int, classes: int, *, sample_rate: int = 250,
        resolution_hz: float = 0.25, band_hz: tuple[float, float] = (6.0, 45.0),
        subband_low_hz: tuple[float, ...] = (6.0, 14.0, 22.0),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if not subband_low_hz:
            raise ValueError("at least one filter-bank subband is required")
        if any(low < band_hz[0] or low >= band_hz[1] for low in subband_low_hz):
            raise ValueError("subband lower cutoffs must lie inside band_hz")
        self.sample_rate = int(sample_rate)
        self.resolution_hz = float(resolution_hz)
        self.band_hz = tuple(float(value) for value in band_hz)
        self.subband_low_hz = tuple(float(value) for value in subband_low_hz)
        self.subnetworks = nn.ModuleList([
            SSVEPFormerReference(
                channels=channels, classes=classes, sample_rate=sample_rate,
                resolution_hz=resolution_hz, band_hz=band_hz, dropout=dropout,
            )
            for _ in self.subband_low_hz
        ])
        # Conv1d implements the paper's learned per-subband score fusion with
        # weights shared across target classes.
        self.fusion = nn.Conv1d(len(self.subband_low_hz), 1, kernel_size=1, bias=True)
        nn.init.constant_(self.fusion.weight, 1.0 / len(self.subband_low_hz))
        nn.init.zeros_(self.fusion.bias)

    def _mask(self, reference: torch.Tensor, low_hz: float) -> torch.Tensor:
        bins = reference.shape[-1]
        start = self.subnetworks[0].start
        frequencies = (
            torch.arange(bins, device=reference.device, dtype=reference.dtype) + start
        ) * self.resolution_hz
        return (frequencies >= low_hz).to(reference.dtype).view(1, 1, bins)

    def subnetwork_logits(self, x: torch.Tensor) -> torch.Tensor:
        real, imaginary = self.subnetworks[0]._complex_spectrum(x)
        outputs = []
        for low_hz, network in zip(self.subband_low_hz, self.subnetworks, strict=True):
            mask = self._mask(real, low_hz)
            outputs.append(network.forward_from_spectrum(real * mask, imaginary * mask))
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(self.subnetwork_logits(x)).squeeze(1)
