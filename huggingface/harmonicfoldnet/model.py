from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv1d(cin, cout, kernel, stride=stride, padding=kernel // 2,
                      groups=groups, bias=False),
            nn.BatchNorm1d(cout),
            nn.GELU(),
        )


class LocalMixer(nn.Module):
    """Multi-scale depthwise temporal mixing followed by channel mixing."""
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dw3 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.dw7 = nn.Conv1d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.norm = nn.BatchNorm1d(dim)
        hidden = dim * expansion
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, hidden, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, dim, 1), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.norm(self.dw3(x) + self.dw7(x))
        return x + self.ffn(x)


def _fuse_conv_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> tuple[torch.Tensor, torch.Tensor]:
    weight = conv.weight
    bias = conv.bias if conv.bias is not None else torch.zeros(
        weight.shape[0], device=weight.device, dtype=weight.dtype,
    )
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return weight * scale.reshape(-1, 1, 1), bn.bias + (bias - bn.running_mean) * scale


class RepTokenMixer1d(nn.Module):
    """Train with three paths and fuse them into one depthwise Conv1d for deployment."""

    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.dim = dim
        self.kernel_size = kernel_size
        self.branch_large = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2,
                      groups=dim, bias=False),
            nn.BatchNorm1d(dim),
        )
        self.branch_pointwise = nn.Sequential(
            nn.Conv1d(dim, dim, 1, groups=dim, bias=False), nn.BatchNorm1d(dim),
        )
        self.branch_identity = nn.BatchNorm1d(dim)
        self.deploy_conv: nn.Conv1d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy_conv is not None:
            return self.deploy_conv(x)
        return self.branch_large(x) + self.branch_pointwise(x) + self.branch_identity(x)

    def reparameterize(self) -> None:
        if self.deploy_conv is not None:
            return
        base = self.branch_large[0]
        large_w, large_b = _fuse_conv_bn(base, self.branch_large[1])
        point_w, point_b = _fuse_conv_bn(self.branch_pointwise[0], self.branch_pointwise[1])
        point_w = F.pad(point_w, [self.kernel_size // 2] * 2)
        identity = nn.Conv1d(
            self.dim, self.dim, self.kernel_size, padding=self.kernel_size // 2,
            groups=self.dim, bias=False,
        ).to(device=base.weight.device, dtype=base.weight.dtype)
        identity.weight.data.zero_()
        identity.weight.data[:, 0, self.kernel_size // 2] = 1.0
        identity_w, identity_b = _fuse_conv_bn(identity, self.branch_identity)
        deploy = nn.Conv1d(
            self.dim, self.dim, self.kernel_size, padding=self.kernel_size // 2,
            groups=self.dim, bias=True,
        ).to(device=base.weight.device, dtype=base.weight.dtype)
        deploy.weight.data.copy_(large_w + point_w + identity_w)
        deploy.bias.data.copy_(large_b + point_b + identity_b)
        self.deploy_conv = deploy
        del self.branch_large
        del self.branch_pointwise
        del self.branch_identity


class RepLocalMixer(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.token_mixer = RepTokenMixer1d(dim, kernel_size=7)
        hidden = dim * expansion
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, hidden, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, dim, 1), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.token_mixer(x))
        return x + self.ffn(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 3, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        x = x + self.attn(z, z, z, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class LegacyLocalAttentionNet(nn.Module):
    """EEG local-first, attention-late network inspired by HarmonicFold's design logic.

    It is an original 1-D EEG implementation and does not use Apple weights or code.
    """
    def __init__(
        self, channels: int = 8, classes: int = 4, width: int = 64,
        local_depth: int = 3, attention_depth: int = 2,
        heads: int = 4, dropout: float = 0.15,
    ):
        super().__init__()
        self.channel_projection = nn.Sequential(
            nn.Conv1d(channels, width, 1, bias=False), nn.BatchNorm1d(width), nn.GELU(),
        )
        self.stem = ConvNormAct(width, width, 9, stride=2, groups=width)
        self.local = nn.Sequential(*[
            LocalMixer(width, expansion=2, dropout=dropout) for _ in range(local_depth)
        ])
        self.downsample = ConvNormAct(width, width * 2, 5, stride=4)
        dim = width * 2
        self.global_blocks = nn.Sequential(*[
            AttentionBlock(dim, heads=heads, dropout=dropout) for _ in range(attention_depth)
        ])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_projection(x)
        x = self.local(self.stem(x))
        x = self.downsample(x).transpose(1, 2)
        x = self.global_blocks(x)
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reparameterize_model(model: nn.Module) -> nn.Module:
    deploy_model = copy.deepcopy(model).eval()
    for module in deploy_model.modules():
        if isinstance(module, RepTokenMixer1d):
            module.reparameterize()
    return deploy_model


class _LocalAttentionEncoder(nn.Module):
    def __init__(
        self, channels: int, width: int, local_depth: int,
        attention_depth: int, heads: int, dropout: float,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(channels, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.stem = ConvNormAct(width, width, 9, stride=2, groups=width)
        self.local = nn.Sequential(*[
            RepLocalMixer(width, expansion=2, dropout=dropout)
            for _ in range(local_depth)
        ])
        self.downsample = ConvNormAct(width, width * 2, 5, stride=4)
        self.global_blocks = nn.Sequential(*[
            AttentionBlock(width * 2, heads=heads, dropout=dropout)
            for _ in range(attention_depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.local(self.stem(x))
        x = self.downsample(x).transpose(1, 2)
        return self.global_blocks(x).mean(dim=1)


class LegacyFusionNet(nn.Module):
    """Local-first time/frequency EEG encoder with attention applied late.

    The spectral branch is frequency-agnostic: it consumes a normalized band
    rather than embedding a fixed set of SSVEP labels, so the same architecture
    can be trained on datasets with different target frequencies.
    """

    def __init__(
        self, channels: int = 8, classes: int = 4, sample_rate: int = 250,
        width: int = 48, local_depth: int = 2, attention_depth: int = 1,
        heads: int = 4, dropout: float = 0.15, spectral_bins: int = 128,
        spectral_band_hz: tuple[float, float] = (0.5, 45.0),
        class_frequencies: list[float] | tuple[float, ...] | None = None,
        class_phases: list[float] | tuple[float, ...] | None = None,
        evidence_harmonics: int = 2,
        evidence_mode: str = "power",
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.spectral_bins = int(spectral_bins)
        self.spectral_band_hz = tuple(float(v) for v in spectral_band_hz)
        self.evidence_harmonics = int(evidence_harmonics)
        if evidence_mode not in {"power", "cca_phase", "phase_demod"}:
            raise ValueError(f"unsupported evidence mode: {evidence_mode}")
        self.evidence_mode = evidence_mode
        self._reference_q_cache: dict[tuple[int, str], torch.Tensor] = {}
        frequencies = [] if class_frequencies is None else list(class_frequencies)
        if frequencies and len(frequencies) != classes:
            raise ValueError("class_frequencies must contain one frequency per class")
        self.register_buffer(
            "class_frequencies",
            torch.tensor(frequencies, dtype=torch.float32), persistent=True,
        )
        phases = [] if class_phases is None else list(class_phases)
        if phases and len(phases) != classes:
            raise ValueError("class_phases must contain one phase per class")
        if evidence_mode == "phase_demod" and len(phases) != classes:
            raise ValueError("phase_demod requires class phases")
        self.register_buffer(
            "class_phases", torch.tensor(phases, dtype=torch.float32), persistent=True,
        )
        self.time_encoder = _LocalAttentionEncoder(
            channels, width, local_depth, attention_depth, heads, dropout,
        )
        self.frequency_encoder = _LocalAttentionEncoder(
            channels, width, local_depth, attention_depth, heads, dropout,
        )
        dim = width * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, classes))
        evidence_features = channels * self.evidence_harmonics * (3 if evidence_mode == "phase_demod" else 1)
        evidence_hidden = max(32 if evidence_mode == "phase_demod" else 16, channels * 2)
        self.evidence_residual = nn.Sequential(
            nn.Linear(evidence_features, evidence_hidden),
            nn.GELU(),
            nn.Linear(evidence_hidden, 1),
        ) if frequencies else None
        self.evidence_scale = nn.Parameter(torch.tensor(1.5)) if frequencies else None
        if self.evidence_residual is not None and self.evidence_mode in {"cca_phase", "phase_demod"}:
            nn.init.zeros_(self.evidence_residual[-1].weight)
            nn.init.zeros_(self.evidence_residual[-1].bias)

    def _spectrum(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1, norm="ortho").abs().square()
        nyquist_bins = spectrum.shape[-1] - 1
        lo = max(0, int(round(self.spectral_band_hz[0] * 2 * nyquist_bins / self.sample_rate)))
        hi = min(spectrum.shape[-1], int(round(self.spectral_band_hz[1] * 2 * nyquist_bins / self.sample_rate)) + 1)
        spectrum = torch.log(spectrum[..., lo:hi].clamp_min(1e-8))
        # Remove subject/session gain and the smooth 1/f spectral envelope.
        smooth = F.avg_pool1d(spectrum, kernel_size=9, stride=1, padding=4)
        spectrum = spectrum - smooth
        spectrum = F.interpolate(
            spectrum, size=self.spectral_bins, mode="linear", align_corners=False,
        )
        return spectrum

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        time_features = self.time_encoder(x)
        frequency_features = self.frequency_encoder(self._spectrum(x))
        return self.fusion(torch.cat([time_features, frequency_features], dim=-1))

    def _fusion_logits(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode not in {"full", "fusion_only", "time_only", "frequency_only"}:
            raise ValueError(f"unsupported fusion mode: {mode}")
        time_features = self.time_encoder(x) if mode != "frequency_only" else None
        frequency_features = (
            self.frequency_encoder(self._spectrum(x)) if mode != "time_only" else None
        )
        if time_features is None:
            time_features = torch.zeros_like(frequency_features)
        if frequency_features is None:
            frequency_features = torch.zeros_like(time_features)
        fused = self.fusion(torch.cat([time_features, frequency_features], dim=-1))
        return self.head(fused)

    def candidate_evidence(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.class_frequencies.numel():
            return None
        if self.evidence_mode == "phase_demod":
            base = self.reference_correlation_evidence(x)
            features = self.phase_aligned_demodulation(x)
        else:
            spectrum = torch.fft.rfft(x, dim=-1, norm="ortho").abs().square()
            max_bin = spectrum.shape[-1] - 1
            harmonic_numbers = torch.arange(
                1, self.evidence_harmonics + 1, device=x.device, dtype=x.dtype,
            )
            target_hz = self.class_frequencies.to(dtype=x.dtype)[:, None] * harmonic_numbers[None, :]
            indices = torch.round(target_hz * (2 * max_bin / self.sample_rate)).long().clamp_(0, max_bin)
            selected = spectrum[..., indices.reshape(-1)]
            selected = selected.reshape(
                x.shape[0], x.shape[1], len(self.class_frequencies), self.evidence_harmonics,
            )
            if self.evidence_mode == "cca_phase":
                base = self.reference_correlation_evidence(x)
            else:
                base = torch.log(selected.mean(dim=(1, 3)).clamp_min(1e-8))
            features = torch.log(selected.clamp_min(1e-8)).permute(0, 2, 1, 3).flatten(2)
        base = (base - base.mean(dim=1, keepdim=True)) / base.std(dim=1, keepdim=True).clamp_min(1e-5)
        features = (features - features.mean(dim=1, keepdim=True)) / features.std(dim=1, keepdim=True).clamp_min(1e-5)
        return base + self.evidence_residual(features).squeeze(-1)

    def phase_aligned_demodulation(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            signal = x.float()
            samples = signal.shape[-1]
            time_axis = torch.arange(samples, device=x.device, dtype=torch.float32) / self.sample_rate
            harmonics = torch.arange(
                1, self.evidence_harmonics + 1, device=x.device, dtype=torch.float32,
            )
            frequencies = self.class_frequencies.to(device=x.device, dtype=torch.float32)
            phases = self.class_phases.to(device=x.device, dtype=torch.float32)
            angles = (
                2.0 * torch.pi * frequencies[:, None, None]
                * harmonics[None, :, None] * time_axis[None, None, :]
                + phases[:, None, None] * harmonics[None, :, None]
            )
            real = torch.einsum("bct,nht->bnch", signal, torch.cos(angles)) / samples
            imaginary = torch.einsum("bct,nht->bnch", signal, torch.sin(angles)) / samples
            amplitude = torch.sqrt(real.square() + imaginary.square() + 1e-8)
            features = torch.stack((real, imaginary, torch.log(amplitude + 1e-8)), dim=-1)
            return features.flatten(2).to(dtype=x.dtype)

    def reference_correlation_evidence(self, x: torch.Tensor) -> torch.Tensor:
        """Regularized CCA evidence against phase-complete sinusoid subspaces.

        This path has no trainable parameters. It retains the classical SSVEP
        prior while the learned residual and time/frequency encoders model
        deviations from that prior.
        """
        if not self.class_frequencies.numel():
            raise RuntimeError("reference evidence requires class frequencies")
        # Keep the whitening decomposition in float32 even under AMP. Dry EEG
        # channels can be nearly collinear, so eigenvalue clipping is safer
        # than assuming every sample covariance admits a Cholesky factor.
        with torch.autocast(device_type=x.device.type, enabled=False):
            signal = x.float()
            _, channels, samples = signal.shape
            signal = signal - signal.mean(dim=-1, keepdim=True)
            covariance = signal @ signal.transpose(-1, -2) / max(samples - 1, 1)
            values, vectors = torch.linalg.eigh(covariance)
            scale = values.amax(dim=-1, keepdim=True).clamp_min(1.0)
            values = values.clamp_min(1e-4 * scale)
            inverse_sqrt = (vectors * values.rsqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
            whitened = (inverse_sqrt @ signal) / float(max(samples - 1, 1)) ** 0.5
            key = (samples, str(signal.device))
            reference_q = self._reference_q_cache.get(key)
            if reference_q is None or reference_q.device != signal.device:
                time_axis = torch.arange(
                    samples, device=signal.device, dtype=torch.float32,
                ) / self.sample_rate
                harmonics = torch.arange(
                    1, self.evidence_harmonics + 1, device=signal.device, dtype=torch.float32,
                )
                frequencies = self.class_frequencies.to(device=signal.device, dtype=torch.float32)
                angles = (
                    2.0 * torch.pi * frequencies[:, None, None]
                    * harmonics[None, :, None] * time_axis[None, None, :]
                )
                references = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
                references = references.permute(0, 2, 1, 3).flatten(2)
                reference_q = torch.linalg.qr(references, mode="reduced").Q
                self._reference_q_cache[key] = reference_q
            cross_covariance = torch.einsum("bct,ntk->bnck", whitened, reference_q)
            evidence = torch.linalg.svdvals(cross_covariance)[..., 0]
        return evidence.to(dtype=x.dtype)

    def forward_mode(self, x: torch.Tensor, mode: str = "full") -> torch.Tensor:
        if mode == "evidence_only":
            evidence = self.candidate_evidence(x)
            if evidence is None:
                raise RuntimeError("evidence_only requires class frequencies")
            return F.softplus(self.evidence_scale) * evidence
        logits = self._fusion_logits(x, mode)
        if mode == "full":
            evidence = self.candidate_evidence(x)
            if evidence is not None:
                logits = logits + F.softplus(self.evidence_scale) * evidence
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_mode(x, "full")
