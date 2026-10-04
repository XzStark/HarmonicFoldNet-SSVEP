from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .model import AttentionBlock, RepTokenMixer1d


FUSION_MODES = ("full", "no_evidence", "no_attention", "no_local")
HARMONIC_FOLD_MODES = (
    "full", "no_attention", "no_cross_attention", "no_candidate_attention",
    "no_local", "no_harmonic_bias",
)
TEMPORAL_FUSION_ARCHITECTURE_REVISION = "temporal_fusion_v1_3"
SPECTRAL_FUSION_ARCHITECTURE_REVISION = "spectral_fusion_v3_0"
HARMONIC_FOLD_V4_ARCHITECTURE_REVISION = "harmonic_fold_v4_0"
HARMONIC_FOLD_V41_ARCHITECTURE_REVISION = "harmonic_fold_v4_1"
HARMONIC_FOLD_V42_ARCHITECTURE_REVISION = "harmonic_fold_v4_2"
HARMONIC_FOLD_V43_ARCHITECTURE_REVISION = "harmonic_fold_v4_3"
HARMONIC_FOLD_V44_ARCHITECTURE_REVISION = "harmonic_fold_v4_4"
HARMONIC_FOLD_V45_ARCHITECTURE_REVISION = "harmonic_fold_v4_5_1"
HARMONIC_FOLD_V46_ARCHITECTURE_REVISION = "harmonic_fold_v4_6"
HARMONIC_FOLD_V47_ARCHITECTURE_REVISION = "harmonic_fold_v4_7"


class ConvFFN1d(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.BatchNorm1d(dim)
        self.layers = nn.Sequential(
            nn.Conv1d(dim, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, dim, 1),
            nn.Dropout(dropout),
        )
        # Start as an identity residual path. This avoids destroying the weak
        # periodic input before the local branch has learned a useful update.
        nn.init.zeros_(self.layers[-2].weight)
        nn.init.zeros_(self.layers[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(self.norm(x))


class RepEEGBlock(nn.Module):
    """Local train-time multi-branch mixer with a foldable inference graph."""

    def __init__(
        self, dim: int, kernel_size: int, dropout: float, *, gate_logit_init: float = -2.0,
    ):
        super().__init__()
        self.mixer = RepTokenMixer1d(dim, kernel_size=kernel_size)
        nn.init.zeros_(self.mixer.branch_large[1].weight)
        nn.init.zeros_(self.mixer.branch_pointwise[1].weight)
        self.ffn = ConvFFN1d(dim, expansion=2, dropout=dropout)
        self.residual_gate_logit = nn.Parameter(torch.tensor(float(gate_logit_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.ffn(self.mixer(x)) - x
        return x + torch.sigmoid(self.residual_gate_logit) * update


class DepthwiseDownsample1d(nn.Module):
    def __init__(self, cin: int, cout: int, *, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(cin, cin, 9, stride=stride, padding=4, groups=cin, bias=False),
            nn.BatchNorm1d(cin),
            nn.GELU(),
            nn.Conv1d(cin, cout, 1, bias=False),
            nn.BatchNorm1d(cout),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CandidateFeatureExtractor(nn.Module):
    """Shared candidate-aligned phase features, independent of class count."""

    def __init__(
        self,
        channels: int,
        dim: int,
        sample_rate: int,
        class_frequencies: list[float] | tuple[float, ...],
        class_phases: list[float] | tuple[float, ...] | None,
        *,
        harmonics: int = 4,
        segments: int = 4,
        phase_dynamics: bool = False,
    ):
        super().__init__()
        if not class_frequencies:
            raise ValueError("class_frequencies cannot be empty")
        phases = list(class_phases or [0.0] * len(class_frequencies))
        if len(phases) != len(class_frequencies):
            raise ValueError("class_phases must match class_frequencies")
        self.channels = int(channels)
        self.sample_rate = float(sample_rate)
        self.harmonics = int(harmonics)
        self.segments = int(segments)
        if self.segments <= 0:
            raise ValueError("segments must be positive")
        self.phase_dynamics_enabled = bool(phase_dynamics)
        if self.phase_dynamics_enabled and self.segments < 2:
            raise ValueError("phase dynamics require at least two temporal segments")
        self.register_buffer(
            "class_frequencies", torch.tensor(class_frequencies, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "class_phases", torch.tensor(phases, dtype=torch.float32), persistent=True,
        )
        normalized = self.class_frequencies / (0.5 * self.sample_rate)
        self.register_buffer(
            "frequency_encoding_features",
            torch.stack(
                (
                    normalized,
                    normalized.square(),
                    torch.sin(2.0 * torch.pi * normalized),
                    torch.cos(2.0 * torch.pi * normalized),
                    torch.sin(4.0 * torch.pi * normalized),
                    torch.cos(4.0 * torch.pi * normalized),
                    torch.sin(self.class_phases),
                    torch.cos(self.class_phases),
                ),
                dim=-1,
            ),
            persistent=False,
        )
        self._demodulation_basis_cache: dict[
            tuple[int, str], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        phase_dynamics_features = 5 if self.phase_dynamics_enabled else 0
        feature_dim = channels * harmonics * (
            segments * 3 + phase_dynamics_features
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.frequency_projection = nn.Sequential(
            nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim),
        )

    def _frequency_encoding(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.frequency_projection(
            self.frequency_encoding_features.to(device=device, dtype=dtype)
        )

    def _demodulation_basis(
        self, samples: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (samples, str(device))
        cached = self._demodulation_basis_cache.get(key)
        if cached is not None and cached[0].device == device:
            return cached
        # Validation calls run under inference_mode. A cached inference tensor
        # cannot later be saved by autograd when the local EEG front end makes
        # the signal differentiable, so build cache entries in normal tensor
        # mode regardless of the caller's context.
        with torch.inference_mode(False), torch.no_grad():
            time_axis = (
                torch.arange(samples, device=device, dtype=torch.float32)
                / self.sample_rate
            )
            harmonics = torch.arange(
                1, self.harmonics + 1, device=device, dtype=torch.float32,
            )
            frequency = self.class_frequencies.to(device=device, dtype=torch.float32)
            phase = self.class_phases.to(device=device, dtype=torch.float32)
            angles = (
                2.0 * torch.pi * frequency[:, None, None]
                * harmonics[None, :, None] * time_axis[None, None, :]
                + phase[:, None, None] * harmonics[None, :, None]
            )
            cosine = torch.cos(angles)
            sine = torch.sin(angles)
            cosine_segments = []
            sine_segments = []
            for segment in range(self.segments):
                begin = (segment * samples) // self.segments
                end = ((segment + 1) * samples) // self.segments
                length = max(end - begin, 1)
                mask = torch.zeros(samples, device=device, dtype=torch.float32)
                mask[begin:end] = 1.0 / length
                cosine_segments.append(cosine * mask)
                sine_segments.append(sine * mask)
            basis = (
                torch.stack(cosine_segments, dim=2),
                torch.stack(sine_segments, dim=2),
            )
        self._demodulation_basis_cache[key] = basis
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            signal = x.float()
            samples = signal.shape[-1]
            cosine, sine = self._demodulation_basis(samples, x.device)
            real = torch.einsum("bct,nhst->bnchs", signal, cosine)
            imaginary = torch.einsum("bct,nhst->bnchs", signal, sine)
            amplitude = torch.sqrt(real.square() + imaginary.square() + 1e-8)
            features = torch.stack(
                (real, imaginary, torch.log(amplitude + 1e-8)), dim=-1,
            ).flatten(2)
            if self.phase_dynamics_enabled:
                unit_real = real / amplitude
                unit_imaginary = imaginary / amplitude
                adjacent_real = (
                    unit_real[..., 1:] * unit_real[..., :-1]
                    + unit_imaginary[..., 1:] * unit_imaginary[..., :-1]
                ).mean(dim=-1)
                adjacent_imaginary = (
                    unit_imaginary[..., 1:] * unit_real[..., :-1]
                    - unit_real[..., 1:] * unit_imaginary[..., :-1]
                ).mean(dim=-1)
                endpoint_real = (
                    unit_real[..., -1] * unit_real[..., 0]
                    + unit_imaginary[..., -1] * unit_imaginary[..., 0]
                )
                endpoint_imaginary = (
                    unit_imaginary[..., -1] * unit_real[..., 0]
                    - unit_real[..., -1] * unit_imaginary[..., 0]
                )
                amplitude_instability = torch.log(amplitude + 1e-8).std(
                    dim=-1, correction=0,
                )
                phase_dynamics = torch.stack(
                    (
                        adjacent_real,
                        adjacent_imaginary,
                        endpoint_real,
                        endpoint_imaginary,
                        amplitude_instability,
                    ),
                    dim=-1,
                ).flatten(2)
                features = torch.cat((features, phase_dynamics), dim=-1)
            features = (features - features.mean(dim=1, keepdim=True)) / features.std(
                dim=1, keepdim=True, correction=0,
            ).clamp_min(1e-5)
        tokens = self.projection(features.to(dtype=x.dtype))
        return tokens + self._frequency_encoding(device=x.device, dtype=x.dtype).unsqueeze(0)


class AnalyticReferencePrior(nn.Module):
    """Parameter-free CCA score used only as a residual prior and ablation."""

    def __init__(
        self,
        sample_rate: int,
        class_frequencies: list[float] | tuple[float, ...],
        *,
        harmonics: int = 4,
    ):
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.harmonics = int(harmonics)
        self.register_buffer(
            "class_frequencies", torch.tensor(class_frequencies, dtype=torch.float32),
            persistent=True,
        )
        self._reference_q_cache: dict[tuple[int, str], torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            signal = x.float()
            _, _, samples = signal.shape
            signal = signal - signal.mean(dim=-1, keepdim=True)
            covariance = signal @ signal.transpose(-1, -2) / max(samples - 1, 1)
            values, vectors = torch.linalg.eigh(covariance)
            scale = values.amax(dim=-1, keepdim=True).clamp_min(1.0)
            values = values.clamp_min(1e-4 * scale)
            inverse_sqrt = (
                vectors * values.rsqrt().unsqueeze(-2)
            ) @ vectors.transpose(-1, -2)
            whitened = (inverse_sqrt @ signal) / math.sqrt(max(samples - 1, 1))
            key = (samples, str(signal.device))
            reference_q = self._reference_q_cache.get(key)
            if reference_q is None or reference_q.device != signal.device:
                time_axis = torch.arange(
                    samples, device=signal.device, dtype=torch.float32,
                ) / self.sample_rate
                harmonics = torch.arange(
                    1, self.harmonics + 1, device=signal.device, dtype=torch.float32,
                )
                angles = (
                    2.0 * torch.pi
                    * self.class_frequencies.to(signal.device)[:, None, None]
                    * harmonics[None, :, None]
                    * time_axis[None, None, :]
                )
                references = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
                references = references.permute(0, 2, 1, 3).flatten(2)
                reference_q = torch.linalg.qr(references, mode="reduced").Q
                self._reference_q_cache[key] = reference_q
            cross = torch.einsum("bct,ntk->bnck", whitened, reference_q)
            scores = torch.linalg.svdvals(cross)[..., 0]
            scores = (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
                dim=1, keepdim=True, correction=0,
            ).clamp_min(1e-5)
        return scores.to(dtype=x.dtype)


class ComplexSpectrumFrontEnd(nn.Module):
    """Fixed-grid complex spectrum used by both global and candidate paths."""

    def __init__(
        self,
        sample_rate: int,
        *,
        resolution_hz: float = 0.25,
        band_hz: tuple[float, float] = (6.0, 45.0),
    ) -> None:
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.resolution_hz = float(resolution_hz)
        self.n_fft = int(round(self.sample_rate / self.resolution_hz))
        self.start = int(round(float(band_hz[0]) / self.resolution_hz))
        self.end = int(round(float(band_hz[1]) / self.resolution_hz)) + 1
        if self.start < 0 or self.end > self.n_fft // 2 + 1 or self.start >= self.end:
            raise ValueError("spectral band must lie inside the Nyquist interval")

    def spectrum(self, x: torch.Tensor) -> torch.Tensor:
        # Divide by the observed duration rather than n_fft so zero padding only
        # changes spectral resolution, not the scale seen by the network.
        return torch.fft.rfft(x.float(), n=self.n_fft, dim=-1) / x.shape[-1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = self.spectrum(x)
            band = spectrum[..., self.start:self.end]
            magnitude = torch.log1p(10.0 * band.abs())
            features = torch.cat((band.real, band.imag, magnitude), dim=1)
        return spectrum, features.to(dtype=x.dtype)


class SpectralCandidateExtractor(nn.Module):
    """Sample complex harmonic neighborhoods on a fixed physical-Hz grid."""

    def __init__(
        self,
        channels: int,
        dim: int,
        sample_rate: int,
        class_frequencies: list[float] | tuple[float, ...],
        class_phases: list[float] | tuple[float, ...] | None,
        *,
        resolution_hz: float,
        harmonics: int = 4,
        neighborhood_bins: int = 2,
    ) -> None:
        super().__init__()
        if not class_frequencies:
            raise ValueError("class_frequencies cannot be empty")
        if neighborhood_bins < 0:
            raise ValueError("neighborhood_bins must be non-negative")
        phases = list(class_phases or [0.0] * len(class_frequencies))
        if len(phases) != len(class_frequencies):
            raise ValueError("class_phases must match class_frequencies")
        self.channels = int(channels)
        self.sample_rate = float(sample_rate)
        self.resolution_hz = float(resolution_hz)
        self.harmonics = int(harmonics)
        self.neighborhood_bins = int(neighborhood_bins)
        self.register_buffer(
            "class_frequencies", torch.tensor(class_frequencies, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "class_phases", torch.tensor(phases, dtype=torch.float32), persistent=True,
        )
        normalized = self.class_frequencies / (0.5 * self.sample_rate)
        self.register_buffer(
            "frequency_encoding_features",
            torch.stack(
                (
                    normalized,
                    normalized.square(),
                    torch.sin(2.0 * torch.pi * normalized),
                    torch.cos(2.0 * torch.pi * normalized),
                    torch.sin(4.0 * torch.pi * normalized),
                    torch.cos(4.0 * torch.pi * normalized),
                    torch.sin(self.class_phases),
                    torch.cos(self.class_phases),
                ),
                dim=-1,
            ),
            persistent=False,
        )
        harmonic = torch.arange(1, self.harmonics + 1, dtype=torch.float32)
        offsets = torch.arange(
            -self.neighborhood_bins, self.neighborhood_bins + 1, dtype=torch.float32,
        ) * self.resolution_hz
        target_hz = self.class_frequencies[:, None, None] * harmonic[None, :, None]
        target_hz = target_hz + offsets[None, None, :]
        max_hz = 0.5 * self.sample_rate
        position = (target_hz / self.resolution_hz).clamp(0, int(max_hz / self.resolution_hz))
        lower = position.floor().long()
        self.register_buffer("sample_lower", lower.flatten(), persistent=False)
        self.register_buffer("sample_upper", (lower + 1).clamp_max(int(max_hz / self.resolution_hz)).flatten(), persistent=False)
        self.register_buffer("sample_fraction", (position - lower).flatten(), persistent=False)
        self.register_buffer(
            "sample_valid", ((target_hz >= 0.0) & (target_hz <= max_hz)).flatten(),
            persistent=False,
        )
        neighborhood = 2 * self.neighborhood_bins + 1
        feature_dim = channels * harmonics * neighborhood * 3
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.frequency_projection = nn.Sequential(
            nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim),
        )

    def _frequency_encoding(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.frequency_projection(
            self.frequency_encoding_features.to(device=device, dtype=dtype)
        )

    def forward(self, spectrum: torch.Tensor, *, output_dtype: torch.dtype) -> torch.Tensor:
        with torch.autocast(device_type=spectrum.device.type, enabled=False):
            spectrum = spectrum.to(torch.complex64)
            lower_value = spectrum.index_select(-1, self.sample_lower)
            upper_value = spectrum.index_select(-1, self.sample_upper)
            fraction = self.sample_fraction.view(1, 1, -1)
            sampled = lower_value * (1.0 - fraction) + upper_value * fraction
            sampled = sampled * self.sample_valid.view(1, 1, -1)
            batch = sampled.shape[0]
            classes = self.class_frequencies.numel()
            neighborhood = 2 * self.neighborhood_bins + 1
            sampled = sampled.view(
                batch, self.channels, classes, self.harmonics, neighborhood,
            ).permute(0, 2, 1, 3, 4)
            feature = torch.stack(
                (
                    sampled.real,
                    sampled.imag,
                    torch.log1p(10.0 * sampled.abs()),
                ),
                dim=-1,
            ).flatten(2)
            feature = (feature - feature.mean(dim=1, keepdim=True)) / feature.std(
                dim=1, keepdim=True, correction=0,
            ).clamp_min(1e-5)
        tokens = self.projection(feature.to(dtype=output_dtype))
        return tokens + self._frequency_encoding(
            device=spectrum.device, dtype=output_dtype,
        ).unsqueeze(0)


class HarmonicCrossAttention(nn.Module):
    """Late class-query attention with a fixed harmonic locality bias."""

    def __init__(
        self,
        dim: int,
        heads: int,
        class_frequencies: list[float] | tuple[float, ...],
        spectral_token_hz: torch.Tensor,
        *,
        harmonics: int,
        bias_width_hz: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("attention dimension must be divisible by heads")
        if bias_width_hz <= 0:
            raise ValueError("bias_width_hz must be positive")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.dropout = float(dropout)
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )
        frequencies = torch.tensor(class_frequencies, dtype=torch.float32)
        harmonic = torch.arange(1, harmonics + 1, dtype=torch.float32)
        targets = frequencies[:, None] * harmonic[None, :]
        distance = spectral_token_hz[None, None, :] - targets[:, :, None]
        harmonic_bias = torch.logsumexp(
            -0.5 * (distance / float(bias_width_hz)).square(), dim=1,
        )
        harmonic_bias = harmonic_bias - harmonic_bias.amax(dim=-1, keepdim=True)
        self.register_buffer("harmonic_bias", harmonic_bias, persistent=False)
        self.register_buffer(
            "spectral_token_hz", spectral_token_hz.detach().clone(), persistent=False,
        )
        self.register_buffer("harmonic_target_hz", targets, persistent=False)
        self.bias_width_hz = float(bias_width_hz)

    def attention_probabilities(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        use_harmonic_bias: bool = True,
    ) -> torch.Tensor:
        """Return analysis-only attention probabilities without dropout."""
        batch, candidates, _ = query.shape
        tokens = context.shape[1]
        q = self.query_projection(self.query_norm(query)).view(
            batch, candidates, self.heads, self.head_dim,
        ).transpose(1, 2)
        normalized_context = self.context_norm(context)
        k = self.key_projection(normalized_context).view(
            batch, tokens, self.heads, self.head_dim,
        ).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if use_harmonic_bias:
            logits = logits + self.harmonic_bias.to(dtype=logits.dtype)[None, None, :, :]
        return logits.softmax(dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        use_harmonic_bias: bool = True,
    ) -> torch.Tensor:
        batch, candidates, _ = query.shape
        tokens = context.shape[1]
        q = self.query_projection(self.query_norm(query)).view(
            batch, candidates, self.heads, self.head_dim,
        ).transpose(1, 2)
        normalized_context = self.context_norm(context)
        k = self.key_projection(normalized_context).view(
            batch, tokens, self.heads, self.head_dim,
        ).transpose(1, 2)
        v = self.value_projection(normalized_context).view(
            batch, tokens, self.heads, self.head_dim,
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=(
                self.harmonic_bias.to(dtype=q.dtype)[None, None, :, :]
                if use_harmonic_bias else None
            ),
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, candidates, self.dim)
        query = query + self.output_projection(attended)
        return query + self.ffn(self.ffn_norm(query))


class HarmonicFoldNet(nn.Module):
    """local-to-global SSVEP decoder with attention as the global fusion path."""

    def __init__(
        self,
        channels: int,
        sample_rate: int,
        class_frequencies: list[float] | tuple[float, ...],
        class_phases: list[float] | tuple[float, ...] | None = None,
        *,
        width: int = 48,
        local_depths: tuple[int, int] = (1, 1),
        heads: int = 4,
        harmonics: int = 4,
        neighborhood_bins: int = 2,
        spectral_resolution_hz: float = 0.25,
        spectral_band_hz: tuple[float, float] = (6.0, 45.0),
        harmonic_bias_width_hz: float = 0.5,
        align_spectral_grid_to_classes: bool = False,
        duration_conditioning: bool = False,
        candidate_local_mixing: bool = False,
        candidate_local_mixing_placement: str = "pre_cross",
        local_domain: str = "spectral",
        temporal_kernel_sizes: tuple[int, int] = (15, 7),
        temporal_segments: int = 4,
        temporal_phase_dynamics: bool = False,
        spectral_subband_low_hz: tuple[float, ...] | None = None,
        evidence_reliability_gate: bool = False,
        decision_reliability_gate: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        dim = width * 2
        if dim % heads:
            raise ValueError("2 * width must be divisible by heads")
        if local_domain not in {"spectral", "temporal", "dual"}:
            raise ValueError("local_domain must be spectral, temporal, or dual")
        if len(temporal_kernel_sizes) != 2 or any(
            int(kernel) <= 0 or int(kernel) % 2 == 0
            for kernel in temporal_kernel_sizes
        ):
            raise ValueError("temporal_kernel_sizes must contain two positive odd values")
        if int(temporal_segments) <= 0:
            raise ValueError("temporal_segments must be positive")
        if candidate_local_mixing_placement not in {"pre_cross", "post_cross"}:
            raise ValueError(
                "candidate_local_mixing_placement must be pre_cross or post_cross"
            )
        if align_spectral_grid_to_classes:
            ordered_frequencies = sorted({float(value) for value in class_frequencies})
            if len(ordered_frequencies) < 2:
                raise ValueError("class-grid alignment requires at least two frequencies")
            minimum_spacing = min(
                right - left
                for left, right in zip(
                    ordered_frequencies[:-1], ordered_frequencies[1:], strict=True,
                )
            )
            if minimum_spacing <= 0.0:
                raise ValueError("class frequencies must be distinct")
            spectral_resolution_hz = min(
                float(spectral_resolution_hz), 0.5 * minimum_spacing,
            )
            harmonic_bias_width_hz = min(
                float(harmonic_bias_width_hz), minimum_spacing,
            )
        self.spectral_resolution_hz = float(spectral_resolution_hz)
        self.harmonic_bias_width_hz = float(harmonic_bias_width_hz)
        self.sample_rate = float(sample_rate)
        self.duration_conditioning_enabled = bool(duration_conditioning)
        self.candidate_local_mixing_enabled = bool(candidate_local_mixing)
        self.candidate_local_mixing_placement = candidate_local_mixing_placement
        self.register_buffer(
            "candidate_frequency_hz",
            torch.tensor(class_frequencies, dtype=torch.float32),
            persistent=False,
        )
        phases = list(class_phases or [0.0] * len(class_frequencies))
        if len(phases) != len(class_frequencies):
            raise ValueError("class_phases must match class_frequencies")
        candidate_order = sorted(
            range(len(class_frequencies)),
            key=lambda index: (float(class_frequencies[index]), float(phases[index])),
        )
        candidate_inverse_order = [0] * len(candidate_order)
        for ordered_index, original_index in enumerate(candidate_order):
            candidate_inverse_order[original_index] = ordered_index
        self.register_buffer(
            "candidate_frequency_order",
            torch.tensor(candidate_order, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "candidate_frequency_inverse_order",
            torch.tensor(candidate_inverse_order, dtype=torch.long),
            persistent=False,
        )
        self.local_domain = local_domain
        self.evidence_reliability_gate_enabled = bool(evidence_reliability_gate)
        self.decision_reliability_gate_enabled = bool(decision_reliability_gate)
        self.spectrum = ComplexSpectrumFrontEnd(
            sample_rate,
            resolution_hz=self.spectral_resolution_hz,
            band_hz=spectral_band_hz,
        )
        subband_low_hz = tuple(float(value) for value in (spectral_subband_low_hz or ()))
        if subband_low_hz and any(
            value < float(spectral_band_hz[0]) or value >= float(spectral_band_hz[1])
            for value in subband_low_hz
        ):
            raise ValueError("spectral subband cutoffs must lie inside spectral_band_hz")
        self.spectral_subband_low_hz = subband_low_hz
        self.register_buffer(
            "spectral_bin_hz",
            float(spectral_band_hz[0]) + torch.arange(
                self.spectrum.end - self.spectrum.start, dtype=torch.float32,
            ) * self.spectral_resolution_hz,
            persistent=False,
        )
        spectral_input_groups = len(subband_low_hz) if subband_low_hz else 1
        self.channel_projection = nn.Sequential(
            nn.Conv1d(channels * 3 * spectral_input_groups, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.local_stage1 = nn.ModuleList()
        self.local_stage2 = nn.ModuleList()
        self.input_local_stage1 = nn.ModuleList()
        self.input_local_stage2 = nn.ModuleList()
        if local_domain in {"spectral", "dual"}:
            self.local_stage1.extend(
                RepEEGBlock(width, kernel_size=9, dropout=dropout, gate_logit_init=0.0)
                for _ in range(local_depths[0])
            )
        if local_domain in {"temporal", "dual"}:
            self.input_local_stage1.extend(
                RepEEGBlock(
                    channels, kernel_size=int(temporal_kernel_sizes[0]),
                    dropout=dropout, gate_logit_init=0.0,
                )
                for _ in range(local_depths[0])
            )
        self.downsample = DepthwiseDownsample1d(width, dim, stride=2)
        if local_domain in {"spectral", "dual"}:
            self.local_stage2.extend(
                RepEEGBlock(dim, kernel_size=5, dropout=dropout, gate_logit_init=0.0)
                for _ in range(local_depths[1])
            )
        if local_domain in {"temporal", "dual"}:
            self.input_local_stage2.extend(
                RepEEGBlock(
                    channels, kernel_size=int(temporal_kernel_sizes[1]),
                    dropout=dropout, gate_logit_init=0.0,
                )
                for _ in range(local_depths[1])
            )
        self.temporal_candidate_features = CandidateFeatureExtractor(
            channels,
            dim,
            sample_rate,
            class_frequencies,
            class_phases,
            harmonics=harmonics,
            segments=int(temporal_segments),
            phase_dynamics=temporal_phase_dynamics,
        )
        self.spectral_candidate_features = SpectralCandidateExtractor(
            channels,
            dim,
            sample_rate,
            class_frequencies,
            class_phases,
            resolution_hz=self.spectral_resolution_hz,
            harmonics=harmonics,
            neighborhood_bins=neighborhood_bins,
        )
        self.candidate_fusion = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(),
        )
        if self.duration_conditioning_enabled:
            self.duration_projection = nn.Sequential(
                nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.duration_projection[-1].weight)
            nn.init.zeros_(self.duration_projection[-1].bias)
        else:
            self.duration_projection = None
        self.candidate_local_stage = (
            RepEEGBlock(dim, kernel_size=3, dropout=dropout, gate_logit_init=0.0)
            if self.candidate_local_mixing_enabled else None
        )
        if self.evidence_reliability_gate_enabled:
            self.evidence_reliability_gate = nn.Sequential(
                nn.LayerNorm(dim * 3),
                nn.Linear(dim * 3, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
            )
            nn.init.zeros_(self.evidence_reliability_gate[-1].weight)
            nn.init.constant_(self.evidence_reliability_gate[-1].bias, -1.5)
        else:
            self.evidence_reliability_gate = None
        if self.decision_reliability_gate_enabled:
            # Five bounded, sample-level probability statistics control how
            # much of the global/harmonic residual is trusted.  The gate is
            # deliberately downstream of candidate scoring: it never builds a
            # second temporal/frequency encoder and adds only 113 parameters.
            self.decision_reliability_gate = nn.Sequential(
                nn.Linear(5, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            nn.init.zeros_(self.decision_reliability_gate[-1].weight)
            nn.init.zeros_(self.decision_reliability_gate[-1].bias)
        else:
            self.decision_reliability_gate = None
        spectral_bins = self.spectrum.end - self.spectrum.start
        reduced_bins = (spectral_bins + 1) // 2
        token_hz = float(spectral_band_hz[0]) + torch.arange(
            reduced_bins, dtype=torch.float32,
        ) * (2.0 * self.spectral_resolution_hz)
        normalized_hz = token_hz / (0.5 * float(sample_rate))
        position_features = torch.stack(
            (
                normalized_hz,
                normalized_hz.square(),
                torch.sin(2.0 * torch.pi * normalized_hz),
                torch.cos(2.0 * torch.pi * normalized_hz),
                torch.sin(4.0 * torch.pi * normalized_hz),
                torch.cos(4.0 * torch.pi * normalized_hz),
                torch.sin(8.0 * torch.pi * normalized_hz),
                torch.cos(8.0 * torch.pi * normalized_hz),
            ),
            dim=-1,
        )
        self.register_buffer("spectral_position_features", position_features, persistent=False)
        self.position_projection = nn.Sequential(
            nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.fusion_attention = HarmonicCrossAttention(
            dim,
            heads,
            class_frequencies,
            token_hz,
            harmonics=harmonics,
            bias_width_hz=self.harmonic_bias_width_hz,
            dropout=dropout,
        )
        self.candidate_attention = AttentionBlock(dim, heads=heads, dropout=dropout)
        self.score_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )

    def _spectral_tokens(
        self, spectral_features: torch.Tensor, *, use_local: bool,
    ) -> torch.Tensor:
        x = self.channel_projection(spectral_features)
        if use_local:
            for block in self.local_stage1:
                x = block(x)
        x = self.downsample(x)
        if use_local:
            for block in self.local_stage2:
                x = block(x)
        tokens = x.transpose(1, 2)
        position = self.position_projection(
            self.spectral_position_features.to(dtype=tokens.dtype)
        )
        return tokens + position.unsqueeze(0)

    def _spectral_stem_features(
        self, spectrum: torch.Tensor, spectral_features: torch.Tensor,
    ) -> torch.Tensor:
        """Create nested subband views for one shared local spectral stem.

        Unlike a filter-bank ensemble, this does not instantiate separate
        fundamental/harmonic convolution branches.  Deterministic masks only
        expose progressively higher-frequency evidence to the same 1-D local
        mixer, preserving the compact deployment graph.
        """
        if not self.spectral_subband_low_hz:
            return spectral_features
        with torch.autocast(device_type=spectrum.device.type, enabled=False):
            band = spectrum.to(torch.complex64)[..., self.spectrum.start:self.spectrum.end]
            features = []
            for low_hz in self.spectral_subband_low_hz:
                mask = (self.spectral_bin_hz >= low_hz).to(
                    device=band.device, dtype=band.real.dtype,
                ).view(1, 1, -1)
                masked = band * mask
                features.append(torch.cat((
                    masked.real,
                    masked.imag,
                    torch.log1p(10.0 * masked.abs()),
                ), dim=1))
        return torch.cat(features, dim=1).to(dtype=spectral_features.dtype)

    def _locally_mixed_signal(
        self, x: torch.Tensor, *, use_local: bool,
    ) -> torch.Tensor:
        if not use_local or self.local_domain not in {"temporal", "dual"}:
            return x
        for block in self.input_local_stage1:
            x = block(x)
        for block in self.input_local_stage2:
            x = block(x)
        return x

    def _duration_conditioning(
        self, *, samples: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.duration_projection is None:
            raise RuntimeError("duration conditioning is disabled")
        duration = torch.tensor(
            float(samples) / self.sample_rate, device=device, dtype=torch.float32,
        )
        frequencies = self.candidate_frequency_hz.to(device=device)
        cycles = frequencies * duration
        features = torch.stack(
            (
                (duration / (1.0 + duration)).expand_as(cycles),
                torch.log1p(duration).expand_as(cycles),
                cycles / (1.0 + cycles),
                torch.log1p(cycles),
            ),
            dim=-1,
        )
        return self.duration_projection(features.to(dtype=dtype)).unsqueeze(0)

    def _locally_mix_candidates(
        self, candidates: torch.Tensor, *, use_local: bool,
    ) -> torch.Tensor:
        """Contrast neighboring stimulus candidates in physical frequency order.

        Dataset class indices need not be frequency-sorted.  The local mixer
        therefore operates on a derived frequency/phase order and restores the
        original class order before attention and scoring.
        """
        if not use_local or self.candidate_local_stage is None:
            return candidates
        ordered = candidates.index_select(1, self.candidate_frequency_order)
        ordered = self.candidate_local_stage(ordered.transpose(1, 2)).transpose(1, 2)
        return ordered.index_select(1, self.candidate_frequency_inverse_order)

    def harmonic_attention_probabilities(
        self,
        x: torch.Tensor,
        *,
        use_harmonic_bias: bool = True,
    ) -> torch.Tensor:
        """Expose the late cross-attention map for mechanism analysis only."""
        local_x = self._locally_mixed_signal(x, use_local=True)
        spectrum, spectral_features = self.spectrum(local_x)
        spectral_features = self._spectral_stem_features(spectrum, spectral_features)
        spectral = self._spectral_tokens(spectral_features, use_local=True)
        temporal = self.temporal_candidate_features(local_x)
        frequency = self.spectral_candidate_features(spectrum, output_dtype=x.dtype)
        candidates = self.candidate_fusion(torch.cat((temporal, frequency), dim=-1))
        if self.duration_projection is not None:
            candidates = candidates + self._duration_conditioning(
                samples=x.shape[-1], device=x.device, dtype=candidates.dtype,
            )
        if self.candidate_local_mixing_placement == "pre_cross":
            candidates = self._locally_mix_candidates(candidates, use_local=True)
        return self.fusion_attention.attention_probabilities(
            candidates,
            spectral,
            use_harmonic_bias=use_harmonic_bias,
        )

    def forward_mode(self, x: torch.Tensor, mode: str = "full") -> torch.Tensor:
        if mode not in set(HARMONIC_FOLD_MODES):
            raise ValueError(f"unsupported HarmonicFold v4 mode: {mode}")
        local_x = self._locally_mixed_signal(x, use_local=mode != "no_local")
        spectrum, spectral_features = self.spectrum(local_x)
        spectral_features = self._spectral_stem_features(spectrum, spectral_features)
        spectral = self._spectral_tokens(
            spectral_features, use_local=mode != "no_local",
        )
        temporal = self.temporal_candidate_features(local_x)
        frequency = self.spectral_candidate_features(spectrum, output_dtype=x.dtype)
        candidates = self.candidate_fusion(torch.cat((temporal, frequency), dim=-1))
        if self.duration_projection is not None:
            candidates = candidates + self._duration_conditioning(
                samples=x.shape[-1], device=x.device, dtype=candidates.dtype,
            )
        if self.candidate_local_mixing_placement == "pre_cross":
            candidates = self._locally_mix_candidates(
                candidates, use_local=mode != "no_local",
            )
        if mode != "no_attention":
            base_candidates = candidates
            refined = candidates
            if mode != "no_cross_attention":
                refined = self.fusion_attention(
                    refined,
                    spectral,
                    use_harmonic_bias=mode != "no_harmonic_bias",
                )
            if self.candidate_local_mixing_placement == "post_cross":
                refined = self._locally_mix_candidates(
                    refined, use_local=mode != "no_local",
                )
            if mode != "no_candidate_attention":
                refined = self.candidate_attention(refined)
            if self.decision_reliability_gate is not None:
                base_logits = self.score_head(base_candidates).squeeze(-1)
                refined_logits = self.score_head(refined).squeeze(-1)
                base_probabilities = torch.softmax(base_logits.float(), dim=-1)
                refined_probabilities = torch.softmax(refined_logits.float(), dim=-1)
                normalizer = math.log(float(base_logits.shape[-1]))
                base_entropy = -(
                    base_probabilities * base_probabilities.clamp_min(1e-8).log()
                ).sum(dim=-1) / normalizer
                refined_entropy = -(
                    refined_probabilities * refined_probabilities.clamp_min(1e-8).log()
                ).sum(dim=-1) / normalizer
                base_top2 = base_probabilities.topk(k=2, dim=-1).values
                refined_top2 = refined_probabilities.topk(k=2, dim=-1).values
                midpoint = 0.5 * (base_probabilities + refined_probabilities)
                js_divergence = 0.5 * (
                    (
                        base_probabilities
                        * (
                            base_probabilities.clamp_min(1e-8).log()
                            - midpoint.clamp_min(1e-8).log()
                        )
                    ).sum(dim=-1)
                    + (
                        refined_probabilities
                        * (
                            refined_probabilities.clamp_min(1e-8).log()
                            - midpoint.clamp_min(1e-8).log()
                        )
                    ).sum(dim=-1)
                )
                reliability_features = torch.stack(
                    (
                        1.0 - base_entropy,
                        1.0 - refined_entropy,
                        base_top2[:, 0] - base_top2[:, 1],
                        refined_top2[:, 0] - refined_top2[:, 1],
                        1.0 - js_divergence / math.log(2.0),
                    ),
                    dim=-1,
                ).to(dtype=base_logits.dtype)
                reliability = torch.sigmoid(
                    self.decision_reliability_gate(reliability_features)
                )
                return base_logits + reliability * (refined_logits - base_logits)
            if self.evidence_reliability_gate is None:
                candidates = refined
            else:
                agreement_features = torch.cat(
                    (temporal, frequency, (temporal - frequency).abs()), dim=-1,
                )
                reliability = torch.sigmoid(
                    self.evidence_reliability_gate(agreement_features)
                )
                candidates = candidates + reliability * (refined - candidates)
        elif self.candidate_local_mixing_placement == "post_cross":
            candidates = self._locally_mix_candidates(
                candidates, use_local=mode != "no_local",
            )
        return self.score_head(candidates).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_mode(x, "full")


class SpectralFusionDecoder(nn.Module):
    """local-to-global complex-spectrum hybrid for calibration-free SSVEP.

    Local re-parameterizable mixing operates on the frequency axis before
    token reduction and late attention. Candidate scoring is shared across
    datasets, so the architecture is not tied to one fixed label count.
    """

    def __init__(
        self,
        channels: int,
        sample_rate: int,
        class_frequencies: list[float] | tuple[float, ...],
        class_phases: list[float] | tuple[float, ...] | None = None,
        *,
        width: int = 48,
        local_depths: tuple[int, int] = (1, 1),
        attention_depth: int = 1,
        heads: int = 4,
        harmonics: int = 4,
        neighborhood_bins: int = 2,
        spectral_resolution_hz: float = 0.25,
        spectral_band_hz: tuple[float, float] = (6.0, 45.0),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        dim = width * 2
        if dim % heads:
            raise ValueError("2 * width must be divisible by heads")
        self.spectrum = ComplexSpectrumFrontEnd(
            sample_rate,
            resolution_hz=spectral_resolution_hz,
            band_hz=spectral_band_hz,
        )
        self.channel_projection = nn.Sequential(
            nn.Conv1d(channels * 3, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.local_stage1 = nn.ModuleList(
            RepEEGBlock(width, kernel_size=9, dropout=dropout, gate_logit_init=0.0)
            for _ in range(local_depths[0])
        )
        self.downsample = DepthwiseDownsample1d(width, dim, stride=2)
        self.local_stage2 = nn.ModuleList(
            RepEEGBlock(dim, kernel_size=5, dropout=dropout, gate_logit_init=0.0)
            for _ in range(local_depths[1])
        )
        self.spectral_attention = nn.ModuleList(
            AttentionBlock(dim, heads=heads, dropout=dropout)
            for _ in range(attention_depth)
        )
        self.candidate_features = SpectralCandidateExtractor(
            channels,
            dim,
            sample_rate,
            class_frequencies,
            class_phases,
            resolution_hz=spectral_resolution_hz,
            harmonics=harmonics,
            neighborhood_bins=neighborhood_bins,
        )
        self.temporal_candidate_features = CandidateFeatureExtractor(
            channels,
            dim,
            sample_rate,
            class_frequencies,
            class_phases,
            harmonics=harmonics,
            segments=4,
        )
        self.candidate_fusion = nn.Linear(dim * 2, dim)
        # Begin from the already validated segmented-demodulation path. The
        # complex-spectrum candidate path must earn a non-zero contribution
        # during training rather than perturbing the baseline at initialization.
        nn.init.zeros_(self.candidate_fusion.weight)
        nn.init.zeros_(self.candidate_fusion.bias)
        with torch.no_grad():
            self.candidate_fusion.weight[:, :dim].copy_(torch.eye(dim))
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True,
        )
        # Likewise, the global spectrum starts as a safe residual branch.
        nn.init.zeros_(self.cross_attention.out_proj.weight)
        nn.init.zeros_(self.cross_attention.out_proj.bias)
        self.candidate_attention = AttentionBlock(dim, heads=heads, dropout=dropout)
        self.pooled_context = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        spectral_bins = self.spectrum.end - self.spectrum.start
        reduced_bins = (spectral_bins + 1) // 2
        self.global_score_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(reduced_bins * dim, len(class_frequencies)),
        )
        nn.init.normal_(self.global_score_head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.global_score_head[-1].bias)
        self.global_score_scale = nn.Parameter(torch.tensor(-2.0))
        self.score_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )
        self.reference_prior = AnalyticReferencePrior(
            sample_rate, class_frequencies, harmonics=harmonics,
        )
        self.prior_scale = nn.Parameter(torch.tensor(1.0))

    def _spectral_tokens(
        self, spectral_features: torch.Tensor, *, use_local: bool,
    ) -> torch.Tensor:
        x = self.channel_projection(spectral_features)
        if use_local:
            for block in self.local_stage1:
                x = block(x)
        x = self.downsample(x)
        if use_local:
            for block in self.local_stage2:
                x = block(x)
        return x.transpose(1, 2)

    def forward_mode(self, x: torch.Tensor, mode: str = "full") -> torch.Tensor:
        if mode not in FUSION_MODES:
            raise ValueError(f"unsupported HarmonicFold mode: {mode}")
        spectrum, spectral_features = self.spectrum(x)
        spectral = self._spectral_tokens(
            spectral_features, use_local=mode != "no_local",
        )
        temporal_candidates = self.temporal_candidate_features(x)
        spectral_candidates = self.candidate_features(spectrum, output_dtype=x.dtype)
        candidates = self.candidate_fusion(
            torch.cat((temporal_candidates, spectral_candidates), dim=-1)
        )
        if mode == "no_attention":
            candidates = candidates + self.pooled_context(spectral.mean(dim=1)).unsqueeze(1)
        else:
            for block in self.spectral_attention:
                spectral = block(spectral)
            query = self.cross_norm(candidates)
            candidates = candidates + self.cross_attention(
                query, spectral, spectral, need_weights=False,
            )[0]
            candidates = self.candidate_attention(candidates)
        logits = self.score_head(candidates).squeeze(-1)
        logits = logits + F.softplus(self.global_score_scale) * self.global_score_head(spectral)
        if mode != "no_evidence":
            logits = logits + F.softplus(self.prior_scale) * self.reference_prior(x)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_mode(x, "full")


class TemporalFusionDecoder(nn.Module):
    """Original 1-D local-first, attention-late SSVEP hybrid.

    The implementation follows a local-to-global design constraint, but it does
    not reuse Apple source code, weights, image operators, or model outputs.
    """

    def __init__(
        self,
        channels: int,
        sample_rate: int,
        class_frequencies: list[float] | tuple[float, ...],
        class_phases: list[float] | tuple[float, ...] | None = None,
        *,
        width: int = 48,
        local_depths: tuple[int, int] = (2, 2),
        attention_depth: int = 1,
        heads: int = 4,
        harmonics: int = 4,
        demod_segments: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        dim = width * 2
        if dim % heads:
            raise ValueError("2 * width must be divisible by heads")
        self.channel_projection = nn.Sequential(
            nn.Conv1d(channels, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.stem = DepthwiseDownsample1d(width, width, stride=2)
        self.local_stage1 = nn.ModuleList(
            RepEEGBlock(width, kernel_size=15, dropout=dropout)
            for _ in range(local_depths[0])
        )
        self.downsample = DepthwiseDownsample1d(width, dim, stride=4)
        self.local_stage2 = nn.ModuleList(
            RepEEGBlock(dim, kernel_size=7, dropout=dropout)
            for _ in range(local_depths[1])
        )
        self.temporal_attention = nn.ModuleList(
            AttentionBlock(dim, heads=heads, dropout=dropout)
            for _ in range(attention_depth)
        )
        self.candidate_features = CandidateFeatureExtractor(
            channels,
            dim,
            sample_rate,
            class_frequencies,
            class_phases,
            harmonics=harmonics,
            segments=demod_segments,
        )
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True,
        )
        self.candidate_attention = AttentionBlock(dim, heads=heads, dropout=dropout)
        self.pooled_context = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.score_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )
        self.reference_prior = AnalyticReferencePrior(
            sample_rate, class_frequencies, harmonics=harmonics,
        )
        self.prior_scale = nn.Parameter(torch.tensor(1.0))

    def _temporal_tokens(self, x: torch.Tensor, *, use_local: bool) -> torch.Tensor:
        x = self.stem(self.channel_projection(x))
        if use_local:
            for block in self.local_stage1:
                x = block(x)
        x = self.downsample(x)
        if use_local:
            for block in self.local_stage2:
                x = block(x)
        return x.transpose(1, 2)

    def forward_mode(self, x: torch.Tensor, mode: str = "full") -> torch.Tensor:
        if mode not in FUSION_MODES:
            raise ValueError(f"unsupported HarmonicFold mode: {mode}")
        temporal = self._temporal_tokens(x, use_local=mode != "no_local")
        candidates = self.candidate_features(x)
        if mode == "no_attention":
            candidates = candidates + self.pooled_context(temporal.mean(dim=1)).unsqueeze(1)
        else:
            for block in self.temporal_attention:
                temporal = block(temporal)
            query = self.cross_norm(candidates)
            candidates = candidates + self.cross_attention(
                query, temporal, temporal, need_weights=False,
            )[0]
            candidates = self.candidate_attention(candidates)
        logits = self.score_head(candidates).squeeze(-1)
        if mode != "no_evidence":
            logits = logits + F.softplus(self.prior_scale) * self.reference_prior(x)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_mode(x, "full")
