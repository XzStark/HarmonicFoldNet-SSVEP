import unittest

import torch

from src.harmonic_fold import (
    FUSION_MODES,
    CandidateFeatureExtractor,
    SpectralFusionDecoder,
    HarmonicFoldNet,
    TemporalFusionDecoder,
    SpectralCandidateExtractor,
)
from src.model import parameter_count, reparameterize_model


def reference_temporal_features(
    extractor: CandidateFeatureExtractor, x: torch.Tensor,
) -> torch.Tensor:
    signal = x.float()
    samples = signal.shape[-1]
    time_axis = torch.arange(samples, dtype=torch.float32) / extractor.sample_rate
    harmonics = torch.arange(1, extractor.harmonics + 1, dtype=torch.float32)
    angles = (
        2.0 * torch.pi * extractor.class_frequencies[:, None, None]
        * harmonics[None, :, None] * time_axis[None, None, :]
        + extractor.class_phases[:, None, None] * harmonics[None, :, None]
    )
    real_segments = []
    imaginary_segments = []
    for segment in range(extractor.segments):
        begin = (segment * samples) // extractor.segments
        end = ((segment + 1) * samples) // extractor.segments
        length = max(end - begin, 1)
        segment_signal = signal[..., begin:end]
        real_segments.append(
            torch.einsum("bct,nht->bnch", segment_signal, torch.cos(angles[..., begin:end]))
            / length
        )
        imaginary_segments.append(
            torch.einsum("bct,nht->bnch", segment_signal, torch.sin(angles[..., begin:end]))
            / length
        )
    real = torch.stack(real_segments, dim=-1)
    imaginary = torch.stack(imaginary_segments, dim=-1)
    amplitude = torch.sqrt(real.square() + imaginary.square() + 1e-8)
    features = torch.stack(
        (real, imaginary, torch.log(amplitude + 1e-8)), dim=-1,
    ).flatten(2)
    features = (features - features.mean(dim=1, keepdim=True)) / features.std(
        dim=1, keepdim=True, correction=0,
    ).clamp_min(1e-5)
    return extractor.projection(features) + extractor._frequency_encoding(
        device=x.device, dtype=x.dtype,
    ).unsqueeze(0)


def reference_spectral_features(
    extractor: SpectralCandidateExtractor,
    spectrum: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    harmonic = torch.arange(1, extractor.harmonics + 1, dtype=torch.float32)
    offsets = torch.arange(
        -extractor.neighborhood_bins,
        extractor.neighborhood_bins + 1,
        dtype=torch.float32,
    ) * extractor.resolution_hz
    target_hz = extractor.class_frequencies[:, None, None] * harmonic[None, :, None]
    target_hz = target_hz + offsets[None, None, :]
    position = target_hz / extractor.resolution_hz
    lower = position.floor().long().clamp(0, spectrum.shape[-1] - 1)
    upper = (lower + 1).clamp_max(spectrum.shape[-1] - 1)
    fraction = (position - position.floor()).clamp(0.0, 1.0)
    lower_value = spectrum[..., lower.flatten()]
    upper_value = spectrum[..., upper.flatten()]
    sampled = lower_value * (1.0 - fraction.flatten()) + upper_value * fraction.flatten()
    valid = (target_hz >= 0.0) & (target_hz <= 0.5 * extractor.sample_rate)
    sampled = sampled * valid.flatten()
    batch = sampled.shape[0]
    classes = extractor.class_frequencies.numel()
    neighborhood = 2 * extractor.neighborhood_bins + 1
    sampled = sampled.view(
        batch, extractor.channels, classes, extractor.harmonics, neighborhood,
    ).permute(0, 2, 1, 3, 4)
    feature = torch.stack(
        (sampled.real, sampled.imag, torch.log1p(10.0 * sampled.abs())), dim=-1,
    ).flatten(2)
    feature = (feature - feature.mean(dim=1, keepdim=True)) / feature.std(
        dim=1, keepdim=True, correction=0,
    ).clamp_min(1e-5)
    return extractor.projection(feature.to(dtype=output_dtype)) + extractor._frequency_encoding(
        device=spectrum.device, dtype=output_dtype,
    ).unsqueeze(0)


class TemporalFusionDecoderTests(unittest.TestCase):
    def _model(self, classes: int = 12) -> TemporalFusionDecoder:
        return TemporalFusionDecoder(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(classes)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(classes)],
            width=16,
            local_depths=(1, 1),
            attention_depth=1,
            heads=4,
            harmonics=2,
            demod_segments=2,
            dropout=0.0,
        )

    def test_modes_and_variable_candidate_count(self):
        x = torch.randn(2, 8, 100)
        for classes in (12, 40):
            model = self._model(classes).eval()
            for mode in FUSION_MODES:
                with self.subTest(classes=classes, mode=mode):
                    self.assertEqual(tuple(model.forward_mode(x, mode).shape), (2, classes))

    def test_folded_deployment_graph_is_equivalent(self):
        torch.manual_seed(17)
        model = self._model().eval()
        x = torch.randn(2, 8, 250)
        expected = model(x)
        actual = reparameterize_model(model)(x)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)

    def test_parameter_budget_is_deployment_scale(self):
        model = TemporalFusionDecoder(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(40)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(40)],
            width=48,
            local_depths=(2, 2),
            attention_depth=1,
            heads=4,
            harmonics=4,
            demod_segments=4,
        )
        self.assertLess(parameter_count(model), 1_000_000)

    def test_temporal_candidate_segment_count_is_configurable(self):
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="temporal",
            temporal_segments=2,
            dropout=0.0,
        )
        self.assertEqual(model.temporal_candidate_features.segments, 2)
        self.assertEqual(tuple(model(torch.randn(2, 8, 100)).shape), (2, 12))

    def test_spectral_grid_can_follow_minimum_class_spacing(self):
        dense = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            align_spectral_grid_to_classes=True,
            dropout=0.0,
        )
        sparse = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.5 * index for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            align_spectral_grid_to_classes=True,
            dropout=0.0,
        )
        self.assertAlmostEqual(dense.spectral_resolution_hz, 0.1, places=5)
        self.assertAlmostEqual(dense.harmonic_bias_width_hz, 0.2, places=5)
        self.assertAlmostEqual(sparse.spectral_resolution_hz, 0.25, places=5)
        self.assertAlmostEqual(sparse.harmonic_bias_width_hz, 0.5, places=5)

    def test_duration_conditioning_is_continuous_and_window_sensitive(self):
        torch.manual_seed(97)
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            duration_conditioning=True,
            dropout=0.0,
        )
        self.assertIsNotNone(model.duration_projection)
        short = model._duration_conditioning(
            samples=100, device=torch.device("cpu"), dtype=torch.float32,
        )
        long = model._duration_conditioning(
            samples=300, device=torch.device("cpu"), dtype=torch.float32,
        )
        self.assertEqual(tuple(short.shape), (1, 12, 32))
        # The final projection is deliberately zero-initialized so the new
        # path begins as the frozen baseline, then receives gradients.
        torch.testing.assert_close(short, torch.zeros_like(short))
        model(torch.randn(2, 8, 100)).square().mean().backward()
        self.assertTrue(all(
            parameter.grad is not None
            for parameter in model.duration_projection.parameters()
        ))
        self.assertEqual(tuple(long.shape), tuple(short.shape))

    def test_candidate_local_mixer_uses_physical_frequency_order(self):
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[10.0, 8.0, 9.0, 8.0],
            class_phases=[0.0, 1.0, 0.0, 0.0],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            candidate_local_mixing=True,
            dropout=0.0,
        )
        self.assertEqual(model.candidate_frequency_order.tolist(), [3, 1, 2, 0])
        ordered = torch.arange(4)[model.candidate_frequency_order]
        restored = ordered[model.candidate_frequency_inverse_order]
        torch.testing.assert_close(restored, torch.arange(4))

    def test_candidate_local_mixer_is_foldable_and_mode_safe(self):
        torch.manual_seed(101)
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.4, 8.0, 8.6, 8.2, 8.8, 9.0],
            class_phases=[0.0] * 6,
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="temporal",
            candidate_local_mixing=True,
            dropout=0.0,
        ).eval()
        self.assertIsNotNone(model.candidate_local_stage)
        for samples in (100, 250):
            x = torch.randn(2, 8, samples)
            for mode in ("full", "no_attention", "no_local"):
                expected = model.forward_mode(x, mode)
                actual = reparameterize_model(model).forward_mode(x, mode)
                self.assertEqual(tuple(expected.shape), (2, 6))
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)

    def test_candidate_local_mixer_can_follow_harmonic_cross_attention(self):
        torch.manual_seed(103)
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.4, 8.0, 8.6, 8.2, 8.8, 9.0],
            class_phases=[0.0] * 6,
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="temporal",
            candidate_local_mixing=True,
            candidate_local_mixing_placement="post_cross",
            dropout=0.0,
        ).eval()
        x = torch.randn(2, 8, 150)
        for mode in ("full", "no_attention", "no_cross_attention", "no_local"):
            expected = model.forward_mode(x, mode)
            actual = reparameterize_model(model).forward_mode(x, mode)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)

    def test_candidate_local_mixer_stays_inside_parameter_budget(self):
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(40)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(40)],
            width=48,
            local_depths=(1, 1),
            heads=4,
            harmonics=4,
            neighborhood_bins=2,
            local_domain="temporal",
            candidate_local_mixing=True,
        )
        self.assertLess(parameter_count(model), 500_000)


class SpectralFusionDecoderTests(unittest.TestCase):
    def _model(self, classes: int = 12) -> SpectralFusionDecoder:
        return SpectralFusionDecoder(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(classes)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(classes)],
            width=16,
            local_depths=(1, 1),
            attention_depth=1,
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            dropout=0.0,
        )

    def test_modes_support_variable_windows_and_candidate_counts(self):
        for samples in (100, 250):
            x = torch.randn(2, 8, samples)
            for classes in (12, 40):
                model = self._model(classes).eval()
                for mode in FUSION_MODES:
                    with self.subTest(samples=samples, classes=classes, mode=mode):
                        self.assertEqual(tuple(model.forward_mode(x, mode).shape), (2, classes))

    def test_folded_deployment_graph_is_equivalent(self):
        torch.manual_seed(23)
        model = self._model().eval()
        x = torch.randn(2, 8, 250)
        expected = model(x)
        actual = reparameterize_model(model)(x)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)

    def test_parameter_budget_is_deployment_scale(self):
        model = SpectralFusionDecoder(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(40)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(40)],
            width=48,
            local_depths=(1, 1),
            attention_depth=1,
            heads=4,
            harmonics=4,
            neighborhood_bins=2,
        )
        self.assertLess(parameter_count(model), 1_000_000)


class CandidateExtractorOptimizationTests(unittest.TestCase):
    def test_vectorized_temporal_extractor_matches_segment_reference(self):
        torch.manual_seed(31)
        extractor = CandidateFeatureExtractor(
            channels=3,
            dim=16,
            sample_rate=250,
            class_frequencies=[8.0, 9.25, 12.0],
            class_phases=[0.0, 0.5, 1.0],
            harmonics=3,
            segments=4,
        ).eval()
        for samples in (100, 151, 250):
            x = torch.randn(2, 3, samples)
            with self.subTest(samples=samples):
                torch.testing.assert_close(
                    extractor(x), reference_temporal_features(extractor, x),
                    rtol=1e-5, atol=1e-6,
                )

    def test_cached_spectral_indices_match_direct_reference(self):
        torch.manual_seed(37)
        extractor = SpectralCandidateExtractor(
            channels=3,
            dim=16,
            sample_rate=250,
            class_frequencies=[8.0, 9.25, 12.0],
            class_phases=[0.0, 0.5, 1.0],
            resolution_hz=0.25,
            harmonics=3,
            neighborhood_bins=2,
        ).eval()
        spectrum = torch.fft.rfft(torch.randn(2, 3, 250), n=1000, dim=-1) / 250
        torch.testing.assert_close(
            extractor(spectrum, output_dtype=torch.float32),
            reference_spectral_features(extractor, spectrum, output_dtype=torch.float32),
            rtol=1e-5, atol=1e-6,
        )


class HarmonicFoldNetTests(unittest.TestCase):
    def _model(self, classes: int = 12) -> HarmonicFoldNet:
        return HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(classes)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(classes)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            dropout=0.0,
        )

    def test_modes_support_variable_windows_and_candidate_counts(self):
        for samples in (100, 250):
            x = torch.randn(2, 8, samples)
            for classes in (12, 40):
                model = self._model(classes).eval()
                for mode in (
                    "full", "no_attention", "no_local", "no_harmonic_bias",
                    "no_temporal_candidate", "no_spectral_candidate",
                ):
                    with self.subTest(samples=samples, classes=classes, mode=mode):
                        self.assertEqual(tuple(model.forward_mode(x, mode).shape), (2, classes))

    def test_spectral_token_stride_controls_reduced_sequence_length(self):
        x = torch.randn(2, 8, 250)
        lengths = {}
        for stride in (1, 2, 4):
            model = HarmonicFoldNet(
                channels=8,
                sample_rate=250,
                class_frequencies=[8.0 + 0.5 * index for index in range(9)],
                class_phases=[0.0] * 9,
                width=16,
                local_depths=(1, 1),
                heads=4,
                harmonics=2,
                neighborhood_bins=1,
                spectral_token_stride=stride,
                dropout=0.0,
            ).eval()
            spectrum, features = model.spectrum(x)
            del spectrum
            tokens = model._spectral_tokens(features, use_local=True)
            lengths[stride] = tokens.shape[1]
            self.assertEqual(tuple(model(x).shape), (2, 9))
        self.assertGreater(lengths[1], lengths[2])
        self.assertGreater(lengths[2], lengths[4])

    def test_folded_deployment_graph_is_equivalent(self):
        torch.manual_seed(41)
        model = self._model().eval()
        x = torch.randn(2, 8, 250)
        expected = model(x)
        actual = reparameterize_model(model)(x)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)

    def test_temporal_local_frontend_is_foldable_and_mode_safe(self):
        torch.manual_seed(42)
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="temporal",
            dropout=0.0,
        ).eval()
        x = torch.randn(2, 8, 150)
        for mode in ("full", "no_attention", "no_local"):
            expected = model.forward_mode(x, mode)
            actual = reparameterize_model(model).forward_mode(x, mode)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)

    def test_temporal_local_frontend_can_train_after_inference_cache_fill(self):
        torch.manual_seed(47)
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="temporal",
            dropout=0.0,
        )
        x = torch.randn(2, 8, 150)
        with torch.inference_mode():
            model.eval()(x)
        model.train()
        loss = model(x).square().mean()
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.input_local_stage1.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in gradients))

    def test_temporal_phase_dynamics_are_compact_variable_window_features(self):
        torch.manual_seed(51)
        common = dict(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="temporal",
            temporal_segments=4,
            dropout=0.0,
        )
        base = HarmonicFoldNet(**common).eval()
        candidate = HarmonicFoldNet(
            **common, temporal_phase_dynamics=True,
        ).eval()
        self.assertGreater(parameter_count(candidate), parameter_count(base))
        self.assertLess(parameter_count(candidate) - parameter_count(base), 12_000)
        for samples in (100, 200, 300):
            x = torch.randn(2, 8, samples)
            expected = candidate(x)
            folded = reparameterize_model(candidate)(x)
            self.assertEqual(tuple(expected.shape), (2, 12))
            torch.testing.assert_close(folded, expected, rtol=2e-4, atol=3e-5)

    def test_temporal_phase_dynamics_require_two_segments(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            CandidateFeatureExtractor(
                channels=8,
                dim=32,
                sample_rate=250,
                class_frequencies=[8.0, 8.2],
                class_phases=[0.0, 0.5 * torch.pi],
                harmonics=2,
                segments=1,
                phase_dynamics=True,
            )

    def test_dual_local_frontend_is_foldable(self):
        torch.manual_seed(53)
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="dual",
            dropout=0.0,
        ).eval()
        x = torch.randn(2, 8, 150)
        expected = model(x)
        actual = reparameterize_model(model)(x)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)

    def test_long_receptive_field_frontend_is_foldable(self):
        torch.manual_seed(59)
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="temporal",
            temporal_kernel_sizes=(31, 15),
            dropout=0.0,
        ).eval()
        x = torch.randn(2, 8, 100)
        expected = model(x)
        actual = reparameterize_model(model)(x)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)

    def test_attention_is_an_obligatory_non_identity_global_path(self):
        torch.manual_seed(43)
        model = self._model().eval()
        x = torch.randn(2, 8, 150)
        self.assertFalse(torch.equal(model.forward_mode(x, "full"), model.forward_mode(x, "no_attention")))

    def test_nested_subband_reliability_gate_is_variable_window_and_fold_safe(self):
        torch.manual_seed(71)
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="temporal",
            spectral_subband_low_hz=(6.0, 14.0, 22.0),
            evidence_reliability_gate=True,
            dropout=0.0,
        ).eval()
        self.assertEqual(model.channel_projection[0].in_channels, 8 * 3 * 3)
        for samples in (100, 200, 300):
            x = torch.randn(2, 8, samples)
            expected = model(x)
            actual = reparameterize_model(model)(x)
            self.assertEqual(tuple(expected.shape), (2, 12))
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)

    def test_reliability_gate_receives_gradients(self):
        torch.manual_seed(73)
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="temporal",
            spectral_subband_low_hz=(6.0, 14.0, 22.0),
            evidence_reliability_gate=True,
            dropout=0.0,
        )
        model(torch.randn(2, 8, 100)).square().mean().backward()
        gradients = [
            parameter.grad for parameter in model.evidence_reliability_gate.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))

    def test_decision_reliability_gate_is_fold_safe_and_trainable(self):
        torch.manual_seed(79)
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="temporal",
            decision_reliability_gate=True,
            dropout=0.0,
        )
        model.train()
        output = model(torch.randn(3, 8, 100))
        self.assertEqual(tuple(output.shape), (3, 12))
        output.square().mean().backward()
        gradients = [
            parameter.grad for parameter in model.decision_reliability_gate.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))

        model.eval()
        x = torch.randn(2, 8, 200)
        expected = model(x)
        actual = reparameterize_model(model)(x)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)

    def test_decision_gate_adds_only_113_parameters(self):
        common = dict(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="temporal",
            dropout=0.0,
        )
        base = HarmonicFoldNet(**common)
        gated = HarmonicFoldNet(
            **common,
            decision_reliability_gate=True,
        )
        self.assertEqual(parameter_count(gated) - parameter_count(base), 113)

    def test_decision_gate_interpolates_distinct_base_and_attention_paths(self):
        torch.manual_seed(83)
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(12)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(12)],
            width=16,
            local_depths=(1, 1),
            heads=4,
            harmonics=2,
            neighborhood_bins=1,
            local_domain="temporal",
            decision_reliability_gate=True,
            dropout=0.0,
        ).eval()
        x = torch.randn(2, 8, 150)
        base = model.forward_mode(x, "no_attention")
        with torch.no_grad():
            model.decision_reliability_gate[-1].weight.zero_()
            model.decision_reliability_gate[-1].bias.fill_(-100.0)
        low_reliability = model(x)
        with torch.no_grad():
            model.decision_reliability_gate[-1].bias.fill_(100.0)
        high_reliability = model(x)
        torch.testing.assert_close(low_reliability, base)
        self.assertFalse(torch.equal(high_reliability, low_reliability))

    def test_harmonic_bias_has_an_independent_ablation(self):
        torch.manual_seed(61)
        model = self._model().eval()
        x = torch.randn(2, 8, 150)
        full = model.forward_mode(x, "full")
        no_bias = model.forward_mode(x, "no_harmonic_bias")
        self.assertEqual(tuple(no_bias.shape), tuple(full.shape))
        self.assertFalse(torch.equal(full, no_bias))

    def test_cross_and_candidate_attention_paths_have_independent_ablations(self):
        torch.manual_seed(89)
        model = self._model().eval()
        x = torch.randn(2, 8, 150)
        full = model.forward_mode(x, "full")
        no_cross = model.forward_mode(x, "no_cross_attention")
        no_candidate = model.forward_mode(x, "no_candidate_attention")
        self.assertEqual(tuple(no_cross.shape), tuple(full.shape))
        self.assertEqual(tuple(no_candidate.shape), tuple(full.shape))
        self.assertFalse(torch.equal(full, no_cross))
        self.assertFalse(torch.equal(full, no_candidate))
        self.assertFalse(torch.equal(no_cross, no_candidate))

    def test_attention_probabilities_are_normalized(self):
        torch.manual_seed(67)
        model = self._model(classes=12).eval()
        query = torch.randn(2, 12, 32)
        context = torch.randn(2, model.fusion_attention.harmonic_bias.shape[1], 32)
        probabilities = model.fusion_attention.attention_probabilities(query, context)
        self.assertEqual(tuple(probabilities.shape[:3]), (2, 4, 12))
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones_like(probabilities[..., 0]),
        )
        x = torch.randn(2, 8, 150)
        end_to_end = model.harmonic_attention_probabilities(x)
        self.assertEqual(tuple(end_to_end.shape), tuple(probabilities.shape))
        torch.testing.assert_close(
            end_to_end.sum(dim=-1),
            torch.ones_like(end_to_end[..., 0]),
        )

    def test_parameter_budget_is_deployment_scale(self):
        model = HarmonicFoldNet(
            channels=8,
            sample_rate=250,
            class_frequencies=[8.0 + 0.2 * index for index in range(40)],
            class_phases=[0.5 * torch.pi * (index % 4) for index in range(40)],
            width=48,
            local_depths=(1, 1),
            heads=4,
            harmonics=4,
            neighborhood_bins=2,
        )
        self.assertLess(parameter_count(model), 1_000_000)


if __name__ == "__main__":
    unittest.main()
