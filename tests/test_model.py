import unittest

import torch

from src.model import LegacyFusionNet, LegacyLocalAttentionNet, parameter_count, reparameterize_model


class ModelTests(unittest.TestCase):
    def test_shape_and_parameter_budget(self):
        model = LegacyLocalAttentionNet(channels=8, classes=4, width=32, local_depth=2,
                             attention_depth=1, heads=4)
        output = model(torch.randn(3, 8, 500))
        self.assertEqual((3, 4), tuple(output.shape))
        self.assertLess(parameter_count(model), 1_000_000)

    def test_fusion_shape_and_parameter_budget(self):
        model = LegacyFusionNet(
            channels=8, classes=4, sample_rate=250, width=32,
            local_depth=1, attention_depth=1, heads=4,
            class_frequencies=[8.0, 10.0, 12.0, 14.0],
        )
        output = model(torch.randn(3, 8, 500))
        self.assertEqual(tuple(output.shape), (3, 4))
        self.assertLess(parameter_count(model), 1_000_000)

    def test_reparameterized_output_matches(self):
        torch.manual_seed(7)
        model = LegacyFusionNet(
            channels=8, classes=4, sample_rate=250, width=32,
            local_depth=2, attention_depth=1, heads=4, dropout=0.0,
            class_frequencies=[8.0, 10.0, 12.0, 14.0],
        ).eval()
        x = torch.randn(2, 8, 500)
        expected = model(x)
        actual = reparameterize_model(model)(x)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_registered_ablation_modes_have_expected_shape(self):
        model = LegacyFusionNet(
            channels=8, classes=4, sample_rate=250, width=16,
            local_depth=1, attention_depth=1, heads=4, dropout=0.0,
            class_frequencies=[8.0, 10.0, 12.0, 14.0],
        ).eval()
        x = torch.randn(2, 8, 250)
        for mode in ("full", "evidence_only", "fusion_only", "time_only", "frequency_only"):
            with self.subTest(mode=mode):
                self.assertEqual(tuple(model.forward_mode(x, mode).shape), (2, 4))

    def test_phase_complete_evidence_recovers_synthetic_targets(self):
        frequencies = [8.0, 10.0, 12.0, 14.0]
        model = LegacyFusionNet(
            channels=8, classes=4, sample_rate=250, width=16,
            local_depth=1, attention_depth=1, heads=4, dropout=0.0,
            class_frequencies=frequencies, evidence_harmonics=4,
            evidence_mode="cca_phase",
        ).eval()
        time_axis = torch.arange(250) / 250
        trials = torch.stack([
            torch.stack([torch.sin(2 * torch.pi * frequency * time_axis) for _ in range(8)])
            for frequency in frequencies
        ])
        prediction = model.forward_mode(trials, "evidence_only").argmax(dim=1)
        torch.testing.assert_close(prediction, torch.arange(4))

    def test_phase_demodulation_features_are_finite(self):
        model = LegacyFusionNet(
            channels=8, classes=4, sample_rate=250, width=16,
            local_depth=1, attention_depth=1, heads=4, dropout=0.0,
            class_frequencies=[8.0, 10.0, 12.0, 14.0],
            class_phases=[0.0, 0.5 * torch.pi, torch.pi, 1.5 * torch.pi],
            evidence_harmonics=4, evidence_mode="phase_demod",
        ).eval()
        features = model.phase_aligned_demodulation(torch.randn(2, 8, 250))
        self.assertEqual(tuple(features.shape), (2, 4, 8 * 4 * 3))
        self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
