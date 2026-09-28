import unittest

import torch

from src.model import FastSSVEPFusionNet, FastSSVEPNet, parameter_count, reparameterize_model


class ModelTests(unittest.TestCase):
    def test_shape_and_parameter_budget(self):
        model = FastSSVEPNet(channels=8, classes=4, width=32, local_depth=2,
                             attention_depth=1, heads=4)
        output = model(torch.randn(3, 8, 500))
        self.assertEqual((3, 4), tuple(output.shape))
        self.assertLess(parameter_count(model), 1_000_000)

    def test_fusion_shape_and_parameter_budget(self):
        model = FastSSVEPFusionNet(
            channels=8, classes=4, sample_rate=250, width=32,
            local_depth=1, attention_depth=1, heads=4,
            class_frequencies=[8.0, 10.0, 12.0, 14.0],
        )
        output = model(torch.randn(3, 8, 500))
        self.assertEqual(tuple(output.shape), (3, 4))
        self.assertLess(parameter_count(model), 1_000_000)

    def test_reparameterized_output_matches(self):
        torch.manual_seed(7)
        model = FastSSVEPFusionNet(
            channels=8, classes=4, sample_rate=250, width=32,
            local_depth=2, attention_depth=1, heads=4, dropout=0.0,
            class_frequencies=[8.0, 10.0, 12.0, 14.0],
        ).eval()
        x = torch.randn(2, 8, 500)
        expected = model(x)
        actual = reparameterize_model(model)(x)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
