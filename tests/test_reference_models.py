import unittest

import torch

from src.reference_models import EEGNetAdaptive, FBSSVEPFormerReference, SSVEPFormerReference


class ReferenceModelTests(unittest.TestCase):
    def test_reference_models_accept_registered_window_lengths(self):
        for model in (
            EEGNetAdaptive(channels=8, classes=12),
            SSVEPFormerReference(channels=8, classes=12),
            FBSSVEPFormerReference(channels=8, classes=12),
        ):
            model.eval()
            for samples in (100, 250, 500):
                with self.subTest(model=type(model).__name__, samples=samples):
                    output = model(torch.randn(2, 8, samples))
                    self.assertEqual(tuple(output.shape), (2, 12))

    def test_fb_ssvepformer_exposes_three_independent_subbands(self):
        model = FBSSVEPFormerReference(channels=8, classes=12)
        model.eval()
        x = torch.randn(2, 8, 100)
        with torch.no_grad():
            branch_logits = model.subnetwork_logits(x)
            fused = model(x)
        self.assertEqual(tuple(branch_logits.shape), (2, 3, 12))
        self.assertEqual(tuple(fused.shape), (2, 12))
        self.assertFalse(
            torch.equal(
                model.subnetworks[0].head[-1].weight,
                model.subnetworks[1].head[-1].weight,
            )
        )


if __name__ == "__main__":
    unittest.main()
