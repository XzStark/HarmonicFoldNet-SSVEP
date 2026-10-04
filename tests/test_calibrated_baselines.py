import unittest

import numpy as np

from src.calibrated_baselines import EnsembleTRCA, TDCA, TRCA
from src.paper_calibrated_eval import _preceding_calibration_blocks


def synthetic_trials(*, blocks: int = 4, classes: int = 3, samples: int = 255):
    rng = np.random.default_rng(17)
    frequencies = np.asarray([8.0, 10.0, 12.0])[:classes]
    x, y, block_ids = [], [], []
    time = np.arange(samples) / 250.0
    mixing = rng.normal(size=(classes, 4))
    for block in range(blocks):
        for label, frequency in enumerate(frequencies):
            source = np.sin(2 * np.pi * frequency * time + 0.15 * block)
            trial = mixing[label, :, None] * source[None, :]
            trial += 0.08 * rng.normal(size=trial.shape)
            x.append(trial); y.append(label); block_ids.append(block)
    return np.asarray(x), np.asarray(y), np.asarray(block_ids), frequencies


class CalibratedBaselineTests(unittest.TestCase):
    def test_class_specific_trca_separates_held_out_blocks(self):
        x, y, blocks, _ = synthetic_trials()
        train = blocks != 3
        model = TRCA().fit(x[train, :, :250], y[train])
        np.testing.assert_array_equal(model.predict(x[~train, :, :250]), y[~train])

    def test_ensemble_trca_separates_held_out_blocks(self):
        x, y, blocks, _ = synthetic_trials()
        train = blocks != 3
        model = EnsembleTRCA().fit(x[train, :, :250], y[train])
        np.testing.assert_array_equal(model.predict(x[~train, :, :250]), y[~train])

    def test_tdca_supports_one_block_calibration(self):
        x, y, blocks, frequencies = synthetic_trials()
        train = blocks == 0
        model = TDCA(
            frequencies=frequencies, sample_rate=250, samples=250,
            harmonics=2, delay_samples=5,
        ).fit(x[train], y[train])
        prediction = model.predict(x[blocks == 1])
        self.assertGreaterEqual(float((prediction == y[blocks == 1]).mean()), 2 / 3)

    def test_trca_rejects_single_trial_per_class(self):
        x, y, blocks, _ = synthetic_trials()
        for estimator in (TRCA, EnsembleTRCA):
            with self.assertRaises(ValueError):
                estimator().fit(x[blocks == 0, :, :250], y[blocks == 0])

    def test_calibration_schedule_never_contains_held_out_block(self):
        blocks = ["0", "1", "2", "3"]
        for held_out_index, held_out in enumerate(blocks):
            for count in (1, 2, -1):
                selected = _preceding_calibration_blocks(blocks, held_out_index, count)
                self.assertNotIn(held_out, selected)
                self.assertEqual(len(selected), 3 if count == -1 else count)


if __name__ == "__main__":
    unittest.main()
