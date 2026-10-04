import unittest

import numpy as np

from src.paper_baselines import cca_scores, fbcca_scores, fixed_harmonic_scores


class PaperBaselineTests(unittest.TestCase):
    def test_synthetic_frequency_is_recovered(self):
        sample_rate = 250
        samples = 500
        frequencies = [8.0, 10.0, 12.0, 14.0]
        time_axis = np.arange(samples) / sample_rate
        rng = np.random.default_rng(4)
        trials = []
        expected = []
        for label, frequency in enumerate(frequencies):
            signal = np.sin(2 * np.pi * frequency * time_axis)
            channels = np.stack([
                signal + 0.15 * rng.normal(size=samples) for _ in range(8)
            ])
            trials.append(channels.astype(np.float32))
            expected.append(label)
        x = np.stack(trials)
        for scorer in (fixed_harmonic_scores, cca_scores, fbcca_scores):
            with self.subTest(scorer=scorer.__name__):
                actual = scorer(x, frequencies, sample_rate).argmax(axis=1)
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
