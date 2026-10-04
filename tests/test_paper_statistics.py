import unittest

import numpy as np

from src.paper_statistics import holm_adjust, merge_out_of_fold_documents, paired_test, summarize


class PaperStatisticsTests(unittest.TestCase):
    def test_holm_is_monotone_and_bounded(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        self.assertGreaterEqual(adjusted[1], 0.04)

    def test_paired_test_uses_participant_values(self):
        result = paired_test(np.asarray([0.8, 0.7, 0.9]), np.asarray([0.5, 0.6, 0.4]))
        self.assertEqual(result["participants"], 3)
        self.assertGreater(result["mean_difference"], 0)

    def test_bootstrap_summary(self):
        result = summarize(np.asarray([0.2, 0.4, 0.6]), seed=1, bootstrap_samples=100)
        self.assertAlmostEqual(result["mean"], 0.4)
        self.assertLessEqual(result["bootstrap_mean_ci95_low"], result["mean"])
        self.assertGreaterEqual(result["bootstrap_mean_ci95_high"], result["mean"])

    def test_grouped_folds_merge_once_per_subject_and_seed(self):
        def document(seed, subjects):
            return {
                "seed": seed,
                "test": {"1.0": {"per_subject": {
                    subject: {"accuracy": value} for subject, value in subjects.items()
                }}},
            }

        documents = [
            document(1, {"1": 0.5}), document(1, {"2": 0.7}),
            document(2, {"1": 0.6}), document(2, {"2": 0.8}),
        ]
        subjects, values, seeds = merge_out_of_fold_documents(
            documents, window="1.0", metric="accuracy",
        )
        self.assertEqual(subjects, ["1", "2"])
        self.assertEqual(seeds, [1, 2])
        np.testing.assert_allclose(values, [[0.5, 0.7], [0.6, 0.8]])

    def test_grouped_folds_reject_duplicate_subject(self):
        documents = [
            {"seed": 1, "test": {"1.0": {"per_subject": {"1": {"accuracy": 0.5}}}}},
            {"seed": 1, "test": {"1.0": {"per_subject": {"1": {"accuracy": 0.6}}}}},
        ]
        with self.assertRaises(ValueError):
            merge_out_of_fold_documents(documents, window="1.0", metric="accuracy")


if __name__ == "__main__":
    unittest.main()
