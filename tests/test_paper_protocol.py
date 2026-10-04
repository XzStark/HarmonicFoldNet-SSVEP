import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.paper_data import crop_and_standardize, grouped_kfold_split, load_subjects_window, loso_split
from src.paper_metrics import information_transfer_rate
from src.paper_train import distillation_kl_loss, parse_trial_filters


class PaperProtocolTests(unittest.TestCase):
    def test_parse_trial_filters_is_generic_and_repeatable(self):
        self.assertEqual(
            parse_trial_filters(["electrode=dry", "site=A,B"]),
            {"electrode": ["dry"], "site": ["A", "B"]},
        )
        self.assertIsNone(parse_trial_filters(None))

    def test_window_is_normalized_without_future_samples(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(2, 3, 500)).astype(np.float32)
        first = crop_and_standardize(x, 1.0, 250)
        changed_future = x.copy()
        changed_future[..., 250:] += 1_000_000
        second = crop_and_standardize(changed_future, 1.0, 250)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first.mean(axis=-1), 0.0, atol=1e-6)
        np.testing.assert_allclose(first.std(axis=-1), 1.0, atol=1e-6)

    def test_loader_selects_channels_and_trial_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = np.arange(4 * 3 * 20, dtype=np.float32).reshape(4, 3, 20)
            np.savez_compressed(
                root / "sub-A.npz", x=x, y=np.arange(4), sample_rate=np.asarray(10),
                channels=np.asarray(["A", "B", "C"]), condition=np.asarray(["x", "y", "x", "y"]),
            )
            selected, labels, subjects = load_subjects_window(
                root, ["A"], window_seconds=1.0, sample_rate=10,
                channels=["C", "A"], trial_filters={"condition": ["x"]},
            )
            self.assertEqual(selected.shape, (2, 2, 10))
            np.testing.assert_array_equal(labels, [0, 2])
            np.testing.assert_array_equal(subjects, ["A", "A"])

    def test_loso_is_disjoint_deterministic_and_accepts_non_numeric_ids(self):
        subjects = ["sub-a", "sub-b", "sub-c", "sub-d", "sub-e"]
        split1 = loso_split(subjects, "sub-c", validation_count=1, split_seed=9)
        split2 = loso_split(subjects, "sub-c", validation_count=1, split_seed=9)
        self.assertEqual(split1, split2)
        train, validation, test = map(set, split1)
        self.assertFalse(train & validation or train & test or validation & test)
        self.assertEqual(train | validation | test, set(subjects))

    def test_itr_boundary_conditions(self):
        self.assertEqual(information_transfer_rate(4, 0.25, 1.0), 0.0)
        self.assertAlmostEqual(information_transfer_rate(4, 1.0, 1.0), 120.0)

    def test_grouped_folds_cover_every_subject_once(self):
        subjects = [str(index) for index in range(23)]
        all_test = []
        for fold in range(5):
            train, validation, test = grouped_kfold_split(
                subjects, fold_index=fold, fold_count=5,
                validation_count=3, split_seed=12,
            )
            self.assertFalse(set(train) & set(validation))
            self.assertFalse(set(train) & set(test))
            self.assertFalse(set(validation) & set(test))
            self.assertEqual(set(train) | set(validation) | set(test), set(subjects))
            all_test.extend(test)
        self.assertCountEqual(all_test, subjects)

    def test_distillation_kl_is_zero_for_equal_logits_and_positive_otherwise(self):
        student = torch.tensor([[2.0, -1.0, 0.5]])
        equal = distillation_kl_loss(student, student.clone(), temperature=2.0)
        different = distillation_kl_loss(
            student, torch.tensor([[-1.0, 2.0, 0.5]]), temperature=2.0,
        )
        self.assertAlmostEqual(float(equal), 0.0, places=6)
        self.assertGreater(float(different), 0.0)
        with self.assertRaises(ValueError):
            distillation_kl_loss(student, student, temperature=0.0)


if __name__ == "__main__":
    unittest.main()
