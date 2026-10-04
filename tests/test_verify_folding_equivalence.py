from __future__ import annotations

import numpy as np

from scripts.verify_folding_equivalence import _representative_indices


def test_representative_indices_cover_subjects_and_classes() -> None:
    subjects = np.asarray(["1", "1", "1", "1", "2", "2", "2", "2"])
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    indices = _representative_indices(subjects, labels)
    assert indices.tolist() == [0, 2, 4, 6]
    assert set(subjects[indices]) == {"1", "2"}
    assert set(labels[indices]) == {0, 1}
