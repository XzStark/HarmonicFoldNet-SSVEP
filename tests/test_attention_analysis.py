import json
import tempfile
import unittest
from pathlib import Path

from src.aggregate_attention_analysis import METRICS, aggregate


def record(offset: float) -> dict:
    def mode(base: float) -> dict:
        return {
            "per_subject": {
                subject: {
                    metric: base + offset + subject_index * 0.01 + metric_index * 0.001
                    for metric_index, metric in enumerate(METRICS)
                }
                for subject_index, subject in enumerate(("1", "2"))
            }
        }

    return {
        "dataset": "synthetic",
        "window_seconds": 0.8,
        "with_harmonic_bias": mode(0.2),
        "same_checkpoint_without_harmonic_bias": mode(0.4),
    }


class AttentionAggregationTests(unittest.TestCase):
    def test_repeated_seeds_are_averaged_by_subject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for index, offset in enumerate((0.0, 0.02)):
                path = root / f"seed-{index}.json"
                path.write_text(json.dumps(record(offset)), encoding="utf-8")
                inputs.append(path)
            output = root / "summary.json"
            payload = aggregate(inputs, output)
            self.assertEqual(payload["checkpoints"], 2)
            self.assertEqual(payload["unique_subjects"], 2)
            self.assertEqual(payload["per_subject"]["1"]["replicates"], 2)
            self.assertAlmostEqual(
                payload["metrics"]["harmonic_neighborhood_mass"]["with_minus_without_mean"],
                -0.2,
            )
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
