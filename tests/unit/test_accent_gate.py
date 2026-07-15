import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.gate_joint_persona_mapper import main


class AccentGateTest(unittest.TestCase):
    def _run(self, stock_prob, candidate_prob, stock_frac=0.0, candidate_frac=0.0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.csv"
            fields = [
                "set",
                "mos_mean",
                "wer_mean",
                "sim_mean",
                "indian_frac",
                "indian_prob_mean",
            ]
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "set": "stock_xvc",
                        "mos_mean": 3.1,
                        "wer_mean": 0.01,
                        "sim_mean": 0.66,
                        "indian_frac": stock_frac,
                        "indian_prob_mean": stock_prob,
                    }
                )
                writer.writerow(
                    {
                        "set": "joint_persona_mapper",
                        "mos_mean": 3.1,
                        "wer_mean": 0.01,
                        "sim_mean": 0.66,
                        "indian_frac": candidate_frac,
                        "indian_prob_mean": candidate_prob,
                    }
                )
            output = root / "gate.json"
            status = main(["--summary", str(summary), "--out", str(output)])
            return status, json.loads(output.read_text(encoding="utf-8"))

    def test_continuous_indian_gain_can_pass_without_argmax_flip(self):
        status, result = self._run(0.03, 0.06)
        self.assertEqual(status, 0)
        self.assertEqual(result["status"], "pass")
        self.assertAlmostEqual(result["deltas"]["indian_prob"], 0.03)

    def test_single_hard_flip_does_not_override_weak_probability_gain(self):
        status, result = self._run(0.03, 0.035, candidate_frac=0.05)
        self.assertEqual(status, 1)
        self.assertIn("Indian posterior gain", result["failures"][0])


if __name__ == "__main__":
    unittest.main()
