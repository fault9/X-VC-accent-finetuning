import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.gate_joint_persona_mapper import main


class AccentGateTest(unittest.TestCase):
    def _run(
        self,
        stock_prob,
        candidate_prob,
        stock_frac=0.0,
        candidate_frac=0.0,
        calibrated=False,
    ):
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
            argv = ["--summary", str(summary), "--out", str(output)]
            if calibrated:
                calibration = root / "calibration.csv"
                with calibration.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=["set", "indian_prob_mean"]
                    )
                    writer.writeheader()
                    writer.writerow(
                        {"set": "genuine_asi", "indian_prob_mean": 0.0865}
                    )
                    writer.writerow(
                        {"set": "native_unseen", "indian_prob_mean": 0.0636}
                    )
                argv.extend(["--calibration", str(calibration)])
            status = main(argv)
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

    def test_calibrated_gate_uses_fraction_of_human_gap(self):
        # Human gap is 0.0229; a gain of 0.006 closes about 26.2%.
        status, result = self._run(0.0636, 0.0696, calibrated=True)
        self.assertEqual(status, 0)
        self.assertGreater(
            result["accent_calibration"]["candidate_gap_closed"], 0.25
        )


if __name__ == "__main__":
    unittest.main()
