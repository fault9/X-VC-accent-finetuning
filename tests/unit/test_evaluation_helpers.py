"""CPU tests for the shared eval-runner/accent-gate helpers (stdlib only)."""

import csv
import json
from pathlib import Path

import pytest

from xvc.evaluation import (
    check_training_prompt_overlap,
    check_unseen_sources,
    prompt_from_name,
    run_accent_gate,
    speaker_from_name,
)


class TestNaming:
    def test_speaker_from_arctic_name(self):
        assert speaker_from_name(Path("CLB_arctic_a0001.wav")) == "clb"

    def test_speaker_fallback_without_marker(self):
        assert speaker_from_name(Path("Guest_take2.wav")) == "guest"

    def test_prompt_from_name(self):
        assert prompt_from_name(Path("slt_ARCTIC_B0538__persona.wav")) == "arctic_b0538"
        assert prompt_from_name(Path("no_prompt_here.wav")) is None


class TestUnseenSources:
    PATHS = [Path("clb_arctic_a0001.wav"), Path("slt_arctic_a0002.wav")]

    def test_disjoint_passes(self):
        eval_speakers, overlap = check_unseen_sources(
            self.PATHS, {"aup", "asi"}, require_unseen=True, min_unseen_speakers=2
        )
        assert eval_speakers == {"clb", "slt"}
        assert overlap == set()

    def test_overlap_fails_closed(self):
        with pytest.raises(SystemExit, match="seen in training"):
            check_unseen_sources(
                self.PATHS, {"CLB"}, require_unseen=True, min_unseen_speakers=2
            )

    def test_too_few_speakers_fails_closed(self):
        with pytest.raises(SystemExit, match="unseen evaluation speaker"):
            check_unseen_sources(
                self.PATHS[:1], set(), require_unseen=True, min_unseen_speakers=2
            )

    def test_overlap_reported_when_not_required(self):
        _, overlap = check_unseen_sources(
            self.PATHS, {"clb"}, require_unseen=False, min_unseen_speakers=2
        )
        assert overlap == {"clb"}


class TestPromptOverlap:
    def write_manifest(self, tmp_path, prompts):
        manifest = tmp_path / "train.jsonl"
        manifest.write_text(
            "\n".join(json.dumps({"prompt_id": p}) for p in prompts) + "\n",
            encoding="utf-8",
        )
        return str(manifest)

    def test_disjoint_passes(self, tmp_path):
        manifest = self.write_manifest(tmp_path, ["arctic_a0001"])
        paths = [Path("clb_arctic_b0100.wav")]
        assert check_training_prompt_overlap(paths, manifest) == []

    def test_overlap_fails_closed(self, tmp_path):
        manifest = self.write_manifest(tmp_path, ["arctic_a0001"])
        paths = [Path("clb_ARCTIC_A0001.wav")]
        with pytest.raises(SystemExit, match="overlaps training prompts"):
            check_training_prompt_overlap(paths, manifest)

    def test_no_manifest_is_noop(self):
        assert check_training_prompt_overlap([Path("x.wav")], None) == []


FIELDS = ["set", "mos_mean", "wer_mean", "sim_mean", "indian_frac", "indian_prob_mean"]


def write_summary(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TestAccentGate:
    def gate(self, tmp_path, candidate_row, **overrides):
        summary = tmp_path / "condition_summary.csv"
        write_summary(
            summary,
            [
                {"set": "stock_xvc", "mos_mean": 4.0, "wer_mean": 0.10,
                 "sim_mean": 0.80, "indian_frac": 0.1, "indian_prob_mean": 0.20},
                {"set": "persona_codec_teacher", **candidate_row},
            ],
        )
        calibration = tmp_path / "calibration.csv"
        write_summary(
            calibration,
            [
                {"set": "genuine_asi", "mos_mean": 4.0, "wer_mean": 0.1,
                 "sim_mean": 0.8, "indian_frac": 0.9, "indian_prob_mean": 0.80},
                {"set": "native_unseen", "mos_mean": 4.0, "wer_mean": 0.1,
                 "sim_mean": 0.8, "indian_frac": 0.0, "indian_prob_mean": 0.20},
            ],
        )
        kwargs = dict(
            summary=str(summary),
            out=str(tmp_path / "gate.json"),
            stock_set="stock_xvc",
            candidate_set="persona_codec_teacher",
            max_mos_drop=0.25,
            max_wer_increase=0.05,
            max_sim_drop=0.03,
            min_indian_prob_gain=0.02,
            calibration=str(calibration),
            calibration_target_set="genuine_asi",
            calibration_native_set="native_unseen",
            min_accent_gap_closed=0.25,
            interpretation="test",
        )
        kwargs.update(overrides)
        code = run_accent_gate(**kwargs)
        return code, json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))

    def test_pass(self, tmp_path, capsys):
        code, result = self.gate(
            tmp_path,
            {"mos_mean": 3.9, "wer_mean": 0.12, "sim_mean": 0.79,
             "indian_frac": 0.3, "indian_prob_mean": 0.38},  # gap closed 0.30
        )
        assert code == 0
        assert result["status"] == "pass"
        assert result["accent_calibration"]["candidate_gap_closed"] == pytest.approx(0.3)

    def test_insufficient_gap_fails(self, tmp_path, capsys):
        code, result = self.gate(
            tmp_path,
            {"mos_mean": 4.0, "wer_mean": 0.10, "sim_mean": 0.80,
             "indian_frac": 0.1, "indian_prob_mean": 0.26},  # gap closed 0.10
        )
        assert code == 1
        assert any("gap closed" in failure for failure in result["failures"])

    def test_quality_regression_fails(self, tmp_path, capsys):
        code, result = self.gate(
            tmp_path,
            {"mos_mean": 3.5, "wer_mean": 0.20, "sim_mean": 0.70,
             "indian_frac": 0.5, "indian_prob_mean": 0.50},
        )
        assert code == 1
        assert len(result["failures"]) == 3  # MOS, WER, similarity

    def test_missing_condition_fails_closed(self, tmp_path):
        summary = tmp_path / "condition_summary.csv"
        write_summary(
            summary,
            [{"set": "stock_xvc", "mos_mean": 4.0, "wer_mean": 0.1,
              "sim_mean": 0.8, "indian_frac": 0.1, "indian_prob_mean": 0.2}],
        )
        with pytest.raises(SystemExit, match="missing summary rows"):
            run_accent_gate(
                summary=str(summary),
                out=str(tmp_path / "gate.json"),
                stock_set="stock_xvc",
                candidate_set="persona_codec_teacher",
                max_mos_drop=0.25,
                max_wer_increase=0.05,
                max_sim_drop=0.03,
                min_indian_prob_gain=0.02,
                calibration=None,
                calibration_target_set="genuine_asi",
                calibration_native_set="native_unseen",
                min_accent_gap_closed=0.25,
                interpretation="test",
            )

    def test_inverted_calibration_fails_closed(self, tmp_path):
        summary = tmp_path / "condition_summary.csv"
        write_summary(
            summary,
            [
                {"set": "stock_xvc", "mos_mean": 4.0, "wer_mean": 0.1,
                 "sim_mean": 0.8, "indian_frac": 0.1, "indian_prob_mean": 0.5},
                {"set": "persona_codec_teacher", "mos_mean": 4.0, "wer_mean": 0.1,
                 "sim_mean": 0.8, "indian_frac": 0.1, "indian_prob_mean": 0.6},
            ],
        )
        calibration = tmp_path / "calibration.csv"
        write_summary(
            calibration,
            [
                {"set": "genuine_asi", "mos_mean": 4, "wer_mean": 0.1,
                 "sim_mean": 0.8, "indian_frac": 0.9, "indian_prob_mean": 0.20},
                {"set": "native_unseen", "mos_mean": 4, "wer_mean": 0.1,
                 "sim_mean": 0.8, "indian_frac": 0.0, "indian_prob_mean": 0.80},
            ],
        )
        with pytest.raises(SystemExit, match="must exceed native calibration"):
            run_accent_gate(
                summary=str(summary),
                out=str(tmp_path / "gate.json"),
                stock_set="stock_xvc",
                candidate_set="persona_codec_teacher",
                max_mos_drop=0.25,
                max_wer_increase=0.05,
                max_sim_drop=0.03,
                min_indian_prob_gain=0.02,
                calibration=str(calibration),
                calibration_target_set="genuine_asi",
                calibration_native_set="native_unseen",
                min_accent_gap_closed=0.25,
                interpretation="test",
            )
