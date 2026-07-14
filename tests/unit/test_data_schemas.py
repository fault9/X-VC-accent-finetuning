"""Unit tests for xvc.data.schemas (no filesystem, no audio)."""

import pytest

from xvc.data.schemas import (
    CROSSPAIR_SCHEMA_VERSION,
    QCGates,
    check_qc_row,
    is_supported_schema_version,
    missing_manifest_fields,
    required_manifest_fields_for_meta,
)


def test_schema_version_supported():
    assert is_supported_schema_version(CROSSPAIR_SCHEMA_VERSION)
    assert not is_supported_schema_version(0)


def test_missing_manifest_fields():
    row = {
        "source_utt": "clb_arctic_a0001",
        "source_wav_path": "/x/s.wav",
        "target_utt": "ASI_arctic_a0001_ch",
        "target_wav_path": "/x/t.wav",
    }
    assert missing_manifest_fields(row) == []
    assert missing_manifest_fields({**row, "source_utt": ""}) == ["source_utt"]
    assert set(missing_manifest_fields({})) == {
        "source_utt", "source_wav_path", "target_utt", "target_wav_path",
    }


@pytest.mark.parametrize(
    "meta, expected_fields",
    [
        ({}, set()),
        ({"warp_method": "rubberband"},
         {"raw_target_wav_path", "target_reference_wav_path"}),
        ({"warp_method": "latent"},
         {"raw_source_wav_path", "raw_target_wav_path", "latent_alignment_path"}),
        ({"warp_side": "source"}, {"raw_source_wav_path"}),
    ],
)
def test_required_fields_follow_align_meta(meta, expected_fields):
    assert set(required_manifest_fields_for_meta(meta)) == expected_fields


def test_inactive_gates_ignore_missing_fields():
    gates = QCGates.from_align_meta({})
    assert check_qc_row("utt", {}, gates) == []


def test_active_gate_requires_field_with_actionable_reason():
    gates = QCGates.from_align_meta({"max_anchor_removal_fraction": 0.4})
    failures = check_qc_row("utt", {}, gates)
    assert len(failures) == 1
    assert "anchor_removal_fraction" in failures[0]
    assert "no safe default" in failures[0]


def test_active_gates_enforce_thresholds():
    gates = QCGates.from_align_meta(
        {"allowed_global_stretch": [0.8, 1.25], "max_anchor_removal_fraction": 0.4}
    )
    ok = {"global_stretch_ratio": 1.0, "anchor_removal_fraction": 0.2}
    assert check_qc_row("utt", ok, gates) == []
    bad = {"global_stretch_ratio": 2.0, "anchor_removal_fraction": 0.5}
    failures = check_qc_row("utt", bad, gates)
    assert len(failures) == 2
    assert any("global stretch" in f for f in failures)
    assert any("anchor removal" in f for f in failures)
