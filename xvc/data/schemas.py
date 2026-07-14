"""Versioned schema helpers for native-to-target cross-pair datasets."""

from __future__ import annotations

from dataclasses import dataclass


CROSSPAIR_SCHEMA_VERSION = 1
BASE_MANIFEST_FIELDS = (
    "source_utt",
    "source_wav_path",
    "target_utt",
    "target_wav_path",
)


def is_supported_schema_version(version: int) -> bool:
    return version == CROSSPAIR_SCHEMA_VERSION


def missing_manifest_fields(row: dict) -> list[str]:
    return [field for field in BASE_MANIFEST_FIELDS if not row.get(field)]


def required_manifest_fields_for_meta(meta: dict) -> list[str]:
    required: list[str] = []
    if meta.get("warp_method") == "rubberband":
        required.extend(("raw_target_wav_path", "target_reference_wav_path"))
    if meta.get("warp_side") == "source":
        required.append("raw_source_wav_path")
    if meta.get("warp_method") == "latent":
        required.extend(
            ("raw_source_wav_path", "raw_target_wav_path", "latent_alignment_path")
        )
    return list(dict.fromkeys(required))


@dataclass(frozen=True)
class QCGates:
    allowed_global_stretch: tuple[float, float] | None = None
    max_anchor_removal_fraction: float | None = None

    @classmethod
    def from_align_meta(cls, meta: dict) -> "QCGates":
        allowed = meta.get("allowed_global_stretch")
        return cls(
            allowed_global_stretch=(
                (float(allowed[0]), float(allowed[1])) if allowed else None
            ),
            max_anchor_removal_fraction=(
                float(meta["max_anchor_removal_fraction"])
                if "max_anchor_removal_fraction" in meta
                else None
            ),
        )


def check_qc_row(source_utt: str, row: dict, gates: QCGates) -> list[str]:
    failures: list[str] = []
    if gates.allowed_global_stretch is not None:
        if "global_stretch_ratio" not in row:
            failures.append(
                f"{source_utt}: alignment QC lacks global_stretch_ratio; "
                "configured gate has no safe default"
            )
        else:
            ratio = float(row["global_stretch_ratio"])
            lo, hi = gates.allowed_global_stretch
            if not lo <= ratio <= hi:
                failures.append(
                    f"{source_utt}: global stretch {ratio:.4f} outside {[lo, hi]}"
                )
    if gates.max_anchor_removal_fraction is not None:
        if "anchor_removal_fraction" not in row:
            failures.append(
                f"{source_utt}: alignment QC lacks anchor_removal_fraction; "
                "configured gate has no safe default"
            )
        else:
            removal = float(row["anchor_removal_fraction"])
            limit = gates.max_anchor_removal_fraction
            if removal > limit:
                failures.append(
                    f"{source_utt}: anchor removal {removal:.1%} > {limit:.1%}"
                )
    return failures
