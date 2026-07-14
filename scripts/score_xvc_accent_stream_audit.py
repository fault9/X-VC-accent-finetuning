#!/usr/bin/env python
"""Score every condition from ``audit_xvc_accent_streams.py`` in one pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default="ckpts/xvc.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--xvc-device", type=int, default=0)
    parser.add_argument("--wer-model", default="small")
    args = parser.parse_args(argv)

    import soundfile as sf

    from eval_checkpoints import (
        AccentClassifier,
        MOSPredictor,
        Whisper,
        norm_text,
        word_error_rate,
    )
    from gate_teacher_renders import (
        ACCENT_DEPTH,
        _load_similarity_scorer,
        _read_manifests,
        _similarity,
        _source_reference,
        _summarize,
        _write_csv,
    )

    audit_root = Path(args.audit_root)
    meta_path = audit_root / "audit_meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"[error] missing audit metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    conditions = list(meta["conditions"])
    manifest_condition = meta.get("manifest_condition", conditions[0])
    if manifest_condition not in conditions:
        raise SystemExit(
            f"[error] manifest_condition {manifest_condition!r} is not in conditions"
        )
    manifest_root = audit_root / manifest_condition
    _, by_target = _read_manifests(manifest_root)

    # Match the established evaluator's memory-safe model load order.
    sim_scorer = _load_similarity_scorer(args)
    whisper = Whisper(args.wer_model, args.device)
    classifier = AccentClassifier(args.device)
    mos = MOSPredictor(args.device)
    source_cache = {}
    rows = []
    summaries = []

    for condition in conditions:
        wavs = sorted((audit_root / condition / "wavs").glob("*.wav"))
        if len(wavs) != len(by_target):
            raise SystemExit(
                f"[error] {condition}: {len(wavs)} wavs but {len(by_target)} manifest rows"
            )
        condition_rows = []
        for wav_path in wavs:
            manifest_row = by_target.get(wav_path.name)
            if manifest_row is None:
                raise SystemExit(f"[error] no manifest row for {wav_path.name}")
            reference_words, wer_mode = _source_reference(
                manifest_row, whisper, source_cache
            )
            hypothesis_words = norm_text(whisper.transcribe(str(wav_path)))
            accent_label, accent_conf = classifier.classify(str(wav_path))
            wav, sample_rate = sf.read(str(wav_path), always_2d=False)
            row = {
                "set": condition,
                "clip": wav_path.name,
                "source": manifest_row["source_wav_path"],
                "split": manifest_row["_split"],
                "mos_pred": float(mos.score(wav, sample_rate)),
                "wer": word_error_rate(reference_words, hypothesis_words),
                "wer_mode": wer_mode,
                "sim_cosine": _similarity(wav_path, sim_scorer),
                "accent_label": accent_label,
                "accent_conf": float(accent_conf),
                "accent_depth": ACCENT_DEPTH.get(accent_label, 0.0),
            }
            condition_rows.append(row)
            rows.append(row)
        summary = _summarize(condition, condition_rows)
        summaries.append(summary)
        print(
            f"{condition:30s} n={summary['n']:>2} MOS={summary['mos_mean']} "
            f"WER={summary['wer_mean']} sim={summary['sim_mean']} "
            f"Indian={summary['indian_frac']} ({summary['accent_hist']})"
        )

    out = Path(args.out)
    fields = [
        "set", "clip", "source", "split", "mos_pred", "wer", "wer_mode",
        "sim_cosine", "accent_label", "accent_conf", "accent_depth",
    ]
    _write_csv(out / "per_clip.csv", rows, fields)
    _write_csv(out / "condition_summary.csv", summaries, list(summaries[0]))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "audit_root": str(audit_root),
                "reference": args.reference,
                "config": args.config,
                "checkpoint": args.ckpt,
                "wer_note": "sidecar reference text where available; otherwise ASR-on-source proxy",
                "accent_note": "classifier labels are a noisy screen and require listening",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[metrics] {out / 'condition_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
