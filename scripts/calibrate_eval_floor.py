#!/usr/bin/env python
"""Score REAL recordings with the exact eval metric stack — no conversion, no
checkpoint under test. This calibrates what the proxies CAN return for genuinely
natural, genuinely accented human speech, so run tables are read against the
achievable ceiling instead of an imagined 3.7-MOS / 20-of-20-indian ideal.

UTMOS-style MOS predictors and Whisper WER can systematically score accented /
non-studio speech lower. Every fine-tune so
far "loses" a MOS point before the accent even registers — this script answers
whether that loss is model artifacts or the measuring stick.

What to score, typically:
  * the raw L2 target-corpus clips (TNI / ASI) — the distribution training pulls
    the model toward; their scores are the TARGET BAND for a perfect conversion;
  * optionally a cleaner accented corpus — separates accent penalty from
    recording-channel penalty;
  * optionally the native eval sources — the clean upper anchor.

Outputs per clip: mos_pred, accent_label/conf, wer (only when reference text is
available — sidecar ``<clip>.txt`` or ``--transcripts`` TSV; ASR-of-self would be
trivially ~0 and is deliberately not computed), lufs, duration, and — when
``--config``/``--ckpt`` are given — ERes2Net cosine against the pinned target
reference of the SAME speaker (matched by target stem appearing in the clip
path). That intra-speaker cosine is the realistic sim ceiling: different
utterances of the same real speaker never score 1.0.

Interpretation guide (written into summary.csv):
  * mos_mean of the L2 corpus ~2.5-3.0  -> fine-tunes near ceiling; MOS decay is
    largely convergence to the target distribution; fixes are data-side.
  * mos_mean ~3.5+                      -> proxy is fine with this accent/channel;
    low run scores are real artifacts.
  * indian_frac on real L2 clips        -> the accent classifier's actual recall;
    the canary's achievable ceiling (NOT 20/20).
  * sim_mean vs pinned refs             -> the sim ceiling for a perfect clone.
  * wer_mean (with true transcripts)    -> the ASR accent penalty baseline.

Example (container, conda xvc, repo root):
    python scripts/calibrate_eval_floor.py \
        --wavs "data/<accent>/wavs/TNI/*.wav" --label tni_corpus \
        --wavs "data/<accent>/wavs/ASI/*.wav" --label asi_corpus \
        --targets-dir data/eval_targets \
        --config configs/xvc.yaml --ckpt ckpts/xvc.pt \
        --out exp/eval_calibration

Part of the X-VC accent fine-tuning pipeline. Upstream: https://github.com/Jerrister/X-VC (MIT).
"""

from __future__ import annotations

import argparse
import csv
import glob as globlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # scripts/ -> eval_checkpoints
sys.path.insert(0, str(_HERE.parent))   # repo root -> bins.infer_utils

from eval_checkpoints import (  # noqa: E402  (same metric stack as the run eval)
    AccentClassifier, MOSPredictor, Whisper, maybe_resample, norm_text,
    word_error_rate,
)


def _load_mono(path: Path):
    import numpy as np
    import soundfile as sf
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(wav.mean(axis=1)), int(sr)


def _lufs(wav, sr: int) -> Optional[float]:
    try:
        import numpy as np
        import pyloudnorm as pyln
        return round(float(pyln.Meter(sr).integrated_loudness(
            np.asarray(wav, dtype="float64"))), 2)
    except Exception:
        return None


def _read_transcripts(tsv: Optional[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if tsv:
        for line in Path(tsv).read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                name, text = line.split("\t", 1)
                out[name.strip()] = text.strip()
    return out


def _ref_text(clip: Path, transcripts: Dict[str, str]) -> Optional[str]:
    sidecar = clip.with_suffix(".txt")
    if sidecar.is_file():
        return sidecar.read_text(encoding="utf-8").strip()
    return transcripts.get(clip.stem)


def _match_target(clip: Path, target_stems: List[str]) -> Optional[str]:
    """Pinned-target speaker whose stem appears in the clip's path (unique hit)."""
    p = str(clip).lower()
    hits = [t for t in target_stems if t.lower() in p]
    return hits[0] if len(hits) == 1 else None


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _std(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return round((sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5, 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wavs", action="append", required=True,
                    help="glob of real clips to score; repeatable, paired with --label")
    ap.add_argument("--label", action="append", required=True,
                    help="set name for the matching --wavs (repeat once per --wavs)")
    ap.add_argument("--out", default="exp/eval_calibration")
    ap.add_argument("--targets-dir", default=None,
                    help="pinned target refs (<speaker>.wav); enables the sim ceiling "
                         "when --config/--ckpt are also given")
    ap.add_argument("--config", default=None, help="XVC config (e.g. configs/xvc.yaml)")
    ap.add_argument("--ckpt", default=None, help="stock checkpoint (e.g. ckpts/xvc.pt)")
    ap.add_argument("--transcripts", default=None, help="TSV name<TAB>text for WER reference")
    ap.add_argument("--wer-model", default="small", help="whisper model size (default small)")
    ap.add_argument("--skip-wer", action="store_true")
    ap.add_argument("--skip-accent-clf", action="store_true")
    ap.add_argument("--skip-mos", action="store_true")
    ap.add_argument("--score-sr", type=int, default=16000,
                    help="resample clips to this rate before MOS, matching the rate "
                         "the run eval scores model output at (default 16000)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--latent-hop-length", type=int, default=1280)
    ap.add_argument("--mask-target-condition", action="store_true")
    ap.add_argument("--max-clips", type=int, default=None, help="cap per set")
    args = ap.parse_args(argv)

    if len(args.wavs) != len(args.label):
        raise SystemExit("[error] need exactly one --label per --wavs")

    sets: List[tuple] = []
    for pattern, label in zip(args.wavs, args.label):
        clips = sorted(Path(p) for p in globlib.glob(pattern, recursive=True))
        clips = [c for c in clips if c.suffix.lower() == ".wav"]
        if not clips:
            raise SystemExit(f"[error] --wavs {pattern!r} matched no .wav files")
        if args.max_clips:
            clips = clips[: args.max_clips]
        sets.append((label, pattern, clips))

    import numpy as np
    import torch  # container-only, as everywhere else in the pipeline

    cuda = "cuda" if torch.cuda.is_available() else "cpu"
    mos_model = None if args.skip_mos else MOSPredictor(cuda)
    accent_clf = None if args.skip_accent_clf else AccentClassifier(cuda)
    whisper = None if args.skip_wer else Whisper(args.wer_model, cuda)
    transcripts = _read_transcripts(args.transcripts)

    # Speaker-sim ceiling needs the SAME encoder + reference-embedding path the
    # run eval uses: refs through precompute_conditions, clips through
    # model.speaker_encoder on raw audio at the model sample rate.
    model = cfg = device = None
    tgt_embs: Dict[str, "torch.Tensor"] = {}
    target_stems: List[str] = []
    if args.config and args.ckpt and args.targets_dir:
        from bins.infer_utils import load_pair_as_tensors, load_xvc, precompute_conditions
        cfg, model, device = load_xvc(args.config, args.ckpt, args.device, False)
        for t in sorted(Path(args.targets_dir).glob("*.wav")):
            _, tw, twc = load_pair_as_tensors(
                str(t), str(t), cfg, device, args.latent_hop_length,
                args.mask_target_condition)
            emb, _ = precompute_conditions(model, tw, twc)
            tgt_embs[t.stem] = emb
        target_stems = list(tgt_embs)
        print(f"[sim] {len(target_stems)} pinned reference(s): {target_stems}")
    elif args.targets_dir or args.config or args.ckpt:
        print("[sim] skipped -- needs --targets-dir AND --config AND --ckpt",
              file=sys.stderr)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    fieldnames = ["set", "clip", "speaker", "sim_cosine", "wer", "wer_mode",
                  "mos_pred", "dur_s", "sr", "accent_label", "accent_conf",
                  "indian_prob", "lufs"]

    for label, pattern, clips in sets:
        print(f"\n=== set {label} ({len(clips)} clips, {pattern}) ===")
        for clip in clips:
            wav, sr = _load_mono(clip)
            scored, ssr = maybe_resample(wav, sr, args.score_sr)
            mos = round(mos_model.score(scored, ssr), 3) if mos_model else None

            alabel = aconf = indian_prob = None
            if accent_clf:
                alabel, aconf, posterior = accent_clf.classify_detailed(str(clip))
                indian_prob = posterior["indian"]
                aconf = round(aconf, 4)

            wer = wmode = None
            if whisper:
                ref = _ref_text(clip, transcripts)
                if ref:  # no ASR-of-self proxy: it is ~0 by construction
                    wer = round(word_error_rate(
                        norm_text(ref), norm_text(whisper.transcribe(str(clip)))), 4)
                    wmode = "ref_text"

            speaker = _match_target(clip, target_stems) if target_stems else None
            sim = None
            if speaker and model is not None:
                import torch.nn.functional as F
                enc, esr = maybe_resample(wav, sr, int(cfg["sample_rate"]))
                with torch.inference_mode():
                    x = torch.from_numpy(np.ascontiguousarray(enc)).float().to(device)
                    emb, _ = model.speaker_encoder(x.view(1, 1, -1))
                    sim = round(float(F.cosine_similarity(
                        emb.flatten(1), tgt_embs[speaker].flatten(1)).item()), 4)

            rows.append({
                "set": label, "clip": clip.stem, "speaker": speaker,
                "sim_cosine": sim, "wer": wer, "wer_mode": wmode, "mos_pred": mos,
                "dur_s": round(len(wav) / sr, 3), "sr": sr,
                "accent_label": alabel, "accent_conf": aconf,
                "indian_prob": round(indian_prob, 6) if indian_prob is not None else None,
                "lufs": _lufs(wav, sr),
            })
            print(f"  {clip.stem:<28} mos={mos} accent={alabel} wer={wer} sim={sim}")

    with open(out_root / "per_clip.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Per-set summary: the numbers run tables should be read against.
    summaries = []
    for label, _, _ in sets:
        sub = [r for r in rows if r["set"] == label]
        hist: Dict[str, int] = {}
        for r in sub:
            if r["accent_label"]:
                hist[r["accent_label"]] = hist.get(r["accent_label"], 0) + 1
        top = " ".join(f"{k}:{v}" for k, v in
                       sorted(hist.items(), key=lambda kv: -kv[1])[:5])
        summaries.append({
            "set": label, "n": len(sub),
            "mos_mean": _mean([r["mos_pred"] for r in sub]),
            "mos_std": _std([r["mos_pred"] for r in sub]),
            "wer_mean": _mean([r["wer"] for r in sub]),
            "n_wer": sum(1 for r in sub if r["wer"] is not None),
            "sim_mean": _mean([r["sim_cosine"] for r in sub]),
            "sim_std": _std([r["sim_cosine"] for r in sub]),
            "n_sim": sum(1 for r in sub if r["sim_cosine"] is not None),
            "indian_frac": _mean([1.0 if r["accent_label"] == "indian" else 0.0
                                  for r in sub if r["accent_label"]]),
            "indian_prob_mean": _mean([r["indian_prob"] for r in sub]),
            "accent_hist": top,
        })
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"written_utc": datetime.now(timezone.utc).isoformat(),
                   "args": vars(args),
                   "sets": {label: len(clips) for label, _, clips in sets}},
                  f, indent=2, default=str)

    print(f"\n[calibration] wrote {out_root}/per_clip.csv, summary.csv")
    for s in summaries:
        print(f"  {s['set']:<16} n={s['n']:<4} mos={s['mos_mean']} "
              f"(std {s['mos_std']}) wer={s['wer_mean']} (n={s['n_wer']}) "
              f"sim={s['sim_mean']} (n={s['n_sim']}) "
              f"indian_frac={s['indian_frac']} p={s['indian_prob_mean']} "
              f"[{s['accent_hist']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
