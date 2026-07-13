#!/usr/bin/env python3
"""
Per-checkpoint conversion eval for X-VC accent fine-tunes — THE checkpoint selector.

Deployment converts UNSEEN sources (the experimenter live, participants) into the
fine-tuned targets, so checkpoints must be chosen from that direction — not from
self-reconstruction validation loss. For every checkpoint of a run this script:

    1. converts clips from a FIXED source folder according to an optional,
       explicit evaluation plan (offline mode; optionally streaming too),
    2. computes per clip: ERes2Net cosine similarity to the target reference,
       Whisper WER against the script text (or an ASR-on-source proxy),
       output-vs-source duration delta, optional CommonAccent confidence,
    3. dumps audible samples per checkpoint (optionally EBU-R128 loudness
       normalized and/or resampled, e.g. 24 kHz for PP delivery),
    4. writes metrics.csv (per clip) + summary.csv (per checkpoint) and prints a
       ranking. The frozen checkpoint is picked from THESE curves.

Target references are PINNED: each target is one named wav file whose sha256 is
recorded in targets_used.json — the reference clip is part of the stimulus
definition (the speaker embedding is computed from exactly this clip).

Typical use (GPU container):

    # one-time: pin one reference clip per speaker from the val split
    python scripts/eval_checkpoints.py make-targets \
        --data-root data/finetuning_audio --out data/eval_targets

    # evaluate every checkpoint of a run against the fixed sources
    python scripts/eval_checkpoints.py run \
        --run-dir exp/finetune_arabic \
        --source-dir data/eval_sources \
        --targets-dir data/eval_targets \
        --evaluation-plan configs/eval_hindi_native_to_accent.json \
        --include-base ckpts/xvc.pt

WER references: put a sidecar ``<clip>.txt`` next to each source wav, or pass
``--transcripts file.tsv`` (``name<TAB>text``). Without either, the reference is
Whisper's transcript of the SOURCE clip (intelligibility-drift proxy; flagged in
the ``wer_mode`` column).

Requires torch + the repo deps; Whisper backend is faster-whisper or
openai-whisper (``--skip-wer`` to omit); ``--accent-clf`` needs speechbrain;
``--loudnorm`` needs pyloudnorm.

Part of the X-VC accent fine-tuning pipeline. Upstream: https://github.com/Jerrister/X-VC (MIT).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# make-targets: pin one reference clip per speaker (deterministic)
# --------------------------------------------------------------------------- #
def _trim_silence(x, sr: int, thresh_db: float = -40.0, pad_ms: int = 50):
    """Trim leading/trailing low-energy samples (frame RMS gate), keep small pads."""
    import numpy as np

    frame = sr // 100  # 10ms
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt((x[: n * frame].reshape(n, frame).astype(np.float64) ** 2).mean(axis=1))
    ref = rms.max() or 1.0
    db = 20 * np.log10(np.maximum(rms / ref, 1e-10))
    voiced = np.nonzero(db > thresh_db)[0]
    if len(voiced) == 0:
        return x
    pad = pad_ms * sr // 1000
    start = max(0, voiced[0] * frame - pad)
    end = min(len(x), (voiced[-1] + 1) * frame + pad)
    return x[start:end]


def _concat_wavs(paths: List[Path], dst: Path, seconds: float) -> List[str]:
    """Concatenate 16k mono PCM16 wavs (silence-trimmed) into dst until >= seconds
    of actual speech. Returns names used."""
    import wave

    import numpy as np

    used = []
    pieces = []
    params = None
    total = 0.0
    for p in paths:
        with wave.open(str(p), "rb") as w:
            if params is None:
                params = w.getparams()
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        x = _trim_silence(x, sr)
        pieces.append(x)
        total += len(x) / float(sr)
        used.append(p.name)
        if total >= seconds:
            break
    # Cross-fade joins (20ms) — hard splices click, and the reference feeds the
    # frame-condition path, so splice artifacts can leak into rendering.
    sr = params.framerate
    xf = max(1, 20 * sr // 1000)
    out = pieces[0].astype(np.float32)
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    for piece in pieces[1:]:
        nxt = piece.astype(np.float32)
        n = min(xf, len(out), len(nxt))
        out[-n:] = out[-n:] * (1 - ramp[-n:]) + nxt[:n] * ramp[-n:]
        out = np.concatenate([out, nxt[n:]])
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(np.clip(out, -32768, 32767).astype(np.int16).tobytes())
    return used


def cmd_make_targets(args) -> int:
    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = {}
    for group_dir in sorted((data_root / "val").iterdir()):
        if not group_dir.is_dir():
            continue
        for spk_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            wavs = sorted(spk_dir.glob("*.wav"))
            if not wavs:
                print(f"[warn] no val wavs for {spk_dir}", file=sys.stderr)
                continue
            # Deterministic: alphabetical order. The val split never trains,
            # so the reference is unseen by the loss.
            dst = out_dir / f"{spk_dir.name}.wav"
            if args.seconds and args.seconds > 0:
                used = _concat_wavs(wavs, dst, args.seconds)
                src_desc = f"{len(used)} clips ({used[0]}..{used[-1]})"
            else:
                shutil.copy2(wavs[0], dst)
                used = [wavs[0].name]
                src_desc = wavs[0].name
            picked[spk_dir.name] = {
                "group": group_dir.name,
                "source_dir": str(spk_dir),
                "clips": used,
                "sha256": sha256_file(dst),
            }
            print(f"{group_dir.name:8s} {spk_dir.name:6s} <- {src_desc}")
    # Remove wavs from previous rosters — the eval harness scans this directory,
    # so a stale target would silently re-enter the study.
    for wav in sorted(out_dir.glob("*.wav")):
        if wav.stem not in picked:
            wav.unlink()
            print(f"[clean] removed stale target {wav.name} (not in current roster)")
    with open(out_dir / "targets_meta.json", "w", encoding="utf-8") as f:
        json.dump(picked, f, indent=2)
    print(f"\n{len(picked)} pinned target references in {out_dir} "
          f"(sha256s in targets_meta.json)")
    return 0


# --------------------------------------------------------------------------- #
# Metrics helpers (lazy heavy imports)
# --------------------------------------------------------------------------- #
_NORM_RE = re.compile(r"[^a-z0-9' ]+")


def norm_text(s: str) -> List[str]:
    return _NORM_RE.sub(" ", s.lower()).split()


def word_error_rate(ref: List[str], hyp: List[str]) -> float:
    """Plain Levenshtein WER; no external deps."""
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / len(ref)


class Whisper:
    """Thin ASR wrapper: faster-whisper preferred, openai-whisper fallback."""

    def __init__(self, model_name: str, device: str):
        self.backend = None
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_name, device=device,
                                      compute_type="float16" if device == "cuda" else "int8")
            self.backend = "faster-whisper"
        except ImportError:
            try:
                import whisper
                self.model = whisper.load_model(model_name, device=device)
                self.backend = "openai-whisper"
            except ImportError:
                raise RuntimeError(
                    "No Whisper backend. pip install faster-whisper (or openai-whisper), "
                    "or pass --skip-wer.")

    def transcribe(self, wav_path: str) -> str:
        if self.backend == "faster-whisper":
            segments, _ = self.model.transcribe(wav_path, language="en")
            return " ".join(s.text for s in segments)
        return self.model.transcribe(wav_path, language="en")["text"]


class AccentClassifier:
    """Optional CommonAccent ECAPA classifier (speechbrain)."""

    def __init__(self, device: str):
        from speechbrain.inference.classifiers import EncoderClassifier
        self.clf = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            savedir="pretrained/accent-id-commonaccent_ecapa",
            run_opts={"device": device},
        )

    def classify(self, wav_path: str):
        out_prob, score, index, label = self.clf.classify_file(wav_path)
        return label[0], float(score.exp().item() if hasattr(score, "exp") else score)


class MOSPredictor:
    """Predicted MOS (UTMOS22-strong via torch.hub, tarepan/SpeechMOS).

    Captures the synthesis-quality axis that cosine/WER are blind to
    (robotic texture, gating, artifacts). Scores the model's raw 16k output,
    before any loudnorm/resample post-processing.
    """

    def __init__(self, device: str):
        import torch
        self.device = device
        self.model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        ).to(device).eval()

    def score(self, wav_np, sr: int) -> float:
        import torch
        with torch.inference_mode():
            x = torch.from_numpy(wav_np).float().unsqueeze(0).to(self.device)
            return float(self.model(x, sr).item())


def maybe_loudnorm(wav, sr: int, target_lufs: float):
    """EBU R128 loudness-normalize a float mono array. Returns (wav, measured_lufs)."""
    import numpy as np
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(np.asarray(wav, dtype="float64"))
    out = pyln.normalize.loudness(wav, loud, target_lufs)
    peak = max(1e-9, float(abs(out).max()))
    if peak > 0.99:  # prevent clipping introduced by the gain
        out = out * (0.99 / peak)
    return out, loud


def maybe_resample(wav, sr: int, out_sr: int):
    if out_sr == sr:
        return wav, sr
    try:
        import soxr
        return soxr.resample(wav, sr, out_sr), out_sr
    except ImportError:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, out_sr)
        return resample_poly(wav, out_sr // g, sr // g), out_sr


# --------------------------------------------------------------------------- #
# run: the eval loop
# --------------------------------------------------------------------------- #
def list_checkpoints(run_dir: Path, steps: Optional[str]) -> List[Dict]:
    ckpt_dir = run_dir / "ckpt"
    found = []
    for p in sorted(ckpt_dir.glob("*.pt")):
        if p.is_symlink() or not p.stem.isdigit():
            continue
        found.append({"step": int(p.stem), "path": p})
    if steps and steps != "all":
        wanted = {int(s) for s in steps.split(",")}
        found = [c for c in found if c["step"] in wanted]
    return found


def load_transcripts(source_dir: Path, transcripts: Optional[str]) -> Dict[str, str]:
    table: Dict[str, str] = {}
    if transcripts:
        with open(transcripts, "r", encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    name, text = line.rstrip("\n").split("\t", 1)
                    table[Path(name).stem] = text
    for wav in source_dir.glob("*.wav"):
        side = wav.with_suffix(".txt")
        if wav.stem not in table and side.is_file():
            table[wav.stem] = side.read_text(encoding="utf-8").strip()
    return table


def source_speaker(path: Path) -> str:
    """Return the corpus speaker prefix from ``speaker_arctic_prompt.wav``."""
    marker = "_arctic_"
    if marker not in path.stem:
        raise ValueError(
            f"cannot derive source speaker from {path.name!r}; an evaluation "
            f"plan requires names like speaker_arctic_prompt.wav"
        )
    return path.stem.split(marker, 1)[0]


def build_evaluation_pairs(sources, targets, plan_path: Optional[str], targets_dir: Path):
    """Return explicitly assigned ``(source, target)`` pairs.

    Without a plan this retains the historical Cartesian-product behaviour.
    A plan is deliberately strict: every source speaker must be allow-listed,
    every source must resolve to exactly one assigned target, and target-group
    membership is verified against ``targets_meta.json``.
    """
    if not plan_path:
        return [(source, target) for source in sources for target in targets], None

    plan_file = Path(plan_path)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if not plan.get("source_group"):
        # Historical guard: plans originally had to assert the native-carrier
        # stimulus ('native_english'). Diverse/unseen-speaker sets are now
        # legitimate stimuli, so any non-empty class label is accepted -- it
        # documents what the rows mean and keeps unlabeled plans from running.
        raise SystemExit(
            "[error] evaluation plan must declare a source_group label "
            "(e.g. 'native_english', 'diverse_unseen')"
        )

    allowed = set(plan.get("allowed_source_speakers") or [])
    assignments = plan.get("assignments") or {}
    if not allowed or not assignments:
        raise SystemExit(
            "[error] evaluation plan needs allowed_source_speakers and assignments"
        )

    target_by_name = {target.stem: target for target in targets}
    target_group = plan.get("target_group")
    if not target_group:
        raise SystemExit("[error] evaluation plan needs target_group")
    meta_path = targets_dir / "targets_meta.json"
    if not meta_path.is_file():
        raise SystemExit(
            f"[error] target-group validation requires {meta_path}"
        )
    target_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    pairs = []
    used_targets = set()
    for source in sources:
        try:
            speaker = source_speaker(source)
        except ValueError as exc:
            raise SystemExit(f"[error] {exc}") from exc
        if speaker not in allowed:
            raise SystemExit(
                f"[error] non-native/unapproved source speaker {speaker!r}: "
                f"{source.name}"
            )
        target_name = assignments.get(source.stem, assignments.get(speaker))
        if not target_name:
            raise SystemExit(
                f"[error] no assigned target for source {source.stem!r} "
                f"(speaker {speaker!r})"
            )
        if target_name not in target_by_name:
            raise SystemExit(
                f"[error] assigned target {target_name!r} is absent from "
                f"{targets_dir}"
            )
        actual_group = (target_meta.get(target_name) or {}).get("group")
        if actual_group != target_group:
            raise SystemExit(
                f"[error] assigned target {target_name!r} belongs to "
                f"{actual_group!r}, expected {target_group!r}"
            )
        pairs.append((source, target_by_name[target_name]))
        used_targets.add(target_name)

    declared_targets = set(assignments.values())
    if used_targets != declared_targets:
        raise SystemExit(
            "[error] evaluation plan has unused target assignments: "
            + ", ".join(sorted(declared_targets - used_targets))
        )
    return pairs, plan


def cmd_run(args) -> int:
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F

    from bins.infer_utils import (
        load_pair_as_tensors, load_xvc, precompute_conditions,
        run_offline, run_streaming, to_numpy_audio,
    )

    run_dir = Path(args.run_dir)
    config_path = Path(args.config) if args.config else run_dir / "config.yaml"
    out_root = Path(args.out) if args.out else run_dir / "eval"
    out_root.mkdir(parents=True, exist_ok=True)

    sources = sorted(Path(args.source_dir).glob("*.wav"))
    targets = sorted(p for p in Path(args.targets_dir).glob("*.wav"))
    if not sources:
        raise SystemExit(f"[error] no source wavs in {args.source_dir}")
    if not targets:
        raise SystemExit(f"[error] no target reference wavs in {args.targets_dir}")
    if args.max_sources:
        sources = sources[: args.max_sources]

    eval_pairs, evaluation_plan = build_evaluation_pairs(
        sources, targets, args.evaluation_plan, Path(args.targets_dir)
    )
    targets = sorted({target for _, target in eval_pairs})

    ckpts = list_checkpoints(run_dir, args.steps)
    if args.include_base:
        ckpts.insert(0, {"step": "base", "path": Path(args.include_base)})
    if not ckpts:
        raise SystemExit(f"[error] no checkpoints found under {run_dir}/ckpt")

    # Pin the stimulus definition: exactly these reference files, these hashes.
    with open(out_root / "targets_used.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "written_utc": datetime.now(timezone.utc).isoformat(),
                "config": str(config_path),
                "source_dir": str(args.source_dir),
                "sources": {p.stem: sha256_file(p) for p in sources},
                "targets": {p.stem: sha256_file(p) for p in targets},
                "evaluation_plan": evaluation_plan,
                "pairs": [
                    {"source": source.stem, "target": target.stem}
                    for source, target in eval_pairs
                ],
                "mode": "streaming" if args.streaming else "offline",
            },
            f, indent=2,
        )

    transcripts = load_transcripts(Path(args.source_dir), args.transcripts)
    whisper = None
    if not args.skip_wer:
        whisper = Whisper(args.wer_model, "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[wer] backend: {whisper.backend} ({args.wer_model})")
        missing_txt = [s.stem for s in sources if s.stem not in transcripts]
        if missing_txt:
            print(f"[wer] no reference text for {len(missing_txt)} source(s) -> "
                  f"ASR-on-source proxy for those", file=sys.stderr)
        # Reference cache: transcript text, else ASR of the source clip.
        wer_ref: Dict[str, tuple] = {}
        for s in sources:
            if s.stem in transcripts:
                wer_ref[s.stem] = (norm_text(transcripts[s.stem]), "ref_text")
            else:
                wer_ref[s.stem] = (norm_text(whisper.transcribe(str(s))), "asr_source_proxy")

    accent_clf = AccentClassifier("cuda" if torch.cuda.is_available() else "cpu") \
        if args.accent_clf else None
    mos_model = MOSPredictor("cuda" if torch.cuda.is_available() else "cpu") \
        if args.mos else None

    rows = []
    fieldnames = ["step", "mode", "source", "target", "sim_cosine", "wer", "wer_mode",
                  "mos_pred", "dur_source_s", "dur_out_s", "dur_delta_pct",
                  "accent_label", "accent_conf", "lufs_in", "avg_latency_ms", "out_path"]

    for ck in ckpts:
        step = ck["step"]
        print(f"\n=== checkpoint {step} ({ck['path']}) ===")
        cfg, model, device = load_xvc(str(config_path), str(ck["path"]), args.device, False)
        sr = int(cfg["sample_rate"])
        sample_dir = out_root / "samples" / str(step)
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Target speaker embeddings from EXACTLY the pinned reference clips.
        tgt_cache = {}
        for t in targets:
            _, target_wav, target_wav_cond = load_pair_as_tensors(
                str(t), str(t), cfg, device, args.latent_hop_length,
                args.mask_target_condition)
            emb, frame_cond = precompute_conditions(model, target_wav, target_wav_cond)
            tgt_cache[t.stem] = (target_wav, target_wav_cond, emb, frame_cond)

        for src, t in eval_pairs:
                target_wav, target_wav_cond, tgt_emb, frame_cond = tgt_cache[t.stem]
                source_wav, _, _ = load_pair_as_tensors(
                    str(src), str(t), cfg, device, args.latent_hop_length,
                    args.mask_target_condition)

                t0 = time.time()
                latency = None
                if args.streaming:
                    recon, latency_list = run_streaming(
                        model=model, source_wav=source_wav,
                        speaker_condition=tgt_emb, frame_condition=frame_cond,
                        sample_rate=sr, chunk_ms=args.chunk,
                        current_ms=args.current, future_ms=args.future,
                        smooth_ms=args.smooth)
                    latency = sum(latency_list) / max(1, len(latency_list))
                    mode = "stream"
                else:
                    recon = run_offline(model, source_wav, target_wav, target_wav_cond)
                    mode = "offline"

                out_np = to_numpy_audio(recon)
                dur_src = source_wav.shape[-1] / sr
                dur_out = len(out_np) / sr

                mos = mos_model.score(out_np, sr) if mos_model is not None else None

                # ERes2Net cosine: converted output vs the pinned target reference.
                with torch.inference_mode():
                    out_t = torch.from_numpy(out_np).float().to(device)
                    out_emb, _ = model.speaker_encoder(out_t.view(1, 1, -1))
                    sim = float(F.cosine_similarity(
                        out_emb.flatten(1), tgt_emb.flatten(1)).item())

                lufs_in = None
                if args.loudnorm:
                    out_np, lufs_in = maybe_loudnorm(out_np, sr, args.loudnorm_lufs)
                save_np, save_sr = maybe_resample(out_np, sr, args.out_sr) \
                    if args.out_sr else (out_np, sr)
                out_path = sample_dir / f"{t.stem}__{src.stem}.wav"
                sf.write(str(out_path), np.asarray(save_np, dtype="float32"), save_sr)

                wer = wmode = None
                if whisper is not None:
                    hyp = norm_text(whisper.transcribe(str(out_path)))
                    ref, wmode = wer_ref[src.stem]
                    wer = round(word_error_rate(ref, hyp), 4)

                alabel = aconf = None
                if accent_clf is not None:
                    alabel, aconf = accent_clf.classify(str(out_path))

                rows.append({
                    "step": step, "mode": mode, "source": src.stem, "target": t.stem,
                    "sim_cosine": round(sim, 4), "wer": wer, "wer_mode": wmode,
                    "mos_pred": round(mos, 3) if mos is not None else None,
                    "dur_source_s": round(dur_src, 3), "dur_out_s": round(dur_out, 3),
                    "dur_delta_pct": round(100.0 * (dur_out - dur_src) / max(dur_src, 1e-6), 2),
                    "accent_label": alabel,
                    "accent_conf": round(aconf, 4) if aconf is not None else None,
                    "lufs_in": round(lufs_in, 2) if lufs_in is not None else None,
                    "avg_latency_ms": round(latency, 1) if latency is not None else None,
                    "out_path": str(out_path),
                })
                print(
                    f"  {src.stem} -> {t.stem} done "
                    f"({time.time() - t0:.1f}s)"
                )

        del model, tgt_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Persist incrementally — a crash at checkpoint N keeps 1..N-1.
        with open(out_root / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    # Per-checkpoint summary + ranking.
    def agg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    steps_seen = sorted({r["step"] for r in rows}, key=lambda s: (s != "base", s))
    summary = []
    for s in steps_seen:
        sub = [r for r in rows if r["step"] == s]
        summary.append({
            "step": s,
            "n": len(sub),
            "sim_cosine_mean": agg([r["sim_cosine"] for r in sub]),
            "wer_mean": agg([r["wer"] for r in sub]),
            "mos_pred_mean": agg([r["mos_pred"] for r in sub]),
            "abs_dur_delta_pct_mean": agg([abs(r["dur_delta_pct"]) for r in sub]),
            "avg_latency_ms": agg([r["avg_latency_ms"] for r in sub]),
        })
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\n===== summary (pick the frozen checkpoint from THIS, not val loss) =====")
    for s in summary:
        print(f"  step {str(s['step']):>6s} | sim {s['sim_cosine_mean']} | "
              f"wer {s['wer_mean']} | |dur%| {s['abs_dur_delta_pct_mean']}"
              + (f" | mos {s['mos_pred_mean']}" if s["mos_pred_mean"] else "")
              + (f" | lat {s['avg_latency_ms']}ms" if s["avg_latency_ms"] else ""))
    print(f"\nmetrics.csv / summary.csv / samples/ under {out_root}")
    print("Listen to samples/<step>/ before freezing anything — ears gate metrics.")
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("make-targets", help="pin one reference clip per speaker from the val split")
    t.add_argument("--data-root", default="data/finetuning_audio")
    t.add_argument("--out", default="data/eval_targets")
    t.add_argument("--seconds", type=float, default=15.0,
                   help="concatenate the speaker's silence-trimmed val clips "
                        "(alphabetical, cross-faded) until the reference reaches this "
                        "many seconds of speech (default 15; 0 = single clip). 15s won "
                        "the listening ladder: it stabilizes the speaker embedding, "
                        "while 30s is out-of-distribution for the frame-condition path "
                        "and reintroduces artifacts. Changes the pinned stimulus "
                        "definition — re-run evals after re-pinning.")
    t.set_defaults(func=cmd_make_targets)

    r = sub.add_parser("run", help="evaluate checkpoints: convert, measure, rank")
    r.add_argument("--run-dir", required=True, help="training run dir (exp/finetune_<x>)")
    r.add_argument("--source-dir", required=True,
                   help="FIXED folder of source wavs (restricted by evaluation plan)")
    r.add_argument("--targets-dir", required=True,
                   help="pinned target reference wavs (<speaker>.wav), see make-targets")
    r.add_argument(
        "--evaluation-plan",
        help="JSON plan that enforces native source speakers, target accent group, "
             "and one assigned target per source; without it, legacy Cartesian "
             "source x target evaluation is used",
    )
    r.add_argument("--config", default=None, help="default: <run-dir>/config.yaml")
    r.add_argument("--out", default=None, help="default: <run-dir>/eval")
    r.add_argument("--steps", default="all", help="'all' or comma list, e.g. 250,500")
    r.add_argument("--include-base", default=None,
                   help="also evaluate this checkpoint as step='base' (e.g. ckpts/xvc.pt)")
    r.add_argument("--transcripts", default=None, help="TSV name<TAB>text for WER reference")
    r.add_argument("--wer-model", default="small", help="whisper model size (default small)")
    r.add_argument("--skip-wer", action="store_true")
    r.add_argument("--accent-clf", action="store_true",
                   help="CommonAccent ECAPA accent-ID confidence (needs speechbrain)")
    r.add_argument("--mos", action="store_true",
                   help="predicted MOS per conversion (UTMOS22-strong via torch.hub; "
                        "measures the robotic/artifact axis that cosine and WER miss)")
    r.add_argument("--loudnorm", action="store_true",
                   help="EBU R128 loudness-normalize saved samples (needs pyloudnorm)")
    r.add_argument("--loudnorm-lufs", type=float, default=-23.0)
    r.add_argument("--out-sr", type=int, default=None,
                   help="resample saved samples, e.g. 24000 for PP delivery")
    r.add_argument("--max-sources", type=int, default=None)
    r.add_argument("--device", type=int, default=0)
    r.add_argument("--latent-hop-length", type=int, default=1280)
    r.add_argument("--mask-target-condition", action="store_true")
    r.add_argument("--streaming", action="store_true",
                   help="streaming mode (HMO deployment path) instead of offline")
    r.add_argument("--chunk", type=int, default=2400)
    r.add_argument("--current", type=int, default=320)
    r.add_argument("--future", type=int, default=0)
    r.add_argument("--smooth", type=int, default=0)
    r.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
