#!/usr/bin/env python3
"""
Prepare X-VC fine-tuning data: select audio from the raw ARCTIC corpora and build
per-group JSONL manifests.

Speaker groups (which speakers, which gender, which raw corpus) live in
``configs/data_groups.yaml`` — never hardcoded here. Default groups: three L2
accents with 4 speakers each (2M+2F) plus a ``native`` CMU-ARCTIC group that gets
the identical treatment (required so native-vs-non-native is not confounded with
base-vs-fine-tuned model; see CHANGES.md).

Two subcommands
---------------
select    Curate ``data/finetuning_audio/{train,val}/<group>/<speaker>/*.wav`` from
          the raw corpora: deterministic per-speaker utterance split (zero
          train/val overlap by construction), ``--minutes-per-speaker all|N``,
          per-file ingest asserts (16 kHz, mono, non-empty, >= min duration) with
          an optional ``--resample`` fallback. Writes ``selection_meta.json``.

manifest  Build ``manifests/<group>/{train,val}.jsonl`` (+ optional ``joint``)
          from the curated tree. Re-verifies every file, hard-asserts zero
          train/val utterance overlap per speaker, supports ``--pair-mode
          {self,cross,mixed}`` (cross pairs via shared ARCTIC prompt IDs) and a
          rehearsal ``--filler-dir``. Writes ``manifest_meta.json`` with sha256
          per manifest, per-speaker minutes and a gender tally.

Typical use::

    python scripts/prepare_finetuning_data.py select   --minutes-per-speaker 10
    python scripts/prepare_finetuning_data.py manifest --joint

IMPORTANT (pair modes): the training dataloader re-assigns roles at load time.
With ``dataloader.reconstruction_ratio: 1.0`` (the self-recon configs) every
manifest line is forced back to source==target regardless of what is written
here. Cross/mixed manifests therefore require a config with
``reconstruction_ratio: 0.0`` and ``reversed_ratio: 0.0``. Cross pairs share the
prompt text but are NOT frame-aligned; the current reconstruction losses compare
output to target frame-by-frame, so cross training is EXPERIMENTAL until an
alignment-tolerant objective exists (see docs/finetuning.md). The script warns
loudly when you ask for it.

Dependencies: standard library + PyYAML (for the groups file). ``--resample``
additionally needs soundfile + (soxr or scipy).

Part of the X-VC accent fine-tuning pipeline. Upstream: https://github.com/Jerrister/X-VC (MIT).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TARGET_SR = 16000
DEFAULT_MIN_DURATION = 3.0  # s; > segment_duration (2.4 s) so every clip yields a full segment


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_groups_config(path: Path) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        try:
            from ruamel.yaml import YAML
            with open(path, "r", encoding="utf-8") as f:
                cfg = YAML(typ="safe").load(f)
        except ImportError:
            print("[error] PyYAML or ruamel.yaml is required to read the groups "
                  "config (pip install pyyaml).", file=sys.stderr)
            raise SystemExit(3)
    for key in ("groups", "sources"):
        if key not in cfg:
            print(f"[error] groups config {path} missing top-level key '{key}'", file=sys.stderr)
            raise SystemExit(3)
    return cfg


# --------------------------------------------------------------------------- #
# WAV probing / ingest asserts
# --------------------------------------------------------------------------- #
def wav_info(path: Path) -> Optional[Tuple[int, int, int, float]]:
    """(sample_rate, channels, sampwidth, seconds) or None if unreadable."""
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as w:
            rate, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            if rate <= 0:
                return None
            return rate, ch, sw, n / float(rate)
    except Exception:
        return None


def check_wav(path: Path, min_duration: float) -> Tuple[bool, str, float]:
    """Ingest asserts for one file. Returns (ok, reason, duration_seconds)."""
    info = wav_info(path)
    if info is None:
        return False, "unreadable", 0.0
    rate, ch, _, dur = info
    if dur <= 0.0:
        return False, "empty", dur
    if rate != TARGET_SR:
        return False, f"sample_rate={rate}!={TARGET_SR}", dur
    if ch != 1:
        return False, f"channels={ch}!=1", dur
    if dur < min_duration:
        return False, f"duration={dur:.2f}s<{min_duration:g}s", dur
    return True, "", dur


def resample_copy(src: Path, dst: Path) -> None:
    """Convert src to 16 kHz mono 16-bit PCM at dst (lazy heavy imports)."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(f"--resample needs soundfile+numpy ({e})")
    data, sr = sf.read(str(src), always_2d=True)
    data = data.mean(axis=1)  # downmix to mono
    if sr != TARGET_SR:
        try:
            import soxr
            data = soxr.resample(data, sr, TARGET_SR)
        except ImportError:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(sr, TARGET_SR)
                data = resample_poly(data, TARGET_SR // g, sr // g)
            except ImportError as e:
                raise RuntimeError(f"--resample needs soxr or scipy ({e})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), np.asarray(data, dtype="float32"), TARGET_SR, subtype="PCM_16")


# --------------------------------------------------------------------------- #
# `select`: raw corpora -> curated data tree
# --------------------------------------------------------------------------- #
def select_speaker(
    speaker: str,
    spec: dict,
    group: str,
    sources: dict,
    data_root: Path,
    minutes_per_speaker: Optional[float],  # None == all
    val_minutes: float,
    min_duration: float,
    seed: int,
    do_resample: bool,
    overwrite: bool,
) -> dict:
    src_cfg = sources[spec["source"]]
    pattern = src_cfg["wav_glob"].format(speaker=speaker)
    raw_files = sorted(Path(src_cfg["root"]).glob(pattern))
    if not raw_files:
        raise SystemExit(
            f"[error] no raw wavs for speaker '{speaker}' "
            f"({Path(src_cfg['root'])}/{pattern}) — check configs/data_groups.yaml"
        )

    # Probe every candidate; violations either exclude (short) or fail/resample.
    usable: List[Tuple[Path, float]] = []
    excluded_short, fixed = 0, 0
    for f in raw_files:
        ok, reason, dur = check_wav(f, min_duration)
        if ok:
            usable.append((f, dur))
        elif reason.startswith("duration"):
            excluded_short += 1  # a selection criterion, not corruption
        elif do_resample and (reason.startswith("sample_rate") or reason.startswith("channels")):
            fixed += 1
            usable.append((f, dur))  # converted at copy time below
        else:
            raise SystemExit(f"[error] ingest assert failed for {f}: {reason} "
                             f"(use --resample for format fixes)")

    # Deterministic split: sorted file list + per-speaker seeded shuffle.
    rng = random.Random(f"{seed}|{group}|{speaker}")
    order = list(range(len(usable)))
    rng.shuffle(order)

    val_sel, train_sel = [], []
    val_sec = train_sec = 0.0
    budget = None if minutes_per_speaker is None else minutes_per_speaker * 60.0
    for i in order:
        f, dur = usable[i]
        if val_sec < val_minutes * 60.0:
            val_sel.append((f, dur)); val_sec += dur
        elif budget is None or train_sec < budget:
            train_sel.append((f, dur)); train_sec += dur

    for split, sel in (("train", train_sel), ("val", val_sel)):
        out_dir = data_root / split / group / speaker
        if out_dir.exists():
            if not overwrite:
                raise SystemExit(
                    f"[error] {out_dir} already exists; pass --overwrite to re-select "
                    f"(re-selection changes the train/val split — do it only before counted runs)"
                )
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for f, _ in sel:
            dst = out_dir / f.name
            ok, _, _ = check_wav(f, min_duration)
            if ok:
                shutil.copy2(f, dst)
            else:
                resample_copy(f, dst)
            # Post-copy assert: never trust the write.
            ok2, reason2, _ = check_wav(dst, min_duration)
            if not ok2:
                raise SystemExit(f"[error] post-copy assert failed for {dst}: {reason2}")

    return {
        "group": group,
        "gender": spec["gender"],
        "source": spec["source"],
        "raw_candidates": len(raw_files),
        "excluded_below_min_duration": excluded_short,
        "format_fixed_by_resample": fixed,
        "train": {"utterances": len(train_sel), "minutes": round(train_sec / 60.0, 3)},
        "val": {"utterances": len(val_sel), "minutes": round(val_sec / 60.0, 3)},
    }


def cmd_select(args) -> int:
    repo_root = repo_root_from_here()
    groups_cfg = load_groups_config(Path(args.groups_config))
    data_root = Path(args.data_root) if args.data_root else repo_root / "data" / "finetuning_audio"
    seed = args.seed if args.seed is not None else int(groups_cfg.get("seed", 1234))

    minutes = None if str(args.minutes_per_speaker).lower() == "all" else float(args.minutes_per_speaker)
    wanted = args.groups or sorted(groups_cfg["groups"].keys())

    print(f"Groups config : {args.groups_config}")
    print(f"Data root     : {data_root}")
    print(f"Minutes/spk   : {'all' if minutes is None else minutes} (train) + {args.val_minutes} (val)")
    print(f"Min duration  : {args.min_duration}s | seed {seed} | resample={args.resample}")
    print("-" * 76)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "groups_config": str(args.groups_config),
        "seed": seed,
        "minutes_per_speaker": "all" if minutes is None else minutes,
        "val_minutes": args.val_minutes,
        "min_duration_s": args.min_duration,
        "target_sample_rate": TARGET_SR,
        "speakers": {},
    }
    for group in wanted:
        if group not in groups_cfg["groups"]:
            raise SystemExit(f"[error] unknown group '{group}' "
                             f"(available: {sorted(groups_cfg['groups'])})")
        for speaker, spec in sorted(groups_cfg["groups"][group]["speakers"].items()):
            stat = select_speaker(
                speaker, spec, group, groups_cfg["sources"], data_root,
                minutes, args.val_minutes, args.min_duration, seed,
                args.resample, args.overwrite,
            )
            meta["speakers"][speaker] = stat
            print(f"{group:8s} {speaker:6s} [{spec['gender']}] | "
                  f"train {stat['train']['utterances']:4d} utt ({stat['train']['minutes']:6.2f} min) | "
                  f"val {stat['val']['utterances']:3d} utt ({stat['val']['minutes']:5.2f} min) | "
                  f"short-excluded {stat['excluded_below_min_duration']}")

    meta_path = data_root / "selection_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("-" * 76)
    print(f"Selection meta written to {meta_path}")
    print("Now run:  python scripts/prepare_finetuning_data.py manifest --joint")
    return 0


# --------------------------------------------------------------------------- #
# `manifest`: curated tree -> JSONL manifests (+ meta, + overlap asserts)
# --------------------------------------------------------------------------- #
def to_manifest_path(wav: Path, repo_root: Path, path_style: str, abs_prefix: Optional[str]) -> str:
    if path_style == "absolute":
        return wav.resolve().as_posix()
    rel_posix = Path(os.path.relpath(wav.resolve(), start=repo_root)).as_posix()
    if abs_prefix:
        return f"{abs_prefix.rstrip('/')}/{rel_posix}"
    return rel_posix


def scan_split(
    data_root: Path, split: str, group: str, min_duration: float,
) -> Dict[str, List[Tuple[str, Path, float]]]:
    """{speaker: [(utt_id, wav_path, duration), ...]} with per-file hard asserts."""
    out: Dict[str, List[Tuple[str, Path, float]]] = {}
    group_dir = data_root / split / group
    if not group_dir.is_dir():
        return out
    for spk_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        rows = []
        for wav_path in sorted(spk_dir.glob("*.wav")):
            ok, reason, dur = check_wav(wav_path, min_duration)
            if not ok:
                raise SystemExit(f"[error] manifest assert failed for {wav_path}: {reason} — "
                                 f"re-run `select` (with --resample if needed)")
            rows.append((wav_path.stem, wav_path, dur))
        out[spk_dir.name] = rows
    return out


def self_record(speaker: str, utt: str, path_str: str) -> dict:
    # target_utt carries a 3-char "_ft" suffix because the loader indexes with
    # target_utt[:-3]; stripping it recovers "{speaker}_{uttid}".
    label = f"{speaker}_{utt}_ft"
    return {"source_utt": label, "source_wav_path": path_str,
            "target_utt": label, "target_wav_path": path_str}


def cross_records(
    split_data: Dict[str, List[Tuple[str, Path, float]]],
    paths: Dict[Tuple[str, str], str],
    n_wanted: int,
    rng: random.Random,
) -> List[dict]:
    """Sample ordered cross-speaker pairs sharing an ARCTIC prompt ID."""
    by_utt: Dict[str, List[str]] = {}
    for spk, rows in split_data.items():
        for utt, _, _ in rows:
            by_utt.setdefault(utt, []).append(spk)
    candidates = []
    for utt, spks in sorted(by_utt.items()):
        for a in sorted(spks):
            for b in sorted(spks):
                if a != b:
                    candidates.append((a, b, utt))
    rng.shuffle(candidates)
    out = []
    for a, b, utt in candidates[:n_wanted]:
        out.append({
            "source_utt": f"{a}_{utt}_ft", "source_wav_path": paths[(a, utt)],
            "target_utt": f"{b}_{utt}_ft", "target_wav_path": paths[(b, utt)],
        })
    if len(out) < n_wanted:
        print(
            f"[WARN] cross pairs: only {len(out)} shared-prompt pairs available "
            f"({n_wanted} requested). Per-speaker random selection shares few prompt "
            f"IDs; re-run `select` with '--minutes-per-speaker all' (or coordinate "
            f"prompts across speakers) to raise the ceiling.", file=sys.stderr,
        )
    return out


def cmd_manifest(args) -> int:
    repo_root = repo_root_from_here()
    groups_cfg = load_groups_config(Path(args.groups_config))
    data_root = Path(args.data_root) if args.data_root else repo_root / "data" / "finetuning_audio"
    out_root = Path(args.out_root) if args.out_root else data_root / "manifests"
    seed = args.seed if args.seed is not None else int(groups_cfg.get("seed", 1234))
    rng = random.Random(f"{seed}|manifest")

    wanted = args.groups or sorted(g for g in groups_cfg["groups"]
                                   if (data_root / "train" / g).is_dir())
    if not wanted:
        raise SystemExit(f"[error] no group directories under {data_root / 'train'} — run `select` first")

    if args.pair_mode != "self":
        print(
            "[WARN] pair-mode '%s': cross pairs share prompt text but are NOT frame-aligned;\n"
            "       the reconstruction losses compare output to target frame-by-frame, so\n"
            "       cross training is EXPERIMENTAL. Also: the training config MUST set\n"
            "       dataloader.reconstruction_ratio: 0.0 (and reversed_ratio: 0.0) or the\n"
            "       loader silently rewrites every pair back to self-reconstruction."
            % args.pair_mode, file=sys.stderr,
        )

    # Optional rehearsal filler (e.g. a VCTK slice) — self-recon only, train only.
    filler_files: List[Path] = []
    if args.filler_dir:
        for f in sorted(Path(args.filler_dir).rglob("*.wav")):
            ok, reason, _ = check_wav(f, args.min_duration)
            if not ok:
                raise SystemExit(f"[error] filler assert failed for {f}: {reason}")
            filler_files.append(f)
        if not filler_files:
            raise SystemExit(f"[error] --filler-dir {args.filler_dir} contains no wavs")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "groups_config": str(args.groups_config),
        "seed": seed,
        "pair_mode": args.pair_mode,
        "cross_ratio": args.cross_ratio if args.pair_mode == "mixed" else None,
        "val_pair_mode": "self (always: val loss stays comparable across pair modes)",
        "filler_dir": args.filler_dir,
        "filler_ratio": args.filler_ratio if args.filler_dir else None,
        "min_duration_s": args.min_duration,
        "config_requirement": (
            "reconstruction_ratio must be 0.0 for cross/mixed manifests"
            if args.pair_mode != "self" else
            "self-recon manifests; reconstruction_ratio 1.0 is consistent"
        ),
        "groups": {},
    }

    joint = {"train": [], "val": []}
    exit_code = 0
    for group in wanted:
        gspec = groups_cfg["groups"].get(group, {}).get("speakers", {})
        gmeta = {"speakers": {}, "gender_tally": {"M": 0, "F": 0, "?": 0}}
        records = {"train": [], "val": []}
        split_scan = {}

        for split in ("train", "val"):
            split_scan[split] = scan_split(data_root, split, group, args.min_duration)
            if not split_scan[split]:
                print(f"[warn] no {split} wavs for group '{group}'", file=sys.stderr)
                exit_code = 1

        # Roster check + zero train/val overlap per speaker (hard).
        found = sorted(set(split_scan["train"]) | set(split_scan["val"]))
        expected = sorted(gspec.keys())
        if expected and found != expected:
            print(f"[warn] group '{group}': on-disk speakers {found} != groups.yaml {expected}",
                  file=sys.stderr)
            exit_code = 1
        for spk in found:
            tr = {u for u, _, _ in split_scan["train"].get(spk, [])}
            va = {u for u, _, _ in split_scan["val"].get(spk, [])}
            overlap = tr & va
            assert not overlap, (
                f"train/val utterance overlap for {group}/{spk}: {sorted(overlap)[:5]}"
            )
            gender = gspec.get(spk, {}).get("gender", "?")
            gmeta["gender_tally"][gender] = gmeta["gender_tally"].get(gender, 0) + 1
            gmeta["speakers"][spk] = {
                "gender": gender,
                "train_utts": len(tr), "val_utts": len(va),
                "train_minutes": round(sum(d for _, _, d in split_scan["train"].get(spk, [])) / 60.0, 3),
                "val_minutes": round(sum(d for _, _, d in split_scan["val"].get(spk, [])) / 60.0, 3),
            }

        # Build records. Val is ALWAYS self-reconstruction.
        path_of: Dict[Tuple[str, str], str] = {}
        for split in ("train", "val"):
            for spk, rows in split_scan[split].items():
                for utt, wav_path, _ in rows:
                    path_of[(spk, utt)] = to_manifest_path(
                        wav_path, repo_root, args.path_style, args.abs_prefix)

        for spk, rows in sorted(split_scan["val"].items()):
            for utt, _, _ in rows:
                records["val"].append(self_record(spk, utt, path_of[(spk, utt)]))

        self_train = []
        for spk, rows in sorted(split_scan["train"].items()):
            for utt, _, _ in rows:
                self_train.append(self_record(spk, utt, path_of[(spk, utt)]))

        if args.pair_mode == "self":
            records["train"] = self_train
        elif args.pair_mode == "cross":
            # size-matched to the self-recon budget for comparability
            records["train"] = cross_records(split_scan["train"], path_of, len(self_train), rng)
        else:  # mixed
            n_cross = round(args.cross_ratio / (1.0 - args.cross_ratio) * len(self_train))
            records["train"] = self_train + cross_records(
                split_scan["train"], path_of, n_cross, rng)

        # Filler (train only, self-recon, speaker label "filler").
        if filler_files:
            n_filler = round(args.filler_ratio * len(records["train"]))
            picks = random.Random(f"{seed}|filler|{group}").sample(
                filler_files, min(n_filler, len(filler_files)))
            for f in picks:
                records["train"].append(self_record(
                    "filler", f.stem,
                    to_manifest_path(f, repo_root, args.path_style, args.abs_prefix)))
            gmeta["filler_utts"] = len(picks)

        # Write.
        for split in ("train", "val"):
            out_path = out_root / group / f"{split}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in records[split]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            gmeta[f"{split}_manifest"] = Path(os.path.relpath(out_path, repo_root)).as_posix()
            gmeta[f"{split}_sha256"] = sha256_file(out_path)
            gmeta[f"{split}_records"] = len(records[split])
            joint[split].extend(records[split])

        meta["groups"][group] = gmeta
        tally = gmeta["gender_tally"]
        print(f"{group:8s} | train {len(records['train']):4d} rec | "
              f"val {len(records['val']):3d} rec | speakers {found} "
              f"| gender M={tally['M']} F={tally['F']}")
        if expected and (tally["M"] == 0 or tally["F"] == 0) and group != "native":
            print(f"         [warn] group '{group}' is single-gender on disk — "
                  f"gender is confounded with accent for this group", file=sys.stderr)

    # Joint manifests (single-checkpoint training over all groups).
    if args.joint:
        jmeta = {}
        for split in ("train", "val"):
            out_path = out_root / "joint" / f"{split}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in joint[split]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jmeta[f"{split}_manifest"] = Path(os.path.relpath(out_path, repo_root)).as_posix()
            jmeta[f"{split}_sha256"] = sha256_file(out_path)
            jmeta[f"{split}_records"] = len(joint[split])
        meta["groups"]["joint"] = jmeta
        print(f"{'joint':8s} | train {len(joint['train']):4d} rec | val {len(joint['val']):3d} rec")

    meta_path = out_root / "manifest_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("-" * 76)
    print(f"Manifest meta written to {meta_path}")
    return exit_code


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--groups-config", default="configs/data_groups.yaml",
                       help="speaker roster YAML (default: configs/data_groups.yaml)")
        p.add_argument("--data-root", default=None,
                       help="curated tree root (default: <repo>/data/finetuning_audio)")
        p.add_argument("--groups", nargs="*", default=None,
                       help="subset of groups (default: all)")
        p.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION,
                       help=f"minimum clip seconds (default {DEFAULT_MIN_DURATION:g})")
        p.add_argument("--seed", type=int, default=None,
                       help="RNG seed (default: `seed` in the groups config)")

    p = sub.add_parser("select", help="curate the data tree from the raw corpora")
    common(p)
    p.add_argument("--minutes-per-speaker", default="10",
                   help="train minutes per speaker, or 'all' (default 10)")
    p.add_argument("--val-minutes", type=float, default=1.0,
                   help="val minutes per speaker (default 1.0)")
    p.add_argument("--resample", action="store_true",
                   help="convert wrong-format files to 16 kHz mono instead of failing")
    p.add_argument("--overwrite", action="store_true",
                   help="replace existing speaker directories (changes the split!)")
    p.set_defaults(func=cmd_select)

    m = sub.add_parser("manifest", help="build JSONL manifests from the curated tree")
    common(m)
    m.add_argument("--out-root", default=None,
                   help="manifest output root (default: <data-root>/manifests)")
    m.add_argument("--pair-mode", choices=["self", "cross", "mixed"], default="self",
                   help="train pairing (default self; cross/mixed are EXPERIMENTAL, "
                        "see module docstring)")
    m.add_argument("--cross-ratio", type=float, default=0.5,
                   help="cross fraction for --pair-mode mixed (default 0.5)")
    m.add_argument("--filler-dir", default=None,
                   help="directory of rehearsal wavs (e.g. VCTK slice), mixed into train")
    m.add_argument("--filler-ratio", type=float, default=0.25,
                   help="filler records as a fraction of group train records (default 0.25)")
    m.add_argument("--joint", action="store_true",
                   help="also write manifests/joint/{train,val}.jsonl over all groups")
    m.add_argument("--path-style", choices=["relative", "absolute"], default="relative")
    m.add_argument("--abs-prefix", default=None,
                   help="prefix for relative paths, e.g. '/workspace/X-VC'")
    m.set_defaults(func=cmd_manifest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
