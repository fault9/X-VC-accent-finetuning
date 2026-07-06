#!/usr/bin/env python3
"""
DTW-align parallel cross pairs for accent conversion (Option B, phase B2).

The B1 pilot used UNIFORM time-stretch (global duration match only) and
collapsed into garbled output (WER > 1.0), whether or not semantic_adapter was
frozen.  That rules out the adapter choice as the sole cause; it does not by
itself prove that alignment is the only cause.

For each (native source, L2 target) pair reading the
SAME ARCTIC prompt, it DTW-aligns the two on speaker-normalised MFCCs (pairs are
gender-matched to reduce speaker mismatch), then warps the target audio onto
the source timeline. The validated B2 overlap-add backend remains the default;
an optional Rubber Band backend performs phase-coherent variable-rate warping
from sparse DTW anchors.
Frame t of the warped target is then (approximately) the SAME phoneme as frame t
of the source, so the losses become coherent.

It reports a diagnostic mel-distance after DTW warp and uniform stretch. Since
DTW is fitted on related spectral features, this is not an independent proof
of alignment quality. Validate and listen to the warped targets before training.

    python scripts/align_crosspairs.py \
        --train-manifest data/crosspair_hindi/manifests/train.jsonl \
        --val-manifest data/crosspair_hindi/manifests/val.jsonl \
        --resplit-val-prompts 40 \
        --warp-method rubberband \
        --l2-root /mnt/d/datasets/arctic-audio/l2arctic \
        --out data/crosspair_hindi_dtw

Rubber Band mode requires version 3 or newer and explicitly selects its R3
(Finer) engine. Use ``--limit`` for a listening A/B before a full rebuild.

Copies source WAVs and writes warped targets, split-preserving manifests, and
align_meta.json. Part of the X-VC pipeline (upstream Jerrister/X-VC, MIT).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import librosa
import numpy as np

SR = 16000
HOP = 160          # 10 ms frames
N_MFCC = 13
EVAL_PROMPTS = {f"arctic_b{i:04d}" for i in (2, 4, 5, 6, 7, 8, 9, 10, 11, 12)}


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(
                f"{path}: expected 16 kHz mono PCM16, got "
                f"rate={w.getframerate()}, channels={w.getnchannels()}, "
                f"sample_width={w.getsampwidth()}"
            )
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def mfcc_cmn(x: np.ndarray) -> np.ndarray:
    """MFCCs + delta, cepstral-mean-normalised (speaker-robust-ish). (frames, dim)."""
    m = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=N_MFCC, hop_length=HOP)
    d = librosa.feature.delta(m)
    f = np.vstack([m, d]).T
    f = f - f.mean(axis=0, keepdims=True)
    f = f / (f.std(axis=0, keepdims=True) + 1e-8)
    return f


def dtw_path(A: np.ndarray, B: np.ndarray, band: float = 0.2):
    """Classic DTW with a Sakoe-Chiba band. Returns list of (i,j) source->target."""
    na, nb = len(A), len(B)
    w = max(int(band * max(na, nb)), abs(na - nb) + 1)
    INF = np.inf
    D = np.full((na + 1, nb + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0
    # cost = euclidean; compute per-cell within band
    for i in range(1, na + 1):
        jlo = max(1, int(i * nb / na) - w)
        jhi = min(nb, int(i * nb / na) + w)
        a = A[i - 1]
        for j in range(jlo, jhi + 1):
            cost = np.linalg.norm(a - B[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # backtrack
    i, j = na, nb
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def _dense_mapping(path, n_source_frames: int) -> np.ndarray:
    """Interpolate a monotonic target-frame position for every source frame."""
    from collections import defaultdict

    s2t = defaultdict(list)
    for i, j in path:
        s2t[i].append(j)
    known_i = np.asarray(sorted(s2t), dtype=np.float64)
    if len(known_i) == 0:
        raise ValueError("empty DTW path")
    known_j = np.asarray([np.median(s2t[int(i)]) for i in known_i], dtype=np.float64)
    return np.interp(np.arange(n_source_frames), known_i, known_j)


def warp_target_to_source(src: np.ndarray, tgt: np.ndarray, path, win_ms: int = 40):
    """DTW-guided, gap-free overlap-add; output length equals source length.

    This is OLA, not WSOLA: the DTW path supplies the target position and no
    secondary waveform-similarity search is performed. Reflect-padding and a
    dense mapping ensure boundary frames are covered.
    """
    n = int(SR * win_ms / 1000)
    n -= n % 2
    half = n // 2
    win = np.hanning(n).astype(np.float32)
    n_source_frames = max(1, int(np.ceil(len(src) / HOP)) + 1)
    mapping = _dense_mapping(path, n_source_frames)

    tgt_pad = np.pad(tgt, (half, half), mode="reflect")
    out = np.zeros(len(src) + 2 * half + HOP, dtype=np.float32)
    norm = np.zeros_like(out)
    for i, target_frame in enumerate(mapping):
        c_out = half + i * HOP
        c_in = half + int(round(target_frame * HOP))
        c_in = int(np.clip(c_in, half, len(tgt_pad) - half))
        segment = tgt_pad[c_in - half:c_in + half]
        if len(segment) != n:
            continue
        out[c_out - half:c_out + half] += segment * win
        norm[c_out - half:c_out + half] += win

    crop = slice(half, half + len(src))
    if np.any(norm[crop] <= 1e-6):
        raise RuntimeError("DTW OLA left uncovered output samples")
    return (out[crop] / norm[crop]).astype(np.float32)


def _rubberband_time_map(path, source_len: int, target_len: int,
                         anchor_hop_frames: int = 10) -> list[tuple[int, int]]:
    """Convert source->target DTW frames to sparse target->source sample anchors.

    Rubber Band maps positions in its input audio (the raw target recording) to
    positions in its output audio (the native source timeline).  Duplicate and
    flat DTW steps are collapsed because Rubber Band expects a useful monotonic
    key-frame map rather than every cell in the DTW path.
    """
    if source_len <= 0 or target_len <= 0:
        raise ValueError("source and target audio must be non-empty")
    if anchor_hop_frames < 1:
        raise ValueError("anchor_hop_frames must be positive")

    # Aggregate all source positions assigned to each target frame, then use a
    # robust centre.  DTW is monotonic, so these centres are non-decreasing.
    target_to_source: dict[int, list[int]] = {}
    for source_frame, target_frame in path:
        target_to_source.setdefault(int(target_frame), []).append(int(source_frame))
    if not target_to_source:
        raise ValueError("empty DTW path")

    dense = [
        (target_frame, int(round(float(np.median(source_frames)))))
        for target_frame, source_frames in sorted(target_to_source.items())
    ]

    # Keep anchors about 100 ms apart.  Enforce strict progress in both axes;
    # endpoints use audio lengths (not last indices), as required by
    # Rubber Band's time-map format.
    anchors: list[tuple[int, int]] = [(0, 0)]
    last_target_frame = -anchor_hop_frames
    for target_frame, source_frame in dense:
        if target_frame - last_target_frame < anchor_hop_frames:
            continue
        target_sample = min(target_len, target_frame * HOP)
        source_sample = min(source_len, source_frame * HOP)
        if target_sample > anchors[-1][0] and source_sample > anchors[-1][1]:
            anchors.append((target_sample, source_sample))
            last_target_frame = target_frame

    while len(anchors) > 1 and (
        anchors[-1][0] >= target_len or anchors[-1][1] >= source_len
    ):
        anchors.pop()
    anchors.append((target_len, source_len))

    if any(a[0] >= b[0] or a[1] >= b[1] for a, b in zip(anchors, anchors[1:])):
        raise RuntimeError("Rubber Band time map is not strictly monotonic")
    return anchors


def rubberband_version(executable: str = "rubberband") -> str:
    """Return executable version for dataset provenance."""
    try:
        proc = subprocess.run(
            [executable, "--version"], check=True, capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    output = (proc.stdout or proc.stderr).strip().splitlines()
    return output[0] if output else "unknown"


def _rubberband_major(version: str) -> int | None:
    match = re.search(r"(?<!\d)(\d+)\.(?:\d+)", version)
    return int(match.group(1)) if match else None


def _local_stretch_ratios(time_map: list[tuple[int, int]]) -> np.ndarray:
    """Output-duration/input-duration ratio between adjacent key frames."""
    points = np.asarray(time_map, dtype=np.float64)
    delta = np.diff(points, axis=0)
    return delta[:, 1] / delta[:, 0]


def _latent_alignment_positions(
    time_map: list[tuple[int, int]],
    source_len: int,
    target_len: int,
    frame_hop: int = 320,
) -> np.ndarray:
    """Map target 50-Hz frame centres to normalized source time.

    The normalized representation is independent of small encoder boundary
    differences. Training uses it to sample both post-adapter WhisperVQ
    features and quantized SAC acoustic features onto the target timeline.
    """
    if source_len < 2 or target_len < frame_hop:
        raise ValueError("audio is too short for latent alignment")
    points = np.asarray(time_map, dtype=np.float64)
    target_frames = target_len // frame_hop
    target_centres = (np.arange(target_frames, dtype=np.float64) + 0.5) * frame_hop
    source_samples = np.interp(target_centres, points[:, 0], points[:, 1])
    positions = source_samples / float(source_len - 1)
    positions = np.maximum.accumulate(np.clip(positions, 0.0, 1.0))
    return positions.astype(np.float32)


def _rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-12))


def _stabilize_time_map(
    time_map: list[tuple[int, int]],
    min_ratio: float,
    max_ratio: float,
    target_len: int,
    min_anchor_hz: float = 2.0,
) -> tuple[list[tuple[int, int]], int]:
    """Remove only anchors responsible for unsafe instantaneous stretch rates.

    Merging a pathological 100-ms interval with a neighbour lets Rubber Band
    vary tempo over a longer context instead of producing a flutter or rejecting
    the whole utterance. Endpoints are immutable. A density floor prevents the
    repaired map from silently collapsing into a global duration stretch.
    """
    anchors = list(time_map)
    original_count = len(anchors)

    def violation_score(candidate: list[tuple[int, int]]) -> tuple[float, float]:
        ratios = _local_stretch_ratios(candidate)
        low = np.maximum(min_ratio - ratios, 0.0) / min_ratio
        high = np.maximum(ratios - max_ratio, 0.0) / max_ratio
        violation = low + high
        return float(np.max(violation)), float(np.sum(violation))

    while len(anchors) > 2:
        ratios = _local_stretch_ratios(anchors)
        bad = np.flatnonzero((ratios < min_ratio) | (ratios > max_ratio))
        if len(bad) == 0:
            break
        interval = int(bad[np.argmax(
            np.maximum(min_ratio / np.maximum(ratios[bad], 1e-12),
                       ratios[bad] / max_ratio)
        )])
        removable = []
        if interval > 0:
            removable.append(interval)
        if interval + 1 < len(anchors) - 1:
            removable.append(interval + 1)
        if not removable:
            break
        candidates = []
        for index in removable:
            candidate = anchors[:index] + anchors[index + 1:]
            candidates.append((violation_score(candidate), index, candidate))
        _, _, anchors = min(candidates, key=lambda item: item[0])

    ratios = _local_stretch_ratios(anchors)
    if np.any((ratios < min_ratio) | (ratios > max_ratio)):
        raise ValueError(
            "DTW map cannot satisfy bounded local stretch even after coarsening"
        )
    duration = target_len / SR
    minimum_count = max(3, int(np.ceil(duration * min_anchor_hz)) + 1)
    if len(anchors) < minimum_count:
        raise ValueError(
            f"DTW map collapsed to {len(anchors)} anchors; need at least "
            f"{minimum_count} for {duration:.2f}s"
        )
    return anchors, original_count - len(anchors)


def rubberband_warp_target_to_source(
    src: np.ndarray,
    tgt: np.ndarray,
    path,
    anchor_hop_frames: int = 10,
    engine: str = "r3",
    min_local_stretch: float = 1 / 3,
    max_local_stretch: float = 3.0,
    warp_side: str = "target",
) -> tuple[np.ndarray, int, np.ndarray, int, int]:
    """DTW-guided variable-rate Rubber Band warp of one side of a pair.

    ``target`` preserves the native source timeline and warps the accented
    recording, which is the historical behaviour. ``source`` inverts the same
    key-frame map: the native recording is warped onto the accented timeline
    and the accented supervision waveform can remain pristine. The latter is
    preferable for training when audible time-stretch artefacts would otherwise
    become part of the reconstruction/GAN target.

    The command is invoked directly instead of through pyrubberband so engine
    selection is explicit.  ``rubberband`` defaults to R2 even in 3.x.
    """
    executable = shutil.which("rubberband")
    if executable is None:
        raise RuntimeError(
            "rubberband executable not found; install rubberband-cli and ensure "
            "it is on PATH"
        )
    version = rubberband_version(executable)
    major = _rubberband_major(version)
    if engine == "r3" and (major is None or major < 3):
        raise RuntimeError(
            f"Rubber Band R3 requires version >=3; found {version!r}. "
            "Ubuntu 22.04 ships R2-only 2.0; use Ubuntu 24.04 or build >=3."
        )

    if warp_side not in {"target", "source"}:
        raise ValueError(f"unsupported warp_side: {warp_side!r}")

    # The canonical map is raw target input -> native source output. Inverting
    # every point gives native source input -> raw target output while retaining
    # exactly the same DTW correspondence for a controlled A/B experiment.
    time_map = _rubberband_time_map(
        path, source_len=len(src), target_len=len(tgt),
        anchor_hop_frames=anchor_hop_frames,
    )
    if warp_side == "source":
        time_map = [(output_sample, input_sample)
                    for input_sample, output_sample in time_map]
        input_audio = src
        expected_len = len(tgt)
    else:
        input_audio = tgt
        expected_len = len(src)
    original_anchor_count = len(time_map)
    time_map, anchors_removed = _stabilize_time_map(
        time_map,
        min_ratio=min_local_stretch,
        max_ratio=max_local_stretch,
        target_len=len(input_audio),
    )
    local_ratios = _local_stretch_ratios(time_map)
    if (
        np.min(local_ratios) < min_local_stretch
        or np.max(local_ratios) > max_local_stretch
    ):
        raise ValueError(
            "pathological local DTW stretch: "
            f"min={np.min(local_ratios):.3f}, max={np.max(local_ratios):.3f}, "
            f"allowed=[{min_local_stretch:.3f}, {max_local_stretch:.3f}]"
        )

    with tempfile.TemporaryDirectory(prefix="xvc-rubberband-") as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / "input.wav"
        output_path = tmpdir / "output.wav"
        map_path = tmpdir / "timemap.txt"
        write_wav(input_path, input_audio)
        with map_path.open("w", encoding="ascii", newline="\n") as handle:
            for input_sample, output_sample in time_map:
                handle.write(f"{input_sample} {output_sample}\n")
        command = [
            executable,
            "-q",
            "-3" if engine == "r3" else "-2",
            "--time", f"{expected_len / len(input_audio):.12f}",
            "--timemap", str(map_path),
            str(input_path), str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Rubber Band failed: " + (exc.stderr or exc.stdout or str(exc))
            ) from exc
        warped = read_wav(str(output_path))

    # Rubber Band may differ at the synthesis boundary. Never hide lost speech:
    # a larger correction is permitted only when the affected tail is already
    # effectively silent.
    delta = len(warped) - expected_len
    tolerance = int(0.35 * SR)
    tail_window = min(len(warped), int(0.10 * SR))
    tail_dbfs = _rms_dbfs(warped[-tail_window:]) if tail_window else -240.0
    if abs(delta) > tolerance:
        raise RuntimeError(
            f"Rubber Band output length mismatch: got {len(warped)}, "
            f"expected {expected_len} (delta {delta}, tail={tail_dbfs:.1f} dBFS)"
        )
    if delta > 0:
        removed_dbfs = _rms_dbfs(warped[expected_len:])
        if removed_dbfs > -35.0:
            raise RuntimeError(
                f"refusing to crop audible Rubber Band tail ({removed_dbfs:.1f} dBFS)"
            )
        warped = warped[:expected_len]
    elif delta < 0:
        if tail_dbfs > -35.0:
            raise RuntimeError(
                f"refusing to pad after audible Rubber Band tail ({tail_dbfs:.1f} dBFS)"
            )
        # Avoid a click if the returned signal does not already end in silence.
        fade = min(int(0.01 * SR), len(warped))
        if fade:
            warped[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        warped = np.pad(warped, (0, -delta), mode="constant")
    if not np.all(np.isfinite(warped)):
        raise RuntimeError("Rubber Band produced non-finite samples")
    return (
        warped.astype(np.float32, copy=False), delta, local_ratios,
        anchors_removed, original_anchor_count,
    )


def uniform_resample_len(tgt: np.ndarray, target_len: int) -> np.ndarray:
    if len(tgt) == target_len:
        return tgt
    idx = np.linspace(0, len(tgt) - 1, target_len)
    return np.interp(idx, np.arange(len(tgt)), tgt).astype(np.float32)


def mel_dist(a: np.ndarray, b: np.ndarray) -> float:
    L = min(len(a), len(b))
    ma = librosa.feature.melspectrogram(y=a[:L], sr=SR, hop_length=HOP, n_mels=40)
    mb = librosa.feature.melspectrogram(y=b[:L], sr=SR, hop_length=HOP, n_mels=40)
    ma = librosa.power_to_db(ma); mb = librosa.power_to_db(mb)
    k = min(ma.shape[1], mb.shape[1])
    return float(np.mean(np.abs(ma[:, :k] - mb[:, :k])))


def prompt_id(row: dict) -> str:
    stem = Path(row["source_wav_path"]).stem
    if "_arctic_" not in stem:
        raise ValueError(f"cannot derive ARCTIC prompt from {stem!r}")
    return "arctic_" + stem.split("_arctic_", 1)[1]


def load_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def raw_target_path(row: dict, l2_root: Path) -> Path:
    utt = row["target_utt"][:-3] if row["target_utt"].endswith("_ft") else row["target_utt"]
    target_speaker, rest = utt.split("__", 1)
    prompt = rest.split("_", 1)[1]
    return l2_root / target_speaker / "wav" / f"{prompt}.wav"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def filter_input_rows(
    rows: list[dict],
    l2_root: Path,
    source_root: Path,
    minimum: float,
    min_global_stretch: float,
    max_global_stretch: float,
) -> tuple[list[dict], dict[str, int]]:
    kept = []
    stats = {"short": 0, "global_stretch": 0}
    for row in rows:
        source = Path(row["source_wav_path"])
        if not source.is_absolute():
            source = source_root / source
        target = raw_target_path(row, l2_root)
        source_duration = wav_duration(source)
        target_duration = wav_duration(target)
        if source_duration < minimum or target_duration < minimum:
            stats["short"] += 1
            continue
        ratio = source_duration / target_duration
        if not min_global_stretch <= ratio <= max_global_stretch:
            stats["global_stretch"] += 1
            continue
        kept.append(row)
    return kept, stats


def target_key(row: dict) -> str:
    target = raw_target_path(row, Path("."))
    return f"{target.parent.parent.name}:{target.stem}"


def load_exclusions(path: str | None) -> tuple[set[str], set[str]]:
    """Read pair IDs or ``TARGET_SPEAKER:arctic_prompt`` exclusions."""
    pair_ids: set[str] = set()
    target_ids: set[str] = set()
    if not path:
        return pair_ids, target_ids
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        if ":" in value:
            target_ids.add(value)
        else:
            pair_ids.add(value)
    return pair_ids, target_ids


def source_speaker(row: dict) -> str:
    return Path(row["source_wav_path"]).stem.split("_arctic_", 1)[0]


def stratified_limit(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Deterministically round-robin a per-split cap across source speakers."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(source_speaker(row), []).append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    speakers = sorted(groups)
    while len(result) < limit and any(groups.values()):
        for speaker in speakers:
            if groups[speaker] and len(result) < limit:
                result.append(groups[speaker].pop())
    return result


def clean_reference_path(row: dict, l2_root: Path,
                         pools: dict[str, list[Path]], minimum: float,
                         forbidden_prompts: set[str]) -> Path:
    """Choose a deterministic clean, different-prompt reference for the L2 speaker."""
    target = raw_target_path(row, l2_root)
    speaker = target.parent.parent.name
    if speaker not in pools:
        pools[speaker] = [
            path for path in sorted(target.parent.glob("*.wav"))
            if path.stem not in forbidden_prompts
            and wav_duration(path) >= minimum
        ]
    candidates = [path for path in pools[speaker] if path.stem != target.stem]
    if not candidates:
        raise RuntimeError(f"no clean reference candidates for {target.parent}")
    digest = hashlib.sha256(row["source_utt"].encode("utf-8")).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-manifest", required=True,
                    help="training manifest from build_crosspairs.py")
    ap.add_argument("--val-manifest", required=True,
                    help="validation manifest from build_crosspairs.py")
    ap.add_argument("--resplit-val-prompts", type=int, default=None,
                    help="repair legacy leaky manifests by combining them and "
                         "holding out this many prompt IDs globally")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--source-root", default=".",
        help="root used to resolve relative source_wav_path values",
    )
    ap.add_argument("--l2-root", default="D:/datasets/arctic-audio/l2arctic",
                    help="raw L2 root — align the RAW target (single warp), not the "
                         "build's pre-stretched one (avoids double time-warp artifacts)")
    ap.add_argument("--limit", type=int, default=None, help="cap pairs (debug)")
    ap.add_argument("--min-duration", type=float, default=3.0,
                    help="require both raw recordings to be at least this long")
    ap.add_argument("--min-global-stretch", type=float, default=0.85,
                    help="minimum source/raw-target duration ratio")
    ap.add_argument("--max-global-stretch", type=float, default=1.20,
                    help="maximum source/raw-target duration ratio")
    ap.add_argument("--exclude-file",
                    help="optional pair IDs or SPEAKER:prompt IDs to omit")
    ap.add_argument("--prompts-file",
                    help="optional L2-ARCTIC PROMPTS file copied into the dataset")
    ap.add_argument(
        "--warp-method", choices=("ola", "rubberband", "latent"), default="ola",
        help="alignment backend; latent keeps both waveforms pristine and "
             "writes a training-only source-feature time map",
    )
    ap.add_argument(
        "--warp-side", choices=("target", "source"), default="target",
        help="side to time-warp: target preserves the native timeline; source "
             "keeps the accented supervision waveform pristine (Rubber Band only)",
    )
    ap.add_argument("--anchor-hop-frames", type=int, default=10,
                    help="Rubber Band DTW anchor spacing in 10 ms frames (default: 10)")
    ap.add_argument("--rubberband-engine", choices=("r2", "r3"), default="r3",
                    help="Rubber Band engine; R3 is required for final-quality data")
    ap.add_argument("--min-local-stretch", type=float, default=1 / 3,
                    help="reject DTW maps with a smaller local output/input ratio")
    ap.add_argument("--max-local-stretch", type=float, default=3.0,
                    help="reject DTW maps with a larger local output/input ratio")
    ap.add_argument("--max-anchor-removal-fraction", type=float, default=0.20,
                    help="reject maps needing more adaptive anchor removal")
    args = ap.parse_args()

    if args.warp_side == "source" and args.warp_method != "rubberband":
        ap.error("--warp-side source currently requires --warp-method rubberband")

    out = Path(args.out)
    source_root = Path(args.source_root)
    l2_root = Path(args.l2_root)
    splits = {
        "train": load_manifest(args.train_manifest),
        "val": load_manifest(args.val_manifest),
    }
    forbidden_reference_prompts = (
        {prompt_id(row) for rows in splits.values() for row in rows}
        | EVAL_PROMPTS
    )
    excluded_pairs, excluded_targets = load_exclusions(args.exclude_file)
    excluded_manual = 0
    for name in splits:
        before = len(splits[name])
        splits[name] = [
            row for row in splits[name]
            if row["source_utt"] not in excluded_pairs
            and target_key(row) not in excluded_targets
        ]
        excluded_manual += before - len(splits[name])
    skipped_duration = 0
    skipped_global_stretch = 0
    for name in splits:
        splits[name], filter_stats = filter_input_rows(
            splits[name], l2_root, source_root, args.min_duration,
            args.min_global_stretch, args.max_global_stretch,
        )
        skipped_duration += filter_stats["short"]
        skipped_global_stretch += filter_stats["global_stretch"]
    print(f"[filter] minimum raw duration {args.min_duration:.2f}s: "
          f"kept={sum(map(len, splits.values()))}, short={skipped_duration}, "
          f"global-stretch={skipped_global_stretch}, manual={excluded_manual}")
    train_prompts = {prompt_id(r) for r in splits["train"]}
    val_prompts = {prompt_id(r) for r in splits["val"]}
    overlap = train_prompts & val_prompts
    if overlap:
        if args.resplit_val_prompts is None:
            raise SystemExit(
                "[error] input manifests leak prompt IDs across train/val: "
                + ", ".join(sorted(overlap)[:10])
                + "; rebuild with fixed build_crosspairs.py or pass "
                  "--resplit-val-prompts N to repair legacy manifests"
            )
        combined = splits["train"] + splits["val"]
        prompt_ids = sorted({prompt_id(row) for row in combined})
        random.Random(1234).shuffle(prompt_ids)
        held_out = set(prompt_ids[:args.resplit_val_prompts])
        splits = {
            "train": [row for row in combined if prompt_id(row) not in held_out],
            "val": [row for row in combined if prompt_id(row) in held_out],
        }
        train_prompts = {prompt_id(r) for r in splits["train"]}
        val_prompts = {prompt_id(r) for r in splits["val"]}
        if train_prompts & val_prompts:
            raise RuntimeError("global prompt re-split failed")
        print(f"[repair] re-split legacy manifests by prompt: "
              f"train={len(splits['train'])}, val={len(splits['val'])}")
    if args.limit:
        splits = {
            name: stratified_limit(rows, args.limit, 1234 + index)
            for index, (name, rows) in enumerate(splits.items())
        }

    if args.warp_method == "rubberband":
        executable = shutil.which("rubberband")
        if executable is None:
            raise SystemExit("[error] rubberband executable not found on PATH")
        version = rubberband_version(executable)
        major = _rubberband_major(version)
        if args.rubberband_engine == "r3" and (major is None or major < 3):
            raise SystemExit(
                f"[error] final-quality R3 needs Rubber Band >=3; found {version!r}"
            )
        print(f"[rubberband] version={version!r}, engine={args.rubberband_engine}")

    rows_out = {"train": [], "val": []}
    d_dtw, d_uni = [], []
    rubberband_length_deltas: list[int] = []
    local_stretch_ratios: list[float] = []
    removed_anchor_counts: list[int] = []
    alignment_qc_rows: list[dict] = []
    skipped_pathological = 0
    skipped_anchor_repairs = 0
    reference_pools: dict[str, list[Path]] = {}
    total = sum(len(rows) for rows in splits.values())
    completed = 0
    for split, rows in splits.items():
        for r in rows:
            src_path = Path(r["source_wav_path"])
            if not src_path.is_absolute():
                src_path = source_root / src_path
            src = read_wav(str(src_path))
            # Re-derive the raw L2 target; never warp the already stretched B1 file.
            raw_tgt_path = raw_target_path(r, l2_root)
            tgt_raw = read_wav(str(raw_tgt_path))
            path = dtw_path(mfcc_cmn(src), mfcc_cmn(tgt_raw))
            alignment_positions = None
            if args.warp_method == "latent":
                try:
                    time_map = _rubberband_time_map(
                        path, source_len=len(src), target_len=len(tgt_raw),
                        anchor_hop_frames=args.anchor_hop_frames,
                    )
                    original_anchor_count = len(time_map)
                    time_map, anchors_removed = _stabilize_time_map(
                        time_map,
                        min_ratio=args.min_local_stretch,
                        max_ratio=args.max_local_stretch,
                        target_len=len(tgt_raw),
                    )
                    pair_ratios = _local_stretch_ratios(time_map)
                    alignment_positions = _latent_alignment_positions(
                        time_map, source_len=len(src), target_len=len(tgt_raw)
                    )
                except ValueError as exc:
                    skipped_pathological += 1
                    print(f"  [skip] {r['source_utt']}: {exc}")
                    continue
                removal_fraction = anchors_removed / original_anchor_count
                if removal_fraction > args.max_anchor_removal_fraction:
                    skipped_anchor_repairs += 1
                    print(
                        f"  [skip] {r['source_utt']}: removed "
                        f"{removal_fraction:.1%} of DTW anchors"
                    )
                    continue
                local_stretch_ratios.extend(pair_ratios.tolist())
                removed_anchor_counts.append(anchors_removed)
                length_delta = 0
                # Diagnostic only. Neither waveform written to the latent
                # dataset is time-warped.
                diagnostic_warp = warp_target_to_source(src, tgt_raw, path)
                train_src, train_tgt = src, tgt_raw
                uniform = uniform_resample_len(tgt_raw, len(src))
                pair_dtw_distance = mel_dist(src, diagnostic_warp)
                pair_resample_distance = mel_dist(src, uniform)
            elif args.warp_method == "rubberband":
                try:
                    (warped, length_delta, pair_ratios, anchors_removed,
                     original_anchor_count) = rubberband_warp_target_to_source(
                        src, tgt_raw, path,
                        anchor_hop_frames=args.anchor_hop_frames,
                        engine=args.rubberband_engine,
                        min_local_stretch=args.min_local_stretch,
                        max_local_stretch=args.max_local_stretch,
                        warp_side=args.warp_side,
                    )
                except ValueError as exc:
                    skipped_pathological += 1
                    print(f"  [skip] {r['source_utt']}: {exc}")
                    continue
                removal_fraction = anchors_removed / original_anchor_count
                if removal_fraction > args.max_anchor_removal_fraction:
                    skipped_anchor_repairs += 1
                    print(
                        f"  [skip] {r['source_utt']}: removed "
                        f"{removal_fraction:.1%} of DTW anchors"
                    )
                    continue
                rubberband_length_deltas.append(length_delta)
                local_stretch_ratios.extend(pair_ratios.tolist())
                removed_anchor_counts.append(anchors_removed)
            else:
                warped = warp_target_to_source(src, tgt_raw, path)
                length_delta = 0
                pair_ratios = np.asarray([], dtype=np.float64)
                anchors_removed = 0
                original_anchor_count = 0

            if args.warp_method != "latent":
                if args.warp_side == "source":
                    train_src = warped
                    train_tgt = tgt_raw
                    uniform = uniform_resample_len(src, len(tgt_raw))
                else:
                    train_src = src
                    train_tgt = warped
                    uniform = uniform_resample_len(tgt_raw, len(src))

                # Fitted diagnostic only; independent QC happens in
                # validate_crosspairs.py. Both arguments are on the selected
                # training timeline for either warp direction.
                pair_dtw_distance = mel_dist(train_src, train_tgt)
                pair_resample_distance = mel_dist(uniform, train_tgt)
            d_dtw.append(pair_dtw_distance)
            d_uni.append(pair_resample_distance)

            src_out = out / "wavs" / "src" / src_path.name
            src_out.parent.mkdir(parents=True, exist_ok=True)
            if args.warp_method == "latent":
                shutil.copyfile(src_path, src_out)
            elif args.warp_side == "source":
                write_wav(src_out, train_src)
            elif not src_out.exists():
                # copy2/copy preserves POSIX timestamps, which can fail on a
                # Windows-mounted /mnt/c filesystem under WSL. Only bytes are
                # relevant to the training dataset.
                shutil.copyfile(src_path, src_out)
            tgt_out = out / "wavs" / "tgt" / Path(r["target_wav_path"]).name
            if args.warp_method == "latent":
                tgt_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(raw_tgt_path, tgt_out)
            elif args.warp_side == "source":
                # Do not requantize or otherwise process the supervision
                # waveform. Byte-for-byte identity is checked by the validator.
                tgt_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(raw_tgt_path, tgt_out)
            else:
                write_wav(tgt_out, train_tgt)
            raw_src_out = out / "wavs" / "raw_src" / src_path.name
            raw_src_out.parent.mkdir(parents=True, exist_ok=True)
            if not raw_src_out.exists():
                shutil.copyfile(src_path, raw_src_out)
            raw_tgt_out = (
                out / "wavs" / "raw_tgt"
                / f"{raw_tgt_path.parent.parent.name}_{raw_tgt_path.name}"
            )
            raw_tgt_out.parent.mkdir(parents=True, exist_ok=True)
            if not raw_tgt_out.exists():
                shutil.copyfile(raw_tgt_path, raw_tgt_out)
            reference = clean_reference_path(
                r, l2_root, reference_pools, args.min_duration,
                forbidden_reference_prompts,
            )
            ref_out = out / "wavs" / "ref" / f"{reference.parent.parent.name}_{reference.name}"
            ref_out.parent.mkdir(parents=True, exist_ok=True)
            if not ref_out.exists():
                shutil.copyfile(reference, ref_out)
            rr = dict(r)
            rr["source_wav_path"] = str(src_out).replace("\\", "/")
            rr["target_wav_path"] = str(tgt_out).replace("\\", "/")
            rr["raw_source_wav_path"] = str(raw_src_out).replace("\\", "/")
            rr["raw_target_wav_path"] = str(raw_tgt_out).replace("\\", "/")
            rr["target_reference_wav_path"] = str(ref_out).replace("\\", "/")
            if alignment_positions is not None:
                alignment_out = out / "alignment" / f"{r['source_utt']}.npy"
                alignment_out.parent.mkdir(parents=True, exist_ok=True)
                np.save(alignment_out, alignment_positions, allow_pickle=False)
                rr["latent_alignment_path"] = str(alignment_out).replace("\\", "/")
            rr["raw_source_duration"] = round(len(src) / SR, 6)
            rr["raw_target_duration"] = round(len(tgt_raw) / SR, 6)
            rows_out[split].append(rr)
            alignment_qc_rows.append({
                "split": split,
                "source_utt": r["source_utt"],
                "target_key": target_key(r),
                "prompt_id": prompt_id(r),
                "warp_side": (
                    "latent_features" if args.warp_method == "latent"
                    else args.warp_side
                ),
                "raw_source_duration": round(len(src) / SR, 6),
                "raw_target_duration": round(len(tgt_raw) / SR, 6),
                "global_stretch_ratio": round(len(src) / len(tgt_raw), 6),
                "anchors_initial": original_anchor_count,
                "anchors_removed": anchors_removed,
                "anchor_removal_fraction": (
                    round(anchors_removed / original_anchor_count, 6)
                    if original_anchor_count else 0.0
                ),
                "local_stretch_min": (
                    round(float(np.min(pair_ratios)), 6)
                    if len(pair_ratios) else None
                ),
                "local_stretch_max": (
                    round(float(np.max(pair_ratios)), 6)
                    if len(pair_ratios) else None
                ),
                "rubberband_length_delta_samples": length_delta,
                "diagnostic_mel_dist_dtw": round(pair_dtw_distance, 6),
                "diagnostic_mel_dist_resample": round(pair_resample_distance, 6),
                "diagnostic_mel_improvement": round(
                    pair_resample_distance - pair_dtw_distance, 6
                ),
            })

            completed += 1
            if completed % 200 == 0:
                print(f"  {completed}/{total}  diagnostic mel-dist "
                      f"dtw {np.mean(d_dtw):.2f} vs resample {np.mean(d_uni):.2f}")

    if not rows_out["train"] or not rows_out["val"]:
        raise RuntimeError(
            "alignment produced an empty train or validation split; inspect "
            "duration and local-stretch filtering"
        )

    mdir = out / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    for name, rows in rows_out.items():
        with open(mdir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    with open(out / "alignment_qc.jsonl", "w", encoding="utf-8") as f:
        for row in alignment_qc_rows:
            f.write(json.dumps(row) + "\n")
    if args.prompts_file:
        shutil.copyfile(args.prompts_file, out / "PROMPTS")

    meta = {
        "pairs": sum(len(rows) for rows in rows_out.values()),
        "train_pairs": len(rows_out["train"]),
        "val_pairs": len(rows_out["val"]),
        "train_prompts": len({prompt_id(r) for r in rows_out["train"]}),
        "val_prompts": len({prompt_id(r) for r in rows_out["val"]}),
        "prompt_overlap": 0,
        "minimum_raw_duration_seconds": args.min_duration,
        "allowed_global_stretch": [args.min_global_stretch, args.max_global_stretch],
        "skipped_short_pairs": skipped_duration,
        "skipped_global_stretch_pairs": skipped_global_stretch,
        "skipped_manual_exclusions": excluded_manual,
        "skipped_pathological_dtw_pairs": skipped_pathological,
        "skipped_excessive_anchor_repairs": skipped_anchor_repairs,
        "max_anchor_removal_fraction": args.max_anchor_removal_fraction,
        "prompts_file": "PROMPTS" if args.prompts_file else None,
        "source_speakers": {
            split: {
                speaker: sum(source_speaker(row) == speaker for row in rows)
                for speaker in sorted({source_speaker(row) for row in rows})
            }
            for split, rows in rows_out.items()
        },
        "warp_method": args.warp_method,
        "warp_side": (
            "latent_features" if args.warp_method == "latent" else args.warp_side
        ),
        "anchor_hop_frames": (
            args.anchor_hop_frames
            if args.warp_method in {"rubberband", "latent"} else None
        ),
        "rubberband_version": (
            rubberband_version() if args.warp_method == "rubberband" else None
        ),
        "rubberband_engine": (
            args.rubberband_engine if args.warp_method == "rubberband" else None
        ),
        "local_stretch_ratio": ({
            "min": round(float(np.min(local_stretch_ratios)), 4),
            "p01": round(float(np.percentile(local_stretch_ratios, 1)), 4),
            "median": round(float(np.median(local_stretch_ratios)), 4),
            "p99": round(float(np.percentile(local_stretch_ratios, 99)), 4),
            "max": round(float(np.max(local_stretch_ratios)), 4),
            "allowed_min": args.min_local_stretch,
            "allowed_max": args.max_local_stretch,
        } if local_stretch_ratios else None),
        "dtw_anchors_removed": ({
            "mean": round(float(np.mean(removed_anchor_counts)), 2),
            "max": max(removed_anchor_counts),
            "pairs_with_repairs": sum(value > 0 for value in removed_anchor_counts),
        } if removed_anchor_counts else None),
        "rubberband_length_delta_samples_mean": (
            round(float(np.mean(rubberband_length_deltas)), 1)
            if rubberband_length_deltas else None
        ),
        "rubberband_length_delta_samples_abs_max": (
            max(map(abs, rubberband_length_deltas))
            if rubberband_length_deltas else None
        ),
        "diagnostic_mel_dist_dtw_mean": round(float(np.mean(d_dtw)), 3),
        "diagnostic_mel_dist_resample_mean": round(float(np.mean(d_uni)), 3),
        "diagnostic_improvement": round(float(np.mean(d_uni) - np.mean(d_dtw)), 3),
    }
    with open(out / "align_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nFitted diagnostic mel-dist: DTW {meta['diagnostic_mel_dist_dtw_mean']} "
          f"vs resample {meta['diagnostic_mel_dist_resample_mean']} "
          f"(improvement {meta['diagnostic_improvement']}; not an independent QC metric)")
    print(f"manifests + align_meta.json under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
