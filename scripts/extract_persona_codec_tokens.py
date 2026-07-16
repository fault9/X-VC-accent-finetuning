#!/usr/bin/env python
"""Phase 0: extract discrete X-VC token sequences for pristine parallel pairs.

For every native/ASI pair in the canonical pristine dataset this encodes the
two *unmodified* waveforms independently through the frozen X-VC encoders and
stores, per pair: source/target WhisperVQ semantic token IDs (12.5 Hz),
source/target acoustic-code IDs (50 Hz), raw stream lengths, and prompt /
speaker metadata.  No waveform or feature is warped, resampled, or cropped —
`process_audio` only zero-pads each file to the 1280-sample token grid, the
same padding stock inference applies.

The output shards feed both the geometry report
(`scripts/report_persona_token_geometry.py`) and the offline teacher trainer
(`scripts/train_persona_codec_teacher.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/hindi_asi_pristine_parallel_221")
    parser.add_argument("--config", default="configs/xvc.yaml")
    parser.add_argument("--ckpt", default="ckpts/xvc.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all pairs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    import numpy as np
    import torch

    from bins.infer_utils import load_xvc
    from models.codec.sac.utils import process_audio
    from xvc.persona_codec.runtime import (
        adapter_upsample_ratio,
        extract_source_token_streams,
        file_sha256,
        live_token_geometry,
        tensor_sha256,
        acoustic_codebook,
        semantic_codebook,
    )
    from xvc.persona_codec.tokens import (
        EXPECTED_GROUP_SIZE,
        assert_prompt_disjoint,
        trim_to_group_ratio,
        validate_token_ids,
    )

    data_root = Path(args.data_root)
    output = Path(args.out)
    if output.exists() and any(output.glob("*.pt")) and not args.overwrite:
        raise SystemExit(f"[error] output already contains shards: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for split in ("train", "val"):
        path = data_root / "manifests" / f"{split}.jsonl"
        if not path.is_file():
            raise SystemExit(f"[error] missing manifest: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.max_pairs:
            rows = rows[: args.max_pairs]
        if not rows:
            raise SystemExit(
                f"[error] {split} manifest has zero rows: {path}; the teacher "
                f"trainer needs a non-empty prompt-disjoint {split} split"
            )
        manifests[split] = rows
    assert_prompt_disjoint(manifests["train"], manifests["val"])

    # The persona is fixed per checkpoint: every pair in every split must
    # share one target speaker, or the shards (and everything downstream)
    # would silently mislabel a mixed-persona experiment.
    personas = {
        str(row["target_speaker"]) for rows in manifests.values() for row in rows
    }
    if len(personas) != 1:
        raise SystemExit(
            f"[error] manifests mix target speakers {sorted(personas)}; the "
            "persona codec is trained for exactly one target persona"
        )
    target_persona = personas.pop()

    cfg, model, device = load_xvc(args.config, args.ckpt, args.device, False)
    hop = int(cfg["latent_hop_length"])
    geometry = live_token_geometry(model)
    sem_vocab = geometry["semantic_vocab_size"]
    ac_vocab = geometry["acoustic_vocab_size"]
    adapter_group = adapter_upsample_ratio(model)
    if adapter_group != EXPECTED_GROUP_SIZE:
        raise SystemExit(
            f"[error] semantic adapter upsamples x{adapter_group}, expected "
            f"x{EXPECTED_GROUP_SIZE}; the persona codec latency contract "
            "(one ~80 ms step carrying one acoustic group) does not hold"
        )

    @torch.inference_mode()
    def encode(wav_path: str) -> dict:
        resolved = Path(wav_path)
        if not resolved.is_file():
            resolved = _ROOT / wav_path
        wav_np = process_audio(str(resolved), cfg, hop)
        wav = torch.from_numpy(wav_np).view(1, 1, -1).float().to(device)
        streams = extract_source_token_streams(model, wav)
        semantic = streams["semantic_tokens"][0].cpu().numpy().astype(np.int32)
        acoustic = streams["acoustic_indices"][0].cpu().numpy().astype(np.int32)
        validate_token_ids(semantic, sem_vocab, f"{resolved.name}:semantic")
        validate_token_ids(acoustic, ac_vocab, f"{resolved.name}:acoustic")
        return {
            "semantic": semantic,
            "acoustic": acoustic,
            "samples": int(wav.shape[-1]),
        }

    stats_rows = []
    for split, rows in manifests.items():
        items = []
        for index, row in enumerate(rows, start=1):
            source = encode(row["source_wav_path"])
            target = encode(row["target_wav_path"])
            observed_group = round(len(source["acoustic"]) / max(len(source["semantic"]), 1))
            if observed_group != adapter_group:
                raise SystemExit(
                    f"[error] pair {row['pair_id']}: observed acoustic/semantic "
                    f"frame ratio {observed_group} != live adapter upsample "
                    f"ratio {adapter_group}"
                )
            trims = {}
            for name, item in (("source", source), ("target", target)):
                sem_len, ac_len, dropped = trim_to_group_ratio(
                    len(item["semantic"]), len(item["acoustic"]), observed_group
                )
                trims[name] = {
                    "semantic_raw": len(item["semantic"]),
                    "acoustic_raw": len(item["acoustic"]),
                    "semantic": sem_len,
                    "acoustic": ac_len,
                    "dropped_acoustic_frames": dropped,
                }
                item["semantic"] = item["semantic"][:sem_len]
                item["acoustic"] = item["acoustic"][:ac_len]
            items.append(
                {
                    "pair_id": row["pair_id"],
                    "prompt_id": row["prompt_id"],
                    "prompt_text": row.get("prompt_text", ""),
                    "source_speaker": row["source_speaker"],
                    "target_speaker": row["target_speaker"],
                    "src_semantic": torch.from_numpy(source["semantic"]),
                    "src_acoustic": torch.from_numpy(source["acoustic"]),
                    "tgt_semantic": torch.from_numpy(target["semantic"]),
                    "tgt_acoustic": torch.from_numpy(target["acoustic"]),
                    "trims": trims,
                }
            )
            stats_rows.append(
                {
                    "split": split,
                    "pair_id": row["pair_id"],
                    "prompt_id": row["prompt_id"],
                    "source_speaker": row["source_speaker"],
                    "src_semantic_len": trims["source"]["semantic"],
                    "tgt_semantic_len": trims["target"]["semantic"],
                    "src_acoustic_len": trims["source"]["acoustic"],
                    "tgt_acoustic_len": trims["target"]["acoustic"],
                    "src_dropped_acoustic": trims["source"]["dropped_acoustic_frames"],
                    "tgt_dropped_acoustic": trims["target"]["dropped_acoustic_frames"],
                }
            )
            if index % 25 == 0 or index == len(rows):
                print(f"  [{split}] {index}/{len(rows)}", flush=True)
        torch.save(items, output / f"{split}.pt")

    group_size = adapter_group
    sample_rate = int(cfg["sample_rate"])
    meta = {
        "schema_version": 1,
        "purpose": "persona codec token pairs (Phase 0/1)",
        "data_root": str(data_root).replace("\\", "/"),
        "dataset_meta": json.loads((data_root / "dataset_meta.json").read_text(encoding="utf-8")),
        "xvc_config": args.config,
        "xvc_checkpoint": args.ckpt,
        "xvc_checkpoint_sha256": file_sha256(args.ckpt),
        "semantic_vocab_size": sem_vocab,
        "acoustic_vocab_size": ac_vocab,
        "acoustic_group_size": group_size,
        "semantic_token_hz": sample_rate / hop,
        "acoustic_code_hz": group_size * sample_rate / hop,
        "latent_hop_length": hop,
        "semantic_codebook_sha256": tensor_sha256(semantic_codebook(model)),
        "quantizer_codebook_sha256": tensor_sha256(acoustic_codebook(model)),
        "waveform_or_feature_warping": False,
        "target_persona": target_persona,
        # Val speakers count as seen: the val split drives checkpoint
        # selection, so unseen-source evaluation must exclude them too.
        "source_speakers_seen": sorted(
            {
                row["source_speaker"].casefold()
                for rows in manifests.values()
                for row in rows
            }
        ),
        "train_prompt_ids": sorted({row["prompt_id"].casefold() for row in manifests["train"]}),
        "val_prompt_ids": sorted({row["prompt_id"].casefold() for row in manifests["val"]}),
        "n_train": len(manifests["train"]),
        "n_val": len(manifests["val"]),
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output / "geometry_stats.json").write_text(
        json.dumps(stats_rows, indent=2), encoding="utf-8"
    )
    print(
        f"[extract] {meta['n_train']} train / {meta['n_val']} val pairs -> {output} "
        f"(sem vocab {sem_vocab}, ac vocab {ac_vocab}, group {group_size}, "
        f"{meta['semantic_token_hz']:.2f} Hz semantic)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
