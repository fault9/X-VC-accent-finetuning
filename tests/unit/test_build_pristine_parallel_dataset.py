import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_pristine_parallel_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_pristine_parallel_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_wav(path: Path, seconds: float = 0.1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * int(16000 * seconds))


def write_grid(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('File type = "ooTextFile"\nname = "phones"\n', encoding="utf-8")


class BuildPristineParallelDatasetTest(unittest.TestCase):
    def test_builds_relocatable_dataset_with_phone_grids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            source_id = "bdl_arctic_a0001"
            target_id = "ASI__bdl_arctic_a0001"
            source_root = prepared / "mfa_corpus" / "source" / "bdl"
            target_root = prepared / "mfa_corpus" / "target" / "ASI"
            write_wav(source_root / f"{source_id}.wav")
            write_wav(target_root / f"{target_id}.wav")
            (source_root / f"{source_id}.lab").write_text("hello\n", encoding="utf-8")
            (target_root / f"{target_id}.lab").write_text("hello\n", encoding="utf-8")
            write_grid(prepared / "mfa_align" / "source" / f"{source_id}.TextGrid")
            write_grid(prepared / "mfa_align" / "target" / f"{target_id}.TextGrid")
            selected = prepared / "selected_manifests"
            selected.mkdir(parents=True)
            row = {
                "source_utt": source_id,
                "target_utt": target_id + "_ft",
                "mfa_source_id": source_id,
                "mfa_target_id": target_id,
                "prompt_text": "hello",
            }
            (selected / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (selected / "val.jsonl").write_text("", encoding="utf-8")
            out = root / "out"

            result = MODULE.main(
                [
                    "--prepared-root", str(prepared),
                    "--out", str(out),
                    "--path-prefix", "data/hindi_asi_pristine_parallel_221",
                    "--min-duration", "0.05",
                ]
            )
            self.assertEqual(result, 0)
            manifest = json.loads((out / "manifests" / "train.jsonl").read_text())
            self.assertEqual(
                manifest["source_wav_path"],
                f"data/hindi_asi_pristine_parallel_221/wavs/source/bdl/{source_id}.wav",
            )
            self.assertTrue((out / "alignments" / "target" / "ASI" / f"{target_id}.TextGrid").is_file())
            self.assertTrue((out / "checksums.sha256").is_file())


if __name__ == "__main__":
    unittest.main()
