import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_phoneaware_mfa_corpus import main


def write_wav(path: Path, seconds: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * int(16000 * seconds))


class PreparePhoneawareMfaCorpusTest(unittest.TestCase):
    def test_selects_exact_speaker_and_copies_only_pristine_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw_src.wav"
            asi = root / "raw_asi.wav"
            tni = root / "raw_tni.wav"
            for path in (source, asi, tni):
                write_wav(path)

            rows = [
                {
                    "source_utt": "bdl_arctic_a0001",
                    "target_utt": "ASI__bdl_arctic_a0001_ft",
                    "raw_source_wav_path": str(source),
                    "raw_target_wav_path": str(asi),
                },
                {
                    "source_utt": "bdl_arctic_a0002",
                    "target_utt": "TNI__bdl_arctic_a0002_ft",
                    "raw_source_wav_path": str(source),
                    "raw_target_wav_path": str(tni),
                },
            ]
            train = root / "train.jsonl"
            train.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            val = root / "val.jsonl"
            val.write_text("", encoding="utf-8")
            prompts = root / "PROMPTS"
            prompts.write_text(
                "arctic_a0001 This is one.\n"
                "arctic_a0002 This is two.\n",
                encoding="utf-8",
            )
            output = root / "out"
            argv = [
                "prepare_phoneaware_mfa_corpus.py",
                "--train-manifest",
                str(train),
                "--val-manifest",
                str(val),
                "--prompts-file",
                str(prompts),
                "--target-speaker",
                "ASI",
                "--expected-train",
                "1",
                "--expected-val",
                "0",
                "--out",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(main(), 0)

            self.assertTrue(
                (output / "mfa_corpus/source/bdl/bdl_arctic_a0001.wav").is_file()
            )
            self.assertTrue(
                (
                    output
                    / "mfa_corpus/target/ASI/ASI__bdl_arctic_a0001.wav"
                ).is_file()
            )
            self.assertFalse(
                (
                    output
                    / "mfa_corpus/target/TNI/TNI__bdl_arctic_a0002.wav"
                ).exists()
            )
            meta = json.loads((output / "mfa_prepare_meta.json").read_text())
            self.assertEqual(meta["train_pairs"], 1)
            self.assertEqual(meta["target_speaker"], "ASI")


if __name__ == "__main__":
    unittest.main()
