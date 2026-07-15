import json
import tempfile
import unittest
import wave
from pathlib import Path

from scripts.check_persona_dataset_scale import main


def write_wav(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * int(16000 * seconds))


class PersonaDatasetScaleTest(unittest.TestCase):
    def test_repeated_pair_does_not_inflate_unique_target_minutes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            manifests = dataset / "manifests"
            manifests.mkdir(parents=True)
            target = root / "ASI_arctic_a0001.wav"
            write_wav(target)
            train = [
                {
                    "prompt_id": "arctic_a0001",
                    "source_speaker": speaker,
                    "target_speaker": "ASI",
                    "target_wav_path": str(target),
                }
                for speaker in ("bdl", "rms")
            ]
            (manifests / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in train) + "\n",
                encoding="utf-8",
            )
            (manifests / "val.jsonl").write_text(
                json.dumps({
                    "prompt_id": "arctic_a0002",
                    "source_speaker": "bdl",
                    "target_speaker": "ASI",
                    "target_wav_path": str(target),
                }) + "\n",
                encoding="utf-8",
            )
            evaluation = root / "eval"
            write_wav(evaluation / "aba_arctic_b0001.wav")
            common = [
                "--dataset-root", str(dataset),
                "--eval-source-dir", str(evaluation),
                "--min-train-pairs", "2",
                "--min-source-speakers", "2",
                "--min-eval-speakers", "1",
            ]
            self.assertEqual(
                main(common + ["--min-unique-target-minutes", "0.02"]), 1
            )
            self.assertEqual(
                main(common + ["--min-unique-target-minutes", "0.01"]), 0
            )


if __name__ == "__main__":
    unittest.main()
