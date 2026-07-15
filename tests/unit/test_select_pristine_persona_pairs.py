import json
import tempfile
import unittest
import wave
from pathlib import Path

from scripts.select_pristine_persona_pairs import main


def write_wav(path: Path, seconds: float = 3.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * int(16000 * seconds))


class SelectPristinePersonaPairsTest(unittest.TestCase):
    def test_uses_each_target_once_and_balances_source_speakers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            speakers = ("bdl", "rms", "clb", "slt")
            prompts = [f"arctic_a{index:04d}" for index in range(1, 6)]
            for prompt in prompts:
                write_wav(root / "target" / f"ASI_{prompt}.wav")
                for speaker in speakers:
                    write_wav(root / speaker / f"{speaker}_{prompt}.wav")
            output = root / "selection"
            arguments = []
            for speaker in speakers:
                arguments.extend(["--source", f"{speaker}={root / speaker / '*.wav'}"])
            result = main(arguments + [
                "--target-glob", str(root / "target" / "*.wav"),
                "--out", str(output),
                "--val-prompts", "1",
                "--min-train-pairs", "4",
                "--min-unique-target-minutes", "0.19",
            ])
            self.assertEqual(result, 0)
            train = [
                json.loads(line)
                for line in (output / "train.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(train), 4)
            self.assertEqual(len({row["raw_target_wav_path"] for row in train}), 4)
            self.assertEqual({row["source_speaker"] for row in train}, set(speakers))


if __name__ == "__main__":
    unittest.main()
