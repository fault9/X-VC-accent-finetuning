import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from phone_supervision import align_phone_sequences, phone_tier, read_textgrid
from annotate_accentbridge_phone_supervision import _resolve_textgrid, _trim_range


TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "test"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.2
            text = "sil"
        intervals [2]:
            xmin = 0.2
            xmax = 0.6
            text = "EH1"
        intervals [3]:
            xmin = 0.6
            xmax = 1
            text = "S"
'''


class PhoneSupervisionTest(unittest.TestCase):
    def test_real_phone_tier_is_selected_and_stress_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.TextGrid"
            path.write_text(TEXTGRID, encoding="utf-8")
            tiers = read_textgrid(path)
            self.assertEqual(set(tiers), {"words", "phones"})
            name, phones, duration = phone_tier(path)
            self.assertEqual(name, "phones")
            self.assertEqual([p.label for p in phones], ["EH1", "S"])
            self.assertEqual(duration, 1.0)
            pairs, rate = align_phone_sequences(phones, phones)
            self.assertEqual([p[2] for p in pairs], ["eh", "s"])
            self.assertEqual(rate, 1.0)

    def test_word_only_grid_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.TextGrid"
            path.write_text(TEXTGRID.split("    item [2]:")[0], encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no genuine phone tier"):
                phone_tier(path)

    def test_adaptive_trim_preserves_short_phone(self):
        self.assertEqual(_trim_range(10, 12, 0.15, 1, 2), (10, 12))
        self.assertEqual(_trim_range(10, 18, 0.15, 1, 2), (11, 17))
        self.assertIsNone(_trim_range(10, 11, 0.15, 1, 2))

    def test_target_textgrid_resolves_from_speaker_and_prompt(self):
        expected = Path("ASI_arctic_a0042.TextGrid")
        index = {"ASI_arctic_a0042": expected}
        meta = {"target_wav_path": "derived/ASI__bdl_a0042_ft.wav",
                "target_utt": "ASI__bdl_a0042", "target_speaker": "ASI",
                "prompt": "a0042"}
        self.assertEqual(_resolve_textgrid(index, meta, "target"), expected)


if __name__ == "__main__":
    unittest.main()
