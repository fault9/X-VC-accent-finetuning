import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from xvc.data.stream_swap import (
    build_phone_frame_map,
    phone_segments_from_textgrids,
    resolve_audio_path,
)


TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 1
tiers? <exists>
size = 1
item []:
    item [1]:
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


class PhoneFrameMapTest(unittest.TestCase):
    def test_identity_intervals_produce_identity_map(self):
        segments = [
            {"src": [0, 5], "tgt": [0, 5]},
            {"src": [5, 10], "tgt": [5, 10]},
        ]
        positions, info = build_phone_frame_map(10, 10, segments)
        np.testing.assert_allclose(positions, np.arange(10), atol=1e-6)
        self.assertEqual(info["matched_phones"], 2)
        self.assertEqual(info["monotonic_anchor_repairs"], 0)

    def test_stretched_target_is_monotonic_and_pinned(self):
        segments = [
            {"src": [1, 4], "tgt": [2, 8]},
            {"src": [5, 9], "tgt": [10, 18]},
        ]
        positions, _ = build_phone_frame_map(10, 20, segments)
        self.assertAlmostEqual(float(positions[0]), 0.0)
        self.assertAlmostEqual(float(positions[-1]), 19.0)
        self.assertTrue(np.all(np.diff(positions) >= 0))

    def test_phone_at_zero_cannot_displace_endpoint_pin(self):
        segments = [{"src": [0, 5], "tgt": [2, 7]}]
        positions, _ = build_phone_frame_map(10, 12, segments)
        self.assertEqual(float(positions[0]), 0.0)
        self.assertEqual(float(positions[-1]), 11.0)

    def test_annotation_lengths_are_rescaled(self):
        segments = [{"src": [0, 100], "tgt": [0, 200]}]
        positions, _ = build_phone_frame_map(
            50,
            80,
            segments,
            source_annotation_frames=100,
            target_annotation_frames=200,
        )
        self.assertEqual(positions.shape, (50,))
        self.assertAlmostEqual(float(positions[-1]), 79.0)

    def test_empty_segments_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_phone_frame_map(10, 10, [])


class AudioPathResolutionTest(unittest.TestCase):
    def test_resolves_unique_copy_by_exact_basename(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            copy = root / "mfa" / "speaker" / "clip.wav"
            copy.parent.mkdir(parents=True)
            copy.write_bytes(b"RIFF-test")
            resolved, remapped = resolve_audio_path("deleted/clip.wav", [root])
            self.assertEqual(resolved, copy.resolve())
            self.assertTrue(remapped)

    def test_conflicting_duplicate_names_fail_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a" / "clip.wav"
            second = root / "b" / "clip.wav"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                resolve_audio_path("deleted/clip.wav", [root])


class TextGridPhoneSegmentTest(unittest.TestCase):
    def test_derives_segments_without_warping(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.TextGrid"
            target = root / "target.TextGrid"
            source.write_text(TEXTGRID, encoding="utf-8")
            target.write_text(TEXTGRID, encoding="utf-8")
            segments, metadata = phone_segments_from_textgrids(
                source,
                target,
                50,
                60,
                min_matched_phones=2,
            )
            self.assertEqual([segment["phone"] for segment in segments], ["eh", "s"])
            self.assertEqual(metadata["label_match_rate"], 1.0)
            self.assertFalse(metadata["source_was_warped"])
            self.assertTrue(all(segment["src"][1] > segment["src"][0] for segment in segments))
            self.assertTrue(all(segment["tgt"][1] > segment["tgt"][0] for segment in segments))


if __name__ == "__main__":
    unittest.main()
