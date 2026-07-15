import unittest

import numpy as np

from scripts.extract_joint_persona_pairs import _crop_audio, _window_phone_segments


class JointPersonaWindowExtractionTest(unittest.TestCase):
    def test_waveform_crop_has_exact_length_and_edge_padding(self):
        audio = np.arange(8, dtype=np.float32)
        np.testing.assert_array_equal(
            _crop_audio(audio, -2, 6),
            np.array([0, 0, 0, 1, 2, 3], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            _crop_audio(audio, 5, 6),
            np.array([5, 6, 7, 0, 0, 0], dtype=np.float32),
        )

    def test_phone_annotations_keep_independent_source_and_target_timelines(self):
        matches = [
            {
                "phone": "aa",
                "src_seconds": [1.2, 1.4],
                "tgt_seconds": [2.1, 2.5],
                "duration_ratio": 0.5,
                "confidence": 0.5,
            }
        ]
        segments = _window_phone_segments(
            matches,
            source_start_seconds=0.8,
            target_start_seconds=1.5,
            window_seconds=2.4,
            source_frames=120,
            target_frames=120,
        )
        self.assertEqual(len(segments), 1)
        self.assertNotEqual(segments[0]["src"], segments[0]["tgt"])
        self.assertEqual(segments[0]["phone"], "aa")


if __name__ == "__main__":
    unittest.main()
