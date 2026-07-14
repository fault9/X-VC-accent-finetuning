import unittest

import numpy as np

from xvc.data.stream_swap import build_phone_frame_map


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


if __name__ == "__main__":
    unittest.main()
