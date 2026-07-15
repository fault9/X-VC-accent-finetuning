import unittest

import torch

from models.joint_accent_mapper import JointAccentMapper
from xvc.training.monotonic import (
    cosine_cost,
    phonewise_aligned_code_agreement,
    phonewise_discrete_code_loss,
    phonewise_dual_stream_loss,
    soft_dtw,
)


class JointAccentMapperTest(unittest.TestCase):
    def test_zero_initialization_is_exact_dual_stream_identity(self):
        mapper = JointAccentMapper(
            semantic_dim=12,
            acoustic_dim=10,
            code_dim=4,
            hidden=16,
            layers=2,
            lookahead_frames=1,
            input_dropout=0.0,
        ).eval()
        semantic = torch.randn(2, 12, 9)
        acoustic = torch.randn(2, 10, 9)
        code = torch.randn(2, 4, 9)
        edited_semantic, edited_code = mapper(semantic, acoustic, code)
        torch.testing.assert_close(edited_semantic, semantic)
        torch.testing.assert_close(edited_code, code)

    def test_model_has_no_speaker_conditioning_argument_or_parameter(self):
        mapper = JointAccentMapper(
            semantic_dim=8,
            acoustic_dim=8,
            code_dim=2,
            hidden=16,
            layers=1,
        )
        self.assertFalse(any("speaker" in name for name, _ in mapper.named_parameters()))

    def test_nearest_code_uses_frozen_codebook_geometry(self):
        codebook = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        query = torch.tensor([[[0.9, 0.0], [0.1, 1.0]]])
        indices = JointAccentMapper.nearest_codes(query, codebook)
        self.assertEqual(indices.tolist(), [[0, 1]])


class MonotonicLossTest(unittest.TestCase):
    def test_identical_sequences_have_lower_cost(self):
        source = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        identical = source.clone()
        reversed_target = source.flip(-1)
        self.assertLess(
            float(soft_dtw(cosine_cost(source, identical))),
            float(soft_dtw(cosine_cost(source, reversed_target))),
        )

    def test_dual_loss_backpropagates_without_resampling(self):
        predicted_semantic = torch.randn(1, 6, 7, requires_grad=True)
        predicted_code = torch.randn(1, 3, 7, requires_grad=True)
        target_semantic = torch.randn(1, 6, 9)
        target_code = torch.randn(1, 3, 9)
        segments = [[{"src": [1, 6], "tgt": [2, 8], "confidence": 0.9}]]
        semantic_loss, code_loss, phones = phonewise_dual_stream_loss(
            predicted_semantic,
            predicted_code,
            target_semantic,
            target_code,
            segments,
        )
        (semantic_loss + code_loss).backward()
        self.assertEqual(phones, 1)
        self.assertTrue(torch.isfinite(predicted_semantic.grad).all())
        self.assertTrue(torch.isfinite(predicted_code.grad).all())

    def test_discrete_code_loss_prefers_real_target_ids_and_backpropagates(self):
        segments = [[{"src": [0, 2], "tgt": [0, 2], "confidence": 1.0}]]
        target_indices = torch.tensor([[1, 1]])
        good_logits = torch.tensor(
            [[[0.0, 5.0, -1.0], [0.0, 5.0, -1.0]]], requires_grad=True
        )
        bad_logits = torch.tensor(
            [[[5.0, 0.0, -1.0], [5.0, 0.0, -1.0]]], requires_grad=True
        )
        good_loss, good_phones = phonewise_discrete_code_loss(
            good_logits, target_indices, segments
        )
        bad_loss, bad_phones = phonewise_discrete_code_loss(
            bad_logits, target_indices, segments
        )
        self.assertEqual(good_phones, 1)
        self.assertEqual(bad_phones, 1)
        self.assertLess(float(good_loss), float(bad_loss))
        good_loss.backward()
        self.assertTrue(torch.isfinite(good_logits.grad).all())

    def test_aligned_agreement_rewards_target_directed_not_arbitrary_changes(self):
        codebook = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        source_indices = torch.tensor([[0, 0, 0]])
        target_indices = torch.tensor([[1, 1, 1]])
        predicted_indices = torch.tensor([[1, 1, 1]])
        source_code = codebook[source_indices].transpose(1, 2)
        target_code = codebook[target_indices].transpose(1, 2)
        segments = [[{"src": [0, 3], "tgt": [0, 3], "confidence": 1.0}]]
        metrics = phonewise_aligned_code_agreement(
            predicted_indices,
            source_indices,
            target_indices,
            source_code,
            target_code,
            segments,
        )
        self.assertEqual(metrics["source"], 0.0)
        self.assertEqual(metrics["predicted"], 1.0)
        self.assertEqual(metrics["gain"], 1.0)
        arbitrary = phonewise_aligned_code_agreement(
            torch.tensor([[2, 2, 2]]),
            source_indices,
            target_indices,
            source_code,
            target_code,
            segments,
        )
        self.assertEqual(arbitrary["predicted"], 0.0)
        self.assertEqual(arbitrary["gain"], 0.0)


if __name__ == "__main__":
    unittest.main()
