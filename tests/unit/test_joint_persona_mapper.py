import unittest

import torch

from models.joint_accent_mapper import JointAccentMapper, PostPrenetAccentMapper
from models.pronunciation_editor import CausalPronunciationEditor
from scripts.train_joint_persona_mapper import context_valid_items
from xvc.runtime.persona_editor import PersonaStreamEditor
from xvc.training.monotonic import (
    cosine_cost,
    phonewise_aligned_code_agreement,
    phone_duration_class_targets,
    phonewise_discrete_code_loss,
    phonewise_dual_stream_loss,
    phone_sequence_single_stream_loss,
    phonewise_target_code_margin_loss,
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

    def test_stream_window_rejects_hidden_latency_and_excess_history(self):
        safe = JointAccentMapper(
            semantic_dim=8, acoustic_dim=8, code_dim=2, hidden=16,
            layers=5, lookahead_frames=4,
        )
        safe.validate_stream_window(history_ms=2160, smooth_ms=20, future_ms=100)
        with self.assertRaises(ValueError):
            safe.validate_stream_window(history_ms=2160, smooth_ms=0, future_ms=40)
        too_deep = JointAccentMapper(
            semantic_dim=8, acoustic_dim=8, code_dim=2, hidden=16,
            layers=6, lookahead_frames=0,
        )
        with self.assertRaises(ValueError):
            too_deep.validate_stream_window(history_ms=2160, smooth_ms=20, future_ms=100)

    def test_post_prenet_mapper_starts_as_exact_identity(self):
        mapper = PostPrenetAccentMapper(
            input_dim=12, hidden=16, layers=2, lookahead_frames=1,
            input_dropout=0.0,
        ).eval()
        hidden = torch.randn(2, 12, 9)
        torch.testing.assert_close(mapper(hidden), hidden)

    def test_pronunciation_editor_starts_as_identity_and_emits_duration_logits(self):
        editor = CausalPronunciationEditor(
            input_dim=12, hidden=16, layers=1, lookahead_frames=2,
            required_history_frames=4, input_dropout=0.0,
        ).eval()
        hidden = torch.randn(2, 12, 20)
        edited, delta, actions = editor(hidden, return_aux=True)
        torch.testing.assert_close(edited, hidden)
        torch.testing.assert_close(delta, torch.zeros_like(delta))
        self.assertEqual(actions.shape, (2, 3, 20))
        self.assertTrue(torch.all(actions.argmax(dim=1) == 1))
        editor.validate_stream_window(history_ms=80, smooth_ms=20, future_ms=20)
        with self.assertRaises(ValueError):
            editor.validate_stream_window(history_ms=80, smooth_ms=0, future_ms=20)

    def test_context_filter_uses_all_causally_valid_window_frames(self):
        item = {
            "source_semantic": torch.randn(4, 20),
            "source_zq": torch.randn(4, 20),
            "source_codes": torch.arange(20),
            "target_semantic": torch.randn(4, 20),
            "target_zq": torch.randn(4, 20),
            "target_codes": torch.arange(20),
            "phone_segments": [
                {"src": [6, 12], "tgt": [2, 8], "confidence": 1.0},
                {"src": [15, 20], "tgt": [10, 15], "confidence": 1.0},
            ],
            "meta": {
                "window_id": "example__w00",
                "source_encoded_from_window": True,
                "target_encoded_from_window": True,
            },
        }
        prepared = context_valid_items(
            [item], history_frames=8, lookahead_frames=2,
            expected_window_frames=20, min_phone_segments=2,
        )[0]
        self.assertEqual(prepared["source_semantic"].shape[-1], 20)
        self.assertEqual(
            [segment["src"] for segment in prepared["phone_segments"]],
            [[8, 12], [15, 18]],
        )
        self.assertEqual(
            [segment["tgt"] for segment in prepared["phone_segments"]],
            [[4, 8], [10, 13]],
        )
        self.assertEqual(int(prepared["source_valid_mask"].sum()), 10)
        self.assertEqual(int(prepared["target_valid_mask"].sum()), 10)

    def test_context_filter_rejects_old_full_utterance_feature_cache(self):
        item = {
            "source_semantic": torch.randn(4, 20),
            "source_zq": torch.randn(4, 20),
            "source_codes": torch.arange(20),
            "target_semantic": torch.randn(4, 20),
            "target_zq": torch.randn(4, 20),
            "target_codes": torch.arange(20),
            "phone_segments": [{"src": [8, 12], "tgt": [8, 12]}],
            "meta": {},
        }
        with self.assertRaisesRegex(ValueError, "predates raw 2.4 s"):
            context_valid_items(
                [item], history_frames=8, lookahead_frames=2,
                expected_window_frames=20,
            )

    def test_runtime_editor_uses_channel_first_native_quantizer_geometry(self):
        class FakeQuantizer:
            def __init__(self):
                self.weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

            def embed_code(self, indices):
                return self.weight[indices]

            def decode_latents(self, latents):
                self.seen_shape = tuple(latents.shape)
                indices = latents.argmax(dim=1)
                return latents, indices, torch.zeros(1)

            def vq2emb(self, indices, out_proj=True):
                return self.weight[indices].transpose(1, 2).repeat(1, 5, 1)

        mapper = JointAccentMapper(
            semantic_dim=12, acoustic_dim=10, code_dim=2, hidden=16,
            layers=1, lookahead_frames=0, input_dropout=0.0,
        ).eval()
        quantizer = FakeQuantizer()
        editor = PersonaStreamEditor(mapper, "pre_prenet", {})
        indices = torch.tensor([[0, 1, 0]])
        zq = quantizer.vq2emb(indices)
        semantic = torch.randn(1, 3, 12)
        edited_semantic, edited_zq = editor.edit_pre_prenet(
            semantic, zq, indices, quantizer
        )
        self.assertEqual(quantizer.seen_shape, (1, 2, 3))
        torch.testing.assert_close(edited_semantic, semantic)
        torch.testing.assert_close(edited_zq, zq)

    def test_chunk_output_matches_full_context_inside_declared_receptive_field(self):
        torch.manual_seed(7)
        mapper = JointAccentMapper(
            semantic_dim=8, acoustic_dim=8, code_dim=2, hidden=16,
            layers=2, kernel_size=3, lookahead_frames=2, input_dropout=0.0,
        ).eval()
        torch.nn.init.normal_(mapper.semantic_delta.weight, std=0.02)
        torch.nn.init.normal_(mapper.code_delta.weight, std=0.02)
        semantic = torch.randn(1, 8, 200)
        acoustic = torch.randn(1, 8, 200)
        code = torch.randn(1, 2, 200)
        full_sem, full_code = mapper(semantic, acoustic, code)
        start, end = 50, 170
        chunk_sem, chunk_code = mapper(
            semantic[:, :, start:end],
            acoustic[:, :, start:end],
            code[:, :, start:end],
        )
        left = mapper.receptive_history_frames
        right = mapper.lookahead_frames
        torch.testing.assert_close(
            chunk_sem[:, :, left:-right], full_sem[:, :, start + left:end - right]
        )
        torch.testing.assert_close(
            chunk_code[:, :, left:-right], full_code[:, :, start + left:end - right]
        )

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

    def test_sequence_loss_concatenates_ordered_phone_interiors(self):
        predicted = torch.randn(1, 4, 10, requires_grad=True)
        target = torch.randn(1, 4, 12)
        segments = [[
            {"src": [1, 4], "tgt": [2, 6], "confidence": 0.9},
            {"src": [6, 9], "tgt": [8, 11], "confidence": 1.0},
        ]]
        loss, items = phone_sequence_single_stream_loss(
            predicted, target, segments
        )
        loss.backward()
        self.assertEqual(items, 1)
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_duration_targets_encode_phone_length_ratio(self):
        segments = [[
            {"src": [1, 3], "tgt": [1, 2], "duration_ratio": 0.5},
            {"src": [4, 6], "tgt": [4, 6], "duration_ratio": 1.0},
            {"src": [7, 9], "tgt": [7, 11], "duration_ratio": 2.0},
        ]]
        labels, weights = phone_duration_class_targets(
            segments, 12, torch.device("cpu")
        )
        self.assertEqual(labels[0, 1:3].tolist(), [2, 2])
        self.assertEqual(labels[0, 4:6].tolist(), [1, 1])
        self.assertEqual(labels[0, 7:9].tolist(), [0, 0])
        self.assertEqual(int(torch.count_nonzero(weights)), 6)

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

    def test_target_margin_prefers_aligned_target_over_native_code(self):
        source_indices = torch.tensor([[0, 0]])
        target_indices = torch.tensor([[1, 1]])
        codebook = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        source_code = codebook[source_indices].transpose(1, 2)
        target_code = codebook[target_indices].transpose(1, 2)
        segments = [[{"src": [0, 2], "tgt": [0, 2], "confidence": 1.0}]]
        good = torch.tensor([[[0.0, 3.0], [0.0, 3.0]]], requires_grad=True)
        bad = torch.tensor([[[3.0, 0.0], [3.0, 0.0]]], requires_grad=True)
        good_loss, positions = phonewise_target_code_margin_loss(
            good, source_indices, target_indices, source_code, target_code, segments
        )
        bad_loss, _ = phonewise_target_code_margin_loss(
            bad, source_indices, target_indices, source_code, target_code, segments
        )
        self.assertEqual(positions, 2)
        self.assertLess(float(good_loss), float(bad_loss))
        bad_loss.backward()
        self.assertTrue(torch.isfinite(bad.grad).all())


if __name__ == "__main__":
    unittest.main()
