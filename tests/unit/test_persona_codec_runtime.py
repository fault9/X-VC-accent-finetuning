"""CPU tests for fail-closed persona codec checkpoint loading and the
identity/bypass guarantee of the token re-embedding path (tiny mock X-VC)."""

import pytest
import torch
import torch.nn as nn

fvq_module = pytest.importorskip(
    "models.codec.base.quantizer.factorized_vector_quantize",
    reason="needs einops (pip install -r requirements.txt)",
)

from models.persona_codec_generator import PersonaCodecTeacher
from xvc.persona_codec.runtime import (
    adapter_upsample_ratio,
    build_generator_payload,
    embed_token_streams,
    geometry_from_extraction_meta,
    geometry_from_xvc,
    live_token_geometry,
    load_persona_codec_generator,
    tensor_sha256,
)

SEM_VOCAB = 24
AC_VOCAB = 32
GROUP = 4
SEM_DIM = 16
AC_DIM = 20


class MockSemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(21)
        self.codebook = nn.Embedding(SEM_VOCAB, SEM_DIM)

    def embed_ids(self, ids):
        return self.codebook(ids)


class MockAdapter(nn.Module):
    """B x C x T -> B x C' x (GROUP*T), mirroring Decoder_with_upsample."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(22)
        self.proj = nn.Conv1d(SEM_DIM, AC_DIM, kernel_size=1)

    def forward(self, x):
        return self.proj(x).repeat_interleave(GROUP, dim=-1)


class MockXVC(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(23)
        self.semantic_encoder = MockSemanticEncoder()
        self.semantic_adapter = MockAdapter()
        self.acoustic_quantizer = fvq_module.FactorizedVectorQuantize(
            input_dim=AC_DIM, codebook_size=AC_VOCAB, codebook_dim=4,
            commitment=0.25, codebook_loss_weight=4.0, use_l2_normlize=True,
        )


def tiny_teacher_and_config():
    config = dict(
        semantic_vocab_size=SEM_VOCAB,
        acoustic_vocab_size=AC_VOCAB,
        group_size=GROUP,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_source_positions=64,
        max_target_positions=96,
    )
    torch.manual_seed(9)
    return PersonaCodecTeacher(**config), config


def make_payload(xvc, **overrides):
    teacher, config = tiny_teacher_and_config()
    payload = build_generator_payload(
        model_name="PersonaCodecTeacher",
        state_dict=teacher.state_dict(),
        config=config,
        target_persona="ASI",
        geometry=geometry_from_xvc(xvc),
        group_size=GROUP,
        step=1,
    )
    payload.update(overrides)
    return payload


class TestFailClosedLoader:
    def test_happy_path_round_trip(self, tmp_path):
        xvc = MockXVC()
        path = tmp_path / "teacher.pt"
        torch.save(make_payload(xvc), path)
        generator = load_persona_codec_generator(path, xvc, "cpu",
                                                 expected_target_persona="asi")
        assert not generator.training
        assert generator.group_size == GROUP
        assert generator.payload["target_persona"] == "ASI"
        assert generator.payload["_checkpoint_sha256"]

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"source_speaker_conditioning": True}, "source-speaker"),
            ({"source_speaker_conditioning": None}, "source-speaker"),
            ({"xvc_voice_conditioning_required": False}, "voice conditioning"),
            ({"model": "SneakyModel"}, "unsupported"),
            ({"semantic_vocab_size": SEM_VOCAB + 1}, "semantic_vocab_size mismatch"),
            ({"acoustic_vocab_size": AC_VOCAB - 1}, "acoustic_vocab_size mismatch"),
            ({"semantic_codebook_sha256": None}, "missing semantic_codebook_sha256"),
            ({"quantizer_codebook_sha256": "0" * 64}, "different codebooks"),
            ({"acoustic_group_size": GROUP + 1}, "acoustic_group_size"),
        ],
    )
    def test_incompatible_checkpoints_are_refused(self, tmp_path, overrides, match):
        xvc = MockXVC()
        path = tmp_path / "teacher.pt"
        torch.save(make_payload(xvc, **overrides), path)
        with pytest.raises(ValueError, match=match):
            load_persona_codec_generator(path, xvc, "cpu")

    def test_codebook_drift_is_refused(self, tmp_path):
        xvc = MockXVC()
        path = tmp_path / "teacher.pt"
        torch.save(make_payload(xvc), path)
        with torch.no_grad():
            xvc.acoustic_quantizer.codebook.weight += 0.01
        with pytest.raises(ValueError, match="different codebooks"):
            load_persona_codec_generator(path, xvc, "cpu")

    def test_persona_mismatch_is_refused(self, tmp_path):
        xvc = MockXVC()
        path = tmp_path / "teacher.pt"
        torch.save(make_payload(xvc), path)
        with pytest.raises(ValueError, match="target persona"):
            load_persona_codec_generator(path, xvc, "cpu",
                                         expected_target_persona="TNI")

    def test_payload_certifies_no_warping(self):
        payload = make_payload(MockXVC())
        assert payload["waveform_or_feature_warping"] is False

    def test_group_size_vs_live_adapter_is_refused(self, tmp_path):
        """A checkpoint whose group size is internally consistent but wrong
        for the live adapter's upsample ratio must be refused."""
        xvc = MockXVC()
        _, config = tiny_teacher_and_config()
        config = {**config, "group_size": GROUP + 1}
        torch.manual_seed(11)
        teacher = PersonaCodecTeacher(**config)
        payload = build_generator_payload(
            model_name="PersonaCodecTeacher",
            state_dict=teacher.state_dict(),
            config=config,
            target_persona="ASI",
            geometry=geometry_from_xvc(xvc),
            group_size=GROUP + 1,
            step=1,
        )
        path = tmp_path / "teacher.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError, match="adapter upsample"):
            load_persona_codec_generator(path, xvc, "cpu")


class TestGeometryHelpers:
    def test_live_token_geometry_reads_model(self):
        geometry = live_token_geometry(MockXVC())
        assert geometry == {
            "semantic_vocab_size": SEM_VOCAB,
            "acoustic_vocab_size": AC_VOCAB,
        }

    def test_extraction_meta_geometry_matches_live(self):
        xvc = MockXVC()
        live = geometry_from_xvc(xvc)
        meta = {
            "semantic_vocab_size": SEM_VOCAB,
            "acoustic_vocab_size": AC_VOCAB,
            "semantic_codebook_sha256": live["semantic_codebook_sha256"],
            "quantizer_codebook_sha256": live["quantizer_codebook_sha256"],
        }
        assert geometry_from_extraction_meta(meta) == live

    def test_extraction_meta_missing_hash_fails(self):
        with pytest.raises(ValueError, match="missing codebook hashes"):
            geometry_from_extraction_meta(
                {
                    "semantic_vocab_size": SEM_VOCAB,
                    "acoustic_vocab_size": AC_VOCAB,
                    "semantic_codebook_sha256": "",
                    "quantizer_codebook_sha256": "abc",
                }
            )

    def test_tensor_sha_changes_with_content(self):
        a = torch.ones(4, 2)
        b = torch.ones(4, 2)
        b[0, 0] = 2.0
        assert tensor_sha256(a) != tensor_sha256(b)

    def test_adapter_upsample_ratio_is_measured(self):
        assert adapter_upsample_ratio(MockXVC()) == GROUP


class TestIdentityBypass:
    def test_vq2emb_reproduces_quantizer_output(self):
        """Re-embedding code IDs must equal the stock quantized stream."""
        xvc = MockXVC().eval()
        z = torch.randn(1, AC_DIM, 12)
        with torch.no_grad():
            zq, indices, *_ = xvc.acoustic_quantizer(z)
            re_embedded = xvc.acoustic_quantizer.vq2emb(indices, out_proj=True)
        torch.testing.assert_close(zq, re_embedded)

    def test_embed_token_streams_matches_stock_embedding(self):
        """Passing the source's own token IDs reproduces the stock streams."""
        xvc = MockXVC().eval()
        sem_ids = torch.randint(0, SEM_VOCAB, (1, 6))
        z = torch.randn(1, AC_DIM, 6 * GROUP)
        with torch.no_grad():
            zq_stock, ac_ids, *_ = xvc.acoustic_quantizer(z)
            sem_stock = xvc.semantic_adapter(
                xvc.semantic_encoder.embed_ids(sem_ids).transpose(1, 2)
            ).transpose(1, 2)
            streams = embed_token_streams(xvc, sem_ids, ac_ids)
        torch.testing.assert_close(streams["semantic"], sem_stock)
        torch.testing.assert_close(streams["acoustic_zq"], zq_stock)
        assert torch.equal(streams["acoustic_indices"], ac_ids)

    def test_streams_share_one_timeline(self):
        xvc = MockXVC().eval()
        sem_ids = torch.randint(0, SEM_VOCAB, (1, 6))
        # One acoustic frame short of the semantic upsample (edge behaviour).
        ac_ids = torch.randint(0, AC_VOCAB, (1, 6 * GROUP - 1))
        with torch.no_grad():
            streams = embed_token_streams(xvc, sem_ids, ac_ids)
        frames = 6 * GROUP - 1
        assert streams["semantic"].shape[1] == frames
        assert streams["acoustic_zq"].shape[-1] == frames
        assert streams["acoustic_indices"].shape[-1] == frames

    def test_full_step_misalignment_is_refused(self):
        """A whole semantic step of drift must fail, not be silently trimmed."""
        xvc = MockXVC().eval()
        sem_ids = torch.randint(0, SEM_VOCAB, (1, 6))
        ac_ids = torch.randint(0, AC_VOCAB, (1, 5 * GROUP))
        with pytest.raises(ValueError, match="full semantic step"):
            with torch.no_grad():
                embed_token_streams(xvc, sem_ids, ac_ids)
