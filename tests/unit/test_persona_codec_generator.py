"""CPU tests for PersonaCodecTeacher / PersonaCodecStudent (tiny dims)."""

import inspect

import pytest
import torch

from models.persona_codec_generator import (
    ACTION_BLANK,
    ACTION_EMIT,
    ACTION_EMIT_REPEAT,
    IGNORE_INDEX,
    PersonaCodecStudent,
    PersonaCodecTeacher,
)

SEM_VOCAB = 24
AC_VOCAB = 32
GROUP = 4


def tiny_teacher(**overrides) -> PersonaCodecTeacher:
    config = dict(
        semantic_vocab_size=SEM_VOCAB,
        acoustic_vocab_size=AC_VOCAB,
        group_size=GROUP,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        max_source_positions=64,
        max_target_positions=96,
    )
    config.update(overrides)
    torch.manual_seed(7)
    return PersonaCodecTeacher(**config)


def tiny_student(**overrides) -> PersonaCodecStudent:
    config = dict(
        semantic_vocab_size=SEM_VOCAB,
        acoustic_vocab_size=AC_VOCAB,
        group_size=GROUP,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        delay_steps=1,
        max_positions=64,
        max_expansion_ratio=1.5,
    )
    config.update(overrides)
    torch.manual_seed(11)
    return PersonaCodecStudent(**config)


def example_batch(batch=2, src_len=10, tgt_lens=(8, 5)):
    torch.manual_seed(3)
    src = torch.randint(0, SEM_VOCAB, (batch, src_len))
    src_pad = torch.zeros(batch, src_len, dtype=torch.bool)
    max_tgt = max(tgt_lens)
    tgt_sem = torch.randint(0, SEM_VOCAB, (batch, max_tgt))
    tgt_ac = torch.randint(0, AC_VOCAB, (batch, max_tgt, GROUP))
    lengths = torch.tensor(tgt_lens, dtype=torch.long)
    return src, src_pad, tgt_sem, tgt_ac, lengths


class TestNoSourceSpeakerConditioning:
    def test_teacher_has_no_speaker_inputs(self):
        params = inspect.signature(PersonaCodecTeacher.forward).parameters
        assert not any("speaker" in name or "spk" in name for name in params)
        params = inspect.signature(PersonaCodecTeacher.__init__).parameters
        assert not any("speaker" in name or "spk" in name for name in params)

    def test_student_has_no_speaker_inputs(self):
        for fn in (PersonaCodecStudent.__init__, PersonaCodecStudent.forward_full,
                   PersonaCodecStudent.step):
            params = inspect.signature(fn).parameters
            assert not any("speaker" in name or "spk" in name for name in params)


class TestTeacherForcing:
    def test_labels_and_shifted_inputs(self):
        teacher = tiny_teacher()
        _, _, tgt_sem, tgt_ac, lengths = example_batch()
        batch = teacher.build_teacher_forcing_batch(tgt_sem, tgt_ac, lengths)
        # BOS-first shifted inputs.
        assert (batch["sem_in"][:, 0] == teacher.sem_bos_id).all()
        assert (batch["ac_in"][:, 0] == teacher.ac_bos_id).all()
        assert (batch["sem_in"][0, 1:] == tgt_sem[0][: batch["sem_in"].shape[1] - 1]).all()
        # EOS labelled exactly at each sequence's length.
        for row, length in enumerate([8, 5]):
            assert batch["sem_labels"][row, length] == teacher.sem_eos_id
            assert (batch["sem_labels"][row, :length] == tgt_sem[row, :length]).all()
            # Steps beyond EOS are ignored for loss and masked for attention.
            assert (batch["sem_labels"][row, length + 1:] == IGNORE_INDEX).all()
            assert (batch["ac_labels"][row, length:] == IGNORE_INDEX).all()
            assert not batch["tgt_padding_mask"][row, length]
            assert batch["tgt_padding_mask"][row, length + 1:].all()

    def test_loss_is_finite_and_decreases_on_overfit(self):
        teacher = tiny_teacher()
        src, src_pad, tgt_sem, tgt_ac, lengths = example_batch()
        optimizer = torch.optim.Adam(teacher.parameters(), lr=2e-3)
        first = None
        for _ in range(30):
            losses = teacher.compute_loss(src, src_pad, tgt_sem, tgt_ac, lengths)
            assert torch.isfinite(losses["loss"])
            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()
            if first is None:
                first = float(losses["loss"].detach())
        assert float(losses["loss"].detach()) < first * 0.8

    def test_source_padding_mask_blocks_content(self):
        teacher = tiny_teacher().eval()
        src, src_pad, tgt_sem, tgt_ac, lengths = example_batch()
        src_pad[:, -3:] = True
        batch = teacher.build_teacher_forcing_batch(tgt_sem, tgt_ac, lengths)
        with torch.no_grad():
            out_a = teacher(src, batch["sem_in"], batch["ac_in"],
                            src_padding_mask=src_pad,
                            tgt_padding_mask=batch["tgt_padding_mask"])
            src_b = src.clone()
            src_b[:, -3:] = (src_b[:, -3:] + 1) % SEM_VOCAB
            out_b = teacher(src_b, batch["sem_in"], batch["ac_in"],
                            src_padding_mask=src_pad,
                            tgt_padding_mask=batch["tgt_padding_mask"])
        torch.testing.assert_close(out_a["sem_logits"], out_b["sem_logits"])
        torch.testing.assert_close(out_a["ac_logits"], out_b["ac_logits"])

    def test_decoder_is_causal_over_target_steps(self):
        teacher = tiny_teacher().eval()
        src, src_pad, tgt_sem, tgt_ac, lengths = example_batch()
        batch = teacher.build_teacher_forcing_batch(tgt_sem, tgt_ac, lengths)
        with torch.no_grad():
            out_a = teacher(src, batch["sem_in"], batch["ac_in"], src_padding_mask=src_pad)
            sem_in_b = batch["sem_in"].clone()
            sem_in_b[:, 5:] = (sem_in_b[:, 5:] + 1) % SEM_VOCAB
            out_b = teacher(src, sem_in_b, batch["ac_in"], src_padding_mask=src_pad)
        torch.testing.assert_close(
            out_a["sem_logits"][:, :5], out_b["sem_logits"][:, :5]
        )


class TestTeacherDecode:
    def test_eos_terminates(self):
        teacher = tiny_teacher().eval()
        # Bias the semantic head so EOS wins immediately after min_steps.
        with torch.no_grad():
            teacher.sem_head.bias[teacher.sem_eos_id] = 50.0
        out = teacher.greedy_decode(torch.randint(0, SEM_VOCAB, (1, 12)), min_steps=2)
        assert out["stopped_by_eos"]
        assert out["semantic_tokens"].shape[1] == 2
        assert out["acoustic_groups"].shape == (1, 2, GROUP)

    def test_output_is_bounded_without_eos(self):
        teacher = tiny_teacher().eval()
        with torch.no_grad():
            teacher.sem_head.bias[teacher.sem_eos_id] = -50.0
        src_len = 10
        out = teacher.greedy_decode(
            torch.randint(0, SEM_VOCAB, (1, src_len)), max_len_ratio=1.5, extra_steps=2
        )
        assert not out["stopped_by_eos"]
        assert out["semantic_tokens"].shape[1] == out["max_steps"] == 17
        assert out["acoustic_groups"].shape[1] == 17

    def test_grouped_acoustic_output_shape(self):
        teacher = tiny_teacher().eval()
        out = teacher.greedy_decode(torch.randint(0, SEM_VOCAB, (1, 6)))
        assert out["acoustic_groups"].shape[2] == GROUP
        assert out["acoustic_groups"].max() < AC_VOCAB

    def test_rejects_batched_decode(self):
        teacher = tiny_teacher().eval()
        with pytest.raises(ValueError, match="single"):
            teacher.greedy_decode(torch.randint(0, SEM_VOCAB, (2, 6)))


class TestStudentScaffold:
    def test_cached_steps_match_full_forward(self):
        student = tiny_student().eval()
        src = torch.randint(0, SEM_VOCAB, (1, 12))
        with torch.no_grad():
            full = student.forward_full(src)
            cache = student.init_cache()
            stepped = []
            for position in range(src.shape[1]):
                out = student.step(src[:, position:position + 1], cache, position)
                stepped.append(out["sem_logits"])
            stepped = torch.cat(stepped, dim=1)
        torch.testing.assert_close(full["sem_logits"], stepped, rtol=1e-4, atol=1e-5)

    def test_no_future_leakage(self):
        student = tiny_student().eval()
        src = torch.randint(0, SEM_VOCAB, (1, 12))
        with torch.no_grad():
            out_a = student.forward_full(src)
            src_b = src.clone()
            src_b[:, 8:] = (src_b[:, 8:] + 1) % SEM_VOCAB
            out_b = student.forward_full(src_b)
        torch.testing.assert_close(
            out_a["sem_logits"][:, :8], out_b["sem_logits"][:, :8]
        )
        torch.testing.assert_close(
            out_a["action_logits"][:, :8], out_b["action_logits"][:, :8]
        )

    def test_delay_shift_pads_right_edge_only(self):
        student = tiny_student(delay_steps=1)
        src = torch.arange(5).view(1, 5)
        shifted = student.shift_for_delay(src)
        assert shifted[0, :4].tolist() == [1, 2, 3, 4]
        assert shifted[0, 4] == student.pad_id

    def test_delay_zero_is_identity(self):
        student = tiny_student(delay_steps=0)
        src = torch.arange(5).view(1, 5)
        assert torch.equal(student.shift_for_delay(src), src)

    def test_invalid_delay_rejected(self):
        with pytest.raises(ValueError, match="delay_steps"):
            tiny_student(delay_steps=2)

    def test_blank_emits_nothing(self):
        student = tiny_student().eval()
        with torch.no_grad():
            student.action_head.bias.zero_()
            student.action_head.bias[ACTION_BLANK] = 50.0
        out = student.streaming_decode(torch.randint(0, SEM_VOCAB, (1, 10)))
        assert out["semantic_tokens"].shape[1] == 0
        assert out["acoustic_groups"].shape == (1, 0, GROUP)

    def test_emit_produces_one_step_per_input(self):
        student = tiny_student().eval()
        with torch.no_grad():
            student.action_head.bias.zero_()
            student.action_head.bias[ACTION_EMIT] = 50.0
        out = student.streaming_decode(torch.randint(0, SEM_VOCAB, (1, 10)))
        assert out["semantic_tokens"].shape[1] == 10

    def test_repeat_is_bounded_by_expansion_budget(self):
        student = tiny_student(max_expansion_ratio=1.5).eval()
        with torch.no_grad():
            student.action_head.bias.zero_()
            student.action_head.bias[ACTION_EMIT_REPEAT] = 50.0
        src_len = 10
        out = student.streaming_decode(torch.randint(0, SEM_VOCAB, (1, src_len)))
        # Unbounded doubling would give 20; the budget caps at 15.
        assert out["semantic_tokens"].shape[1] == int(src_len * 1.5)
        assert out["emission_budget"] == 15

    def test_semantic_masking_uses_mask_row(self):
        student = tiny_student()
        src = torch.randint(0, SEM_VOCAB, (1, 40))
        generator = torch.Generator().manual_seed(0)
        corrupted = student.corrupt_source(src, mask_prob=0.2, span=3, generator=generator)
        assert (corrupted == student.mask_id).any()
        untouched = corrupted != student.mask_id
        assert torch.equal(corrupted[untouched], src[untouched])
        assert torch.equal(
            student.corrupt_source(src, mask_prob=0.0), src
        )


def test_vocab_validation():
    with pytest.raises(ValueError, match="semantic_vocab_size"):
        tiny_teacher(semantic_vocab_size=1)
    with pytest.raises(ValueError, match="group_size"):
        tiny_teacher(group_size=0)
