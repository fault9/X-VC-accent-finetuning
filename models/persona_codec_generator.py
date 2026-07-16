"""Target-persona codec-sequence generators (StreamVoice-inspired) for X-VC.

Two models over X-VC's *discrete* token spaces (WhisperVQ semantic token IDs
at 12.5 Hz; acoustic-code IDs at 50 Hz, ``group_size`` codes per semantic
step):

``PersonaCodecTeacher``
    Offline upper-bound model (Phase 1).  A non-causal transformer encoder
    reads the native source semantic token sequence; a causal decoder emits
    the target persona's semantic token *and* its grouped acoustic codes,
    one semantic step (~80 ms) per autoregressive step, with EOS for
    variable-length output.  This borrows StreamVoice's per-step grouped
    acoustic prediction, applied to X-VC's two-stream geometry.

``PersonaCodecStudent``
    Streaming scaffold (Phase 3).  A causal decoder-only stack with an
    explicit per-layer KV cache, 0/1-semantic-step input delay, and a
    blank/emit/emit-repeat action head that bounds local expansion.  The
    scaffold is mechanically complete (causality and cached-step equivalence
    are unit-tested) but is only to be trained by distilling a *passing*
    teacher.

Neither model accepts any source-speaker identity, embedding, or label; the
target persona is fixed per checkpoint and documented in its payload.  Both
models emit token IDs only — rendering happens exclusively through the frozen
X-VC codebooks/renderer (see ``xvc.persona_codec.runtime``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100

# Student action classes.
ACTION_BLANK = 0        # consume the input step, emit nothing (contraction)
ACTION_EMIT = 1         # emit exactly one target step
ACTION_EMIT_REPEAT = 2  # emit the predicted step twice (bounded expansion)
NUM_ACTIONS = 3


def _check_vocab(name: str, value: int) -> int:
    if int(value) < 2:
        raise ValueError(f"{name} must be >= 2, got {value}")
    return int(value)


class PersonaCodecTeacher(nn.Module):
    """Offline native-semantic -> target persona (semantic + acoustic) tokens."""

    def __init__(
        self,
        semantic_vocab_size: int,
        acoustic_vocab_size: int,
        group_size: int = 4,
        d_model: int = 384,
        nhead: int = 6,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_source_positions: int = 512,
        max_target_positions: int = 768,
    ):
        super().__init__()
        self.semantic_vocab_size = _check_vocab("semantic_vocab_size", semantic_vocab_size)
        self.acoustic_vocab_size = _check_vocab("acoustic_vocab_size", acoustic_vocab_size)
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.group_size = int(group_size)
        self.d_model = int(d_model)
        self.max_source_positions = int(max_source_positions)
        self.max_target_positions = int(max_target_positions)

        # Decoder-input ids: real semantic tokens plus one BOS row.
        self.sem_bos_id = self.semantic_vocab_size
        # Semantic output classes: real tokens plus one EOS class.
        self.sem_eos_id = self.semantic_vocab_size
        # Acoustic-input ids: real codes plus one BOS row (step 0 has no
        # previous acoustic group).
        self.ac_bos_id = self.acoustic_vocab_size

        self.src_embed = nn.Embedding(self.semantic_vocab_size, d_model)
        self.src_pos = nn.Embedding(self.max_source_positions, d_model)
        self.tgt_sem_embed = nn.Embedding(self.semantic_vocab_size + 1, d_model)
        self.tgt_ac_embed = nn.Embedding(self.acoustic_vocab_size + 1, d_model)
        self.tgt_pos = nn.Embedding(self.max_target_positions, d_model)
        self.ac_group_proj = nn.Linear(d_model, d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                batch_first=True, norm_first=True,
            ),
            num_encoder_layers,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout,
                batch_first=True, norm_first=True,
            ),
            num_decoder_layers,
        )
        self.sem_head = nn.Linear(d_model, self.semantic_vocab_size + 1)
        self.ac_head = nn.Linear(d_model, self.group_size * self.acoustic_vocab_size)

    # ------------------------------------------------------------------ #

    def _embed_source(self, src_tokens: torch.Tensor) -> torch.Tensor:
        if src_tokens.shape[1] > self.max_source_positions:
            raise ValueError(
                f"source length {src_tokens.shape[1]} exceeds "
                f"max_source_positions {self.max_source_positions}"
            )
        positions = torch.arange(src_tokens.shape[1], device=src_tokens.device)
        return self.src_embed(src_tokens) + self.src_pos(positions)[None]

    def _embed_target_step_inputs(
        self, sem_in: torch.Tensor, ac_in: torch.Tensor
    ) -> torch.Tensor:
        """Embed shifted decoder inputs: previous semantic token + previous group."""
        if sem_in.shape[1] > self.max_target_positions:
            raise ValueError(
                f"target length {sem_in.shape[1]} exceeds "
                f"max_target_positions {self.max_target_positions}"
            )
        if ac_in.shape[:2] != sem_in.shape or ac_in.shape[2] != self.group_size:
            raise ValueError(
                f"acoustic input {tuple(ac_in.shape)} does not match semantic "
                f"input {tuple(sem_in.shape)} with group size {self.group_size}"
            )
        positions = torch.arange(sem_in.shape[1], device=sem_in.device)
        ac = self.ac_group_proj(self.tgt_ac_embed(ac_in).mean(dim=2))
        return self.tgt_sem_embed(sem_in) + ac + self.tgt_pos(positions)[None]

    def encode(
        self, src_tokens: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.encoder(
            self._embed_source(src_tokens), src_key_padding_mask=src_padding_mask
        )

    def forward(
        self,
        src_tokens: torch.Tensor,          # [B, S] semantic token ids
        tgt_sem_in: torch.Tensor,          # [B, T] shifted (BOS-first) target ids
        tgt_ac_in: torch.Tensor,           # [B, T, G] shifted (BOS-first) groups
        src_padding_mask: Optional[torch.Tensor] = None,  # [B, S] True = pad
        tgt_padding_mask: Optional[torch.Tensor] = None,  # [B, T] True = pad
        memory: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if memory is None:
            memory = self.encode(src_tokens, src_padding_mask)
        steps = self._embed_target_step_inputs(tgt_sem_in, tgt_ac_in)
        # Bool mask (True = blocked) keeps dtype consistent with the bool
        # key-padding masks.
        causal = torch.triu(
            torch.ones(steps.shape[1], steps.shape[1], dtype=torch.bool, device=steps.device),
            diagonal=1,
        )
        hidden = self.decoder(
            steps,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        sem_logits = self.sem_head(hidden)
        ac_logits = self.ac_head(hidden).view(
            hidden.shape[0], hidden.shape[1], self.group_size, self.acoustic_vocab_size
        )
        return {"sem_logits": sem_logits, "ac_logits": ac_logits, "memory": memory}

    # ------------------------------------------------------------------ #

    def build_teacher_forcing_batch(
        self,
        tgt_sem: torch.Tensor,             # [B, T] real target semantic ids
        tgt_ac: torch.Tensor,              # [B, T, G] real target groups
        tgt_lengths: torch.Tensor,         # [B] valid target lengths
    ) -> dict[str, torch.Tensor]:
        """Shifted inputs and CE labels with EOS appended at each valid end.

        Output tensors have T+1 steps: step t predicts target step t, and the
        step at ``length`` predicts EOS.  Padded positions carry IGNORE_INDEX.
        """
        batch, steps = tgt_sem.shape
        device = tgt_sem.device
        sem_in = torch.full((batch, steps + 1), self.sem_bos_id, dtype=torch.long, device=device)
        sem_in[:, 1:] = tgt_sem
        ac_in = torch.full(
            (batch, steps + 1, self.group_size), self.ac_bos_id, dtype=torch.long, device=device
        )
        ac_in[:, 1:] = tgt_ac

        arange = torch.arange(steps + 1, device=device)[None]
        lengths = tgt_lengths[:, None]
        valid_step = arange < lengths            # real target steps
        eos_step = arange == lengths             # one EOS label per sequence

        sem_labels = torch.full((batch, steps + 1), IGNORE_INDEX, dtype=torch.long, device=device)
        sem_labels[:, :steps][valid_step[:, :steps]] = tgt_sem[valid_step[:, :steps]]
        sem_labels[eos_step] = self.sem_eos_id

        ac_labels = torch.full(
            (batch, steps + 1, self.group_size), IGNORE_INDEX, dtype=torch.long, device=device
        )
        ac_labels[:, :steps][valid_step[:, :steps]] = tgt_ac[valid_step[:, :steps]]

        padding_mask = ~(valid_step | eos_step)  # True = pad for attention
        # Inputs at padded steps must still be valid embedding ids.
        sem_in = sem_in.masked_fill(padding_mask, self.sem_bos_id)
        ac_in = ac_in.masked_fill(padding_mask[..., None], self.ac_bos_id)
        return {
            "sem_in": sem_in,
            "ac_in": ac_in,
            "sem_labels": sem_labels,
            "ac_labels": ac_labels,
            "tgt_padding_mask": padding_mask,
        }

    def compute_loss(
        self,
        src_tokens: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_sem: torch.Tensor,
        tgt_ac: torch.Tensor,
        tgt_lengths: torch.Tensor,
        acoustic_loss_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced cross-entropy over discrete token IDs (no L2)."""
        batch = self.build_teacher_forcing_batch(tgt_sem, tgt_ac, tgt_lengths)
        out = self.forward(
            src_tokens,
            batch["sem_in"],
            batch["ac_in"],
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=batch["tgt_padding_mask"],
        )
        sem_loss = F.cross_entropy(
            out["sem_logits"].flatten(0, 1),
            batch["sem_labels"].flatten(),
            ignore_index=IGNORE_INDEX,
        )
        ac_loss = F.cross_entropy(
            out["ac_logits"].flatten(0, 2),
            batch["ac_labels"].flatten(),
            ignore_index=IGNORE_INDEX,
        )
        loss = sem_loss + acoustic_loss_weight * ac_loss
        with torch.no_grad():
            sem_valid = batch["sem_labels"] != IGNORE_INDEX
            sem_acc = (
                (out["sem_logits"].argmax(-1) == batch["sem_labels"])[sem_valid]
                .float().mean()
            )
            ac_valid = batch["ac_labels"] != IGNORE_INDEX
            ac_acc = (
                (out["ac_logits"].argmax(-1) == batch["ac_labels"])[ac_valid]
                .float().mean()
            )
        return {
            "loss": loss,
            "sem_loss": sem_loss,
            "ac_loss": ac_loss,
            "sem_accuracy": sem_acc,
            "ac_accuracy": ac_acc,
        }

    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def greedy_decode(
        self,
        src_tokens: torch.Tensor,          # [1, S]
        max_len_ratio: float = 1.6,
        min_steps: int = 1,
        extra_steps: int = 8,
    ) -> dict[str, torch.Tensor]:
        """Beam-1 decoding with a hard variable-length bound.

        The output cannot run away: generation stops at EOS or at
        ``ceil(S * max_len_ratio) + extra_steps`` steps, whichever is first
        (also capped by ``max_target_positions``).
        """
        if src_tokens.ndim != 2 or src_tokens.shape[0] != 1:
            raise ValueError("greedy_decode expects a single [1, S] sequence")
        self.eval()
        device = src_tokens.device
        source_len = src_tokens.shape[1]
        max_steps = min(
            self.max_target_positions - 1,
            int(math.ceil(source_len * max_len_ratio)) + extra_steps,
        )
        memory = self.encode(src_tokens)

        sem_in = torch.full((1, 1), self.sem_bos_id, dtype=torch.long, device=device)
        ac_in = torch.full(
            (1, 1, self.group_size), self.ac_bos_id, dtype=torch.long, device=device
        )
        sem_out: list[int] = []
        ac_out: list[list[int]] = []
        stopped_by_eos = False
        for step in range(max_steps):
            out = self.forward(src_tokens, sem_in, ac_in, memory=memory)
            sem_logits = out["sem_logits"][0, -1]
            if step < min_steps:
                sem_logits = sem_logits.clone()
                sem_logits[self.sem_eos_id] = float("-inf")
            sem_id = int(sem_logits.argmax().item())
            if sem_id == self.sem_eos_id:
                stopped_by_eos = True
                break
            group = out["ac_logits"][0, -1].argmax(-1)          # [G]
            sem_out.append(sem_id)
            ac_out.append([int(x) for x in group.tolist()])
            sem_in = torch.cat(
                [sem_in, torch.tensor([[sem_id]], dtype=torch.long, device=device)], dim=1
            )
            ac_in = torch.cat([ac_in, group.view(1, 1, self.group_size)], dim=1)
        return {
            "semantic_tokens": torch.tensor([sem_out], dtype=torch.long, device=device),
            "acoustic_groups": torch.tensor([ac_out], dtype=torch.long, device=device)
            if ac_out else torch.zeros((1, 0, self.group_size), dtype=torch.long, device=device),
            "stopped_by_eos": stopped_by_eos,
            "max_steps": max_steps,
        }


class _CausalSelfAttention(nn.Module):
    """Single-head-group causal self-attention with an explicit KV cache."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                              # [B, T, D]
        cache: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch, steps, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def shape(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, -1, self.nhead, self.head_dim).transpose(1, 2)

        q, k, v = shape(q), shape(k), shape(v)
        past = 0
        if cache is not None:
            if "k" in cache:
                past = cache["k"].shape[2]
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        # Causal over the full (past + current) timeline.
        total = k.shape[2]
        mask = torch.ones(steps, total, dtype=torch.bool, device=x.device)
        mask = torch.tril(mask, diagonal=past)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[None, None]
        )
        attended = attended.transpose(1, 2).reshape(batch, steps, -1)
        return self.dropout(self.out(attended))


class _CausalBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, cache: Optional[dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cache=cache)
        return x + self.ff(self.norm2(x))


class PersonaCodecStudent(nn.Module):
    """Causal streaming scaffold: per-source-step action + token emission.

    The student consumes source semantic tokens step by step (optionally
    delayed by ``delay_steps`` so step ``t`` may see ``t + delay`` inputs) and
    per consumed step emits one of {blank, emit, emit-repeat} plus the target
    semantic token and its acoustic-code group.  Local expansion is bounded by
    construction: at most two emitted steps per consumed step, and the decode
    wrapper enforces a global expansion budget.

    Semantic masking (StreamVoice-style context corruption) is provided via
    ``corrupt_source``; the mask row is a dedicated learned embedding.
    """

    def __init__(
        self,
        semantic_vocab_size: int,
        acoustic_vocab_size: int,
        group_size: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        delay_steps: int = 1,
        max_positions: int = 512,
        max_expansion_ratio: float = 1.5,
    ):
        super().__init__()
        self.semantic_vocab_size = _check_vocab("semantic_vocab_size", semantic_vocab_size)
        self.acoustic_vocab_size = _check_vocab("acoustic_vocab_size", acoustic_vocab_size)
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        if delay_steps not in (0, 1):
            raise ValueError("delay_steps must be 0 or 1 (80 ms right-context budget)")
        self.group_size = int(group_size)
        self.delay_steps = int(delay_steps)
        self.max_positions = int(max_positions)
        self.max_expansion_ratio = float(max_expansion_ratio)

        self.mask_id = self.semantic_vocab_size          # corruption token
        self.pad_id = self.semantic_vocab_size + 1       # right-edge padding
        self.src_embed = nn.Embedding(self.semantic_vocab_size + 2, d_model)
        self.pos_embed = nn.Embedding(self.max_positions, d_model)
        self.blocks = nn.ModuleList(
            _CausalBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, NUM_ACTIONS)
        self.sem_head = nn.Linear(d_model, self.semantic_vocab_size)
        self.ac_head = nn.Linear(d_model, self.group_size * self.acoustic_vocab_size)

    def corrupt_source(
        self, src_tokens: torch.Tensor, mask_prob: float, span: int = 10,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Mask random spans of the source (StreamVoice semantic masking)."""
        if not 0.0 <= mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1]")
        corrupted = src_tokens.clone()
        # torch.rand requires the generator and the output tensor to share a
        # device; sample on the generator's device (CPU for the portable
        # seeded default) and move, so seeded masking also works on CUDA.
        rand_device = generator.device if generator is not None else src_tokens.device
        starts = (
            torch.rand(src_tokens.shape, generator=generator, device=rand_device)
            < mask_prob
        ).to(src_tokens.device)
        mask = torch.zeros_like(starts)
        for offset in range(span):
            shifted = starts[:, : src_tokens.shape[1] - offset]
            mask[:, offset:] |= shifted
        corrupted[mask] = self.mask_id
        return corrupted

    def shift_for_delay(self, src_tokens: torch.Tensor) -> torch.Tensor:
        """Give step ``t`` access to input ``t + delay`` by shifting left.

        The final ``delay`` positions see the pad row — the model never reads
        beyond the provided sequence (no future fabrication).
        """
        if self.delay_steps == 0:
            return src_tokens
        shifted = torch.full_like(src_tokens, self.pad_id)
        shifted[:, : -self.delay_steps] = src_tokens[:, self.delay_steps:]
        return shifted

    def _embed(self, tokens: torch.Tensor, start_position: int) -> torch.Tensor:
        end = start_position + tokens.shape[1]
        if end > self.max_positions:
            raise ValueError(f"positions {end} exceed max_positions {self.max_positions}")
        positions = torch.arange(start_position, end, device=tokens.device)
        return self.src_embed(tokens) + self.pos_embed(positions)[None]

    def forward_full(self, src_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full-sequence causal forward (training / test reference)."""
        x = self._embed(src_tokens, 0)
        for block in self.blocks:
            x = block(x)
        return self._heads(self.norm(x))

    def init_cache(self) -> list[dict[str, torch.Tensor]]:
        return [{} for _ in self.blocks]

    def step(
        self,
        src_token: torch.Tensor,                        # [B, 1]
        cache: list[dict[str, torch.Tensor]],
        position: int,
    ) -> dict[str, torch.Tensor]:
        """Incremental forward for one consumed source step using the KV cache."""
        x = self._embed(src_token, position)
        for block, block_cache in zip(self.blocks, cache):
            x = block(x, cache=block_cache)
        return self._heads(self.norm(x))

    def _heads(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        ac = self.ac_head(hidden).view(
            hidden.shape[0], hidden.shape[1], self.group_size, self.acoustic_vocab_size
        )
        return {
            "action_logits": self.action_head(hidden),
            "sem_logits": self.sem_head(hidden),
            "ac_logits": ac,
        }

    @torch.inference_mode()
    def streaming_decode(self, src_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Cache-based decode with hard local and global expansion bounds."""
        if src_tokens.ndim != 2 or src_tokens.shape[0] != 1:
            raise ValueError("streaming_decode expects a single [1, S] sequence")
        self.eval()
        device = src_tokens.device
        shifted = self.shift_for_delay(src_tokens)
        cache = self.init_cache()
        sem_out: list[int] = []
        ac_out: list[list[int]] = []
        emission_budget = max(1, int(src_tokens.shape[1] * self.max_expansion_ratio))
        for position in range(shifted.shape[1]):
            out = self.step(shifted[:, position:position + 1], cache, position)
            action = int(out["action_logits"][0, -1].argmax().item())
            if action == ACTION_BLANK:
                continue
            sem_id = int(out["sem_logits"][0, -1].argmax().item())
            group = [int(x) for x in out["ac_logits"][0, -1].argmax(-1).tolist()]
            repeats = 2 if action == ACTION_EMIT_REPEAT else 1
            for _ in range(repeats):
                if len(sem_out) >= emission_budget:
                    break
                sem_out.append(sem_id)
                ac_out.append(group)
        return {
            "semantic_tokens": torch.tensor([sem_out], dtype=torch.long, device=device),
            "acoustic_groups": torch.tensor([ac_out], dtype=torch.long, device=device)
            if ac_out else torch.zeros((1, 0, self.group_size), dtype=torch.long, device=device),
            "emission_budget": emission_budget,
        }
