"""Discrete action-token generation ("D" mode in GroundAct Table 1) for Alpamayo VLMs.

The released Alpamayo inference path masks the 4,000 trajectory tokens during chain-of-causation
(CoC) generation and hands off to the flow-matching action expert at ``<|traj_future_start|>``.
Both checkpoints were nevertheless trained with a discrete trajectory head: after
``<|traj_future_start|>`` the VLM emits 128 tokens ``<i0>..<i2999>`` (64 waypoints x
[acceleration, curvature], 3,000 bins each) followed by ``<|traj_future_end|>``. This module
constrains ``generate`` to that grammar and decodes the tokens with the checkpoint's own
``DiscreteTrajectoryTokenizer`` (unicycle accel/curvature action space).
"""
from __future__ import annotations

import torch
from transformers import LogitsProcessor


class DiscreteTrajectoryLogitsProcessor(LogitsProcessor):
    """Grammar for `CoC -> <|traj_future_start|> -> 128 action tokens -> <|traj_future_end|>`.

    Stateless: the phase of every row is recomputed from ``input_ids`` at each step, which keeps
    it correct under ``num_return_sequences > 1``.
    """

    def __init__(self, prompt_len: int, traj_future_start_id: int, traj_future_end_id: int,
                 traj_token_start_idx: int, traj_vocab_size: int, future_vocab_size: int,
                 tokens_per_future_traj: int, eos_token_id: int):
        self.prompt_len = prompt_len
        self.start_id = traj_future_start_id
        self.end_id = traj_future_end_id
        self.traj_lo = traj_token_start_idx
        self.traj_hi = traj_token_start_idx + traj_vocab_size  # all <i*> tokens (future + history)
        self.fut_lo = traj_token_start_idx
        self.fut_hi = traj_token_start_idx + future_vocab_size
        self.n_traj = tokens_per_future_traj
        self.eos_id = eos_token_id

    def phase(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gen = input_ids[:, self.prompt_len:]
        is_start = gen == self.start_id
        has_start = is_start.any(dim=1)
        pos = is_start.int().argmax(dim=1)
        n_after = gen.shape[1] - 1 - pos
        return has_start, n_after

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        has_start, n_after = self.phase(input_ids)
        neg = torch.finfo(scores.dtype).min
        cot_rows = ~has_start
        traj_rows = has_start & (n_after < self.n_traj)
        end_rows = has_start & (n_after == self.n_traj)
        done_rows = has_start & (n_after > self.n_traj)

        # CoC phase: no trajectory tokens at all (same as the released ExpertLogitsProcessor).
        scores[cot_rows, self.traj_lo:self.traj_hi] = neg
        if traj_rows.any():
            keep = scores[traj_rows, self.fut_lo:self.fut_hi].clone()
            scores[traj_rows] = neg
            scores[traj_rows, self.fut_lo:self.fut_hi] = keep
        if end_rows.any():
            keep = scores[end_rows, self.end_id].clone()
            scores[end_rows] = neg
            scores[end_rows, self.end_id] = keep
        if done_rows.any():
            keep = scores[done_rows, self.eos_id].clone()
            scores[done_rows] = neg
            scores[done_rows, self.eos_id] = keep
        return scores


def count_valid_traj_tokens(sequences: torch.Tensor, prompt_len: int, start_id: int, end_id: int,
                            fut_lo: int, fut_hi: int) -> torch.Tensor:
    """Number of in-range action tokens between the first start and first end token, per row."""
    gen = sequences[:, prompt_len:]
    out = torch.zeros(gen.shape[0], dtype=torch.long, device=gen.device)
    for i in range(gen.shape[0]):
        row = gen[i]
        s = (row == start_id).nonzero()
        if s.numel() == 0:
            continue
        s = int(s[0])
        e = (row[s + 1:] == end_id).nonzero()
        e = s + 1 + int(e[0]) if e.numel() else row.shape[0]
        seg = row[s + 1:e]
        out[i] = int(((seg >= fut_lo) & (seg < fut_hi)).sum())
    return out
