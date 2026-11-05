"""Token-as-Reference (Section 3.3): reference generation, resolution and consumption.

Eq. 1   p(y_{t+1} = e_i | h_t, M) = pi_t * softmax_i[ tau * cos(W_q h_t, W_k M_i) ],
        pi_t = sigmoid(g(h_t)); the vocabulary head carries the remaining mass 1 - pi_t.
        Detector-null slots are excluded from the pointer softmax whenever a non-null candidate
        exists (gated pointer-generator, See et al. 2017).
Eq. 2   Resolve(e_i, M) = M_i,   u_{t+1} = (1 - alpha) b_ref + alpha (b_i + P(M_i)),
        b_i = entity-query base embedding + slot identity, P projects/scales M_i into the decoder
        input space, alpha in [0, 1].
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from groundact_repro.groundact.entities import EntityBank

REFERENCE_RE = re.compile(r"<\|entity_(\d+)\|>")


def reference_token(i: int) -> str:
    return f"<|entity_{i}|>"


def parse_references(text: str) -> list[int]:
    """Entity indices referenced in a reasoning string, in order of appearance (with repeats)."""
    return [int(m) for m in REFERENCE_RE.findall(text)]


@dataclass
class NextTokenDistribution:
    log_p_vocab: torch.Tensor  # (B, V)   log[(1 - pi) softmax(vocab logits)]
    log_p_ref: torch.Tensor  # (B, K)   log[pi softmax_i(pointer scores)]
    gate: torch.Tensor  # (B,)     pi_t

    def sample(self, temperature: float = 1.0, top_p: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from the joint distribution. Returns (is_reference (B,), index (B,))."""
        joint = torch.cat([self.log_p_vocab, self.log_p_ref], dim=-1) / temperature
        probs = joint.softmax(-1)
        if top_p < 1.0:
            sorted_p, order = probs.sort(descending=True, dim=-1)
            cum = sorted_p.cumsum(-1)
            drop = cum - sorted_p > top_p
            sorted_p = sorted_p.masked_fill(drop, 0.0)
            probs = torch.zeros_like(probs).scatter(-1, order, sorted_p)
            probs = probs / probs.sum(-1, keepdim=True)
        idx = torch.multinomial(probs, 1).squeeze(-1)
        V = self.log_p_vocab.shape[-1]
        is_ref = idx >= V
        return is_ref, torch.where(is_ref, idx - V, idx)


class GatedPointerHead(nn.Module):
    """Reference gate g and entity pointer (W_q, W_k, tau) producing Eq. 1."""

    def __init__(self, hidden_size: int, entity_size: int, proj_size: int = 256,
                 tau: float = 10.0, learn_tau: bool = True):
        super().__init__()
        self.W_q = nn.Linear(hidden_size, proj_size, bias=False)
        self.W_k = nn.Linear(entity_size, proj_size, bias=False)
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size // 4), nn.GELU(),
                                  nn.Linear(hidden_size // 4, 1))
        log_tau = torch.tensor(float(tau)).log()
        self.log_tau = nn.Parameter(log_tau) if learn_tau else log_tau

    def pointer_scores(self, h: torch.Tensor, M: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        """tau * cos(W_q h, W_k M_i), null slots masked when any non-null candidate exists.

        h (B, D_h); M (B, K, D_e) or (K, D_e); valid (B, K) or (K,) bool.
        """
        if M.dim() == 2:
            M = M.unsqueeze(0).expand(h.shape[0], -1, -1)
        q = F.normalize(self.W_q(h), dim=-1)  # (B, P)
        k = F.normalize(self.W_k(M), dim=-1)  # (B, K, P)
        scores = self.log_tau.exp() * torch.einsum("bp,bkp->bk", q, k)
        if valid is not None:
            if valid.dim() == 1:
                valid = valid.unsqueeze(0).expand_as(scores)
            has_valid = valid.any(-1, keepdim=True)
            mask = (~valid) & has_valid
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        return scores

    def forward(self, h: torch.Tensor, vocab_logits: torch.Tensor, M: torch.Tensor,
                valid: torch.Tensor | None = None) -> NextTokenDistribution:
        pi = torch.sigmoid(self.gate(h).squeeze(-1))  # (B,)
        log_pi = F.logsigmoid(self.gate(h).squeeze(-1))
        log_1m_pi = F.logsigmoid(-self.gate(h).squeeze(-1))
        log_p_ref = log_pi[:, None] + self.pointer_scores(h, M, valid).log_softmax(-1)
        log_p_vocab = log_1m_pi[:, None] + vocab_logits.log_softmax(-1)
        return NextTokenDistribution(log_p_vocab, log_p_ref, pi)

    def association_loss(self, h: torch.Tensor, M: torch.Tensor, target: torch.Tensor,
                         valid_pos: torch.Tensor) -> torch.Tensor:
        """L_assoc (Eq. 7): cross-entropy of the matched slot at valid reference positions.

        Note: the paper's Eq. 7 normalizes over all K states (no null masking)."""
        if valid_pos.sum() == 0:
            return h.new_zeros(())
        scores = self.pointer_scores(h[valid_pos], M[valid_pos] if M.dim() == 3 else M, None)
        return F.cross_entropy(scores, target[valid_pos])

    def gate_loss(self, h: torch.Tensor, is_reference: torch.Tensor,
                  valid: torch.Tensor | None = None) -> torch.Tensor:
        """L_gate (Eq. 8): class-balanced BCE between reference and ordinary-text positions."""
        logit = self.gate(h).squeeze(-1)
        if valid is None:
            valid = torch.ones_like(is_reference, dtype=torch.bool)
        losses = []
        for b in (True, False):
            sel = valid & (is_reference == b)
            if sel.any():
                t = torch.full_like(logit[sel], float(b))
                losses.append(F.binary_cross_entropy_with_logits(logit[sel], t))
        return torch.stack(losses).mean() if losses else logit.new_zeros(())


def resolve_reference(bank: EntityBank, index: torch.Tensor) -> torch.Tensor:
    """Resolve(e_i, M) = M_i: read the original prefill state from the pool. index (B,) -> (B, D)."""
    states = bank.states
    if states.dim() == 2:
        return states[index]
    return states.gather(1, index.view(-1, 1, 1).expand(-1, 1, states.shape[-1])).squeeze(1)


class ReferenceConsumer(nn.Module):
    """u_{t+1} = (1 - alpha) b_ref + alpha (b_i + P(M_i))  (Eq. 2)."""

    def __init__(self, entity_size: int, embed_size: int, num_entities: int = 64,
                 alpha: float = 0.5, base_embedding: torch.Tensor | None = None):
        super().__init__()
        self.b_ref = nn.Parameter(torch.zeros(embed_size))  # shared reference embedding
        self.base = nn.Parameter(base_embedding.clone() if base_embedding is not None
                                 else torch.zeros(embed_size))  # entity-query base embedding
        self.slot = nn.Embedding(num_entities, embed_size)  # slot identity
        self.P = nn.Linear(entity_size, embed_size)  # projects and scales the retrieved state
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.alpha = alpha

    def address(self, bank: EntityBank, index: torch.Tensor, stop_grad: bool = False) -> torch.Tensor:
        """a_i = b_i + P(M_i) (before feedback interpolation; sg[M_i] when stop_grad)."""
        M_i = resolve_reference(bank, index)
        if stop_grad:
            M_i = M_i.detach()
        return self.base + self.slot(index) + self.scale * self.P(M_i)

    def forward(self, bank: EntityBank, index: torch.Tensor) -> torch.Tensor:
        a = self.address(bank, index)
        return (1.0 - self.alpha) * self.b_ref + self.alpha * a


@torch.no_grad()
def reference_decode_step(head: GatedPointerHead, consumer: ReferenceConsumer, bank: EntityBank,
                          h_t: torch.Tensor, vocab_logits: torch.Tensor, embed_tokens: nn.Embedding,
                          temperature: float = 1.0, top_p: float = 1.0):
    """One decoding step with the pointer: returns (is_reference, token_or_entity_index, u_{t+1}).

    For an ordinary token the next input embedding is the token embedding; for a reference e_i it
    is the consumed address u_{t+1} that carries M_i into the next step.
    """
    dist = head(h_t, vocab_logits, bank.states, bank.valid)
    is_ref, idx = dist.sample(temperature, top_p)
    u = torch.where(is_ref[:, None], consumer(bank, idx.clamp(max=bank.k - 1)), embed_tokens(idx.clamp(max=embed_tokens.num_embeddings - 1)))
    return is_ref, idx, u
