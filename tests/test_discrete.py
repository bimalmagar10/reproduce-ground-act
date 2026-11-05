import torch

from groundact_repro.discrete import DiscreteTrajectoryLogitsProcessor, count_valid_traj_tokens

V = 100
START, END, EOS = 90, 91, 92
TRAJ_LO, TRAJ_VOCAB, FUT_VOCAB, N = 10, 40, 30, 4  # traj tokens 10..49, future tokens 10..39


def proc(prompt_len=3):
    return DiscreteTrajectoryLogitsProcessor(prompt_len, START, END, TRAJ_LO, TRAJ_VOCAB, FUT_VOCAB, N, EOS)


def allowed(scores):
    return set((scores[0] > torch.finfo(scores.dtype).min / 2).nonzero().flatten().tolist())


def test_phases():
    p = proc()
    prompt = [1, 2, 3]
    # CoC phase: trajectory tokens masked, everything else allowed
    s = p(torch.tensor([prompt + [5, 6]]), torch.zeros(1, V))
    a = allowed(s)
    assert not (a & set(range(TRAJ_LO, TRAJ_LO + TRAJ_VOCAB))) and 5 in a and START in a
    # after start: only future tokens
    s = p(torch.tensor([prompt + [5, START]]), torch.zeros(1, V))
    assert allowed(s) == set(range(TRAJ_LO, TRAJ_LO + FUT_VOCAB))
    # after N tokens: only end
    s = p(torch.tensor([prompt + [5, START] + [11] * N]), torch.zeros(1, V))
    assert allowed(s) == {END}
    # after end: only eos
    s = p(torch.tensor([prompt + [5, START] + [11] * N + [END]]), torch.zeros(1, V))
    assert allowed(s) == {EOS}


def test_rows_independent():
    p = proc()
    ids = torch.tensor([[1, 2, 3, 5, 6, 7, 8], [1, 2, 3, START, 11, 12, 13]])
    s = p(ids, torch.zeros(2, V))
    assert 5 in allowed(s[:1]) and allowed(s[1:]) == set(range(TRAJ_LO, TRAJ_LO + FUT_VOCAB))


def test_count_valid():
    seq = torch.tensor([[1, 2, 3, 7, START, 11, 12, 45, 13, END, EOS], [1, 2, 3, 7, 7, 7, 7, 7, 7, 7, 7]])
    n = count_valid_traj_tokens(seq, 3, START, END, TRAJ_LO, TRAJ_LO + FUT_VOCAB)
    assert n.tolist() == [3, 0]
