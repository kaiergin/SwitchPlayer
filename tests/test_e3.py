"""E3 (Kai, 2026-11-05; CURRICULUM.md "Queued experiment (pre-registered) --
E3", amendments E3-A2..A10): tokens mined from the master's own play,
appended at every swap, mined WITHOUT grouping. Pins: the token helpers and
the canonicalizer (E3-A2: a duration token stays an (action, frames) pair,
a mined token is a tuple of steps); play_token's per-base-step inputs; edge
discounts, n-step frames and replay units by total frames; the gate; the
long-token bias cap (E3-A7); the row buckets and read (2)'s shares
(E3-A10); the vocabulary rules on both sides (E3-A4, E3-A8: the 1% rule,
appended, capped at the freed slots and 8, prefix-free, a vocabulary-equal
key dropped, the declared tails, JSON round trips); the mining score count
x (k - 1) x consistency on synthetic logs (E3-A3); the target rule's path
products on tiny nets (E3-A5); relabeling (E3-A6); the second swap (E3-A8);
the supports (E3-A9) -- every duration-only path bit for bit as before --
and, slow, on real Ms. Pac-Man (CPU): a mined prep from a T0-like master's
log, its student offline and online, and the second swap from that student
as the master."""
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import distill
import muzero
from muzero import IMAGE_RESHAPE, initial_frame_stack, preprocess_frame, \
    push_frame
from tests.test_distill import _ReplayEnv, _replay_shell
from tests.test_prioract2 import restore_rngs  # noqa: F401
from tests.test_testb import T, TK_VOCAB, arms, logged  # noqa: F401
from tests.test_tokens import (_dtraj, _Steps, _token_shell,  # noqa: F401
                               prior_tokens, tok_arm)

M1 = ((1, 8), (3, 4))                  # 12 frames: inputs 1, 1, 3
M2 = ((2, 4), (4, 12), (1, 4))         # 20 frames: inputs 2, 4, 4, 4, 1
BIG = ((1, 32), (3, 32))               # 64 frames: 16 base steps
DROP = ((0, 8), (0, 32), (5, 16))
KEPT = tuple(t for t in TK_VOCAB if t not in DROP)            # 33 tokens
G = 0.997 ** 4
SLOT = {t: i for i, t in enumerate(TK_VOCAB)}


def J(vocab):
    return [muzero.token_json(t) for t in vocab]


# --------------------------------------------------------------------------
# E3-A2: the helpers, the canonicalizer, JSON
# --------------------------------------------------------------------------
def test_token_helpers_on_both_kinds():
    assert not muzero.token_is_mined((3, 16)) and muzero.token_is_mined(M1)
    assert muzero.token_steps((3, 16)) == ((3, 16),)
    assert muzero.token_steps(M2) == M2
    assert muzero.token_total_frames((3, 16)) == 16
    assert muzero.token_total_frames(M1) == 12
    assert muzero.token_total_frames(M2) == 20
    assert muzero.token_inputs((3, 16)) == (3, 3, 3, 3)
    assert muzero.token_inputs(M1) == (1, 1, 3)
    assert muzero.token_inputs(M2) == (2, 4, 4, 4, 1)
    assert muzero.token_inputs(BIG) == (1,) * 8 + (3,) * 8
    # numpy ints are a duration token's numbers, never steps
    assert not muzero.token_is_mined((np.int64(3), np.int64(16)))
    assert muzero.token_total_frames((np.int64(3), np.int64(16))) == 16
    # two spellings of the same inputs are one token's inputs (A13)
    assert muzero.token_inputs(((1, 4), (1, 4), (3, 4))) == \
        muzero.token_inputs(M1)


def test_canon_token_reads_json_lists_and_collapses_one_step():
    t = muzero.canon_token([3, 16])
    assert t == (3, 16) and type(t) is tuple and type(t[0]) is int
    assert muzero.canon_token([[1, 8], [3, 4]]) == M1
    assert muzero.canon_token(((1.0, 8.0), (3, 4))) == M1
    assert all(type(x) is int for s in muzero.canon_token(
        ((1.0, 8.0), (3, 4))) for x in s)
    assert muzero.canon_token([[1, 8]]) == (1, 8)      # a one-step sequence
    assert muzero.canon_token(np.array([2, 8])) == (2, 8)
    for bad in ("ab", [], [1, 2, 3], [[1, 8], 3], [[1, 8], [[2, 4]]], 5,
                [[1, 2, 3]], [[]]):
        with pytest.raises((TypeError, ValueError)):
            muzero.canon_token(bad)
    assert muzero.canon_vocab([[0, 4], [[1, 8], [3, 4]], [[2, 8]]]) == \
        ((0, 4), M1, (2, 8))
    with pytest.raises(TypeError):
        muzero.canon_vocab("nonsense")


def test_json_round_trip_and_string_keys():
    vocab = ((0, 4), (3, 32), M1, M2)
    js = json.loads(json.dumps(J(vocab)))
    assert js[0] == [0, 4] == list((0, 4))         # as every pair was written
    assert js[2] == [[1, 8], [3, 4]]
    assert muzero.canon_vocab(js) == vocab
    assert [muzero.token_key(t) for t in vocab] == [
        "0,4", "3,32", "1,8;3,4", "2,4;4,12;1,4"]
    doc = json.loads(json.dumps({"mined": [{"token": J([M1])[0]},
                                           {"token": J([M2])[0]}],
                                 "master_mined": J([BIG])}))
    assert ablation.declared_mined(doc) == (M1, M2)
    assert ablation.declared_mined(doc, "master_mined") == (BIG,)
    assert ablation.declared_mined({}) == ()
    with pytest.raises(SystemExit, match="not a list"):
        ablation.declared_mined({"mined": "x"})


# --------------------------------------------------------------------------
# E3-A2: frames, discounts, n-step targets, replay units, play_token
# --------------------------------------------------------------------------
def test_duration_tokens_keep_frames_discounts_and_bias_bit_for_bit(
        monkeypatch):
    monkeypatch.setattr(muzero, "DISCOUNT", G)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", TK_VOCAB)
    # the pre-E3 formulas, verbatim
    assert [muzero.token_frames(i) for i in range(36)] == [
        int(TK_VOCAB[i][1]) for i in range(36)]
    assert muzero.token_edge_discounts() == tuple(
        float(G) ** (int(f) / 4) for _, f in TK_VOCAB)
    for b in (0.5, 0.25, 0.01):
        old = np.array([(int(f) / 4) ** b for _, f in TK_VOCAB],
                       dtype=np.float64)
        assert np.array_equal(ablation.token_bias_weights(b), old)
    assert ablation.mined_overlap(TK_VOCAB) is None


def test_a_mined_token_lasts_and_discounts_its_total_frames(monkeypatch):
    D = 0.9
    monkeypatch.setattr(muzero, "DISCOUNT", D)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), M2, (5, 20)))
    assert [muzero.token_frames(i) for i in range(3)] == [4, 20, 20]
    d = muzero.token_edge_discounts()
    assert d[1] == d[2] == D ** (20 / 4) and d[0] == D
    # the mined token is one edge of its total frames: the same n-step
    # targets as a duration token of the same length, at every horizon
    rewards = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    tr_m = _dtraj([1, 0, 0, 1, 0, 0], rewards)
    tr_d = _dtraj([2, 0, 0, 2, 0, 0], rewards)
    for idx in range(6):
        for h in (1, 2, 5, 6, 9):
            assert muzero.paper_value_parts(tr_m, idx, h, D) == \
                muzero.paper_value_parts(tr_d, idx, h, D)
    # horizon 5 = 20 frames: the mined token alone covers it
    assert muzero.paper_value_parts(tr_m, 0, 5, D) == (1.0, D ** (20 / 4), 1)
    # horizon 6 = 24 frames: then one 4-frame token
    assert muzero.paper_value_parts(tr_m, 0, 6, D) == (
        1.0 + D ** (20 / 4) * 2.0, D ** (20 / 4) * D ** (4 / 4), 2)
    # the replay counts its 5 base steps (+ the final record)
    assert ablation.replay_units([{"action": 1}, {"action": 0},
                                  {"action": 0}]) == 5 + 1 + 1


def test_play_token_plays_each_base_step_with_its_own_action(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), M2))
    monkeypatch.setattr(ablation, "INNER_FRAMES", [])
    env = _Steps(reward=lambda t: float(t))
    obs, _ = env.reset()
    stack = initial_frame_stack(preprocess_frame(obs, IMAGE_RESHAPE))
    inner = []
    obs, r, term, trunc, out, n = ablation.play_token(env, 1, stack, inner)
    assert env.actions == [2, 4, 4, 4, 1] and n == 5 and not term
    assert r == 1.0 + 2.0 + 3.0 + 4.0 + 5.0 and inner == [1.0, 2.0, 3.0,
                                                           4.0, 5.0]
    expect = stack
    for t in (1, 2, 3, 4):     # the 5th frame is pushed at the next decision
        expect = push_frame(expect, preprocess_frame(env.frame(t),
                                                     IMAGE_RESHAPE))
    assert np.array_equal(out, expect) and np.array_equal(obs, env.frame(5))
    assert len(ablation.INNER_FRAMES) == 1
    assert len(ablation.INNER_FRAMES[0]) == 4
    # a terminal inside it stops there, its actions so far played
    env2 = _Steps(end=3)
    obs, _ = env2.reset()
    s2 = initial_frame_stack(preprocess_frame(obs, IMAGE_RESHAPE))
    _, _, term, _, _, n = ablation.play_token(env2, 1, s2)
    assert env2.actions == [2, 4, 4] and term and n == 3
    assert len(ablation.INNER_FRAMES[1]) == 2


def test_play_episode_counts_a_mined_tokens_base_steps(prior_tokens,
                                                       monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), M1, M2))
    monkeypatch.setattr(ablation, "ACTIONS", 3)
    env = _Steps(reward=lambda t: 1.0)
    run = _token_shell(env, [0.1, 0.1, 0.8])            # greedy: M2
    traj, total, steps, term = run.play_episode(greedy=True, cap=12)
    # M2 = 5 base steps: 0 -> 5 -> 10 -> 15 (cap 12)
    assert steps == 15 and len(traj) == 4 and not term
    assert env.actions == [2, 4, 4, 4, 1] * 3
    assert [r[2] for r in traj] == [0.0, 5.0, 5.0, 5.0]
    np.random.seed(3)
    env = _Steps()
    run = _token_shell(env, [1 / 3] * 3)
    traj, _, steps, _ = run.play_episode(greedy=False, cap=40)
    assert steps == sum(muzero.token_frames(r[1]) // 4 for r in traj[:-1])
    assert steps == len(env.actions) == run.env_steps


# --------------------------------------------------------------------------
# the gate; the long-token bias (E3-A7); the supports (E3-A9)
# --------------------------------------------------------------------------
def test_the_gate_takes_mined_tokens(tok_arm, monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (2, 8), M1, M2,
                                                ((1, 32), (2, 32))))
    ablation.token_gate(tok_arm, True, True, 0)


@pytest.mark.parametrize("vocab, match", [
    (((0, 4), ((1, 6), (3, 4))), "positive multiple"),
    (((0, 4), ((1, 8), (7, 4))), "outside"),
    (((0, 4), ((1, 8),)), "at least two steps"),
    (((0, 4), ((1, 32), (3, 36))), "at most 64"),
    (((0, 4), (1, 8), ((1, 4), (1, 4))), "duplicate"),     # (1, 8)'s inputs
    (((0, 4), M1, ((1, 4), (1, 4), (3, 4))), "duplicate"),  # M1's inputs
    (((0, 4), (0, 4)), "duplicate"),
])
def test_the_gate_refuses(tok_arm, monkeypatch, vocab, match):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", vocab)
    with pytest.raises(SystemExit, match=match):
        ablation.token_gate(tok_arm, True, True, 0)


def test_the_bias_caps_a_mined_token_at_the_longest_duration(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (0, 32), M1, BIG))
    w = ablation.token_bias_weights(0.5)
    # M1 (12 frames) uncapped; BIG (64) at the 32-frame token's weight
    assert w.tolist() == [1.0, (32 / 4) ** 0.5, (12 / 4) ** 0.5,
                          (32 / 4) ** 0.5]
    assert w[3] < (64 / 4) ** 0.5
    # the vocabulary's own longest duration token sets the cap
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (0, 16), BIG))
    assert ablation.token_bias_weights(0.5)[2] == (16 / 4) ** 0.5


def test_supports_read_the_longest_token_only_with_mined_tokens(
        monkeypatch):
    sup = muzero.scalar_categorical.support_for

    def old(c):        # the pre-E3 check, verbatim
        return not (sup(c) > muzero.REWARD_SUPPORT
                    or sup(c * muzero.UNROLL_STEPS)
                    > muzero.VALUE_PREFIX_SUPPORT)

    for c in (0.5, 1.0, 7.26, 17.4, 160.0, 399.0, 400.0, 400.5, 1000.0):
        assert ablation.ceiling_fits_supports(c) == old(c)
        assert ablation.ceiling_fits_supports(c, 1) == old(c)
    assert ablation.support_steps(None) == 1
    assert ablation.support_steps(TK_VOCAB) == 1      # duration-only: as ever
    assert ablation.support_steps(TK_VOCAB + (BIG,)) == 16
    assert ablation.support_steps(TK_VOCAB + (M1,)) == 8   # its (a, 32)s
    assert ablation.ceiling_fits_supports(17.4, 16)    # T2's prep: 17.4
    assert ablation.ceiling_fits_supports(25.0, 16)    # 400 per record
    assert not ablation.ceiling_fits_supports(25.01, 16)
    assert not ablation.ceiling_fits_supports(160.0, 16)


def test_the_growth_gate_reads_the_longest_mined_token(monkeypatch):
    tk = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-" \
         "mspacman-tokens-rgrow"
    for k in ablation.ARMS[tk]:
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ENV_NAME", "FRAMESKIP"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    ablation.ENV_NAME, ablation.FRAMESKIP = "MsPacmanNoFrameskip-v4", 4
    with ablation.arm_config(tk):
        ablation.reward_growth_gate(True, True, 0)    # 160 x 1: as ever
        muzero.TOKEN_VOCAB = KEPT + (BIG,)
        with pytest.raises(SystemExit, match="record of 16 base steps"):
            ablation.reward_growth_gate(True, True, 0)
        muzero.REWARD_GROWTH_START = muzero.REWARD_GROWTH_CEILING = 17.4
        ablation.reward_growth_gate(True, True, 0)


# --------------------------------------------------------------------------
# E3-A10: the row diagnostics
# --------------------------------------------------------------------------
def test_rows_bucket_mined_tokens_and_report_each_ones_shares(monkeypatch):
    vocab = ((0, 4), (1, 4), (2, 32), M1)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", vocab)
    run = object.__new__(ablation.Run)
    run.replay = []
    pols = [np.array([0.1, 0.4, 0.2, 0.3]), np.array([0.0, 0.5, 0.0, 0.5]),
            np.array([0.2, 0.2, 0.2, 0.4]), np.full(4, 0.25)]
    out = [{"action": a, "target_policy": p, "root_value": 0.0}
           for a, p in zip((1, 3, 2, 0), pols)]          # the last: final
    run._note_tokens(out, [0.0] * 4)
    errs = {}
    run._merge_row_stats(errs)
    third = round(1 / 3, 4)
    assert errs["tok_duration_play_share"] == {"4": third, "32": third,
                                               "mined": third}
    mass = pols[0] + pols[1] + pols[2]
    share = mass / mass.sum()
    assert errs["tok_duration_search_share"] == {
        "4": round(float(share[0] + share[1]), 4),
        "32": round(float(share[2]), 4), "mined": round(float(share[3]), 4)}
    assert errs["tok_mined_play_share"] == {"1,8;3,4": third}
    assert errs["tok_mined_search_share"] == {"1,8;3,4": round(1.2 / 3, 4)}
    # the redundancy: only decision 0 played a token whose inputs (1)
    # overlap M1's (1, 1, 3) -- decision 1 played M1 itself, decision 2
    # (2, 32): M1's mass 0.3 there, over 3 decisions
    assert errs["tok_mined_overlap_search_share"] == round(0.3 / 3, 4)
    assert errs["tok_frames_per_decision"] == round((4 + 12 + 32) / 3, 3)
    # a duration-only vocabulary: no mined field at all
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", vocab[:3])
    run._note_tokens([{"action": 1, "target_policy": np.full(3, 1 / 3),
                       "root_value": 0.0}] * 2, [0.0, 0.0])
    errs = {}
    run._merge_row_stats(errs)
    assert set(errs["tok_duration_play_share"]) == {"4", "32"}
    assert not any(k.startswith("tok_mined") for k in errs)


def test_the_overlap_matrix():
    vocab = ((0, 4), (1, 4), (1, 8), M1, ((1, 4), (2, 4)), (3, 4))
    o = ablation.mined_overlap(vocab)
    # columns: only the mined tokens (3: inputs 1 1 3; 4: inputs 1 2)
    assert o[:, :3].sum() == 0 and o[:, 5].sum() == 0
    assert o[:, 3].tolist() == [0, 1, 1, 0, 0, 0]     # (1), (1,1); not (1,2)
    assert o[:, 4].tolist() == [0, 1, 0, 0, 0, 0]     # (1); (1,1) differs
    assert o[3, 3] == 0                                # never itself


# --------------------------------------------------------------------------
# E3-A4 / E3-A8: the student's vocabulary, both sides
# --------------------------------------------------------------------------
def test_the_drop_rule_treats_a_masters_mined_tokens_as_tokens():
    master = ((0, 4), (1, 4), (1, 8), M1, M2)
    vocab, dropped = distill.student_vocabulary(
        master, [0.0, 0.2, 0.005, 0.2, 0.009])
    assert vocab == ((0, 4), (1, 4), M1)      # a base token never goes
    assert dropped == ((1, 8), M2)
    # JSON lists in, canonical tokens out
    vocab, _ = distill.student_vocabulary(J(master), [0.2] * 5)
    assert vocab == master


def test_validate_accepts_exactly_the_declared_tail():
    new = (M1, M2)
    v = ablation.validate_student_vocab(J(KEPT + new), TK_VOCAB,
                                        mined=J(new), max_mined=8)
    assert v == KEPT + new
    # without a tail: the rule as it always was
    assert ablation.validate_student_vocab(J(KEPT), TK_VOCAB) == KEPT


@pytest.mark.parametrize("vocab, mined, match", [
    (KEPT + (M1,), None, "outside the master's"),          # undeclared
    (KEPT + (M2, M1), (M1, M2), "declared mined"),          # order
    (KEPT[:5] + (M1,) + KEPT[5:] + (M2,), (M1, M2), "declared mined"),
    ((M1,), (M1,), "drops a 4-frame"),
    (KEPT + (((1, 4), (1, 4)),), (((1, 4), (1, 4)),), "same per-base-step"),
    (KEPT + ((5, 16),), ((5, 16),), "is a duration token"),
    (KEPT + (((1, 32), (3, 36)),), (((1, 32), (3, 36)),), "more than 64"),
    (KEPT + (((1, 6), (3, 4)),), (((1, 6), (3, 4)),), "whole 4-frame"),
    (KEPT + tuple(((a, 4), (b, 8)) for a in range(3) for b in range(3)),
     tuple(((a, 4), (b, 8)) for a in range(3) for b in range(3)),
     "at most 8 per swap"),
    (KEPT + tuple(((a, 4), (b, 8)) for a in range(2) for b in range(2)),
     tuple(((a, 4), (b, 8)) for a in range(2) for b in range(2)),
     "past the master's 36"),
])
def test_validate_refusals(vocab, mined, match):
    with pytest.raises(SystemExit, match=match):
        ablation.validate_student_vocab(J(vocab), TK_VOCAB,
                                        mined=None if mined is None
                                        else J(mined), max_mined=8)


def _e3_doc(tmp_path, vocab, mined, master_arm=T[0], master_vocab=None,
            master_mined=None, c=7.26, set_path=None, name="vocab.json"):
    doc = {"vocab": J(vocab), "student_ceiling": c, "master_arm": master_arm,
           "master_iterations": 47, "student_discount": G,
           "discount_rule": "keep", "master_discount": G,
           "mined": [{"token": muzero.token_json(t)} for t in mined]}
    if master_vocab is not None:
        doc["master_vocab"] = J(master_vocab)
    if master_mined is not None:
        doc["master_mined"] = J(master_mined)
    if set_path is not None:
        doc["set_sha256"] = ablation.file_sha256(set_path)
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


M1_VOCAB = KEPT + (M1, M2)                         # what M1 played
M3 = ((6, 4), (7, 8))
# M2's: M1's minus (1, 16) and the mined M2; M1 kept; M3 appended
M2_VOCAB = tuple(t for t in M1_VOCAB if t not in ((1, 16), M2)) + (M3,)


def test_a_first_swap_student_takes_its_mined_tail(tmp_path, arms):
    static = ablation.ARMS[T[1]]
    spec = ablation.student_arm(T[1], _e3_doc(tmp_path, M1_VOCAB, (M1, M2)))
    assert spec["TOKEN_VOCAB"] == M1_VOCAB
    assert spec["_mined_tokens"] == (M1, M2)
    assert "_master_vocab" not in spec and "_master_mined" not in spec
    assert spec["REWARD_GROWTH_CEILING"] == 7.26
    with ablation.arm_config(T[1]), ablation.token_actions():
        assert muzero.TOKEN_VOCAB == M1_VOCAB and ablation.ACTIONS == 35
        out = ablation.student_expect(spec, None)
    assert out["token_vocab"] == J(M1_VOCAB)
    assert out["token_vocab"][-2] == [[1, 8], [3, 4]]
    # a "master_mined" for a master that is no student
    ablation.ARMS[T[1]] = static
    with pytest.raises(SystemExit, match="is no student"):
        ablation.student_arm(T[1], _e3_doc(tmp_path, M1_VOCAB, (M1, M2),
                                           master_mined=(BIG,)))
    # the ceiling's support check reads the longest token (E3-A9)
    ablation.ARMS[T[1]] = static
    with pytest.raises(SystemExit, match="record of 16 base steps"):
        ablation.student_arm(T[1], _e3_doc(tmp_path, KEPT + (BIG,), (BIG,),
                                           c=25.5))
    ablation.ARMS[T[1]] = static
    ablation.student_arm(T[1], _e3_doc(tmp_path, KEPT + (M1,), (M1,),
                                       c=25.5))      # 8 base steps x 25.5
    # more than 8 mined tokens in one swap
    ablation.ARMS[T[1]] = static
    nine = tuple(((a, 4), (b, 8)) for a in range(3) for b in range(3))
    with pytest.raises(SystemExit, match="at most 8 per swap"):
        ablation.student_arm(T[1], _e3_doc(
            tmp_path, tuple(t for t in KEPT if t[1] != 16) + nine, nine))


def test_a_vocab_json_without_mined_tokens_applies_as_before(tmp_path,
                                                             arms):
    spec = ablation.student_arm(T[1], _e3_doc(tmp_path, KEPT, ()))
    assert spec["TOKEN_VOCAB"] == KEPT
    assert not {"_mined_tokens", "_master_mined"} & set(spec)


def test_a_second_swap_student_of_a_master_holding_mined_tokens(tmp_path,
                                                                arms):
    doc = _e3_doc(tmp_path, M2_VOCAB, (M3,), master_arm=T[1],
                  master_vocab=M1_VOCAB, master_mined=(M1, M2))
    static = ablation.ARMS[T[2]]
    spec = ablation.student_arm(T[2], doc)
    assert spec["TOKEN_VOCAB"] == M2_VOCAB
    assert spec["_master_vocab"] == M1_VOCAB
    assert spec["_master_mined"] == (M1, M2) and spec["_mined_tokens"] == (M3,)
    head = tuple(t for t in M2_VOCAB if t not in (M1, M3))
    for vocab, mined, master_mined, why in (
            # the master's mined tail undeclared
            (M2_VOCAB, (M3,), None, "outside the master's"),
            # declared, but not the tail the master's vocabulary holds
            (head + (M1, BIG, M3), (M3,), (M1, M2, BIG), "declared mined"),
            # a mined token declared nowhere
            (M2_VOCAB, (), (M1, M2), "neither in 'mined' nor"),
            (M2_VOCAB, (BIG,), (M1, M2), "neither in 'mined' nor"),
            # this prep's tail must close the vocabulary
            (head + (M3, M1), (M3,), (M1, M2), "declared mined")):
        ablation.ARMS[T[2]] = static
        with pytest.raises(SystemExit, match=why):
            ablation.student_arm(T[2], _e3_doc(
                tmp_path, vocab, mined, master_arm=T[1],
                master_vocab=M1_VOCAB, master_mined=master_mined))


def _student_spec(arm, vocab, **extra):
    return {**ablation.ARMS[arm], "TOKEN_VOCAB": vocab,
            "_token_vocab_sha256": "a" * 64, "_vocab_set_sha256": "b" * 64,
            "REWARD_GROWTH_START": 7.26, "REWARD_GROWTH_CEILING": 7.26,
            "_student_ceiling": 7.26, "DISCOUNT": G, "_student_discount": G,
            **extra}


def test_the_master_spec_of_masters_holding_mined_tokens():
    m1 = _student_spec(T[1], M1_VOCAB, _mined_tokens=(M1, M2))
    assert distill.master_spec(T[1], {"spec": m1}, "m") == m1
    with pytest.raises(SystemExit, match="outside the master's"):
        distill.master_spec(T[1], {"spec": {k: v for k, v in m1.items()
                                            if k != "_mined_tokens"}}, "m")
    with pytest.raises(SystemExit, match="no student"):
        distill.master_spec(T[1], {"spec": {**m1,
                                            "_master_mined": (M1,)}}, "m")
    # M2 as a master: its own master M1 a student holding mined tokens
    m2 = _student_spec(T[2], M2_VOCAB, _mined_tokens=(M3,),
                       _master_vocab=M1_VOCAB, _master_mined=(M1, M2))
    assert distill.master_spec(T[2], {"spec": m2}, "m") == m2
    with pytest.raises(SystemExit, match="outside the master's"):
        distill.master_spec(T[2], {"spec": {k: v for k, v in m2.items()
                                            if k != "_master_mined"}}, "m")


def test_the_gate_compares_canonical_vocabularies(tmp_path, arms):
    header = {"format": distill.FORMAT, "vocab": J(M1_VOCAB),
              "master_vocab": J(TK_VOCAB), "master_snapshot": "m.pth",
              "master_arm": T[0], "student_ceiling": 7.26,
              "student_discount": G, "source": "own-play log"}
    good = tmp_path / "set.pkl"
    distill.write_set(good, header, [])
    other = tmp_path / "other.pkl"
    distill.write_set(other, {**header, "vocab": J(KEPT + (M2, M1))}, [])
    static = ablation.ARMS[T[1]]
    for dset, ok in ((good, True), (other, False)):
        ablation.ARMS[T[1]] = static
        ablation.student_arm(T[1], _e3_doc(tmp_path, M1_VOCAB, (M1, M2),
                                           set_path=dset))
        with ablation.arm_config(T[1]) as spec:
            if ok:
                ablation.student_gate(spec, dset, True)
            else:
                with pytest.raises(SystemExit, match="differs from"):
                    ablation.student_gate(spec, dset, True)


# --------------------------------------------------------------------------
# E3-A3: mining on synthetic logs
# --------------------------------------------------------------------------
def test_mining_keys_by_inputs_and_drops_only_student_tokens():
    s = SLOT
    lives = [{"slots": [s[(5, 16)], s[(5, 16)], s[(3, 16)]],
              "full": [True] * 3, "offset": 0},
             {"slots": [s[(5, 32)], s[(3, 16)], s[(6, 4)]],
              "full": [True, True, False], "offset": 3}]
    high = {0, 1, 3, 4}
    cands, dropped, stats = distill.mine_candidates(lives, TK_VOCAB, high,
                                                    TK_VOCAB)
    key = (5,) * 8 + (3,) * 4
    assert [c["inputs"] for c in cands] == [key, (5,) * 4 + (3,) * 4]
    first = cands[0]
    assert first["count"] == 2 and first["n"] == 3        # the first seen
    assert first["key"] == ((5, 16), (5, 16), (3, 16))
    assert first["spellings"] == [(((5, 16), (5, 16), (3, 16)), 1),
                                  (((5, 32), (3, 16)), 1)]
    assert first["occurrences"] == [(0, 0, 3), (1, 0, 2)]
    assert first["length"] == 48
    # (5,16)(5,16) holds (5, 32)'s inputs -- a student token: dropped
    assert dropped == {(5,) * 8: 1}
    assert stats["vocab_equal"] == 1 and stats["cut_short"] == 2
    # (5, 32) gone from the student's vocabulary: the key is a candidate
    student = tuple(t for t in TK_VOCAB if t != (5, 32))
    cands, dropped, _ = distill.mine_candidates(lives, TK_VOCAB, high,
                                                student)
    assert (5,) * 8 in [c["inputs"] for c in cands] and dropped == {}
    # a master's mined token: its own steps' inputs
    vocab = TK_VOCAB + (M1,)
    lives = [{"slots": [36, s[(2, 4)]], "full": [True, True], "offset": 0}]
    cands, _, _ = distill.mine_candidates(lives, vocab, {0}, vocab)
    assert cands[0]["inputs"] == (1, 1, 3, 2)
    assert distill.mined_token(cands[0]["key"]) == ((1, 8), (3, 4), (2, 4))


def test_effect_consistency_and_the_score():
    ec = distill.effect_consistency
    assert ec([]) is None
    assert ec([[3.0, 4.0]]) == pytest.approx(1.0)
    assert ec([[1.0, 2.0], [2.0, 4.0]]) == pytest.approx(1.0)
    # orthogonal changes: each at 45 degrees from the mean
    assert ec([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(2 ** -0.5)
    # opposite changes: a zero mean change -> 0
    assert ec([[1.0, 0.0], [-1.0, 0.0]]) == 0.0
    # 1 - the mean cosine DISTANCE, by hand: changes (1,0), (1,1), (0,1)
    d = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    mu = d.mean(0)
    cos = [x @ mu / np.linalg.norm(x) / np.linalg.norm(mu) for x in d]
    assert ec(d) == pytest.approx(1 - np.mean([1 - c for c in cos]))
    assert distill.mined_score(5, 3, 0.5) == 5.0
    assert distill.mined_score(5, 2, 0.5) == 2.5
    assert distill.mined_score(5, 3, None) == 0.0


def test_the_port_matches_e3as_probe_on_duration_vocabularies():
    """The prep's mining is E3a's (benchmarks/probes/probe_e3a_sequences):
    the same returns, high half and candidates -- E3a's duration-token drop
    is the vocabulary-equal drop when the student keeps the vocabulary --
    on random logs over T2's vocabulary."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                           "benchmarks" / "probes"))
    import probe_e3a_sequences as e3
    from tests.test_e3a import T2_VOCAB
    rng = np.random.default_rng(11)
    for _ in range(5):
        lives, adv, off = [], [], 0
        for _ in range(int(rng.integers(1, 5))):
            m = int(rng.integers(1, 30))
            slots = rng.integers(0, len(T2_VOCAB), m).tolist()
            full = (rng.random(m) > 0.1).tolist()
            lives.append({"slots": slots, "full": full, "offset": off})
            off += m
            traj = [(None, slots[0], ())] + [
                (None, slots[i + 1] if i + 1 < m else 0,
                 tuple(float(x) for x in rng.choice(
                     [0, 10, 50, 200, -10],
                     T2_VOCAB[slots[i]][1] // 4)))
                for i in range(m)]
            a, sa = e3.decision_returns(traj, 10.0, 5.0, 0.97)
            b, sb = distill.decision_returns(traj, 10.0, 5.0, 0.97)
            assert np.array_equal(a, b) and sa == sb
            adv.extend((a - rng.normal(size=m)).tolist())
        assert e3.high_half(adv) == distill.high_half(adv)
        high = distill.high_half(adv)[0]
        want, wd, ws = e3.mine_candidates(lives, T2_VOCAB, high, top=10 ** 6)
        got, gd, gs = distill.mine_candidates(lives, T2_VOCAB, high,
                                              T2_VOCAB)
        assert [(c["inputs"], c["key"], c["count"], c["n"], c["length"],
                 c["spellings"]) for c in got] == [
            (c["inputs"], c["key"], c["count"], c["n"], c["length"],
             c["spellings"]) for c in want]
        assert gd == wd
        assert gs.get("vocab_equal", 0) == ws.get("duration_tokens", 0)
        assert sum(c["count"] for c in got) == gs.get("counted", 0) == \
            ws.get("counted", 0)
        assert gs["keys"] == ws["keys"]


def test_the_selection_is_prefix_free_by_score_and_capped():
    def c(inputs, score):
        return {"inputs": inputs, "score": score}

    cands = [c((1, 2, 3), 9.0), c((1, 2), 5.0), c((1, 2, 3, 4), 4.0),
             c((2, 1), 4.0), c((3, 3), 0.0), c((4, 4), 3.0),
             c((5, 5), 4.0), c((6, 6), -1.0)]
    chosen, skipped = distill.select_mined(cands, 8)
    assert [x["inputs"] for x in chosen] == [(1, 2, 3), (2, 1), (5, 5),
                                             (4, 4)]
    # (1, 2) is a prefix of a higher-scored selection, (1, 2, 3, 4) extends
    # one; the nonpositive scores never enter
    assert skipped == {"prefix": 2, "score": 2}
    assert [x["inputs"] for x in distill.select_mined(cands, 2)[0]] == \
        [(1, 2, 3), (2, 1)]
    assert distill.select_mined(cands, 0)[0] == []


class _LatRep(nn.Module):
    """The latent IS the stack's bytes (flattened); eval mode."""

    def forward(self, x):
        return (x * 255.0).round().reshape(len(x), -1)


class _ValPred(nn.Module):
    """Value = the latent's first coordinate x `scale`."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, z):
        return (torch.zeros(len(z), 3), self.scale * z[:, :1], None)


def _lat_run(scale=0.0):
    return SimpleNamespace(trainer=SimpleNamespace(
        representation=_LatRep().eval(), prediction=_ValPred(scale).eval(),
        device=torch.device("cpu")))


def test_mine_wires_the_advantage_the_latents_and_the_score():
    """Two lives over V; latents designed per stack; every base step pays
    1 (unit 1, no clip, discount 1): a decision's return is its life's base
    steps left; the value is latent[0] x 0.25 (exact in float32). The score
    is count x (k - 1) x c, c over h(stack after the run) - h(stack
    before) -- recomputed here from the stacks -- with an occurrence ending
    at a game-ending life's final record left out of c."""
    V = ((0, 4), (1, 4), (2, 4), (0, 8), (1, 8), (2, 8))
    rng = np.random.default_rng(7)

    def life(slots, game_end, cut=False):
        stacks = [rng.integers(0, 255, (1, 1, 3), dtype=np.uint8)
                  for _ in range(len(slots) + 1)]
        traj = [(stacks[0], slots[0], ())]
        for i, s in enumerate(slots):
            n = muzero.token_total_frames(V[s]) // 4
            if cut and i == len(slots) - 1:
                n -= 1
            traj.append((stacks[i + 1], slots[i + 1] if i + 1 < len(slots)
                         else 0, (1.0,) * n))
        return {"traj": traj, "terminated": True, "game_end": game_end}

    lives = [life([0, 1, 0, 1, 2, 4, 3], False),
             life([0, 1, 0, 1, 5], True)]
    chosen, rec = distill.mine_tokens(_lat_run(0.25), lives, V, V, 3, 1.0,
                                      100.0, 1.0)
    # the advantage by hand
    rets, vals = [], []
    for lf in lives:
        traj = lf["traj"]
        steps = [len(traj[i + 1][2]) for i in range(len(traj) - 1)]
        for i in range(len(steps)):
            rets.append(float(sum(steps[i:])))
            vals.append(0.25 * float(traj[i][0].reshape(-1)[0]))
    adv = np.array(rets) - np.array(vals)
    high, thr = distill.high_half(adv)
    assert rec["high"] == len(high) == 6 and rec["high_threshold"] == thr
    assert rec["decisions"] == 12 and rec["lives"] == 2
    infos = [{"slots": [int(r[1]) for r in lf["traj"][:-1]],
              "full": [True] * (len(lf["traj"]) - 1),
              "offset": o} for lf, o in zip(lives, (0, 7))]
    cands, _, _ = distill.mine_candidates(infos, V, high, V)
    assert rec["candidates"] == len(cands) > 0
    want = []
    for c in cands:
        deltas = []
        for li, i, n in c["occurrences"]:
            traj = lives[li]["traj"]
            if lives[li]["game_end"] and i + n == len(traj) - 1:
                continue
            deltas.append(traj[i + n][0].reshape(-1).astype(float)
                          - traj[i][0].reshape(-1).astype(float))
        cc = distill.effect_consistency(deltas)
        want.append((c["inputs"], distill.mined_score(c["count"], c["n"],
                                                      cc)))
    got = {tuple(x["inputs"]): x["score"] for x in rec["scored"]}
    for inputs, score in want:
        assert got[tuple(inputs)] == pytest.approx(score)
    best = distill.select_mined([{"inputs": i, "score": s} for i, s in want],
                                3)[0]
    assert [c["inputs"] for c in chosen] == [b["inputs"] for b in best]
    slot_v = {t: k for k, t in enumerate(V)}
    for c in chosen:
        assert c["path"] == tuple(slot_v[t] for t in c["key"])
        assert muzero.token_inputs(c["token"]) == c["inputs"]
        assert c["token"] == distill.mined_token(c["key"])
    assert rec["cap"] == 3
    json.dumps(rec)                                   # vocab.json-ready


def test_a_stale_final_stack_is_left_out_of_the_consistency():
    """The same run (0, 1) twice, both first decisions high -- once inside
    life a, once ending at the final record of life b, which ended the
    game: b's is counted but left out of c. With it, c would be cos 45."""
    V = ((0, 4), (1, 4), (2, 4))

    def st(*v):
        return np.array(v, np.uint8).reshape(1, 1, 3)

    a = {"traj": [(st(0, 0, 0), 0, ()), (st(1, 0, 0), 1, (9.0,)),
                  (st(2, 0, 0), 2, (0.5,)), (st(0, 0, 0), 0, (0.5,))],
         "terminated": True, "game_end": False}
    b = {"traj": [(st(0, 0, 0), 0, ()), (st(0, 5, 0), 1, (9.0,)),
                  (st(0, 2, 0), 0, (0.5,))],
         "terminated": True, "game_end": True}
    # returns: a 10, 1, 0.5; b 9.5, 0.5 -> the high half {a's 0, b's 0}
    chosen, rec = distill.mine_tokens(_lat_run(0.0), [a, b], V, V, 8, 1.0,
                                      9.0, 1.0)
    run01 = [x for x in rec["scored"] if x["inputs"] == [0, 1]]
    assert run01 and run01[0]["count"] == 2
    # c from life a's occurrence alone (b's ends at its stale final stack)
    assert run01[0]["consistency"] == pytest.approx(1.0)
    assert run01[0]["score"] == pytest.approx(2.0)
    assert rec["stale_occurrences"] == 1
    # the same lives, b not ending the game: both occurrences, cos 45
    b["game_end"] = False
    _, rec = distill.mine_tokens(_lat_run(0.0), [a, b], V, V, 8, 1.0, 9.0,
                                 1.0)
    run01 = [x for x in rec["scored"] if x["inputs"] == [0, 1]][0]
    assert run01["consistency"] == pytest.approx(2 ** -0.5)
    assert rec["stale_occurrences"] == 0


# --------------------------------------------------------------------------
# E3-A5: the target rule
# --------------------------------------------------------------------------
class _OneHotRep(nn.Module):
    def __init__(self, S):
        super().__init__()
        self.S = S

    def forward(self, x):
        s = (x[:, 0, 0, 0] * 255.0).round().long()
        return torch.nn.functional.one_hot(s, self.S).float()


class _TableDyn(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = torch.as_tensor(table)

    def forward(self, z, a):
        s = z.argmax(1)
        nxt = self.table[s, a]
        return torch.nn.functional.one_hot(nxt, z.shape[1]).float(), None


class _TablePred(nn.Module):
    def __init__(self, P):
        super().__init__()
        self.logp = torch.log(torch.as_tensor(P, dtype=torch.float32))

    def forward(self, z):
        return self.logp[z.argmax(1)], torch.zeros(len(z), 1), None


def test_the_path_products_on_tiny_nets():
    """3 states x 4 actions: T[s, a] the next state, P[s] the prior. A
    path (s1, s2, s3) from state x weighs P[T[x, s1], s2] x
    P[T[T[x, s1], s2], s3]; one token: 1."""
    T_ = np.array([[1, 2, 0, 1], [2, 0, 1, 2], [0, 1, 2, 0]])
    P = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.25, 0.125, 0.125],
                  [0.7, 0.1, 0.1, 0.1]])
    nets = {"representation": _OneHotRep(3), "dynamics": _TableDyn(T_),
            "prediction": _TablePred(P)}
    nets["dynamics"].train()                     # restored after the call
    nets["prediction"].eval()
    stacks = [np.full((1, 1, 1), x, np.uint8) for x in (0, 1, 2, 1)]
    paths = [(0, 1), (3, 0, 2), (2,)]
    w = distill.mined_path_weights(nets, stacks, paths, torch.device("cpu"),
                                   batch=3)
    for i, st in enumerate(stacks):
        x = int(st[0, 0, 0])
        a = T_[x, 0]
        assert w[i, 0] == pytest.approx(P[a, 1], rel=1e-6)
        b = T_[x, 3]
        c = T_[b, 0]
        assert w[i, 1] == pytest.approx(P[b, 0] * P[c, 2], rel=1e-6)
        assert w[i, 2] == 1.0
    assert nets["dynamics"].training and not nets["prediction"].training


def _old_masked(pols, master_vocab, student_vocab):
    """The pre-E3 masked_targets, verbatim."""
    idx = [list(map(tuple, master_vocab)).index(t) for t in student_vocab]
    out, fallbacks, gone = [], 0, []
    for p in pols:
        full = np.asarray(p, dtype=np.float64)
        q = full[idx]
        s = q.sum()
        gone.append(float(full.sum() - s))
        if s > 0:
            out.append(q / s)
        else:
            out.append(np.ones(len(idx)) / len(idx))
            fallbacks += 1
    return out, fallbacks, gone


def test_without_mined_tokens_the_targets_are_bit_for_bit():
    rng = np.random.default_rng(0)
    pols = [rng.dirichlet(np.ones(36)) for _ in range(20)]
    pols += [np.where(np.arange(36) == SLOT[(0, 8)], 1.0, 0.0)]
    a = distill.masked_targets(pols, TK_VOCAB, KEPT)
    b = _old_masked(pols, TK_VOCAB, KEPT)
    assert a[1:] == b[1:] and all(np.array_equal(x, y)
                                  for x, y in zip(a[0], b[0]))


MASTER5 = ((0, 4), (1, 4), (2, 4), (1, 8), (2, 8))
MA = ((1, 4), (2, 4))       # path (1, 2): from (1, 4)'s slot
MB = ((2, 8), (0, 4))       # path (4, 0): from the DROPPED (2, 8)
STUDENT6 = ((0, 4), (1, 4), (2, 4), (1, 8), MA, MB)


def test_the_target_rule_by_hand():
    pols = [np.array([0.1, 0.4, 0.2, 0.1, 0.2]),
            np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            np.zeros(5)]
    weights = np.array([[0.5, 0.25], [0.3, 0.6], [0.5, 0.5]])
    out, fb, gone = distill.masked_targets(pols, MASTER5, STUDENT6,
                                           mined=(1, 4), weights=weights)
    # record 0: MA = 0.4 x 0.5 moved from (1, 4); MB = 0.2 x 0.25 moved from
    # (2, 8), whose remaining 0.15 is discarded with it; renormalized
    q0 = np.array([0.1, 0.2, 0.2, 0.1, 0.2, 0.05])
    assert out[0] == pytest.approx(q0 / 0.85)
    assert gone[0] == pytest.approx(0.15)
    # MA's slot is nonzero though nothing played it here
    assert out[0][4] > 0
    # record 1: all the master's mass on the dropped (2, 8) -- MB keeps
    # its path's share, the rest is discarded
    assert out[1] == pytest.approx([0, 0, 0, 0, 0, 1.0])
    assert gone[1] == pytest.approx(0.4)
    # record 2: no mass anywhere -> uniform over the 6 student slots
    assert out[2] == pytest.approx([1 / 6] * 6) and fb == 1


def test_two_mined_tokens_from_one_slot_split_its_mass():
    """Prefix-free paths from the same t1 are disjoint: their moved masses
    sum to at most q(t1), and t1 keeps the rest."""
    mc = ((1, 4), (0, 4))
    student = ((0, 4), (1, 4), (2, 4), (1, 8), MA, mc)
    p = np.array([0.1, 0.5, 0.2, 0.1, 0.1])
    out, _, gone = distill.masked_targets([p], MASTER5, student,
                                          mined=(1, 1),
                                          weights=[[0.6, 0.3]])
    q = np.array([0.1, 0.5 * 0.1, 0.2, 0.1, 0.5 * 0.6, 0.5 * 0.3])
    assert out[0] == pytest.approx(q / q.sum())
    assert gone[0] == pytest.approx(0.1)          # only (2, 8)'s
    assert out[0][1] + out[0][4] + out[0][5] == pytest.approx(0.5 / 0.9)


# --------------------------------------------------------------------------
# E3-A6: relabeling; E3-A8: a master's dropped mined token
# --------------------------------------------------------------------------
V6 = ((0, 4), (1, 4), (2, 4), (1, 8), (2, 8), (0, 16))
MX = ((1, 8), (2, 4))                                  # inputs 1, 1, 2
S6 = ((0, 4), (1, 4), (2, 4), (1, 8), (2, 8), MX)      # (0, 16) dropped


def _fr(v):
    return np.full((3, 4, 4), v, dtype=np.uint8)


def _plan_life(vocab, plan):
    """A recorded life: `plan` [(slot, base steps played)]; base step t
    (from 1) pays t and shows frame t."""
    s = muzero.initial_frame_stack(_fr(0))
    traj, inner, t, incoming = [], [], 0, ()
    for slot, m in plan:
        traj.append((s, slot, incoming))
        pushed = []
        for j in range(1, m + 1):
            t += 1
            if j < m:
                pushed.append(_fr(t))
            s = muzero.push_frame(s, _fr(t))
        inner.append(pushed)
        incoming = tuple(float(x) for x in range(t - m + 1, t + 1))
    traj.append((s, 0, incoming))
    return traj, inner


def test_relabeling_greedy_at_decision_boundaries():
    # d0 (1,8) | d1 (2,4) | d2 (0,16) dropped | d3 (1,4) | d4 (1,4) |
    # d5 (2,4) | d6 (1,8) cut short after 1 of 2 base steps
    traj, inner = _plan_life(V6, [(3, 2), (2, 1), (5, 4), (1, 1), (1, 1),
                                  (2, 1), (3, 1)])
    distill.check_life(traj, inner)
    stats = {}
    out = distill.retokenize(traj, inner, V6, S6,
                             mined={muzero.token_inputs(MX): 5}, stats=stats)
    assert [r[1] for r in out] == [5, 0, 0, 0, 0, 5, 3, 0]
    assert [r[2] for r in out] == [(), (1.0, 2.0, 3.0), (4.0,), (5.0,),
                                   (6.0,), (7.0,), (8.0, 9.0, 10.0),
                                   (11.0,)]
    # the relabeled records sit on their stretch's start stack; the
    # interior stacks (decisions 1, 4, 5) leave the set
    assert out[0][0] is traj[0][0] and out[5][0] is traj[3][0]
    assert out[6][0] is traj[6][0] and out[7][0] is traj[7][0]
    kept = [id(r[0]) for r in out]
    assert all(id(traj[i][0]) not in kept for i in (1, 4, 5))
    # the split dropped token: its base steps' stacks
    for j in range(4):
        assert out[1 + j][0][-3:].max() == 3 + j
    assert stats == {"records": {5: 2}, "decisions": 5, "split": 1}
    # every raw reward and base step exactly once
    assert sum(sum(r[2]) for r in out) == sum(sum(r[2]) for r in traj)
    assert sum(len(r[2]) for r in out) == sum(len(r[2]) for r in traj)


def test_relabeling_takes_the_longest_and_only_whole_decisions():
    traj, inner = _plan_life(V6, [(3, 2), (2, 1), (1, 1)])
    # two mined keys from one boundary (not prefix-free here): the longest
    out = distill.retokenize(traj, inner, V6, S6 + (((1, 4), (1, 4)),),
                             mined={(1, 1): 6, (1, 1, 2): 5})
    assert [r[1] for r in out] == [5, 1, 0]
    out = distill.retokenize(traj, inner, V6, S6 + (((1, 4), (1, 4)),),
                             mined={(1, 1): 6})
    assert [r[1] for r in out] == [6, 2, 1, 0]
    # (1, 2) occurs only across d0's second base step and d1: never matched
    out = distill.retokenize(traj, inner, V6, S6, mined={(1, 2): 5})
    assert [r[1] for r in out] == [3, 2, 1, 0]
    # a stretch ending at the life's last, full decision is matched
    traj, inner = _plan_life(V6, [(0, 1), (3, 2), (2, 1)])
    out = distill.retokenize(traj, inner, V6, S6, mined={(1, 1, 2): 5})
    assert [r[1] for r in out] == [0, 5, 0]
    assert out[-1][2] == (2.0, 3.0, 4.0)
    # its last decision cut short: no match
    traj, inner = _plan_life(V6, [(0, 1), (3, 1)])
    out = distill.retokenize(traj, inner, V6, S6, mined={(1, 1): 5})
    assert [r[1] for r in out] == [0, 3, 0]
    # mined=None: the pre-E3 rule (no relabel even where inputs match)
    traj, inner = _plan_life(V6, [(3, 2), (2, 1)])
    assert [r[1] for r in distill.retokenize(traj, inner, V6, S6)] == \
        [3, 2, 0]


def test_a_masters_dropped_mined_token_replays_its_own_actions():
    """E3-A8: the master M1 holds MX (kept) and MZ (dropped): MX stays one
    record; MZ's three base steps become (1,4), (1,4), (0,4) records, each
    on its stack, each with its own reward (its last play cut short after
    two)."""
    MZ = ((1, 8), (0, 4))
    master = ((0, 4), (1, 4), (2, 4), MX, MZ)
    student = ((0, 4), (1, 4), (2, 4), MX)
    traj, inner = _plan_life(master, [(4, 3), (3, 3), (4, 2)])
    stats = {}
    out = distill.retokenize(traj, inner, master, student, stats=stats)
    assert stats == {"split": 2}         # both MZ decisions, nothing relabeled
    assert [r[1] for r in out] == [1, 1, 0, 3, 1, 1, 0]
    assert [r[2] for r in out] == [(), (1.0,), (2.0,), (3.0,),
                                   (4.0, 5.0, 6.0), (7.0,), (8.0,)]
    for j in range(3):
        assert out[j][0][-3:].max() == j
    assert out[3][0] is traj[1][0]


class _ActEnv(_ReplayEnv):
    """_ReplayEnv recording every base step's action."""

    def __init__(self, end=None):
        super().__init__(end)
        self.actions = []

    def step(self, a):
        self.actions.append(int(a))
        return super().step(a)


@pytest.mark.parametrize("trial", range(16))
def test_a_student_replaying_relabeled_records_sees_the_same_records(
        trial, monkeypatch):
    """A master (durations plus, from trial 8, two mined tokens of its own,
    one of them dropped) plays through the real play_episode; mined tokens
    are taken from its own runs; retokenize relabels; a student replaying
    those slots under its vocabulary records the same stacks, slots and
    raw reward tuples and sends the env the same action at every base step
    -- life losses inside tokens, the cap and the first record included."""
    master = tuple((a, f) for a in range(3) for f in (4, 8, 16, 32))
    if trial >= 8:
        master += (((0, 8), (2, 4)), ((1, 4), (2, 8), (0, 4)))
    for k, v in (("REWARD_GROWTH", True), ("USE_SEARCH_ACTING", False),
                 ("PAPER_TARGETS", True), ("BIRTH_SEARCH", True),
                 ("PRIOR_ACT_STAGED_TEMP", False), ("ACTING_TEMPERATURE", 1.0),
                 ("PRIOR_ACT_TEMP_SCALE", 1.0), ("TOKEN_LONG_BIAS", 0.0),
                 ("TOKEN_VOCAB", master)):
        monkeypatch.setattr(muzero, k, v)
    for k in ("ACTIONS", "EPISODE_CAP", "INNER_FRAMES"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    rng = np.random.default_rng(100 + trial)
    end = int(rng.integers(20, 200)) if trial % 3 else None
    drop = tuple(t for t in master
                 if (muzero.token_is_mined(t) or t[1] != 4)
                 and rng.random() < 0.4)
    if trial >= 8:
        drop = tuple(set(drop) | {master[-1]})
    kept = tuple(t for t in master if t not in drop)
    ablation.ACTIONS, ablation.EPISODE_CAP = len(master), int(
        rng.integers(40, 300))
    p = rng.dirichlet(np.ones(len(master)))
    ablation.INNER_FRAMES = []
    np.random.seed(trial)
    menv = _ActEnv(end)
    traj, _ret, steps, term = ablation.Run.play_episode(
        _replay_shell(menv, lambda s: p.copy()))
    inner, ablation.INNER_FRAMES = ablation.INNER_FRAMES, None
    distill.check_life(traj, inner)
    # mined tokens from the master's own runs of 2-3 full decisions
    taken = {muzero.token_inputs(t) for t in kept}
    mined = []
    for i in range(0, len(traj) - 3, 3):
        run = [master[int(traj[j][1])] for j in range(i, i + 2 + i % 2)]
        tok = distill.mined_token(run)
        ins = muzero.token_inputs(tok)
        full = all(len(traj[j + 1][2]) == muzero.token_total_frames(
            master[int(traj[j][1])]) // 4 for j in range(i, i + len(run)))
        if (full and muzero.token_total_frames(tok) <= 64 and ins not in taken
                and not any(distill._is_prefix(ins, m) or
                            distill._is_prefix(m, ins) for m in
                            map(muzero.token_inputs, mined))):
            mined.append(tok)
        if len(mined) == 4:
            break
    student = kept + tuple(mined)
    relabel = {muzero.token_inputs(t): len(kept) + j
               for j, t in enumerate(mined)} or None
    stats = {}
    out = distill.retokenize(traj, inner, master, student, mined=relabel,
                             stats=stats)
    if mined:
        assert stats.get("decisions", 0) >= 1       # its own runs relabel
    muzero.TOKEN_VOCAB, ablation.ACTIONS = student, len(student)
    ablation.EPISODE_CAP = steps
    acts, k = [r[1] for r in out[:-1]], [0]

    def fn(stack):
        q = np.zeros(len(student))
        q[acts[k[0]]] = 1.0
        k[0] += 1
        return q

    senv = _ActEnv(end)
    straj, _r, ssteps, sterm = ablation.Run.play_episode(
        _replay_shell(senv, fn))
    assert (len(straj), ssteps, sterm) == (len(out), steps, term)
    for a, b in zip(straj, out):
        assert np.array_equal(a[0], b[0]) and a[1] == b[1]
        assert tuple(a[2]) == tuple(b[2])
    assert senv.actions == menv.actions and len(menv.actions) == steps


# --------------------------------------------------------------------------
# slow, real Ms. Pac-Man (CPU): the first swap with mining, its student,
# and the second swap from that student
# --------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("logged", [T[0]], indirect=True)
def test_a_mined_lineage_from_a_logged_master(logged, monkeypatch, tmp_path):
    header0, _out0, _logs0, lives0 = logged
    master = tmp_path / "weights-iter002.pth"
    for t in (T[1], T[2]):
        monkeypatch.setitem(ablation.ARMS, t, {
            **ablation.ARMS[t], "_distill_offline_steps": 4,
            "_distill_snapshot_every": 2, "SCHEDULE_OFFSET_STEPS": 4,
            "ACT_SEARCH_SIMS": 4})
    # -- the first swap, mined ------------------------------------------
    out = tmp_path / "prep-mine"
    logs = []
    h = distill.prep(master, 2, out, log=logs.append,
                     from_log=tmp_path / "own-play", mine=True,
                     drop_share=0.03)
    doc = json.loads((out / "vocab.json").read_text())
    master_vocab = muzero.canon_vocab(h["master_vocab"])
    vocab = muzero.canon_vocab(h["vocab"])
    mined = ablation.declared_mined(doc)
    assert h["mine"] is True and doc["mined"] == h["mined"]
    assert 1 <= len(mined) <= min(8, len(h["dropped"])) == h["mining"]["cap"]
    assert vocab[len(vocab) - len(mined):] == mined
    assert vocab[:len(vocab) - len(mined)] == muzero.canon_vocab(
        [t for t in h["master_vocab"] if t not in h["dropped"]])
    assert len(vocab) <= len(master_vocab)
    # the entries: spelling -> token, path -> master slots, the score
    for j, e in enumerate(h["mined"]):
        spelled = muzero.canon_vocab(e["spelling"])
        assert distill.mined_token(spelled) == mined[j]
        assert e["path"] == [master_vocab.index(t) for t in spelled]
        assert e["slot"] == len(vocab) - len(mined) + j
        assert e["score"] == pytest.approx(e["count"] * (e["k"] - 1)
                                           * e["consistency"])
        assert e["score"] > 0 and e["frames"] <= 64
    scores = [e["score"] for e in h["mined"]]
    assert scores == sorted(scores, reverse=True)
    # relabels, targets, every base step once
    assert h["relabeled_decisions"] >= 2
    assert sum(h["relabeled"].values()) >= 1
    assert set(h["target_rule"]["moved_mean"]) == {
        muzero.token_key(t) for t in mined}
    assert sum(h["target_rule"]["nonzero_share"].values()) > 0
    hh, set_lives = distill.read_set(out / "set.pkl")
    assert hh == h
    S = len(vocab)
    base_steps = 0
    for lf in set_lives:
        for obs, slot, raw, pol, val in lf["traj"]:
            assert 0 <= slot < S and pol.shape == (S,)
            assert pol.sum() == pytest.approx(1.0)
        base_steps += sum(len(r[2]) for r in lf["traj"])
    assert base_steps == h["base_steps"]
    labelled = sum(1 for lf in set_lives for r in lf["traj"][:-1]
                   if r[1] >= S - len(mined))
    assert labelled == sum(h["relabeled"].values())
    assert any("mined" in line for line in logs)
    # the non-mined prep of the same master and log: no E3 key at all
    assert not {"mine", "mined", "mining", "relabeled", "target_rule",
                "master_mined"} & set(header0)
    # -- the student M1: offline, then online with its mined tokens ------
    s1 = T[1]
    static1 = ablation.ARMS[s1]
    spec1 = ablation.student_arm(s1, out / "vocab.json")
    assert spec1["_mined_tokens"] == mined and spec1["TOKEN_VOCAB"] == vocab
    results = tmp_path / "m1"
    results.mkdir()
    ablation.RESULTS, ablation.TAG = results, "-test"
    own = tmp_path / "m1-own-play"
    log1 = ablation.OwnPlayLog(own)
    with ablation.arm_config(s1) as spec, \
            ablation._run_length_globals(30, 3), ablation.token_actions(), \
            ablation.env_schedule(20000):
        muzero.TRAINING_BATCHES = 8
        ablation.student_gate(spec, out / "set.pkl", True)
        ablation.token_gate(spec, True, True, 0)
        ablation.reward_growth_gate(True, True, 0)
        run = ablation.Run(s1, 0, spec)
        try:
            ablation.distill_prefill(run, set_lives)
            ablation.distill_offline(run, s1, 0, spec, True)
            assert run.trainer.training_steps == 4
            n, played = 0, set()
            for it in (1, 2):
                for _ in range(2):
                    ablation.INNER_FRAMES = []
                    try:
                        traj, _, steps, term = run.play_episode(greedy=False)
                        inner = ablation.INNER_FRAMES
                    finally:
                        ablation.INNER_FRAMES = None
                    distill.check_life(traj, inner)
                    # one raw reward per base step played, mined or not
                    assert steps == sum(len(r[2]) for r in traj[1:])
                    assert all(len(traj[i + 1][2]) <= muzero.token_frames(
                        r[1]) // 4 for i, r in enumerate(traj[:-1]))
                    played |= {int(r[1]) for r in traj[:-1]}
                    n += steps
                    log1.write(it, traj, {"inner": inner,
                                          "real_done": bool(
                                              run.env.was_real_done),
                                          "temperature": 0.5},
                               term, steps, n)
            nets = {k: getattr(run.trainer, k) for k in ablation.NET_NAMES}
            blob = {"arm": s1, "seed": 0, "spec": dict(spec),
                    "training_steps": 4 + 2 * distill.TRAIN_STEPS,
                    "iteration": 2,
                    **{k: v.state_dict() for k, v in nets.items()},
                    "acting_states": {k: nets[k].state_dict()
                                      for k in ("representation",
                                                "prediction")}}
        finally:
            run.env.close()
    m1 = tmp_path / "m1-weights-iter002.pth"
    torch.save(blob, m1)
    # -- the second swap: M1 the master, holding mined tokens ------------
    ablation.ARMS[s1] = static1
    out2 = tmp_path / "prep-mine-2"
    h2 = distill.prep(m1, 2, out2, log=lambda m: None, from_log=own,
                      mine=True, drop_share=0.05)
    assert muzero.canon_vocab(h2["master_vocab"]) == vocab
    assert ablation.declared_mined(h2, "master_mined") == mined
    vocab2 = muzero.canon_vocab(h2["vocab"])
    dropped2 = muzero.canon_vocab(h2["dropped"])
    mined2 = ablation.declared_mined(h2)
    kept2 = vocab2[:len(vocab2) - len(mined2)]
    # M1's mined tokens are tokens: each kept or dropped by the 1% rule
    assert all((t in kept2) != (t in dropped2) for t in mined)
    # every base step once, whatever was split or relabeled
    _, lives2 = distill.read_set(out2 / "set.pkl")
    assert sum(len(r[2]) for lf in lives2 for r in lf["traj"]) == \
        h2["base_steps"]
    # the student M2 applies it against M1's vocabulary
    spec2 = ablation.student_arm(T[2], out2 / "vocab.json")
    assert spec2["_master_vocab"] == vocab
    assert spec2["_master_mined"] == mined
    assert spec2["TOKEN_VOCAB"] == vocab2
    assert spec2.get("_mined_tokens", ()) == mined2
