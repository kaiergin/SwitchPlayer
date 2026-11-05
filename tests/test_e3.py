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
# Test B2's T2's 18 tokens (tests/test_e3a.py's, written out here)
T2_VOCAB = ((0, 4), (1, 4), (2, 4), (3, 4), (3, 16), (3, 32), (4, 4), (4, 32),
            (5, 4), (5, 16), (5, 32), (6, 4), (7, 4), (7, 32), (8, 4), (8, 8),
            (8, 16), (8, 32))
# E3's arms (part 2): T1 / T2 but for _mine and M2's master
MINE_ARM = ("effzero-repro-faithful-prioract-noredo-temp05-bnrecal-"
            "mspacman-tokens-mined")
MARMS = [MINE_ARM + "-m1", MINE_ARM + "-m2"]


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
    assert muzero.canon_token(np.array([[1, 8], [3, 4]])) == M1
    assert muzero.canon_token((np.int64(2), np.float64(8.0))) == (2, 8)
    for bad in ("ab", [], [1, 2, 3], [[1, 8], 3], [[1, 8], [[2, 4]]], 5,
                [[1, 2, 3]], [[]],
                # (audit r1a) a 0-d array is a number, not a step; frames
                # and actions are integral numbers only
                np.array(5), [np.array(1), np.array(8)][0:1],
                [1, 8.9], [[1, 8], [3, 4.5]], [True, 4], ["1", "8"],
                [1.5, 8]):
        with pytest.raises((TypeError, ValueError)):
            muzero.canon_token(bad)
    assert not muzero._token_seq(np.array(5))
    assert muzero._token_seq(np.array([5]))
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


def test_the_bias_caps_a_mined_token_at_the_constant_32_frames(monkeypatch):
    """E3-A7 as written (E3-A13): the cap is the constant 32 frames."""
    assert muzero.TOKEN_BIAS_CAP_FRAMES == 32
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (0, 32), M1, BIG))
    w = ablation.token_bias_weights(0.5)
    # M1 (12 frames) uncapped; BIG (64) at 32 frames
    assert w.tolist() == [1.0, (32 / 4) ** 0.5, (12 / 4) ** 0.5,
                          (32 / 4) ** 0.5]
    assert w[3] < (64 / 4) ** 0.5
    # the vocabulary's longest duration token (16) and its longest other
    # token of either kind (M2, 20 frames) do not move the cap; a duration
    # token past 32 frames is never capped
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (0, 16), M2, BIG,
                                                (1, 48)))
    w = ablation.token_bias_weights(0.5)
    assert w.tolist() == [1.0, (16 / 4) ** 0.5, (20 / 4) ** 0.5,
                          (32 / 4) ** 0.5, (48 / 4) ** 0.5]


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


def test_steps_scale_both_support_bounds(monkeypatch):
    """(audit r1a) The two bounds apart: a value-prefix support for 1,000
    (a record of c x steps, UNROLL_STEPS of them, must fit) -- then a reward
    support for 100 -- so `steps` applied to only one bound fails."""
    sup = muzero.scalar_categorical.support_for
    monkeypatch.setattr(muzero, "VALUE_PREFIX_SUPPORT", sup(1000.0))
    # c 15 x 16 steps = 240 per record (under the reward's 400) but x 5 =
    # 1,200 per prefix (past 1,000)
    assert not ablation.ceiling_fits_supports(15.0, 16)
    assert ablation.ceiling_fits_supports(15.0, 1)     # 75 per prefix
    assert ablation.ceiling_fits_supports(12.5, 16)    # 1,000 exactly
    monkeypatch.setattr(muzero, "VALUE_PREFIX_SUPPORT", sup(2000.0))
    monkeypatch.setattr(muzero, "REWARD_SUPPORT", sup(100.0))
    # c 10 x 16 = 160 per record (past 100) though x 5 = 800 fits 2,000
    assert not ablation.ceiling_fits_supports(10.0, 16)
    assert ablation.ceiling_fits_supports(10.0, 1)
    assert ablation.ceiling_fits_supports(6.25, 16)    # 100 exactly


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
            master_mined=None, c=7.26, set_path=None, name="vocab.json",
            **extra):
    doc = {"vocab": J(vocab), "student_ceiling": c, "master_arm": master_arm,
           "master_iterations": 47, "student_discount": G,
           "discount_rule": "keep", "master_discount": G,
           "mined": [{"token": muzero.token_json(t)} for t in mined],
           **extra}
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


def test_the_duration_token_probes_refuse_a_mined_vocabulary():
    """(audit r1a) E0's census and the E1 / E3a probes built on it read
    (action, frames) tokens only: their shared snapshot check refuses a
    mined vocabulary by name."""
    import inspect
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                           "benchmarks" / "probes"))
    import probe_action_census as pc
    import probe_e1_equivalence as e1
    import probe_e3a_sequences as e3
    with pytest.raises(SystemExit, match=r"holds mined tokens \(E3\)"):
        pc.check_snapshot(TK_VOCAB + (M1,), pc.ENV_ID)
    pc.check_snapshot(TK_VOCAB, pc.ENV_ID)            # the grid, as ever
    with pytest.raises(SystemExit, match="outside the 36-token grid"):
        pc.check_snapshot(((0, 12),), pc.ENV_ID)
    for mod in (e1, e3):            # each refuses before anything plays
        src = inspect.getsource(mod.main)
        assert src.index("pc.check_snapshot(vocab, pe.ENV)") < src.index(
            "ablation.Run(")


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
    # (audit r1b, memory) only the counted runs' start / end latents kept
    need = {(li, x) for c in cands for li, i, n in c["occurrences"]
            for x in (i, i + n)}
    assert rec["latents_kept"] == len(need) < 12 + 2
    json.dumps(rec)                                   # vocab.json-ready
    # no slot freed: nothing computed (the run is never touched)
    assert distill.mine_tokens(None, lives, V, V, 0, 1.0, 100.0, 1.0) == (
        [], {"cap": 0, "candidates": 0, "skipped": "no slot freed"})


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


V12 = ((0, 4), (1, 4), (2, 4), (1, 8), (1, 12))
MH = ((1, 8), (1, 4))                       # inputs 1, 1, 1: (1, 12)'s


def _slot_life(plan):
    """A life of (slot, base steps played) decisions over V12 (stacks
    irrelevant here)."""
    traj = [(None, plan[0][0], ())]
    for i, (slot, m) in enumerate(plan):
        nxt = plan[i + 1][0] if i + 1 < len(plan) else 0
        traj.append((None, nxt, (1.0,) * m))
    return {"traj": traj}


def test_target_spellings_are_every_logged_path_and_the_masters_token():
    """E3-A13: MH's spellings -- the master's own (1, 12) (one token, first,
    whether or not it was played), then every run of the log whose inputs
    are (1, 1, 1), any start, every decision played in full."""
    lives = [_slot_life([(0, 1), (3, 2), (1, 1), (2, 1)]),     # (1,8)(1,4)
             _slot_life([(1, 1), (3, 2), (0, 1)]),             # (1,4)(1,8)
             _slot_life([(1, 1), (1, 1), (1, 1), (1, 1)]),     # x2: 0, 1
             _slot_life([(3, 2), (1, 1)]),                     # (1,8)(1,4)
             _slot_life([(4, 3)]),                             # (1,12)
             _slot_life([(3, 2), (1, 0)])]                     # cut short
    sp = distill.target_spellings(lives, V12, [MH, ((2, 4), (0, 4))])
    assert sp[0] == [((4,), 1), ((3, 1), 2), ((1, 3), 1), ((1, 1, 1), 2)]
    assert sp[1] == []                          # (2, 0) never played
    # the master's own token counts even when the log never played it
    sp = distill.target_spellings(lives[:4], V12, [MH])
    assert sp[0][0] == ((4,), 0)
    # never a slot-prefix of another spelling of the same inputs
    paths = [p for p, _n in sp[0]]
    assert not any(a != b and b[:len(a)] == a for a in paths for b in paths)
    assert distill.target_spellings(lives, V12, []) == []


def test_a_mined_token_sums_its_spellings_one_token_included():
    """E3-A13 by hand: MH's P = q(1,12) (one token, whole) + q(1,8) x w1 +
    q(1,4) x w2, each moved from its own first slot; (1, 12) dropped."""
    student = ((0, 4), (1, 4), (2, 4), (1, 8), MH)
    q = np.array([0.1, 0.3, 0.1, 0.2, 0.3])
    out, fb, gone = distill.masked_targets(
        [q], V12, student, mined=[[4, 3, 1]], weights=[[1.0, 0.5, 0.25]])
    mass = 0.3 + 0.2 * 0.5 + 0.3 * 0.25
    want = np.array([0.1, 0.3 - 0.075, 0.1, 0.2 - 0.1, mass])
    assert out[0] == pytest.approx(want / want.sum())
    assert gone[0] == pytest.approx(0.0)        # (1, 12)'s q all moved
    # the most-played spelling alone would credit only q(1,8) x w1
    alone, _, _ = distill.masked_targets([q], V12, student, mined=[[3]],
                                         weights=[[0.5]])
    assert alone[0][4] < out[0][4]


def test_overlapping_paths_fail_loudly():
    student = ((0, 4), (1, 4), (2, 4), (1, 8), MH, ((1, 4), (2, 4)))
    q = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    # two paths from slot 1 whose products sum past 1: not disjoint
    with pytest.raises(AssertionError):
        distill.masked_targets([q], V12, student, mined=[[1], [1]],
                               weights=[[0.7, 0.4]])
    # within the products' float tolerance but a slot driven below 0
    with pytest.raises(AssertionError):
        distill.masked_targets([q], V12, student, mined=[[1], [1]],
                               weights=[[0.5, 0.5 + 5e-7]])
    out, _, _ = distill.masked_targets([q], V12, student, mined=[[1], [1]],
                                       weights=[[0.5, 0.5]])
    assert out[0] == pytest.approx([0, 0, 0, 0, 0.5, 0.5])
    with pytest.raises(ValueError, match="path products"):
        distill.masked_targets([q], V12, student, mined=[[1], [1]],
                               weights=[[0.5]])


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
    assert stats == {"records": {5: 2}, "decisions": 5, "split": 1,
                     "split_extra": 3}
    # every raw reward and base step exactly once
    assert sum(sum(r[2]) for r in out) == sum(sum(r[2]) for r in traj)
    assert sum(len(r[2]) for r in out) == sum(len(r[2]) for r in traj)


def test_relabeling_takes_the_longest_and_only_whole_decisions():
    traj, inner = _plan_life(V6, [(3, 2), (2, 1), (1, 1)])
    # a VALID student vocabulary (audit r1b): (1, 8) and (0, 16) dropped,
    # MX and MK appended -- MK's inputs (1, 1) are the dropped (1, 8)'s
    MK = ((1, 4), (1, 4))
    student = ((0, 4), (1, 4), (2, 4), (2, 8), MX, MK)
    assert ablation.validate_student_vocab(student, V6, mined=(MX, MK),
                                           max_mined=8) == student
    # two mined keys from one boundary (not prefix-free here: the
    # mechanics): the longest
    out = distill.retokenize(traj, inner, V6, student,
                             mined={(1, 1): 5, (1, 1, 2): 4})
    assert [r[1] for r in out] == [4, 1, 0]
    # the k = 1 relabel: the master's dropped (1, 8) decision IS MK's inputs
    stats = {}
    out = distill.retokenize(traj, inner, V6, student, mined={(1, 1): 5},
                             stats=stats)
    assert [r[1] for r in out] == [5, 2, 1, 0]
    assert [r[2] for r in out] == [(), (1.0, 2.0), (3.0,), (4.0,)]
    assert stats == {"records": {5: 1}, "decisions": 1}
    # without MK the dropped (1, 8) is split into two (1, 4) records
    stats = {}
    out = distill.retokenize(traj, inner, V6, student[:5], stats=stats)
    assert [r[1] for r in out] == [1, 1, 2, 1, 0]
    assert stats == {"split": 1, "split_extra": 1}
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
    # both MZ decisions, nothing relabeled: 3 - 1 and 2 - 1 extra records
    assert stats == {"split": 2, "split_extra": 3}
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
# the prep's core on fakes (audit r1b): distill_lives' wiring -- mining on
# the ACTING states with the master's clip and per-base-step discount, the
# path products on the RAW states at the re-tokenized records, the tail's
# order, every record's target per slot
# --------------------------------------------------------------------------
VP = ((0, 4), (1, 4), (2, 4), (1, 8), (0, 16), (2, 8))
PA = ((1, 8), (2, 4))           # also spelled (1, 4) (1, 4) (2, 4)
PB = ((0, 16), (2, 4))          # through the dropped (0, 16)
NS = 64                         # the fakes' latent states


class _SRep(nn.Module):
    """The latent: one-hot of the stack's newest frame's value."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mode", torch.zeros(()))

    def forward(self, x):
        s = (x[:, -1, 0, 0] * 255.0).round().long()
        return torch.nn.functional.one_hot(s, NS).float()


class _SPred(nn.Module):
    """Prior P[mode][s], value V[mode][s]: mode 0 the RAW states, 1 the
    ACTING ones (its buffer, which the states load)."""

    def __init__(self, P, V):
        super().__init__()
        self.register_buffer("mode", torch.zeros(()))
        self.P = torch.as_tensor(P, dtype=torch.float32)
        self.V = torch.as_tensor(V, dtype=torch.float32)

    def forward(self, z):
        s, m = z.argmax(1), int(self.mode.item())
        return torch.log(self.P[m][s]), self.V[m][s].unsqueeze(1), None


class _SDyn(nn.Module):
    def __init__(self, Tr):
        super().__init__()
        self.Tr = torch.as_tensor(Tr)

    def forward(self, z, a):
        return (torch.nn.functional.one_hot(self.Tr[z.argmax(1), a],
                                            NS).float(), None)


class _SRun:
    """What distill_lives reads of a Run: the trainer's nets, set_train_mode
    and a birth search whose policy is Q[the stack's state] (uniform on a
    terminated tail, as Run._birth_search's)."""

    def __init__(self, rep, dyn, pred, Q):
        self.trainer = SimpleNamespace(representation=rep, dynamics=dyn,
                                       prediction=pred,
                                       device=torch.device("cpu"))
        self.Q = Q

    def set_train_mode(self, training):
        for n in (self.trainer.representation, self.trainer.dynamics,
                  self.trainer.prediction):
            n.train(training)

    def _birth_search(self, traj, terminated):
        out = []
        for i, r in enumerate(traj):
            s = int(r[0][-1, 0, 0])
            tail = terminated and i == len(traj) - 1
            pol = np.ones(len(VP)) / len(VP) if tail else self.Q[s]
            out.append((r[0], r[1], r[2], pol, float(s)))
        return out


def _fake_lives():
    """10 lives over VP, every base step's frame a fresh state: PA spelled
    (1,8)(2,4) four times and (1,4)(1,4)(2,4) twice, PB three times (its
    (2,4) pays 200: 20 units, past the master's ceiling 1), then (0,4) and
    the dropped (2,8)."""
    ids = iter(range(1, NS))

    def frame(v):
        return np.full((3, 1, 1), v, np.uint8)

    def life(plan):
        s = muzero.initial_frame_stack(frame(next(ids)))
        traj, inner, incoming = [], [], ()
        for slot, rewards in plan:
            traj.append((s, slot, incoming))
            pushed = []
            for j in range(len(rewards)):
                f = frame(next(ids))
                if j < len(rewards) - 1:
                    pushed.append(f)
                s = muzero.push_frame(s, f)
            inner.append(pushed)
            incoming = tuple(float(r) for r in rewards)
        traj.append((s, 0, incoming))
        distill.check_life(traj, inner)
        return {"traj": traj, "inner": inner, "terminated": True,
                "game_end": False}

    return ([life([(3, (10, 10)), (2, (10,))]) for _ in range(4)]
            + [life([(1, (10,)), (1, (10,)), (2, (10,))]) for _ in range(2)]
            + [life([(4, (10,) * 4), (2, (200,))]) for _ in range(3)]
            + [life([(0, (10,)), (5, (10, 10))])])


def test_the_preps_core_wires_states_clip_discount_tail_and_records():
    rng = np.random.default_rng(5)
    Q = rng.dirichlet(np.ones(len(VP)), NS)
    P = np.stack([rng.dirichlet(np.ones(len(VP)), NS) for _ in range(2)])
    Tr = rng.integers(0, NS, (NS, len(VP)))
    V = np.zeros((2, NS))
    V[0] = 2.0                  # the RAW values; the acting ones are 0
    run = _SRun(_SRep(), _SDyn(Tr), _SPred(P, V), Q)
    states = {m: {"representation": {"mode": torch.tensor(float(m))},
                  "prediction": {"mode": torch.tensor(float(m))}}
              for m in (0, 1)}
    lives = _fake_lives()
    stacks = [[r[0] for r in lf["traj"]] for lf in lives]
    clip = {"reward_unit": 10.0, "master_ceiling": 1.0,
            "student_ceiling": 5.0}
    logs = []
    got = distill.distill_lives(run, lives, VP, states[0], states[1], clip,
                                0.9, drop_share=0.15, mine=True,
                                log=logs.append)
    # the 1% rule (at 15% here) and the appended tail, in score order
    assert got["dropped"] == ((0, 16), (2, 8))
    assert got["vocab"] == ((0, 4), (1, 4), (2, 4), (1, 8), PA, PB)
    assert [c["token"] for c in got["chosen"]] == [PA, PB]
    assert got["chosen"][0]["target_spellings"] == [((3, 2), 4),
                                                    ((1, 1, 2), 2)]
    assert got["chosen"][1]["target_spellings"] == [((4, 2), 3)]
    # mining read the ACTING values (0) and the master's clip (1 unit) and
    # discount (0.9) per BASE step -- by hand, in log order
    d = 0.9
    r3 = 1 + d + d * d
    rets = ([r3, 1.0] * 4 + [r3, 1 + d, 1.0] * 2
            + [1 + d + d ** 2 + d ** 3 + d ** 4, 1.0] * 3 + [r3, 1 + d])
    q = np.percentile(rets, [10, 50, 90])
    want = {"mean": round(float(np.mean(rets)), 4),
            "p10": round(float(q[0]), 4), "p50": round(float(q[1]), 4),
            "p90": round(float(q[2]), 4)}
    m = got["mining"]
    assert m["returns"] == want and m["advantage"] == want
    assert m["values"] == {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    assert m["high"] == 11 and m["cap"] == 2
    assert m["value_terms"] == {"reward_unit": 10.0, "ceiling": 1.0,
                                "discount_per_base_step": 0.9}
    # the records: relabeled (PA, PB) and not (the last life's)
    want_slots = [[4, 0]] * 6 + [[5, 0]] * 3 + [[0, 2, 2, 0]]
    out = got["out_lives"]
    assert [[r[1] for r in lf["traj"]] for lf in out] == want_slots
    assert [r[2] for r in out[0]["traj"]] == [(), (10.0, 10.0, 10.0)]
    assert [r[2] for r in out[6]["traj"]] == [(), (10.0,) * 4 + (200.0,)]
    assert [r[2] for r in out[9]["traj"]] == [(), (10.0,), (10.0,), (10.0,)]
    assert out[0]["traj"][0][0] is stacks[0][0]
    assert out[0]["traj"][1][0] is stacks[0][2]          # the interior left

    # every record's target by hand: q at its state (uniform on the
    # terminated tail), each spelling's path product on the RAW prior
    def w(path, s):
        z, prod = Tr[s, path[0]], 1.0
        for t in path[1:]:
            prod *= P[0][z][t]
            z = Tr[z, t]
        return prod

    def target(q, s):
        rest, mass = np.array(q, dtype=np.float64), [0.0, 0.0]
        for j, paths in enumerate(([(3, 2), (1, 1, 2)], [(4, 2)])):
            for p in paths:
                moved = q[p[0]] * w(p, s)
                rest[p[0]] -= moved
                mass[j] += moved
        st = np.array([rest[0], rest[1], rest[2], rest[3], *mass])
        return st / st.sum()

    n, moved_32 = 0, 0.0
    for lf in out:
        for i, (obs, slot, raw, pol, val) in enumerate(lf["traj"]):
            s = int(obs[-1, 0, 0])
            tail = i == len(lf["traj"]) - 1
            qq = np.ones(len(VP)) / len(VP) if tail else Q[s]
            assert pol == pytest.approx(target(qq, s), rel=1e-5, abs=1e-7)
            if not tail:
                n += 1
                moved_32 += qq[3] * w((3, 2), s)
    assert n == got["trule"]["records"] == 12
    head = distill.mined_header(got)
    sp = head["target_rule"]["spellings"][muzero.token_key(PA)]
    assert [(x["path"], x["logged"]) for x in sp] == [([3, 2], 4),
                                                      ([1, 1, 2], 2)]
    assert sp[0]["moved_mean"] == pytest.approx(moved_32 / n, abs=2e-6)
    assert head["relabeled"] == {muzero.token_key(PA): 6,
                                 muzero.token_key(PB): 3}
    assert head["relabeled_decisions"] == 20
    assert head["mined"][0]["target_spellings"] == [
        {"path": [3, 2], "logged": 4}, {"path": [1, 1, 2], "logged": 2}]
    # (audit r1b NITs) the splits' extra records apart from the relabels'
    assert (got["splits"], got["split_extra"], got["net_extra"]) == (1, 1,
                                                                     -10)
    assert head["net_extra_records"] == -10
    line = [x for x in logs if x.startswith("re-tokenized")][0]
    assert "20 decisions relabeled as 9 mined records) [" in line
    assert ") ;" not in line
    json.dumps(head)


def test_the_preps_support_check_runs_without_mining_too():
    """(audit r1b) Without --mine, a master's KEPT mined token (16 base
    steps) still bounds the student's ceiling; once it is dropped, the
    vocabulary is duration-only and the check the per-base-step one, as
    ever."""
    rng = np.random.default_rng(1)
    Q7 = rng.dirichlet(np.ones(len(VP) + 1), NS)
    run = _SRun(_SRep(), _SDyn(rng.integers(0, NS, (NS, len(VP) + 1))),
                _SPred(np.stack([Q7, Q7]), np.zeros((2, NS))), Q7)
    run._birth_search = lambda traj, term: [
        (r[0], r[1], r[2], Q7[int(r[0][-1, 0, 0])], 0.0) for r in traj]
    master = VP + (BIG,)          # a 16-step mined token the master holds
    states = {"representation": {"mode": torch.tensor(0.0)},
              "prediction": {"mode": torch.tensor(0.0)}}

    def go(c, share):
        return distill.distill_lives(
            run, _fake_lives(), master, states, states,
            {"reward_unit": 10.0, "master_ceiling": 1.0,
             "student_ceiling": c}, 0.9, drop_share=share,
            log=lambda m: None)

    with pytest.raises(SystemExit, match="record of 16 base steps"):
        go(30.0, 0.0)                   # BIG kept: 30 x 16 = 480 per record
    assert BIG in go(20.0, 0.0)["vocab"]                # 320 fits
    assert BIG not in go(30.0, 0.01)["vocab"]           # never played


# --------------------------------------------------------------------------
# E3's arms (part 2): M1 / M2 -- Test B2's T1 / T2 but for _mine (the prep's
# --mine) and M2's master M1
# --------------------------------------------------------------------------
@pytest.fixture
def marms(arms, monkeypatch):
    """ARMS restored (T's, by `arms`; M1 / M2's here) after a test applies
    --token-vocab."""
    for arm in MARMS:
        monkeypatch.setitem(ablation.ARMS, arm, ablation.ARMS[arm])


def test_the_mined_arms_are_t1_and_t2_but_for_mining():
    assert ablation.MINE_ARM == MINE_ARM
    for k, arm in ((1, MARMS[0]), (2, MARMS[1])):
        spec, t = ablation.ARMS[arm], ablation.ARMS[T[k]]
        delta = {x for x in set(spec) | set(t) if spec.get(x) != t.get(x)}
        # M1's master is T1's (T0); M2's is M1, not T1
        assert delta == ({"_mine", "_doc"} if k == 1
                         else {"_mine", "_master_arm", "_doc"}), arm
        assert spec["_mine"] is True and "_mine" not in t
    assert ablation.ARMS[MARMS[0]]["_master_arm"] == T[0]
    assert ablation.ARMS[MARMS[1]]["_master_arm"] == MARMS[0]
    # T0 is generation 0 of both lineages (E3-A1): no M0 arm; no other arm
    # mines
    assert MINE_ARM + "-m0" not in ablation.ARMS
    assert {n for n, s in ablation.ARMS.items()
            if s.get("_mine")} == set(MARMS)
    with ablation.arm_config(MARMS[0]):
        assert muzero.DISCOUNT == G              # the T arms' constant


def test_a_mined_arm_takes_only_a_mined_prep(tmp_path, marms):
    m1 = MARMS[0]
    static = ablation.ARMS[m1]
    for mine in (None, False, 1, "true"):
        extra = {} if mine is None else {"mine": mine}
        with pytest.raises(SystemExit, match="not a mined prep"):
            ablation.student_arm(m1, _e3_doc(tmp_path, M1_VOCAB, (M1, M2),
                                             **extra))
        ablation.ARMS[m1] = static
    spec = ablation.student_arm(m1, _e3_doc(tmp_path, M1_VOCAB, (M1, M2),
                                            mine=True))
    assert spec["TOKEN_VOCAB"] == M1_VOCAB and spec["_mine"] is True
    assert spec["_mined_tokens"] == (M1, M2)
    # a T arm (no _mine) applies either (part 1's rule, unchanged)
    ablation.student_arm(T[1], _e3_doc(tmp_path, M1_VOCAB, (M1, M2)))
    # M2: a mined prep of M1, a student holding mined tokens
    m2 = ablation.student_arm(MARMS[1], _e3_doc(
        tmp_path, M2_VOCAB, (M3,), master_arm=m1, master_vocab=M1_VOCAB,
        master_mined=(M1, M2), mine=True))
    assert m2["_master_vocab"] == M1_VOCAB and m2["_mine"] is True
    assert m2["_master_mined"] == (M1, M2) and m2["_mined_tokens"] == (M3,)


def test_the_master_spec_of_a_mined_arm():
    """M2's prep reads M1's snapshot: its spec is M1's arm with the student
    keys (its mined tail declared) -- and M1's arm's _mine."""
    m1 = _student_spec(MARMS[0], M1_VOCAB, _mined_tokens=(M1, M2))
    assert distill.master_spec(MARMS[0], {"spec": m1}, "m") == m1
    assert distill.discount_rule(m1) == "keep"
    with pytest.raises(SystemExit, match="beyond the student keys"):
        distill.master_spec(MARMS[0], {"spec": {**m1, "_mine": False}}, "m")


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
            # (audit r1a) a FORCED prior, seeded: the first mined slot
            # holds 0.3 of it (it is played, deterministically), the other
            # mined slots 0 (never played: the second swap drops them),
            # the kept tokens the rest
            first = len(vocab) - len(mined)
            forced = np.full(len(vocab), 0.7 / first)
            forced[first:] = 0.0
            forced[first] = 0.3
            run.searcher.network_manager.initial_inference = (
                lambda stack: (forced.copy(), 0.0))
            np.random.seed(11)
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
            # a mined slot WAS played, through play_token's per-step inputs
            assert first in played
            assert not played & set(range(first + 1, len(vocab)))
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
    # M1's mined tokens are tokens: each kept or dropped by the 1% rule --
    # the one it played kept, the ones it never played dropped
    assert all((t in kept2) != (t in dropped2) for t in mined)
    assert mined[0] in kept2
    assert all(t in dropped2 for t in mined[1:])
    assert h2["shares"][muzero.token_key(mined[0])] >= 0.05
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
