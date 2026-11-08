"""Test O stage 3 (CURRICULUM.md "Test O": DESIGN "Head targets", "Jump
training", "Prefix BN"; DEPARTURES 1, 3, 4, 6, 7): the option arms' records,
the jump unroll, the head targets and the loss, and the birth search on the
option engine.

Expected numbers are worked BY HAND from the registration and OptionZero's
code (rlglab/optionzero iclr2025 a383d66): base_env.h getStepLength 306-316,
getOptionPolicy 318-372, getOptionLossScale 374-416, getMaxActionID 419-425;
data_loader.cpp setMuZeroTrainingData 157-213, updateMax 282-298; train.py
calculate_loss 158-165, 223 / 244 / 274 / 275; atari.cpp getReward 279-290.
They are never read off the implementation.

FLAG OFF: two full O0 batches and train steps (batch arrays, loss, every
gradient, parameters, BN buffers, optimizer state, priorities, RNG streams)
and an O0 life's birth search and records are pinned bit for bit against
the build tree's code before this stage (git show e11fd5f8: muzero.py,
networks.py, experiments/ablation.py), loaded from git the way
test_options_net.py loads networks.py -- the pins FAIL, never skip, when
git cannot read those files. CPU only."""
import asyncio
import copy
import importlib.util
import math
import pathlib
import random
import re
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import ablation  # noqa: E402
import muzero  # noqa: E402
import muzero_option_targets as mot  # noqa: E402
import muzero_vtree  # noqa: E402
import muzero_vtree_options  # noqa: E402
from tests.test_prioract2 import restore_rngs  # noqa: E402,F401
from tests.test_testc import make_run  # noqa: E402

PRE_S3 = "e11fd5f8"          # the build tree's head before stage 3
O0 = ablation.PLAIN_ARM + "-o0"
A9 = 9
CPU = torch.device("cpu")
A, L = 4, 3                  # the hand cases: 4 moves (stop = class 4), L 3
STOP = A


# ==========================================================================
# fixtures and helpers
# ==========================================================================
@pytest.fixture
def o0(restore_rngs):  # noqa: F811
    """Inside O0's config at Ms. Pac-Man's 9 moves on the CPU (as
    test_testo's in_arm); yields the spec. Request it BEFORE monkeypatch, so
    a test's patches are undone before the arm is left."""
    saved = {k: getattr(ablation, k)
             for k in ("ACTIONS", "ENV_NAME", "FRAMESKIP", "DEVICE")}
    ablation.ACTIONS, ablation.ENV_NAME = A9, "MsPacmanNoFrameskip-v4"
    ablation.FRAMESKIP, ablation.DEVICE = 4, CPU
    try:
        with ablation.arm_config(O0) as spec, ablation.token_actions():
            yield spec
    finally:
        for k, v in saved.items():
            setattr(ablation, k, v)


def opt(*moves, length=L):
    return tuple(moves) + (-1,) * (length - len(moves))


def hand_life(actions, pis=None, pmax=None, options=None, rewards=None,
              terminated=True, closing_action=0, closing_pi=None, A_=A,
              L_=L):
    """A life's records as paper_targets stores them: the acted ones, then
    the closing record (placeholder action). pmax[t] None: no field."""
    n = len(actions)
    recs = []
    for t in range(n + 1):
        acted = t < n
        if acted:
            pi = pis[t] if pis is not None else np.full(A_, 1.0 / A_)
        else:
            pi = closing_pi if closing_pi is not None else np.full(A_, 1.0 / A_)
        r = {"obs": np.full((1,), t, dtype=np.uint8),
             "action": actions[t] if acted else closing_action,
             "reward": float(rewards[t]) if rewards is not None else 0.0,
             "target_policy": np.asarray(pi, dtype=np.float64),
             "root_value": 0.0, "target_value": 0.0,
             "has_policy_target": acted or not terminated,
             "born_step": 0, "priority": 1.0}
        if acted:
            if pmax is not None and pmax[t] is not None:
                r["pmax"] = pmax[t]
            if options and t in options:
                r["root_option"] = opt(*options[t], length=L_)
        recs.append(r)
    recs[-1]["truncated"] = not terminated
    return recs


def targets_at(life, pos, L_=L, A_=A):
    tgt = np.full((L_ - 1, A_ + 1), np.nan)
    scale = np.full(L_ - 1, np.nan)
    mot.head_targets(life, pos, L_, A_, tgt, scale)
    return tgt, scale


def head(**mass):
    """One head's target row: {move: mass, "stop": mass}."""
    row = np.zeros(A + 1)
    for k, v in mass.items():
        row[STOP if k == "stop" else int(k[1:])] = v
    return row


STOP_ROW = head(stop=1.0)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def rng_states():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state())


def same_rng(a, b):
    return (a[0] == b[0] and all(np.array_equal(x, y) if isinstance(
        x, np.ndarray) else x == y for x, y in zip(a[1], b[1]))
        and torch.equal(a[2], b[2]))


def stub_trainer(L_=L, steps=0):
    """A TrainingHelper with no nets but the prediction's option length
    (enough for the batch builder without the target model)."""
    tr = object.__new__(muzero.TrainingHelper)
    tr.prediction = types.SimpleNamespace(option_length=L_)
    tr.training_steps = steps
    tr.device = CPU
    return tr


@pytest.fixture
def plain_knobs(monkeypatch):
    """Batch-builder knobs for hand numbers: stored bootstraps (no target
    model), TD targets, discount 0.5, 5 steps, no age shrink, K 5."""
    for k, v in (("USE_TARGET_MODEL", False), ("VALUE_TARGET_MODE", "td"),
                 ("DISCOUNT", 0.5), ("TD_STEPS", 5),
                 ("OFFPOLICY_DECAY_STEPS", 0), ("TOKEN_VOCAB", None),
                 ("PER_ENABLED", True), ("UNROLL_STEPS", 5)):
        monkeypatch.setattr(muzero, k, v)
    return monkeypatch


def build_batch(tr, replay, B, A_=A):
    return tr.finish_batch_paper(tr.assemble_batch_paper(replay, B, 5, A_))


def row_of(batch, rec):
    rows = [i for i, r in enumerate(batch["sample_refs"]) if r is rec]
    assert len(rows) == 1
    return rows[0]


def at(life, rec):
    """The record's position in its life (by identity)."""
    hits = [i for i, r in enumerate(life) if r is rec]
    assert len(hits) == 1
    return hits[0]


# ==========================================================================
# getStepLength (base_env.h 306-316), first-move mode
# ==========================================================================
def test_step_length_match_partial_and_first_move():
    #               0  1  2  3  4  | closing (placeholder 0)
    played = [1, 2, 3, 0, 1]
    assert mot.step_length(hand_life(played, options={0: (1, 2, 3)}), 0) == 3
    assert mot.step_length(hand_life(played, options={0: (1, 2)}), 0) == 2
    # partial: the third move was not played -> one step, not two
    assert mot.step_length(hand_life(played, options={0: (1, 2, 0)}), 0) == 1
    # the first move itself was not played
    assert mot.step_length(hand_life(played, options={0: (2, 2)}), 0) == 1
    # mid-life
    assert mot.step_length(hand_life(played, options={2: (3, 0, 1)}), 2) == 3


def test_step_length_without_a_suggestion_is_one():
    life = hand_life([1, 2, 3], options={1: ()})
    assert life[1]["root_option"] == (-1, -1, -1)
    assert [mot.step_length(life, t) for t in range(5)] == [1] * 5
    del life[1]["root_option"]
    assert mot.step_length(life, 1) == 1


def test_a_match_stops_before_the_closing_record():
    # option (0, 1, 0) at 3: positions 3, 4 played 0, 1, and the closing
    # record's PLACEHOLDER action is 0 too -- it must not complete the match
    played = [1, 2, 3, 0, 1]
    life = hand_life(played, options={3: (0, 1, 0)}, closing_action=0)
    assert mot.step_length(life, 3) == 1
    # a length-2 option ending on the last acted position lands ON the
    # closing record: allowed
    life = hand_life(played, options={3: (0, 1)})
    assert mot.step_length(life, 3) == 2
    # the last acted position: its second move would be the placeholder
    life = hand_life(played, options={4: (1, 0)}, closing_action=0)
    assert mot.step_length(life, 4) == 1
    # a cut life's closing record (a real search) is no different
    life = hand_life(played, options={3: (0, 1, 0)}, terminated=False)
    assert mot.step_length(life, 3) == 1


def test_step_length_at_and_past_the_closing_record_is_one():
    life = hand_life([1, 2], options={0: (1, 2)})
    life[2]["root_option"] = opt(0, 0)      # even if the closing one had one
    assert mot.step_length(life, 2) == 1
    assert mot.step_length(life, 7) == 1    # padding


# ==========================================================================
# the window (setMuZeroTrainingData: pos += getStepLength(pos))
# ==========================================================================
#                0  1  2  3  4  5  6 | closing 7 (placeholder 0)
WIN_PLAYED = [1, 2, 3, 0, 1, 2, 2]
WIN_OPTIONS = {0: (1, 2, 3), 3: (0, 1), 5: (2, 2, 2), 6: (2, 0)}


def win_life(terminated=True):
    return hand_life(WIN_PLAYED, options=WIN_OPTIONS,
                     pmax=list(WIN_PLAYED), rewards=list(range(8)),
                     terminated=terminated)


def test_window_positions_and_spans_by_hand():
    # 0 -(1,2,3)-> 3 -(0,1)-> 5; at 5 (2,2,2) needs the closing record's
    # placeholder -> 1 step; at 6 (2,0) the same -> 1; 7 is the closing
    # record (1 step on, into padding)
    pos, spans = mot.window(win_life(), 0, 5)
    assert pos == [0, 3, 5, 6, 7, 8]
    assert spans == [3, 2, 1, 1, 1]
    pos, spans = mot.window(win_life(), 4, 5)
    assert pos == [4, 5, 6, 7, 8, 9] and spans == [1] * 5
    assert mot.span_of(win_life(), 0, 5) == 8


def _trunc_life():
    # a CUT life of 11 acted positions (len 12); a length-3 option played at
    # 4 and a length-2 one at 6
    played = [0, 1, 2, 3, 1, 1, 2, 2, 0, 1, 3]
    return hand_life(played, options={4: (1, 1, 2), 6: (2, 2)},
                     pmax=list(played), terminated=False)


def test_truncated_roots_count_the_positions_a_window_really_spans():
    life = _trunc_life()
    # K 5: the plain rule keeps roots 0..6 (len - K = 7). Windows:
    #   0: 0 1 2 3 4 7      span 7 -> 7  <= 11
    #   1: 1 2 3 4 7 8      -> 8
    #   2: 2 3 4 7 8 9      -> 9
    #   3: 3 4 7 8 9 10     -> 10
    #   4: 4 7 8 9 10 11    -> 11 (the closing record: allowed)
    #   5: 5 6 8 9 10 11    -> 11
    #   6: 6 8 9 10 11 12   -> 12 > 11: OUT (would run past the cut tail)
    assert mot.truncated_roots(life, 5, 3) == [0, 1, 2, 3, 4, 5]
    # without options: exactly the plain rule
    plain = hand_life(list(range(4)) * 3, terminated=False)
    assert mot.truncated_roots(plain, 5, 3) == list(range(len(plain) - 5))


def test_truncated_roots_on_a_long_life_match_the_definition():
    rng = np.random.default_rng(3)
    played = [int(x) for x in rng.integers(0, 2, 60)]
    options = {t: tuple(played[t:t + k]) for t in range(0, 58, 2)
               for k in (2 + (t % 4 == 0),)}
    life = hand_life(played, options=options, A_=2, terminated=False)
    n = len(life) - 1
    want = [i for i in range(len(life) - 5)
            if i + mot.span_of(life, i, 5) <= n]
    got = mot.truncated_roots(life, 5, 3)
    assert got == want
    assert any(mot.span_of(life, i, 5) > 5 for i in got)
    assert len(got) < len(life) - 5         # some tail roots dropped


def _no_fit_life():
    # a CUT life of 6 acted positions (len 7, K 5 < 7 <= K x L = 15) with
    # length-3 jumps played at 0 and 1: the plain candidates are 0 and 1,
    #   0: 0 3 4 5 6 7   span 7 -> 7 > 6
    #   1: 1 4 5 6 7 8   span 7 -> 8 > 6
    # so no window fits (audit r1 of stage 3)
    played = [0, 1, 2, 1, 2, 3]
    return hand_life(played, options={0: (0, 1, 2), 1: (1, 2, 1)},
                     pmax=list(played), terminated=False)


def test_a_cut_life_no_window_fits_keeps_root_0():
    life = _no_fit_life()
    assert [mot.span_of(life, i, 5) for i in (0, 1)] == [7, 7]
    assert mot.truncated_roots(life, 5, 3) == [0]


def test_a_replay_of_such_lives_alone_still_samples(plain_knobs):
    life = _no_fit_life()
    b = build_batch(stub_trainer(), [life], 1)
    assert b["sample_refs"][0] is life[0]
    # root 0's window under the plain fallback: positions 0 3 4 5 6 7, the
    # last one padding past the cut tail
    assert b["real_mask"][0].tolist() == [True] * 5 + [False]
    assert b["option_span"][0].tolist() == [3, 1, 1, 1, 1]


def test_a_train_step_on_a_cut_life_fallback_window(o0, monkeypatch):
    """Stage 3's audits left this for stage 4 (the test above pins only the
    root, real_mask and spans): a TRAIN STEP on that fallback window, real
    frames and option nets. Every row is root 0 (sampled with replacement),
    its window 0 3 4 5 6 7: one jump, three one-step edges onto real
    positions (6 the closing record), the last edge onto padding. The step
    is finite; PMAX is rewritten at the acted positions it visits (0 3 4 5)
    and never at the closing record or the padding; the policy target is
    masked at the padding only (a cut life's closing record keeps its
    search); the closing record and the padding are terminal for the heads
    (stop, scale 1); the padding edge takes no consistency."""
    run = option_run(o0, monkeypatch)
    tr = run.trainer
    rng = np.random.default_rng(12)
    played = [0, 1, 2, 1, 2, 3]
    life = hand_life(played, options={0: (0, 1, 2), 1: (1, 2, 1)},
                     pmax=list(played), terminated=False, A_=A9)
    for r in life:
        r["obs"] = rng.integers(0, 255, (12, 84, 84), dtype=np.uint8)
        r["target_policy"] = rng.dirichlet(np.ones(A9))
    assert mot.truncated_roots(life, 5, 3) == [0]
    dezero_policy(tr)
    seed_all(13)
    b = build_batch(tr, [life], 4, A9)
    assert all(r is life[0] for r in b["sample_refs"])
    assert b["option_span"].tolist() == [[3, 1, 1, 1, 1]] * 4
    assert b["real_mask"].tolist() == [[True] * 5 + [False]] * 4
    assert b["has_policy_target"].tolist() == [[True] * 5 + [False]] * 4
    assert b["actions"][:, 0].tolist() == [[0, 1, 2]] * 4
    assert (b["option_scale"][:, 4:] == 1.0).all()
    assert (b["option_target"][:, 4:, :, A9] == 1.0).all()
    assert [[None if x is None else at(life, x) for x in row]
            for row in b["pmax_refs"]] == [[0, 3, 4, 5, None, None]] * 4
    for r in life[:-1]:
        r["pmax"] = -7                      # mark: rewritten or not
    loss = asyncio.run(tr.train_step(b))
    assert math.isfinite(loss)
    assert [life[k]["pmax"] for k in (1, 2)] == [-7, -7]   # never visited
    assert all(isinstance(life[k]["pmax"], int) and 0 <= life[k]["pmax"] < A9
               for k in (0, 3, 4, 5))
    assert "pmax" not in life[6]
    last = tr._option_last
    assert last["option_jump_share"] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert last["option_span_mean"] == 7.0
    assert last["cons_jump_n"] == 4 and last["cons_single_n"] == 3 * 4
    assert math.isfinite(float(last["option_loss"]))


# ==========================================================================
# head targets (getOptionPolicy / getOptionLossScale + DEPARTURE 7)
# ==========================================================================
#        t:    0                     1                     2 (miss)
HT_PI = [[0.1, 0.6, 0.2, 0.1], [0.7, 0.1, 0.1, 0.1], [0.2, 0.1, 0.6, 0.1],
         #     3                        4
         [0.25, 0.25, 0.25, 0.25], [0.0, 0.9, 0.1, 0.0]]
HT_PMAX = [1, 0, 0, 3, 1]     # at 2 the NETWORK's top is 0, pi's top is 2
HT_PLAYED = [1, 0, 3, 3, 1]   # 2 misses its PMAX


def ht_life():
    return hand_life(HT_PLAYED, pis=HT_PI, pmax=HT_PMAX)


def test_head_targets_product_by_hand():
    life = ht_life()
    tgt, scale = targets_at(life, 0)
    # head 1: m1 = PMAX(1) = 0, pi(1)[0] = 0.7
    # head 2: m2 = PMAX(2) = 0 (not pi(2)'s top, 2), 0.7 x pi(2)[0] = 0.14
    assert np.allclose(tgt, [head(m0=0.7, stop=0.3), head(m0=0.14, stop=0.86)])
    assert scale.tolist() == [1.0, 1.0]


def test_a_miss_mid_chain_keeps_its_head_and_stops_the_rest():
    tgt, scale = targets_at(ht_life(), 1)
    # head 1: m = PMAX(2) = 0, pi(2)[0] = 0.2; the move played at 2 (3)
    # misses it, so head 2 is stop -- trained (scale 1)
    assert np.allclose(tgt, [head(m0=0.2, stop=0.8), STOP_ROW])
    assert scale.tolist() == [1.0, 1.0]


def test_a_miss_at_t_masks_the_extra_heads():
    tgt, scale = targets_at(ht_life(), 2)
    assert np.allclose(tgt, [STOP_ROW, STOP_ROW])
    assert scale.tolist() == [0.0, 0.0]


def test_the_closing_record_is_terminal():
    life = ht_life()
    tgt, scale = targets_at(life, 3)
    # head 1: m = PMAX(4) = 1, 0.9; head 2 would read position 5, the
    # closing record: stop
    assert np.allclose(tgt, [head(m1=0.9, stop=0.1), STOP_ROW])
    assert scale.tolist() == [1.0, 1.0]
    for t in (4, 5, 6):          # the last acted one, the closing one, padding
        tgt, scale = targets_at(life, t)
        assert np.allclose(tgt, [STOP_ROW, STOP_ROW]), t
        assert scale.tolist() == [1.0, 1.0], t


def test_a_cut_lifes_closing_search_still_counts_as_terminal():
    # the closing record of a cut life carries a real search, all on 2, and
    # its placeholder action is 2: nothing past the last acted position
    life = hand_life([1, 2], pis=[[0, 1, 0, 0], [0, 0, 1, 0]], pmax=[1, 2],
                     terminated=False, closing_action=2,
                     closing_pi=[0, 0, 1.0, 0])
    tgt, scale = targets_at(life, 1)
    assert np.allclose(tgt, [STOP_ROW, STOP_ROW])
    assert scale.tolist() == [1.0, 1.0]
    tgt, _ = targets_at(life, 0)
    assert np.allclose(tgt, [head(m2=1.0, stop=0.0), STOP_ROW])


def test_refreshed_policies_flow_through():
    life = ht_life()
    life[1]["target_policy"] = np.array([0.4, 0.2, 0.2, 0.2])  # a refresh
    tgt, _ = targets_at(life, 0)
    assert np.allclose(tgt, [head(m0=0.4, stop=0.6), head(m0=0.08, stop=0.92)])


def test_pmax_is_the_records_field_else_the_policys_argmax():
    # position 2 has no PMAX field: getMaxActionID's fallback, pi(2)'s top 2
    life = hand_life([1, 0, 2, 3, 1], pis=HT_PI, pmax=[1, 0, None, 3, 1])
    tgt, _ = targets_at(life, 1)
    # head 1: m = 2, pi(2)[2] = 0.6; played 2 hits; head 2: m = PMAX(3) = 3,
    # 0.6 x 0.25 = 0.15
    assert np.allclose(tgt, [head(m2=0.6, stop=0.4), head(m3=0.15, stop=0.85)])
    # a negative PMAX reads the fallback too; ties go to the first
    life[2]["pmax"] = -1
    life[2]["target_policy"] = np.array([0.3, 0.3, 0.3, 0.1])
    assert mot.record_pmax(life[2]) == 0


def test_the_heads_are_l_dependent():
    # L 4: three extra heads, the third the product of three terms
    life = hand_life([0, 0, 0, 0, 0],
                     pis=[[0.9, .1, 0, 0], [0.8, .2, 0, 0], [0.5, .5, 0, 0],
                          [0.5, .5, 0, 0], [1.0, 0, 0, 0]],
                     pmax=[0] * 5, options={}, L_=4)
    tgt, scale = targets_at(life, 0, L_=4)
    assert np.allclose(tgt[:, 0], [0.8, 0.4, 0.2])
    assert np.allclose(tgt[:, STOP], [0.2, 0.6, 0.8])
    assert scale.tolist() == [1.0, 1.0, 1.0]


# ==========================================================================
# the batch: jumps, summed rewards, value / prefix targets, spans, refs
# ==========================================================================
def test_the_batch_row_of_a_jump_window_by_hand(plain_knobs):
    life = win_life()
    tr = stub_trainer()
    b = build_batch(tr, [life], 7)          # every root, without replacement
    r = row_of(b, life[0])
    # positions 0, 3, 5, 6, 7 (closing), 8 (padding); spans 3 2 1 1 1
    assert b["actions"][r].tolist() == [[1, 2, 3], [0, 1, -1], [2, -1, -1],
                                        [2, -1, -1], [0, -1, -1]]
    assert b["option_span"][r].tolist() == [3, 2, 1, 1, 1]
    # rewards = the record's index; an edge sums the rewards it enters,
    # undiscounted: 1+2+3, 4+5, 6, 7, padding 0
    assert b["rewards"][r].tolist() == [0.0, 6.0, 9.0, 6.0, 7.0, 0.0]
    prefix = muzero.discounted_prefix_targets(
        torch.tensor(b["rewards"][r:r + 1]), discount=1.0)
    assert prefix[0].tolist() == [0.0, 6.0, 15.0, 21.0, 28.0, 28.0]
    # value targets are each POSITION's own (g 0.5, 5 steps, stored 0):
    # z(0) = 1 + .5x2 + .25x3 + .125x4 + .0625x5; z(3) = 4 + .5x5 + .25x6 +
    # .125x7; z(5) = 6 + .5x7; z(6) = 7; z(7) = 0; padding 0
    assert np.allclose(b["target_value"][r],
                       [3.5625, 8.875, 9.5, 7.0, 0.0, 0.0])
    assert b["real_mask"][r].tolist() == [True] * 5 + [False]
    assert b["has_policy_target"][r].tolist() == [True] * 4 + [False] * 2
    assert np.array_equal(b["obs"][r, :, 0], [0, 3, 5, 6, 7, 7])
    refs = b["pmax_refs"][r]
    assert [x is y for x, y in zip(refs[:4], [life[0], life[3], life[5],
                                               life[6]])] == [True] * 4
    assert refs[4] is None and refs[5] is None
    # head targets (PMAX = the move played everywhere, pi uniform 0.25):
    #   0: m1 = 2 (0.25), m2 = 3 (0.0625); 3: m1 = 1, m2 = 2; 5: m1 = 2,
    #   then the closing record; 6, 7, 8: stop
    t = b["option_target"][r]
    assert np.allclose(t[0], [head(m2=0.25, stop=0.75),
                              head(m3=0.0625, stop=0.9375)])
    assert np.allclose(t[1], [head(m1=0.25, stop=0.75),
                              head(m2=0.0625, stop=0.9375)])
    assert np.allclose(t[2], [head(m2=0.25, stop=0.75), STOP_ROW])
    for s in (3, 4, 5):
        assert np.allclose(t[s], [STOP_ROW, STOP_ROW]), s
    assert (b["option_scale"][r] == 1.0).all()
    assert b["option_length"] == 3


def test_a_one_step_row_is_the_plain_rows_data(plain_knobs):
    life = win_life()
    b = build_batch(stub_trainer(), [life], 7)
    r = row_of(b, life[4])
    assert b["actions"][r].tolist() == [[1, -1, -1], [2, -1, -1],
                                        [2, -1, -1], [0, -1, -1],
                                        [0, -1, -1]]
    assert b["rewards"][r].tolist() == [4.0, 5.0, 6.0, 7.0, 0.0, 0.0]
    assert b["option_span"][r].tolist() == [1] * 5


def test_a_truncated_lifes_roots_in_the_batch(plain_knobs):
    life = _trunc_life()
    b = build_batch(stub_trainer(), [life], 6)
    roots = sorted(at(life, r) for r in b["sample_refs"])
    assert roots == [0, 1, 2, 3, 4, 5]
    assert b["real_mask"].all()             # no window runs past the cut tail


def test_masks_and_shares_in_the_batch(plain_knobs):
    # position 1 misses its PMAX: every window holding it masks it there
    life = hand_life([1, 2, 3, 0], pmax=[1, 0, 3, 0], options={})
    b = build_batch(stub_trainer(), [life], 4)
    for r in range(4):
        start = at(life, b["sample_refs"][r])
        for s in range(6):
            want = 0.0 if start + s == 1 else 1.0
            assert b["option_scale"][r, s].tolist() == [want, want]


def test_the_closing_record_and_padding_never_take_a_pmax(plain_knobs):
    life = win_life()
    b = build_batch(stub_trainer(), [life], 7)
    for r, refs in enumerate(b["pmax_refs"]):
        for ref in refs:
            assert ref is None or at(life, ref) < len(life) - 1


# ==========================================================================
# host-side jump validation
# ==========================================================================
def test_validate_jump_rows_passes_good_rows():
    mot.validate_jump_rows(np.array([[[1, 2, 3], [0, -1, -1], [3, 3, -1]]]),
                           4)


@pytest.mark.parametrize("bad, msg", [
    ([[-1, 2, -1]], "first move"), ([[-1, -1, -1]], "first move"),
    ([[1, -1, 2]], "suffix"), ([[4, -1, -1]], "outside"),
    ([[1, -2, -1]], "outside"), ([[1, 2, 7]], "outside")])
def test_validate_jump_rows_refuses(bad, msg):
    with pytest.raises(ValueError, match=msg):
        mot.validate_jump_rows(np.array([bad]), 4)


def test_finish_refuses_a_bad_row_before_any_forward(plain_knobs):
    tr = stub_trainer()
    a = tr.assemble_batch_paper([win_life()], 7, 5, A)
    a["actions"][2][1] = [1, -1, 2]
    with pytest.raises(ValueError, match="suffix"):
        tr.finish_batch_paper(a)
    a["actions"][2][1] = [-1, -1, -1]
    with pytest.raises(ValueError, match="first move"):
        tr.finish_batch_paper(a)


# ==========================================================================
# PMAX rewrite order (updateMax) and intern
# ==========================================================================
def test_write_pmax_rows_then_positions_later_wins():
    a, b, c = {}, {}, {}
    refs = [[a, b, None], [b, c, None]]
    argmax = np.array([[1, 2], [3, 4], [5, 6]])     # [K+1, B]
    mot.write_pmax(refs, argmax)
    # row 0 writes a = 1, b = 3; row 1 then b = 2, c = 4
    assert (a, b, c) == ({"pmax": 1}, {"pmax": 2}, {"pmax": 4})


def test_options_are_interned_and_padded():
    x = mot.intern_option(np.array([3, 1, 5]), 2)
    y = mot.intern_option([3, 1, 0], 2)
    assert x == (3, 1, -1) and x is y
    assert mot.intern_option([2, 2, 2], 1) == (-1, -1, -1)
    assert mot.intern_option([2, 2, 2], 0) is mot.intern_option([0, 0, 0], 0)
    assert mot.option_moves({"root_option": x}) == (3, 1)


# ==========================================================================
# the loss (calculate_loss 158-165)
# ==========================================================================
def test_option_loss_mask_and_averaging_by_hand():
    tr = stub_trainer()
    B = 2
    # zero logits: log_softmax = -log 3 over A + 1 = 3 classes, so every
    # head's cross-entropy is log 3 whatever its target
    olog = torch.zeros(B, 3, 3, requires_grad=True)
    # [B, K+1 = 2, L-1 = 2, A+1 = 3]; both positions alike
    oin = {"target": torch.tensor([[[[0.5, 0.0, 0.5], [0.0, 0.0, 1.0]]] * 2]
                                  * B),
           "scale": torch.tensor([[[1.0, 0.0]] * 2, [[1.0, 1.0]] * 2]),
           "argmax": [], "loss": None}
    w = torch.tensor([1.0, 0.5])
    got = tr._option_loss(oin, 0, olog, torch.zeros(B, 2), w, 5)
    # head 1: (1 x 1 + 0.5 x 1) log3 / 2 = 0.75 log3; head 2: (0 + 0.5) log3
    # / 2 = 0.25 log3; masked terms count in the mean; / (L - 1) = 2
    assert float(got) == pytest.approx(0.5 * math.log(3))
    got.backward()
    assert olog.grad[:, 1:].abs().sum() > 0
    assert olog.grad[:, 0].abs().sum() == 0          # head 0 is the policy's
    # the reported loss: step 0 plus the unrolled positions / K
    got2 = tr._option_loss(oin, 1, olog.detach(), torch.zeros(B, 2), w, 5)
    assert float(oin["loss"]) == pytest.approx(
        0.5 * math.log(3) + float(got2) / 5)
    assert len(oin["argmax"]) == 2


def test_option_loss_without_weights_and_with_a_real_target():
    tr = stub_trainer()
    olog = torch.tensor([[[0.0, 0.0, 0.0], [math.log(3.0), 0.0, 0.0],
                          [0.0, 0.0, 0.0]]])
    oin = {"target": torch.tensor([[[[0.25, 0.0, 0.75], [0.0, 0.0, 1.0]]]]),
           "scale": torch.ones(1, 1, 2), "argmax": [], "loss": None}
    got = tr._option_loss(oin, 0, olog, torch.zeros(1, 2), None, 5)
    # head 1: softmax (3, 1, 1)/5 -> CE = -(0.25 ln .6 + 0.75 ln .2);
    # head 2: ln 3; mean of the two
    want = (-(0.25 * math.log(0.6) + 0.75 * math.log(0.2)) + math.log(3)) / 2
    assert float(got) == pytest.approx(want)


# ==========================================================================
# the train step on real option nets
# ==========================================================================
def option_run(spec, monkeypatch, L_=3, edge=True, seed=0):
    monkeypatch.setattr(muzero, "OPTION_LENGTH", L_)
    monkeypatch.setattr(muzero, "OPTION_EDGE", edge)
    torch.manual_seed(seed)
    return make_run(spec, O0)


def real_lives(seed, L_=3, n_lives=((14, True), (12, False))):
    """Lives with real frames; suggestions that match the play every third
    position (jumps), PMAX = the move played (unmasked) but every fifth."""
    rng = np.random.default_rng(seed)
    lives = []
    for n, terminated in n_lives:
        acts = [int(rng.integers(0, A9)) for _ in range(n)]
        recs = []
        for t in range(n + 1):
            acted = t < n
            r = {"obs": rng.integers(0, 255, (12, 84, 84), dtype=np.uint8),
                 "action": acts[t] if acted else 0,
                 "reward": float(rng.integers(0, 2)) if t else 0.0,
                 "target_policy": rng.dirichlet(np.ones(A9)),
                 "root_value": float(rng.normal()), "target_value": 0.0,
                 "has_policy_target": acted or not terminated,
                 "born_step": 39990, "priority": float(rng.uniform(.5, 1.5))}
            r["raw_reward"] = (10.0,) if r["reward"] else ()
            if acted:
                r["pmax"] = acts[t] if t % 5 != 4 else (acts[t] + 1) % A9
                k = min(L_, n - t)
                if t % 3 == 0 and k >= 2:
                    r["root_option"] = mot.intern_option(
                        acts[t:t + k] + [-1] * (L_ - k), k)
                else:
                    r["root_option"] = mot.intern_option([-1] * L_, 0)
            recs.append(r)
        recs[-1]["truncated"] = not terminated
        lives.append(recs)
    return lives


def dezero_policy(tr, seed=4, scale=2.0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in tr.prediction.policy_fc.parameters():
            p.normal_(0, scale, generator=g)


def spy(module, store, kind):
    orig = module.forward

    def fwd(*args, **kw):
        out = orig(*args, **kw)
        store.append((kind, args, kw, out))
        return out
    module.forward = fwd
    return orig


def test_train_step_jumps_one_call_per_edge_and_pools_prefix_sets(
        o0, monkeypatch):
    run = option_run(o0, monkeypatch)
    tr = run.trainer
    replay = real_lives(0)
    seed_all(1)
    b = build_batch(tr, replay, 8, A9)
    assert (b["option_span"] > 1).any(), "the batch must hold jumps"
    calls = []
    spy(tr.dynamics, calls, "dyn")
    spy(tr.value_prefix, calls, "vp")
    vp = tr.value_prefix
    counts = {n: getattr(vp, f"{n}_step_count").clone()
              for n in vp.norm_names()}
    asyncio.run(tr.train_step(b))
    dyn = [c for c in calls if c[0] == "dyn"]
    pre = [c for c in calls if c[0] == "vp"]
    # one dynamics call per edge, every row [B, L], the check skipped (the
    # rows were validated on the host)
    assert len(dyn) == 5
    for i, (_, args, kw, _) in enumerate(dyn):
        assert tuple(args[1].shape) == (8, 3)
        assert kw.get("check_jump") is False
        assert torch.equal(args[1], torch.as_tensor(b["actions"][:, i]))
    # one LSTM call per edge, all B rows (jump and single) at step i
    assert [(kw["step"], args[0].shape[0]) for _, args, kw, _ in pre] == [
        (i, 8) for i in range(1, 6)]
    for n in vp.norm_names():
        assert (getattr(vp, f"{n}_step_count") - counts[n]).tolist() == \
            [1] * 5, n


def test_gradient_reaches_the_option_heads_and_the_jump_rows(o0, monkeypatch):
    run = option_run(o0, monkeypatch)
    tr = run.trainer
    replay = real_lives(0)
    seed_all(2)
    b = build_batch(tr, replay, 8, A9)
    asyncio.run(tr.train_step(b))
    for i in range(2):
        for m in (tr.prediction.option_convs[i], tr.prediction.option_fcs[i]):
            assert all(p.grad is not None and p.grad.abs().sum() > 0
                       for p in m.parameters()), (i, m)
    acts = b["actions"]
    for j, emb in enumerate(tr.dynamics.option_embeddings, start=1):
        used = set(int(a) for a in acts[..., j].ravel() if a >= 0)
        g = emb.weight.grad.abs().sum(dim=1)
        assert used, j
        assert {int(k) for k in torch.nonzero(g).flatten()} == used, j


def test_pmax_is_read_before_the_step_and_rewritten_after(o0, monkeypatch):
    run = option_run(o0, monkeypatch)
    tr = run.trainer
    dezero_policy(tr)                     # distinct argmaxes per position
    replay = real_lives(5, n_lives=((16, True),))
    life = replay[0]
    for r in life[:-1]:
        r["pmax"] = r["action"]           # every acted position a hit
    seed_all(3)
    b = build_batch(tr, replay, 12, A9)
    # the targets read the PMAX stored NOW: every acted position unmasked
    for r in range(12):
        for s in range(6):
            assert (b["option_scale"][r, s] == 1.0).all()
    calls = []
    spy(tr.prediction, calls, "pred")
    before = [r.get("pmax") for r in life]
    asyncio.run(tr.train_step(b))
    preds = [c for c in calls if c[0] == "pred"]
    assert len(preds) == 6
    am = [out[0].argmax(dim=1) for _, _, _, out in preds]
    want = {}
    disagree = False
    for row, refs in enumerate(b["pmax_refs"]):
        for s, ref in enumerate(refs):
            if ref is not None:
                k = at(life, ref)
                v = int(am[s][row])
                disagree |= k in want and want[k] != v
                want[k] = v                # rows in order: a later row wins
    assert disagree, "the test needs a position two rows read differently"
    for k, r in enumerate(life):
        if k in want:
            assert r["pmax"] == want[k], k
        else:
            assert r.get("pmax") == before[k], k
    assert "pmax" not in life[-1]
    # the next batch reads the rewritten PMAX: masked iff played != PMAX
    b2 = build_batch(tr, replay, 12, A9)
    for row in range(12):
        start = at(life, b2["sample_refs"][row])
        pos, _ = mot.window(life, start, 5)
        for s, p in enumerate(pos):
            masked = p < len(life) - 1 and life[p]["action"] != life[p]["pmax"]
            assert (b2["option_scale"][row, s] == (0.0 if masked else 1.0)
                    ).all()


def option_terms(olog_by_pos, b):
    """calculate_loss's option term per position, written out here: per
    extra head h, the batch MEAN of is_weight x mask x cross-entropy (the
    masked rows count as zeros), summed over heads, / (L - 1)."""
    w = torch.tensor(b["is_weights"], dtype=torch.float32)
    tgt = torch.tensor(b["option_target"], dtype=torch.float32)
    sc = torch.tensor(b["option_scale"], dtype=torch.float32)
    out = []
    for s, olog in enumerate(olog_by_pos):
        tot = 0.0
        for h in range(1, olog.shape[1]):
            ce = -(tgt[:, s, h - 1] * torch.log_softmax(olog[:, h], dim=1)
                   ).sum(dim=1)
            tot += float((w * sc[:, s, h - 1] * ce).sum()) / olog.shape[0]
        out.append(tot / (olog.shape[1] - 1))
    return out


def test_the_option_loss_enters_the_total_at_scale_1(o0, monkeypatch):
    """Two copies of one trainer on one batch and one RNG draw, the second
    with the option loss zeroed: the totals differ by the option loss of
    every position (calculate_loss, written out in option_terms from the
    heads' logits), unweighted in value (the 1/K is gradient-only), so the
    term enters at scale 1."""
    run = option_run(o0, monkeypatch)
    tr = run.trainer
    dezero_policy(tr)
    replay = real_lives(1)
    seed_all(4)
    b = build_batch(tr, replay, 8, A9)
    assert (b["option_scale"] == 0).any() and (b["option_scale"] == 1).any()
    tr2 = copy.deepcopy(tr)
    calls = []
    spy(tr.prediction, calls, "pred")
    real = muzero.TrainingHelper._option_loss

    def zero(self, oin, i, *a):
        return real(self, oin, i, *a) * 0.0

    seed_all(9)
    full = asyncio.run(tr.train_step(copy.deepcopy(b)))
    monkeypatch.setattr(muzero.TrainingHelper, "_option_loss", zero)
    seed_all(9)
    none = asyncio.run(tr2.train_step(copy.deepcopy(b)))
    olog = [out[3].detach() for _, _, kw, out in calls
            if kw.get("with_options")]
    assert len(olog) == 6
    terms = option_terms(olog, b)
    assert all(t > 0 for t in terms)
    assert full - none == pytest.approx(sum(terms), rel=1e-4)
    # the reported loss is step 0 plus the unrolled positions / 5
    got = float(tr._option_last["option_loss"])
    assert got == pytest.approx(terms[0] + sum(terms[1:]) / 5, rel=1e-5)


@pytest.mark.parametrize("effzero", [True, False])
def test_the_option_terms_gradient_weight_at_every_position(o0, monkeypatch,
                                                            effzero):
    """Audit r1 of stage 3 (its test_audit_grad_weight.py, ported): the
    loss VALUE cannot see where the option term sits against the 1/K
    gradient scale, so its GRADIENT is pinned. dTotal / d option_logits_i
    = w_i x d l_i / d option_logits_i, l_i calculate_loss's term written
    out here and w_i the position's weight: 1/K on every unrolled position
    and, under EFFZERO_LOSS_SCALING (O0's), on the root too; 1 on the root
    otherwise (OptionZero's loss_option = step 0 + the unrolled / K,
    train.py 235 / 258). Head 0 of the stacked logits takes none (the
    policy loss reads the policy logits). And every written PMAX is its own
    forward's policy-logit argmax, a later row winning."""
    assert muzero.EFFZERO_LOSS_SCALING is True
    monkeypatch.setattr(muzero, "EFFZERO_LOSS_SCALING", effzero)
    run = option_run(o0, monkeypatch)
    tr = run.trainer
    dezero_policy(tr)
    replay = real_lives(1)
    seed_all(4)
    b = build_batch(tr, replay, 8, A9)
    K = b["actions"].shape[1]
    kept = []
    orig = tr.prediction.forward

    def fwd(*a, **kw):
        out = orig(*a, **kw)
        if kw.get("with_options"):
            out[3].retain_grad()
            kept.append(out)
        return out

    tr.prediction.forward = fwd
    asyncio.run(tr.train_step(b))
    assert len(kept) == K + 1
    w = torch.tensor(b["is_weights"], dtype=torch.float32)
    tgt = torch.tensor(b["option_target"], dtype=torch.float32)
    sc = torch.tensor(b["option_scale"], dtype=torch.float32)
    for s, out in enumerate(kept):
        lg = out[3].detach().clone().requires_grad_(True)
        Lh = lg.shape[1]
        tot = 0.0
        for h in range(1, Lh):
            ce = -(tgt[:, s, h - 1] * torch.log_softmax(lg[:, h], dim=1)
                   ).sum(dim=1)
            tot = tot + (w * sc[:, s, h - 1] * ce).mean()
        (tot / (Lh - 1)).backward()
        weight = 1.0 / K if (s > 0 or effzero) else 1.0
        want = lg.grad[:, 1:] * weight
        got = out[3].grad[:, 1:]
        assert want.abs().sum() > 0, s
        assert torch.allclose(got, want, atol=1e-7, rtol=1e-4), (s, float(
            got.abs().sum() / want.abs().sum()))
        assert float(out[3].grad[:, 0].abs().sum()) == 0.0, s
    want = {}
    for row, refs in enumerate(b["pmax_refs"]):
        for s, ref in enumerate(refs):
            if ref is not None:
                want[id(ref)] = (ref, int(kept[s][0].argmax(dim=1)[row]))
    assert want
    assert all(ref["pmax"] == v for ref, v in want.values())


def test_the_reported_shares(o0, monkeypatch):
    run = option_run(o0, monkeypatch)
    tr = run.trainer
    replay = real_lives(0)
    seed_all(5)
    b = build_batch(tr, replay, 8, A9)
    asyncio.run(tr.train_step(b))
    got = tr._option_last
    spans = b["option_span"]
    assert got["option_jump_share"] == pytest.approx(
        [float((spans[:, i] > 1).mean()) for i in range(5)])
    assert got["option_mask_share"] == pytest.approx(
        float((b["option_scale"] == 0).mean()))
    assert 0 < got["option_mask_share"] < 1
    assert got["option_span_mean"] == pytest.approx(
        float(spans.sum(axis=1).mean()))
    assert got["option_loss"].requires_grad is False


def test_option_nets_refuse_a_batch_without_option_targets(o0, monkeypatch):
    run = option_run(o0, monkeypatch)
    tr = run.trainer
    replay = real_lives(0)
    seed_all(6)
    b = build_batch(tr, replay, 4, A9)
    for k in ("option_target", "option_scale", "option_span", "pmax_refs",
              "option_length", "option_rows_validated"):
        b.pop(k)
    b["actions"] = b["actions"][..., 0]
    with pytest.raises(ValueError, match="no option targets"):
        asyncio.run(tr.train_step(b))


def test_option_nets_refuse_rows_not_validated_on_the_host(o0, monkeypatch):
    """The forward skips the network's jump-row check (check_jump=False), so
    train_step requires finish_batch_paper's mark (audit r1 of stage 3)."""
    run = option_run(o0, monkeypatch)
    tr = run.trainer
    replay = real_lives(0)
    seed_all(6)
    b = build_batch(tr, replay, 4, A9)
    assert b["option_rows_validated"] is True
    calls = []
    spy(tr.dynamics, calls, "dyn")
    for bad in (None, False, 1):
        c = dict(b)
        if bad is None:
            c.pop("option_rows_validated")
        else:
            c["option_rows_validated"] = bad
        with pytest.raises(ValueError, match="not validated on the host"):
            asyncio.run(tr.train_step(c))
    assert calls == [], "refused before any forward"


def test_a_flag_off_batch_carries_no_option_field(plain_knobs):
    life = hand_life([1, 2, 3, 0])
    b = build_batch(stub_trainer(L_=1), [life], 4)
    assert not any(k.startswith("option") or k == "pmax_refs" for k in b)
    assert b["actions"].shape == (4, 5)


def test_a_skipped_nonfinite_step_writes_no_pmax(o0, monkeypatch):
    run = option_run(o0, monkeypatch)
    tr = run.trainer
    replay = real_lives(0)
    seed_all(7)
    b = build_batch(tr, replay, 8, A9)
    before = [[r.get("pmax") for r in life] for life in replay]
    monkeypatch.setattr(muzero, "SKIP_NONFINITE_GRAD_STEPS", True)
    real = torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_",
                        lambda *a, **k: real(*a, **k) * float("nan"))
    asyncio.run(tr.train_step(b))
    assert [[r.get("pmax") for r in life] for life in replay] == before


# ==========================================================================
# the birth search on the option engine, the records, the refresh
# ==========================================================================
def life3(n, acts, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.integers(0, 255, (12, 84, 84), dtype=np.uint8),
             acts[t] if t < n else 0, (10.0,) if t % 2 else ())
            for t in range(n + 1)]


def suggest(tr, moves=(4, 4)):
    """Make every node's dominant option (0, *moves): the policy head's
    weights are zero-init, so a bias of 3 on move 0 makes 0 its argmax
    (and its first move most visited, so the edge gets crossed), and each
    extra head's bias puts its mass on its move."""
    with torch.no_grad():
        tr.prediction.policy_fc.bias.zero_()
        tr.prediction.policy_fc.bias[0] = 3.0
        for fc, m in zip(tr.prediction.option_fcs, moves):
            fc.bias.zero_()
            fc.bias[m] = 20.0


def net_argmax(tr, frames):
    """The learner's own top move per stack, computed apart from the search
    (eval mode, /255 as the search's root forward; modes restored)."""
    nets = (tr.representation, tr.prediction)
    modes = [n.training for n in nets]
    for n in nets:
        n.eval()
    try:
        with torch.no_grad():
            x = torch.tensor(np.stack(frames), dtype=torch.float32) / 255.0
            logits, _, _ = tr.prediction(tr.representation(x))
    finally:
        for n, m in zip(nets, modes):
            n.train(m)
    return [int(a) for a in logits.argmax(dim=1)]


@pytest.mark.parametrize("edge", [True, False])
def test_the_birth_search_runs_the_option_engine(o0, monkeypatch, edge):
    """De-zeroed policy weights (the network's top move varies by stack and
    differs from the visit argmax somewhere), each extra head's bias on move
    4, so every node's dominant option is (its top move, 4, 4). The records
    must hold the NETWORK's top move as PMAX (zero_actor.cpp 306), not the
    visits' (audit r1 of stage 3: zero-init ties gave both 0)."""
    run = option_run(o0, monkeypatch, edge=edge)
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 12)
    tr = run.trainer
    dezero_policy(tr, seed=11, scale=1.0)
    with torch.no_grad():
        for fc in tr.prediction.option_fcs:
            fc.bias.zero_()
            fc.bias[4] = 20.0
    n = 8
    frames = [f for f, _, _ in life3(n, [0] * n, seed=5)]
    m = net_argmax(tr, frames[:n])
    # play: (m0, 4, 4) from 0 -- a jump; after it, never the top move, nor 4
    acts = [m[0], 4, 4] + [next(v for v in range(A9) if v not in (m[t], 4))
                           for t in range(3, n)]
    traj = [(f, (acts[t] if t < n else 0), (10.0,) if t % 2 else ())
            for t, f in enumerate(frames)]
    seen = []
    real = muzero_vtree_options.VectorizedOptionPUCT.search

    def watch(self, obs, **kw):
        out = real(self, obs, **kw)
        seen.append((kw, out[3]))
        return out

    monkeypatch.setattr(muzero_vtree_options.VectorizedOptionPUCT, "search",
                        watch)

    def no_v1(*a, **k):
        raise AssertionError("the option arm ran the v1 birth search")

    monkeypatch.setattr(muzero_vtree.VectorizedPUCT, "search", no_v1)
    seed_all(21)
    out = run.paper_targets(traj, False, born_step=0)
    assert [kw["option_edge"] for kw, _ in seen] == [edge, edge]
    assert [kw["root_noise"] for kw, _ in seen] == [True, False]
    assert all(kw["return_info"] is True for kw, _ in seen)
    info = seen[0][1]
    assert info["option_edge"] is edge
    visit_top = [int(np.argmax(r["target_policy"])) for r in out[:n]]
    assert any(v != t for v, t in zip(visit_top, m)), \
        "the case needs a stack whose visit argmax is not the network's"
    for i in range(n):
        assert out[i]["pmax"] == m[i] == int(info["root_pmax"][i]), i
        assert out[i]["root_option"] == (m[i], 4, 4), i
        assert out[i]["root_option"] is mot.intern_option([m[i], 4, 4], 3)
    assert "root_option" not in out[n] and "pmax" not in out[n]
    assert run._birth_options is None
    counts = run._birth_option_counts
    assert counts["roots"] == n + 1 and counts["simulations"] == 12 * (n + 1)
    assert counts["expanded_with_option"] >= n + 1
    if edge:
        assert counts["sims_crossing_option"] > 0
    else:
        assert counts["sims_crossing_option"] == 0
    # the records jump where the play matched the suggestion, only there
    assert [mot.step_length(out, t) for t in range(n + 1)] == \
        [3] + [1] * n


def test_the_precheck_still_runs_under_options(o0, monkeypatch):
    run = option_run(o0, monkeypatch)
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 4)
    dezero_policy(run.trainer)
    assert muzero.OPTION_PRECHECK is True
    out = run.paper_targets(life3(6, [1, 2, 3, 4, 5, 6]), True, born_step=0)
    assert run._birth_pmax.tolist() == [r["pmax"] for r in out[:-1]]
    assert run._pc_window["n"] == 6


def test_the_birth_search_refuses_nets_without_the_heads(o0, monkeypatch):
    run = option_run(o0, monkeypatch, L_=1)
    monkeypatch.setattr(muzero, "OPTION_LENGTH", 3)
    with pytest.raises(ValueError, match="option heads of length 1"):
        run._birth_search(life3(2, [0, 1]), True)


def test_the_birth_search_refuses_heads_under_option_length_1(o0,
                                                              monkeypatch):
    """The other way (audit r1 of stage 3): nets with heads under
    OPTION_LENGTH 1 must not run v1 silently while the trainer keys on the
    nets' option_length. Refused before anything runs: no RNG drawn."""
    run = option_run(o0, monkeypatch, L_=3)
    monkeypatch.setattr(muzero, "OPTION_LENGTH", 1)
    before = rng_states()
    with pytest.raises(ValueError,
                       match="OPTION_LENGTH 1: .*option heads of length 3"):
        run._birth_search(life3(2, [0, 1]), True)
    assert same_rng(before, rng_states())
    assert not hasattr(run, "_birth_stats")
    assert all(n.training for n in (run.trainer.representation,
                                    run.trainer.prediction))


def test_the_refresh_keeps_the_suggestions_and_pmax(o0, monkeypatch):
    run = option_run(o0, monkeypatch)
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 4)
    life = real_lives(2, n_lives=((6, True),))[0]
    keep = [(r.get("root_option"), r.get("pmax")) for r in life]
    old_pol = [r["target_policy"].copy() for r in life]
    engine = run._vtree_engine()
    assert not isinstance(engine, muzero_vtree_options.VectorizedOptionPUCT)
    run._refresh_trajs_vtree([life], engine, 40000)
    assert [(r.get("root_option"), r.get("pmax")) for r in life] == keep
    assert all(r.get("root_option") is k for r, (k, _) in zip(life, keep))
    assert any(not np.allclose(r["target_policy"], p)
               for r, p in zip(life[:-1], old_pol)), "the refresh ran"


def test_a_jump_edge_is_discounted_by_its_base_steps(monkeypatch):
    """The search side (stage 2) of the jump's discount: a landing the jump
    evaluates holds r_jump + DISCOUNT ** len x v, r_jump the jump row's one
    (undiscounted) reward. Fake nets from the stage-2 tests: root code 7,
    option (0, 1, 2), the jump's reward 0.6, the landing's value 2.0."""
    from tests.test_vtree_options import FakeNets, code_of, run as vrun
    for k, v in (("USE_GUMBEL_SEARCH", False), ("TOKEN_VOCAB", None),
                 ("USE_VALUE_PREFIX_IN_TREE", False), ("MAX_DEPTH", 12),
                 ("MINMAX_RANGE_FLOOR", 0.01), ("OPTION_EDGE", True)):
        monkeypatch.setattr(muzero, k, v)
    land = code_of((0, 1, 2))
    fake = FakeNets(A=3, L=3, preds={
        7: {"logits": [1.0, 0.0, 0.0], "value": 0.0,
            "option": ((0, 1, 2), 0.9)},
        (land, 1): {"logits": [0.0, 0.0, 0.0], "value": 2.0}},
        rewards={(7, (0, 1, 2)): 0.6})
    tree, *_ = vrun(fake, 1, discount=0.9)
    nid = int(tree["opt_land"][0, 0])
    assert nid >= 0 and int(tree["opt_len"][0, 0]) == 3
    assert float(tree["opt_r"][0, 0]) == pytest.approx(0.6)
    assert float(tree["tot"][0, nid]) == pytest.approx(0.6)
    assert int(tree["depth"][0, nid]) == 3
    # the landing's backed-up total: 0.6 + 0.9 ** 3 x 2.0
    assert float(tree["tot"][0, nid] + 0.9 ** int(tree["depth"][0, nid])
                 * tree["lval"][0, nid]) == pytest.approx(0.6 + 0.729 * 2.0)


def test_end_to_end_birth_records_batch_and_step(o0, monkeypatch):
    """A life through the birth search, into a batch (jumps where the play
    matched the root suggestion), through a train step, PMAX rewritten."""
    run = option_run(o0, monkeypatch)
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 4)
    suggest(run.trainer)
    acts = [0, 4, 4, 2, 0, 4, 4, 0, 4, 1]
    life = run.paper_targets(life3(10, acts, seed=3), True, born_step=39990)
    assert [mot.step_length(life, t) for t in range(10)] == \
        [3, 1, 1, 1, 3, 1, 1, 1, 1, 1]
    tr = run.trainer
    seed_all(8)
    b = build_batch(tr, [life], 10, A9)       # every root
    r = row_of(b, life[0])
    assert b["option_span"][r].tolist() == [3, 1, 3, 1, 1]
    assert b["actions"][r, 0].tolist() == [0, 4, 4]
    asyncio.run(tr.train_step(b))
    assert all(isinstance(x["pmax"], int) for x in life[:-1])
    assert "pmax" not in life[-1]


def test_the_row_diagnostics_run_on_option_nets_and_jump_batches(
        o0, monkeypatch):
    """evaluate_diagnostics (every eval row) builds paper batches -- jump
    rows now -- and unrolls them through the dynamics with its check on;
    the dormant-unit read walks the nets. Stage 4a's option fields (opt_*,
    tests/test_option_diag.py) join the row; nothing under another name."""
    run = option_run(o0, monkeypatch)
    run.replay = real_lives(3)
    seed_all(10)
    errs = run.evaluate_diagnostics(batches=1)
    assert math.isfinite(errs["prefix"]) and math.isfinite(errs["value"])
    assert "dormant_025" in errs
    assert not any(k.startswith("option") for k in errs)
    assert "opt_prefix_jump_rows" in errs and "opt_probe_roots" in errs


# ==========================================================================
# memory per record
# ==========================================================================
def test_the_record_fields_cost_two_dict_slots():
    rec = {"obs": None, "action": 1, "reward": 0.0, "target_policy": None,
           "root_value": 0.0, "target_value": 0.0, "has_policy_target": True,
           "born_step": 0, "priority": 1.0, "raw_reward": (),
           "refreshed_step": 5}
    plain = sys.getsizeof(rec)
    rec["root_option"] = mot.intern_option([0, 4, 4], 3)
    rec["pmax"] = 3
    added = sys.getsizeof(rec) - plain
    # the values are shared (interned tuples, small ints): only slots
    assert added <= 1200
    print(f"record dict {plain} -> {sys.getsizeof(rec)} bytes")


# ==========================================================================
# FLAG OFF: the pre-stage-3 code, bit for bit (git show e11fd5f8)
# ==========================================================================
def _git_show(path):
    try:
        return subprocess.run(["git", "show", f"{PRE_S3}:{path}"], cwd=ROOT,
                              capture_output=True, text=True, check=True,
                              encoding="utf-8").stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.fail(f"git show {PRE_S3}:{path} failed, so the flag-off pin "
                    f"cannot run: {exc}")


def _load(name, src, d):
    path = d / f"{name}.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pre_s3(tmp_path_factory):
    """networks.py, muzero.py and experiments/ablation.py as they were
    before stage 3: the old muzero imports the OLD networks (sys.modules
    swapped while it loads); the old ablation imports today's muzero (its
    own code is what it pins). Unreadable = a FAILED pin."""
    srcs = {p: _git_show(p) for p in ("networks.py", "muzero.py",
                                      "experiments/ablation.py")}
    assert "check_jump" not in srcs["networks.py"]
    assert "_option_inputs" not in srcs["muzero.py"]
    assert "_birth_options" not in srcs["experiments/ablation.py"]
    d = tmp_path_factory.mktemp("pre_s3")
    old_net = _load("networks_pre_s3", srcs["networks.py"], d)
    saved, path = sys.modules.get("networks"), list(sys.path)
    sys.modules["networks"] = old_net
    try:
        old_mz = _load("muzero_pre_s3", srcs["muzero.py"], d)
    finally:
        sys.modules["networks"] = saved
    try:
        old_ab = _load("ablation_pre_s3", srcs["experiments/ablation.py"], d)
    finally:
        sys.path[:] = path
    assert old_mz.DynamicsNetwork is old_net.DynamicsNetwork
    assert old_ab.muzero is muzero
    return types.SimpleNamespace(networks=old_net, muzero=old_mz,
                                 ablation=old_ab)


KNOB = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _copy_knobs(src, dst):
    for k in dir(src):
        if KNOB.match(k) and hasattr(dst, k):
            setattr(dst, k, getattr(src, k))


def _nets(mz):
    return dict(
        representation=mz.build_representation(),
        dynamics=mz.build_dynamics(A9, support_size=mz.REWARD_SUPPORT,
                                   num_bins=mz.REWARD_BINS, transform=True),
        prediction=mz.build_prediction(A9, support_size=mz.VALUE_SUPPORT,
                                       num_bins=mz.VALUE_BINS, transform=True),
        value_prefix=mz.build_value_prefix(
            support_size=mz.VALUE_PREFIX_SUPPORT,
            num_bins=mz.VALUE_PREFIX_BINS, transform=True),
        projector=mz.build_projector())


def _trainer(mz, nets):
    tr = object.__new__(mz.TrainingHelper)
    tr.device, tr.action_space, tr.training_steps = CPU, A9, 40000
    for n, net in nets.items():
        setattr(tr, n, net)
    for n, o in zip(ablation.NET_NAMES, ablation.OPT_NAMES):
        setattr(tr, o, mz.make_optimizer(getattr(tr, n).parameters()))
    return tr


def _o0_replay(seed):
    rng = np.random.default_rng(seed)
    lives = []
    for n, terminated in ((30, True), (40, False), (3, True)):
        recs = []
        for t in range(n + 1):
            acted = t < n
            recs.append({
                "obs": rng.integers(0, 255, (12, 84, 84), dtype=np.uint8),
                "action": int(rng.integers(0, A9)) if acted else 0,
                "reward": float(rng.integers(0, 2)) if t else 0.0,
                "target_policy": rng.dirichlet(np.ones(A9)),
                "root_value": float(rng.normal()), "target_value": 0.0,
                "has_policy_target": acted or not terminated,
                "born_step": int(rng.integers(0, 40000)),
                "priority": float(rng.uniform(0.1, 2.0)),
                "raw_reward": ()})
        recs[-1]["truncated"] = not terminated
        lives.append(recs)
    return lives


def _where(replay, rec):
    for i, life in enumerate(replay):
        for j, r in enumerate(life):
            if r is rec:
                return i, j
    raise AssertionError("a ref outside the replay")


def _same_batch(a, b, ra, rb):
    assert list(a) == list(b), "the flag-off batch must carry the same keys"
    for k in a:
        if k == "sample_refs":
            assert [_where(ra, x) for x in a[k]] == [_where(rb, x)
                                                     for x in b[k]]
        else:
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k


def _same_state(x, y):
    sx, sy = x.state_dict(), y.state_dict()
    assert list(sx) == list(sy)
    for k in sx:
        assert torch.equal(sx[k], sy[k]), k


def _same_opt(x, y):
    sx, sy = x.state_dict(), y.state_dict()
    assert sx["param_groups"] == sy["param_groups"]
    assert list(sx["state"]) == list(sy["state"])
    for k in sx["state"]:
        for f in sx["state"][k]:
            u, v = sx["state"][k][f], sy["state"][k][f]
            assert (torch.equal(u, v) if torch.is_tensor(u) else u == v), (k, f)


def test_flag_off_train_steps_are_the_pre_stage3_code_bit_for_bit(o0, pre_s3):
    assert muzero.OPTION_LENGTH == 1 and muzero.OPTION_EDGE is True
    old = pre_s3.muzero
    _copy_knobs(muzero, old)
    torch.manual_seed(0)
    new_nets = _nets(muzero)
    torch.manual_seed(0)
    harness = ablation.build_networks(o0)
    torch.manual_seed(0)
    old_nets = _nets(old)
    for k in new_nets:
        _same_state(new_nets[k], harness[k])
        _same_state(new_nets[k], old_nets[k])
        assert type(old_nets[k]).__module__ in ("networks_pre_s3",), k
    t_new, t_old = _trainer(muzero, new_nets), _trainer(old, old_nets)
    r_new, r_old = _o0_replay(11), _o0_replay(11)
    for step in range(2):
        seed_all(100 + step)
        b_old = t_old.finish_batch_paper(
            t_old.assemble_batch_paper(r_old, 12, 5, A9))
        s_old = rng_states()
        seed_all(100 + step)
        b_new = t_new.finish_batch_paper(
            t_new.assemble_batch_paper(r_new, 12, 5, A9))
        assert same_rng(s_old, rng_states())
        _same_batch(b_new, b_old, r_new, r_old)
        seed_all(200 + step)
        l_old = asyncio.run(t_old.train_step(b_old))
        s_old = rng_states()
        seed_all(200 + step)
        l_new = asyncio.run(t_new.train_step(b_new))
        assert same_rng(s_old, rng_states())
        assert l_new == l_old
        for n in ablation.NET_NAMES:
            pn = dict(getattr(t_new, n).named_parameters())
            po = dict(getattr(t_old, n).named_parameters())
            assert list(pn) == list(po)
            for k in pn:
                gn, go = pn[k].grad, po[k].grad
                assert (gn is None) == (go is None), (n, k)
                assert gn is None or torch.equal(gn, go), (n, k)
            _same_state(getattr(t_new, n), getattr(t_old, n))
        for o in ablation.OPT_NAMES:
            _same_opt(getattr(t_new, o), getattr(t_old, o))
        for n in ("target_representation", "target_prediction",
                  "target_dynamics", "target_value_prefix"):
            _same_state(getattr(t_new, n), getattr(t_old, n))
        assert torch.equal(t_new._last_step[1], t_old._last_step[1])
        assert [[r["priority"] for r in life] for life in r_new] == \
            [[r["priority"] for r in life] for life in r_old]
        assert [sorted(r) for life in r_new for r in life] == \
            [sorted(r) for life in r_old for r in life]
        t_new.training_steps += 1
        t_old.training_steps += 1
    assert not hasattr(t_new, "_option_last")


def test_flag_off_birth_search_and_records_are_the_pre_stage3_code(
        o0, pre_s3, monkeypatch):
    assert muzero.OPTION_LENGTH == 1
    monkeypatch.setattr(muzero, "ACT_SEARCH_SIMS", 8)
    old_ab = pre_s3.ablation
    for k in ("ACTIONS", "ENV_NAME", "FRAMESKIP", "DEVICE"):
        monkeypatch.setattr(old_ab, k, getattr(ablation, k))
    torch.manual_seed(0)
    run = make_run(o0, O0)
    dezero_policy(run.trainer)
    old_run = object.__new__(old_ab.Run)
    old_run.__dict__.update(copy.copy(run.__dict__))
    for terminated in (True, False):
        traj = life3(7, [1, 2, 3, 4, 5, 6, 7], seed=int(terminated))
        seed_all(5)
        out_old = old_ab.Run.paper_targets(old_run, traj, terminated,
                                           born_step=100)
        s_old = rng_states()
        seed_all(5)
        out_new = run.paper_targets(traj, terminated, born_step=100)
        assert same_rng(s_old, rng_states())
        assert len(out_new) == len(out_old)
        for x, y in zip(out_new, out_old):
            assert list(x) == list(y)
            for k in x:
                u, v = x[k], y[k]
                if isinstance(u, np.ndarray):
                    assert u.dtype == v.dtype and np.array_equal(u, v), k
                else:
                    assert u == v and type(u) is type(v), k
        assert np.array_equal(run._birth_priors, old_run._birth_priors)
        assert np.array_equal(run._birth_pmax, old_run._birth_pmax)
        assert run._pc_window == old_run._pc_window
        for k in ("birth_positions_total", "birth_positions_last"):
            assert run._birth_stats[k] == old_run._birth_stats[k]
        assert run._fresh_target_H == old_run._fresh_target_H
        w_new, w_old = run._beh_window, old_run._beh_window
        assert w_new["n"] == w_old["n"] and w_new["counts"] == w_old["counts"]
        assert w_new["sums"] == w_old["sums"]
    assert not hasattr(run, "_birth_options")
    assert not hasattr(run, "_birth_option_counts")
