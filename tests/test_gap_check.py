"""Test G, stage 1, the blind pilot (CURRICULUM "Next experiment
(pre-registered) -- Test G"). Pins: rho_w on hand-computed cases, ties and
singletons included; the residual statistic's -1/15 null and a hand case;
Spearman-Brown and the split-half read; the pilot's GO and STOP on both bars and
on undefined statistics; G6's effective games (T0's own row: 9 distinct scores,
6.4) and its single lengthening; the strata; the drops' order and counts; every
seeded draw; the opening's generator; the clip and the per-frame discount on a
hand example, delay rewards in the first base step; and, on the real game with no
snapshot, the raw frames against MaxAndSkipEnv, the loop against Run.play_game,
the stale-lives hazard in both directions, delays, the frame cap, a branch against
its raw replay and across a fresh process, a delay-0 branch against the main line
it left, the check that catches one that left it, the refusals. In a subprocess:
importing the probe changes nothing process-wide, and (slow, needing the snapshots)
a whole small pilot on T0 and T2."""
import hashlib
import inspect
import json
import os
import pathlib
import pickle
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBES = ROOT / "benchmarks" / "probes"
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(PROBES))
import ale_py  # noqa: E402
import muzero  # noqa: E402
import ablation  # noqa: E402
import probe_action_census as pc  # noqa: E402
import probe_gap_check as gc  # noqa: E402

VOCAB = ((0, 4), (1, 8), (2, 16), (3, 32), (4, 4), (2, 4), (3, 8))


def test_importing_the_probe_changes_nothing_process_wide():
    assert "probe_exhaustive_eval" not in gc.__dict__
    code = ("import os, sys; sys.path[:0] = [{p!r}, {e!r}, {r!r}]; "
            "import torch; threads = torch.get_num_threads(); "
            "import ablation; before = (ablation.ENV_NAME, ablation.ACTIONS); "
            "import probe_gap_check; "
            "assert (ablation.ENV_NAME, ablation.ACTIONS) == before; "
            "assert torch.get_num_threads() == threads; "
            "assert 'probe_exhaustive_eval' not in sys.modules; "
            "assert os.environ.get('PROBE_ACTING_STATES') is None; "
            "assert os.environ.get('PROBE_ENV') is None; "
            "assert os.environ.get('PROBE_THREADS') is None; "
            "print('ok')").format(p=str(PROBES), e=str(ROOT / "experiments"),
                                  r=str(ROOT))
    env = {k: v for k, v in os.environ.items()
           if k not in ("PROBE_ACTING_STATES", "PROBE_ENV", "PROBE_THREADS")}
    env["CUDA_VISIBLE_DEVICES"] = "-1"        # never the live run's GPU
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_the_registered_numbers():
    assert (gc.OPENING, gc.OPENING_LONG) == (80, 100)
    assert gc.STATES_PER_GAME == 6 and gc.STRATUM_STEPS == 20
    assert gc.GAMES == 240 and gc.PLAIN_GAMES == 30 and gc.N_STRATA == 20
    assert gc.PER_STRATUM == 3 and gc.G6_BAR == 30 and not gc.SMOKE
    assert gc.DELAYS == tuple(range(16)) and gc.HALF == 8
    assert gc.FRAME_CAP == 1536 and gc.BASE == 4
    assert gc.DISCOUNTS == (0.997, 0.99, 0.98, 0.95)
    assert gc.SCALES[0] == "clip" and gc.LAGS == (1, 4, 8)
    assert gc.LAG_BAR == 0.3 and gc.RELIABILITY_BAR == 0.3
    assert gc.LAG_NULL == pytest.approx(-1 / 15)
    assert gc.FREE_GRID == tuple((a, 32) for a in range(9))
    assert set(gc.FREE_GRID) < set(pc.GRID)
    assert gc.PLAYERS["T0"]["iteration"] == 45
    assert gc.PLAYERS["T2"]["iteration"] == 42


# --------------------------------------------------------------------------
# rho_w
# --------------------------------------------------------------------------
def test_centred_ranks_by_hand():
    assert gc.centred_ranks([10, 20, 30]) == pytest.approx([-1 / 3, 0, 1 / 3])
    assert gc.centred_ranks([30, 10]) == pytest.approx([0.25, -0.25])
    # ties averaged: ranks 3.5, 3.5, 1, 2 of 4
    assert gc.centred_ranks([5, 5, 1, 2]) == pytest.approx(
        [0.25, 0.25, -0.375, -0.125])
    assert gc.centred_ranks([7, 7, 7]) == pytest.approx([0, 0, 0])
    assert gc.centred_ranks([3]) == pytest.approx([0])


def test_rho_w_two_strata_by_hand():
    # stratum a: x ranks -1/3, 0, 1/3; y ranks -1/3, 1/3, 0
    # stratum b: x ranks -1/4, 1/4; y ranks 1/4, -1/4
    # sum xy = 1/9 - 1/8 = -1/72; sum xx = sum yy = 2/9 + 1/8 = 25/72
    x = [1, 2, 3, 1, 2]
    y = [1, 3, 2, 2, 1]
    assert gc.rho_w(x, y, list("aaabb")) == pytest.approx(-1 / 25)


def test_rho_w_with_ties_by_hand():
    # x ranks 1.5, 1.5, 3, 4; y ranks 1, 2, 3, 4, both of n = 4
    # sum xy = sum xx = 0.28125, sum yy = 0.3125, so rho = sqrt(0.9)
    assert gc.rho_w([1, 1, 2, 3], [1, 2, 3, 4], [0] * 4) == pytest.approx(
        np.sqrt(0.9))


def test_rho_w_is_inside_strata_only():
    """A level shared by a stratum means nothing: x and y agree across strata
    and run opposite inside them."""
    x = [1, 2, 3, 101, 102, 103]
    y = [3, 2, 1, 103, 102, 101]
    assert gc.rho_w(x, y, [0, 0, 0, 1, 1, 1]) == pytest.approx(-1)
    assert gc.rho_w(x, x, [0, 0, 0, 1, 1, 1]) == pytest.approx(1)


def test_a_stratum_of_one_drops_out():
    x, y, s = [1, 2, 3, 1, 2], [1, 3, 2, 2, 1], list("aaabb")
    assert gc.rho_w(x + [50], y + [-50], s + ["c"]) == pytest.approx(-1 / 25)
    assert gc.rho_w([1, 2], [2, 1], ["a", "b"]) is None
    assert gc.rho_w([], [], []) is None


def test_rho_w_without_spread_is_undefined():
    assert gc.rho_w([1, 1, 2, 2], [1, 2, 1, 2], [0, 0, 1, 1]) is None
    assert gc.rho_w([1, 2], [5, 5], [0, 0]) is None


# --------------------------------------------------------------------------
# the residual statistic, the split half, the verdict
# --------------------------------------------------------------------------
def test_the_residual_null_is_minus_one_over_fifteen():
    rng = np.random.default_rng(1)
    returns = rng.normal(5.0, 2.0, size=(60000, 16))
    returns += rng.normal(0, 30, size=(60000, 1))     # a state's level is out
    for lag in gc.LAGS:
        assert gc.lag_correlation(returns, lag) == pytest.approx(-1 / 15,
                                                                 abs=0.004)


def test_the_residual_statistic_by_hand():
    # one state, delays [0, 2, 4, 6] then lag 2: residuals -3, -1, 1, 3,
    # pairs (-3, 1), (-1, 3): x and y rise together, r = 1
    assert gc.lag_correlation([[0, 2, 4, 6]], 2) == pytest.approx(1)
    # two states: residuals [-1, 1, -1, 1] and [1, -1, 1, -1] at lag 1:
    # every pair opposes
    r = gc.lag_correlation([[0, 2, 0, 2], [2, 0, 2, 0]], 1)
    assert r == pytest.approx(-1)
    # pooled pairs, not a mean of per-state correlations, at lag 1:
    # residuals -1.5, -.5, .5, 1.5 and 3, 1, -1, -3; pairs by hand
    x = np.array([-1.5, -.5, .5, 3, 1, -1])
    y = np.array([-.5, .5, 1.5, 1, -1, -3])
    assert gc.lag_correlation([[0, 1, 2, 3], [6, 4, 2, 0]], 1) == pytest.approx(
        np.corrcoef(x, y)[0, 1])


def test_neighbours_show_up_at_their_lag():
    """A noise shared by neighbouring delays, as a held phase would give."""
    rng = np.random.default_rng(2)
    e = rng.normal(size=(40000, 17))
    returns = e[:, :-1] + e[:, 1:]
    assert gc.lag_correlation(returns, 1) > 0.3
    assert gc.lag_correlation(returns, 8) < 0.3


def test_no_residuals_is_undefined():
    assert gc.lag_correlation(np.ones((5, 16)), 1) is None


def test_spearman_brown():
    assert gc.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert gc.spearman_brown(0.0) == 0
    assert gc.spearman_brown(1.0) == pytest.approx(1)
    assert gc.spearman_brown(-0.5) == pytest.approx(-2)
    assert gc.spearman_brown(-1.0) is None and gc.spearman_brown(None) is None


def test_the_split_is_delays_0_7_against_8_15():
    strata = [0, 0, 0, 1, 1, 1]
    gap = np.zeros((6, 16))
    level = np.array([1, 2, 3, 1, 3, 2.0])[:, None]
    gap[:, :8] = level
    gap[:, 8:] = level
    assert gc.split_half(gap, strata) == pytest.approx((1, 1))
    gap[:, 8:] = -level
    r, reliability = gc.split_half(gap, strata)
    assert r == pytest.approx(-1) and reliability is None
    # the means, not one delay of each half
    gap = np.zeros((6, 16))
    gap[:, 3] = level[:, 0] * 8
    gap[:, 12] = level[:, 0] * 8
    assert gc.split_half(gap, strata)[0] == pytest.approx(1)
    # interleaving the delays must not read as the same split
    gap = np.zeros((6, 16))
    gap[:, ::2] = level
    gap[:, 1::2] = -level
    assert gc.split_half(gap, strata)[0] is None


def test_noisy_halves_read_as_rank_correlations_of_three():
    """Halves at a true Pearson 0.5 inside strata of 3: the rank correlation
    of n normals has expectation 6 / (pi (n + 1)) (asin r + (n - 2) asin(r / 2)),
    0.371 at n = 3, so rho_w reads below the Pearson (the pilot's strata hold 3
    each, which is part of why its bar is lower)."""
    rng = np.random.default_rng(3)
    n, strata = 3000, np.repeat(np.arange(1000), 3)
    truth = rng.normal(size=(n, 1))
    gap = truth + rng.normal(0, np.sqrt(8), size=(n, 16))   # each half: var 1
    r, reliability = gc.split_half(gap, strata)
    want = 6 / (np.pi * 4) * (np.arcsin(0.5) + np.arcsin(0.25))
    assert r == pytest.approx(want, abs=0.04)
    assert reliability == pytest.approx(gc.spearman_brown(r))


GOOD = {"T0": {1: 0.1, 4: -0.2, 8: 0.3}, "T2": {1: 0.0, 4: 0.3, 8: -0.1}}


def test_go_needs_every_bar():
    assert gc.pilot_verdict(GOOD, 0.3) == ("GO", [])
    assert gc.pilot_verdict(GOOD, 0.9) == ("GO", [])


def test_each_correlation_can_stop_it():
    for player in ("T0", "T2"):
        for lag in gc.LAGS:
            c = {p: dict(v) for p, v in GOOD.items()}
            c[player][lag] = 0.31
            verdict, reasons = gc.pilot_verdict(c, 0.9)
            assert verdict == "STOP" and len(reasons) == 1
            assert f"{player} lag {lag}" in reasons[0]


def test_the_reliability_bar_stops_it():
    verdict, reasons = gc.pilot_verdict(GOOD, 0.299)
    assert verdict == "STOP" and "G5" in reasons[0]


def test_undefined_is_never_go():
    c = {p: dict(v) for p, v in GOOD.items()}
    c["T2"][4] = None
    assert gc.pilot_verdict(c, 0.9)[0] == "STOP"
    assert gc.pilot_verdict(GOOD, None)[0] == "STOP"


def test_the_pilot_read_end_to_end():
    rng = np.random.default_rng(4)
    keys = [(0, k) for k in range(20) for _ in range(3)]
    level = rng.normal(0, 3, size=(60, 1))
    t0 = rng.normal(0, 1, size=(60, 16))
    t2 = t0 * 0 + level + rng.normal(0, 1, size=(60, 16))
    out = gc.pilot_read(t0, t2, keys)
    assert out["verdict"] == "GO" and out["reliability"] > 0.8
    assert out["null"] == pytest.approx(-1 / 15)
    assert set(out["correlations"]) == {"T0", "T2"}
    assert all(set(v) == {1, 4, 8} for v in out["correlations"].values())
    # a phase of 4 delays: lag 4 stops it, for that player
    t0 = np.tile(np.array([3.0, 0, 0, 0] * 4), (60, 1)) + rng.normal(
        0, .1, size=(60, 16))
    out = gc.pilot_read(t0, t2, keys)
    assert out["verdict"] == "STOP"
    assert any("T0 lag 4" in r for r in out["reasons"])


# --------------------------------------------------------------------------
# G6
# --------------------------------------------------------------------------
def test_effective_games_on_t0s_row():
    """The registration's own figure: T0's row at iteration 45."""
    counts = {570: 2, 670: 7, 700: 1, 710: 5, 730: 2, 790: 6, 840: 2, 870: 4,
              930: 1}
    scores = [s for s, c in counts.items() for _ in range(c)]
    assert len(scores) == 30
    assert gc.effective_games(scores) == pytest.approx(6.43, abs=0.01)


def test_effective_games_on_synthetic_games():
    assert gc.effective_games(range(240)) == pytest.approx(240)
    assert gc.effective_games([1.0] * 240) == pytest.approx(1)
    assert gc.effective_games(list(range(30)) * 8) == pytest.approx(30)
    # lengths play no part, near repeats on one score are one game
    assert gc.effective_games([100.0] * 120 + [200.0] * 120) == pytest.approx(2)


def test_g6_lengthens_once_then_voids():
    assert gc.g6_next([30.0]) == "pass"
    assert gc.g6_next([29.99]) == "lengthen"
    assert gc.g6_next([12.0, 30.0]) == "pass"
    assert gc.g6_next([12.0, 29.0]) == "void"


# --------------------------------------------------------------------------
# strata, drops, draws
# --------------------------------------------------------------------------
def test_strata_are_life_by_twenty_base_steps():
    assert gc.stratum(0, 80) == (0, 4)
    assert gc.stratum(0, 99) == (0, 4) and gc.stratum(0, 100) == (0, 5)
    assert gc.stratum(2, 0) == (2, 0) and gc.stratum(2, 19) == (2, 0)
    assert gc.stratum(2, 20) == (2, 1)
    assert gc.stratum(1, 5) != gc.stratum(2, 5)


def _state(game, tag, free=False):
    stack = np.full((12, 84, 84), tag, dtype=np.uint8)
    return {"game": game, "stack": stack, "free": free}


def test_the_drops_count_and_keep_the_first():
    states = [_state(0, 1), _state(0, 2, free=True), _state(1, 1),
              _state(2, 3), _state(3, 1), _state(4, 3)]
    kept, counts = gc.select_states(states, lambda s: s["free"])
    assert [s["game"] for s in kept] == [0, 2]
    assert kept[0]["merged"] == [1, 3] and kept[1]["merged"] == [4]
    assert counts == {"sampled": 6, "free": 1, "merged": 3, "kept": 2}


def test_free_goes_first():
    """The registration's order: a FREE first copy does not take its twin."""
    states = [_state(0, 1, free=True), _state(1, 1)]
    kept, counts = gc.select_states(states, lambda s: s["free"])
    assert [s["game"] for s in kept] == [1]
    assert counts["free"] == 1 and counts["merged"] == 0


def _keys(sizes):
    return [(0, k) for k, n in enumerate(sizes) for _ in range(n)]


def test_the_draw_takes_three_in_each_of_twenty():
    keys = _keys([3 + (k % 5) for k in range(30)])
    chosen, eligible = gc.draw_pilot(keys)
    assert len(chosen) == 60 and len(set(chosen)) == 60
    assert len(eligible) == 30
    per = {}
    for i in chosen:
        per[keys[i]] = per.get(keys[i], 0) + 1
    assert len(per) == 20 and set(per.values()) == {3}


def test_small_strata_are_not_drawn():
    keys = _keys([2] * 40 + [3] * 20)
    chosen, eligible = gc.draw_pilot(keys)
    assert len(eligible) == 20
    assert all(keys[i][1] >= 40 for i in chosen)


def test_fewer_than_twenty_strata_is_void():
    chosen, eligible = gc.draw_pilot(_keys([2] * 40 + [3] * 19))
    assert chosen is None and len(eligible) == 19


def test_every_draw_is_deterministic():
    keys = _keys([3 + (k % 5) for k in range(30)])
    assert gc.draw_pilot(keys) == gc.draw_pilot(keys)
    assert gc.draw_pilot(keys)[0] != gc.draw_pilot(keys[3:])[0]
    assert gc.draw_game_states(7, 100) == gc.draw_game_states(7, 100)
    assert gc.draw_game_states(7, 100) != gc.draw_game_states(8, 100)
    assert gc.g4_delays(60) == gc.g4_delays(60)
    assert gc.opening_moves(5, 80) == gc.opening_moves(5, 80)
    # pinned values: a change to a seed or a generator shows here
    assert gc.opening_moves(0, 5) == [6, 6, 0, 4, 8]
    assert gc.draw_game_states(0, 100) == [13, 29, 41, 58, 74, 85]
    assert gc.g4_delays(6) == [14, 12, 2, 0, 13, 1]


def test_a_games_draw_takes_six_distinct_decisions():
    for n in (6, 7, 500):
        picks = gc.draw_game_states(3, n)
        assert len(picks) == len(set(picks)) == 6
        assert picks == sorted(picks) and 0 <= picks[0] and picks[-1] < n
    assert gc.draw_game_states(3, 4) == [0, 1, 2, 3]
    assert gc.draw_game_states(3, 0) == []


def test_g4s_delays_reach_the_whole_range():
    picks = gc.g4_delays(60)
    assert len(picks) == 60 and set(picks) <= set(gc.DELAYS)
    assert len(set(picks)) > 8


def test_the_opening_is_peek_readys():
    for seed in (0, 1, 239):
        r = random.Random(seed)
        assert gc.opening_moves(seed, 100) == [r.randrange(9)
                                               for _ in range(100)]
        assert gc.opening_moves(seed, 100)[:80] == gc.opening_moves(seed, 80)
    assert [gc.game_noops(s) for s in (0, 1, 29, 30, 239)] == [1, 2, 30, 1, 30]
    assert {gc.game_noops(s) for s in range(240)} == set(range(1, 31))


# --------------------------------------------------------------------------
# the clip and the discounts
# --------------------------------------------------------------------------
def test_the_clip_and_discount_sums_by_hand():
    raw = [20.0, 0.0, 50.0, 10.0]          # the delay's 10 already in the 20
    clip = [1.0, 0.0, 1.0, 1.0]            # 20 / 10, 50 / 10 reach the ceiling
    g = 0.99
    w = [1.0, g ** 4, g ** 8, g ** 12]     # per frame, 4 frames a base step
    heads = gc.head_returns(raw)
    assert heads.shape == (2, 4)
    assert heads[0, 1] == pytest.approx(sum(a * b for a, b in zip(clip, w)))
    assert heads[1, 1] == pytest.approx(sum(a * b for a, b in zip(raw, w)))
    assert heads[0, 1] == pytest.approx(1 + 0.99 ** 8 + 0.99 ** 12)
    assert heads[1, 1] == pytest.approx(20 + 50 * 0.99 ** 8 + 10 * 0.99 ** 12)
    assert heads[0, 0] == pytest.approx(1 + 0.997 ** 8 + 0.997 ** 12)
    assert heads[1, 3] == pytest.approx(20 + 50 * 0.95 ** 8 + 10 * 0.95 ** 12)


def test_a_sub_unit_reward_stays_a_fraction_of_a_unit():
    assert gc.head_returns([5.0])[0, 0] == pytest.approx(0.5)


def test_the_clip_is_per_base_step_not_per_branch():
    """Two pellets in one base step clip to one unit; in two, to two."""
    assert gc.head_returns([20.0])[0, 3] == pytest.approx(1)
    assert gc.head_returns([10.0, 10.0])[0, 3] == pytest.approx(1 + 0.95 ** 4)


def test_primary_is_the_clip_at_0997():
    b = {"rewards": [10.0, 200.0]}
    assert gc.primary_returns([[b, b]]) == pytest.approx(
        np.array([[1 + 0.997 ** 4] * 2]))


# --------------------------------------------------------------------------
# the real game, no snapshot
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def env():
    e = gc.raw_env()
    yield e
    e.close()


@pytest.fixture
def vocab(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", VOCAB)
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", False)


def inference(stack):
    """A deterministic stand-in for a policy: every stack picks a token."""
    h = int.from_bytes(hashlib.blake2b(stack.tobytes(),
                                       digest_size=4).digest(), "little")
    p = np.zeros(len(muzero.TOKEN_VOCAB))
    p[h % len(p)] = 1.0
    return p, 0.0


def decide(stack):
    return int(inference(stack)[0].argmax())


def walk(stack):
    return 2                                     # (2, 16), right, 4 base steps


def state_after(env, frames, seed=0):
    """A live state some way into a game, its stack carried."""
    moves = gc.opening_moves(seed, 80)
    stack, *_ = gc.start_game(env, gc.game_noops(seed), moves)
    ale = env.unwrapped.ale
    stack, _, lost = gc.raw_frames(ale, env.unwrapped._action_set[2], frames,
                                   stack)
    assert not lost
    return ale.cloneState(), stack


def death_states(env):
    """Standing still from the start, in a 3-life game: a state just before
    the first life goes, and one just after it. (Lives fall by frame 826.)"""
    u = env.unwrapped
    ale = u.ale
    u.reset(seed=gc.EVAL_SEED)
    u.reset()
    before = None
    while ale.lives() == 3:
        before = ale.cloneState()
        ale.act(u._action_set[0])
    return before, ale.cloneState()


def test_raw_frames_pool_as_max_and_skip(env):
    """The same four inputs through raw_frames and through the wrappers."""
    ale, actions = env.unwrapped.ale, env.unwrapped._action_set
    start = state_after(env, 40)[0]
    frame = np.zeros((3, 84, 84), dtype=np.uint8)
    stack = ablation.initial_frame_stack(frame)
    gc.restore(env, start)
    wrapper, reward = stack, 0.0
    for a in (1, 3, 2):
        obs, r, *_ = env.step(a)
        reward += r
        wrapper = ablation.push_frame(wrapper, ablation.preprocess_frame(
            obs, ablation.IMAGE_RESHAPE))
    ale.restoreState(start)
    raw, total = stack, 0.0
    for a in (1, 3, 2):
        raw, r, lost = gc.raw_frames(ale, actions[a], 4, raw)
        total += r
        assert not lost
    assert np.array_equal(raw, wrapper) and total == reward


def test_a_delay_pushes_a_frame_every_four_frames(env):
    ale, noop = env.unwrapped.ale, env.unwrapped._action_set[0]
    start, stack = state_after(env, 40)
    frames = {}
    for d in (0, 3, 4, 7, 8, 15):
        ale.restoreState(start)
        frames[d] = gc.raw_frames(ale, noop, d, stack)[0]
    assert frames[0] is stack and frames[3] is stack
    assert np.array_equal(frames[4], frames[7])
    assert not np.array_equal(frames[4], stack)
    assert np.array_equal(frames[8][:-3], np.concatenate(
        [frames[4][3:-3], frames[8][6:9]]))
    # at 15 delay frames, three have entered: the oldest of `stack` is left
    assert np.array_equal(frames[15][:3], stack[9:])


def test_the_loop_is_run_play_game(env, vocab):
    """A plain game through the probe's loop, against Run.play_episode and
    play_game themselves, on the real game: the G1 cross-check in miniature."""
    from atari_wrappers import NoopResetEnv
    for noops in (1, 2, 3):
        ne = env
        while not isinstance(ne, NoopResetEnv):
            ne = ne.env

        def fixed(self, **kw):
            obs, info = self.env.reset(**kw)
            for _ in range(noops):
                obs, _, _, _, info = self.env.step(0)
            return obs, info

        ne.reset = fixed.__get__(ne)
        count = [0]
        shell = SimpleNamespace(env=env, searcher=SimpleNamespace(
            network_manager=SimpleNamespace(initial_inference=inference)))

        def episode(**kw):
            out = ablation.Run.play_episode(shell, **kw)
            count[0] += len(out[0]) - 1
            return out

        shell.play_episode = episode
        try:
            env.was_real_done = True
            env.reset(seed=gc.EVAL_SEED)
            ret, steps, term = ablation.Run.play_game(shell, cap=gc.LIFE_CAP)
        finally:
            del ne.reset
        assert term and steps > 0
        assert gc.plain_game(env, decide, noops) == (ret, steps, term, count[0])


def test_a_main_line_records_what_a_state_needs(env, vocab):
    moves = gc.opening_moves(4, 80)
    rec, held = gc.main_game(env, decide, 4, moves)
    assert rec["noops"] == 5 and rec["moves"] == 80
    assert rec["terminated"] and rec["life"][-1] == 2
    assert rec["decisions"] == len(held) == len(rec["life"]) == len(rec["steps"])
    # base steps count from the life's start, the opening included
    assert rec["steps"][0] == 80
    assert rec["life"][0] == 0 and rec["life"] == sorted(rec["life"])
    first_of = [rec["life"].index(k) for k in (1, 2)]
    assert all(rec["steps"][i] == 0 for i in first_of)
    assert rec["length"] == 80 + sum(len(v) for v in rec["rewards"].values())
    # the score is the policy's: the opening's reward is held apart from it
    assert rec["score"] == sum(sum(v) for v in rec["rewards"].values())
    assert rec["opening_reward"] == gc.start_game(env, 5, moves)[3] > 0
    # each decision's stack and state, the stack of the decision it played on
    state, stack = held[0]
    assert isinstance(state, bytes) and stack.shape == (12, 84, 84)
    assert stack.dtype == np.uint8
    assert rec["digests"][0] == pc._digest(stack.tobytes())
    assert rec["tokens"][0] == decide(stack)


def test_the_opening_carries_a_pooled_stack(env):
    """Four distinct pooled frames at the first decision, where a tiled frame
    would give four copies of one."""
    moves = gc.opening_moves(0, 80)
    stack, life, steps, _ = gc.start_game(env, 1, moves)
    assert (life, steps) == (0, 80)
    assert env.lives == 3 and env.was_real_done is False
    # replayed through the wrappers' own pooling
    ale, actions = env.unwrapped.ale, env.unwrapped._action_set
    u = env.unwrapped
    u.reset(seed=gc.EVAL_SEED)
    u.reset()
    ale.act(actions[0])
    frame = ablation.preprocess_frame(ale.getScreenRGB(),
                                      ablation.IMAGE_RESHAPE)
    want = ablation.initial_frame_stack(frame)
    env.lives, env.was_real_done = 3, False
    for a in moves:
        obs, *_ = env.step(a)
        want = ablation.push_frame(want, ablation.preprocess_frame(
            obs, ablation.IMAGE_RESHAPE))
    assert np.array_equal(stack, want)


def test_a_life_lost_in_the_opening_plays_on(env):
    """Registered: the game plays on. Standing still loses the first life
    near frame 826, so a NOOP opening of 250 moves crosses it."""
    stack, life, steps, _ = gc.start_game(env, 1, [0] * 250)
    assert life == 1 and 0 < steps < 250
    assert env.lives == 2 and env.was_real_done is False


# the stale-lives regression -------------------------------------------------
def test_a_stale_count_ends_a_life_that_goes_on(env):
    """The hazard, shown on the wrappers: a 2-lives state restored under a
    count of 3 reads its first step as a death, and restore() does not."""
    _, after = death_states(env)
    ale = env.unwrapped.ale
    env.lives = 3
    ale.restoreState(after)
    assert ale.lives() == 2
    term = env.step(0)[2]
    assert term is True                          # the bug, stale

    gc.restore(env, after)
    assert env.lives == 2
    assert env.step(0)[2] is False               # the same step, reset


def test_a_stale_count_misses_a_real_death(env):
    """The other way: the state just before the first life goes, under a
    count of 2 left by a later life's branch. The death reads as nothing."""
    before, _ = death_states(env)
    ale = env.unwrapped.ale
    env.lives = 2
    ale.restoreState(before)
    assert ale.lives() == 3
    assert env.step(0)[2] is False               # the bug, stale
    assert ale.lives() == 2

    gc.restore(env, before)
    assert env.step(0)[2] is True                # the same step, reset


def test_branches_read_true_lives_in_any_order(env, vocab):
    """The same state's branch is the same after a branch from another life
    left its count behind, and it dies where it dies."""
    before, after = death_states(env)
    frame = np.zeros((3, 84, 84), dtype=np.uint8)
    stack = ablation.initial_frame_stack(frame)
    late = gc.branch(env, walk, after, stack, 0, cap=64)
    early = gc.branch(env, lambda s: 0, before, stack, 0)
    assert early["ended"] == "life" and len(early["rewards"]) == 1
    again = gc.branch(env, walk, after, stack, 0, cap=64)
    assert gc.same_branch(late, again) == []
    assert late["ended"] == "cap" and len(late["rewards"]) == 16
    early2 = gc.branch(env, lambda s: 0, before, stack, 0)
    assert gc.same_branch(early, early2) == []


def test_a_game_over_restores_clean(env, vocab):
    """A branch that ends the game leaves nothing behind for the next."""
    u = env.unwrapped
    ale = u.ale
    u.reset(seed=gc.EVAL_SEED)
    u.reset()
    last = None
    while not ale.game_over():
        last = ale.cloneState()
        ale.act(u._action_set[0])
    stack = ablation.initial_frame_stack(np.zeros((3, 84, 84), dtype=np.uint8))
    a = gc.branch(env, lambda s: 0, last, stack, 0)
    assert a["ended"] == "life" and env.was_real_done
    assert gc.same_branch(a, gc.branch(env, lambda s: 0, last, stack, 0)) == []


# delays and the cap ---------------------------------------------------------
def test_a_life_lost_during_a_delay_ends_the_branch(env, vocab):
    before, _ = death_states(env)
    stack = ablation.initial_frame_stack(np.zeros((3, 84, 84), dtype=np.uint8))
    b = gc.branch(env, walk, before, stack, 5)
    assert b["ended"] == "delay" and b["tokens"] == []
    assert b["frames"] == 1 and len(b["rewards"]) == 1
    assert gc.primary_returns([[b]])[0, 0] == b["rewards"][0] / 10


def test_delay_rewards_count_with_the_first_base_step(env, vocab):
    """On a state where the NOOP frames themselves score: a walk until
    a delay of 15 collects a pellet, then the branch's first base step holds
    the delay's reward plus its own."""
    ale, u = env.unwrapped.ale, env.unwrapped
    noop = u._action_set[0]
    for frames in range(0, 600, 4):
        start, stack = state_after(env, frames)
        ale.restoreState(start)
        delay_reward = gc.raw_frames(ale, noop, 15, stack)[1]
        if delay_reward > 0:
            break
    assert delay_reward > 0
    b = gc.branch(env, lambda s: 0, start, stack, 15, cap=15 + 8)
    ale.restoreState(start)
    gc.raw_frames(ale, noop, 15, stack)
    own = sum(float(ale.act(noop)) for _ in range(4))
    assert len(b["rewards"]) == 2 and b["ended"] == "cap"
    assert b["rewards"][0] == delay_reward + own


def test_the_cap_counts_frames_with_the_delay(env, vocab):
    start, stack = state_after(env, 40)
    b = gc.branch(env, walk, start, stack, 0, cap=64)
    assert len(b["rewards"]) == 16 and b["ended"] == "cap"
    assert b["frames"] == 64
    b = gc.branch(env, walk, start, stack, 3, cap=64)
    assert len(b["rewards"]) == 15 and b["ended"] == "cap"    # 3 + 15 x 4 = 63
    b = gc.branch(env, walk, start, stack, 4, cap=64)
    assert len(b["rewards"]) == 15 and b["ended"] == "cap"
    # a token crossing the cap is cut there: 12 base steps of a 16-frame token
    b = gc.branch(env, walk, start, stack, 0, cap=48 + 8)
    assert len(b["rewards"]) == 14 and b["frames"] == 64


def test_a_branch_is_capped_at_the_registered_frames_by_default():
    cap = inspect.signature(gc.branch).parameters["cap"].default
    assert cap == gc.FRAME_CAP == 1536


def test_the_policy_redecides_on_the_delayed_stack(env, vocab):
    """Delay 4 feeds the decision a stack delay 3 never saw."""
    start, stack = state_after(env, 40)
    seen = {}
    for d in (3, 4):
        def spy(s, d=d):
            seen.setdefault(d, s)
            return 0
        gc.branch(env, spy, start, stack, d, cap=32)
    assert np.array_equal(seen[3], stack)
    assert not np.array_equal(seen[4], stack)


# replays and processes -------------------------------------------------------
def test_a_branch_equals_its_raw_replay(env, vocab):
    start, stack = state_after(env, 200)
    for d in (0, 6, 15):
        b = gc.branch(env, decide, start, stack, d)
        assert len(gc.branch_inputs(list(env.unwrapped._action_set), b)) \
            == b["frames"]
        frames, end = gc.frames_of(env, start, b)
        assert end == b["end"]
        assert frames == gc.frames_of(env, start, b)[0]
        again = gc.branch(env, decide, start, stack, d)
        assert gc.same_branch(b, again) == []


def test_same_branch_names_what_differs():
    a = {"delay": 1, "rewards": [1.0], "ended": "life", "tokens": [2],
         "digests": [b"x"], "frames": 9, "end": b"y"}
    assert gc.same_branch(a, dict(a)) == []
    assert gc.same_branch(a, {**a, "rewards": [2.0], "end": b"z"}) == [
        "rewards", "end"]


_FRESH = r"""
import json, pickle, sys
sys.path[:0] = [{probes!r}, {exp!r}, {root!r}]
import ale_py, muzero
import probe_gap_check as gc
muzero.TOKEN_VOCAB = {vocab!r}
import hashlib
import numpy as np
def decide(stack):
    h = int.from_bytes(hashlib.blake2b(stack.tobytes(), digest_size=4).digest(), "little")
    return h % len(muzero.TOKEN_VOCAB)
data = pickle.load(open({path!r}, "rb"))
env = gc.raw_env()
state = ale_py.ALEState(data["state"])
out = []
for d in data["delays"]:
    b = gc.branch(env, decide, state, data["stack"], d)
    b["replay"] = gc.frames_of(env, state, b)
    out.append(b)
pickle.dump(out, open({res!r}, "wb"))
"""


def test_a_restored_branch_gives_the_same_frames_in_a_fresh_process(
        env, vocab, tmp_path):
    """G4's mechanism: a state crossing as ALEState(bytes), the raw frames and
    the end state equal, the emulator built anew."""
    start, stack = state_after(env, 200, seed=3)
    delays = (0, 5, 12)
    here = []
    for d in delays:
        b = gc.branch(env, decide, start, stack, d)
        b["replay"] = gc.frames_of(env, start, b)
        here.append(b)
    src, res = tmp_path / "state.pkl", tmp_path / "out.pkl"
    with open(src, "wb") as f:
        pickle.dump({"state": start.serialize(), "stack": stack,
                     "delays": delays}, f)
    code = _FRESH.format(probes=str(PROBES), exp=str(ROOT / "experiments"),
                         root=str(ROOT), vocab=VOCAB, path=str(src),
                         res=str(res))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    with open(res, "rb") as f:
        there = pickle.load(f)
    for a, b in zip(here, there):
        assert gc.same_branch(a, b) == [] and a["replay"] == b["replay"]


def test_a_zero_delay_branch_is_the_main_line(env, vocab):
    """The registered check's strong form, on a stand-in policy: T0's delay-0
    branch from a sampled decision repeats the main line, stacks included."""
    rec, held = gc.main_game(env, decide, 2, gc.opening_moves(2, 80))
    picks = gc.draw_game_states(2, len(held))
    states = [{"game": 0, "decision": i} for i in picks]
    zero = [gc.branch(env, decide, ale_py.ALEState(held[i][0]), held[i][1], 0)
            for i in picks]
    assert gc.main_line_reasons(states, [rec], zero) == []
    assert any(b["ended"] == "life" for b in zero)
    # the check is live: a changed reward, a changed token, a short branch
    changed = [dict(b) for b in zero]
    changed[0] = {**zero[0], "rewards": [zero[0]["rewards"][0] + 10]
                  + zero[0]["rewards"][1:]}
    assert "rewards" in gc.main_line_reasons(states, [rec], changed)[0]
    changed[0] = {**zero[0], "tokens": [zero[0]["tokens"][0] + 1]
                  + zero[0]["tokens"][1:]}
    assert "decisions" in gc.main_line_reasons(states, [rec], changed)[0]
    ended = next(k for k, b in enumerate(zero) if b["ended"] == "life")
    changed = [dict(b) for b in zero]
    changed[ended] = {**zero[ended], "rewards": zero[ended]["rewards"][:-1]}
    assert gc.main_line_reasons(states, [rec], changed)


def test_continuation_reads_inside_one_life():
    game = {"life": [0, 0, 1, 1, 1], "steps": [80, 84, 0, 2, 6],
            "tokens": [1, 2, 3, 4, 5], "digests": list("abcde"),
            "rewards": {0: [1.0] * 8, 1: [0, 1, 2, 3, 4, 5, 6, 7.0]}}
    assert gc.continuation(game, 0) == ([1, 2], ["a", "b"], [1.0] * 8)
    assert gc.continuation(game, 1) == ([2], ["b"], [1.0] * 4)
    assert gc.continuation(game, 3) == ([4, 5], ["d", "e"],
                                        [2, 3, 4, 5, 6, 7.0])


# --------------------------------------------------------------------------
# refusals, files
# --------------------------------------------------------------------------
def test_a_snapshot_that_is_not_the_player_is_refused():
    t0 = {"TOKEN_VOCAB": VOCAB * 5 + VOCAB[:1], "REWARD_UNIT": 10.0,
          "REWARD_GROWTH_START": 1.0, "REWARD_GROWTH_CEILING": 1.0}
    gc.check_player("T0", t0)
    with pytest.raises(SystemExit):
        gc.check_player("T2", t0)
    with pytest.raises(SystemExit):
        gc.check_player("T0", {**t0, "REWARD_GROWTH_CEILING": 17.4})
    with pytest.raises(SystemExit):
        gc.check_player("T0", {**t0, "REWARD_UNIT": 25.0})
    with pytest.raises(SystemExit):
        gc.check_player("T0", {**t0, "TOKEN_VOCAB": VOCAB})


def test_files_write_whole(tmp_path):
    gc.write(tmp_path / "a.json", {"x": 1})
    gc.write(tmp_path / "b.pkl", {"y": b"z"})
    assert gc.read(tmp_path / "a.json") == {"x": 1}
    assert gc.read(tmp_path / "b.pkl") == {"y": b"z"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "b.pkl"]


def test_a_finished_pilot_is_never_overwritten(tmp_path):
    (tmp_path / "pilot.json").write_text("{}")
    with pytest.raises(SystemExit, match="never overwritten"):
        gc.pilot(tmp_path)


# --------------------------------------------------------------------------
# end to end: a small pilot on the real players
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_a_small_pilot_on_the_real_players(tmp_path):
    try:
        for name in gc.PLAYERS:
            gc.snapshot(name)
    except SystemExit:
        pytest.skip("the players' snapshots are not on this machine")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "GAP_GAMES": "24",
           "GAP_PLAIN_GAMES": "3", "GAP_STRATA": "2", "GAP_G6_BAR": "2"}
    r = subprocess.run([sys.executable, str(PROBES / "probe_gap_check.py"),
                        "pilot", "--out", str(tmp_path)], cwd=str(ROOT),
                       env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    print(r.stdout)
    rec = json.loads((tmp_path / "pilot.json").read_text())
    assert rec["smoke"] is True and rec["verdict"] in ("GO", "STOP"), rec
    assert "PILOT: " + rec["verdict"] in r.stdout
    for name in gc.PLAYERS:
        g1 = json.loads((tmp_path / f"g1-{name}.json").read_text())
        assert g1["ok"] and g1["threads"] == 2, g1["reasons"]
        assert json.loads((tmp_path / f"g4-{name}.json").read_text())["ok"]
    sel = gc.read(tmp_path / "select.pkl")
    assert len(sel["chosen"]) == 6
    table = {n: gc.read(tmp_path / f"branches-{n}.pkl") for n in gc.PLAYERS}
    assert all(len(table[n]["table"]) == 6 for n in gc.PLAYERS)
    assert all(len(row) == 16 for n in gc.PLAYERS for row in table[n]["table"])
    assert set(rec["correlations"]) == {"T0", "T2"}
