"""E3a: sequences mined from the agent's own play, and whether grouping them
by latents makes sense (CURRICULUM "Queued experiment (pre-registered) --
E3", its E3a part and amendments (A1)-(A8), Kai 2026-11-04). Pins: the
advantage's returns (the prep's clip, unit and per-base-step discount)
against a hand computation and distill's value check, the high half and
its ties, the candidate mining (counts, ties by first occurrence, the
64-frame cap, the dropped duration tokens, the cut-short runs, the
high-half filter), the multi-step calibration errors and tau_L per length,
the unrolled and real-outcome projections, the raw-ALE branch (E0's
digest, MaxAndSkip's observations), the main line's hook on a scripted
emulator (the candidate plays' eligibility and pairing, the equivalence
classes, the duplicate flags, the replay and reached-stack checks, the
emulator restored, a smoke cut, a leaky emulator caught), read (1)'s rate,
reads (2) / (3)'s pair scoring against the trivial rules (strictly below
tau_L, the components), every verdict branch and the under-100 rule, the
validity checks, the value terms and decoder refusals, the acting nets, the
output files, and -- in subprocesses -- that importing the probe changes
nothing process-wide and (slow) a smoke run on T2's real snapshot, log and
census, valid and then VOID."""
import copy
import io
import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBES = ROOT / "benchmarks" / "probes"
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(PROBES))
import ablation  # noqa: E402
import distill  # noqa: E402
import muzero  # noqa: E402
import networks  # noqa: E402
import probe_action_census as pc  # noqa: E402
import probe_e1_equivalence as e1  # noqa: E402
import probe_e3a_sequences as e3  # noqa: E402

T2_VOCAB = ((0, 4), (1, 4), (2, 4), (3, 4), (3, 16), (3, 32), (4, 4), (4, 32),
            (5, 4), (5, 16), (5, 32), (6, 4), (7, 4), (7, 32), (8, 4), (8, 8),
            (8, 16), (8, 32))


def test_importing_the_probe_changes_nothing_process_wide():
    """The script-mode settings stay out of a module import (the exact
    eval's import would set ablation's env globals and the threads)."""
    assert "probe_exhaustive_eval" not in e3.__dict__
    code = ("import os, sys; sys.path[:0] = [{p!r}, {e!r}, {r!r}]; "
            "import torch; threads = torch.get_num_threads(); "
            "import ablation; before = (ablation.ENV_NAME, ablation.ACTIONS); "
            "rng = torch.get_rng_state(); "
            "import probe_e3a_sequences; "
            "assert (ablation.ENV_NAME, ablation.ACTIONS) == before; "
            "assert torch.get_num_threads() == threads; "
            "assert torch.equal(torch.get_rng_state(), rng); "
            "assert 'probe_exhaustive_eval' not in sys.modules; "
            "assert os.environ.get('PROBE_ACTING_STATES') is None; "
            "assert os.environ.get('PROBE_ENV') is None; "
            "assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-7'; "
            "print('ok')").format(p=str(PROBES), e=str(ROOT / "experiments"),
                                  r=str(ROOT))
    env = {k: v for k, v in os.environ.items()
           if k not in ("PROBE_ACTING_STATES", "PROBE_ENV")}
    # a sentinel the script mode would overwrite with "-1" (an invalid id:
    # still no GPU, never the live run's)
    env["CUDA_VISIBLE_DEVICES"] = "-7"
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_the_pre_registered_constants():
    assert e3.RUN_SIZES == (2, 3, 4) and e3.MAX_RUN == 4          # (A3)
    assert (e3.MAX_LENGTH, e3.N_CANDIDATES) == (64, 32)
    assert e3.HORIZON == 60 and e3.BASE == 4                      # (A4)
    assert (e3.GROUP_P, e3.GROUP_R) == (0.7, 0.2)                 # (A8)
    assert (e3.DUP_BAR, e3.MIN_PLAYS) == (0.10, 100)
    assert e3.TAU_MULTS == {"0.5": 0.5, "2": 2.0}
    assert not e3.SMOKE or any(k.startswith("E3A_") for k in os.environ)


# --------------------------------------------------------------------------
# (A2) the advantage
# --------------------------------------------------------------------------
def _frame(rng):
    return rng.integers(0, 256, size=(3, 84, 84), dtype=np.uint8)


def _life(vocab, slots, rng, rewards=None, cut=None):
    """A life as play_episode / play_token record it under REWARD_GROWTH:
    records (stack, slot, the raw per-base-step reward tuple entering it);
    decision i's base steps earn rewards[i] (zeros by default); `cut` {i:
    base steps played} (a life's last token cut short by its end)."""
    obs = _frame(rng)
    stack = muzero.initial_frame_stack(_frame(rng))
    traj, prev = [], ()
    for i, slot in enumerate(slots):
        stack = muzero.push_frame(stack, obs)
        n = (cut or {}).get(i, vocab[slot][1] // 4)
        after, inner = stack, []
        for k in range(n):
            obs = _frame(rng)
            inner.append(float(rewards[i][k]) if rewards else 0.0)
            if k < n - 1:
                after = muzero.push_frame(after, obs)
        traj.append((stack, int(slot), prev))
        stack, prev = after, tuple(inner)
    traj.append((muzero.push_frame(stack, obs), 0, prev))
    return traj


def _reward_traj(slots, rewards):
    """Records carrying only what the returns read (slot, reward tuple)."""
    return ([(None, slots[0], ())]
            + [(None, slots[i + 1] if i + 1 < len(slots) else 0,
                tuple(rewards[i])) for i in range(len(slots))])


def test_the_returns_against_a_hand_computation():
    """Unit 10, ceiling 5 (sign kept), discount 0.5 per BASE STEP: the base
    steps' clipped rewards are 0, 5, 0, 1 | -3 | 5, 0, and each decision's
    return starts at its own first base step."""
    slots = [4, 0, 15]                        # (3,16), (0,4), (8,8)
    rewards = [(0, 100, 0, 10), (-30,), (2000, 0)]
    traj = _reward_traj(slots, rewards)
    g, steps = e3.decision_returns(traj, 10.0, 5.0, 0.5)
    assert steps == [4, 1, 2]
    c = [0, 5, 0, 1, -3, 5, 0]

    def ret(start):
        return sum(0.5 ** k * x for k, x in enumerate(c[start:]))
    assert g.tolist() == pytest.approx([ret(0), ret(4), ret(5)])
    assert g.tolist() == pytest.approx([2.59375, -0.5, 5.0])
    # distill's value check clips and discounts the same base steps: its
    # largest |return| is base step 1's
    assert distill.discounted_return_max([{"traj": traj}], 10.0, 5.0, 0.5) \
        == pytest.approx(ret(1))
    # the value targets' token-edge discount (a token's clipped rewards
    # summed undiscounted, its edge discounted 0.5 ** (frames / 4)) reads
    # 6 + 0.5 ** 4 * -3 + 0.5 ** 5 * 5 = 5.96875: not the registered return
    assert g[0] != pytest.approx(6 + 0.5 ** 4 * -3 + 0.5 ** 5 * 5)
    assert e3.clipped(-70, 10.0, 5.0) == -5.0 and e3.clipped(30, 10.0, 5.0) \
        == 3.0
    empty, none = e3.decision_returns(traj[:1], 10.0, 5.0, 0.5)
    assert empty.shape == (0,) and none == []
    with pytest.raises(SystemExit, match="raw per-base-step rewards"):
        e3.decision_returns([(None, 4, 0.0), (None, 0, 1.0)], 10.0, 5.0, 0.5)


def test_the_high_half_and_its_ties():
    high, low = e3.high_half([0.5, 2.0, 2.0, -1.0, 0.5, 3.0, 0.5])
    assert high == {5, 1, 2} and low == 2.0             # floor(7 / 2) = 3
    high, low = e3.high_half([1.0, 1.0, 1.0, 1.0])
    assert high == {0, 1} and low == 1.0                # ties: the earlier
    high, low = e3.high_half([0.0, -2.0, 7.0, 7.0, 1.0, 9.0])
    assert high == {5, 2, 3}
    assert e3.high_half([4.0]) == (set(), None)
    assert e3.high_half([]) == (set(), None)


# --------------------------------------------------------------------------
# (A3) the candidates
# --------------------------------------------------------------------------
def test_the_logged_runs():
    runs = list(e3.logged_runs([9, 4, 0], T2_VOCAB))
    assert runs == [(0, 2, ((5, 16), (3, 16)), 32),
                    (0, 3, ((5, 16), (3, 16), (0, 4)), 36),
                    (1, 2, ((3, 16), (0, 4)), 20)]
    assert list(e3.logged_runs([9], T2_VOCAB)) == []


def test_the_duration_tokens():
    vs = set(T2_VOCAB)
    assert e3.is_duration_token(((8, 4), (8, 4)), vs)          # (8, 8)
    assert e3.is_duration_token(((3, 16), (3, 16)), vs)        # (3, 32)
    assert e3.is_duration_token(((8, 8), (8, 8)), vs)          # (8, 16)
    assert not e3.is_duration_token(((6, 4), (6, 4)), vs)      # no (6, 8)
    assert not e3.is_duration_token(((8, 16),) * 3, vs)        # no (8, 48)
    assert not e3.is_duration_token(((3, 16), (5, 16)), vs)    # two actions


def _mining_lives():
    """Life A (log 0-6): (5,16) (5,16) (3,16) (3,16) (5,32) (6,4) (6,4);
    life B (log 7-11): (5,16) (3,16) (3,16) (5,16) (3,16), its last cut
    short. High: A's 0, 1, 3, 5 and B's 0, 1, 3."""
    return ([{"slots": [9, 9, 4, 4, 10, 11, 11], "full": [True] * 7,
              "offset": 0},
             {"slots": [9, 4, 4, 9, 4], "full": [True] * 4 + [False],
              "offset": 7}],
            {0, 1, 3, 5, 7, 8, 10})


def test_mining_counts_ties_cap_drops_and_the_high_half():
    lives, high = _mining_lives()
    cands, dropped, stats = e3.mine_candidates(lives, T2_VOCAB, high, top=10)
    keys = [c["key"] for c in cands]
    s16, t16 = (5, 16), (3, 16)
    # the two keys seen twice first, then the rest in first-occurrence order
    assert keys == [(s16, t16), (s16, t16, t16),
                    (s16, s16, t16), (s16, s16, t16, t16),
                    (t16, (5, 32)), (t16, (5, 32), (6, 4)),
                    (t16, (5, 32), (6, 4), (6, 4)), ((6, 4), (6, 4)),
                    (s16, t16, t16, s16), (t16, t16, s16)]
    assert [c["count"] for c in cands] == [2, 2] + [1] * 8
    assert [c["id"] for c in cands] == list(range(10))
    assert [c["length"] for c in cands] == [32, 48, 48, 64, 48, 52, 56, 8,
                                            64, 48]
    assert [c["n"] for c in cands] == [2, 3, 3, 4, 2, 3, 4, 2, 4, 3]
    # (5,16)x2 and (3,16)x2 are duration tokens (5,32) / (3,32); (6,4)x2 is
    # not (no (6, 8)); a 64-frame run is in, an 80-frame one out
    assert dropped == {(s16, s16): 1, (t16, t16): 1}
    assert stats == {"high_runs": 17, "cut_short": 2, "over_length": 1,
                     "duration_tokens": 2, "counted": 12, "keys": 10}
    # the low starts (A 2 and 4, B 2) count nothing
    assert (t16, t16, (5, 32)) not in keys and ((5, 32), (6, 4)) not in keys
    assert (t16, s16) not in keys                       # B's 2
    # the top 4: the tie at count 1 is cut in first-occurrence order
    top4, _, _ = e3.mine_candidates(lives, T2_VOCAB, high, top=4)
    assert [c["key"] for c in top4] == keys[:4]
    # every decision high: the low starts count too
    every, _, s = e3.mine_candidates(lives, T2_VOCAB, set(range(12)), top=99)
    assert (t16, t16, (5, 32)) in [c["key"] for c in every]
    assert s["high_runs"] == sum(1 for life in lives for _ in
                                 e3.logged_runs(life["slots"], T2_VOCAB))
    assert e3.mine_candidates(lives, T2_VOCAB, set())[0] == []


def test_by_length():
    cands = [{"id": 0, "length": 32}, {"id": 1, "length": 48},
             {"id": 2, "length": 32}]
    assert e3.by_length(cands) == {32: [0, 2], 48: [1]}


# --------------------------------------------------------------------------
# (A6) the forward model on tiny nets, its calibration
# --------------------------------------------------------------------------
def _tiny_nets(actions=18, seed=0):
    torch.manual_seed(seed)
    rep = networks.RepresentationNetwork(input_channels=12, latent_channels=16,
                                         trunk_blocks=1, renormalize=False)
    dyn = networks.DynamicsNetwork(latent_channels=16, action_space=actions,
                                   blocks=1, action_encoding="onehot",
                                   reward_channels=4, renormalize=False)
    proj = networks.LatentProjector(16 * 36, proj_hidden=32, proj_out=32,
                                    pred_hidden=16)
    for net in (rep, dyn, proj):
        for m in net.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.running_mean.uniform_(-0.2, 0.2)
                m.running_var.uniform_(0.5, 1.5)
        net.eval()
    return e1.Nets(rep, dyn, proj)


def _tiny_pred(actions=18, seed=1, support=10):
    torch.manual_seed(seed)
    pred = networks.PredictionNetwork(latent_channels=16, action_space=actions,
                                      head_channels=4, support_size=support)
    torch.nn.init.normal_(pred.policy_fc.weight, std=0.5)
    torch.nn.init.normal_(pred.value_fc.weight, std=0.5)
    return pred.eval()


def _x(stack):
    return torch.tensor(np.asarray(stack)[None], dtype=torch.float32) / 255.0


def _hand_unroll(nets, stack, slots):
    with torch.no_grad():
        z = nets.rep(_x(stack))
        for s in slots:
            z = nets.dyn(z, torch.tensor([int(s)]))[0]
        return nets.proj(z, with_prediction=True)[0]


def _hand_target(nets, stack):
    with torch.no_grad():
        return nets.proj(nets.rep(_x(stack)), with_prediction=False)[0]


def _cosd(a, b):
    return float(1.0 - torch.nn.functional.cosine_similarity(a[None], b[None]))


def test_the_run_errors_against_a_hand_computation():
    nets = _tiny_nets()
    rng = np.random.default_rng(1)
    slots = [0, 4, 15, 11, 17, 2]
    traj = _life(T2_VOCAB, slots, rng)
    hand = {(i, n): _cosd(_hand_unroll(nets, traj[i][0], slots[i:i + n]),
                          _hand_target(nets, traj[i + n][0]))
            for n in (2, 3, 4) for i in range(len(slots) - n + 1)}
    assert len(hand) == 5 + 4 + 3
    for batch in (1, 4, 64):
        lat = e3.life_latents(nets.rep, traj, batch)
        got = e3.life_run_errors(nets, lat, slots, batch=batch)
        assert set(got) == set(hand)
        for k, v in hand.items():
            assert got[k] == pytest.approx(v, abs=1e-5), k
    assert len({round(v, 6) for v in hand.values()}) == len(hand)
    # the slots matter: other tokens, other errors
    other = e3.life_run_errors(nets, e3.life_latents(nets.rep, traj),
                               [(s + 1) % 18 for s in slots])
    assert not all(other[k] == pytest.approx(v, abs=1e-6)
                   for k, v in hand.items())
    one = e3.life_run_errors(nets, e3.life_latents(nets.rep, traj[:2]),
                             slots[:1])
    assert one == {}


def test_the_length_calibration():
    """tau_L is the median per total length, the run sizes pooled (a mean
    would read 0.2667 for L 32)."""
    cal = e3.calibrate_lengths([32, 32, 48, 32, 48], [2, 3, 2, 2, 4],
                               [0.1, 0.5, 0.2, 0.2, 0.4])
    assert set(cal) == {32, 48}
    assert cal[32]["tau"] == pytest.approx(0.2) and cal[32]["decisions"] == 3
    assert cal[48]["tau"] == pytest.approx(0.3)
    assert set(cal[32]["by_size"]) == {"2", "3"}
    assert cal[32]["by_size"]["2"]["p50"] == pytest.approx(0.15)
    assert cal[32]["by_size"]["3"]["decisions"] == 1
    assert set(cal[48]["by_size"]) == {"2", "4"}
    assert e3.calibrate_lengths([], [], []) == {}


def test_the_own_play_pass():
    """One pass over two lives: offsets, returns, values (acting nets),
    advantages, the calibration's runs (full, at most 64 frames), counts."""
    nets, pred = _tiny_nets(), _tiny_pred()
    rng = np.random.default_rng(3)
    s0 = [9, 5, 5, 4, 0]                     # 16, 32, 32, 16, 4 frames
    r0 = [(0, 10, 0, 0), (0,) * 7 + (50,), (0,) * 8, (200, 0, 0, 0), (10,)]
    a = _life(T2_VOCAB, s0, rng, r0)
    s1 = [14, 9, 15]                         # 4, 16, 8: the last cut short
    b = _life(T2_VOCAB, s1, rng, [(10,), (0, 0, 0, 0), (0,)], cut={2: 1})
    lives_iter = [(1, {"traj": a, "terminated": True}),
                  (2, {"traj": b, "terminated": False})]
    lives, adv, rets, vals, (lengths, sizes, errors), counts = \
        e3.own_play_pass(lives_iter, T2_VOCAB, nets, pred, 10.0, 17.4, 0.9)
    assert [l["offset"] for l in lives] == [0, 5]
    assert [l["iteration"] for l in lives] == [1, 2]
    assert lives[0]["full"] == [True] * 5
    assert lives[1]["full"] == [True, True, False]
    want = np.concatenate([e3.decision_returns(t, 10.0, 17.4, 0.9)[0]
                           for t in (a, b)])
    assert rets == pytest.approx(want)
    with torch.no_grad():
        hand_v = [float(pred(nets.rep(_x(t[i][0])))[1])
                  for t in (a, b) for i in range(len(t) - 1)]
    # decoded values pass through h's inverse: batched against batch-1
    # float32 differences are relative
    assert vals == pytest.approx(hand_v, rel=1e-4, abs=1e-4)
    assert adv == pytest.approx(want - np.asarray(hand_v), rel=1e-4,
                                abs=1e-4)
    assert len(set(np.round(vals, 3))) > 1           # the stacks matter
    # life a: lengths 48, 64, 48, 20 (n 2); 80, 80, 52 (n 3); 96, 84 (n 4):
    # five runs at most 64 frames. Life b: 20 (n 2, full); the rest touch
    # its cut decision
    assert sorted(zip(lengths, sizes)) == [(20, 2), (20, 2), (48, 2),
                                           (48, 2), (52, 3), (64, 2)]
    errs_a = e3.life_run_errors(nets, e3.life_latents(nets.rep, a), s0)
    assert errors[0] == pytest.approx(errs_a[(0, 2)], abs=1e-6)
    assert counts == {"lives": 2, "truncated_lives": 1, "decisions": 8,
                      "cut_short_decisions": 1}


def test_the_unrolled_and_real_projections():
    nets = _tiny_nets()
    rng = np.random.default_rng(2)
    stack = muzero.initial_frame_stack(_frame(rng))
    seqs = [[1, 4], [5, 0, 1], [0, 4], [17, 17, 17, 2]]
    z = e3.unrolled_projections(nets, stack, seqs)
    assert z.shape == (4, 32)
    for row, q in zip(z, seqs):
        assert torch.allclose(row, _hand_unroll(nets, stack, q), atol=1e-5)
    stacks = [muzero.initial_frame_stack(_frame(rng)) for _ in range(3)]
    r = e3.stack_projections(nets, stacks)
    for row, s in zip(r, stacks):
        assert torch.allclose(row, _hand_target(nets, s), atol=1e-5)
    d = e3.pair_distances(z)
    full = e1.cos_distances(z)
    assert d == pytest.approx([full[a, b] for a, b in e1.all_pairs(4)])
    assert len(d) == 6 and all(isinstance(x, float) for x in d)


# --------------------------------------------------------------------------
# (A4) the truth: a scripted emulator with screens
# --------------------------------------------------------------------------
class _State:
    def __init__(self, t):
        self.t = t

    def serialize(self):
        return repr(self.t).encode()


class _ImgALE:
    """A dot on a strip: RIGHT (2) and the diagonals 5-8 move it right, LEFT
    (3) left, NOOP / UP / DOWN (0, 1, 4) leave it; a corner flickers on odd
    frames (so MaxAndSkip's max matters)."""
    EFFECT = {0: 0, 1: 0, 2: 1, 3: -1, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1}

    def __init__(self):
        self.pos, self.frame, self.acts = 20, 0, 0

    def act(self, a):
        self.pos += self.EFFECT[int(a)]
        self.frame += 1
        self.acts += 1

    def getScreenRGB(self):
        img = np.zeros((210, 160, 3), dtype=np.uint8)
        x = (self.pos * 3) % 150
        img[100:112, x:x + 6, :] = 200
        if self.frame % 2:
            img[0:12, 0:12, 0] = 255
        return img

    def cloneState(self):
        return _State((self.pos, self.frame))

    def restoreState(self, s):
        self.pos, self.frame = s.t


class _LeakyImgALE(_ImgALE):
    """The screen also shows how many frames were ever emulated: an
    order dependence restoreState does not undo."""

    def getScreenRGB(self):
        img = super().getScreenRGB()
        img[50:62, 0:12, 1] = self.acts % 256
        return img


def test_the_branch_is_e0s_with_maxandskip_observations():
    ale = _ImgALE()
    start = ale.cloneState()
    inputs = [2] * 4 + [3] * 4 + [5] * 8
    dg, obs = e3.play_branch(ale, start, inputs, 16 + 9, 0)
    assert dg == pc.play_window(ale, start, inputs, 16 + 9, 0)[0]
    # by hand: each base step's last two raw frames, maxed
    ale.restoreState(start)
    screens = []
    for a in inputs:
        ale.act(a)
        screens.append(ale.getScreenRGB())
    want = [np.maximum(screens[4 * b + 2], screens[4 * b + 3])
            for b in range(4)]
    assert len(obs) == 4
    assert all(np.array_equal(o, w) for o, w in zip(obs, want))
    assert all(o[0:12, 0:12, 0].max() == 255 for o in obs)   # the flicker
    # another input stream, another digest; the same one again, the same
    assert e3.play_branch(ale, start, [3] * 16, 25, 0)[0] != dg
    assert e3.play_branch(ale, start, inputs, 25, 0)[0] == dg
    with pytest.raises(ValueError):
        e3.play_branch(ale, start, [2] * 6, 25, 0)
    with pytest.raises(ValueError):
        e3.play_branch(ale, start, [2] * 8, 7, 0)
    assert e3.token_inputs(((2, 4), (3, 8)), list(range(10, 19))) == \
        [12] * 4 + [13] * 8
    # the reached stack pushes each observation, preprocessed
    stack = muzero.initial_frame_stack(muzero.preprocess_frame(
        screens[0], muzero.IMAGE_RESHAPE))
    s = stack
    for o in want:
        s = muzero.push_frame(s, muzero.preprocess_frame(o,
                                                         muzero.IMAGE_RESHAPE))
    assert np.array_equal(e3.reached_stack(stack, obs), s)
    assert e3.reached_stack(stack, []) is stack


# the scripted main line: slots of a small vocabulary, candidates over it
V = ((0, 4), (2, 4), (3, 4), (5, 4), (2, 8), (1, 4))
CAND_KEYS = (((2, 4), (2, 4)),            # c0  L8
             ((5, 4), (2, 4)),            # c1  L8  = c0 (5 moves as 2)
             ((3, 4), (2, 4)),            # c2  L8
             ((2, 4), (2, 8)),            # c3  L12
             ((0, 4), (2, 8)),            # c4  L12
             ((1, 4), (0, 4), (2, 4)),    # c5  L12
             ((0, 4), (1, 4), (2, 4)),    # c6  L12 = c5 (1 stays as 0)
             ((3, 4), (2, 8)))            # c7  L12
# d0..d9: (2,4) (2,4) (2,8) (1,4) (0,4) (2,4) (2,4)| (2,4) (3,4) (2,8)
TOKENS = (1, 1, 4, 5, 0, 1, 1, 1, 2, 4)
# d6 loses a life at its last base step; d9 ends the game after one of its
# two base steps (cut short)
ENDS = {6: (0, False), 9: (0, True)}


def _cands(keys=CAND_KEYS):
    return [{"id": j, "key": k, "length": sum(f for _, f in k), "n": len(k),
             "count": 1} for j, k in enumerate(keys)]


class _Game:
    """The env play_token sees: the scripted ALE, ablation.play_token's
    loop over a MaxAndSkip stack (4 raw frames per base step, the max of
    the last two), EpisodicLife's was_real_done; `ends` {decision: (the
    base step it ends after, a real game over)}."""

    def __init__(self, ale, ends=ENDS):
        self.unwrapped = SimpleNamespace(ale=ale, _action_set=list(range(9)))
        self.ends, self.count, self.was_real_done = ends, 0, False

    def base_step(self, a):
        screens = []
        for k in range(4):
            self.unwrapped.ale.act(a)
            if k >= 2:
                screens.append(self.unwrapped.ale.getScreenRGB())
        return np.maximum(screens[0], screens[1])

    def play_token(self, env, token, stack, inner=None):
        a, f = V[int(token)]
        end = self.ends.get(self.count)
        self.count += 1
        for i in range(f // 4):
            obs = self.base_step(a)
            term = end is not None and i == end[0]
            self.was_real_done = bool(term and end[1])
            if term:
                return obs, 0.0, True, False, stack, i + 1
            if i < f // 4 - 1:
                stack = muzero.push_frame(stack, muzero.preprocess_frame(
                    obs, muzero.IMAGE_RESHAPE))
        return obs, 0.0, False, False, stack, f // 4


def _drive(game, tokens, priors):
    """play_episode's loop over every life of one game: each decision's
    stack (its prior's argmax registered by the stack's id), the main
    line's observations; a lost life restarts the stack after one NOOP
    base step (EpisodicLife's reset)."""
    def pre(o):
        return muzero.preprocess_frame(o, muzero.IMAGE_RESHAPE)
    obs = game.unwrapped.ale.getScreenRGB()
    after = muzero.initial_frame_stack(pre(obs))
    stacks, seen = [], []
    for t in tokens:
        stack = muzero.push_frame(after, pre(obs))
        priors[id(stack)] = t
        stacks.append(stack)
        out = ablation.play_token(game, t, stack)
        obs, after = out[0], out[4]
        seen.append(obs.tobytes())
        if out[2] and not game.was_real_done:
            obs = game.base_step(0)
            after = muzero.initial_frame_stack(pre(obs))
    return stacks, seen


def _prior(token, n=len(V)):
    p = np.full(n, 0.01)
    p[token] = 0.9
    return p


def _truth(dead=(3,), drop=(9,), wrong=None):
    """The census's lines for game 1: live (36 classes) but `dead` (one
    class), `drop` missing, `wrong` {d: another token}."""
    wrong = wrong or {}
    return {(1, d): (wrong.get(d, t), [0] * 36 if d in dead
                     else list(range(36)))
            for d, t in enumerate(TOKENS) if d not in drop}


def _probe(monkeypatch, ale_cls=_ImgALE, max_decisions=0, truth=None):
    """The scripted game under the Prober; returns (prober, stacks, the
    main line's observations, the final emulator state, out)."""
    monkeypatch.setattr(e3, "MAX_DECISIONS", max_decisions)
    game = _Game(ale_cls())
    monkeypatch.setattr(ablation, "play_token", game.play_token)
    priors = {}
    nm = SimpleNamespace(initial_inference=lambda s: (_prior(priors[id(s)]),
                                                      0.0))
    run = SimpleNamespace(searcher=SimpleNamespace(network_manager=nm))
    out = io.StringIO()
    prober = e3.Prober(run, _tiny_nets(actions=len(V)), V,
                       _truth() if truth is None else truth, _cands(),
                       {8: 0.05, 12: 0.05}, out)
    prober.game = 1
    with prober.hooked():
        stacks, seen = _drive(game, TOKENS, priors)
    prober.flush(game)
    assert ablation.play_token == game.play_token       # the hook let go
    return prober, stacks, seen, game.unwrapped.ale.cloneState().t, out


def _plain(ale_cls=_ImgALE):
    game = _Game(ale_cls())
    orig = ablation.play_token
    ablation.play_token = game.play_token
    try:
        _, seen = _drive(game, TOKENS, {})
    finally:
        ablation.play_token = orig
    return seen, game.unwrapped.ale.cloneState().t


def test_the_main_line_hook_on_a_scripted_game(monkeypatch):
    prober, stacks, seen, end, out = _probe(monkeypatch)
    # the branches never touched the main line: the same game, the same end
    plain_seen, plain_end = _plain()
    assert seen == plain_seen and end == plain_end
    recs = prober.recs
    assert [r["d"] for r in recs] == list(range(10))
    assert [json.loads(l)["d"] for l in out.getvalue().splitlines()] == \
        list(range(10))
    assert prober.scored == 10 and prober.missing == 1
    assert recs[9].get("missing") is True and "live" not in recs[9]
    assert prober.token_mismatches == prober.argmax_mismatches == 0
    assert prober.replay_mismatches == prober.stack_mismatches == 0
    groups = {r["d"]: r["groups"] for r in recs if r.get("groups")}
    # d3's (1,4)(0,4)(2,4) is c5 but d3 is dead: no group
    assert set(groups) == {0, 1, 5}
    g0, g1, g5 = groups[0][0], groups[1][0], groups[5][0]
    # d0 plays c0 (L8): c1 spells it too, c2 does not
    assert (g0["length"], g0["n"], g0["candidate"]) == (8, 2, 0)
    assert g0["members"] == [0, 1, 2] and g0["played"] == 0
    assert g0["truth"] == [0, 0, 1] and g0["duplicate"] is True
    # d1 plays c3 (L12): five members, c5 = c6, none equals c3
    assert (g1["length"], g1["candidate"], g1["played"]) == (12, 3, 0)
    assert g1["members"] == [3, 4, 5, 6, 7]
    assert g1["truth"] == [0, 1, 2, 2, 3] and g1["duplicate"] is False
    # d5 plays c0 into d6's lost life (a life-loss at the run's last base
    # step is in)
    assert (g5["length"], g5["candidate"], g5["duplicate"]) == (8, 0, True)
    for g in (g0, g1, g5):
        assert g["replay_ok"] is True and g["stack_ok"] is True
        assert g["tau"] == 0.05
        k = len(g["members"])
        assert len(g["dist_pred"]) == len(g["dist_real"]) == k * (k - 1) // 2
    # identical frames reach identical stacks: c0 and c1 at distance 0
    assert g0["dist_real"][0] == pytest.approx(0.0, abs=1e-6)
    seqs = [[1, 1], [3, 1], [2, 1]]
    want = e3.pair_distances(e3.unrolled_projections(prober.nets, stacks[0],
                                                     seqs))
    assert g0["dist_pred"] == pytest.approx(want, abs=1e-6)
    # T2's runs: eligibility at the life's and the game's ends
    runs = {r["d"]: r["runs"] for r in recs if "runs" in r}
    assert runs[5] == [{"n": 2, "length": 8, "eligible": True,
                        "candidate": 0}]
    assert runs[6] == []                       # the life's last decision
    assert runs[7] == [{"n": 2, "length": 8, "eligible": True,
                        "candidate": None},
                       {"n": 3, "length": 16, "eligible": False,
                        "candidate": None}]
    # d8 -> d9: c7's key, but d9 was cut short by the game's end
    assert runs[8] == [{"n": 2, "length": 12, "eligible": False,
                        "candidate": 7}]
    assert runs[4] == [{"n": 2, "length": 8, "eligible": True,
                        "candidate": None},
                       {"n": 3, "length": 12, "eligible": True,
                        "candidate": None}]
    assert [x["candidate"] for x in runs[3]] == [None, 5, None]
    assert recs[3]["live"] is False and recs[3]["groups"] == []
    # the reads on these records
    s = e3.summarize(recs)
    r1 = s["read_1"]
    assert (r1["plays"], r1["duplicates"]) == (3, 2)
    assert r1["rate"] == round(2 / 3, 4)
    assert (r1["pairs"], r1["equivalent_pairs"]) == (16, 3)
    assert r1["classes_mean"] == round(8 / 3, 4)
    assert r1["candidates_mean"] == round(11 / 3, 4)
    assert r1["by_length"] == {
        "8": {"plays": 2, "duplicates": 2, "members": 3, "rate": 1.0},
        "12": {"plays": 1, "duplicates": 0, "members": 5, "rate": 0.0}}
    assert s["checks"] == {"replay_ok": 3, "stack_ok": 3, "plays": 3}
    assert s["live_decisions"] == 8 and s["decisions"] == 10
    assert s["runs_on_live"]["2"]["candidate_plays"] == 3
    assert s["verdict"] == "KAI_CALL_TOO_FEW"              # 3 < 100 plays


def test_a_smoke_cut_scores_fewer_decisions_not_a_shorter_game(monkeypatch):
    prober, _, seen, end, _ = _probe(monkeypatch, max_decisions=2,
                                     truth=_truth(wrong={1: 3}))
    assert (seen, end) == _plain()
    assert [r["d"] for r in prober.recs] == [0, 1] and prober.scored == 2
    assert [len(r["groups"]) for r in prober.recs] == [1, 1]
    assert prober.token_mismatches == 1 and prober.missing == 0


def test_a_leaky_emulator_is_caught(monkeypatch):
    """Frames that depend on the branches played: the played candidate's
    replay differs from its first digest, and its reached stack is not the
    main line's."""
    prober, _, _, _, _ = _probe(monkeypatch, ale_cls=_LeakyImgALE)
    assert prober.replay_mismatches == 3 and prober.stack_mismatches == 3
    counts = {k: getattr(prober, k) for k in (
        "token_mismatches", "argmax_mismatches", "missing",
        "replay_mismatches", "stack_mismatches")}
    monkeypatch.setattr(e3, "MAX_DECISIONS", 0)
    ok, why = e3.validity({1: (0.0, 25, True, 10)}, None, {1: 10}, {1: 10},
                          counts, True)
    assert ok is False
    assert "3 played candidates whose replay differs" in " ".join(why)
    assert "3 played candidates whose reached stack" in " ".join(why)


# --------------------------------------------------------------------------
# the reads on hand-made candidate plays
# --------------------------------------------------------------------------
def _g(truth, played, dist_pred, dist_real=None, tau=0.1, length=8):
    k = len(truth)
    return {"length": length, "n": 2, "candidate": 0,
            "members": list(range(k)), "played": played, "truth": truth,
            "duplicate": any(truth[j] == truth[played] for j in range(k)
                             if j != played),
            "tau": tau, "dist_pred": dist_pred,
            "dist_real": dist_pred if dist_real is None else dist_real,
            "replay_ok": True, "stack_ok": True}


def test_read_1_the_duplicate_spelling_rate():
    groups = [_g([0, 0, 1], 0, [0.0] * 3),             # duplicate
              _g([0, 1, 1], 0, [0.0] * 3),             # c1 = c2, not c0
              _g([0], 0, [], length=12),               # alone: no duplicate
              _g([0, 1, 2, 2], 3, [0.0] * 6, length=16)]
    r = e3.read_1(groups)
    assert (r["plays"], r["duplicates"]) == (4, 2) and r["rate"] == 0.5
    assert r["_rate"] == 0.5
    assert r["plays_with_another_candidate"] == 3
    assert r["rate_with_another_candidate"] == round(2 / 3, 4)
    assert (r["pairs"], r["equivalent_pairs"]) == (12, 3)
    assert r["equivalent_pair_share"] == 0.25
    assert r["classes_mean"] == round((2 + 2 + 1 + 3) / 4, 4)
    assert r["candidates_mean"] == 2.75
    assert list(r["by_length"]) == ["8", "12", "16"]
    assert r["by_length"]["8"] == {"plays": 2, "duplicates": 1,
                                   "members": 3, "rate": 0.5}
    assert e3.read_1([])["rate"] is None and e3.read_1([])["_rate"] is None


def test_reads_2_and_3_against_the_trivial_rules():
    """g1 groups its one true pair; g2 merges a wrong pair and misses its
    true one: P 0.5, R 0.5. All one class: P 2/9, R 1; all distinct: no
    precision, R 0."""
    g1 = _g([0, 0, 1], 0, [0.05, 0.5, 0.5])
    g2 = _g([0, 1, 2, 2], 2, [0.05, 0.5, 0.5, 0.5, 0.5, 0.5],
            dist_real=[0.5, 0.5, 0.5, 0.5, 0.5, 0.01])
    r = e3.latent_read([g1, g2], "dist_pred")
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1) and r["groups"] == 2
    assert (r["precision"], r["recall"]) == (0.5, 0.5)
    assert r["trivial"] == {
        "one": {"precision": round(2 / 9, 4), "recall": 1.0,
                "f1": round(4 / 11, 4)},
        "distinct": {"precision": None, "recall": 0.0, "f1": 0.0}}
    assert r["above_trivial"] is True and r["meets_bars"] is False
    # (3) on the real-outcome distances: both true pairs, no false one
    r3 = e3.latent_read([g1, g2], "dist_real")
    assert (r3["tp"], r3["fp"], r3["fn"]) == (2, 0, 0)
    assert r3["meets_bars"] is True and r3["above_trivial"] is True
    md = r3["median_distance"]
    assert md["equivalent"] == pytest.approx(0.03)            # 0.05, 0.01
    assert md["non_equivalent"] == 0.5
    assert md["pairs"] == {"equivalent": 2, "non_equivalent": 7}


def test_grouping_is_strictly_below_tau_and_closes_components():
    # a distance AT tau is not grouped
    at = _g([0, 0], 0, [0.1])
    c, _ = e3.latent_counts([at], "dist_pred")
    assert tuple(c) == (0, 0, 1, 0)
    c, _ = e3.latent_counts([at], "dist_pred", mult=2.0)
    assert tuple(c) == (1, 0, 0, 0)
    # (0,1) and (1,2) grouped, (0,2) not: the components close it
    chain = _g([0, 0, 0], 0, [0.05, 0.5, 0.05])
    assert tuple(e3.latent_counts([chain], "dist_pred")[0]) == (3, 0, 0, 0)
    assert tuple(e3.latent_counts([chain], "dist_pred", closure=False)[0]) \
        == (2, 0, 1, 0)
    r = e3.latent_read([chain, at], "dist_pred")
    assert (r["pairwise"]["tp"], r["pairwise"]["fn"]) == (2, 2)
    # tau x 2 = 0.2: the chain closes (0.5 stays out) and `at` joins
    assert r["robustness"]["2"]["tp"] == 4
    # tau x 0.5 = 0.05: 0.05 is not below it, nothing is grouped
    assert r["robustness"]["0.5"]["tp"] == 0
    assert r["robustness"]["0.5"]["fn"] == 4
    # a lone candidate has no pair
    assert tuple(e3.latent_counts([_g([0], 0, [])], "dist_pred")[0]) == \
        (0, 0, 0, 0)
    with pytest.raises(ValueError):
        e3.latent_counts([_g([0, 1, 2], 0, [0.1])], "dist_pred")


@pytest.mark.parametrize("args,want", [
    # (plays, rate, precision, recall, trivial precisions)
    ((100, 0.5, 0.7, 0.2, [0.69, None]), "GROUP_BY_PREDICTED"),  # inclusive
    ((99, 0.0, 0.95, 0.9, [0.1, None]), "KAI_CALL_TOO_FEW"),     # < 100
    ((0, None, None, None, [None, None]), "KAI_CALL_TOO_FEW"),
    ((100, 0.0999, 0.6999, 0.9, [0.1, None]), "MINE_WITHOUT_GROUPING"),
    ((100, 0.1, 0.6999, 0.9, [0.1, None]), "KAI_CALL"),     # 10%: not under
    ((100, 0.05, 0.9, 0.1999, [0.1, None]), "MINE_WITHOUT_GROUPING"),
    ((100, 0.5, 0.9, 0.1999, [0.1, None]), "KAI_CALL"),
    # not strictly above a trivial rule's precision
    ((100, 0.5, 0.75, 0.5, [0.75, None]), "KAI_CALL"),
    ((100, 0.05, 0.75, 0.5, [0.8, None]), "MINE_WITHOUT_GROUPING"),
    # no predicted merge: no precision, fails
    ((500, 0.5, None, 0.0, [0.3, None]), "KAI_CALL"),
    ((500, 0.5, 0.8, None, [0.3, None]), "KAI_CALL"),
    # all distinct alone sets no bar
    ((100, 0.5, 0.7, 0.2, [None]), "GROUP_BY_PREDICTED")])
def test_the_verdict_branches(args, want):
    assert e3.verdict(*args) == want


def test_the_summary_reads_the_live_plays_and_survives_the_file():
    g_live = _g([0, 0, 1], 0, [0.05, 0.5, 0.5])
    g_dead = _g([0, 1], 0, [0.0])
    recs = [{"game": 1, "d": 0, "token": 3, "live": True,
             "runs": [{"n": 2, "length": 8, "eligible": True,
                       "candidate": 0}], "groups": [g_live]},
            {"game": 1, "d": 1, "token": 3, "live": False,
             "runs": [{"n": 2, "length": 8, "eligible": True,
                       "candidate": 0}], "groups": [g_dead]},
            {"game": 1, "d": 2, "token": 3, "missing": True}]
    s = e3.summarize(recs)
    assert s["read_1"]["plays"] == 1 and s["read_2"]["tp"] == 1
    assert s["read_2"]["fp"] == 0 and s["live_decisions"] == 1
    assert s["runs_on_live"]["2"] == {"eligible": 1, "candidate_plays": 1}
    assert s["read_3_meets_bars"] is True
    assert s["verdict"] == "KAI_CALL_TOO_FEW"
    back = [json.loads(json.dumps(r)) for r in recs]
    assert e3.summarize(back) == s
    text = json.dumps(s)
    assert '"_p"' not in text and '"_rate"' not in text
    assert '"_trivial_p"' not in text
    # 100 such plays read GROUP_BY_PREDICTED (P 1, R 1 above all one's 1/3)
    many = [dict(recs[0], d=d) for d in range(100)]
    assert e3.summarize(many)["verdict"] == "GROUP_BY_PREDICTED"
    # ... and with nothing merged, a duplicate on every play: Kai's call
    none = [dict(recs[0], d=d, groups=[_g([0, 0, 1], 0, [0.5] * 3)])
            for d in range(100)]
    assert e3.summarize(none)["verdict"] == "KAI_CALL"
    # ... and with no duplicate anywhere: mine without grouping
    clean = [dict(recs[0], d=d, groups=[_g([0, 1, 2], 0, [0.5] * 3)])
             for d in range(100)]
    assert e3.summarize(clean)["verdict"] == "MINE_WITHOUT_GROUPING"
    # (A11) single-candidate plays never dilute the decision rate: 100 plays
    # with alternatives, 30 of them duplicates (0.3), plus 300 lone plays
    # (all-plays rate 30 / 400 = 0.075 < 0.1) -- still Kai's call
    mixed = ([dict(recs[0], d=d, groups=[_g([0, 0, 1], 0, [0.5] * 3)])
              for d in range(30)]
             + [dict(recs[0], d=d, groups=[_g([0, 1, 2], 0, [0.5] * 3)])
                for d in range(30, 100)]
             + [dict(recs[0], d=d, groups=[_g([0], 0, [])])
                for d in range(100, 400)])
    sm = e3.summarize(mixed)
    assert sm["read_1"]["rate"] == 0.075
    assert sm["read_1"]["rate_with_another_candidate"] == 0.3
    assert sm["verdict"] == "KAI_CALL"


# --------------------------------------------------------------------------
# validity, the value terms, the nets, the files
# --------------------------------------------------------------------------
COUNTS0 = {"token_mismatches": 0, "argmax_mismatches": 0, "missing": 0,
           "replay_mismatches": 0, "stack_mismatches": 0}


def test_the_validity_checks(monkeypatch):
    monkeypatch.setattr(e3, "MAX_DECISIONS", 0)
    ref = {"games": {"1": [100.0, 700, True, 150], "2": [80.0, 600, True,
                                                         120]}}
    games = {1: (100.0, 700, True, 150), 2: (80.0, 600, True, 120)}
    truth_games = {1: 150, 2: 120}
    scored = {1: 150, 2: 120}

    def v(**kw):
        a = dict(games=games, ref=ref, truth_games=truth_games,
                 scored=scored, counts=COUNTS0, census_ok=True)
        a.update(kw)
        n = a.pop("n_games", None)
        return e3.validity(**a, n_games=n)

    assert v() == (True, [])
    assert v(ref=None) == (None, [])
    ok, why = v(census_ok=None)
    assert ok is False and "main_line_ok" in why[0]
    ok, why = v(games={**games, 2: (80.0, 604, True, 120)})
    assert ok is False and why[0].startswith("game 2")
    ok, why = v(truth_games={1: 150, 2: 121})
    assert ok is False and "the census has 121" in why[0]
    ok, why = v(scored={1: 150, 2: 119})
    assert ok is False and "119 decisions scored of 120" in why[0]
    for k in COUNTS0:
        ok, why = v(counts={**COUNTS0, k: 2})
        assert ok is False and len(why) == 1 and why[0].startswith("2 "), k
    ok, why = v(games={1: games[1]}, n_games=2)
    assert ok is False and why == ["stopped after 1 of 2 games"]
    monkeypatch.setattr(e3, "MAX_DECISIONS", 5)          # a smoke: <= 5
    assert v(scored={1: 5, 2: 5}) == (True, [])
    ok, why = v(scored={1: 5, 2: 4})
    assert ok is False and "4 decisions scored of 5" in why[0]


def test_a_void_summary_keeps_the_valid_verdict():
    rec = {"verdict": "MINE_WITHOUT_GROUPING"}
    e1.finalize(rec, False, ["2 played candidates whose replay differs"])
    assert rec["verdict"].startswith("VOID (2 played candidates")
    assert rec["verdict_if_valid"] == "MINE_WITHOUT_GROUPING"


T2_SPEC = {"REWARD_GROWTH": True, "REWARD_UNIT": 10.0,
           "_student_ceiling": 17.398055555555555,
           "REWARD_GROWTH_START": 17.398055555555555,
           "REWARD_GROWTH_CEILING": 17.398055555555555,
           "DISCOUNT": 0.988053892081}
T2_PREP = {"reward_unit": 10.0, "student_ceiling": 17.398055555555555,
           "student_discount": 0.988053892081}


def test_the_value_terms_and_their_refusals():
    assert e3.value_terms(T2_SPEC, T2_PREP) == (10.0, 17.398055555555555,
                                                0.988053892081)
    for change, match in (({"REWARD_GROWTH": False}, "REWARD_GROWTH"),
                          ({"REWARD_UNIT": 0.0}, "REWARD_UNIT"),
                          ({"REWARD_UNIT": True}, "REWARD_UNIT"),
                          ({"REWARD_GROWTH_START": 1.0}, "constant"),
                          ({"REWARD_GROWTH_CEILING": 160.0}, "constant"),
                          ({"DISCOUNT": 1.0}, "DISCOUNT")):
        with pytest.raises(SystemExit, match=match):
            e3.value_terms({**T2_SPEC, **change}, T2_PREP)
    with pytest.raises(SystemExit, match="its prep's"):
        e3.value_terms(T2_SPEC, {**T2_PREP, "student_discount": 0.997})
    with pytest.raises(SystemExit, match="its prep's"):
        e3.value_terms(T2_SPEC, {})


def test_the_value_decoder_check():
    pred = _tiny_pred(support=muzero.VALUE_SUPPORT)
    e3.check_value_head(pred, {})
    e3.check_value_head(pred, {"VALUE_CEILING": muzero.VALUE_CEILING})
    with pytest.raises(SystemExit, match="h-transform"):
        e3.check_value_head(pred, {"_linear_support": True})
    with pytest.raises(SystemExit, match="value support"):
        e3.check_value_head(pred, {"VALUE_CEILING": 400})
    with pytest.raises(SystemExit, match="value support"):
        e3.check_value_head(_tiny_pred(support=10), {})
    flat = _tiny_pred(support=muzero.VALUE_SUPPORT)
    flat.transform = False
    with pytest.raises(SystemExit, match="h-transform"):
        e3.check_value_head(flat, {})


def _bn_shifted(state, by):
    out = copy.deepcopy(state)
    for k, v in out.items():
        if k.endswith("running_mean"):
            v += by
        elif k.endswith("running_var"):
            v *= 1.0 + by
    return out


def _apply_acting(run, d):
    """probe_exhaustive_eval.apply_acting_states's steps (that module's
    import would set ablation's env globals in this process)."""
    for n, sd in d["acting_states"].items():
        getattr(run.trainer, n).load_state_dict(sd)
    run.set_train_mode(False)


def test_the_acting_nets():
    """(A6) The representation and prediction carry the acting states (the
    deployed ones, unlike E1's raw-statistics (A)); dynamics and projector
    are the trainer's; an acting state for them is refused."""
    nets0, pred = _tiny_nets(), _tiny_pred()
    raw = copy.deepcopy(nets0.rep.state_dict())

    def set_train_mode(training):
        for net in (nets0.rep, pred, nets0.dyn, nets0.proj):
            net.train(training)
    run = SimpleNamespace(trainer=SimpleNamespace(
        representation=nets0.rep, prediction=pred, dynamics=nets0.dyn,
        projector=nets0.proj), set_train_mode=set_train_mode)
    acting = {"representation": _bn_shifted(raw, 0.3),
              "prediction": _bn_shifted(pred.state_dict(), 0.3)}
    nets0.rep.train()
    nets, p = e3.acting_nets(run, {"acting_states": acting}, _apply_acting)
    assert nets.rep is nets0.rep and p is pred
    assert nets.dyn is nets0.dyn and nets.proj is nets0.proj
    assert not any(n.training for n in (*nets, p))
    for k, v in nets.rep.state_dict().items():
        assert torch.equal(v, acting["representation"][k])
    assert any(not torch.equal(v, raw[k])
               for k, v in nets.rep.state_dict().items())
    with pytest.raises(SystemExit, match="acting states for"):
        e3.acting_nets(run, {"acting_states": {"projector": {}}},
                       _apply_acting)


def test_a_full_e3a_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(e3, "HERE", tmp_path)
    summary, dec, meta = e3.out_files("run-x", smoke=False)
    assert summary.name == "probe_e3a_sequences-run-x.jsonl"
    assert dec.name == "probe_e3a_sequences-run-x-decisions.jsonl"
    assert meta.name == "probe_e3a_sequences-run-x-meta.jsonl"
    for p in (dec, meta):
        p.write_text("{}\n")
        with pytest.raises(SystemExit, match="never overwritten"):
            e3.out_files("run-x", smoke=False)
        p.unlink()
    s, d, m = e3.out_files("run-x", smoke=True)
    assert s.name == "probe_e3a_sequences-run-x-smoke.jsonl"
    assert d.name == "probe_e3a_sequences-run-x-smoke-decisions.jsonl"
    assert m.name == "probe_e3a_sequences-run-x-smoke-meta.jsonl"


def test_the_knob_refusals():
    e3.check_knobs(30, 0, 0)
    e3.check_knobs(1, 60, 12)
    for bad in ((0, 0, 0), (31, 0, 0), (30, -1, 0), (30, 0, -1)):
        with pytest.raises(SystemExit, match="E3A_GAMES"):
            e3.check_knobs(*bad)


# --------------------------------------------------------------------------
# end to end: a smoke on T2's real snapshot, own-play log and census
# --------------------------------------------------------------------------
CENSUS_RUN = "mspacman-two-t2-s0-2026-10-29"
REAL_SNAP = pathlib.Path(
    r"Y:\SwitchPlayer-testb2-run\benchmarks\results"
    r"\mspacman-two-t2-s0-2026-10-29\snapshots-mspacmannoframeskip-skip4-"
    r"effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-"
    r"two-t2-seed0\weights-iter042.pth")

_E2E = r"""
import json, pathlib, shutil, sys
sys.path[:0] = [{probes!r}, {exp!r}, {root!r}]
import probe_action_census as pc
import probe_e3a_sequences as e3
out = pathlib.Path({out!r})
e3.HERE = out
snap = {snap!r}
# (a) a valid smoke: game 1, its first decisions scored, a few logged lives
e3.main([snap])
tag = "probe_e3a_sequences-{run}-smoke"
shutil.copy(out / (tag + "-decisions.jsonl"), out / "a-decisions.jsonl")
shutil.copy(out / (tag + "-meta.jsonl"), out / "a-meta.jsonl")
# (b) against an exact-eval row whose game 1 differs: VOID
real = pc.exact_eval_row
def differing(path):
    row = json.loads(json.dumps(real(path)))
    row["games"]["1"][0] += 10
    return row
pc.exact_eval_row = differing
e3.MAX_DECISIONS = 5
e3.main([snap])
"""


@pytest.mark.slow
def test_a_smoke_on_t2s_real_data(tmp_path):
    summary, dec = e1.census_files(CENSUS_RUN)
    if not (summary.exists() and dec.exists()):
        pytest.skip("E0's census is not in this tree")
    if not REAL_SNAP.exists():
        pytest.skip("T2's snapshot is not on this machine")
    env = {**os.environ, "PROBE_ENV": e3.ENV_ID, "PROBE_ACTING_STATES": "1",
           "PROBE_THREADS": "2", "CUDA_VISIBLE_DEVICES": "-1",
           "E3A_GAMES": "1", "E3A_MAX_DECISIONS": "80",
           "E3A_LOG_LIVES": "8"}
    code = _E2E.format(probes=str(PROBES), exp=str(ROOT / "experiments"),
                       root=str(ROOT), out=str(tmp_path),
                       snap=str(REAL_SNAP), run=CENSUS_RUN)
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stderr[-3000:]
    runs = [json.loads(l) for l in (tmp_path / f"probe_e3a_sequences-"
                                    f"{CENSUS_RUN}-smoke.jsonl")
            .read_text().splitlines()]
    assert len(runs) == 2
    a, b = runs
    # (a) the main line is the census's and the exact eval's game 1
    assert a["smoke"] is True and a["knobs"]["log_lives"] == 8
    assert a["main_line_ok"] is True and a["void_reasons"] == []
    assert all(v == 0 for v in a["counts"].values())
    truth, per_game = e1.read_truth(dec)
    assert a["games"]["1"][3] == per_game[1]
    decs = [json.loads(l) for l in (tmp_path / "a-decisions.jsonl")
            .read_text().splitlines()]
    assert [x["d"] for x in decs] == list(range(min(80, per_game[1])))
    assert all(x["token"] == truth[(1, x["d"])][0] for x in decs)
    assert all(x["live"] == (len(set(truth[(1, x["d"])][1])) > 1)
               for x in decs)
    # the candidates (A3) and tau_L (A6)
    vocab = set(T2_VOCAB)
    cands = a["candidates"]
    assert 1 <= len(cands) <= 32
    assert [c["count"] for c in cands] == sorted(
        (c["count"] for c in cands), reverse=True)
    for c in cands:
        key = tuple(tuple(t) for t in c["key"])
        assert 2 <= c["n"] == len(key) <= 4 and set(key) <= vocab
        assert c["length"] == sum(f for _, f in key) <= 64
        assert not e3.is_duration_token(key, vocab)
        assert a["taus"].get(str(c["length"]), 0) > 0
    adv = a["advantage"]
    assert adv["high"] == adv["decisions"] // 2
    meta = json.loads((tmp_path / "a-meta.jsonl").read_text())
    assert meta["candidates"] == cands and meta["taus"] == a["taus"]
    assert meta["value_terms"] == {"reward_unit": 10.0,
                                   "ceiling": 17.398055555555555,
                                   "discount_per_base_step": 0.988053892081}
    # the candidate plays (A4): every check passed; members of one length
    groups = [g for x in decs if x.get("live") for g in x["groups"]]
    assert a["read_1"]["plays"] == len(groups) >= 1, \
        "no candidate play in game 1's first 80 decisions"
    by_id = {c["id"]: c for c in cands}
    for g in groups:
        assert g["replay_ok"] is True and g["stack_ok"] is True
        assert {by_id[m]["length"] for m in g["members"]} == {g["length"]}
        assert g["members"][g["played"]] == g["candidate"]
        k = len(g["members"])
        assert len(g["truth"]) == k and len(g["dist_pred"]) == k * (k - 1) // 2
    assert a["checks"]["replay_ok"] == a["checks"]["stack_ok"] == len(groups)
    assert a["verdict"] in ("KAI_CALL_TOO_FEW", "GROUP_BY_PREDICTED",
                            "MINE_WITHOUT_GROUPING", "KAI_CALL")
    # (b) a differing exact-eval row: VOID, the would-be verdict kept
    assert b["main_line_ok"] is False and b["verdict"].startswith("VOID")
    assert "verdict_if_valid" in b
    assert b["void_reasons"][0].startswith("game 1 ")
