"""E1, a model trained on the agent's own play against E0's outcome classes
(CURRICULUM "Next experiment (pre-registered) -- E1", Kai 2026-11-02).
Pins: the partitions' restriction and components, pair counting and its
pooled precision / recall / F1 (the 0/0 conventions), the trivial-rule
floors, (B)'s ratio class and member counts, the verdict's boundaries and
VOID, the token groups, the own-play reader against ablation.read_own_play,
the IDM's input (the 4-frame shift) on a simulated life, the reservoir and
the held-out split, (A)'s consistency errors and token projections on tiny
random nets against a hand computation, the IDM's training determinism (and
the process RNG left alone), one scored decision, the hook's next stack, the
reads on hand-made records, the census / lineage / output files, and -- in
a subprocess -- that importing the probe changes nothing process-wide."""
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
import muzero  # noqa: E402
import networks  # noqa: E402
import probe_e1_equivalence as e1  # noqa: E402

T2_VOCAB = ((0, 4), (1, 4), (2, 4), (3, 4), (3, 16), (3, 32), (4, 4), (4, 32),
            (5, 4), (5, 16), (5, 32), (6, 4), (7, 4), (7, 32), (8, 4), (8, 8),
            (8, 16), (8, 32))
T2_GROUPS = {"4": [0, 1, 2, 3, 6, 8, 11, 12, 14], "16": [4, 9, 16],
             "32": [5, 7, 10, 13, 17]}


def test_importing_the_probe_changes_nothing_process_wide():
    """The script-mode settings stay out of a module import (the exact
    eval's import would set ablation's env globals and the threads)."""
    assert "probe_exhaustive_eval" not in e1.__dict__
    code = ("import os, sys; sys.path[:0] = [{p!r}, {e!r}, {r!r}]; "
            "import torch; threads = torch.get_num_threads(); "
            "import ablation; before = (ablation.ENV_NAME, ablation.ACTIONS); "
            "rng = torch.get_rng_state(); "
            "import probe_e1_equivalence; "
            "assert (ablation.ENV_NAME, ablation.ACTIONS) == before; "
            "assert torch.get_num_threads() == threads; "
            "assert torch.equal(torch.get_rng_state(), rng); "
            "assert 'probe_exhaustive_eval' not in sys.modules; "
            "assert os.environ.get('PROBE_ACTING_STATES') is None; "
            "assert os.environ.get('PROBE_ENV') is None; "
            "assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'; "
            "print('ok')").format(p=str(PROBES), e=str(ROOT / "experiments"),
                                  r=str(ROOT))
    env = {k: v for k, v in os.environ.items()
           if k not in ("PROBE_ACTING_STATES", "PROBE_ENV")}
    env["CUDA_VISIBLE_DEVICES"] = "-1"        # never the live run's GPU
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_the_pre_registered_constants():
    assert e1.GRID == tuple((a, f) for a in range(9) for f in (4, 8, 16, 32))
    assert e1.TAU_MULTS == {"0.5": 0.5, "1": 1.0, "2": 2.0}
    assert e1.B_RATIOS == {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2}
    assert e1.B_RATIO == "0.1" and e1.PRIMARY == "4"
    assert (e1.E2_AT, e1.LOW_BELOW) == (0.8, 0.5)
    assert e1.LATE_ITERS == 20 and e1.HELD_DIV == 10
    assert not e1.SMOKE or any(k.startswith("E1_") for k in os.environ)


# --------------------------------------------------------------------------
# partitions, pair counting, floors, the verdict
# --------------------------------------------------------------------------
def _part(classes_of_action, by_duration=None):
    """A 36-token census partition from a class per action (all its
    durations alike unless `by_duration` {(a, f): class})."""
    by_duration = by_duration or {}
    return [by_duration.get((a, f), classes_of_action[a]) for a, f in e1.GRID]


def test_restrict_relabels_by_first_appearance():
    part = _part([7, 7, 3, 3, 9, 3, 3, 3, 3], {(3, 16): 11})
    four = [e1.GRID.index((a, 4)) for a in range(9)]
    assert four == [0, 4, 8, 12, 16, 20, 24, 28, 32]
    assert e1.restrict(part, four) == [0, 0, 1, 1, 2, 1, 1, 1, 1]
    sixteen = [e1.GRID.index(tuple(T2_VOCAB[s])) for s in T2_GROUPS["16"]]
    assert sixteen == [14, 22, 34]
    assert e1.restrict(part, sixteen) == [0, 1, 1]      # (3,16) split off
    assert e1.restrict(part, []) == []


def test_components():
    assert e1.components(0, []) == []
    assert e1.components(1, [(0, 0)]) == [0]
    assert e1.components(5, [(3, 4), (0, 2), (2, 4)]) == [0, 1, 0, 0, 0]
    assert e1.components(5, [(4, 2), (2, 0), (4, 3)]) == [0, 1, 0, 0, 0]
    assert e1.components(4, []) == [0, 1, 2, 3]
    assert e1.components(4, [(1, 3), (1, 1)]) == [0, 1, 2, 1]


def test_pair_counts():
    assert e1.pair_counts([0, 0, 1, 1], [0, 0, 0, 1]) == (1, 2, 1, 2)
    assert e1.pair_counts([0, 0, 1, 1], [0, 0, 0, 1],
                          pairs=[(0, 1), (2, 3)]) == (1, 0, 1, 0)
    assert e1.pair_counts([0], [0]) == (0, 0, 0, 0)
    assert sum(e1.pair_counts(list(range(9)), [0] * 9)) == 36
    with pytest.raises(ValueError):
        e1.pair_counts([0, 0], [0])


def test_prf_and_its_zero_conventions():
    p, r, f = e1.prf(1, 2, 1)
    assert p == pytest.approx(1 / 3) and r == 0.5 and f == pytest.approx(0.4)
    assert e1.prf(0, 0, 3) == (None, 0.0, 0.0)     # all distinct, truth not
    assert e1.prf(0, 0, 0) == (None, None, 1.0)    # nothing equivalent: agree
    assert e1.prf(2, 0, 0) == (1.0, 1.0, 1.0)
    assert e1.prf(0, 4, 0) == (0.0, None, 0.0)


def test_the_trivial_rules_and_the_floor():
    truth = [0, 0, 1]
    triv = e1.trivial_partitions(3)
    assert e1.pair_counts(truth, triv["one"]) == (1, 2, 0, 0)
    assert e1.pair_counts(truth, triv["distinct"]) == (0, 0, 1, 2)
    f1s = {k: e1.prf(*e1.pair_counts(truth, v)[:3])[2]
           for k, v in triv.items()}
    assert f1s == {"one": pytest.approx(0.5), "distinct": 0.0}
    assert e1.floor_of(f1s) == pytest.approx(0.5)
    # a truth without any equivalent pair: all distinct is perfect
    f1s = {k: e1.prf(*e1.pair_counts([0, 1, 2], v)[:3])[2]
           for k, v in triv.items()}
    assert f1s == {"one": 0.0, "distinct": 1.0} and e1.floor_of(f1s) == 1.0


def test_the_ratio_class_and_the_member_counts():
    q = [0.5, 0.05, 0.0499, 0.4, 0.0001, 0, 0, 0, 0]
    assert e1.predicted_class(q, 0.1) == [0, 1, 3]   # 0.05 >= 0.1 x 0.5
    assert e1.predicted_class(q, 0.2) == [0, 3]
    assert e1.predicted_class(q, 0.05) == [0, 1, 2, 3]
    assert e1.predicted_class([1 / 9] * 9, 0.9) == list(range(9))
    assert e1.membership_counts([0, 1, 2], [1, 2, 5]) == (2, 1, 1)
    assert e1.membership_counts([4], [4]) == (1, 0, 0)


@pytest.mark.parametrize("args,want", [
    ((0.8, 0.3, None, None), ("E2", 0.8)),
    ((0.7999, 0.3, None, None), ("KAI_CALL_MID", 0.7999)),
    ((0.5, 0.3, None, None), ("KAI_CALL_MID", 0.5)),
    ((0.4999, 0.3, None, None), ("KAI_CALL_LOW", 0.4999)),
    # (1) does not beat its floor: (2)'s F1 decides though (1)'s is higher
    ((0.9, 0.95, 0.6, 0.5), ("KAI_CALL_MID", 0.6)),
    # equal to the floor is not beating it
    ((0.9, 0.9, 0.85, 0.85), ("KAI_CALL_LOW", None)),
    ((None, None, None, None), ("KAI_CALL_LOW", None)),
    ((0.55, 0.2, 0.81, 0.7), ("E2", 0.81)),
    ((0.85, 0.2, 0.3, 0.1), ("E2", 0.85)),
    ((0.3, 0.2, 0.45, 0.1), ("KAI_CALL_LOW", 0.45))])
def test_the_verdict_boundaries(args, want):
    assert e1.verdict(*args) == want


# --------------------------------------------------------------------------
# validity
# --------------------------------------------------------------------------
def test_the_validity_checks(monkeypatch):
    monkeypatch.setattr(e1, "MAX_DECISIONS", 0)
    ref = {"games": {"1": [100.0, 700, True, 150], "2": [80.0, 600, True,
                                                         120]}}
    games = {1: (100.0, 700, True, 150), 2: (80.0, 600, True, 120)}
    truth_games = {1: 150, 2: 120}
    scored = {1: 150, 2: 120}

    def v(**kw):
        a = dict(games=games, ref=ref, truth_games=truth_games,
                 scored=scored, token_mismatches=0, argmax_mismatches=0,
                 missing=0, census_ok=True)
        a.update(kw)
        n = a.pop("n_games", None)
        return e1.validity(**a, n_games=n)

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
    ok, why = v(token_mismatches=1, argmax_mismatches=2, missing=3)
    assert ok is False and len(why) == 3
    ok, why = v(games={1: games[1]}, n_games=2)
    assert ok is False and why == ["stopped after 1 of 2 games"]
    monkeypatch.setattr(e1, "MAX_DECISIONS", 5)          # a smoke: <= 5
    assert v(scored={1: 5, 2: 5}) == (True, [])
    ok, why = v(scored={1: 5, 2: 4})
    assert ok is False and "4 decisions scored of 5" in why[0]


def test_a_void_verdict_keeps_the_valid_one():
    rec = {"verdict": "E2"}
    e1.finalize(rec, False, ["game 3 differs"])
    assert rec["verdict"].startswith("VOID (game 3 differs")
    assert rec["verdict_if_valid"] == "E2"
    rec = {"verdict": "KAI_CALL_MID"}
    e1.finalize(rec, True, [])
    assert rec == {"verdict": "KAI_CALL_MID"}
    e1.finalize(rec, None, [])
    assert rec == {"verdict": "KAI_CALL_MID"}


# --------------------------------------------------------------------------
# the token groups
# --------------------------------------------------------------------------
def test_t2s_token_groups():
    g = e1.token_groups(T2_VOCAB)
    assert g == T2_GROUPS
    assert g["4"] == [s for s in range(18) if T2_VOCAB[s][1] == 4]
    assert [T2_VOCAB[s][0] for s in g["4"]] == list(range(9))
    with pytest.raises(SystemExit, match="one per action"):
        e1.token_groups(T2_VOCAB[1:])
    with pytest.raises(SystemExit, match="one per action"):
        e1.token_groups(((1, 4), (0, 4)) + tuple((a, 4) for a in range(2, 9)))
    # TK's grid (T0): every group has nine tokens, 8 frames none
    grid = e1.token_groups(e1.GRID)
    assert [len(grid[k]) for k in ("4", "16", "32")] == [9, 9, 9]


# --------------------------------------------------------------------------
# own play: a simulated life, the reader, the IDM's input
# --------------------------------------------------------------------------
def _frame(rng):
    return rng.integers(0, 256, size=(3, 84, 84), dtype=np.uint8)


def _life(vocab, slots, rng):
    """A life as play_episode / play_token record it (REWARD_GROWTH raw
    reward tuples): (traj, the new frames each decision's last base step
    showed)."""
    obs = _frame(rng)
    stack = muzero.initial_frame_stack(_frame(rng))
    traj, prev, shown = [], (), []
    for slot in slots:
        stack = muzero.push_frame(stack, obs)
        n = vocab[slot][1] // muzero.TOKEN_BASE_FRAMES
        after, inner = stack, []
        for i in range(n):
            obs = _frame(rng)
            inner.append(0.0)
            if i < n - 1:
                after = muzero.push_frame(after, obs)
        traj.append((stack, int(slot), prev))
        shown.append(obs)
        stack, prev = after, tuple(inner)
    traj.append((muzero.push_frame(stack, obs), 0, prev))
    return traj, shown


def test_the_idm_input_and_a_lifes_samples():
    rng = np.random.default_rng(0)
    slots = [0, 15, 14, 4, 11, 11, 5, 3]      # 4, 8, 4, 16, 4, 4, 32, 4
    traj, shown = _life(T2_VOCAB, slots, rng)
    samples = e1.life_idm_samples(traj, T2_VOCAB)
    four = [i for i, s in enumerate(slots) if T2_VOCAB[s][1] == 4]
    assert [y for _, y in samples] == [T2_VOCAB[slots[i]][0] for i in four]
    assert [y for _, y in samples] == [0, 8, 6, 6, 3]
    for (x, _), i in zip(samples, four):
        assert x.shape == (9, 84, 84) and x.dtype == np.uint8
        assert np.array_equal(x[:6], traj[i][0][6:])   # s_t's last two
        assert np.array_equal(x[6:], shown[i])         # the new frame
        assert np.array_equal(x[6:], traj[i + 1][0][9:])
    # an 8-frame token's next stack is two frames on: not the 4-frame shift
    with pytest.raises(AssertionError, match="one-frame shift"):
        e1.idm_input(traj[1][0], traj[2][0])
    with pytest.raises(AssertionError):
        e1.idm_input(traj[0][0], traj[1][0][:9])
    assert e1.life_idm_samples(traj[:1], T2_VOCAB) == []


def _write_log(folder, lives_per_it, seed=0):
    """An OwnPlayLog with `lives_per_it[i]` simulated lives at iteration
    i + 1; returns the lives in order."""
    log = ablation.OwnPlayLog(folder)
    rng = np.random.default_rng(seed)
    out = []
    for it, n in enumerate(lives_per_it, start=1):
        for k in range(n):
            slots = [int(s) for s in rng.integers(0, 18, size=3 + k)]
            traj, _ = _life(T2_VOCAB, slots, rng)
            inner = [[] for _ in slots]
            log.write(it, traj, {"inner": inner, "real_done": k == n - 1,
                                 "temperature": 0.5}, True, 10, 100 * it)
            out.append((it, traj))
    return out


def _same_life(a, b):
    return (a["iteration"] == b["iteration"] and len(a["traj"]) == len(b["traj"])
            and all(np.array_equal(x[0], y[0]) and x[1:] == y[1:]
                    for x, y in zip(a["traj"], b["traj"])))


def test_the_streaming_reader_is_read_own_play(tmp_path):
    folder = tmp_path / "own-play-x"
    written = _write_log(folder, [2, 1, 3])
    want = ablation.read_own_play(folder, 3)
    got = list(e1.iter_own_play(folder, 3))
    assert [it for it, _ in got] == [it for it, _ in written] == \
        [1, 1, 2, 3, 3, 3]
    assert len(got) == len(want) == 6
    assert all(_same_life(g, w) for (_, g), w in zip(got, want))
    assert [it for it, _ in e1.iter_own_play(folder, 2)] == [1, 1, 2]
    assert [it for it, _ in e1.iter_own_play(folder, 3, max_lives=4)] == \
        [1, 1, 2, 3]
    with pytest.raises(SystemExit, match="iteration 4 is missing"):
        list(e1.iter_own_play(folder, 4))


def test_the_reservoir_and_the_held_out_lives():
    r = e1.Reservoir(10, 0)
    for i in range(7):
        r.add(i)
    assert r.items == list(range(7)) and r.seen == 7
    a, b, c = e1.Reservoir(5, 0), e1.Reservoir(5, 0), e1.Reservoir(5, 1)
    for i in range(100):
        a.add(i), b.add(i), c.add(i)
    assert len(a.items) == 5 and a.items == b.items and a.items != c.items
    assert len(set(a.items)) == 5 and set(a.items) <= set(range(100))
    assert a.seen == 100
    assert [e1.held_out_count(n) for n in (0, 1, 9, 10, 11, 30, 126)] == \
        [0, 1, 1, 1, 2, 3, 13]


# --------------------------------------------------------------------------
# (A) on tiny random nets
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


def _hand_error(nets, s, slot, s_next):
    with torch.no_grad():
        x = torch.tensor(s[None], dtype=torch.float32) / 255.0
        xn = torch.tensor(s_next[None], dtype=torch.float32) / 255.0
        nxt, _, _ = nets.dyn(nets.rep(x), torch.tensor([slot]))
        online = nets.proj(nxt, with_prediction=True)
        target = nets.proj(nets.rep(xn), with_prediction=False)
        # muzero.consistency_loss's per-sample term is -cos
        loss = muzero.consistency_loss(online, target)
    return 1.0 + float(loss)


def test_the_consistency_errors_against_a_hand_computation():
    nets = _tiny_nets()
    rng = np.random.default_rng(1)
    traj, _ = _life(T2_VOCAB, [0, 4, 15, 11, 17, 2, 9], rng)
    hand = [_hand_error(nets, traj[i][0], traj[i][1], traj[i + 1][0])
            for i in range(len(traj) - 1)]
    for batch in (1, 3, 64):
        got = e1.life_errors(nets, traj, batch=batch)
        assert got.shape == (7,)
        assert got == pytest.approx(hand, abs=1e-5)
    assert len(set(np.round(hand, 6))) == 7          # the inputs matter
    assert e1.life_errors(nets, traj[:1]).shape == (0,)
    # the slot matters: another token, another error
    other = [(s, (a + 1) % 18, p) for s, a, p in traj]
    assert not np.allclose(e1.life_errors(nets, other), hand)


def test_t2s_pass_calibrates_samples_and_counts(tmp_path):
    nets = _tiny_nets()
    folder = tmp_path / "own-play-t2"
    written = _write_log(folder, [1, 2, 1, 2], seed=3)
    errors, per_life, late = e1.t2_pass(e1.iter_own_play(folder, 4),
                                        T2_VOCAB, nets, last_iteration=4)
    hand = np.concatenate([e1.life_errors(nets, t) for _, t in written])
    assert errors == pytest.approx(hand) and len(errors) == sum(
        len(t) - 1 for _, t in written)
    assert len(per_life) == 6
    assert [len(s) for s in per_life] == [len(e1.life_idm_samples(
        t, T2_VOCAB)) for _, t in written]
    want = np.zeros(18, dtype=np.int64)     # LATE_ITERS 20 >= 4: all of them
    for _, t in written:
        for rec in t[:-1]:
            want[rec[1]] += 1
    assert late.tolist() == want.tolist()
    # a 2-iteration window: only iterations 3 and 4
    old = e1.LATE_ITERS
    try:
        e1.LATE_ITERS = 2
        _, _, late2 = e1.t2_pass(e1.iter_own_play(folder, 4), T2_VOCAB,
                                 None, last_iteration=4)
    finally:
        e1.LATE_ITERS = old
    want2 = np.zeros(18, dtype=np.int64)
    for it, t in written:
        if it >= 3:
            for rec in t[:-1]:
                want2[rec[1]] += 1
    assert late2.tolist() == want2.tolist() and want2.sum() < want.sum()
    tau = float(np.median(errors))
    assert e1.calibration_read(errors)["p50"] == pytest.approx(tau, abs=1e-6)


def test_the_token_projections_and_their_classes():
    nets = _tiny_nets()
    rng = np.random.default_rng(2)
    stack = muzero.initial_frame_stack(_frame(rng))
    z = e1.token_projections(nets, stack, 18)
    assert z.shape == (18, 32)
    with torch.no_grad():
        lat = nets.rep(torch.tensor(stack[None], dtype=torch.float32) / 255.0)
        for s in (0, 7, 17):
            nxt, _, _ = nets.dyn(lat, torch.tensor([s]))
            assert torch.allclose(nets.proj(nxt, with_prediction=True)[0],
                                  z[s], atol=1e-5)
    d = e1.cos_distances(z)
    assert d.shape == (18, 18) and np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0, atol=1e-6)
    slots = T2_GROUPS["4"]
    assert e1.group_prediction(d, slots, -1.0) == list(range(9))
    assert e1.group_prediction(d, slots, 2.5) == [0] * 9
    # a threshold between two pair distances: the components of those pairs
    sub = sorted(d[slots[a], slots[b]] for a, b in e1.all_pairs(9))
    tau = (sub[3] + sub[4]) / 2
    edges = [(a, b) for a, b in e1.all_pairs(9)
             if d[slots[a], slots[b]] <= tau]
    assert len(edges) == 4
    assert e1.group_prediction(d, slots, tau) == e1.components(9, edges)


# --------------------------------------------------------------------------
# (B): the IDM
# --------------------------------------------------------------------------
def test_the_idm_training_is_deterministic_and_leaves_the_rng_alone():
    rng = np.random.default_rng(4)
    x = rng.integers(0, 256, size=(24, 9, 84, 84), dtype=np.uint8)
    y = rng.integers(0, 9, size=24)
    held = (x[:5], y[:5])
    torch_rng, np_rng = torch.get_rng_state(), np.random.get_state()
    a, ha = e1.train_idm(x, y, 2, seed=0, batch=8, held=held)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert all(np.array_equal(u, v) if isinstance(u, np.ndarray) else u == v
               for u, v in zip(np.random.get_state(), np_rng))
    b, hb = e1.train_idm(x, y, 2, seed=0, batch=8, held=held)
    c, _ = e1.train_idm(x, y, 2, seed=1, batch=8, held=held)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa) and ha == hb
    assert not all(torch.equal(sa[k], sc[k]) for k in sa)
    assert [h["epoch"] for h in ha] == [1, 2]
    assert {"train_loss", "held_acc", "held_nll"} <= set(ha[0])
    q = e1.idm_probs(a, x)
    assert q.shape == (24, 9) and np.allclose(q.sum(1), 1.0)
    acc, nll = e1.idm_scores(a, x, y)
    assert 0.0 <= acc <= 1.0 and nll > 0
    assert e1.idm_scores(a, x[:0], y[:0]) == (None, None)
    # an untrained init is the seed's, whatever the process RNG says
    torch.manual_seed(123)
    a0, _ = e1.train_idm(x, y, 0, seed=0)
    torch.manual_seed(456)
    b0, _ = e1.train_idm(x, y, 0, seed=0)
    assert all(torch.equal(u, v) for u, v in zip(a0.state_dict().values(),
                                                 b0.state_dict().values()))


def test_the_idm_learns_an_easy_set():
    """The labels are written into the new frame: the net must learn
    them (the training loop updates the right thing)."""
    rng = np.random.default_rng(5)
    y = np.arange(72) % 9
    x = rng.integers(0, 40, size=(72, 9, 84, 84), dtype=np.uint8)
    for i, a in enumerate(y):
        x[i, 6:, :, a * 9:(a + 1) * 9] = 255
    net, hist = e1.train_idm(x, y, 15, seed=0, batch=16, lr=1e-3)
    acc, _ = e1.idm_scores(net, x, y)
    assert acc >= 0.9 and hist[-1]["train_loss"] < hist[0]["train_loss"]


# --------------------------------------------------------------------------
# one scored decision and the hook
# --------------------------------------------------------------------------
def _fake_run(prior):
    nm = SimpleNamespace(initial_inference=lambda stack: (np.array(prior),
                                                          0.0))
    return SimpleNamespace(searcher=SimpleNamespace(network_manager=nm))


def _prior(token):
    p = np.full(18, 0.01)
    p[token] = 0.9
    return p


def test_a_scored_decision():
    nets = _tiny_nets()
    rng = np.random.default_rng(6)
    traj, _ = _life(T2_VOCAB, [11, 9], rng)
    stack, nxt = traj[0][0], traj[1][0]
    part = _part([0, 0, 1, 1, 1, 2, 1, 1, 1], {(3, 32): 5})
    truth = {(1, 0): (11, part), (1, 1): (9, part), (1, 2): (3, part)}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        idm = e1.IDM().eval()
    out = io.StringIO()
    sc = e1.Scorer(_fake_run(_prior(11)), nets, T2_VOCAB, truth, 0.05, idm,
                   out)
    sc.game = 1
    sc.decision(11, stack, nxt)
    rec = json.loads(out.getvalue().splitlines()[0])
    assert (rec["game"], rec["d"], rec["token"]) == (1, 0, 11)
    g = rec["groups"]
    assert g["4"]["truth"] == [0, 0, 1, 1, 1, 2, 1, 1, 1]
    assert g["16"]["truth"] == [0, 1, 0]           # actions 3, 5, 8
    assert g["32"]["truth"] == [0, 1, 2, 1, 1]     # (3, 32) split off
    assert [len(g[k]["dist"]) for k in ("4", "16", "32")] == [36, 3, 10]
    assert all(set(g[k]["pred"]) == {"0.5", "1", "2"} for k in g)
    d = e1.cos_distances(e1.token_projections(nets, stack, 18))
    for k, slots in T2_GROUPS.items():
        assert g[k]["dist"] == pytest.approx(
            [d[slots[a], slots[b]] for a, b in e1.all_pairs(len(slots))],
            abs=1e-6)
        assert g[k]["pred"]["1"] == e1.group_prediction(d, slots, 0.05)
        assert g[k]["pred"]["2"] == e1.group_prediction(d, slots, 0.1)
    b = rec["b"]
    assert b["action"] == 6                         # slot 11 = (6, 4)
    assert b["truth_class"] == [2, 3, 4, 6, 7, 8]
    q = e1.idm_probs(idm, e1.idm_input(stack, nxt)[None])[0]
    assert b["q"] == pytest.approx(q, abs=1e-6)
    assert b["pred_class"]["0.1"] == e1.predicted_class(q, 0.1)
    assert sc.token_mismatches == sc.argmax_mismatches == sc.missing == 0
    # a 16-frame token: no (B); the prior's argmax is not the token
    sc.run = _fake_run(_prior(4))
    sc.decision(9, nxt, nxt)
    assert "b" not in sc.recs[-1] and sc.argmax_mismatches == 1
    # the census's token differs (slot 3, 4-frame, 4-frame shift required)
    sc.run = _fake_run(_prior(3))
    with pytest.raises(AssertionError, match="one-frame shift"):
        sc.decision(3, stack, stack)
    sc.index = 2
    sc.decision(3, traj[0][0], traj[1][0])
    assert sc.token_mismatches == 0
    sc.truth[(1, 3)] = (5, part)
    sc.decision(3, traj[0][0], traj[1][0])
    assert sc.token_mismatches == 1
    sc.decision(3, traj[0][0], traj[1][0])           # (1, 4): no census line
    assert sc.missing == 1 and sc.recs[-1]["missing"] is True
    assert "groups" not in sc.recs[-1]
    assert [r["d"] for r in sc.recs] == [0, 1, 2, 3, 4]


def test_the_hook_scores_the_next_stack_and_restores(monkeypatch):
    rng = np.random.default_rng(7)
    stack = muzero.initial_frame_stack(_frame(rng))
    after = muzero.push_frame(stack, _frame(rng))
    obs = rng.integers(0, 256, size=(210, 160, 3), dtype=np.uint8)
    played = []

    def orig(env, token, stack, inner=None):
        played.append(token)
        return obs, 0.0, False, False, after, 1

    monkeypatch.setattr(ablation, "play_token", orig)
    sc = e1.Scorer(None, None, T2_VOCAB, {}, 0.1, None, None)
    seen = []

    def decide(token, s, nxt):
        seen.append((token, s, nxt))
        sc.index += 1
    monkeypatch.setattr(sc, "decision", decide)
    monkeypatch.setattr(e1, "MAX_DECISIONS", 0)
    with sc.hooked():
        assert ablation.play_token("env", 5, stack) == (obs, 0.0, False,
                                                        False, after, 1)
    assert ablation.play_token is orig
    want = muzero.push_frame(after, muzero.preprocess_frame(
        obs, muzero.IMAGE_RESHAPE))
    assert seen[0][0] == 5 and seen[0][1] is stack
    assert np.array_equal(seen[0][2], want)
    # a smoke stops the scoring, not the game
    monkeypatch.setattr(e1, "MAX_DECISIONS", 2)
    with sc.hooked():
        for t in range(4):
            ablation.play_token(None, t, stack)
    assert sc.index == 2 and played == [5, 0, 1, 2, 3]
    with pytest.raises(RuntimeError):
        with sc.hooked():
            assert ablation.play_token is not orig
            raise RuntimeError
    assert ablation.play_token is orig


# --------------------------------------------------------------------------
# the reads on hand-made records
# --------------------------------------------------------------------------
def _rec(game, d, four_truth, four_pred, b=None):
    rec = {"game": game, "d": d, "token": 0, "groups": {
        "4": {"truth": four_truth, "dist": [0.0] * 36,
              "pred": {"0.5": list(range(9)), "1": four_pred,
                       "2": [0] * 9}},
        "16": {"truth": [0, 0, 1], "dist": [0.0] * 3,
               "pred": {"0.5": [0, 1, 2], "1": [0, 0, 1], "2": [0, 0, 0]}},
        "32": {"truth": [0, 1, 2, 3, 4], "dist": [0.0] * 10,
               "pred": {"0.5": [0, 1, 2, 3, 4], "1": [0, 1, 2, 3, 4],
                        "2": [0] * 5}}}}
    if b is not None:
        rec["b"] = b
    return rec


def _b(action, q, truth_class):
    return {"action": action, "q": q, "truth_class": truth_class,
            "pred_class": {k: e1.predicted_class(q, r)
                           for k, r in e1.B_RATIOS.items()}}


def _records():
    q = [0.01, 0.01, 0.01, 0.01, 0.5, 0.2, 0.2, 0.03, 0.03]
    split = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    return [_rec(1, 0, split, split, _b(4, q, [4, 5, 6, 7, 8])),
            _rec(2, 0, [0] * 9, [0, 0, 0, 0, 0, 1, 1, 1, 1]),
            {"game": 2, "d": 1, "token": 0, "missing": True}]


def test_the_reads_by_hand():
    s = e1.summarize(_records(), T2_GROUPS, [0], {1: 1700, 2: 1880},
                     held_acc=0.5)
    r1 = s["read_1"]
    # rec 1: 16 equivalent pairs, all found; rec 2: 36, 16 found
    assert (r1["tp"], r1["fp"], r1["fn"]) == (32, 0, 20)
    assert r1["decisions"] == 2 and r1["pairs_per_decision"] == 36
    assert r1["precision"] == 1.0 and r1["recall"] == round(32 / 52, 4)
    assert r1["f1"] == round(64 / 84, 4)
    assert r1["floor_rules"] == {"one": round(104 / 124, 4), "distinct": 0.0}
    assert r1["floor"] == round(104 / 124, 4) and r1["beats_floor"] is False
    assert r1["classes_pred_mean"] == 2.0 and r1["classes_truth_mean"] == 1.5
    r2 = s["read_2"]
    assert r2["decisions"] == 1 and (r2["tp"], r2["fp"], r2["fn"]) == (3, 0, 2)
    assert r2["f1"] == 0.75 and r2["recall"] == 0.6
    assert r2["floor_rules"] == {"one": round(10 / 14, 4),
                                 "distinct": round(2 / 6, 4)}
    assert r2["beats_floor"] is True and r2["idm_held_acc"] == 0.5
    assert r2["top1"] == 1.0 and r2["top1_in_class"] == 1.0
    assert r2["pred_class_mean"] == 3.0 and r2["truth_class_mean"] == 5.0
    ex = r2["excluding_chosen"]
    assert (ex["tp"], ex["fp"], ex["fn"]) == (2, 0, 2)
    assert ex["floor_rules"] == {"one": round(8 / 12, 4), "distinct": 0.0}
    # (1) under its floor; (2) beats its own: (2)'s 0.75 decides
    assert s["verdict"] == "KAI_CALL_MID" and s["verdict_f1"] == 0.75
    text = json.dumps(s)
    assert '"_f1"' not in text and '"_floor"' not in text   # internals
    # (3): the frequency split, rare = slot 0 alone (position 0 of the
    # 4-frame group)
    fa = s["read_3"]["frequency"]["a"]["4"]
    assert fa["rare"]["pairs_per_decision"] == 8
    assert (fa["rare"]["tp"], fa["rare"]["fp"], fa["rare"]["fn"]) == (7, 0, 4)
    assert (fa["rest"]["tp"], fa["rest"]["fn"]) == (25, 16)
    assert s["read_3"]["frequency"]["a"]["16"]["rare"] == {
        "decisions": 2, "pairs_per_decision": 0}
    fb = s["read_3"]["frequency"]["b"]
    assert fb["rare"] == {"decisions": 0} and fb["rest"]["decisions"] == 1
    assert s["read_3"]["frequency"]["rare_slots"] == [0]
    # (3): the other groups, the families
    g16 = s["read_3"]["groups"]["16"]
    assert (g16["tp"], g16["fp"], g16["fn"]) == (2, 0, 0) and g16["f1"] == 1.0
    g32 = s["read_3"]["groups"]["32"]
    assert (g32["tp"], g32["fp"], g32["fn"]) == (0, 0, 0) and g32["f1"] == 1.0
    assert g32["floor_rules"] == {"one": 0.0, "distinct": 1.0}
    assert g32["beats_floor"] is False
    fam = s["read_3"]["families"]
    assert set(fam) == {"1700", "1880"} and fam["1700"]["games"] == [1]
    assert fam["1700"]["a"]["f1"] == 1.0 and fam["1880"]["b"] == {
        "decisions": 0}
    # (4): robustness
    a4 = s["read_4"]["a"]
    assert set(a4) == {"0.5", "2"}
    assert (a4["0.5"]["4"]["tp"], a4["0.5"]["4"]["fn"]) == (0, 52)
    assert (a4["2"]["4"]["tp"], a4["2"]["4"]["fp"]) == (52, 20)
    b4 = s["read_4"]["b"]
    assert set(b4) == {"0.05", "0.2"}
    assert (b4["0.05"]["tp"], b4["0.05"]["fp"]) == (5, 0)


def test_the_rare_split_with_t2s_rare_set():
    """(A1) with the full logs' rare set: the seven rare 4-frame tokens
    leave one 4-frame pair (UPLEFT, DOWNLEFT: positions 6, 8) outside it;
    the chosen class {4..8} holds rare actions."""
    rare = [0, 1, 2, 3, 6, 8, 12]
    s = e1.summarize(_records(), T2_GROUPS, rare, {1: 1700, 2: 1880})
    f = s["read_3"]["frequency"]
    assert f["rare_slots"] == rare
    assert f["a"]["4"]["rest"]["pairs_per_decision"] == 1
    assert (f["a"]["4"]["rest"]["tp"], f["a"]["4"]["rest"]["fn"]) == (2, 0)
    assert f["a"]["4"]["rare"]["pairs_per_decision"] == 35
    assert (f["a"]["4"]["rare"]["tp"], f["a"]["4"]["rare"]["fn"]) == (30, 20)
    assert f["a"]["16"]["rare"]["pairs_per_decision"] == 0     # none rare
    assert f["a"]["32"]["rest"]["pairs_per_decision"] == 10
    assert f["b"]["rare"]["decisions"] == 1 and f["b"]["rest"] == {
        "decisions": 0}
    # the split is a report: the decision does not move
    assert s["verdict"] == e1.summarize(_records(), T2_GROUPS, [],
                                        {1: 1700, 2: 1880})["verdict"]


def test_the_rare_rule():
    """(A1): under 1% of ALL the window's decisions, strictly."""
    # T2's own-play log, iterations 23-42 (3,141 decisions; 1% = 31.41)
    plays = [0, 1, 4, 8, 315, 337, 15, 179, 5, 1153, 351, 232, 4, 295, 115,
             37, 44, 46]
    assert sum(plays) == 3141
    assert e1.rare_slots(plays) == [0, 1, 2, 3, 6, 8, 12]
    assert all(T2_VOCAB[s][1] == 4 for s in e1.rare_slots(plays))
    assert e1.rare_slots([1, 99]) == [] and e1.rare_slots([1, 100]) == [0]
    assert e1.rare_slots([0, 50, 50]) == [0]
    assert e1.rare_slots([0, 0, 0]) == []          # an empty window
    assert e1.rare_slots(np.array([2, 300], dtype=np.int64)) == [0]
    assert e1.RARE_PER == 100


def test_the_reads_are_the_same_from_the_decisions_file():
    recs = _records()
    back = [json.loads(json.dumps(r)) for r in recs]
    args = (T2_GROUPS, [0, 12], {1: 1700, 2: 1880})
    assert e1.summarize(back, *args) == e1.summarize(recs, *args)


def test_the_reads_with_nothing_scored():
    s = e1.summarize([], T2_GROUPS, [], {})
    assert s["read_1"] == {"decisions": 0, "pairs_per_decision": 0}
    assert s["read_2"]["decisions"] == 0
    assert s["verdict"] == "KAI_CALL_LOW" and s["verdict_f1"] is None


# --------------------------------------------------------------------------
# the census, the lineage, the output files
# --------------------------------------------------------------------------
def test_the_census_row_and_its_truth(tmp_path):
    p = tmp_path / "census.jsonl"
    rows = [{"run": "r", "iteration": 42, "n_games": 30, "max_decisions": None,
             "env": e1.ENV_ID, "main_line_ok": False},
            {"run": "r", "iteration": 42, "n_games": 1, "max_decisions": 5,
             "env": e1.ENV_ID, "main_line_ok": None},       # a smoke
            {"run": "r", "iteration": 42, "n_games": 30, "max_decisions": None,
             "env": e1.ENV_ID, "main_line_ok": True},
            {"run": "r", "iteration": 41, "n_games": 30, "max_decisions": None,
             "env": e1.ENV_ID, "main_line_ok": False}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    assert e1.census_row(p, "r", 42)["main_line_ok"] is True
    assert e1.census_row(p, "r", 40) is None
    assert e1.census_row(tmp_path / "none.jsonl", "r", 42) is None
    d = tmp_path / "dec.jsonl"
    lines = [{"game": 1, "d": 0, "token": 3, "partition": [0] * 36},
             {"game": 1, "d": 1, "token": 9, "partition": list(range(36))},
             {"game": 2, "d": 0, "token": 5, "partition": [0] * 36}]
    d.write_text("\n".join(json.dumps(l) for l in lines) + "\n\n")
    truth, per_game = e1.read_truth(d)
    assert per_game == {1: 2, 2: 1} and truth[(1, 1)][0] == 9
    assert truth[(1, 1)][1] == list(range(36))
    d.write_text("\n".join(json.dumps(l) for l in lines + lines[:1]) + "\n")
    with pytest.raises(SystemExit, match="twice"):
        e1.read_truth(d)


def _tree(tmp_path, t2_vocab=T2_VOCAB):
    results = tmp_path / "tree" / "benchmarks" / "results"
    t1_vocab = [[0, 4], [0, 16]] + [[a, 4] for a in range(1, 9)]
    for name, j in ((e1.PREPS[0], {"master_vocab": [list(t) for t in e1.GRID],
                                   "vocab": t1_vocab,
                                   "log_dir": "benchmarks/results/run-t0/"
                                              "own-play-a-t0-seed0",
                                   "log_iterations": 47}),
                    (e1.PREPS[1], {"master_vocab": t1_vocab,
                                   "vocab": [list(t) for t in t2_vocab],
                                   "log_dir": "benchmarks/results/run-t1/"
                                              "own-play-a-t1-seed0",
                                   "log_iterations": 46})):
        (results / name).mkdir(parents=True)
        (results / name / "vocab.json").write_text(json.dumps(j))
    for k in (0, 1, 2):
        (results / f"run-t{k}" / f"own-play-a-t{k}-seed0").mkdir(parents=True)
    snaps = results / "run-t2" / "snapshots-a-t2-seed0"
    snaps.mkdir()
    return snaps / "weights-iter042.pth", t1_vocab


def test_the_lineage(tmp_path):
    snap, t1_vocab = _tree(tmp_path)
    logs = e1.lineage(snap, T2_VOCAB, 42)
    assert [l["name"] for l in logs] == ["T2", "T0", "T1"]
    assert [l["last"] for l in logs] == [42, 47, 46]
    assert logs[0]["folder"] == snap.parent.parent / "own-play-a-t2-seed0"
    assert logs[1]["folder"].name == "own-play-a-t0-seed0"
    assert logs[1]["vocab"] == e1.GRID
    assert logs[2]["vocab"] == tuple(tuple(t) for t in t1_vocab)
    assert logs[0]["vocab"] == T2_VOCAB
    with pytest.raises(SystemExit, match="not the snapshot's"):
        e1.lineage(snap, T2_VOCAB[:-1], 42)
    (snap.parent.parent / "own-play-a-t2-seed0").rmdir()
    with pytest.raises(SystemExit, match="T2's own-play log"):
        e1.lineage(snap, T2_VOCAB, 42)


def test_the_lineage_refuses_a_broken_chain(tmp_path):
    snap, _ = _tree(tmp_path)
    p = snap.parents[2] / e1.PREPS[1] / "vocab.json"
    j = json.loads(p.read_text())
    j["master_vocab"] = j["master_vocab"][1:]
    p.write_text(json.dumps(j))
    with pytest.raises(SystemExit, match="master vocabulary"):
        e1.lineage(snap, T2_VOCAB, 42)


def test_a_full_e1_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(e1, "HERE", tmp_path)
    summary, dec = e1.out_files("run-x", smoke=False)
    assert summary.name == "probe_e1_equivalence-run-x.jsonl"
    assert dec.name == "probe_e1_equivalence-run-x-decisions.jsonl"
    dec.write_text("{}\n")
    with pytest.raises(SystemExit, match="never overwritten"):
        e1.out_files("run-x", smoke=False)
    s, d = e1.out_files("run-x", smoke=True)
    assert s.name == "probe_e1_equivalence-run-x-smoke.jsonl"
    assert d.name == "probe_e1_equivalence-run-x-smoke-decisions.jsonl"
