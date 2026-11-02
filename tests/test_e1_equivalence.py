"""E1, a model trained on the agent's own play against E0's outcome classes
(CURRICULUM "Next experiment (pre-registered) -- E1", Kai 2026-11-02).
Pins: the partitions' restriction and components, pair counting and its
pooled precision / recall / F1 (the 0/0 conventions), the trivial-rule
floors, (B)'s ratio class and member counts, the verdict's boundaries
(A5: merge precision / recall, pooled and per family) and VOID, that no
trivial rule reaches E2 on E0's census truth, (A6)'s rarity splits, a
confidently wrong IDM, (A)'s raw-vs-acting nets, the preflight refusals
(threads, the census's snapshot), A4's fixed IDM settings, the token
groups, the own-play reader against ablation.read_own_play,
the IDM's input (the 4-frame shift) on a simulated life, the reservoir and
the held-out split, (A)'s consistency errors and token projections on tiny
random nets against a hand computation, the IDM's training determinism (and
the process RNG left alone), one scored decision, the hook's next stack, the
reads on hand-made records, the census / lineage / output files, and -- in
a subprocess -- that importing the probe changes nothing process-wide."""
import copy
import inspect
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
    assert e1.GRID == tuple((a, f) for a in range(9) for f in (4, 8, 16, 32))
    assert e1.TAU_MULTS == {"0.5": 0.5, "1": 1.0, "2": 2.0}
    assert e1.B_RATIOS == {"0.05": 0.05, "0.1": 0.1, "0.2": 0.2}
    assert e1.B_RATIO == "0.1" and e1.PRIMARY == "4"
    assert (e1.E2_P, e1.E2_R, e1.E2_FAMILIES) == (0.9, 0.5, 6)     # (A5)
    assert (e1.LOW_P, e1.LOW_R) == (0.7, 0.2)
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


def test_the_bars():
    assert e1.meets(0.9, 0.5, 0.9, 0.5)                   # inclusive
    assert type(e1.meets(np.float64(0.95), np.float64(0.6), 0.9, 0.5)) is bool
    assert not e1.meets(0.8999, 0.5, 0.9, 0.5)
    assert not e1.meets(0.9, 0.4999, 0.9, 0.5)
    assert not e1.meets(None, 1.0, 0.0, 0.0)              # no predicted merge
    assert not e1.meets(1.0, None, 0.0, 0.0)              # no true merge


NONE = (None, None, 0, None)


@pytest.mark.parametrize("cands,want", [
    # pooled, in 6 of 8 families, junction precision 0.9 (inclusive)
    ({"A": (0.9, 0.5, 6, 0.9), "B": NONE}, ("E2", ["A"])),
    ({"A": (0.95, 0.9, 8, 0.95), "B": (0.91, 0.6, 7, 1.0)},
     ("E2", ["A", "B"])),
    ({"A": (0.5, 0.1, 0, 1.0), "B": (0.92, 0.55, 6, 0.9)}, ("E2", ["B"])),
    # pooled E2 but 5 families: Kai's call (mid)
    ({"A": (0.95, 0.9, 5, 1.0), "B": NONE}, ("KAI_CALL_MID", ["A"])),
    # families do not rescue a pooled miss
    ({"A": (0.8999, 0.9, 8, 1.0), "B": NONE}, ("KAI_CALL_MID", ["A"])),
    ({"A": (0.95, 0.4999, 8, 1.0), "B": NONE}, ("KAI_CALL_MID", ["A"])),
    # (A8): the junction bar, for the same candidate
    ({"A": (0.95, 0.9, 8, 0.8999), "B": NONE}, ("KAI_CALL_MID", ["A"])),
    ({"A": (0.95, 0.9, 8, None), "B": NONE}, ("KAI_CALL_MID", ["A"])),
    ({"A": (0.95, 0.9, 8, 0.5), "B": (0.95, 0.9, 4, 1.0)},
     ("KAI_CALL_MID", ["A", "B"])),
    # the low bars, inclusive, untouched by the junction bar
    ({"A": (0.7, 0.2, 0, None), "B": (0.1, 0.9, 0, 1.0)},
     ("KAI_CALL_MID", ["A"])),
    ({"A": (0.6999, 0.9, 0, 1.0), "B": (0.99, 0.1999, 8, 1.0)},
     ("KAI_CALL_LOW", [])),
    # a None precision (no predicted merge) fails every bar
    ({"A": (None, 0.0, 8, 1.0), "B": (None, None, 8, 1.0)},
     ("KAI_CALL_LOW", []))])
def test_the_verdict_boundaries(cands, want):
    assert e1.verdict(cands) == want


def test_the_junction_bar_is_a8s_alone():
    """Without the junction bar (A7's rule, reported) the junction
    precision is ignored."""
    cands = {"A": (0.95, 0.9, 8, 0.5), "B": NONE}
    assert e1.verdict(cands) == ("KAI_CALL_MID", ["A"])
    assert e1.verdict(cands, junction_bar=False) == ("E2", ["A"])
    assert (e1.JUNCTION_CLASSES, e1.JUNCTION_P) == (3, 0.9)
    assert not e1.is_junction([0] * 9)
    assert not e1.is_junction([0, 1, 0, 1, 1, 0, 0, 0, 1])
    assert e1.is_junction([0, 1, 2, 0, 0, 0, 0, 0, 0])


def test_the_previous_tokens():
    truth = {(1, 0): (5, None), (1, 1): (9, None), (1, 2): (11, None),
             (2, 0): (3, None), (2, 1): (4, None)}
    assert e1.previous_tokens(truth) == {(1, 0): None, (1, 1): 5, (1, 2): 9,
                                         (2, 0): None, (2, 1): 3}


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


def _write_log(folder, lives_per_it, seed=0, vocab=T2_VOCAB):
    """An OwnPlayLog with `lives_per_it[i]` simulated lives at iteration
    i + 1 (slots of `vocab`); returns the lives in order."""
    log = ablation.OwnPlayLog(folder)
    rng = np.random.default_rng(seed)
    out = []
    for it, n in enumerate(lives_per_it, start=1):
        for k in range(n):
            slots = [int(s) for s in rng.integers(0, len(vocab),
                                                  size=3 + k)]
            traj, _ = _life(vocab, slots, rng)
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
    errors, per_life, late, lengths = e1.t2_pass(
        e1.iter_own_play(folder, 4), T2_VOCAB, nets, last_iteration=4)
    hand = np.concatenate([e1.life_errors(nets, t) for _, t in written])
    assert errors == pytest.approx(hand) and len(errors) == sum(
        len(t) - 1 for _, t in written)
    # (A7) each error's token length
    assert lengths.tolist() == [T2_VOCAB[r[1]][1] for _, t in written
                                for r in t[:-1]]
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
        _, _, late2, no_lengths = e1.t2_pass(e1.iter_own_play(folder, 4),
                                             T2_VOCAB, None, last_iteration=4)
        assert no_lengths.shape == (0,)           # no calibration, no lengths
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
    by = e1.calibration_by_length(errors, lengths)
    assert set(by) == {"4", "8", "16", "32"}
    assert sum(v["decisions"] for v in by.values()) == len(errors)


def test_the_calibration_by_token_length():
    by = e1.calibration_by_length([0.1, 0.2, 0.3, 0.4], [4, 4, 8, 32])
    assert by["4"]["decisions"] == 2 and by["4"]["p50"] == pytest.approx(0.15)
    assert by["8"]["p50"] == pytest.approx(0.3) and by["16"] == {
        "decisions": 0}
    assert by["32"]["decisions"] == 1 and set(by["32"]) == {
        "decisions", "mean", "p10", "p25", "p50", "p75", "p90"}


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
    assert b["argmax"] == int(np.argmax(q))         # the unrounded q's
    assert b["pred_class"]["0.1"] == e1.predicted_class(q, 0.1)
    assert b["chosen_in_pred"] == {k: 6 in v for k, v in
                                   b["pred_class"].items()}
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
    assert "groups" not in sc.recs[-1] and "live" not in sc.recs[-1]
    assert [r["d"] for r in sc.recs] == [0, 1, 2, 3, 4]
    # (A7) live: the census's 36-token partition has more than one class
    assert [r.get("live") for r in sc.recs] == [True] * 4 + [None]
    # (A8) a junction: 3 classes among the nine 4-frame tokens
    assert [r.get("junction") for r in sc.recs] == [True] * 4 + [None]
    sc.truth[(1, 5)] = (3, [0] * 36)                  # a dead decision
    sc.decision(3, traj[0][0], traj[1][0])
    assert sc.recs[-1]["live"] is False and sc.recs[-1]["junction"] is False
    # live but a corridor: two 4-frame classes, the 32-frame ones differ
    sc.truth[(1, 6)] = (3, _part([0, 0, 1, 1, 1, 1, 1, 1, 1], {(3, 32): 7}))
    sc.decision(3, traj[0][0], traj[1][0])
    assert sc.recs[-1]["live"] is True and sc.recs[-1]["junction"] is False
    assert sc.recs[-2]["groups"]["4"]["truth"] == [0] * 9      # the dead one


class _FixedIDM(torch.nn.Module):
    """An IDM whose posterior is `q` whatever it sees."""

    def __init__(self, q):
        super().__init__()
        self.logits = torch.log(torch.tensor(q, dtype=torch.float32))

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1)


def test_a_confidently_wrong_idm():
    """(audit r1 M1) The IDM is sure of an action outside the chosen
    action's class: the chosen action is not in its predicted class
    (chosen_in_pred False, not scored), its one predicted member is a
    false merge, and the true class's other members are all missed."""
    nets = _tiny_nets()
    rng = np.random.default_rng(8)
    traj, _ = _life(T2_VOCAB, [11], rng)
    part = _part([0, 0, 1, 1, 1, 2, 1, 1, 1])
    q = [0.97] + [0.03 / 8] * 8                     # sure of NOOP
    sc = e1.Scorer(_fake_run(_prior(11)), nets, T2_VOCAB, {(1, 0): (11, part)},
                   0.05, _FixedIDM(q), None)
    sc.game = 1
    sc.decision(11, traj[0][0], traj[1][0])
    b = sc.recs[0]["b"]
    assert b["action"] == 6 and b["truth_class"] == [2, 3, 4, 6, 7, 8]
    assert b["argmax"] == 0 and b["pred_class"]["0.1"] == [0]
    assert b["chosen_in_pred"] == {"0.05": False, "0.1": False, "0.2": False}
    r2 = e1.summarize(sc.recs, T2_GROUPS, [], {1: 1700})["read_2"]
    assert (r2["tp"], r2["fp"], r2["fn"]) == (0, 1, 5)
    assert r2["precision"] == 0.0 and r2["recall"] == 0.0
    assert r2["chosen_in_pred"] == 0.0
    assert r2["top1"] == 0.0 and r2["top1_in_class"] == 0.0
    inc = r2["including_chosen"]                    # A4's read: one more miss
    assert (inc["tp"], inc["fp"], inc["fn"]) == (0, 1, 6)


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
def _rec(game, d, four_truth, four_pred, b=None, live=True):
    rec = {"game": game, "d": d, "token": 0, "live": live,
           "junction": e1.is_junction(four_truth), "groups": {
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
    pred = {k: e1.predicted_class(q, r) for k, r in e1.B_RATIOS.items()}
    return {"action": action, "q": q, "argmax": int(np.argmax(q)),
            "truth_class": truth_class, "pred_class": pred,
            "chosen_in_pred": {k: action in v for k, v in pred.items()}}


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
    # the trivial rules, reported (A5): all one = 52 true of 72 merged pairs
    assert r1["trivial"] == {
        "one": {"precision": round(52 / 72, 4), "recall": 1.0,
                "f1": round(104 / 124, 4)},
        "distinct": {"precision": None, "recall": 0.0, "f1": 0.0}}
    assert r1["floor"] == round(104 / 124, 4) and r1["beats_floor"] is False
    assert r1["classes_pred_mean"] == 2.0 and r1["classes_truth_mean"] == 1.5
    # (2) over the members other than the chosen action 4: predicted {5, 6}
    # against the true {5, 6, 7, 8}
    r2 = s["read_2"]
    assert r2["decisions"] == 1 and (r2["tp"], r2["fp"], r2["fn"]) == (2, 0, 2)
    assert r2["precision"] == 1.0 and r2["recall"] == 0.5
    assert r2["f1"] == round(4 / 6, 4)
    assert r2["trivial"] == {                       # all nine; the chosen alone
        "one": {"precision": 0.5, "recall": 1.0, "f1": round(8 / 12, 4)},
        "distinct": {"precision": None, "recall": 0.0, "f1": 0.0}}
    assert r2["floor"] == round(8 / 12, 4) and r2["idm_held_acc"] == 0.5
    assert r2["chosen_in_pred"] == 1.0
    assert r2["top1"] == 1.0 and r2["top1_in_class"] == 1.0
    assert r2["pred_class_mean"] == 3.0 and r2["truth_class_mean"] == 5.0
    inc = r2["including_chosen"]                    # A4's read, reported
    assert (inc["tp"], inc["fp"], inc["fn"]) == (3, 0, 2)
    assert inc["trivial"]["one"]["precision"] == round(5 / 9, 4)
    assert inc["trivial"]["distinct"] == {"precision": 1.0, "recall": 0.2,
                                          "f1": round(2 / 6, 4)}
    # (A5): both pass pooled (recall 0.5 inclusive) but each meets the E2
    # bars in one family of two (1880: (A) recall 16/36; no (B) decision)
    fam = s["read_3"]["families"]
    assert fam["1700"]["meets_e2"] == {"A": True, "B": True}
    assert fam["1880"]["meets_e2"] == {"A": False, "B": False}
    # no junction among them (2 and 1 classes): no junction precision, so
    # the junction bar fails too (A8)
    assert s["decision"] == {
        "A": {"precision": 1.0, "recall": round(32 / 52, 4), "pooled_e2": True,
              "families_e2": 1, "families": 2, "junction_precision": None,
              "junction_bar": False, "pooled_low_bar": True},
        "B": {"precision": 1.0, "recall": 0.5, "pooled_e2": True,
              "families_e2": 1, "families": 2, "junction_precision": None,
              "junction_bar": False, "pooled_low_bar": True}}
    assert s["verdict"] == "KAI_CALL_MID" and s["verdict_on"] == ["A", "B"]
    assert s["verdict_a7"] == "KAI_CALL_MID"       # 1 family of 2 anyway
    assert s["live"]["junction_decisions"] == 0
    assert s["live"]["junction"]["A"] == {"decisions": 0,
                                          "pairs_per_decision": 0}
    assert s["live"]["corridor"]["A"]["tp"] == 32
    assert s["live"]["corridor"]["B"]["decisions"] == 1
    # LOW-1: reads (3) and (4) on the live decisions too
    assert s["live"]["read_3"]["frequency"]["a"]["4"]["rare"]["tp"] == 7
    assert set(s["live"]["read_3"]["families"]) == {"1700", "1880"}
    assert s["live"]["read_4"]["a"]["2"]["4"]["tp"] == 52
    assert s["live"]["read_2"]["including_chosen"]["tp"] == 3
    # (A7): both scored records are live, so the live decision is the
    # all-decision one; the reference merges the 4-frame pairs no truth
    # ever split
    assert s["live"]["decisions"] == 2 and s["live"]["share"] == 1.0
    assert s["decision_all_decisions"] == s["decision"]
    assert s["verdict_all_decisions"] == s["verdict"]
    assert s["live"]["read_1"]["tp"] == 32
    assert s["live"]["families"]["1700"]["meets_e2"] == {"A": True, "B": True}
    ref = s["live"]["reference"]
    assert ref["decisions"] == {"A": 2, "B": 1}
    assert ref["A"]["partitions"] == 21147
    text = json.dumps(s)                    # the summary line is JSON
    assert '"_p"' not in text and '"_r"' not in text        # internals
    # (3): (A)'s frequency split, rare = slot 0 alone (position 0 of the
    # 4-frame group)
    fa = s["read_3"]["frequency"]["a"]["4"]
    assert fa["rare"]["pairs_per_decision"] == 8
    assert (fa["rare"]["tp"], fa["rare"]["fp"], fa["rare"]["fn"]) == (7, 0, 4)
    assert (fa["rest"]["tp"], fa["rest"]["fn"]) == (25, 16)
    assert s["read_3"]["frequency"]["a"]["16"]["rare"] == {
        "decisions": 2, "pairs_per_decision": 0}
    assert s["read_3"]["frequency"]["rare_slots"] == [0]
    # (A6): no member is rare (NOOP); the chosen action 4 is not rare
    fb = s["read_3"]["frequency"]["b"]
    mr = fb["members"]["rare"]
    assert (mr["tp"], mr["fp"], mr["fn"]) == (0, 0, 0)
    assert mr["precision"] is None and mr["recall"] is None
    mo = fb["members"]["rest"]
    assert (mo["tp"], mo["fp"], mo["fn"]) == (2, 0, 2)
    assert fb["chosen"]["rare"] == {"decisions": 0}
    assert fb["chosen"]["rest"]["decisions"] == 1
    # (3): the other groups, the families
    g16 = s["read_3"]["groups"]["16"]
    assert (g16["tp"], g16["fp"], g16["fn"]) == (2, 0, 0) and g16["f1"] == 1.0
    g32 = s["read_3"]["groups"]["32"]
    assert (g32["tp"], g32["fp"], g32["fn"]) == (0, 0, 0) and g32["f1"] == 1.0
    assert g32["precision"] is None and g32["recall"] is None
    assert g32["trivial"]["one"]["precision"] == 0.0
    assert g32["trivial"]["distinct"]["f1"] == 1.0
    assert g32["beats_floor"] is False
    assert set(fam) == {"1700", "1880"} and fam["1700"]["games"] == [1]
    assert fam["1700"]["a"]["precision"] == 1.0 and fam["1880"]["b"] == {
        "decisions": 0}
    # (4): robustness, precision and recall reported
    a4 = s["read_4"]["a"]
    assert set(a4) == {"0.5", "2"}
    assert (a4["0.5"]["4"]["tp"], a4["0.5"]["4"]["fn"]) == (0, 52)
    assert a4["0.5"]["4"]["precision"] is None
    assert (a4["2"]["4"]["tp"], a4["2"]["4"]["fp"]) == (52, 20)
    assert a4["2"]["4"]["precision"] == round(52 / 72, 4)
    b4 = s["read_4"]["b"]
    assert set(b4) == {"0.05", "0.2"}
    assert (b4["0.05"]["tp"], b4["0.05"]["fp"]) == (4, 0)    # {5, 6, 7, 8}
    assert b4["0.05"]["recall"] == 1.0
    assert (b4["0.2"]["tp"], b4["0.2"]["fn"]) == (2, 2)


def test_the_rare_split_with_t2s_rare_set():
    """(A1) with the full logs' rare set: the seven rare 4-frame tokens
    leave one 4-frame pair (UPLEFT, DOWNLEFT: positions 6, 8) outside it.
    (A6): the members {5, 6, 7, 8} split into rare {5, 7} and the rest
    {6, 8}; the chosen action 4 (DOWN) is rare."""
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
    mr, mo = f["b"]["members"]["rare"], f["b"]["members"]["rest"]
    assert (mr["tp"], mr["fp"], mr["fn"]) == (1, 0, 1)    # 5 found, 7 missed
    assert (mo["tp"], mo["fp"], mo["fn"]) == (1, 0, 1)    # 6 found, 8 missed
    # all nine, rare members other than 4: {0, 1, 2, 3, 5, 7}, true 5 and 7
    assert mr["trivial"]["one"]["precision"] == round(2 / 6, 4)
    assert f["b"]["chosen"]["rare"]["decisions"] == 1
    assert f["b"]["chosen"]["rest"] == {"decisions": 0}
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
    assert s["verdict"] == "KAI_CALL_LOW" and s["verdict_on"] == []
    assert s["decision"]["A"]["precision"] is None
    assert s["decision"]["B"]["families"] == 0
    assert s["live"]["decisions"] == 0 and s["live"]["share"] is None
    assert s["live"]["reference"]["A"]["partition"] is None


def test_the_decision_reads_the_live_decisions_only():
    """(A7) A model that merges nothing on a live decision and everything
    on a dead one passes pooled over every decision (P 1.0, R 36/52) but
    has no merge at all on the live ones: LOW. The all-decision reads stay
    reported."""
    split = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    recs = [_rec(1, 0, split, list(range(9))),
            _rec(1, 1, [0] * 9, [0] * 9, live=False),
            {"game": 1, "d": 2, "token": 0, "missing": True}]
    s = e1.summarize(recs, T2_GROUPS, [], {1: 1700})
    dec_all = s["decision_all_decisions"]["A"]
    assert (dec_all["precision"], dec_all["recall"]) == (1.0,
                                                        round(36 / 52, 4))
    assert dec_all["pooled_e2"] is True and dec_all["families_e2"] == 1
    assert s["verdict_all_decisions"] == "KAI_CALL_MID"
    assert s["decision"]["A"]["precision"] is None
    assert s["decision"]["A"]["recall"] == 0.0
    assert s["verdict"] == "KAI_CALL_LOW" and s["verdict_on"] == []
    assert s["live"]["decisions"] == 1 and s["live"]["share"] == 0.5
    assert s["read_1"]["tp"] == 36                  # every decision, reported
    # a record without the flag is not live
    s2 = e1.summarize([{k: v for k, v in recs[0].items() if k != "live"}],
                      T2_GROUPS, [], {1: 1700})
    assert s2["live"]["decisions"] == 0


def test_decide_with_a_family_without_live_decisions():
    """decide(): a family whose games have no live decision has no read and
    fails the bars (it still counts among the families); the junction read
    pools the junction decisions alone: an all-one guess at a junction
    keeps pooled precision >= 0.9 (ten perfect corridors around it) but
    fails A8's junction bar."""
    junction = [0, 1, 2, 0, 0, 0, 0, 0, 0]          # 3 classes; 21 true pairs
    split = [0, 0, 0, 0, 1, 1, 1, 1, 1]             # a corridor; 16 true pairs
    recs = ([_rec(1, 0, junction, [0] * 9)]
            + [_rec(1, d, split, split) for d in range(1, 11)]
            + [_rec(2, 0, [0] * 9, [0] * 9, live=False)])
    live = [r for r in recs if r["live"]]
    fams = e1.family_games({1: 1700, 2: 1880})
    out = e1.decide(live, fams)
    assert out["families"]["1880"]["a"] == {"decisions": 0,
                                            "pairs_per_decision": 0}
    assert out["families"]["1880"]["meets_e2"] == {"A": False, "B": False}
    a = out["block"]["A"]
    assert (a["families"], a["families_e2"]) == (2, 1)
    assert a["precision"] == round((160 + 21) / (160 + 36), 4)
    assert a["recall"] == 1.0 and a["pooled_e2"] is True
    assert a["junction_precision"] == round(21 / 36, 4)
    assert a["junction_bar"] is False
    assert out["junction"]["A"]["decisions"] == 1
    assert out["corridor"]["A"]["decisions"] == 10
    assert out["corridor"]["A"]["precision"] == 1.0
    # one family of two: MID with or without the junction bar here
    assert out["verdict"] == out["verdict_no_junction_bar"] == "KAI_CALL_MID"
    assert out["verdict_on"] == ["A"]


def test_the_set_partitions():
    bell = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147]
    for n, want in enumerate(bell):
        parts = e1.set_partitions(n)
        assert len(parts) == want and len(set(parts)) == want
        for p in parts:      # restricted growth: block ids by first use
            assert all(p[i] <= max(p[:i], default=-1) + 1 for i in range(n))
    parts, same = e1._partition_table()
    assert parts.shape == (21147, 9) and same.shape == (21147, 36)
    assert same[0].sum() == 36 and same[-1].sum() == 0   # one block; none


def test_the_reference_partition_by_hand():
    """Truth: NOOP and LEFT always together, nothing else. The best fixed
    partition merges exactly them, at precision and recall 1; (B) likewise
    for a chosen NOOP."""
    truth = [0, 1, 2, 0, 3, 4, 5, 6, 7]
    q = [0.5] + [0.0625] * 8
    recs = [_rec(1, d, truth, truth, _b(0, q, [0, 3]) if d < 2 else None)
            for d in range(4)]
    ref = e1.reference_partition(recs)
    assert ref["decisions"] == {"A": 4, "B": 2}
    for k in ("A", "B"):
        assert ref[k]["partition"] == truth
        assert ref[k]["blocks"][0] == ["NOOP", "LEFT"]
        assert (ref[k]["precision"], ref[k]["recall"]) == (1.0, 1.0)
        assert ref[k]["meeting_e2_bars"] >= 1
    # nothing truly equivalent: no recall to report
    ref = e1.reference_partition([_rec(1, 0, list(range(9)), list(range(9)))])
    assert ref["A"]["partition"] is None and ref["A"]["meeting_e2_bars"] == 0


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
    summary, dec, meta = e1.out_files("run-x", smoke=False)
    assert summary.name == "probe_e1_equivalence-run-x.jsonl"
    assert dec.name == "probe_e1_equivalence-run-x-decisions.jsonl"
    assert meta.name == "probe_e1_equivalence-run-x-meta.jsonl"
    for p in (dec, meta):
        p.write_text("{}\n")
        with pytest.raises(SystemExit, match="never overwritten"):
            e1.out_files("run-x", smoke=False)
        p.unlink()
    s, d, m = e1.out_files("run-x", smoke=True)
    assert s.name == "probe_e1_equivalence-run-x-smoke.jsonl"
    assert d.name == "probe_e1_equivalence-run-x-smoke-decisions.jsonl"
    assert m.name == "probe_e1_equivalence-run-x-smoke-meta.jsonl"


# --------------------------------------------------------------------------
# (A5) on E0's census truth: no trivial rule can reach E2
# --------------------------------------------------------------------------
CENSUS_RUN = "mspacman-two-t2-s0-2026-10-29"
REAL_SNAP = pathlib.Path(
    r"Y:\SwitchPlayer-testb2-run\benchmarks\results"
    r"\mspacman-two-t2-s0-2026-10-29\snapshots-mspacmannoframeskip-skip4-"
    r"effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman-tokens-"
    r"two-t2-seed0\weights-iter042.pth")


def _census_or_skip():
    summary, dec = e1.census_files(CENSUS_RUN)
    if not (summary.exists() and dec.exists()):
        pytest.skip("E0's census is not in this tree")
    census = e1.census_row(summary, CENSUS_RUN, 42)
    truth, per_game = e1.read_truth(dec)
    return census, truth, per_game


def test_a_dead_aware_rule_passes_the_old_pool_but_not_a7():
    """(A7, audit r2) 'all one class on a dead decision, else {NOOP, LEFT,
    UL, DL} {UP} {RIGHT, UR, DR} {DOWN}' -- one bit of state -- reads E2
    pooled over every decision (P 0.907, R 0.569, 7 families) but only
    0.825 / 0.390 on the live ones: Kai's call (mid) under A7. No
    state-blind partition meets the E2 bars on the live decisions (the
    best (A) recall at P >= 0.9 is 0.333)."""
    census, truth, _ = _census_or_skip()
    families = {int(n): int(round(g[0])) for n, g in census["games"].items()}
    four = [e1.GRID.index(T2_VOCAB[s]) for s in T2_GROUPS["4"]]
    fixed = e1.canonical([0, 1, 2, 0, 3, 2, 0, 2, 0])
    recs = []
    for (game, d), (token, part) in sorted(truth.items()):
        t = e1.restrict(part, four)
        live = len(set(part)) > 1
        pred = fixed if live else [0] * 9
        rec = {"game": game, "d": d, "token": token, "live": live,
               "groups": {"4": {"truth": t,
                                "pred": {k: pred for k in e1.TAU_MULTS}}}}
        if token in T2_GROUPS["4"]:
            a = T2_GROUPS["4"].index(token)
            cls = [x for x in range(9) if pred[x] == pred[a]]
            rec["b"] = {"action": a, "q": [float(x == a) for x in range(9)],
                        "argmax": a,
                        "truth_class": [x for x in range(9) if t[x] == t[a]],
                        "pred_class": {k: cls for k in e1.B_RATIOS}}
        recs.append(rec)
    s = e1.summarize(recs, {"4": T2_GROUPS["4"]}, [0, 1, 2, 3, 6, 8, 12],
                     families)
    da = s["decision_all_decisions"]["A"]
    assert da["precision"] == pytest.approx(0.9071, abs=1e-4)
    assert da["recall"] == pytest.approx(0.5692, abs=1e-4)
    assert da["families_e2"] == 7
    assert s["verdict_all_decisions"] == "E2"
    assert s["verdict_on_all_decisions"] == ["A"]
    dl, bl = s["decision"]["A"], s["decision"]["B"]
    assert dl["precision"] == pytest.approx(0.8252, abs=1e-4)
    assert dl["recall"] == pytest.approx(0.3896, abs=1e-4)
    assert bl["precision"] == pytest.approx(0.8151, abs=1e-4)
    assert bl["recall"] == pytest.approx(0.4983, abs=1e-4)
    assert not dl["pooled_e2"] and not bl["pooled_e2"]
    assert s["verdict"] == "KAI_CALL_MID" and s["verdict_on"] == ["A", "B"]
    live = s["live"]
    assert live["decisions"] == 3798 and live["read_2"]["decisions"] == 411
    assert live["share"] == pytest.approx(3798 / 4636, abs=1e-4)
    ref = live["reference"]
    assert ref["decisions"] == {"A": 3798, "B": 411}
    assert ref["A"]["meeting_e2_bars"] == 0 and ref["B"]["meeting_e2_bars"] == 0
    assert ref["A"]["precision"] >= 0.9
    assert ref["A"]["recall"] == pytest.approx(0.3328, abs=1e-4)
    assert ref["B"]["recall"] is None or ref["B"]["recall"] < e1.E2_R


def test_the_keyed_reference_reads_e2_under_a7_but_not_a8():
    """(A8, audit r3) The heading rule: a lookup keyed on the previous
    token, fitted on the census truth leaving each family out. Under A7's
    bars alone it reads E2 (on (B): P 0.923, R 0.544, 6 families -- audit
    r3's alias sets); its junction precision is 0.72 for (A) and 0.60 for
    (B), so under A8 it reads Kai's call (mid). Its corridors are clean."""
    census, truth, _ = _census_or_skip()
    families = {int(n): int(round(g[0])) for n, g in census["games"].items()}
    k = e1.keyed_reference(truth, families, T2_VOCAB)
    assert k["decisions"] == {"A": 3798, "B": 411}
    assert (k["verdict_a7"], k["verdict_on_a7"]) == ("E2", ["B"])
    assert (k["verdict_a8"], k["verdict_on_a8"]) == ("KAI_CALL_MID",
                                                     ["A", "B"])
    a, b = k["A"], k["B"]
    assert a["precision"] == pytest.approx(0.9022, abs=1e-4)
    assert a["recall"] == pytest.approx(0.5466, abs=1e-4)
    assert a["families_e2"] == 5 and a["junction_bar"] is False
    assert a["junction_precision"] == pytest.approx(0.7221, abs=1e-4)
    assert (b["precision"], b["recall"], b["families_e2"]) == (0.9231, 0.5441,
                                                               6)
    assert b["junction_precision"] == pytest.approx(0.6045, abs=1e-4)
    assert k["junction"]["A"]["decisions"] == 1182
    assert k["corridor"]["A"]["decisions"] == 2616
    assert k["corridor"]["A"]["precision"] > 0.97
    assert k["corridor"]["B"]["precision"] > 0.99
    # joystick logic, shown fitted on every family: after RIGHT-4 (slot 2)
    # the right diagonals go with RIGHT
    assert ["RIGHT", "UPRIGHT", "DOWNRIGHT"] in \
        k["partitions_fitted_on_all"]["2"]
    json.dumps(k)                                   # the summary carries it


def test_the_real_census_eval_row_and_snapshot():
    """The real files: the full census's row, the deployed exact-eval row it
    was checked against, and the census's snapshot is the T2 argument."""
    census, truth, per_game = _census_or_skip()
    if not REAL_SNAP.exists():
        pytest.skip("T2's snapshot is not on this machine")
    assert census["main_line_ok"] is True and census["decisions"] == 4636
    assert census["threads"] == 2 and len(truth) == 4636
    assert e1.same_snapshot(census["snapshot"], REAL_SNAP) is None
    ref = e1.pc.exact_eval_row(REAL_SNAP)
    assert ref is not None and ref["exact_mean"] == 1699.0
    assert len(ref["games"]) == 30
    for n, g in census["games"].items():
        assert e1.pc.same_game(g, ref["games"][n])
        assert per_game[int(n)] == g[3]
    assert e1.preflight(REAL_SNAP, census, truth, ref, False, 2) is None
    with pytest.raises(SystemExit, match="torch threads"):
        e1.preflight(REAL_SNAP, census, truth, ref, False, 4)


# --------------------------------------------------------------------------
# main()'s wiring on fakes (no emulator, no real snapshot)
# --------------------------------------------------------------------------
def _fake_tree(tmp_path):
    """A run tree: the two preps, T0 / T1 / T2 logs (2 / 2 / 3 iterations)
    and a T2 snapshot path at iteration 3."""
    results = tmp_path / "tree" / "benchmarks" / "results"
    t1_vocab = ((0, 4), (0, 16)) + tuple((a, 4) for a in range(1, 9))
    preps = {e1.PREPS[0]: {"master_vocab": [list(t) for t in e1.GRID],
                           "vocab": [list(t) for t in t1_vocab],
                           "log_dir": "benchmarks/results/run-t0/"
                                      "own-play-a-t0-seed0",
                           "log_iterations": 2},
             e1.PREPS[1]: {"master_vocab": [list(t) for t in t1_vocab],
                           "vocab": [list(t) for t in T2_VOCAB],
                           "log_dir": "benchmarks/results/run-t1/"
                                      "own-play-a-t1-seed0",
                           "log_iterations": 2}}
    for name, j in preps.items():
        (results / name).mkdir(parents=True)
        (results / name / "vocab.json").write_text(json.dumps(j))
    logs = {"T0": (_write_log(results / "run-t0" / "own-play-a-t0-seed0",
                              [2, 2], 11, e1.GRID), e1.GRID),
            "T1": (_write_log(results / "run-t1" / "own-play-a-t1-seed0",
                              [2, 2], 12, t1_vocab), t1_vocab),
            "T2": (_write_log(results / "run-t2" / "own-play-a-t2-seed0",
                              [3, 3, 4], 13), T2_VOCAB)}
    snap = results / "run-t2" / "snapshots-a-t2-seed0" / "weights-iter003.pth"
    snap.parent.mkdir(parents=True)
    snap.write_bytes(b"weights")
    return snap, logs


def test_main_end_to_end_on_fakes(tmp_path, monkeypatch):
    """main() with a fake probe_exhaustive_eval, Run, play_token and
    exact-eval row over a fake run tree and census: the meta line before
    the replay (its 4-frame decision and label counts, the keyed
    reference), the decisions (live and junction flags), a valid summary;
    a second full run refuses; a full run whose first game differs from
    the exact eval stops there (MISMATCH -> break) and is VOID."""
    import contextlib
    import types
    snap, logs = _fake_tree(tmp_path)
    probes = tmp_path / "probes"
    probes.mkdir()
    monkeypatch.setattr(e1, "HERE", probes)
    tokens = [11, 9, 14, 5, 11, 3]                 # 4, 16, 4, 32, 4, 4 frames
    parts = [[0] * 36] + [_part([0, 0, 1, 1, 1, 2, 1, 1, 1], {(3, 32): k})
                          for k in range(5)]
    (probes / "probe_action_census-run-t2-decisions.jsonl").write_text(
        "".join(json.dumps({"game": 1, "d": d, "token": t,
                            "partition": p}) + "\n"
                for d, (t, p) in enumerate(zip(tokens, parts))))
    row = {"run": "run-t2", "iteration": 3, "n_games": 30,
           "max_decisions": None, "env": e1.ENV_ID, "main_line_ok": True,
           "decisions": 6, "threads": torch.get_num_threads(),
           "snapshot": str(snap), "games": {"1": [100.0, 40, True, 6]}}
    (probes / "probe_action_census-run-t2.jsonl").write_text(
        json.dumps(row) + "\n")
    ref = {"games": {"1": [100.0, 40, True, 6]}, "exact_mean": 100.0}
    monkeypatch.setattr(e1.pc, "exact_eval_row", lambda p: ref)
    # the fake run: tiny nets; the prior's argmax is the token played
    nets0, pred = _tiny_nets(), _tiny_pred()
    playing = {}
    nm = SimpleNamespace(initial_inference=lambda stack: (
        _prior(playing["token"]), 0.0))

    def set_train_mode(training):
        for net in (nets0.rep, pred, nets0.dyn, nets0.proj):
            net.train(training)
    run = SimpleNamespace(
        trainer=SimpleNamespace(representation=nets0.rep, prediction=pred,
                                dynamics=nets0.dyn, projector=nets0.proj),
        searcher=SimpleNamespace(network_manager=nm),
        set_train_mode=set_train_mode)
    d = {"acting_states": {"representation": _bn_shifted(
        nets0.rep.state_dict(), 0.3)}}

    def play(run_, n):
        rng = np.random.default_rng(n)
        stack = muzero.initial_frame_stack(_frame(rng))
        for t in tokens:
            playing["token"] = t
            out = ablation.play_token(None, t, stack)
            stack = muzero.push_frame(out[4], muzero.preprocess_frame(
                out[0], muzero.IMAGE_RESHAPE))
        return 100.0, 40, True, len(tokens)

    def fake_play_token(env, token, stack, inner=None):
        obs = np.full((210, 160, 3), (13 * token) % 256, dtype=np.uint8)
        return obs, 0.0, False, False, stack, 1

    pe = types.ModuleType("probe_exhaustive_eval")
    pe.SEARCH_SIMS, pe.ACTING, pe.ENV, pe.ARM = 0, True, e1.ENV_ID, "fake-arm"
    pe.snapshot_vocab = lambda p: T2_VOCAB
    pe.snapshot_prefix_bn = lambda p: False
    pe.snapshot_tokens = lambda v: contextlib.nullcontext()
    pe.snapshot_prefix_layout = lambda b: contextlib.nullcontext()
    pe.load = lambda run_, p: d
    pe.apply_acting_states = _apply_acting
    pe.play = play
    monkeypatch.setitem(sys.modules, "probe_exhaustive_eval", pe)
    monkeypatch.setattr(ablation, "arm_config",
                        lambda arm: contextlib.nullcontext())
    monkeypatch.setattr(ablation, "_run_length_globals",
                        lambda i, s: contextlib.nullcontext())
    monkeypatch.setitem(ablation.ARMS, "fake-arm", {})
    monkeypatch.setattr(ablation, "Run", lambda arm, seed, spec: run)
    monkeypatch.setattr(ablation, "play_token", fake_play_token)
    for k, v in (("GAMES", 1), ("MAX_DECISIONS", 0), ("LOG_LIVES", 0),
                 ("IDM_EPOCHS", 1), ("SMOKE", False)):
        monkeypatch.setattr(e1, k, v)
    e1.main([str(snap)])
    assert ablation.play_token is fake_play_token        # the hook let go
    tag = probes / "probe_e1_equivalence-run-t2"
    meta = [json.loads(l) for l in
            pathlib.Path(f"{tag}-meta.jsonl").read_text().splitlines()]
    decs = [json.loads(l) for l in
            pathlib.Path(f"{tag}-decisions.jsonl").read_text().splitlines()]
    summ = [json.loads(l) for l in
            pathlib.Path(f"{tag}.jsonl").read_text().splitlines()]
    assert len(meta) == len(summ) == 1 and len(decs) == 6
    m, s = meta[0], summ[0]
    assert m["tau"] == s["tau"] and m["idm"] == s["idm"]
    assert m["calibration"]["decisions"] == sum(
        v["decisions"] for v in m["calibration"]["by_length"].values())
    assert m["frequency"]["rare_slots"] == s["frequency"]["rare_slots"]
    assert m["idm"]["lives"] == {"T2": 10, "T0": 4, "T1": 4}
    assert m["idm"]["held_lives"] == 1
    # the 4-frame decisions and their labels, from the logs as written
    samples = {k: [y for _, t in written
                   for _, y in e1.life_idm_samples(t, vocab)]
               for k, (written, vocab) in logs.items()}
    assert m["idm"]["four_frame_decisions"] == {k: len(v)
                                                for k, v in samples.items()}
    held = [y for _, t in logs["T2"][0][-1:]
            for _, y in e1.life_idm_samples(t, T2_VOCAB)]
    assert m["idm"]["label_counts_held"] == np.bincount(
        held, minlength=9).tolist()
    assert (np.array(m["idm"]["label_counts_train"])
            + m["idm"]["label_counts_held"]).tolist() == np.bincount(
        sum(samples.values(), []), minlength=9).tolist()
    assert m["idm"]["train"] + m["idm"]["held"] == sum(
        len(v) for v in samples.values())
    assert m["reference_keyed"] == s["live"]["reference_keyed"]
    assert m["reference_keyed"]["decisions"] == {"A": 5, "B": 3}
    assert [x["d"] for x in decs] == list(range(6))
    assert [x["live"] for x in decs] == [False] + [True] * 5
    assert [x["junction"] for x in decs] == [False] + [True] * 5
    assert ["b" in x for x in decs] == [True, False, True, False, True, True]
    assert s["main_line_ok"] is True and s["void_reasons"] == []
    assert s["live"]["decisions"] == 5 and s["read_1"]["decisions"] == 6
    assert s["live"]["junction_decisions"] == 5
    assert s["verdict"] in ("E2", "KAI_CALL_MID", "KAI_CALL_LOW")
    assert s["verdict_a7"] in ("E2", "KAI_CALL_MID", "KAI_CALL_LOW")
    assert s["raw_rep_differs_from_acting"] is True
    assert s["census"]["snapshot_note"] is None and s["meta_file"] == \
        "probe_e1_equivalence-run-t2-meta.jsonl"
    # a second full run refuses before anything plays
    with pytest.raises(SystemExit, match="never overwritten"):
        e1.main([str(snap)])
    # a full two-game run whose first game differs from the exact eval:
    # MISMATCH -> break -> VOID, stopped after one game
    for suffix in ("-meta.jsonl", "-decisions.jsonl", ".jsonl"):
        pathlib.Path(f"{tag}{suffix}").unlink()
    ref["games"]["1"] = [101.0, 40, True, 6]
    monkeypatch.setattr(e1, "GAMES", 2)
    played = []
    monkeypatch.setattr(pe, "play", lambda run_, n: played.append(n)
                        or play(run_, n))
    e1.main([str(snap)])
    assert played == [1]                              # game 2 never played
    v = json.loads(pathlib.Path(f"{tag}.jsonl").read_text())
    assert v["main_line_ok"] is False and v["verdict"].startswith("VOID")
    assert "verdict_if_valid" in v and v["games"] == {"1": [100.0, 40, True,
                                                            6]}
    assert v["void_reasons"][0] == "stopped after 1 of 2 games"
    assert v["void_reasons"][1].startswith("game 1 ")


def test_no_trivial_rule_reaches_e2_on_the_census_truth():
    """All one class has merge precision ~0.61 on (A), all nine ~0.68 on
    (B), all distinct recall 0 (precision None): every combination reads
    KAI_CALL_LOW. A perfect model reaches E2 (the bar is reachable)."""
    summary, dec = e1.census_files(CENSUS_RUN)
    if not (summary.exists() and dec.exists()):
        pytest.skip("E0's census is not in this tree")
    census = e1.census_row(summary, CENSUS_RUN, 42)
    truth, _ = e1.read_truth(dec)
    by_fam = {}
    for n, g in census["games"].items():
        by_fam.setdefault(int(round(g[0])), []).append(int(n))
    assert len(by_fam) == 8
    four = [e1.GRID.index(T2_VOCAB[s]) for s in T2_GROUPS["4"]]
    recs = []
    for (game, d), (token, part) in sorted(truth.items()):
        t = e1.restrict(part, four)
        rec = {"game": game, "d": d, "token": token,
               "groups": {"4": {"truth": t, "pred": {"1": t}}}}
        if token in T2_GROUPS["4"]:
            a = T2_GROUPS["4"].index(token)
            cls = [x for x in range(9) if t[x] == t[a]]
            rec["b"] = {"action": a, "q": [1 / 9] * 9, "argmax": a,
                        "truth_class": cls, "pred_class": {"0.1": cls}}
        recs.append(rec)
    a, b = e1.group_read(recs, "4"), e1.b_read(recs)
    assert a["decisions"] == 4636 and b["decisions"] == 506
    ta, tb = a["trivial"], b["trivial"]
    assert ta["one"]["precision"] == pytest.approx(0.61, abs=0.01)
    assert tb["one"]["precision"] == pytest.approx(0.68, abs=0.01)
    assert ta["one"]["recall"] == tb["one"]["recall"] == 1.0
    for t in (ta, tb):
        assert t["distinct"] == {"precision": None, "recall": 0.0, "f1": 0.0}

    def families_meeting(read, rule):
        n = 0
        for games in by_fam.values():
            r = read([x for x in recs if x["game"] in games])
            if "trivial" in r:
                tr = r["trivial"] if rule is None else r["trivial"][rule]
                n += int(e1.meets(tr["precision"], tr["recall"], e1.E2_P,
                                  e1.E2_R))
        return n

    for ra in ("one", "distinct"):
        for rb in ("one", "distinct"):
            # junction precision 1.0: the most lenient (A8 aside)
            cands = {"A": (ta[ra]["precision"], ta[ra]["recall"],
                           families_meeting(lambda s: e1.group_read(s, "4"),
                                            ra), 1.0),
                     "B": (tb[rb]["precision"], tb[rb]["recall"],
                           families_meeting(e1.b_read, rb), 1.0)}
            assert e1.verdict(cands) == ("KAI_CALL_LOW", []), (ra, rb)
    # the perfect predictions given above: E2, in all 8 families
    fam_a = sum(int(e1.meets(r["precision"], r["recall"], e1.E2_P, e1.E2_R))
                for r in (e1.group_read([x for x in recs if x["game"] in g],
                                        "4") for g in by_fam.values()))
    assert (a["precision"], a["recall"], fam_a) == (1.0, 1.0, 8)
    assert e1.verdict({"A": (1.0, 1.0, fam_a, 1.0)}) == ("E2", ["A"])


# --------------------------------------------------------------------------
# audit r1: (A)'s nets, the refusals, A4's fixed IDM, tau's guard
# --------------------------------------------------------------------------
def _tiny_pred(seed=1):
    torch.manual_seed(seed)
    pred = networks.PredictionNetwork(latent_channels=16, action_space=18,
                                      head_channels=4)
    torch.nn.init.normal_(pred.policy_fc.weight, std=0.5)
    torch.nn.init.normal_(pred.value_fc.weight, std=0.5)
    return pred.eval()


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


def test_as_nets_use_the_raw_bn_stats_while_the_main_line_acts():
    """(audit r1 L4) nets_for_a: (A)'s representation keeps the trainer's
    raw running statistics; the live representation (shared with the
    network manager, as Run shares it) gets the acting ones, and
    initial_inference reads those."""
    nets0 = _tiny_nets()
    rep, pred = nets0.rep, _tiny_pred()
    nm = object.__new__(muzero.NetworkManager)
    nm.device, nm.representation, nm.prediction = torch.device("cpu"), rep, \
        pred
    trainer = SimpleNamespace(representation=rep, prediction=pred,
                              dynamics=nets0.dyn, projector=nets0.proj)

    def set_train_mode(training):
        for net in (rep, pred, nets0.dyn, nets0.proj):
            net.train(training)
    run = SimpleNamespace(trainer=trainer, set_train_mode=set_train_mode,
                          searcher=SimpleNamespace(network_manager=nm))
    raw = copy.deepcopy(rep.state_dict())
    acting = {"representation": _bn_shifted(raw, 0.3),
              "prediction": _bn_shifted(pred.state_dict(), 0.3)}
    nets, differs = e1.nets_for_a(run, {"acting_states": acting},
                                  _apply_acting)
    assert differs and nets.rep is not rep
    assert nets.dyn is nets0.dyn and nets.proj is nets0.proj
    assert not any(n.training for n in nets)
    for k, v in nets.rep.state_dict().items():
        assert torch.equal(v, raw[k])                       # raw stats
    for k, v in rep.state_dict().items():
        assert torch.equal(v, acting["representation"][k])  # acting stats
    rng = np.random.default_rng(9)
    stack = muzero.initial_frame_stack(_frame(rng))
    x = torch.tensor(stack[None], dtype=torch.float32) / 255.0
    a_rep = copy.deepcopy(rep)
    r_rep = copy.deepcopy(rep)
    r_rep.load_state_dict(raw)
    with torch.no_grad():
        policy, _ = nm.initial_inference(stack)
        want = torch.softmax(pred(a_rep(x))[0], dim=1)[0].numpy()
        assert np.allclose(policy, want, atol=1e-6)
        assert torch.allclose(nets.rep(x), r_rep(x), atol=1e-6)
        assert not torch.allclose(nets.rep(x), a_rep(x), atol=1e-4)
    with pytest.raises(SystemExit, match="acting states for"):
        e1.nets_for_a(run, {"acting_states": {"dynamics": {}}},
                      _apply_acting)


def _census(snapshot, threads=2, decisions=3):
    return {"snapshot": str(snapshot), "threads": threads,
            "decisions": decisions}


def test_the_preflight_refusals(tmp_path):
    snap = tmp_path / "run" / "snaps" / "weights-iter042.pth"
    snap.parent.mkdir(parents=True)
    snap.write_bytes(b"weights")
    truth = {(1, 0): None, (1, 1): None, (2, 0): None}
    ref = {"games": {}}
    ok = e1.preflight(snap, _census(snap), truth, ref, False, 2)
    assert ok is None
    # the same file by another spelling
    alias = tmp_path / "run" / "snaps" / ".." / "snaps" / snap.name
    assert e1.preflight(snap, _census(alias), truth, ref, False, 2) is None
    with pytest.raises(SystemExit, match="no full census"):
        e1.preflight(snap, None, truth, ref, False, 2)
    with pytest.raises(SystemExit, match="has 3 decisions, its summary 4"):
        e1.preflight(snap, _census(snap, decisions=4), truth, ref, False, 2)
    # L2: another thread count than the census's
    with pytest.raises(SystemExit, match="4 torch threads; the census ran 2"):
        e1.preflight(snap, _census(snap), truth, ref, False, 4)
    with pytest.raises(SystemExit, match="torch threads"):
        e1.preflight(snap, {"snapshot": str(snap), "decisions": 3}, truth,
                     ref, False, 2)
    # a full run needs the exact eval's row; a smoke does not
    with pytest.raises(SystemExit, match="exact-eval row"):
        e1.preflight(snap, _census(snap), truth, None, False, 2)
    assert e1.preflight(snap, _census(snap), truth, None, True, 2) is None


def test_the_census_snapshot_check(tmp_path):
    """(audit r1 L3) another path: accepted (and recorded) only when the
    two files hash alike."""
    snap = tmp_path / "a" / "weights-iter042.pth"
    snap.parent.mkdir()
    snap.write_bytes(b"the weights")
    assert e1.same_snapshot(str(snap), snap) is None
    copy_ = tmp_path / "b" / "weights-iter042.pth"
    copy_.parent.mkdir()
    copy_.write_bytes(b"the weights")
    note = e1.same_snapshot(str(copy_), snap)
    import hashlib
    assert note == {"census_snapshot": str(copy_),
                    "sha256": hashlib.sha256(b"the weights").hexdigest()}
    copy_.write_bytes(b"other weights")
    with pytest.raises(SystemExit, match="sha256"):
        e1.same_snapshot(str(copy_), snap)
    with pytest.raises(SystemExit, match="cannot be read"):
        e1.same_snapshot(str(tmp_path / "gone.pth"), snap)
    with pytest.raises(SystemExit, match="names no snapshot"):
        e1.same_snapshot(None, snap)
    # preflight passes the note on
    copy_.write_bytes(b"the weights")
    assert e1.preflight(snap, {"snapshot": str(copy_), "threads": 2,
                               "decisions": 0}, {}, {}, False, 2) == note


def test_the_knob_refusals():
    e1.check_knobs(30, 0, 0, 10, 20000)
    e1.check_knobs(1, 60, 12, 1, 1)
    for bad in ((0, 0, 0, 10, 1), (31, 0, 0, 10, 1), (30, -1, 0, 10, 1),
                (30, 0, -1, 10, 1), (30, 0, 0, 0, 1), (30, 0, 0, 10, 0)):
        with pytest.raises(SystemExit, match="E1_GAMES"):
            e1.check_knobs(*bad)


def test_tau_refuses_an_empty_log():
    with pytest.raises(SystemExit, match="calibrate tau"):
        e1.calibrate_tau(np.zeros(0))
    assert e1.calibrate_tau(np.array([0.3, 0.1, 0.2])) == pytest.approx(0.2)


def test_a4s_fixed_idm_settings(monkeypatch):
    """(A4) Nature-DQN convolutions, fc 256 -> 9, Adam lr 3e-4, batch 64,
    10 epochs, seed 0, cap 20000, plain cross-entropy."""
    assert (e1.IDM_LR, e1.IDM_BATCH, e1.IDM_EPOCHS_DEFAULT, e1.IDM_SEED,
            e1.IDM_CAP_DEFAULT) == (3e-4, 64, 10, 0, 20000)
    sig = inspect.signature(e1.train_idm).parameters
    assert (sig["seed"].default, sig["batch"].default, sig["lr"].default) == \
        (0, 64, 3e-4)
    net = e1.IDM()
    convs = [m for m in net.conv if isinstance(m, torch.nn.Conv2d)]
    assert [(c.in_channels, c.out_channels, c.kernel_size, c.stride)
            for c in convs] == [(9, 32, (8, 8), (4, 4)),
                                (32, 64, (4, 4), (2, 2)),
                                (64, 64, (3, 3), (1, 1))]
    lins = [m for m in net.fc if isinstance(m, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in lins] == [(3136, 256),
                                                              (256, 9)]
    assert sum(isinstance(m, torch.nn.ReLU) for m in net.modules()) == 4
    assert not any(isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout))
                   for m in net.modules())
    calls = []
    orig = e1.F.cross_entropy

    def spy(*a, **k):
        calls.append(k)
        return orig(*a, **k)
    monkeypatch.setattr(e1.F, "cross_entropy", spy)
    rng = np.random.default_rng(10)
    x = rng.integers(0, 256, size=(4, 9, 84, 84), dtype=np.uint8)
    e1.train_idm(x, np.arange(4), 1)
    assert calls == [{}]             # one batch, no weights / smoothing
    opt_types = []
    real_adam = torch.optim.Adam

    def adam(params, **k):
        opt_types.append(k)
        return real_adam(params, **k)
    monkeypatch.setattr(torch.optim, "Adam", adam)
    e1.train_idm(x, np.arange(4), 1)
    assert opt_types == [{"lr": 3e-4}]
