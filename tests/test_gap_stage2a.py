"""Test G, stage 2a: the real gap on every kept state and the gates that need
only branches (CURRICULUM "Next experiment (pre-registered) -- Test G").
Pins: the registered numbers and the script's environment (the pilot's); the
per-branch returns on every head by hand (the clip per base step at T0's
ceiling 1 in units of 10, per-frame discounts); G's sign (T2 minus T0) and that
it is a mean over the 16 delays; the 16 delays, in order, for both players; on
the real game with stand-in policies, the branch pass against the pilot's, the
cap reaching every branch, a life lost in the delay and in the first token, a
state's branches independent of the states before it, one policy giving no gap;
the clusters by hand (chains, merged states); the bootstrap's unit (clusters,
each with all its states) and seed; G3's two conditions on the primary head;
G5 on the gap with Spearman-Brown by hand and its inclusive bar; the lag
profile; every VOID route; the stamps (knobs, source, the pilot's stamp, the
states consumed); resume after a crashed pass; the summary never overwritten;
the pilot's files read and never written. A small stage on the real game with
stand-in players runs in-process through the orchestrator, G4 also in a fresh
process. Slow, needing the snapshots: the pilot's own branches replayed
exactly, and a whole small stage on T0 and T2 in subprocesses."""
import contextlib
import hashlib
import json
import os
import pathlib
import pickle
import shutil
import stat
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
import probe_gap_check as gc  # noqa: E402
import probe_gap_stage2a as s2  # noqa: E402

VOCAB = ((0, 4), (1, 8), (2, 16), (3, 32), (4, 4), (2, 4), (3, 8))
SMALL_CAP = 192          # frames: short branches keep the real-game tests quick
SMALL_CHUNK = 3


def policy(salt):
    """A deterministic stand-in for a player's prior: each stack picks a
    token; another salt is another player."""
    def inference(stack):
        h = int.from_bytes(hashlib.blake2b(salt + stack.tobytes(),
                                           digest_size=4).digest(), "little")
        p = np.zeros(len(muzero.TOKEN_VOCAB))
        p[h % len(p)] = 1.0
        return p, 0.0
    return inference


INFERENCE = {"T0": policy(b""), "T2": policy(b"master")}


def decide_of(name):
    return lambda stack: int(INFERENCE[name](stack)[0].argmax())


@contextlib.contextmanager
def stand_in(name):
    env = gc.raw_env()
    try:
        yield SimpleNamespace(env=env, searcher=SimpleNamespace(
            network_manager=SimpleNamespace(initial_inference=INFERENCE[name])))
    finally:
        env.close()


def small(mp):
    """The stand-in players and the small knobs the real-game passes run
    under."""
    mp.setattr(muzero, "TOKEN_VOCAB", VOCAB)
    mp.setattr(muzero, "USE_SEARCH_ACTING", False)
    mp.setattr(gc, "player", stand_in)
    mp.setattr(s2, "FRAME_CAP", SMALL_CAP)
    mp.setattr(s2, "CHUNK", SMALL_CHUNK)


def listing(d):
    """Everything a write could change in a directory."""
    out = {}
    for p in sorted(pathlib.Path(d).iterdir()):
        st = p.stat()
        out[p.name] = (st.st_size, st.st_mtime_ns, st.st_mode,
                       hashlib.blake2b(p.read_bytes(), digest_size=16).digest())
    return out


def read_only(d, on=True):
    for p in pathlib.Path(d).iterdir():
        os.chmod(p, stat.S_IREAD if on else stat.S_IREAD | stat.S_IWRITE)


# --------------------------------------------------------------------------
# the module and its numbers
# --------------------------------------------------------------------------
def test_importing_the_stage_changes_nothing_process_wide():
    code = ("import os, sys; sys.path[:0] = [{p!r}, {e!r}, {r!r}]; "
            "import torch; threads = torch.get_num_threads(); "
            "import ablation; before = (ablation.ENV_NAME, ablation.ACTIONS); "
            "import probe_gap_stage2a; "
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
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_the_script_runs_in_the_pilots_environment():
    """The settings probe_gap_check makes when it runs as a script, which
    every pilot pass ran under, are this stage's."""
    src = (PROBES / "probe_gap_check.py").read_text()
    block = src.split('if __name__ == "__main__":', 1)[1].split("import ", 1)[0]
    want = {}
    for line in block.splitlines():
        line = line.split("#")[0].strip()
        if line.startswith("os.environ["):
            key = line.split('"')[1]
            want[key] = gc.ENV_ID if "ENV_ID" in line else line.split('"')[3]
    assert want == s2.ENV
    assert s2.ENV["PROBE_ENV"] == gc.ENV_ID == "MsPacmanNoFrameskip-v4"


def test_the_registered_numbers():
    assert s2.DELAYS == tuple(range(16)) and s2.DELAYS is gc.DELAYS
    assert s2.FRAME_CAP == gc.FRAME_CAP == 1536
    assert s2.PLAYERS == ("T0", "T2")
    assert gc.PLAYERS["T0"]["iteration"] == 45
    assert gc.PLAYERS["T2"]["iteration"] == 42
    assert s2.PRIMARY == (0, 0)
    assert gc.SCALES[s2.PRIMARY[0]] == "clip"
    assert gc.DISCOUNTS[s2.PRIMARY[1]] == 0.997
    assert gc.DISCOUNTS == (0.997, 0.99, 0.98, 0.95) and gc.BASE == 4
    assert (gc.CLIP_UNIT, gc.CLIP_CEILING) == (10.0, 1.0)
    assert s2.G5_BAR == 0.5 and s2.DRAWS == 10_000
    assert s2.INTERVAL == (2.5, 97.5)
    assert s2.BOOT_SEED == (20261107, 4)
    assert s2.LAG_PROFILE == tuple(range(1, 16))
    assert s2.CHUNK == 100 and s2.LIMIT == 0 and not s2.SMOKE
    assert s2.PILOT == PROBES / "probe_gap_check-pilot"


def test_the_source_hash_covers_this_stage_and_the_nets(tmp_path):
    names = {p.name for p in s2.SOURCES}
    assert names == {"probe_gap_stage2a.py", "networks.py",
                     "scalar_categorical.py"}
    assert all(p.is_file() for p in s2.SOURCES)
    assert s2.SOURCE == gc.source_hash(s2.SOURCES)
    st = s2.stamp([])
    assert st["source"] == s2.SOURCE and st["pilot"] == gc.stamp()
    assert st["knobs"] == {"chunk": 100, "states": 0}


# --------------------------------------------------------------------------
# returns and G
# --------------------------------------------------------------------------
def row_of(rewards_by_delay):
    return [{"delay": d, "rewards": list(r), "ended": "life"}
            for d, r in zip(s2.DELAYS, rewards_by_delay)]


def test_branch_returns_on_every_head_by_hand():
    rewards = [[20.0, 0.0, 50.0, 10.0]] + [[5.0]] + [[10.0, 10.0]] \
        + [[0.0]] * 13
    out = s2.branch_returns([row_of(rewards)])
    assert out.shape == (1, 16, 2, 4)            # state, delay, scale, discount
    for j, g in enumerate((0.997, 0.99, 0.98, 0.95)):
        w = [1.0, g ** 4, g ** 8, g ** 12]       # per frame, 4 frames a step
        # the clip: 20 and 50 reach the ceiling of one unit of 10
        assert out[0, 0, 0, j] == pytest.approx(
            sum(a * b for a, b in zip([1, 0, 1, 1], w)))
        assert out[0, 0, 1, j] == pytest.approx(20 + 50 * g ** 8 + 10 * g ** 12)
        assert out[0, 1, 0, j] == pytest.approx(0.5)   # 5 points: half a unit
        # per base step, not per branch: two steps of 10 are two units
        assert out[0, 2, 0, j] == pytest.approx(1 + g ** 4)
        assert out[0, 2, 1, j] == pytest.approx(10 + 10 * g ** 4)
    assert not out[0, 3:].any()


def test_g_is_t2s_mean_minus_t0s():
    rng = np.random.default_rng(1)
    t0 = rng.normal(size=(5, 16, 2, 4))
    t2 = t0 + 3.0
    t2[:, 7] += 16.0                              # one delay moves the mean by 1
    gap = s2.real_gap(t0, t2)
    assert gap.shape == (5, 2, 4)
    assert gap == pytest.approx(np.full((5, 2, 4), 4.0))
    assert s2.real_gap(t2, t0) == pytest.approx(-gap)
    with pytest.raises(SystemExit):
        s2.real_gap(t0, t2[:, :15])


def test_rows_hold_the_16_delays_in_order():
    rows = [row_of([[0.0]] * 16) for _ in range(3)]
    s2.check_rows(rows, 3, "T0")
    with pytest.raises(SystemExit, match="for 3 states, 4 kept"):
        s2.check_rows(rows, 4, "T0")
    short = [r[:15] for r in rows]
    with pytest.raises(SystemExit, match="delays"):
        s2.check_rows(short, 3, "T2")
    swapped = [list(r) for r in rows]
    swapped[1][3], swapped[1][4] = swapped[1][4], swapped[1][3]
    with pytest.raises(SystemExit, match="T2 state 1"):
        s2.check_rows(swapped, 3, "T2")
    shifted = [[{**b, "delay": b["delay"] + 1} for b in r] for r in rows]
    with pytest.raises(SystemExit):
        s2.check_rows(shifted, 3, "T0")


# --------------------------------------------------------------------------
# the real game, stand-in policies, no snapshot
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


def live_state(env, frames=200, seed=0):
    stack, *_ = gc.start_game(env, gc.game_noops(seed),
                              gc.opening_moves(seed, 80))
    ale = env.unwrapped.ale
    stack, _, lost = gc.raw_frames(ale, env.unwrapped._action_set[2], frames,
                                   stack)
    assert not lost
    return {"state": ale.cloneState().serialize(), "stack": stack}


def death_states(env):
    """Standing still from the start: a state one frame before the first life
    goes, and one just after it."""
    u = env.unwrapped
    ale = u.ale
    u.reset(seed=gc.EVAL_SEED)
    u.reset()
    before = None
    while ale.lives() == 3:
        before = ale.cloneState()
        ale.act(u._action_set[0])
    stack = ablation.initial_frame_stack(np.zeros((3, 84, 84), dtype=np.uint8))
    return ({"state": before.serialize(), "stack": stack},
            {"state": ale.cloneState().serialize(), "stack": stack})


def test_play_state_is_the_pilots_branch_pass(env, vocab, monkeypatch):
    """The pilot's stage_branches, one state: the 16 delays through
    gc.branch, the checked one replayed raw."""
    monkeypatch.setattr(s2, "FRAME_CAP", SMALL_CAP)
    s = live_state(env)
    decide = decide_of("T0")
    row, reason = s2.play_state(env, decide, s, 6)
    assert reason is None
    saved = ale_py.ALEState(s["state"])
    want = [gc.branch(env, decide, saved, s["stack"], d, cap=SMALL_CAP)
            for d in gc.DELAYS]
    assert [b["delay"] for b in row] == list(gc.DELAYS)
    assert all(gc.same_branch(a, b) == [] for a, b in zip(row, want))
    assert [k for k, b in enumerate(row) if "replay" in b] == [6]
    frames, end = gc.frames_of(env, saved, want[6])
    assert row[6]["replay"] == frames and end == row[6]["end"]


def test_the_cap_reaches_every_branch(env, vocab, monkeypatch):
    """The stage's FRAME_CAP is the one every branch is played under: 1536 by
    default, and a smaller one cuts every branch there."""
    caps, real = [], gc.branch

    def spy(env, decide, state, stack, delay, cap=gc.FRAME_CAP):
        caps.append(cap)
        return real(env, decide, state, stack, delay, cap=min(cap, 64))

    s = live_state(env)
    monkeypatch.setattr(gc, "branch", spy)
    s2.play_state(env, decide_of("T2"), s, 0)
    assert caps == [1536] * 16
    monkeypatch.setattr(gc, "branch", real)
    monkeypatch.setattr(s2, "FRAME_CAP", 64)
    row, _ = s2.play_state(env, decide_of("T2"), s, 0)
    for b in row:
        budget = (64 - b["delay"]) // gc.BASE
        if b["ended"] == "cap":
            assert len(b["rewards"]) == budget
        else:
            assert len(b["rewards"]) <= budget
    assert {b["ended"] for b in row} == {"cap"}


def test_a_life_lost_ends_the_branch(env, vocab, monkeypatch):
    """One frame before a life goes: a delay of 1+ ends in the delay, delay 0
    in the first token; the return is that one base step's."""
    before, _ = death_states(env)
    row, reason = s2.play_state(env, lambda stack: 0, before, 3)   # NOOP
    assert reason is None
    assert row[0]["ended"] == "life" and len(row[0]["rewards"]) == 1
    assert all(b["ended"] == "delay" and b["frames"] == 1 and b["tokens"] == []
               for b in row[1:])
    ret = s2.branch_returns([row])
    for d, b in enumerate(row):
        r = b["rewards"][0]
        assert ret[0, d, 0, 0] == pytest.approx(min(r / 10, 1.0))
        assert ret[0, d, 1, 3] == pytest.approx(r)


def test_a_states_branches_do_not_depend_on_the_states_before(
        env, vocab, monkeypatch):
    """A later life's state, then an earlier life's: the same rows as alone
    (the stale-lives hazard, through this stage's loop)."""
    monkeypatch.setattr(s2, "FRAME_CAP", SMALL_CAP)
    before, after = death_states(env)
    decide = decide_of("T2")
    alone, _ = s2.play_state(env, decide, before, 0)
    s2.play_state(env, decide, after, 5)
    again, _ = s2.play_state(env, decide, before, 0)
    assert all(gc.same_branch(a, b) == [] for a, b in zip(alone, again))
    late, _ = s2.play_state(env, decide, after, 5)
    assert len(late[0]["rewards"]) > 1        # a stale count reads a death


def test_one_policy_on_both_sides_is_no_gap(env, vocab):
    """Both players' rows from one state under one policy: G is 0 on every
    head, so the delays line up; another policy is a gap."""
    s = live_state(env, frames=200, seed=3)
    a, _ = s2.play_state(env, decide_of("T0"), s, 0)
    b, _ = s2.play_state(env, decide_of("T0"), s, 9)
    gap = s2.real_gap(s2.branch_returns([a]), s2.branch_returns([b]))
    assert not gap.any()
    c, _ = s2.play_state(env, decide_of("T2"), s, 0)
    assert s2.real_gap(s2.branch_returns([a]), s2.branch_returns([c])).any()


# --------------------------------------------------------------------------
# clusters and the bootstrap
# --------------------------------------------------------------------------
def games_of(digests):
    return [{"game": g, "digests": d} for g, d in enumerate(digests)]


def test_clusters_by_hand():
    games = games_of([["a", "b"], ["c"], ["d", "e"], ["e", "b"], ["f"],
                      ["c", "g"], ["h"]])
    assert s2.game_clusters(games) == {0: 0, 1: 1, 2: 0, 3: 0, 4: 4, 5: 1,
                                       6: 6}


def test_a_chain_joins_whatever_the_order():
    """0-1 through x, 2-3 through y, then 1-3 through z: one cluster, met in
    an order where the joining pair's roots are not the games themselves."""
    games = games_of([["x"], ["x", "z"], ["y"], ["y", "z"], ["w"]])
    assert s2.game_clusters(games) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 4}
    rev = games_of([["w"], ["y", "z"], ["y"], ["x", "z"], ["x"]])
    assert s2.game_clusters(rev) == {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}


def test_a_repeat_inside_one_game_joins_nothing():
    assert s2.game_clusters(games_of([["a", "a"], ["b"]])) == {0: 0, 1: 1}


def test_states_take_their_games_cluster():
    games = games_of([["a"], ["b"], ["a"], ["c"]])
    states = [{"game": 0, "decision": 3, "merged": [2]},
              {"game": 1, "decision": 0, "merged": []},
              {"game": 3, "decision": 1, "merged": []}]
    assert s2.state_clusters(states, games) == [0, 1, 3]
    states[1]["merged"] = [3]                 # merged, yet no shared stack
    with pytest.raises(SystemExit, match="not one cluster"):
        s2.state_clusters(states, games)


def test_the_bootstrap_resamples_whole_clusters():
    """A cluster of 100 states at +1 and one of a single state at -1. Each
    draw takes two clusters with replacement and pools their states, so the
    mean is 1, 99/101 or -1, a quarter of draws -1: the lower bound is -1.
    Resampling states, it would sit near +1."""
    values = np.array([1.0] * 100 + [-1.0])
    clusters = [7] * 100 + [3]
    draws = list(s2.cluster_draws(clusters, 4000))
    means = np.array([values[idx].mean() for idx in draws])
    seen = {round(m, 9) for m in means}
    assert seen == {1.0, round(99 / 101, 9), -1.0}
    assert np.mean(means == -1.0) == pytest.approx(0.25, abs=0.03)
    assert np.mean(means == 1.0) == pytest.approx(0.25, abs=0.03)
    assert all(len(idx) in (2, 101, 200) for idx in draws)
    lo, hi = s2.cluster_interval(lambda idx: values[idx].mean(), clusters)
    assert lo == -1.0 and hi == 1.0


def test_each_draw_takes_as_many_clusters_as_there_are():
    clusters = [0, 0, 1, 2, 2, 2]
    sizes = {0: 2, 1: 1, 2: 3}
    rng = np.random.default_rng((20261107, 4))
    for idx in s2.cluster_draws(clusters, 50):
        pick = rng.integers(0, 3, size=3)
        assert len(idx) == sum(sizes[int(c)] for c in pick)
        want = np.concatenate([np.flatnonzero(np.array(clusters) == c)
                               for c in pick])
        assert np.array_equal(idx, want)


def test_the_bootstrap_is_seeded():
    clusters = list(range(30))
    a = [idx.tolist() for idx in s2.cluster_draws(clusters, 5)]
    assert a == [idx.tolist() for idx in s2.cluster_draws(clusters, 5)]
    assert a != [idx.tolist() for idx in s2.cluster_draws(clusters, 5,
                                                          seed=(1, 4))]


def test_the_interval_is_two_sided_95_percent():
    values = np.arange(1000.0)
    clusters = list(range(1000))
    lo, hi = s2.cluster_interval(lambda idx: values[idx].mean(), clusters,
                                 draws=2000)
    sims = np.array([values[idx].mean()
                     for idx in s2.cluster_draws(clusters, 2000)])
    assert (lo, hi) == pytest.approx(tuple(np.percentile(sims, [2.5, 97.5])))
    assert lo < values.mean() < hi


# --------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------
def full(primary, other=0.0):
    """(states, delays) primary returns placed at the primary head, every
    other head `other`."""
    p = np.asarray(primary, dtype=np.float64)
    out = np.full(p.shape + (2, 4), other, dtype=np.float64)
    out[:, :, 0, 0] = p
    return out


KEYS = [(0, k) for k in range(4) for _ in range(3)]
CLUSTERS = list(range(12))
NO_G4 = {"T0": [], "T2": []}


def gap_world(level, noise=0.1, seed=2):
    rng = np.random.default_rng(seed)
    t0 = rng.normal(0, noise, size=(12, 16))
    t2 = np.asarray(level, dtype=np.float64).reshape(12, 1) \
        + rng.normal(0, noise, size=(12, 16))
    return t0, t2


LEVEL = [1, 2, 3, 3, 1, 2, 2, 3, 1, 1.5, 2.5, 3.5]


def test_pass_so_far_with_every_gate(capsys):
    t0, t2 = gap_world(LEVEL)
    rec = s2.read_gates(full(t0), full(t2), KEYS, CLUSTERS, NO_G4, draws=500)
    assert rec["verdict"] == "PASS-SO-FAR" and rec["reasons"] == []
    assert rec["G3"]["ok"] and rec["G3"]["mean"] == pytest.approx(
        (t2.mean(1) - t0.mean(1)).mean())
    assert rec["G3"]["clusters"] == 12 and rec["G3"]["draws"] == 500
    assert rec["G5"]["ok"] and rec["G5"]["reliability"] > 0.9
    assert rec["G4"] == {"T0": {"ok": True, "reasons": []},
                         "T2": {"ok": True, "reasons": []}}


def test_g3_reads_the_primary_head():
    t0, t2 = gap_world(LEVEL)
    rec = s2.read_gates(full(t2, other=-5), full(t0, other=5), KEYS, CLUSTERS,
                        NO_G4, draws=500)
    assert rec["verdict"] == "VOID" and rec["reasons"][0].startswith("G3")
    assert rec["G3"]["mean"] < 0
    rec = s2.read_gates(full(t0, other=5), full(t2, other=-5), KEYS, CLUSTERS,
                        NO_G4, draws=500)
    assert rec["G3"]["ok"]


def test_g3_needs_its_lower_bound_above_zero():
    """A positive mean carried by most clusters, but one large cluster far
    below: the mean is above 0 and its interval reaches below."""
    t0 = np.zeros((12, 16))
    t2 = np.full((12, 16), 0.5)
    t2[:6] = -0.4                              # six states, one cluster
    clusters = [0] * 6 + list(range(1, 7))
    rec = s2.read_gates(full(t0), full(t2), KEYS, clusters, NO_G4, draws=2000)
    assert rec["G3"]["mean"] == pytest.approx(0.05)
    assert rec["G3"]["interval"][0] <= 0 and not rec["G3"]["ok"]
    assert any(r.startswith("G3") for r in rec["reasons"])


def test_g3_needs_its_mean_above_zero():
    t0 = np.zeros((12, 16))
    rec = s2.read_gates(full(t0), full(t0), KEYS, CLUSTERS, NO_G4, draws=200)
    assert rec["G3"]["mean"] == 0 and not rec["G3"]["ok"]


def test_spearman_brown_through_g5_by_hand():
    """Two strata of 3: delays 0-7 hold x, 8-15 hold y. Centred ranks: x
    (-1/3, 0, 1/3) twice, y (-1/3, 1/3, 0) then x's: sum xy = 1/9 + 2/9,
    sum xx = 4/9, so rho_w = 0.75 and Spearman-Brown 1.5 / 1.75 = 6/7."""
    keys = [0, 0, 0, 1, 1, 1]
    x, y = np.array([1, 2, 3, 1, 2, 3.0]), np.array([1, 3, 2, 1, 2, 3.0])
    gap = np.zeros((6, 16))
    gap[:, :8], gap[:, 8:] = x[:, None], y[:, None]
    rec = s2.read_gates(full(np.zeros((6, 16))), full(gap + 10), keys,
                        list(range(6)), NO_G4, draws=100)
    assert rec["G5"]["rho_half"] == pytest.approx(0.75)
    assert rec["G5"]["reliability"] == pytest.approx(6 / 7)
    # T0 carries the gap with its sign flipped: the same halves
    rec2 = s2.read_gates(full(-gap), full(np.zeros((6, 16)) + 10), keys,
                         list(range(6)), NO_G4, draws=100)
    assert rec2["G5"]["reliability"] == pytest.approx(6 / 7)


def test_g5s_bar_is_inclusive_and_undefined_is_void(monkeypatch):
    t0, t2 = gap_world(LEVEL)
    for value, ok in ((0.5, True), (0.4999, False), (0.9, True)):
        monkeypatch.setattr(gc, "split_half", lambda gap, keys, v=value:
                            (0.3, v))
        rec = s2.read_gates(full(t0), full(t2), KEYS, CLUSTERS, NO_G4,
                            draws=100)
        assert rec["G5"]["ok"] is ok
        assert (rec["verdict"] == "PASS-SO-FAR") is ok
        if not ok:
            assert rec["reasons"] == ["G5: reliability 0.500 under 0.5"]
    monkeypatch.setattr(gc, "split_half", lambda gap, keys: (None, None))
    rec = s2.read_gates(full(t0), full(t2), KEYS, CLUSTERS, NO_G4, draws=100)
    assert rec["reasons"] == ["G5: reliability undefined"]


def test_g5_reads_the_gap_not_a_player():
    """A large level both players share is no gap: G5 fails though either
    player alone would pass; opposite levels are all gap."""
    rng = np.random.default_rng(3)
    keys = [(0, k) for k in range(20) for _ in range(3)]
    level = rng.normal(0, 30, size=(60, 1))
    t0 = level + rng.normal(size=(60, 16))
    t2 = level + rng.normal(size=(60, 16))
    rec = s2.read_gates(full(t0), full(t2), keys, list(range(60)), NO_G4,
                        draws=100)
    assert not rec["G5"]["ok"] and "G5" in rec["reasons"][-1]
    assert gc.split_half(t2, keys)[1] > 0.9
    rec = s2.read_gates(full(-level + rng.normal(size=(60, 16))),
                        full(level + rng.normal(size=(60, 16))), keys,
                        list(range(60)), NO_G4, draws=100)
    assert rec["G5"]["ok"] and rec["G5"]["reliability"] > 0.9


def test_g5_splits_delays_0_7_against_8_15():
    keys = [0, 0, 0, 1, 1, 1]
    level = np.array([1, 2, 3, 1, 3, 2.0])[:, None]
    gap = np.zeros((6, 16))
    gap[:, ::2], gap[:, 1::2] = level, -level   # interleaved: halves are flat
    rec = s2.read_gates(full(np.zeros((6, 16))), full(gap), keys,
                        list(range(6)), NO_G4, draws=100)
    assert rec["G5"]["reliability"] is None
    gap[:, :8], gap[:, 8:] = level, -level
    rec = s2.read_gates(full(np.zeros((6, 16))), full(gap), keys,
                        list(range(6)), NO_G4, draws=100)
    assert rec["G5"]["rho_half"] == pytest.approx(-1)


def test_the_lag_profile():
    rng = np.random.default_rng(4)
    t0, t2 = rng.normal(size=(40, 16)), rng.normal(size=(40, 16))
    t2[:, ::4] += 3.0                           # a phase of 4 in T2 alone
    rec = s2.read_gates(full(t0), full(t2), [0] * 40, list(range(40)), NO_G4,
                        draws=100)
    lags = rec["lags"]
    assert set(lags) == {"T0", "T2", "G"}
    for who, x in (("T0", t0), ("T2", t2), ("G", t2 - t0)):
        assert list(lags[who]) == list(range(1, 16))
        for lag in range(1, 16):
            assert lags[who][lag] == pytest.approx(gc.lag_correlation(x, lag))
    assert lags["T2"][4] > 0.5 > lags["T0"][4]
    assert rec["null"] == pytest.approx(-1 / 15)


@pytest.mark.parametrize("name", ["T0", "T2"])
def test_a_g4_failure_is_void(name):
    t0, t2 = gap_world(LEVEL)
    g4 = {"T0": [], "T2": []}
    g4[name] = ["state 4 delay 2: ['raw frames']", "state 9 delay 1: x"]
    rec = s2.read_gates(full(t0), full(t2), KEYS, CLUSTERS, g4, draws=100)
    assert rec["verdict"] == "VOID"
    assert rec["reasons"] == [f"G4 {name}: state 4 delay 2: ['raw frames'] "
                              f"(2 in all)"]
    assert rec["G4"][name]["ok"] is False


def test_every_failure_is_named():
    t0 = np.zeros((12, 16))
    rec = s2.read_gates(full(t0), full(t0), KEYS, CLUSTERS,
                        {"T0": ["a"], "T2": ["b"]}, draws=100)
    assert [r.split(":")[0] for r in rec["reasons"]] == ["G3", "G4 T0",
                                                          "G4 T2", "G5"]


# --------------------------------------------------------------------------
# files and refusals
# --------------------------------------------------------------------------
def test_files_write_whole(tmp_path):
    st = {"x": 1}
    s2.write(tmp_path / "a.json", {"v": 1}, st)
    s2.write(tmp_path / "b.pkl", {"v": b"z"}, st)
    assert s2.read(tmp_path / "a.json", st)["v"] == 1
    assert s2.read(tmp_path / "b.pkl", st)["v"] == b"z"
    with pytest.raises(Exception):
        s2.write(tmp_path / "c.pkl", {"v": lambda: 0}, st)
    with pytest.raises(Exception):
        s2.write(tmp_path / "d.json", {"v": object()}, st)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "b.pkl"]


def test_a_file_from_another_stamp_is_refused(tmp_path):
    st = s2.stamp([])
    s2.write(tmp_path / "a.json", {"v": 1}, st)
    for other in ({**st, "source": "x"}, {**st, "pilot": {"source": "x"}},
                  {**st, "knobs": {"chunk": 99, "states": 0}},
                  {**st, "states": "x"}):
        with pytest.raises(SystemExit, match="never mixed"):
            s2.read(tmp_path / "a.json", other)
    (tmp_path / "old.json").write_text('{"ok": true}')
    with pytest.raises(SystemExit, match="never mixed"):
        s2.read(tmp_path / "old.json", st)


def test_the_fingerprint_reads_every_state_field():
    base = {"game": 1, "decision": 2, "life": 0, "steps": 90,
            "state": b"abc", "stack": np.zeros((12, 84, 84), np.uint8)}
    fp = s2.fingerprint([base])
    stack = base["stack"].copy()
    stack[5, 5, 5] = 1
    for change in ({"game": 2}, {"decision": 3}, {"life": 1}, {"steps": 91},
                   {"state": b"abd"}, {"stack": stack}):
        assert s2.fingerprint([{**base, **change}]) != fp
    assert s2.fingerprint([base, base]) != fp


def test_chunks_cover_every_state_once(monkeypatch):
    monkeypatch.setattr(s2, "CHUNK", 5)
    assert s2.chunks(12) == [(0, 0, 5), (1, 5, 10), (2, 10, 12)]
    assert s2.chunks(10) == [(0, 0, 5), (1, 5, 10)]
    assert s2.chunk_range(12, 2) == (10, 12)
    with pytest.raises(SystemExit):
        s2.chunk_range(12, 3)
    monkeypatch.setattr(s2, "CHUNK", 100)
    assert [hi - lo for _, lo, hi in s2.chunks(1121)] == [100] * 11 + [21]
    monkeypatch.setattr(s2, "CHUNK", 0)
    with pytest.raises(SystemExit):
        s2.chunks(5)


def test_no_output_inside_the_pilot(tmp_path):
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    for out in (pilot, pilot / "x", pilot / "x" / "y", s2.PILOT,
                s2.PILOT / "sub"):
        with pytest.raises(SystemExit, match="never written"):
            s2.guard(out, pilot)
        with pytest.raises(SystemExit, match="never written"):
            s2.stage2a(out, pilot)
        for stage in (s2.stage_branches, s2.stage_g4):
            with pytest.raises(SystemExit, match="never written"):
                stage(out, pilot, "T0", 0)
    s2.guard(tmp_path / "out", pilot)
    with pytest.raises(SystemExit, match="never written"):
        s2.main(["stage2a", "--out", str(pilot / "o"), "--pilot", str(pilot)])
    assert not (pilot / "o").exists()


# --------------------------------------------------------------------------
# the orchestrator's routes, on stubbed pilot and pass files
# --------------------------------------------------------------------------
N_STUB = 12
STUB_KEYS = [(0, k) for k in range(4) for _ in range(3)]


def stub_states():
    return [{"game": i, "decision": 5, "life": 0, "steps": 20 * (i // 3) + 1,
             "state": b"state%d" % i, "merged": [],
             "stack": np.full((12, 84, 84), i, dtype=np.uint8)}
            for i in range(N_STUB)]


def make_stub_pilot(d, verdict="GO"):
    d.mkdir(parents=True, exist_ok=True)
    kept = stub_states()
    games = [{"game": i, "digests": [b"d%d" % i, b"shared" if i in (1, 4)
                                     else b"own%d" % i]}
             for i in range(N_STUB)]
    gc.write(d / "main-080.pkl", {"opening": 80, "games": games,
                                  "states": kept})
    gc.write(d / "select.pkl", {"opening": 80, "kept": kept,
                                "keys": STUB_KEYS, "chosen": [0, 1, 2],
                                "counts": {"kept": N_STUB}})
    gc.write(d / "pilot.json", {"verdict": verdict, "reasons": [],
                                "effective": [70.8], "opening": 80,
                                "counts": {"kept": N_STUB}})
    for n in s2.PLAYERS:
        gc.write(d / f"g1-{n}.json", {"ok": True, "reasons": []})
    read_only(d)
    return d


def table_of(returns):
    """Rows of branches with these primary returns: one base step each,
    under the clip's ceiling, so a return x is 10 x raw points."""
    return [[{"delay": d, "rewards": [10.0 * float(x)], "ended": "life"}
             for d, x in zip(s2.DELAYS, row)] for row in returns]


class Passes:
    """Stands in for run_pass: writes each pass's file as the real pass
    would, from canned returns, and logs the calls. `die_at`: that call fails
    as a crashed pass."""

    def __init__(self, t0, t2):
        self.returns = {"T0": t0, "T2": t2}
        self.calls, self.die_at = [], None
        self.g4 = {"T0": [], "T2": []}

    def __call__(self, out, pilot, stage, name, k):
        call = f"{stage} {name} {k}"
        self.calls.append(call)
        assert len(self.calls) <= 4 * len(s2.chunks(N_STUB)), self.calls
        if call == self.die_at:
            raise SystemExit(f"pass {call}: rc 1")
        _, states, _ = s2.load_states(pilot)
        st = s2.stamp(states)
        lo, hi = s2.chunk_range(len(states), k)
        if stage == "branches":
            s2.write(s2.branch_file(out, name, k),
                     {"player": name, "chunk": k, "range": [lo, hi],
                      "table": table_of(self.returns[name][lo:hi]),
                      "reasons": [], "secs": 2.0 * (hi - lo),
                      "memory": {"peak_wset_mb": 900 + k}}, st)
        else:
            s2.write(s2.g4_file(out, name, k),
                     {"ok": not self.g4[name], "reasons": self.g4[name],
                      "range": [lo, hi], "secs": 1.0, "memory": {},
                      "branches": s2.file_digest(s2.branch_file(out, name,
                                                                k))}, st)


def stub_returns(shared=False):
    rng = np.random.default_rng(8)
    level = rng.uniform(0, .6, size=(N_STUB, 1))
    t0 = (level if shared else 0) + rng.uniform(0, .2, size=(N_STUB, 16))
    t2 = level + 0.2 + rng.uniform(0, .2, size=(N_STUB, 16))
    return t0, t2


def all_passes():
    ks = [k for k, _, _ in s2.chunks(N_STUB)]
    return [f"{stage} {n} {k}" for stage in ("branches", "g4")
            for n in s2.PLAYERS for k in ks]


@pytest.fixture
def stub(tmp_path, monkeypatch):
    pilot = make_stub_pilot(tmp_path / "pilot")
    monkeypatch.setattr(s2, "CHUNK", 5)

    def build(**kw):
        passes = Passes(*stub_returns(**kw))
        monkeypatch.setattr(s2, "run_pass", passes)
        return passes

    yield SimpleNamespace(pilot=pilot, out=tmp_path / "out", build=build)
    read_only(pilot, on=False)


def test_the_route_to_pass_so_far(stub, capsys):
    passes = stub.build()
    stub.out.mkdir()
    before = listing(stub.pilot)
    rec = s2.stage2a(stub.out, stub.pilot)
    assert passes.calls == all_passes()
    assert rec["verdict"] == "PASS-SO-FAR" and rec["reasons"] == []
    assert listing(stub.pilot) == before
    st = s2.stamp(stub_states())
    saved = s2.read(stub.out / "gates.json", st)
    assert saved["verdict"] == "PASS-SO-FAR" and saved["states"] == N_STUB
    assert saved["pilot"]["G6"] == 70.8 and saved["pilot"]["G1"] == {
        "T0": True, "T2": True}
    # the gap file holds every branch's return, and G is T2's mean minus T0's
    gap = s2.read(stub.out / "gap.pkl", st)
    t0, t2 = passes.returns["T0"], passes.returns["T2"]
    assert gap["returns"]["T0"].shape == (N_STUB, 16, 2, 4)
    assert gap["returns"]["T0"][:, :, 0, 0] == pytest.approx(t0)
    assert gap["returns"]["T2"][:, :, 1, 0] == pytest.approx(10 * t2)
    assert gap["G"][:, 0, 0] == pytest.approx(t2.mean(1) - t0.mean(1))
    assert saved["G3"]["mean"] == pytest.approx((t2 - t0).mean())
    assert gap["keys"] == STUB_KEYS and gap["delays"] == list(range(16))
    # clusters: games 1 and 4 share a stack
    assert gap["clusters"] == [0, 1, 2, 3, 1, 5, 6, 7, 8, 9, 10, 11]
    assert saved["clusters"] == {"count": 11, "largest": 2}
    assert saved["G3"]["clusters"] == 11
    assert saved["ended"]["T0"] == {"life": 16 * N_STUB, "cap": 0, "delay": 0}
    assert saved["cost"]["T0"]["secs_per_state"] == 2.0
    assert saved["cost"]["T2"]["peak_wset_mb"] == 902
    out = capsys.readouterr().out
    assert "STAGE 2a: PASS-SO-FAR" in out and "lag profile G" in out
    assert "SMOKE" not in out


def test_the_summary_is_never_overwritten(stub):
    """Refused before any pass or write; and finish() itself refuses a
    summary that appeared while it ran."""
    passes = stub.build()
    stub.out.mkdir()
    s2.stage2a(stub.out, stub.pilot)
    before = listing(stub.out)
    passes.calls = []
    with pytest.raises(SystemExit, match="exists: the summary is never"):
        s2.stage2a(stub.out, stub.pilot)
    assert listing(stub.out) == before and passes.calls == []
    with pytest.raises(SystemExit, match="appeared: never overwritten"):
        s2.finish(stub.out, stub.pilot, s2.pilot_record(stub.pilot), 0)
    assert listing(stub.out)["gates.json"] == before["gates.json"]


def test_a_pilot_that_did_not_go_is_refused(tmp_path, monkeypatch):
    pilot = make_stub_pilot(tmp_path / "pilot", verdict="STOP")
    passes = Passes(*stub_returns())
    monkeypatch.setattr(s2, "run_pass", passes)
    try:
        with pytest.raises(SystemExit, match="not GO"):
            s2.stage2a(tmp_path / "out", pilot)
        assert passes.calls == []
    finally:
        read_only(pilot, on=False)


def test_g3_and_g5_void_the_stage(stub, capsys):
    stub.build(shared=True)
    passes = s2.run_pass
    passes.returns["T2"] = passes.returns["T0"].copy()
    stub.out.mkdir()
    rec = s2.stage2a(stub.out, stub.pilot)
    assert rec["verdict"] == "VOID"
    assert [r[:3] for r in rec["reasons"]] == ["G3:", "G5:"]
    assert "STAGE 2a: VOID (G3" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["T0", "T2"])
def test_a_g4_pass_failure_voids_the_stage(stub, name):
    passes = stub.build()
    passes.g4[name] = ["state 7 delay 3: ['tokens']"]
    stub.out.mkdir()
    rec = s2.stage2a(stub.out, stub.pilot)
    assert rec["verdict"] == "VOID"
    # each chunk's g4 pass carries it, so three in all
    assert rec["reasons"] == [f"G4 {name}: state 7 delay 3: ['tokens'] "
                              f"(3 in all)"]


def test_a_stopped_stage_resumes_where_it_stopped(stub):
    passes = stub.build()
    passes.die_at = "branches T2 1"
    stub.out.mkdir()
    with pytest.raises(SystemExit, match="rc 1"):
        s2.stage2a(stub.out, stub.pilot)
    assert not (stub.out / "gates.json").exists()
    done = all_passes().index("branches T2 1")
    assert passes.calls == all_passes()[:done + 1]
    passes.calls, passes.die_at = [], None
    assert s2.stage2a(stub.out, stub.pilot)["verdict"] == "PASS-SO-FAR"
    assert passes.calls == all_passes()[done:]


def _partial(stub):
    """Every pass file of a stage, with no summary to stop a second one."""
    stub.build()
    stub.out.mkdir()
    s2.stage2a(stub.out, stub.pilot)
    (stub.out / "gates.json").unlink()


@pytest.mark.parametrize("knob, value", [("CHUNK", 6), ("LIMIT", 11)])
def test_a_pass_from_other_knobs_is_refused(stub, monkeypatch, knob, value):
    _partial(stub)
    monkeypatch.setattr(s2, knob, value)
    with pytest.raises(SystemExit, match="never mixed"):
        s2.finish(stub.out, stub.pilot, s2.pilot_record(stub.pilot), 0)
    assert not (stub.out / "gates.json").exists()


def test_a_pass_from_other_source_is_refused(stub, monkeypatch):
    _partial(stub)
    monkeypatch.setattr(s2, "SOURCE", "another build")
    with pytest.raises(SystemExit, match="never mixed"):
        s2.stage2a(stub.out, stub.pilot)


def test_a_pass_on_other_pilot_code_is_refused(stub, monkeypatch):
    """The pilot's stamp in this stage's: another pilot source refuses the
    pilot's own files first, and this stage's files too."""
    _partial(stub)
    monkeypatch.setattr(gc, "SOURCE", "another pilot")
    with pytest.raises(SystemExit, match="never mixed"):
        s2.stage2a(stub.out, stub.pilot)
    states = stub_states()
    other = {**s2.stamp(states)}
    with pytest.raises(SystemExit, match="never mixed"):
        s2.read(s2.branch_file(stub.out, "T0", 0), other)


def test_a_pass_on_other_states_is_refused(stub):
    """The same knobs and code on a select.pkl whose states differ."""
    _partial(stub)
    read_only(stub.pilot, on=False)
    sel = gc.read(stub.pilot / "select.pkl")
    sel["kept"][7]["stack"] = sel["kept"][7]["stack"].copy()
    sel["kept"][7]["stack"][0, 0, 0] += 1
    gc.write(stub.pilot / "select.pkl", {k: v for k, v in sel.items()
                                         if k != "stamp"})
    with pytest.raises(SystemExit, match="never mixed"):
        s2.stage2a(stub.out, stub.pilot)


@pytest.mark.parametrize("name", ["branches-T0-001.pkl", "g4-T2-002.json"])
def test_every_pass_file_is_checked_alone(stub, name):
    _partial(stub)
    path = stub.out / name
    if path.suffix == ".json":
        obj = json.loads(path.read_text())
        obj["stamp"]["source"] = "stale"
        path.write_text(json.dumps(obj))
    else:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj["stamp"]["source"] = "stale"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    with pytest.raises(SystemExit, match=name.replace(".", "[.]")):
        s2.stage2a(stub.out, stub.pilot)


def test_a_pass_file_for_other_states_is_refused(stub):
    _partial(stub)
    st = s2.stamp(stub_states())
    path = s2.branch_file(stub.out, "T2", 1)
    obj = s2.read(path, st)
    obj["range"] = [4, 10]
    s2.write(path, {k: v for k, v in obj.items() if k != "stamp"}, st)
    with pytest.raises(SystemExit, match="not states 5-10"):
        s2.stage2a(stub.out, stub.pilot)


def test_a_g4_is_bound_to_the_branches_it_replayed(stub):
    """A branches file made again after its G4 (same stamp, other content)
    is not covered by that G4."""
    _partial(stub)
    st = s2.stamp(stub_states())
    path = s2.branch_file(stub.out, "T0", 2)
    obj = s2.read(path, st)
    obj["table"][0][4]["rewards"] = [3.0]
    s2.write(path, {k: v for k, v in obj.items() if k != "stamp"}, st)
    with pytest.raises(SystemExit, match="g4-T0-002.json replayed another"):
        s2.stage2a(stub.out, stub.pilot)


def test_a_smoke_stage_says_so(stub, monkeypatch, capsys):
    stub.build()
    monkeypatch.setattr(s2, "LIMIT", 7)
    monkeypatch.setattr(s2, "SMOKE", True)
    stub.out.mkdir()
    rec = s2.stage2a(stub.out, stub.pilot)
    assert rec["smoke"] and rec["states"] == 7
    assert s2.run_pass.calls == [c for c in all_passes() if c[-1] in "01"]
    assert "SMOKE STAGE 2a" in capsys.readouterr().out


def test_a_crashed_pass_stops_the_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=3))
    with pytest.raises(SystemExit, match="rc 3"):
        s2.run_pass(tmp_path, tmp_path, "branches", "T0", 0)


# --------------------------------------------------------------------------
# a small stage on the real game, stand-in players, in this process
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_pilot(tmp_path_factory):
    """A small pilot's files the pilot's way: two real games played from their
    openings by T0's stand-in, two decisions kept from each."""
    out = tmp_path_factory.mktemp("pilot")
    env = gc.raw_env()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(muzero, "TOKEN_VOCAB", VOCAB)
            games, kept = [], []
            for g in range(2):
                rec, held = gc.main_game(env, decide_of("T0"), g,
                                         gc.opening_moves(g, 80))
                for i in gc.draw_game_states(g, len(held))[:2]:
                    kept.append({"game": g, "decision": i,
                                 "life": rec["life"][i],
                                 "steps": rec["steps"][i],
                                 "state": held[i][0], "stack": held[i][1],
                                 "merged": []})
                games.append(rec)
    finally:
        env.close()
    keys = [gc.stratum(s["life"], s["steps"]) for s in kept]
    gc.write(out / "main-080.pkl", {"opening": 80, "games": games,
                                    "states": kept})
    gc.write(out / "select.pkl", {"opening": 80, "kept": kept, "keys": keys,
                                  "chosen": [0, 1, 2],
                                  "counts": {"kept": len(kept)}})
    gc.write(out / "pilot.json", {"verdict": "GO", "reasons": [],
                                  "effective": [70.8], "opening": 80,
                                  "counts": {"kept": len(kept)}})
    for n in s2.PLAYERS:
        gc.write(out / f"g1-{n}.json", {"ok": True, "reasons": []})
    read_only(out)
    yield out
    read_only(out, on=False)


def in_process(out, pilot, stage, name, k):
    {"branches": s2.stage_branches, "g4": s2.stage_g4}[stage](out, pilot,
                                                             name, k)


@pytest.fixture(scope="module")
def built(real_pilot, tmp_path_factory):
    out = tmp_path_factory.mktemp("stage2a")
    before = listing(real_pilot)
    with pytest.MonkeyPatch.context() as mp:
        small(mp)
        mp.setattr(s2, "run_pass", in_process)
        rec = s2.stage2a(out, real_pilot)
    return SimpleNamespace(out=out, pilot=real_pilot, rec=rec, before=before)


@pytest.fixture
def pipeline(built, monkeypatch):
    small(monkeypatch)
    return built


def test_the_pilots_files_are_read_and_never_written(pipeline):
    assert listing(pipeline.pilot) == pipeline.before
    assert all(not os.access(p, os.W_OK) for p in pipeline.pilot.iterdir())
    assert sorted(p.name for p in pipeline.pilot.iterdir()) == [
        "g1-T0.json", "g1-T2.json", "main-080.pkl", "pilot.json",
        "select.pkl"]


def test_the_real_stage_reads_clean(pipeline):
    rec = pipeline.rec
    assert rec["states"] == 4 and rec["G4"]["T0"]["ok"] and \
        rec["G4"]["T2"]["ok"]
    assert sorted(p.name for p in pipeline.out.iterdir()) == [
        "branches-T0-000.pkl", "branches-T0-001.pkl", "branches-T2-000.pkl",
        "branches-T2-001.pkl", "g4-T0-000.json", "g4-T0-001.json",
        "g4-T2-000.json", "g4-T2-001.json", "gap.pkl", "gates.json"]
    _, states, keys = s2.load_states(pipeline.pilot)
    st = s2.stamp(states)
    table = {n: [] for n in s2.PLAYERS}
    for n in s2.PLAYERS:
        for k in (0, 1):
            b = s2.read(s2.branch_file(pipeline.out, n, k), st)
            assert b["states"] == [[s["game"], s["decision"]]
                                   for s in states[b["range"][0]:
                                                   b["range"][1]]]
            table[n] += b["table"]
    gap = s2.read(pipeline.out / "gap.pkl", st)
    for n in s2.PLAYERS:
        assert gap["returns"][n] == pytest.approx(s2.branch_returns(table[n]))
        for row in table[n]:
            assert [b["delay"] for b in row] == list(range(16))
            for b in row:
                assert len(b["rewards"]) <= (SMALL_CAP - b["delay"]) // 4
    want = s2.branch_returns(table["T2"]).mean(1) \
        - s2.branch_returns(table["T0"]).mean(1)
    assert gap["G"] == pytest.approx(want)
    assert gap["keys"] == keys
    # the two players differ, so G does, and the gates read it
    assert np.abs(gap["G"]).sum() > 0
    assert rec["G3"]["mean"] == pytest.approx(want[:, 0, 0].mean())


def test_g4_catches_a_main_line_the_branches_left(pipeline, tmp_path):
    out = tmp_path / "pass"
    shutil.copytree(pipeline.out, out)
    pilot = tmp_path / "pilot"
    shutil.copytree(pipeline.pilot, pilot)
    read_only(pilot, on=False)
    path = pilot / "main-080.pkl"
    data = gc.read(path)
    _, states, _ = s2.load_states(pilot)
    game = data["games"][states[0]["game"]]
    life = game["life"][states[0]["decision"]]
    game["rewards"][life] = [r + 10.0 for r in game["rewards"][life]]
    gc.write(path, {k: v for k, v in data.items() if k != "stamp"})
    s2.stage_g4(out, pilot, "T0", 0)
    st = s2.stamp(states)
    g4 = s2.read(s2.g4_file(out, "T0", 0), st)
    assert not g4["ok"] and "main line" in g4["reasons"][0]
    assert g4["branches"] == s2.file_digest(s2.branch_file(out, "T0", 0))
    s2.stage_g4(out, pilot, "T2", 0)          # T2 never played the main line
    assert s2.read(s2.g4_file(out, "T2", 0), st)["ok"]


def test_g4_catches_a_branch_that_does_not_repeat(pipeline, tmp_path):
    out = tmp_path / "pass"
    shutil.copytree(pipeline.out, out)
    _, states, _ = s2.load_states(pipeline.pilot)
    st = s2.stamp(states)
    checked = gc.g4_delays(len(states))
    path = s2.branch_file(out, "T2", 0)
    data = s2.read(path, st)
    data["table"][0][checked[0]]["rewards"][0] += 10.0
    data["table"][1][checked[1]]["replay"] = b"other frames"
    data["table"][2][checked[2]]["tokens"] = [99]
    data["reasons"] = ["state 2 delay 5: the raw replay ended elsewhere"]
    s2.write(path, {k: v for k, v in data.items() if k != "stamp"}, st)
    s2.stage_g4(out, pipeline.pilot, "T2", 0)
    g4 = s2.read(s2.g4_file(out, "T2", 0), st)
    assert not g4["ok"] and len(g4["reasons"]) == 4
    assert "replay ended elsewhere" in g4["reasons"][0]
    assert "rewards" in g4["reasons"][1]
    assert "raw frames" in g4["reasons"][2]
    assert "tokens" in g4["reasons"][3]


def test_g4_replays_only_its_own_chunk_and_player(pipeline, tmp_path):
    """A branches file for other states, or another player's, is refused
    before anything plays."""
    out = tmp_path / "pass"
    shutil.copytree(pipeline.out, out)
    _, states, _ = s2.load_states(pipeline.pilot)
    st = s2.stamp(states)
    path = s2.branch_file(out, "T2", 1)
    data = s2.read(path, st)
    s2.write(path, {**{k: v for k, v in data.items() if k != "stamp"},
                    "range": [2, 4]}, st)
    with pytest.raises(SystemExit, match="not T2's states 3-4"):
        s2.stage_g4(out, pipeline.pilot, "T2", 1)
    shutil.copy(s2.branch_file(out, "T0", 0), s2.branch_file(out, "T2", 0))
    with pytest.raises(SystemExit, match="not T2's states 0-3"):
        s2.stage_g4(out, pipeline.pilot, "T2", 0)


_FRESH = r"""
import sys
sys.path[:0] = [{tests!r}, {probes!r}, {exp!r}, {root!r}]
import pathlib
import muzero
import probe_gap_check as gc
import probe_gap_stage2a as s2
import test_gap_stage2a as t
muzero.TOKEN_VOCAB = t.VOCAB
muzero.USE_SEARCH_ACTING = False
gc.player = t.stand_in
s2.FRAME_CAP = t.SMALL_CAP
s2.CHUNK = t.SMALL_CHUNK
for name in s2.PLAYERS:
    for k in (0, 1):
        s2.stage_g4(pathlib.Path({out!r}), pathlib.Path({pilot!r}), name, k)
"""


def test_g4_in_a_fresh_process(pipeline, tmp_path):
    """G4's own route: the saved states cross as bytes, the emulator and the
    policy are built anew, and every checked branch repeats."""
    out = tmp_path / "pass"
    shutil.copytree(pipeline.out, out)
    for p in out.glob("g4-*.json"):
        p.unlink()
    code = _FRESH.format(tests=str(ROOT / "tests"), probes=str(PROBES),
                         exp=str(ROOT / "experiments"), root=str(ROOT),
                         out=str(out), pilot=str(pipeline.pilot))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    _, states, _ = s2.load_states(pipeline.pilot)
    st = s2.stamp(states)
    for n in s2.PLAYERS:
        for k in (0, 1):
            assert s2.read(s2.g4_file(out, n, k), st)["ok"]
    assert listing(pipeline.pilot) == pipeline.before


# --------------------------------------------------------------------------
# slow: the real players
# --------------------------------------------------------------------------
def _real_or_skip():
    try:
        for name in gc.PLAYERS:
            gc.snapshot(name)
        gc.read(s2.PILOT / "pilot.json")
    except (SystemExit, FileNotFoundError):
        pytest.skip("the players' snapshots or the pilot's files are not here")


_REPLAY = r"""
import sys
sys.path[:0] = [{probes!r}, {exp!r}, {root!r}]
import pickle
import probe_gap_check as gc
import probe_gap_stage2a as s2
sel = gc.read(s2.PILOT / "select.pkl")
out = {{}}
for name in s2.PLAYERS:
    pilot_rows = gc.read(s2.PILOT / f"branches-{{name}}.pkl")["table"]
    with gc.player(name) as run:
        decide = gc.deployed(run)
        out[name] = []
        for j in (0, 31):
            row, reason = s2.play_state(run.env, decide,
                                        sel["kept"][sel["chosen"][j]], 0)
            out[name].append(([gc.same_branch(a, b)
                               for a, b in zip(pilot_rows[j], row)], reason))
with open({res!r}, "wb") as f:
    pickle.dump(out, f)
"""


@pytest.mark.slow
def test_the_pilots_branches_replay_exactly(tmp_path):
    """Two of the pilot's 60 states, both players: this stage's 16 branches
    are the pilot's, field for field."""
    _real_or_skip()
    before = listing(s2.PILOT)
    res = tmp_path / "res.pkl"
    code = _REPLAY.format(probes=str(PROBES), exp=str(ROOT / "experiments"),
                          root=str(ROOT), res=str(res))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       env={**os.environ, **s2.ENV}, capture_output=True,
                       text=True, timeout=1800)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    with open(res, "rb") as f:
        out = pickle.load(f)
    for name in s2.PLAYERS:
        for differ, reason in out[name]:
            assert reason is None and differ == [[]] * 16, (name, differ)
    assert listing(s2.PILOT) == before


@pytest.mark.slow
def test_a_small_stage_on_the_real_players(tmp_path, monkeypatch):
    _real_or_skip()
    before = listing(s2.PILOT)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "GAP2_STATES": "3",
           "GAP2_CHUNK": "2"}
    r = subprocess.run([sys.executable, str(PROBES / "probe_gap_stage2a.py"),
                        "stage2a", "--out", str(tmp_path)], cwd=str(ROOT),
                       env=env, capture_output=True, text=True, timeout=2400)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    print(r.stdout)
    rec = json.loads((tmp_path / "gates.json").read_text())
    assert rec["smoke"] is True and rec["states"] == 3
    assert rec["verdict"] in ("PASS-SO-FAR", "VOID")
    assert rec["G4"]["T0"]["ok"] and rec["G4"]["T2"]["ok"], rec["G4"]
    assert ("STAGE 2a: " + rec["verdict"]) in r.stdout
    # this process reads the passes under the knobs that made them
    monkeypatch.setattr(s2, "LIMIT", 3)
    monkeypatch.setattr(s2, "CHUNK", 2)
    _, states, keys = s2.load_states(s2.PILOT)
    st = s2.stamp(states)
    table = {n: [] for n in s2.PLAYERS}
    for n in s2.PLAYERS:
        for k in (0, 1):
            b = s2.read(s2.branch_file(tmp_path, n, k), st)
            assert b["threads"] == 2
            table[n] += b["table"]
    gap = s2.read(tmp_path / "gap.pkl", st)
    returns = {n: s2.branch_returns(table[n]) for n in s2.PLAYERS}
    assert gap["G"] == pytest.approx(s2.real_gap(returns["T0"],
                                                 returns["T2"]))
    assert rec["G3"]["mean"] == pytest.approx(gap["G"][:, 0, 0].mean())
    assert rec["ended"]["T0"]["life"] + rec["ended"]["T0"]["cap"] \
        + rec["ended"]["T0"]["delay"] == 48
    assert listing(s2.PILOT) == before
