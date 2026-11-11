"""Test H, stage A (CURRICULUM "Next experiment (pre-registered) -- Test H"):
the own-play data. Pins: the sampler against legacy np.random.choice at the
same uniform; the counter-based streams (deterministic, keyed, independent of
play); the temperature recomputation against muzero's own schedule under the
pins, at and around both stage boundaries; the gate's refusals; the stack
rebuild from recorded frames; the game loop on a stand-in env (a life lost in
the opening, the life cap, the game end, decision indices, the hand-off from
the opening to the policy); on the real game with no snapshot, the probe's env
against ablation.build_env step for step, the hand-off, and a held-out state
restored with its MaxAndSkip buffer reproducing the recorded step; and (slow,
needing the snapshots) the probe's loop against Run.play_episode move for move
and stack for stack over whole games on O0 at T 0.5 and T 0.125 and on OF, and
a one-game smoke of the stage."""
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBES = ROOT / "benchmarks" / "probes"
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(PROBES))
import ale_py  # noqa: E402
import muzero  # noqa: E402
import ablation  # noqa: E402
import probe_move_guesser as mg  # noqa: E402

HAVE_SNAPSHOTS = all(
    (pathlib.Path(s["tree"]) / "benchmarks" / "results" / s["run"]).exists()
    for s in mg.RUNS.values())


# --------------------------------------------------------------------------
# sampler and streams
# --------------------------------------------------------------------------
def test_sample_matches_legacy_choice_at_the_same_uniform():
    rng = np.random.default_rng(0)
    for trial in range(3000):
        logits = rng.normal(size=mg.MOVES) * rng.uniform(0.1, 4)
        p = np.exp(logits).astype(np.float32)
        p = p / p.sum()
        p = mg.tempered(p, float(rng.choice([0.5, 0.25, 0.125])))
        np.random.seed(trial)
        a = int(np.random.choice(mg.MOVES, p=p))
        np.random.seed(trial)
        u = np.random.random_sample()
        assert mg.sample(p, u) == a


def test_tempered_is_play_episodes_float32_rule():
    p = np.array([0.5, 0.25, 0.25, 0, 0, 0, 0, 0, 0], np.float32)
    q = mg.tempered(p, 0.5)
    assert q.dtype == np.float32
    expect = p ** 2.0
    np.testing.assert_array_equal(q, expect / expect.sum())
    assert mg.tempered(p, 1.0) is p


def test_streams_are_deterministic_keyed_and_in_range():
    k = mg.game_key("run", 10, 0)
    assert k == mg.game_key("run", 10, 0)
    assert len({mg.game_key("run", 10, g) for g in range(10)}
               | {mg.game_key("run", 20, 0), mg.game_key("other", 10, 0)}) == 12
    us = [mg.uniform(k, d) for d in range(500)]
    assert us == [mg.uniform(k, d) for d in range(500)]
    assert all(0.0 <= u < 1.0 for u in us) and len(set(us)) == 500
    assert mg.uniform(k, 3) != mg.uniform(mg.game_key("run", 10, 1), 3)
    assert all(1 <= mg.noop_count(mg.game_key("r", i, 0)) <= 30 for i in range(300))
    assert {mg.noop_count(mg.game_key("r", i, 0)) for i in range(300)} == set(range(1, 31))
    moves = [mg.opening_move(k, j) for j in range(400)]
    assert set(moves) == set(range(mg.MOVES)) and moves == [mg.opening_move(k, j) for j in range(400)]
    # the three streams do not share draws
    assert mg.uniform(k, 0) != float(mg._rng(k, mg.TAG_OPEN, 0).random())


# --------------------------------------------------------------------------
# temperature
# --------------------------------------------------------------------------
@contextlib.contextmanager
def pins(env_steps, scale=0.5):
    saved = (muzero.PRIOR_ACT_STAGED_TEMP, muzero.PRIOR_ACT_TEMP_SCALE,
             muzero.SCHEDULE_ENV_TARGET, muzero.SCHEDULE_ENV_STEPS)
    with ablation._run_length_globals(mg.PLANNED_ITERATIONS, mg.TRAIN_STEPS):
        try:
            muzero.PRIOR_ACT_STAGED_TEMP = True
            muzero.PRIOR_ACT_TEMP_SCALE = scale
            muzero.SCHEDULE_ENV_TARGET = mg.ENV_TARGET
            muzero.SCHEDULE_ENV_STEPS = env_steps
            yield
        finally:
            (muzero.PRIOR_ACT_STAGED_TEMP, muzero.PRIOR_ACT_TEMP_SCALE,
             muzero.SCHEDULE_ENV_TARGET, muzero.SCHEDULE_ENV_STEPS) = saved


@pytest.mark.parametrize("env_steps", [0, 46_507, 49_189, 49_999, 50_000, 50_001,
                                       52_838, 72_021, 74_869, 74_999, 75_000,
                                       75_001, 78_088, 99_318, 100_204, 120_000])
def test_recomputation_matches_muzeros_schedule_under_the_pins(env_steps):
    with pins(env_steps):
        t_code = muzero.prior_acting_temperature(123_456)
    assert t_code == mg.expected_temperature(env_steps, 0.5)


def test_registered_stages():
    assert mg.expected_temperature(46_507, 0.5) == 0.5      # O0 row 80
    assert mg.expected_temperature(52_838, 0.5) == 0.25     # O0 row 90
    assert mg.expected_temperature(74_869, 0.5) == 0.25     # OF row 120
    assert mg.expected_temperature(78_088, 0.5) == 0.125    # O0 row 130
    assert mg.expected_temperature(100_204, 0.5) == 0.125   # OF 159


def test_gate_refusals_and_rows():
    table = {80: {"env_steps": 46_507}, 85: {"env_steps": 49_800,
                                           "beh_T_values": [0.5]},
             90: {"env_steps": 52_838}, 95: {"env_steps": 55_000,
                                            "beh_T_values": [0.25]},
             159: {"env_steps": 100_204}}
    assert mg.temperature_gate("O0", 80, table, 0.5, 0.5)[:2] == (46_507, 85)
    assert mg.temperature_gate("O0", 90, table, 0.25, 0.5)[:2] == (52_838, 95)
    assert mg.temperature_gate("OF", 159, table, 0.125, 0.5)[:2] == (100_204, None)
    with pytest.raises(SystemExit, match="differs from the recomputation"):
        mg.temperature_gate("O0", 80, table, 0.25, 0.5)
    table[85]["beh_T_values"] = [0.25]
    with pytest.raises(SystemExit, match="not among row 85"):
        mg.temperature_gate("O0", 80, table, 0.5, 0.5)
    with pytest.raises(SystemExit, match="no jsonl row"):
        mg.temperature_gate("O0", 100, table, 0.5, 0.5)


# --------------------------------------------------------------------------
# the loop on a stand-in env
# --------------------------------------------------------------------------
class FakeALE:
    def __init__(self, env):
        self.env = env

    def lives(self):
        return self.env.lives_left

    def cloneState(self):
        return SimpleState(self.env.t)


class SimpleState:
    def __init__(self, t):
        self.t = t

    def serialize(self):
        return int(self.t).to_bytes(8, "little")


class FakeEnv:
    """EpisodicLifeEnv's surface: reset() is real when was_real_done, else a
    NOOP step; step() ends the episode at a scripted life loss (life lost at
    the steps in `deaths`, game over at the last)."""

    def __init__(self, deaths):
        self.deaths = list(deaths)
        self.t = 0
        self.lives_left = len(self.deaths)
        self.lives = self.lives_left
        self.was_real_done = True
        self.unwrapped = type("U", (), {})()
        self.unwrapped.ale = FakeALE(self)
        self.resets = []

    def _obs(self):
        rng = np.random.default_rng(self.t)
        return rng.integers(0, 255, (210, 160, 3), dtype=np.uint8)

    def reset(self, **kw):
        if self.was_real_done:
            self.resets.append(("real", self.t))
        else:
            self.resets.append(("noop", self.t))
            self.t += 1
        return self._obs(), {}

    def step(self, a):
        self.t += 1
        term = False
        if self.deaths and self.t >= self.deaths[0]:
            self.deaths.pop(0)
            self.lives_left -= 1
            term = True
        self.was_real_done = term and self.lives_left == 0
        self.lives = self.lives_left
        return self._obs(), 1.0, term, False, {}


class FakeTap:
    def take(self):
        return [b"\x00" * mg.DIGEST] * 4


def fake_choose(stack, j, decision):
    if decision is None:
        return j % mg.MOVES, None, None, None
    return (decision * 7) % mg.MOVES, 0.5, np.full(mg.MOVES, 1 / 9, np.float32), 0.25


def test_loop_life_lost_in_opening_cap_and_game_end():
    env = FakeEnv(deaths=[30, 150, 400])       # life 0 ends inside the opening
    seen = []
    arr, states = mg.play_game(env, FakeTap(), None, fake_choose, opening=80,
                               cap=200, on_decision=lambda s, d: seen.append((d, s)))
    kinds, lives = arr["kind"], arr["life"]
    assert [k for k in env.resets] == [("real", 0), ("noop", 30), ("noop", 150)]
    starts = np.flatnonzero(kinds == mg.KIND_START)
    assert list(lives[starts]) == [0, 1, 2]
    # the opening runs across the life change: 80 base steps, life-change NOOPs excluded
    assert int((kinds == mg.KIND_OPEN).sum()) == 80
    assert np.all(lives[kinds == mg.KIND_OPEN][:30] == 0)
    assert np.all(lives[kinds == mg.KIND_OPEN][30:] == 1)
    # decisions are numbered from 0 after the opening, in order
    dec = arr["decision"][kinds == mg.KIND_MOVE]
    assert list(dec) == list(range(len(dec)))
    assert np.all(np.isnan(arr["u"][kinds != mg.KIND_MOVE]))
    # life 1 ran from t=31 to its loss at 150, life 2 is capped at 200 steps
    ends = arr["end"]
    assert int((ends == 1).sum()) == 2 and int((ends == 2).sum()) == 1
    assert int(((kinds != mg.KIND_START) & (lives == 2)).sum()) == 200
    assert env.was_real_done
    # the hand-off: every decision's live stack is the one rebuilt from frames
    rebuilt = mg.stacks_from_frames(arr)
    assert len(seen) == len(rebuilt) == len(dec)
    for d, s in seen:
        assert np.array_equal(s, rebuilt[d])
    assert states == []


def test_stacks_from_frames_rule():
    f = [np.full((3, 84, 84), i, np.uint8) for i in range(6)]
    arr = {"kind": np.array([0, 1, 2, 2, 0, 2], np.int8),
           "decision": np.array([-1, -1, 0, 1, -1, 2], np.int32),
           "frames": np.stack(f)}
    st = mg.stacks_from_frames(arr)
    # decision 0: start frame 0 tiled, frame 0 pushed, then frame 1 (the opening's)
    assert [int(st[0][c * 3, 0, 0]) for c in range(4)] == [0, 0, 0, 1]
    assert [int(st[1][c * 3, 0, 0]) for c in range(4)] == [0, 0, 1, 2]
    # a new life re-tiles its start frame
    assert [int(st[2][c * 3, 0, 0]) for c in range(4)] == [4, 4, 4, 4]


# --------------------------------------------------------------------------
# the real game, no snapshot
# --------------------------------------------------------------------------
@contextlib.contextmanager
def wrappers_on():
    saved = muzero.USE_ATARI_WRAPPERS
    muzero.USE_ATARI_WRAPPERS = True
    try:
        yield
    finally:
        muzero.USE_ATARI_WRAPPERS = saved


def _noop_env(env):
    e = env
    while not isinstance(e, mg.NoopResetEnv):
        e = e.env
    return e


def test_probe_env_steps_as_build_env():
    with wrappers_on():
        ref = ablation.build_env(mg.ENV_ID, 4)
    env, tap, noop, skip = mg.make_env()
    rng = np.random.default_rng(5)
    try:
        for e, n in ((ref, _noop_env(ref)), (env, noop)):
            e.reset(seed=777)
            e.was_real_done = True
        with mg.fixed_noops(_noop_env(ref), 7), mg.fixed_noops(noop, 7):
            o1, _ = ref.reset()
            o2, _ = env.reset()
        assert np.array_equal(o1, o2)
        lost = 0
        for i in range(1500):
            a = int(rng.integers(0, mg.MOVES))
            r1, r2 = ref.step(a), env.step(a)
            assert np.array_equal(r1[0], r2[0]) and r1[1:4] == r2[1:4]
            if r1[2]:
                lost += 1
                if ref.was_real_done:
                    break
                o1, _ = ref.reset()
                o2, _ = env.reset()
                assert np.array_equal(o1, o2)
        assert lost >= 1
    finally:
        ref.close()
        env.close()


def _random_choose(seed):
    def choose(stack, j, decision):
        rng = np.random.default_rng((seed, j))
        m = int(rng.integers(0, mg.MOVES))
        if decision is None:
            return m, None, None, None
        return m, 0.5, np.full(mg.MOVES, 1 / 9, np.float32), 0.25
    return choose


def test_real_handoff_and_restored_state_reproduces_the_step():
    env, tap, noop, skip = mg.make_env()
    try:
        env.reset(seed=777)
        seen = {}
        with mg.fixed_noops(noop, 5):
            arr, states = mg.play_game(
                env, tap, skip, _random_choose(1), opening=80, cap=250,
                keep_states=True, on_decision=lambda s, d: seen.__setitem__(d, s))
        rebuilt = mg.stacks_from_frames(arr)
        assert seen and set(seen) == set(rebuilt)
        assert all(np.array_equal(seen[d], rebuilt[d]) for d in seen)
        assert [s[0] for s in states] == sorted(seen)
        # restore a held decision with its buffer and replay its recorded move
        ale = env.unwrapped.ale
        idx = {int(d): i for i, d in enumerate(arr["decision"]) if d >= 0}
        for d, blob, buf, lives in states[:: max(1, len(states) // 15)]:
            ale.restoreState(ale_py.ALEState(blob))
            skip._buffer[:] = buf
            env.lives = ale.lives()
            env.was_real_done = False
            i = idx[d]
            tap.take()
            obs, rew, term, trunc, _ = env.step(int(arr["move"][i]))
            assert np.array_equal(mg.frame_of(obs), arr["frames"][i])
            assert float(rew) == float(arr["reward"][i])
            got = tap.take()
            n = int(arr["ndigests"][i])
            assert [bytes(x) for x in arr["digests"][i][:n]] == got
    finally:
        env.close()


# --------------------------------------------------------------------------
# slow: against Run.play_episode on the snapshots
# --------------------------------------------------------------------------
def _play_episode_game(run, noops, seed):
    """Run.play_game's loop of play_episode(greedy=False) on the run's own env,
    np.random seeded: the decisions' (stack, move) in order."""
    ne = _noop_env(run.env)
    out = []
    np.random.seed(seed)
    with mg.fixed_noops(ne, noops):
        run.env.was_real_done = True
        while True:
            traj, _, _, term = run.play_episode(greedy=False, cap=mg.LIFE_CAP)
            out += [(s, int(a)) for s, a, _ in traj[:-1]]
            if not term or run.env.was_real_done:
                break
    return out


def _probe_game(run, noops, seed, t):
    manager = run.searcher.network_manager
    np.random.seed(seed)

    def choose(stack, j, decision):
        policy, _ = manager.initial_inference(stack)
        u = np.random.random_sample()
        return mg.sample(mg.tempered(policy, t), u), u, policy, t

    env, tap, noop, skip = mg.make_env()
    seen = []
    try:
        env.reset(seed=777)                 # the seeding reset the reference had
        env.was_real_done = True
        with mg.fixed_noops(noop, noops):
            arr, _ = mg.play_game(env, tap, skip, choose, opening=0,
                                  on_decision=lambda s, d: seen.append(s))
    finally:
        env.close()
    moves = [int(m) for m in arr["move"][arr["kind"] == mg.KIND_MOVE]]
    return list(zip(seen, moves))


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SNAPSHOTS, reason="needs O0's and OF's snapshots")
@pytest.mark.parametrize("player,iteration,t_expect",
                         [("O0", 10, 0.5), ("O0", 160, 0.125), ("OF", 80, 0.5)])
def test_probe_loop_equals_play_episode(player, iteration, t_expect):
    table = mg.rows(player)
    path = mg.snapshot_path(player, iteration)
    with mg.player(path, table[iteration]["env_steps"]) as (run, d):
        t = float(muzero.prior_acting_temperature(
            run.searcher.network_manager.training_steps))
        assert t == t_expect
        mg.temperature_gate(player, iteration, table, t, 0.5)
        run.env.reset(seed=777)
        ref = _play_episode_game(run, 11, 4242)
        got = _probe_game(run, 11, 4242, t)
    assert len(ref) == len(got) > 50
    for (s1, a1), (s2, a2) in zip(ref, got):
        assert a1 == a2
        assert np.array_equal(s1, s2)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_SNAPSHOTS, reason="needs O0's snapshots")
def test_stage_smoke_one_game(tmp_path):
    out = tmp_path / "smoke"
    rep = mg.stage_data(out, ["O0"], iterations=(160,), games=1, cap=200)
    name = mg.game_name("O0", 160, 0)
    meta = json.loads((out / f"{name}.json").read_text())
    assert meta["temperature"] == 0.125 and meta["play_row"] == 162
    assert meta["split"] == "fit" and meta["life_cap"] == 200
    assert not (out / f"{name}-states.npz").exists()
    with np.load(out / f"{name}.npz") as z:
        assert z["frames"].shape[1:] == (3, 84, 84)
        assert int((z["kind"] == mg.KIND_OPEN).sum()) == mg.OPENING
        assert int((z["kind"] == mg.KIND_MOVE).sum()) == meta["decisions"] > 0
        dec = z["prior"][z["kind"] == mg.KIND_MOVE]
        assert np.allclose(dec.sum(1), 1, atol=1e-5)
    assert rep["games"] == 1 and rep["distinct_share"] == 1.0
    # resume: a second call writes nothing new
    mtime = (out / f"{name}.npz").stat().st_mtime
    mg.stage_data(out, ["O0"], iterations=(160,), games=1, cap=200)
    assert (out / f"{name}.npz").stat().st_mtime == mtime
