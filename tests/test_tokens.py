"""Curriculum P2a, duration TOKENS (Kai, 2026-10-17). Pins: the defaults are
off and the arm's delta is exact; a token plays its action for its frames
(inner wrapper steps, summed reward, every inner frame in the stack, base
steps counted); the early long-token bias; the gate; frame-time value
targets; the per-edge discount DISCOUNT ** (frames / 4) in BOTH vectorized
search engines, bitwise against the sequential engine with mixed
durations; the exact probe on token snapshots -- and the regression: a
vocabulary of the base's six actions at 4 frames reproduces the plain arm
bit for bit (search, targets, the real-Pong actor, a whole training
step)."""
import hashlib
import inspect
import pathlib
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import muzero
from muzero import IMAGE_RESHAPE, initial_frame_stack, preprocess_frame, \
    push_frame
from tests.test_gumbelact8 import probe  # noqa: F401
from tests.test_prioract2 import restore_rngs  # noqa: F401
from tests.test_vtree import build_world as vtree_world
from tests.test_vtree import engine_per_tree, sequential_search
from tests.test_vtree_communal import build_world as communal_world
from tests.test_vtree_communal import (make_traj, run_both,
                                       vectorized_windows)

BASE_ARM = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal"
TOK_ARM = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-tokens"
BASE6 = tuple((a, 4) for a in range(6))
MIXED6 = ((0, 4), (1, 8), (2, 16), (3, 32), (4, 4), (5, 8))
TEST_DISCOUNT = 0.99        # the parity worlds' search discount


# --------------------------------------------------------------------------
# defaults, the arm, the helpers
# --------------------------------------------------------------------------
def test_defaults_are_off():
    assert muzero.TOKEN_VOCAB is None
    assert muzero.TOKEN_LONG_BIAS == 0.0
    assert muzero.TOKEN_BASE_FRAMES == 4
    assert muzero.token_edge_discounts() is None
    assert muzero.token_frames(5) == 4
    assert muzero.token_long_bias(0) == 0.0


def test_arm_delta_is_exactly_the_tokens():
    base, tok = ablation.ARMS[BASE_ARM], ablation.ARMS[TOK_ARM]
    delta = {k for k in set(base) | set(tok) if base.get(k) != tok.get(k)}
    assert delta == {"TOKEN_VOCAB", "TOKEN_LONG_BIAS", "TOKEN_LONG_BIAS_END",
                     "_doc"}
    vocab = tok["TOKEN_VOCAB"]
    assert len(vocab) == 12 and len(set(vocab)) == 12
    assert {a for a, _ in vocab} == {0, 2, 3}           # NOOP, UP, DOWN
    assert {f for _, f in vocab} == {4, 8, 16, 32}
    assert vocab[0] == (0, 4)       # the placeholder action 0 is NOOP x4
    assert tok["TOKEN_LONG_BIAS"] == 0.5 and tok["TOKEN_LONG_BIAS_END"] == 0.25
    assert tok["_launch"] == {"ray": True, "cpu_actor": True}


def test_edge_discounts_are_per_frame(monkeypatch):
    monkeypatch.setattr(muzero, "DISCOUNT", 0.997 ** 4)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (2, 8), (3, 32)))
    d = muzero.token_edge_discounts()
    assert d[0] == muzero.DISCOUNT                 # exactly, not approx
    assert d[1] == pytest.approx(0.997 ** 8, rel=1e-14)
    assert d[2] == pytest.approx(0.997 ** 32, rel=1e-14)
    assert [muzero.token_frames(i) for i in range(3)] == [4, 8, 32]


def test_long_bias_falls_linearly_to_zero(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", BASE6)
    monkeypatch.setattr(muzero, "TOKEN_LONG_BIAS", 0.5)
    monkeypatch.setattr(muzero, "TOKEN_LONG_BIAS_END", 0.25)
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", 1000)
    assert [muzero.token_long_bias(s) for s in (0, 125, 249, 250, 999)] == \
        pytest.approx([0.5, 0.25, 0.002, 0.0, 0.0])
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", None)
    assert muzero.token_long_bias(0) == 0.0         # no tokens -> no bias


def test_token_actions_sizes_and_restores_the_action_space(monkeypatch):
    monkeypatch.setattr(ablation, "ACTIONS", 6)
    with ablation.arm_config(TOK_ARM):
        with ablation.token_actions():
            assert ablation.ACTIONS == 12
        assert ablation.ACTIONS == 6
    with ablation.arm_config(BASE_ARM), ablation.token_actions():
        assert ablation.ACTIONS == 6


def test_run_one_wires_the_gate_and_the_action_space():
    src = inspect.getsource(ablation.run_one)
    assert "token_gate(spec, use_ray, cpu_actor, ray_workers)" in src
    assert "token_actions()" in src
    assert src.index("token_actions()") < src.index("token_gate(")


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
@pytest.fixture
def tok_arm(monkeypatch):
    # Pin every arm key FIRST: monkeypatch's undo runs after arm_config's
    # restore, so a key a test monkeypatches inside the arm would otherwise
    # be "restored" to the arm's value and leak into later tests.
    for k in ablation.ARMS[TOK_ARM]:
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ENV_NAME", "FRAMESKIP", "ACTIONS"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    ablation.ENV_NAME, ablation.FRAMESKIP = "Pong-v4", 4
    with ablation.arm_config(TOK_ARM) as spec:
        yield spec


def test_gate_passes_the_arm_and_ignores_plain_arms(tok_arm, monkeypatch):
    ablation.token_gate(tok_arm, True, True, 0)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", None)
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", True)
    ablation.token_gate(tok_arm, False, False, 2)    # no tokens: no checks


@pytest.mark.parametrize("change,match", [
    ({"USE_SEARCH_ACTING": True}, "tested path"),
    ({"REFRESH_VTREE": False}, "tested path"),
    ({"VALUE_TARGET_MODE": "mixed"}, "tested path"),
    ({"USE_GUMBEL_SEARCH": True}, "tested path"),
    ({"BIRTH_SEARCH": False}, "tested path"),
    ({"USE_TARGET_MODEL": False}, "tested path"),
    ({"ASYNC_LOOP": True}, "tested path"),
    ({"REFRESH_OFFLOAD": True}, "tested path"),
    ({"REFRESH_SATURATE": True}, "tested path"),
    ({"USE_ATARI_WRAPPERS": False}, "tested path"),
    ({"ACTOR_GUMBEL_SEARCH": True}, "tested path"),
    ({"TOKEN_VOCAB": ((0, 4), (2, 6))}, "positive multiple"),
    ({"TOKEN_VOCAB": ((0, 4), (7, 8))}, "outside"),
    ({"TOKEN_VOCAB": ((0, 4), (0, 4))}, "duplicate"),
])
def test_gate_refusals(tok_arm, monkeypatch, change, match):
    for k, v in change.items():
        monkeypatch.setattr(muzero, k, v)
    with pytest.raises(SystemExit, match=match):
        ablation.token_gate(tok_arm, True, True, 0)


def test_gate_refuses_other_launch_modes(tok_arm, monkeypatch):
    for args in ((False, False, 0), (True, False, 0), (True, True, 2)):
        with pytest.raises(SystemExit, match="--ray --cpu-actor"):
            ablation.token_gate(tok_arm, *args)
    monkeypatch.setattr(ablation, "FRAMESKIP", 30)
    with pytest.raises(SystemExit, match="frameskip 4"):
        ablation.token_gate(tok_arm, True, True, 0)


# --------------------------------------------------------------------------
# frame-time value targets
# --------------------------------------------------------------------------
def _dtraj(actions, rewards):
    return [{"action": a, "reward": r} for a, r in zip(actions, rewards)]


def test_base_tokens_reproduce_the_plain_targets(monkeypatch):
    rng = np.random.default_rng(3)
    for _ in range(25):
        L = int(rng.integers(2, 30))
        tr = _dtraj(rng.integers(0, 6, L).tolist(),
                    rng.choice([-1.0, 0.0, 0.0, 1.0], L).tolist())
        for horizon in (1, 2, 5):
            for idx in range(L):
                monkeypatch.setattr(muzero, "TOKEN_VOCAB", None)
                plain = muzero.paper_value_parts(tr, idx, horizon, 0.997 ** 4)
                monkeypatch.setattr(muzero, "TOKEN_VOCAB", BASE6)
                tok = muzero.paper_value_parts(tr, idx, horizon, 0.997 ** 4)
                assert plain == tok            # value, g, tail: bit for bit


def test_mixed_tokens_measure_the_horizon_in_frames(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (1, 8), (2, 16),
                                                (3, 32)))
    D = 0.9
    # reward k is the reward for ENTERING position k (action k-1's result)
    tr = _dtraj([3, 0, 1, 2, 0, 0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # idx 0, horizon 5 (20 frames): one 32-frame token already covers it
    assert muzero.paper_value_parts(tr, 0, 5, D) == (1.0, D ** (32 / 4), 1)
    # idx 1: 4 + 8 + 16 = 28 >= 20 frames after three tokens
    v = 0.0 + 1.0 * 2.0
    g = 1.0 * D ** (4 / 4)
    v += g * 3.0
    g *= D ** (8 / 4)
    v += g * 4.0
    g *= D ** (16 / 4)
    assert muzero.paper_value_parts(tr, 1, 5, D) == (v, g, 4)
    # the 4.3 age-shrunk horizon 1 is 4 frames: one 4-frame token
    assert muzero.paper_value_parts(tr, 1, 1, D) == (2.0, D, 2)
    # the trajectory's end caps it
    assert muzero.paper_value_parts(tr, 4, 5, D) == (5.0, D, 5)
    assert muzero.paper_value_parts(tr, 5, 5, D) == (0.0, 1.0, 5)
    # paper_value_target bootstraps the stored value at that tail
    rec = [dict(t, root_value=10.0 * i) for i, t in enumerate(tr)]
    assert muzero.paper_value_target(rec, 0, 5, D) == 1.0 + D ** 8 * 10.0


# --------------------------------------------------------------------------
# per-edge discounts in the vectorized engines
# --------------------------------------------------------------------------
@pytest.fixture
def token_discount(monkeypatch):
    monkeypatch.setattr(muzero, "DISCOUNT", TEST_DISCOUNT)


def test_fresh_root_engine_base_tokens_are_the_scalar_path(monkeypatch,
                                                           token_discount):
    helper, engine, obs = vtree_world(monkeypatch)
    kw = dict(num_simulations=15, c_puct=1.25, discount=TEST_DISCOUNT,
              max_depth=12, root_noise=False)
    plain = engine_per_tree(engine, obs, **kw)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", BASE6)
    tok = engine_per_tree(engine, obs, **kw)
    assert plain[0].tolist() == tok[0].tolist()
    assert plain[1].tolist() == tok[1].tolist()        # bitwise


@pytest.mark.parametrize("sims,max_depth,horizon", [
    (12, 12, 3), (12, 2, 3), (20, 12, 100), (30, 30, 5),
    # audit r1 A-2: these two reach the range update (upd = r + d * q)
    # where a scalar-discount mutant changes the counts
    (40, 30, 3), (60, 12, 3)])
def test_fresh_root_engine_mixed_durations_match_sequential(
        monkeypatch, token_discount, sims, max_depth, horizon):
    helper, engine, obs = vtree_world(monkeypatch)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", MIXED6)
    monkeypatch.setattr(muzero, "PREFIX_LSTM_HORIZON", horizon)
    seq = [sequential_search(helper, obs[t], sims, max_depth=max_depth,
                             discount=TEST_DISCOUNT)
           for t in range(len(obs))]
    counts, vals = engine_per_tree(engine, obs, num_simulations=sims,
                                   c_puct=1.25, discount=TEST_DISCOUNT,
                                   max_depth=max_depth, root_noise=False)
    assert counts.tolist() == [c.tolist() for c, _ in seq]
    assert np.allclose(vals, [v for _, v in seq], rtol=0, atol=1e-12)


def test_fresh_root_engine_mixed_durations_change_the_search(monkeypatch,
                                                             token_discount):
    """Negative control: the per-edge discount is live."""
    helper, engine, obs = vtree_world(monkeypatch)
    kw = dict(num_simulations=20, c_puct=1.25, discount=TEST_DISCOUNT,
              root_noise=False)
    plain = engine_per_tree(engine, obs, **kw)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", MIXED6)
    mixed = engine_per_tree(engine, obs, **kw)
    assert not np.allclose(plain[1], mixed[1], rtol=0, atol=1e-9)


def test_token_search_refuses_a_foreign_discount(monkeypatch):
    helper, engine, obs = vtree_world(monkeypatch)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", MIXED6)      # DISCOUNT 0.997
    with pytest.raises(AssertionError, match="muzero.DISCOUNT"):
        engine.search(obs[:1], num_simulations=2, discount=TEST_DISCOUNT,
                      root_noise=False)


def test_communal_engine_base_tokens_are_the_scalar_path(monkeypatch,
                                                         token_discount):
    helper, engine = communal_world(monkeypatch, noise=True, horizon=5,
                                    max_depth=50)
    traj = make_traj(70, 40, True)
    windows = [[0, 3, 5, 9, 10, 14], [15, 16, 21, 27, 28, 31]]
    plain, _ = vectorized_windows(engine, traj, windows, 30)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", BASE6)
    tok, _ = vectorized_windows(engine, traj, windows, 30)
    assert set(plain) == set(tok)
    for k in plain:
        pp, pv, ph, praw = plain[k]
        tp, tv, th, traw = tok[k]
        assert pp.tolist() == tp.tolist() and pv == tv and ph == th
        assert praw.tolist() == traw.tolist()


@pytest.mark.parametrize("length,terminated,windows,sims,world", [
    (8, True, [[3]], 10, {}),
    (8, False, [[7]], 10, {}),
    (10, False, [[6, 7, 8, 9]], 12, {}),
    (14, True, [list(range(3, 11))], 16, {}),
    (40, True, [[0, 3, 5, 9, 10, 14], [15, 16, 21, 27, 28, 31]], 30,
     dict(noise=True, horizon=5, max_depth=50)),
])
def test_communal_engine_mixed_durations_match_sequential(
        monkeypatch, token_discount, length, terminated, windows, sims,
        world):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", MIXED6)
    traj = make_traj(2, length, terminated)
    run_both(monkeypatch, traj, windows, sims, **world)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_communal_cuda_graph_waves_match_eager_with_tokens(monkeypatch,
                                                           token_discount):
    """Audit r1 A-3: on CUDA the selection wave body is a captured graph
    reading f["disc"]; with mixed durations it must equal the eager waves."""
    import muzero_vtree
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", MIXED6)
    traj = make_traj(70, 40, True)
    windows = [[0, 3, 5, 9, 10, 14], [15, 16, 21, 27, 28, 31]]
    out = []
    for graphs in (False, True):
        monkeypatch.setattr(muzero_vtree, "USE_CUDA_GRAPHS", graphs)
        helper, engine = communal_world(monkeypatch, noise=True, horizon=5,
                                        max_depth=50)
        cuda = torch.device("cuda")
        for n in engine.nets.values():
            n.to(cuda)
        engine.device = cuda
        res, _ = vectorized_windows(engine, traj, windows, 30)
        out.append({k: (v[0].tolist(), v[1], v[3].tolist())
                    for k, v in res.items()})
    assert out[0] == out[1]


def test_communal_chain_seed_folds_each_in_edge(monkeypatch, token_discount):
    helper, engine = communal_world(monkeypatch)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", MIXED6)
    traj = make_traj(3, 9, True)
    lat, pol, val = engine.evaluate(traj["obs"])
    forest = engine.build([{"latents": lat, "policies": pol, "values": val,
                            "actions": traj["actions"],
                            "rewards": traj["rewards"], "terminated": True}])
    edge = muzero.token_edge_discounts()
    expect, bv = [None] * 9, 0.0                 # terminated: leaf value 0
    for k in range(8, -1, -1):
        expect[k] = bv
        d = TEST_DISCOUNT if k == 0 else edge[traj["actions"][k - 1]]
        bv = traj["rewards"][k] + d * bv
    assert forest.f["wsum"][:9].tolist() == expect
    assert forest.f["disc"][1:9].tolist() == [edge[a]
                                              for a in traj["actions"][:-1]]


# --------------------------------------------------------------------------
# playing tokens
# --------------------------------------------------------------------------
class _Steps(gym.Env):
    """Step t pays reward(t); terminates at step `end` (None: never). The
    observation encodes the step count."""
    observation_space = gym.spaces.Box(0, 255, (8, 8, 3), np.uint8)
    action_space = gym.spaces.Discrete(6)

    def __init__(self, reward=lambda t: 0.0, end=None):
        self.reward, self.end, self.t, self.actions = reward, end, 0, []

    def frame(self, t):
        return np.full((8, 8, 3), (10 * t) % 251, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        self.t = 0
        return self.frame(0), {}

    def step(self, action):
        self.actions.append(int(action))
        self.t += 1
        term = self.end is not None and self.t >= self.end
        return self.frame(self.t), float(self.reward(self.t)), term, False, {}


def test_play_token_holds_the_action_and_stacks_every_inner_frame(
        monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (2, 16)))
    env = _Steps(reward=lambda t: float(t))
    obs, _ = env.reset()
    stack = initial_frame_stack(preprocess_frame(obs, IMAGE_RESHAPE))
    obs, r, term, trunc, out, n = ablation.play_token(env, 1, stack)
    assert env.actions == [2, 2, 2, 2] and n == 4 and not term
    assert r == 1.0 + 2.0 + 3.0 + 4.0
    expect = stack
    for t in (1, 2, 3):      # the 4th frame is pushed at the next decision
        expect = push_frame(expect, preprocess_frame(env.frame(t),
                                                     IMAGE_RESHAPE))
    assert np.array_equal(out, expect)
    assert np.array_equal(obs, env.frame(4))


def test_play_token_stops_at_a_terminal(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((3, 32),))
    env = _Steps(reward=lambda t: 1.0, end=3)
    obs, _ = env.reset()
    stack = initial_frame_stack(preprocess_frame(obs, IMAGE_RESHAPE))
    _, r, term, _, _, n = ablation.play_token(env, 0, stack)
    assert env.actions == [3, 3, 3] and term and n == 3 and r == 3.0


class _TokNM:
    def __init__(self, probs, steps=0):
        self.probs, self.training_steps = np.asarray(probs), steps

    def initial_inference(self, stack):
        return self.probs.copy(), 0.0


def _token_shell(env, probs, steps=0):
    run = object.__new__(ablation.Run)
    run.env = env
    run.searcher = SimpleNamespace(network_manager=_TokNM(probs, steps))
    run.env_steps = 0
    return run


@pytest.fixture
def prior_tokens(monkeypatch):
    """Policy-only acting, no paper path, a 4-token vocabulary."""
    for k, v in (("USE_SEARCH_ACTING", False), ("PAPER_TARGETS", False),
                 ("EVAL_PRIOR_ONLY", False), ("PRIOR_ACT_STAGED_TEMP", False),
                 ("ACTING_TEMPERATURE", 1.0), ("PRIOR_ACT_TEMP_SCALE", 1.0),
                 ("REWARD_CLIP", 400),
                 ("TOKEN_VOCAB", ((0, 4), (2, 8), (3, 16), (0, 32)))):
        monkeypatch.setattr(muzero, k, v)
    monkeypatch.setattr(ablation, "ACTIONS", 4)


def test_play_episode_plays_tokens_and_counts_base_steps(prior_tokens):
    env = _Steps(reward=lambda t: 1.0 if t % 3 == 0 else 0.0)
    run = _token_shell(env, [0.1, 0.1, 0.7, 0.1])        # greedy: token 2
    traj, total, steps, term = run.play_episode(greedy=True, cap=10)
    # token 2 = DOWN x16 = 4 base steps: 0 -> 4 -> 8 -> 12 (cap 10)
    assert len(traj) == 4 and steps == 12 and not term
    assert env.actions == [3] * 12
    assert [r[1] for r in traj[:-1]] == [2, 2, 2]
    # each record's reward is the reward for entering it: the previous
    # token's summed inner rewards (steps 3 | 6 | 9 and 12 pay)
    assert [r[2] for r in traj] == [0.0, 1.0, 1.0, 2.0]
    assert total == 4.0
    assert run.env_steps == 0                       # greedy: never counted
    assert env.was_real_done is True                # capped -> real reset next


class _RecNM(_TokNM):
    def __init__(self, probs):
        super().__init__(probs)
        self.seen = []

    def initial_inference(self, stack):
        self.seen.append(np.array(stack, copy=True))
        return super().initial_inference(stack)


def test_records_keep_the_stack_the_decision_was_made_on(prior_tokens):
    """Audit r1 A-1: each record's observation is the stack the policy
    saw at that decision (not the stack after the token's inner frames),
    and that stack is always the last four BASE steps' frames, whatever
    tokens were played -- the decision-independent observation."""
    np.random.seed(1)
    env = _Steps()
    run = _token_shell(env, [0.25] * 4)
    nm = run.searcher.network_manager = _RecNM([0.25] * 4)
    traj, _, steps, _ = run.play_episode(greedy=False, cap=40)
    assert len(nm.seen) == len(traj) - 1
    assert len({muzero.token_frames(r[1]) for r in traj[:-1]}) > 1   # mixed
    base = 0
    for i, rec in enumerate(traj[:-1]):
        assert np.array_equal(rec[0], nm.seen[i]), i
        expect = np.concatenate([
            preprocess_frame(env.frame(max(0, t)), IMAGE_RESHAPE)
            for t in range(base - 3, base + 1)], axis=0)
        assert np.array_equal(rec[0], expect), i
        base += muzero.token_frames(rec[1]) // 4
    assert base == steps
    final = np.concatenate([preprocess_frame(env.frame(max(0, t)),
                                             IMAGE_RESHAPE)
                            for t in range(base - 3, base + 1)], axis=0)
    assert np.array_equal(traj[-1][0], final)


def test_self_play_counts_base_steps_as_env_steps(prior_tokens):
    np.random.seed(0)
    env = _Steps()
    run = _token_shell(env, [0.25, 0.25, 0.25, 0.25])
    traj, _, steps, _ = run.play_episode(greedy=False, cap=40)
    frames = sum(muzero.token_frames(r[1]) for r in traj[:-1])
    assert steps == frames // 4 == len(env.actions) == run.env_steps
    assert len(traj) - 1 < steps          # fewer decisions than base steps


def test_long_token_bias_reweights_prior_sampling(prior_tokens, monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_LONG_BIAS", 0.5)
    monkeypatch.setattr(muzero, "TOKEN_LONG_BIAS_END", 0.25)
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", 1000)
    seen = []

    def choice(n, p=None):
        seen.append(np.asarray(p, dtype=np.float64))
        return 0

    monkeypatch.setattr(np.random, "choice", choice)
    uniform = [0.25] * 4
    _token_shell(_Steps(), uniform, steps=0).play_episode(greedy=False, cap=1)
    w = np.array([1.0, 2.0 ** 0.5, 2.0, 8.0 ** 0.5])
    assert np.allclose(seen[-1], w / w.sum(), rtol=0, atol=1e-12)
    _token_shell(_Steps(), uniform, steps=250).play_episode(greedy=False,
                                                            cap=1)
    assert np.allclose(seen[-1], 0.25, rtol=0, atol=1e-12)   # bias gone


# --------------------------------------------------------------------------
# the regression: base-length tokens ARE the plain arm
# --------------------------------------------------------------------------
def _pin_globals(monkeypatch, keys):
    for k in keys:
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ACTIONS", "EPISODE_CAP", "ENV_NAME", "FRAMESKIP", "DEVICE"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))


@pytest.mark.slow
def test_base_tokens_reproduce_the_plain_actor_on_real_pong(monkeypatch,
                                                            restore_rngs):  # noqa: F811
    """The real Pong-v4 actor (self-play + eval) with TOKEN_VOCAB = the six
    ALE actions x 4 frames: actions, rewards, stacks, lengths, returns and
    the RNG state equal the plain arm's."""
    import ray_actors
    base = {k: v for k, v in ablation.ARMS[BASE_ARM].items()
            if not k.startswith("_")}
    schedule = {"TEMPERATURE_TOTAL_STEPS": 180000,
                "OFFPOLICY_DECAY_STEPS": 54000}
    _pin_globals(monkeypatch, list(base) + list(schedule) + ["TOKEN_VOCAB"])
    torch.set_num_threads(2)

    def trace(overrides):
        actor = ray_actors.SelfPlayActor(overrides, 6, 5000, schedule,
                                         "Pong-v4", 4, 150, server=None,
                                         server_set="eval", force_cpu=True)
        log = [actor.eval_episodes([500000])]
        for s in (7, 8):
            t, term, n = actor.play(s)
            digest = hashlib.sha1(b"".join(r[0].tobytes() for r in t))
            log.append((term, n, [int(r[1]) for r in t],
                        [float(r[2]) for r in t], digest.hexdigest()))
        log.append((float(np.random.random()), float(torch.rand(1))))
        return log

    plain = trace(base)
    tok = trace({**base, "TOKEN_VOCAB": BASE6})
    assert plain == tok


@pytest.mark.slow
def test_base_tokens_reproduce_a_plain_training_step(monkeypatch,
                                                     restore_rngs):  # noqa: F811
    """A whole sequential slice -- play, birth search, three training steps,
    a communal refresh -- with base-length tokens equals the plain arm's:
    every net's weights, the stored targets and priorities, env steps."""
    base = {k: v for k, v in ablation.ARMS[BASE_ARM].items()
            if not k.startswith("_")}
    _pin_globals(monkeypatch, list(base) + [
        "TOKEN_VOCAB", "ACT_SEARCH_SIMS", "TRAINING_BATCHES",
        "TEMPERATURE_TOTAL_STEPS", "OFFPOLICY_DECAY_STEPS"])
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = "Pong-v4", 4, 6
    ablation.DEVICE = torch.device("cpu")
    torch.set_num_threads(2)

    def slice_(vocab):
        with ablation.arm_config(BASE_ARM), \
                ablation._run_length_globals(2, 3):
            muzero.TOKEN_VOCAB = vocab
            muzero.ACT_SEARCH_SIMS = 6
            muzero.TRAINING_BATCHES = 8
            with ablation.token_actions():
                run = ablation.Run(BASE_ARM, 0, ablation.ARMS[BASE_ARM])
                try:
                    for _ in range(2):
                        traj, _, _, term = run.play_episode(cap=40)
                        run.replay.append(run.reanalyse(traj, term))
                    ablation.train_block(run, 3)
                    run.refresh_targets_vtree(1)
                    nets = {n: {k: v.clone() for k, v in
                                getattr(run.trainer, n).state_dict().items()}
                            for n in ablation.NET_NAMES}
                    stored = [[(t["action"], t["reward"], t["root_value"],
                                t["priority"], t["target_policy"].tolist())
                               for t in tr] for tr in run.replay]
                    return nets, stored, run.env_steps, float(torch.rand(1))
                finally:
                    run.env.close()

    plain = slice_(None)
    tok = slice_(BASE6)
    assert plain[1:] == tok[1:]
    for n in ablation.NET_NAMES:
        for k, v in plain[0][n].items():
            assert torch.equal(v, tok[0][n][k]), (n, k)


@pytest.mark.slow
def test_token_actor_plays_its_vocabulary_on_real_pong(monkeypatch,
                                                       restore_rngs):  # noqa: F811
    """The arm's overrides reach the actor (TOKEN_VOCAB via the muzero
    override transport, ACTIONS = 12): tokens in 0..11, base steps =
    frames / 4, more base steps than decisions."""
    import ray_actors
    tok = {k: v for k, v in ablation.ARMS[TOK_ARM].items()
           if not k.startswith("_")}
    schedule = {"TEMPERATURE_TOTAL_STEPS": 180000,
                "OFFPOLICY_DECAY_STEPS": 54000}
    _pin_globals(monkeypatch, list(tok) + list(schedule))
    torch.set_num_threads(2)
    actor = ray_actors.SelfPlayActor(tok, 12, 0, schedule, "Pong-v4", 4, 200,
                                     server=None, force_cpu=True)
    assert actor.get_constant("TOKEN_VOCAB") == tok["TOKEN_VOCAB"]
    traj, term, steps = actor.play(3)
    acts = [int(r[1]) for r in traj[:-1]]
    assert set(acts) <= set(range(12)) and len(set(acts)) > 3
    frames = sum(tok["TOKEN_VOCAB"][a][1] for a in acts)
    assert steps == frames // 4 or term          # a terminal can cut the last
    assert steps > len(acts)


# --------------------------------------------------------------------------
# the exact probe on token snapshots
# --------------------------------------------------------------------------
def _token_snapshot(tmp_path, monkeypatch, arm=TOK_ARM):
    """A random-init snapshot in the weights-file format, its spec from
    `arm` (the run folder layout the probe reads)."""
    _pin_globals(monkeypatch, ["TOKEN_VOCAB"])
    ablation.ENV_NAME, ablation.FRAMESKIP, ablation.ACTIONS = "Pong-v4", 4, 6
    with ablation.arm_config(arm) as spec, ablation.token_actions():
        torch.manual_seed(1)
        nets = ablation.build_networks(spec)
    d = tmp_path / "pong-x" / "snapshots"
    d.mkdir(parents=True)
    path = d / "weights-iter005.pth"
    torch.save({"arm": arm, "seed": 1, "spec": dict(spec),
                "training_steps": 0,
                **{n: nets[n].state_dict() for n in ablation.NET_NAMES}},
               path)
    return path


@pytest.mark.slow
def test_probe_plays_token_snapshots(probe, monkeypatch, tmp_path):  # noqa: F811
    path = _token_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(probe, "MAX_NOOP", 2)
    monkeypatch.setattr(probe, "HERE", tmp_path)
    actions_before = ablation.ACTIONS
    probe.main([str(path)])
    assert muzero.TOKEN_VOCAB is None and ablation.ACTIONS == actions_before
    import json
    rec = json.loads((tmp_path / "probe_exhaustive_eval-pong-x-noops2.jsonl")
                     .read_text().splitlines()[-1])
    assert rec["tokens"] == 12 and rec["iteration"] == 5
    for g in rec["games"].values():
        ret, steps, term, decisions = g
        assert 0 < decisions <= steps
    assert 4.0 <= rec["frames_per_decision"] <= 32.0


@pytest.mark.slow
def test_probe_refuses_mixed_vocabularies(probe, monkeypatch, tmp_path):  # noqa: F811
    tok = _token_snapshot(tmp_path / "a", monkeypatch)
    plain = _token_snapshot(tmp_path / "b", monkeypatch, arm=BASE_ARM)
    monkeypatch.setattr(probe, "MAX_NOOP", 1)
    monkeypatch.setattr(probe, "HERE", tmp_path)
    with pytest.raises(SystemExit, match="TOKEN_VOCAB"):
        probe.main([str(tok), str(plain)])


# --------------------------------------------------------------------------
# the supervisor
# --------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "benchmarks" / "supervisors" / "supervise_tokens.py"


@pytest.fixture
def sup():
    import importlib.util
    spec = importlib.util.spec_from_file_location("supervise_tokens",
                                                  SUPERVISOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_supervisor_recipe_is_the_twins_with_the_token_arm(sup):
    assert sup.ARM == TOK_ARM and sup.SEEDS == {1: "2026-10-17"}
    assert sup.command(1)[1:] == [
        "experiments/ablation.py", "--arms", TOK_ARM, "--seed", "1",
        "--iterations", "90", "--train-steps", "2000", "--eval-every", "5",
        "--env", "Pong-v4", "--frameskip", "4", "--episode-cap", "3000",
        "--episodes-per-iter", "1", "--checkpoint-every", "2",
        "--snapshot-weights", "--ray", "--cpu-actor"]
    rel, base, jsonl, ckpt, _ = sup.paths(1)
    assert rel == "benchmarks/results/pong-tokens-s1-2026-10-17"
    assert jsonl.endswith("ablation-pong-skip4.jsonl")
    assert ckpt.endswith(f"resume-pong-skip4-{TOK_ARM}-seed1.ckpt")
    assert pathlib.Path(sup.REPO) == ROOT          # runs from its own tree


def test_supervisor_pins_what_the_run_and_preflight_use(sup):
    for rel in sup.AUDITED:
        assert (ROOT / rel).exists(), rel
    assert set(sup.SUITES) <= set(sup.AUDITED)
    for rel in ("muzero.py", "muzero_vtree.py", "networks.py",
                "experiments/ablation.py", "experiments/ray_actors.py",
                "benchmarks/probes/probe_exhaustive_eval.py",
                "benchmarks/supervisors/supervise_tokens.py",
                "tests/test_tokens.py"):
        assert rel in sup.AUDITED
    # every test module test_tokens imports is pinned too
    src = (ROOT / "tests" / "test_tokens.py").read_text(encoding="utf-8")
    for mod in ("test_gumbelact8", "test_prioract2", "test_vtree",
                "test_vtree_communal"):
        assert f"from tests.{mod} import" in src
        assert f"tests/{mod}.py" in sup.AUDITED


def test_supervisor_manifest_catches_an_edit(sup, monkeypatch, tmp_path):
    import shutil
    for rel in sup.AUDITED:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "m.json"))
    import json
    (tmp_path / "m.json").write_text(json.dumps(sup.file_hashes(str(tmp_path))))
    assert sup.manifest_check() == (True, "")
    with open(tmp_path / "muzero.py", "a", encoding="utf-8") as f:
        f.write("\n# edit\n")
    ok, why = sup.manifest_check()
    assert not ok and "muzero.py" in why
    # CRLF vs LF does not matter
    p = tmp_path / "muzero.py"
    shutil.copyfile(ROOT / "muzero.py", p)
    p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
    assert sup.manifest_check() == (True, "")


def test_supervisor_write_manifest_refuses_a_dirty_tree(sup, monkeypatch,
                                                         tmp_path):
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "m.json"))
    monkeypatch.setattr(sup.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=" M muzero.py\n", stderr=""))
    assert sup.main(["--write-manifest"]) == 1
    assert not (tmp_path / "m.json").exists()


def test_supervisor_refuses_unregistered_seeds(sup, capsys):
    for argv in (["0"], ["2"], [], ["x"], ["1", "2"]):
        assert sup.main(argv) == 2
    assert "usage" in capsys.readouterr().out


def test_supervisor_finds_live_children_of_its_arm_and_seed(sup, monkeypatch):
    procs = [
        {"pid": 1, "cmdline": ["py", r"Y:\run\experiments\ablation.py",
                               "--arms", TOK_ARM, "--seed", "1"]},
        {"pid": 2, "cmdline": ["py", "experiments/ablation.py", "--arms",
                               TOK_ARM, "--seed", "0"]},
        {"pid": 3, "cmdline": ["py", "experiments/ablation.py", "--arms",
                               BASE_ARM, "--seed", "1"]},
        {"pid": 4, "cmdline": None}]
    import psutil
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [
        SimpleNamespace(info=p) for p in procs])
    assert sup.live_children(1) == [1]


def test_supervisor_pins_every_module_and_supervisor_the_suites_read(sup):
    """Audit r1 B-L1: every tests.* module a preflight suite imports, and
    every supervisor file a suite reads or execs, is in AUDITED."""
    import re
    for rel in sup.SUITES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        for mod in re.findall(r"from tests\.(\w+) import", src):
            assert f"tests/{mod}.py" in sup.AUDITED, (rel, mod)
        for name in re.findall(r"(supervise_\w+)\.py", src):
            assert f"benchmarks/supervisors/{name}.py" in sup.AUDITED, \
                (rel, name)


# --------------------------------------------------------------------------
# the replay cap in base steps; row fields (audit r1 B-M1 / B-L2)
# --------------------------------------------------------------------------
def _rec_traj(actions):
    return [{"action": a, "reward": 0.0} for a in actions] + [
        {"action": 0, "reward": 0.0}]


def test_replay_cap_counts_base_steps_under_tokens(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (2, 32)))
    monkeypatch.setattr(muzero, "MAX_REPLAY", 85)
    monkeypatch.setattr(muzero, "MAX_REPLAY_TRANSITIONS", 40)
    long_game = _rec_traj([1, 1, 1])          # 3 records + 1, 24 + 1 units
    assert ablation.replay_units(long_game) == 25
    replay = [_rec_traj([1, 1, 1]), _rec_traj([0] * 5), _rec_traj([1, 1])]
    ablation.evict_replay(replay)              # 25 + 6 + 17 = 48 > 40
    assert [ablation.replay_units(t) for t in replay] == [6, 17]
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", None)
    assert ablation.replay_units(long_game) == 4     # plain: records


def test_plain_eviction_is_unchanged(monkeypatch):
    monkeypatch.setattr(muzero, "MAX_REPLAY", 85)
    monkeypatch.setattr(muzero, "MAX_REPLAY_TRANSITIONS", 10)
    replay = [_rec_traj([1] * 5), _rec_traj([1] * 3), _rec_traj([1] * 4)]
    ablation.evict_replay(replay)              # records 6 + 4 + 5 = 15 > 10
    assert [len(t) for t in replay] == [4, 5]


def test_token_rows_carry_buffer_size_and_cumulative_decisions(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (2, 32)))
    run = object.__new__(ablation.Run)
    run.replay = [_rec_traj([1, 0]), _rec_traj([0])]
    for tr in run.replay:
        out = [dict(t, target_policy=np.array([0.5, 0.5]), root_value=0.0)
               for t in tr]
        run._note_tokens(out, [0.0] * len(out))
    errs = {}
    run._merge_row_stats(errs)
    assert errs["tok_decisions_total"] == 3
    assert errs["replay_trajs"] == 2 and errs["replay_records"] == 5
    assert errs["replay_base_steps"] == (1 + 8 + 1) + (1 + 1)
    assert errs["tok_counts"] == [2, 1]
    assert errs["tok_frames_per_decision"] == pytest.approx(40 / 3, abs=1e-3)
    errs2 = {}
    run._merge_row_stats(errs2)          # the window restarts, the total not
    assert errs2["tok_decisions_total"] == 3 and "tok_counts" not in errs2
    src = inspect.getsource(ablation.save_run_checkpoint) \
        if hasattr(ablation, "save_run_checkpoint") else \
        (ROOT / "experiments" / "ablation.py").read_text(encoding="utf-8")
    assert '"_tok_totals")' in src         # checkpointed with the diag stats


def test_plain_rows_gain_no_token_fields(monkeypatch):
    run = object.__new__(ablation.Run)
    run.replay = [_rec_traj([1, 0])]
    errs = {}
    run._merge_row_stats(errs)
    assert not any(k.startswith(("tok_", "replay_")) for k in errs)


# --------------------------------------------------------------------------
# audit r2: the BN window in base steps; mid-token terminal stacks; the
# token / behaviour diagnostics with real values; the arm's refresh mode
# --------------------------------------------------------------------------
def _obs_traj(actions, tag):
    """Records whose obs encode (trajectory tag, position)."""
    recs = [{"action": a, "reward": 0.0,
             "obs": np.full((1, 1, 1), 10 * tag + i, dtype=np.uint8)}
            for i, a in enumerate(actions)]
    recs.append({"action": 0, "reward": 0.0,
                 "obs": np.full((1, 1, 1), 10 * tag + len(actions),
                                dtype=np.uint8)})
    return recs


def test_bn_window_counts_base_steps_under_tokens(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (2, 32)))
    # oldest .. newest; each long game is 3 x 8 + 1 = 25 base steps
    replay = [_obs_traj([1, 1, 1], t) for t in range(4)]
    pool = ablation.bn_calibration_frames(replay, 40)
    # the two newest games (50 >= 40 base steps), all 8 of their states
    assert sorted(int(o[0, 0, 0]) for o in pool) == \
        [20, 21, 22, 23, 30, 31, 32, 33]
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", None)
    plain = ablation.bn_calibration_frames(replay, 6)
    # plain: records -- the newest two games' 8 records thinned to 6
    assert len(plain) == 6 and {int(o[0, 0, 0]) // 10 for o in plain} == {2, 3}


def test_a_terminal_mid_token_stacks_its_last_frame_once(prior_tokens):
    """Audit r2 A-LOW2: DOWN x16 (4 base steps) twice, the game ending at
    base step 6 inside the second token: the final record's stack is the
    last four base frames 3..6, the terminal frame pushed exactly once."""
    env = _Steps(end=6)
    run = _token_shell(env, [0.1, 0.1, 0.7, 0.1])        # greedy: token 2
    traj, total, steps, term = run.play_episode(greedy=True, cap=100)
    assert term and steps == 6 and env.actions == [3] * 6

    def frames(ts):
        return np.concatenate([preprocess_frame(env.frame(max(0, t)),
                                                IMAGE_RESHAPE) for t in ts],
                              axis=0)

    assert len(traj) == 3
    assert np.array_equal(traj[0][0], frames([0, 0, 0, 0]))
    assert np.array_equal(traj[1][0], frames([1, 2, 3, 4]))
    assert np.array_equal(traj[2][0], frames([3, 4, 5, 6]))


def test_token_payoff_is_z_minus_v_per_token(monkeypatch):
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", ((0, 4), (2, 32)))
    run = object.__new__(ablation.Run)
    run.replay = []
    out = [{"action": 0, "target_policy": np.array([0.9, 0.1]),
            "root_value": 1.0},
           {"action": 1, "target_policy": np.array([0.2, 0.8]),
            "root_value": -2.0},
           {"action": 0, "target_policy": np.array([0.5, 0.5]),
            "root_value": 0.5},
           {"action": 0, "target_policy": np.array([0.5, 0.5]),
            "root_value": 0.0}]                      # the final record
    run._note_tokens(out, [3.0, 1.0, 0.0, 9.0])
    errs = {}
    run._merge_row_stats(errs)
    # token 0: (3 - 1) and (0 - 0.5) -> mean 0.75; token 1: 1 - (-2) = 3
    assert errs["tok_payoff"] == [0.75, 3.0]
    assert errs["tok_counts"] == [2, 1]
    assert errs["tok_search_share"] == [round(1.6 / 3, 4), round(1.4 / 3, 4)]
    assert errs["tok_duration_play_share"] == {"4": round(2 / 3, 4),
                                               "32": round(1 / 3, 4)}
    # audit r3 A-LOW1: the search share by duration, decisions per game
    assert errs["tok_duration_search_share"] == {"4": round(1.6 / 3, 4),
                                                 "32": round(1.4 / 3, 4)}
    assert errs["tok_decisions_per_game"] == 3.0


def test_behaviour_entropy_includes_the_long_token_bias(monkeypatch):
    for k, v in (("TOKEN_VOCAB", ((0, 4), (0, 8), (0, 16), (0, 32))),
                 ("TOKEN_LONG_BIAS", 0.5), ("TOKEN_LONG_BIAS_END", 0.25),
                 ("TEMPERATURE_TOTAL_STEPS", 1000),
                 ("PRIOR_ACT_STAGED_TEMP", False),
                 ("ACTING_TEMPERATURE", 1.0), ("PRIOR_ACT_TEMP_SCALE", 1.0)):
        monkeypatch.setattr(muzero, k, v)
    monkeypatch.setattr(ablation, "ACTIONS", 4)
    run = object.__new__(ablation.Run)
    run._birth_priors = np.full((3, 4), 0.25)
    acted = [{"action": a, "target_policy": np.full(4, 0.25)}
             for a in (0, 3, 3)]
    run._note_behaviour(acted, born=0)           # b = 0.5 at step 0
    w = np.array([1.0, 2 ** 0.5, 2.0, 8 ** 0.5])
    p = w / w.sum()
    H = float(-(p * np.log(p)).sum())
    assert run._beh_window["sums"]["H_act"] / 3 == pytest.approx(H, abs=1e-9)
    run._beh_window = None
    run._note_behaviour(acted, born=250)         # the bias has ended
    assert run._beh_window["sums"]["H_act"] / 3 == pytest.approx(
        np.log(4), abs=1e-9)


LONG0 = ((0, 32), (1, 4), (2, 8), (3, 16), (4, 4), (5, 8))


@pytest.mark.parametrize("edge_visits", [False, True])
def test_communal_parity_with_a_long_slot0_and_edge_visits(
        monkeypatch, token_discount, edge_visits):
    """Audit r2 A-NIT2: the arm refreshes with COMMUNAL_EDGE_VISITS on; and
    a vocabulary whose slot 0 (the placeholder) is not a base token."""
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", LONG0)
    monkeypatch.setattr(muzero, "COMMUNAL_EDGE_VISITS", edge_visits)
    traj = make_traj(5, 24, False)
    run_both(monkeypatch, traj, [[0, 2, 3, 7, 8], [9, 12, 15, 20, 23]], 20,
             noise=True, horizon=5, max_depth=30)


def test_fresh_root_parity_with_a_long_slot0(monkeypatch, token_discount):
    helper, engine, obs = vtree_world(monkeypatch)
    monkeypatch.setattr(muzero, "TOKEN_VOCAB", LONG0)
    seq = [sequential_search(helper, obs[t], 40, max_depth=30,
                             discount=TEST_DISCOUNT)
           for t in range(len(obs))]
    counts, vals = engine_per_tree(engine, obs, num_simulations=40,
                                   c_puct=1.25, discount=TEST_DISCOUNT,
                                   max_depth=30, root_noise=False)
    assert counts.tolist() == [c.tolist() for c, _ in seq]
    assert np.allclose(vals, [v for _, v in seq], rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# audit r3: temperature then bias (acting and the behaviour log agree); the
# cumulative-decision counter round-trips through a checkpoint
# --------------------------------------------------------------------------
def _temp_then_bias(p, T, b, frames):
    q = np.asarray(p, dtype=np.float64) ** (1.0 / T)
    q = q / q.sum()
    q = q * np.array([(f / 4) ** b for f in frames])
    return q / q.sum()


def test_acting_applies_the_temperature_then_the_bias(prior_tokens,
                                                      monkeypatch):
    """Audit r3 A-LOW2: at the arm's T < 1 the order matters (bias first
    would give an effective exponent b / T)."""
    monkeypatch.setattr(muzero, "PRIOR_ACT_TEMP_SCALE", 0.5)        # T 0.5
    monkeypatch.setattr(muzero, "TOKEN_LONG_BIAS", 0.5)
    monkeypatch.setattr(muzero, "TOKEN_LONG_BIAS_END", 0.25)
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", 1000)
    seen = []
    monkeypatch.setattr(np.random, "choice",
                        lambda n, p=None: seen.append(np.asarray(p)) or 0)
    prior = [0.4, 0.3, 0.2, 0.1]
    _token_shell(_Steps(), prior, steps=0).play_episode(greedy=False, cap=1)
    frames = [f for _, f in muzero.TOKEN_VOCAB]
    want = _temp_then_bias(prior, 0.5, 0.5, frames)
    assert np.allclose(seen[-1], want, rtol=0, atol=1e-12)
    wrong = _temp_then_bias(np.asarray(prior) * np.array(
        [(f / 4) ** 0.5 for f in frames]), 0.5, 0.0, frames)
    assert not np.allclose(want, wrong, atol=1e-3)     # the order is visible


def test_behaviour_log_matches_the_sampled_distribution_at_low_T(
        monkeypatch):
    frames = (4, 8, 16, 32)
    for k, v in (("TOKEN_VOCAB", tuple((0, f) for f in frames)),
                 ("TOKEN_LONG_BIAS", 0.5), ("TOKEN_LONG_BIAS_END", 0.25),
                 ("TEMPERATURE_TOTAL_STEPS", 1000),
                 ("PRIOR_ACT_STAGED_TEMP", False),
                 ("ACTING_TEMPERATURE", 1.0), ("PRIOR_ACT_TEMP_SCALE", 0.5)):
        monkeypatch.setattr(muzero, k, v)
    monkeypatch.setattr(ablation, "ACTIONS", 4)
    run = object.__new__(ablation.Run)
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    run._birth_priors = np.tile(prior, (2, 1))
    acted = [{"action": 1, "target_policy": np.full(4, 0.25)}] * 2
    run._note_behaviour(acted, born=0)
    q = _temp_then_bias(prior, 0.5, 0.5, frames)
    H = float(-(q * np.log(q)).sum())
    assert run._beh_window["sums"]["H_act"] / 2 == pytest.approx(H, abs=1e-9)


def test_cumulative_decisions_survive_a_checkpoint(tmp_path):
    """Audit r3 A-LOW3: _tok_totals round-trips (save -> load)."""
    from tests.test_checkpoint_resume import EXPECT, META, make_run
    a = make_run()
    a.trainer.training_steps = META["training_steps"]
    a._tok_totals = {"tok_decisions_total": 4321}
    path = tmp_path / "resume.ckpt"
    ablation.save_run_checkpoint(a, path, META)
    b = make_run()
    ablation.load_run_checkpoint(b, path, EXPECT)
    assert b._tok_totals == {"tok_decisions_total": 4321}
    c = make_run()                         # a plain run writes no such key
    ablation.save_run_checkpoint(c, tmp_path / "plain.ckpt", META)
    d = make_run()
    ablation.load_run_checkpoint(d, tmp_path / "plain.ckpt", EXPECT)
    assert not hasattr(d, "_tok_totals")
