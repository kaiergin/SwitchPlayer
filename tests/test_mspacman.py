"""The real-time line's algorithm on Ms. Pac-Man (Kai, 2026-10-16). Pins:
whole-GAME evaluation (Run.play_game: every life of a multi-life game from a
real reset, the cap per life; a single-life game is exactly one
play_episode), its wiring into the sequential eval, the Ray eval actor and
the exact probe (PROBE_ENV), the two -mspacman arms (only the per-trajectory
knobs converted to per-game), the 9-action Gumbel-8 search, and the
supervisor (recipe, paths, the Gumbel seed-1 decision it applies, the wait,
the manifest)."""
import ast
import asyncio
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import muzero
from EpisodicLifeEnv import EpisodicLifeEnv
from tests.test_gumbelact8 import _snapshot, probe  # noqa: F401
from tests.test_prioract2 import restore_rngs  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUP = ROOT / "benchmarks" / "supervisors" / "supervise_mspacman.py"
PROBE = ROOT / "benchmarks" / "probes" / "probe_exhaustive_eval.py"
ENV_ID = "MsPacmanNoFrameskip-v4"
ARMS = {"effzero-repro-faithful-gumbelact8-birth-mspacman":
        "effzero-repro-faithful-gumbelact8-birth",
        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman":
        "effzero-repro-faithful-prioract-noredo-temp05-bnrecal"}


# --------------------------------------------------------------------------
# play_game on a scripted multi-life game (real EpisodicLifeEnv, real
# play_episode, a stub prior)
# --------------------------------------------------------------------------
class _Lives(gym.Env):
    """A scripted game: life k lasts plan[k] env steps (a respawn no-op
    included), each non-no-op step pays 1. ALE's semantics: lives() drops
    at each death, the last death is the game over (terminated)."""
    observation_space = gym.spaces.Box(0, 255, (8, 8, 3), np.uint8)
    action_space = gym.spaces.Discrete(9)

    def __init__(self, plan, end_lives=0):
        self.plan, self.end_lives = tuple(plan), end_lives
        self.resets, self.steps = [], 0
        self.ale = SimpleNamespace(lives=lambda: self._lives)
        self._lives, self._life, self._t = len(self.plan), 0, 0

    def _obs(self):
        return np.full((8, 8, 3), self.steps % 251, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        self.resets.append(seed)
        self._lives, self._life, self._t = len(self.plan), 0, 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        self._t += 1
        term = False
        if self._life < len(self.plan) and self._t >= self.plan[self._life]:
            self._life, self._t = self._life + 1, 0
            self._lives -= 1
            term = self._life == len(self.plan)
            if term:
                self._lives = self.end_lives
        return self._obs(), 0.0 if action == 0 else 1.0, term, False, {}


class _NM:
    training_steps = 0

    def initial_inference(self, stack):
        p = np.full(9, 0.05)
        p[2] = 0.6                                  # greedy: always action 2
        return p, 0.0


def _shell(plan, end_lives=0):
    run = object.__new__(ablation.Run)
    run.env = EpisodicLifeEnv(_Lives(plan, end_lives))
    run.searcher = SimpleNamespace(network_manager=_NM())
    run.env_steps = 0
    return run


@pytest.fixture
def prior_eval(monkeypatch):
    """The probe's policy-only arm (greedy play is the prior argmax), 9
    actions."""
    monkeypatch.setattr(ablation, "ACTIONS", 9)
    with ablation.arm_config("effzero-repro-faithful-prioract"):
        assert muzero.USE_SEARCH_ACTING is False
        yield


def test_play_game_plays_every_life_from_one_real_reset(prior_eval):
    run = _shell((5, 4, 3))
    ret, steps, term = run.play_game(cap=100)
    inner = run.env.env
    # life 1: 5 decisions; lives 2-3: the respawn no-op (EpisodicLifeEnv's
    # continuation reset) is the life's first env step, so 3 and 2.
    assert (ret, steps, term) == (10.0, 10, True)
    assert inner.resets == [None] and inner.steps == 12
    assert run.env.was_real_done is True and inner.ale.lives() == 0
    # the next game starts from a real reset, with all lives
    assert run.play_game(cap=100) == (10.0, 10, True)
    assert len(inner.resets) == 2
    assert run.env_steps == 0                       # eval is protocol-exempt


def test_play_game_starts_a_new_game_even_mid_match(prior_eval,
                                                    restore_rngs):  # noqa: F811
    run = _shell((5, 4, 3))
    run.play_episode(greedy=False, cap=100)         # a training LIFE
    assert run.env.was_real_done is False and run.env.env.ale.lives() == 2
    assert run.play_game(cap=100) == (10.0, 10, True)
    assert len(run.env.env.resets) == 2             # a real reset, not a resume


def test_a_capped_life_ends_the_game_and_reads_capped(prior_eval):
    run = _shell((50, 4, 3))
    assert run.play_game(cap=7) == (7.0, 7, False)
    assert run.env.was_real_done is True            # the next reset is real
    assert ablation.eval_summary([run.play_game(cap=7)])["eval_capped"] == 1


def test_single_life_game_is_exactly_one_play_episode(prior_eval):
    """Pong: one life per game -- play_game is play_episode, unchanged."""
    a, b = _shell((6,), end_lives=0), _shell((6,), end_lives=0)
    _, r, n, t = a.play_episode(greedy=True, cap=100)
    assert b.play_game(cap=100) == (r, n, t) == (6.0, 6, True)
    assert a.env.env.steps == b.env.env.steps == 6
    calls = []
    real = ablation.Run.play_episode

    def spy(self, greedy=False, cap=None):
        calls.append((greedy, cap))
        return real(self, greedy=greedy, cap=cap)

    c = _shell((6,))
    c.play_episode = spy.__get__(c)
    c.play_game(cap=40)
    assert calls == [(True, 40)]


def test_play_game_surface_fits_the_actor_shell():
    """The Ray actor calls Run.play_game on its object.__new__ shell, which
    has env / searcher / env_steps (+ the class's methods)."""
    tree = ast.parse((ROOT / "experiments" / "ablation.py")
                     .read_text(encoding="utf-8"))
    run_cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == "Run")
    fn = next(n for n in run_cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "play_game")
    attrs = {node.attr for node in ast.walk(fn)
             if isinstance(node, ast.Attribute)
             and isinstance(node.value, ast.Name) and node.value.id == "self"}
    assert attrs <= {"env", "play_episode"}


def test_sequential_evaluate_plays_games(prior_eval, monkeypatch):
    run = _shell((5, 4, 3))
    monkeypatch.setattr(ablation.Run, "evaluate_diagnostics",
                        lambda self, batches=8: {"diag": 1})
    row = ablation.Run.evaluate(run, episodes=2)
    assert row["return_mean"] == 10.0 and row["eval_len_mean"] == 10.0
    assert row["eval_capped"] == 0 and row["diag"] == 1
    assert len(run.env.env.resets) == 2


# --------------------------------------------------------------------------
# the Ray eval actor (in-process, CPU, the real Ms. Pac-Man env)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_eval_actor_plays_whole_seeded_games(monkeypatch, restore_rngs):  # noqa: F811
    import ray_actors
    # policy-only acting keeps the training life below fast on the CPU; the
    # eval path is the same for both arms (prior argmax, EVAL_PRIOR_ONLY)
    arm = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal-mspacman"
    overrides = {k: v for k, v in ablation.ARMS[arm].items()
                 if not k.startswith("_")}
    schedule = {"TEMPERATURE_TOTAL_STEPS": 180000,
                "OFFPOLICY_DECAY_STEPS": 54000}
    for k in list(overrides) + list(schedule):
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ACTIONS", "EPISODE_CAP", "ENV_NAME", "FRAMESKIP"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    import torch
    torch.set_num_threads(2)
    actor = ray_actors.SelfPlayActor(overrides, 9, 5000, schedule, ENV_ID, 4,
                                     3000, server=None, server_set="eval",
                                     force_cpu=True)
    assert muzero.USE_SEARCH_ACTING is False      # greedy = the prior argmax
    seeds = []
    real_reset = actor.env.env.reset           # the stack under EpisodicLife

    def spy(**kw):
        seeds.append(kw.get("seed"))
        return real_reset(**kw)

    actor.env.env.reset = spy
    first = actor.eval_episodes([500000])
    assert len(first) == 1
    ret, steps, term = first[0]
    assert term is True and steps > 0
    assert actor.env.unwrapped.ale.lives() == 0          # played to game over
    assert seeds[0] == 500000                            # the seeded reset
    # A training life leaves the match mid-way; the next eval game still
    # starts from the seeded real reset and replays the same game.
    actor.play(7)
    assert actor.env.was_real_done is False
    assert actor.env.unwrapped.ale.lives() == 2
    assert actor.eval_episodes([500000]) == first


def _actor_shell(plan):
    import ray_actors
    actor = object.__new__(ray_actors.SelfPlayActor)
    run = _shell(plan)
    actor.env, actor.shell, actor._ablation = run.env, run, ablation
    actor.episode_cap = 100
    return actor


def test_actor_play_continues_the_match_with_one_respawn_noop(
        prior_eval, restore_rngs):  # noqa: F811
    """Audit r1 L1: mid-match, play() leaves the reset to play_episode --
    ONE respawn no-op per continued life (as eval and the baselines
    wrapper); the seeded reset runs only for a new game."""
    actor = _actor_shell((5, 4, 3))
    inner = actor.env.env
    got = [actor.play(s)[1:] for s in (11, 12, 13)]
    assert got == [(True, 5), (True, 3), (True, 2)]        # (terminated, steps)
    assert inner.resets == [11, None]                       # one game, seeded
    assert inner.steps == 5 + (1 + 3) + (1 + 2)             # one no-op per respawn
    assert actor.shell.env_steps == 10
    actor.play(14)                                          # the next game
    assert inner.resets == [11, None, 14, None]


@pytest.mark.slow
@pytest.mark.parametrize("cap", [150, 3000])
def test_pong_actor_eval_and_play_are_unchanged(monkeypatch, restore_rngs,  # noqa: F811
                                                cap):
    """Audit r1 N2: on the real Pong-v4 env, the new eval_episodes and play
    reproduce the old code (reset(seed) + one play_episode) exactly --
    returns, lengths, flags, actions, the inner reset sequence and the RNG
    state -- interleaved as the pipeline interleaves them; capped games
    (150) and natural game overs (3000, the run's cap: audit r2 L1)."""
    import torch
    import ray_actors
    arm = "effzero-repro-faithful-prioract-noredo-temp05-bnrecal"
    overrides = {k: v for k, v in ablation.ARMS[arm].items()
                 if not k.startswith("_")}
    schedule = {"TEMPERATURE_TOTAL_STEPS": 180000,
                "OFFPOLICY_DECAY_STEPS": 54000}
    for k in list(overrides) + list(schedule):
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    for k in ("ACTIONS", "EPISODE_CAP", "ENV_NAME", "FRAMESKIP"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    torch.set_num_threads(2)

    def old_eval(actor, seeds):
        out = []
        for s in seeds:
            actor.env.reset(seed=int(s))
            _, ret, steps, term = ablation.Run.play_episode(
                actor.shell, greedy=True, cap=actor.episode_cap)
            out.append((float(ret), int(steps), bool(term)))
        return out

    def old_play(actor, seed):
        actor.env.reset(seed=int(seed))
        before = actor.shell.env_steps
        traj, _, _, term = ablation.Run.play_episode(
            actor.shell, greedy=False, cap=actor.episode_cap)
        return traj, term, actor.shell.env_steps - before

    def trace(evalf, playf):
        actor = ray_actors.SelfPlayActor(overrides, 6, 5000, schedule,
                                         "Pong-v4", 4, cap, server=None,
                                         server_set="eval", force_cpu=True)
        resets, inner = [], actor.env.env
        real = inner.reset

        def spy(**kw):
            resets.append(kw.get("seed"))
            return real(**kw)

        inner.reset = spy
        log = [evalf(actor, [500000, 500001])]
        for s in (7, 8):
            t, term, n = playf(actor, s)
            log.append((term, n, [int(r[1]) for r in t]))
        log.append(evalf(actor, [500002]))
        log.append(resets)
        log.append((float(np.random.random()), float(torch.rand(1))))
        return log

    new = trace(lambda a, s: a.eval_episodes(s), lambda a, s: a.play(s))
    old = trace(old_eval, old_play)
    assert new == old
    assert new[-2][:2] == [500000, None]                    # the double reset
    # cap 150 truncates every game; at 3000 a random net loses 21-0 first
    capped = [not t for _, _, t in new[0] + new[3]] + [not new[1][0],
                                                       not new[2][0]]
    assert all(capped) if cap == 150 else not any(capped)


# --------------------------------------------------------------------------
# the exact probe: PROBE_ENV
# --------------------------------------------------------------------------
def test_probe_defaults_to_pong(probe):  # noqa: F811
    assert probe.ENV == "Pong-v4" and probe.env_actions("Pong-v4") == 6


def test_probe_pong_record_is_unchanged(probe, monkeypatch, tmp_path):  # noqa: F811
    import torch
    torch.set_num_threads(2)
    snap = _snapshot(probe, tmp_path)
    real_play = ablation.Run.play_episode
    monkeypatch.setattr(ablation.Run, "play_episode",
                        lambda self, greedy=False, cap=None:
                        real_play(self, greedy=greedy, cap=40))
    probe.main([str(snap)])
    rec = json.loads((tmp_path / "probe_exhaustive_eval-run-x-noops1.jsonl")
                     .read_text().splitlines()[0])
    assert "env" not in rec and rec["variant"] == "raw"
    assert list(rec["games"]) == ["1"] and rec["games"]["1"][1] <= 40


_PROBE_RUN = r"""
import os, sys, json, pathlib
sys.path[:0] = [{root!r}, {exp!r}, {probes!r}]
import torch
import ablation, muzero
import probe_exhaustive_eval as pe
assert pe.ENV == {env!r} and ablation.ACTIONS == 9 and ablation.ENV_NAME == pe.ENV
out = pathlib.Path({out!r})
pe.HERE = out
with ablation.arm_config(pe.ARM), ablation._run_length_globals(90, 2000):
    torch.manual_seed(3)
    run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
    snap = out / "run-m" / "snaps" / "weights-iter005.pth"
    snap.parent.mkdir(parents=True, exist_ok=True)
    torch.save({{**{{n: getattr(run.trainer, n).state_dict()
                    for n in ablation.NET_NAMES}},
                "training_steps": 0, "spec": {{}}}}, snap)
lives = []
real = ablation.Run.play_episode
def spy(self, greedy=False, cap=None):
    r = real(self, greedy=greedy, cap=cap)
    lives.append((r[2], r[3], self.env.unwrapped.ale.lives()))
    return r
ablation.Run.play_episode = spy
pe.main([str(snap), str(snap)])
print("LIVES", json.dumps(lives))
"""


@pytest.mark.slow
def test_probe_plays_whole_mspacman_games(tmp_path):
    env = {**os.environ, "PROBE_ENV": ENV_ID, "PROBE_MAX_NOOP": "2",
           "PROBE_THREADS": "2", "CUDA_VISIBLE_DEVICES": "-1"}
    code = _PROBE_RUN.format(root=str(ROOT), exp=str(ROOT / "experiments"),
                             probes=str(PROBE.parent), env=ENV_ID,
                             out=str(tmp_path))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    lives = json.loads(next(l for l in r.stdout.splitlines()
                            if l.startswith("LIVES "))[6:])
    recs = [json.loads(l) for l in (tmp_path / "probe_exhaustive_eval-run-m-"
                                    "noops2.jsonl").read_text().splitlines()]
    assert len(recs) == 2 and recs[0]["games"] == recs[1]["games"]  # exact
    rec = recs[0]
    assert rec["env"] == ENV_ID and rec["max_noop"] == 2
    assert list(rec["games"]) == ["1", "2"]
    # each game is every life of the match: lives left 2, 1, 0 per game
    # (a capped life would end its game early and read capped)
    per_game, game = [], []
    for steps, term, left in lives[:len(lives) // 2]:
        game.append((steps, term, left))
        if not term or left == 0:
            per_game.append(game)
            game = []
    assert len(per_game) == 2 and not game
    for g, (n, (ret, steps, term)) in zip(per_game, rec["games"].items()):
        assert steps == sum(s for s, _, _ in g)
        assert term is all(t for _, t, _ in g)
        if term:
            assert [left for _, _, left in g] == [2, 1, 0]


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------
@pytest.mark.parametrize("arm,base", sorted(ARMS.items()))
def test_mspacman_arm_converts_only_the_per_trajectory_knobs(arm, base):
    a, b = ablation.ARMS[base], ablation.ARMS[arm]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {
        "MAX_REPLAY", "POLICY_REFRESH_COUNT", "_doc"}
    with ablation.arm_config(base):
        old = (muzero.MAX_REPLAY, muzero.POLICY_REFRESH_COUNT,
               muzero.MAX_REPLAY_TRANSITIONS, muzero.POLICY_REFRESH_EVERY)
    with ablation.arm_config(arm):
        new = (muzero.MAX_REPLAY, muzero.POLICY_REFRESH_COUNT,
               muzero.MAX_REPLAY_TRANSITIONS, muzero.POLICY_REFRESH_EVERY)
    assert old == (85, 4, 31250, 2)
    assert new == (3 * 85, 3 * 4, 31250, 2)                  # per game: x3
    assert b["_launch"] == {"ray": True, "cpu_actor": True}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_startup_gate_passes_with_9_actions(arm, monkeypatch):
    monkeypatch.setattr(ablation, "ACTIONS", 9)
    with ablation.arm_config(arm) as spec:
        ablation.ray_scope_gate(spec)
        ablation.launch_gate(spec, True, True)
        ablation.eager_collect_gate(True)
        ablation.prior_temp_gate()
        ablation.bn_recal_gate(True)
        ablation.search_act_birth_gate()
        ablation.birth_search_gate()
        ablation.sve_gate()


def test_mspacman_env_through_the_lineage_stack():
    with ablation.arm_config(sorted(ARMS)[0]):
        assert muzero.USE_ATARI_WRAPPERS is True
        env = ablation.build_env(ENV_ID, 4)
    try:
        assert env.action_space.n == 9
        env.reset(seed=1)
        assert env.unwrapped.ale.lives() == 3
        assert isinstance(env, EpisodicLifeEnv)
        spec = env.unwrapped.spec.kwargs
        assert spec["repeat_action_probability"] == 0.0
        assert spec["frameskip"] == 1 and not spec["full_action_space"]
    finally:
        env.close()
    assert ablation.env_tag_fragment(ENV_ID) == "mspacmannoframeskip"


def test_gumbel_8_with_9_actions_considers_8_and_spends_8(monkeypatch):
    import torch
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", False)
    torch.manual_seed(0)
    np.random.seed(0)
    h = muzero.MCTSHelper(action_space=9, device=torch.device("cpu"),
                          network_ref=None, storage_ref=None,
                          enable_logging=False)
    for net in ("representation", "prediction", "dynamics", "value_prefix"):
        getattr(h.network_manager, net).eval()
    forced = []
    real = muzero.MCTSHelper.gumbel_selection_task

    async def spy(self, index, root_action, discount, max_depth=None):
        forced.append(root_action)
        return await real(self, index, root_action, discount, max_depth)

    monkeypatch.setattr(muzero.MCTSHelper, "gumbel_selection_task", spy)
    stack = np.zeros((12, 84, 84), dtype=np.uint8)
    assert min(muzero.GUMBEL_MAX_CANDIDATES, 9, 8) == 8
    action, pol, _ = asyncio.run(h.act_search(stack, num_simulations=8,
                                              force_gumbel=True))
    assert len(forced) == 8 and len(set(forced)) == 8      # 8 of 9, once each
    assert 0 <= action < 9 and action in forced and len(pol) == 9


# --------------------------------------------------------------------------
# the supervisor
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sup():
    spec = importlib.util.spec_from_file_location("supervise_mspacman", SUP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_supervisor_recipe(sup):
    assert sup.VARIANTS == {"gumbel": sorted(ARMS)[0], "temp": sorted(ARMS)[1]}
    cmd = sup.command(sup.VARIANTS["gumbel"], 0)

    def flag(name):
        return cmd[cmd.index(name) + 1]

    assert cmd[1] == "experiments/ablation.py" and cmd[-2:] == ["--ray",
                                                                "--cpu-actor"]
    assert flag("--arms") == sup.VARIANTS["gumbel"] and flag("--seed") == "0"
    assert (flag("--iterations"), flag("--train-steps"), flag("--eval-every"),
            flag("--checkpoint-every")) == ("90", "2000", "5", "2")
    assert (flag("--env"), flag("--frameskip"), flag("--episode-cap"),
            flag("--episodes-per-iter")) == (ENV_ID, "4", "3000", "3")
    assert "--snapshot-weights" in cmd and "--resume" not in cmd
    # the files ablation will write for this env and skip
    assert sup.TAG == "-" + ablation.env_tag_fragment(ENV_ID) + "-skip4"


def test_supervisor_paths_share_one_folder(sup):
    res = os.path.join(sup.REPO, "benchmarks", "results",
                       "mspacman-s0-2026-10-16")
    for v, arm in sup.VARIANTS.items():
        base, jsonl, ckpt, deaths = sup.paths(arm, 0)
        assert base == res and deaths == os.path.join(res, "deaths.log")
        assert jsonl == os.path.join(
            res, "ablation-mspacmannoframeskip-skip4.jsonl")
        assert ckpt == os.path.join(
            res, f"resume-mspacmannoframeskip-skip4-{arm}-seed0.ckpt")


def test_supervisor_other_variant_refuses(sup, monkeypatch, tmp_path):
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    assert sup.other_variant_ckpt("gumbel", 0) is None
    ck = sup.paths(sup.VARIANTS["temp"], 0)[2]
    os.makedirs(os.path.dirname(ck))
    open(ck, "wb").close()
    assert sup.other_variant_ckpt("gumbel", 0) == ck
    assert sup.other_variant_ckpt("temp", 0) is None


def _rows(raw, won=None):
    """{iteration: record} for rows 45-90 from a list of 10 means; phases
    won default to 0."""
    won = won or {}
    return {i: {"iteration": i, "exact_mean": m, "phases_won": won.get(i, 0)}
            for i, m in zip(range(45, 95, 5), raw)}


def _recal(vals):
    return {i: {"iteration": i, "exact_mean": m}
            for i, m in zip(range(60, 95, 5), vals)}


S0_RAW = [-13.7, -9.6, -7.2, -7.5, -0.5, 7.3, 17.9, 17.7, 15.4, 17.9]


@pytest.mark.parametrize("raw,won,recal,want", [
    # Gumbel seed 0's own series: @70 by phases (3/4), mean +9.74 -> gumbel
    (S0_RAW, {70: 3}, [-10.4, -0.8, 17.4, 19.1, 18.2, 17.5, 17.0],
     ("gumbel", "REPLICATES")),
    # never +10, never 3/4 phases -> temp
    ([-15] * 7 + [9.9] * 3, {80: 2, 85: 2, 90: 2}, [0] * 7,
     ("temp", "NO BREAKOUT")),
    # breakout @70 but the mean misses the band -> PARTIAL
    ([-15] * 5 + [10, 1, 1, 1, 1], None, [5] * 7, (None, "PARTIAL")),
    # a real collapse after the first success -> PARTIAL ...
    (S0_RAW[:7] + [-5, 15.4, 17.9], {70: 3}, [0, 0, 0, 0, -3, 0, 0],
     (None, "PARTIAL")),
    # ... unless its recal row is >= 0 (BN artifact, not counted)
    (S0_RAW[:7] + [-5, 15.4, 17.9], {70: 3}, [0, 0, 0, 0, 2, 0, 0],
     ("gumbel", "REPLICATES")),
    # a first success at 85 leaves stability UNTESTED -> PARTIAL
    ([-15] * 3 + [5] * 5 + [20, 20], None, [5] * 7, (None, "PARTIAL")),
    # rows before 60 have no recal and count raw
    ([-15, 12, -1] + [15] * 7, None, [15] * 7, (None, "PARTIAL")),
])
def test_supervisor_decision(sup, raw, won, recal, want):
    got = sup.decide(_rows(raw, won), _recal(recal))
    assert got[:2] == want


def test_supervisor_decision_on_seed_0s_real_readout(sup, monkeypatch):
    """Audit r1 NIT-3: Gumbel seed 0's own probe files, read through
    s1_readout: REPLICATES, @70, the RESULT's mean +9.7381."""
    probes = ROOT / "benchmarks" / "probes"
    run = "pong-gumbelact8-s0-2026-10-12"
    monkeypatch.setattr(sup, "S1_RAW",
                        str(probes / f"probe_exhaustive_eval-{run}.jsonl"))
    monkeypatch.setattr(sup, "S1_RECAL",
                        str(probes / f"probe_bn_recal_eval-{run}.jsonl"))
    raw, recal = sup.s1_readout()
    v, label, detail = sup.decide(raw, recal)
    assert (v, label) == ("gumbel", "REPLICATES")
    assert detail == "first success @70, mean 60-90 +9.7381, collapses []"


def test_supervisor_readout_waits_for_every_row(sup, monkeypatch, tmp_path):
    raw_f, recal_f = tmp_path / "raw.jsonl", tmp_path / "recal.jsonl"
    monkeypatch.setattr(sup, "S1_RAW", str(raw_f))
    monkeypatch.setattr(sup, "S1_RECAL", str(recal_f))
    assert sup.s1_readout() is None                   # nothing yet
    rec = lambda i, m, **kw: json.dumps({"iteration": i, "exact_mean": m,
                                         "variant": "raw", **kw})
    raw_f.write_text("\n".join(rec(i, 1.0) for i in range(45, 90, 5)) + "\n")
    cal = sup.S1_CAL.replace("/", chr(92))           # as the probe writes it
    recal_f.write_text("\n".join(json.dumps({"iteration": i, "exact_mean": 2.0,
                                            "calibration": cal})
                                 for i in range(60, 95, 5)) + "\n")
    assert sup.s1_readout() is None                   # raw @90 missing
    # a smoke record, an acting-variant record and a torn line do not count
    with open(raw_f, "a") as f:
        f.write(rec(90, 5.0, max_noop=2) + "\n")
        f.write(json.dumps({"iteration": 90, "exact_mean": 5.0,
                            "variant": "acting"}) + "\n")
        f.write('{"iteration": 90, "exa\n')
    assert sup.s1_readout() is None
    with open(raw_f, "a") as f:
        f.write(rec(90, 5.0) + "\n" + rec(90, 6.0) + "\n")  # a rerun: last wins
    raw, recal = sup.s1_readout()
    assert raw[90]["exact_mean"] == 6.0 and len(raw) == 10 and len(recal) == 7
    # the calibration match ignores case (audit r4 N3)
    with open(recal_f, "a") as f:
        f.write(json.dumps({"iteration": 90, "exact_mean": 3.0,
                            "calibration": cal.upper()}) + "\n")
    assert sup.s1_readout()[1][90]["exact_mean"] == 3.0
    # a recal rerun on another calibration snapshot never replaces a row
    with open(recal_f, "a") as f:
        f.write(json.dumps({"iteration": 90, "exact_mean": -9.0,
                            "calibration": "elsewhere/weights-iter050.pth"})
                + "\n")
    assert sup.s1_readout()[1][90]["exact_mean"] == 3.0


def test_supervisor_seed_state(sup, tmp_path):
    ck = tmp_path / "resume.ckpt"
    assert sup.seed_state(str(ck)) == "running"
    ck.write_bytes(b"x" * 1_000_001)
    assert sup.seed_state(str(ck)) == "running"
    (tmp_path / "supervisor.log").write_text("CRASH LOOP: 5 deaths\n")
    assert sup.seed_state(str(ck)) == "failed"
    (tmp_path / "supervisor.log").write_text("RUN COMPLETE s1\n")
    assert sup.seed_state(str(ck)) == "done"
    ck.write_bytes(b"marker")
    assert sup.seed_state(str(ck)) == "done"


def test_supervisor_waits_on_gumbel_seed_1(sup):
    assert sup.S1_CKPT == os.path.join(
        sup.REPO, "benchmarks", "results", "pong-gumbelact8-s1-2026-10-15",
        "resume-pong-skip4-effzero-repro-faithful-gumbelact8-birth-seed1.ckpt")
    assert sup.S1_RAW.endswith(
        "probe_exhaustive_eval-pong-gumbelact8-s1-2026-10-15.jsonl")
    assert sup.S1_RECAL.endswith(
        "probe_bn_recal_eval-pong-gumbelact8-s1-2026-10-15.jsonl")
    assert sup.RAW_ROWS == tuple(range(45, 95, 5))
    assert sup.RECAL_ROWS == tuple(range(60, 95, 5))
    assert sup.NO_WORSE == 3.68


@pytest.mark.parametrize("argv", [[], ["ray", "0"], ["auto", "1"],
                                  ["gumbel", "0", "x"], ["auto", "x"],
                                  ["--kai-override", "gumbel"]])
def test_supervisor_refuses_bad_arguments(sup, argv):
    assert sup.main(argv) == 64


def test_supervisor_partial_holds_without_launching(sup, monkeypatch, capsys,
                                                   tmp_path):
    """Isolated from the real run folder (audit r3 M1: this test runs in
    the preflight of every relaunch, when the run's own checkpoint IS in
    that folder)."""
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "wait_for_s1", lambda: (
        _rows([-15] * 5 + [10, 1, 1, 1, 1]), _recal([5] * 7)))
    monkeypatch.setattr(sup, "preflight", lambda arm: pytest.fail("launched"))
    assert sup.main(["auto", "0"]) == 2
    assert "HOLD: PARTIAL" in capsys.readouterr().out


def test_supervisor_audits_what_it_runs(sup):
    for rel in sup.AUDITED + sup.SUITES:
        assert (ROOT / rel).exists(), rel
    assert set(sup.SUITES) <= set(sup.AUDITED)
    for rel in ("experiments/ablation.py", "experiments/ray_actors.py",
                "benchmarks/probes/probe_exhaustive_eval.py",
                "benchmarks/supervisors/supervise_mspacman.py",
                "tests/test_mspacman.py", "EpisodicLifeEnv.py",
                "atari_wrappers.py"):
        assert rel in sup.AUDITED
    assert len(sup.SUITES) == 15
    # every older supervisor the suites read / exec / run (audit r4 L3)
    for name in ("gumbelact8", "bnrecal", "searchact8", "factor",
                 "prioract2", "tempscale", "seqref"):
        assert f"benchmarks/supervisors/supervise_{name}.py" in sup.AUDITED


MSPACMAN_LAUNCH = "f07d1d440059821edd0c206fef6b8262c521f4d7"


def test_supervisor_manifest_matches_the_launch_commit(sup):
    """The manifest next to the script (written by --write-manifest) hashes
    the AUDITED files of the run's launch commit; the supervisor READS the
    main tree's copy (audit r1 L2). The run and its evals finished
    2026-10-17 and the freeze lifted, so the live tree has moved on (P2a's
    edits): the manifest is the archived launch record, checked against
    that commit's content through git (skipped in an archive copy without
    git history)."""
    import hashlib
    assert sup.MANIFEST_OUT == str(SUP.parent / "mspacman_manifest.json")
    assert sup.MANIFEST == os.path.join(sup.REPO, "benchmarks", "supervisors",
                                        "mspacman_manifest.json")
    with open(sup.MANIFEST_OUT, encoding="utf-8") as f:
        want = json.load(f)
    assert sorted(want) == sorted(sup.AUDITED)
    have = {}
    for rel in want:
        r = subprocess.run(["git", "show", f"{MSPACMAN_LAUNCH}:{rel}"],
                           cwd=ROOT, capture_output=True)
        if r.returncode:
            pytest.skip("no git history for the launch commit here")
        have[rel] = hashlib.sha256(
            r.stdout.replace(b"\r\n", b"\n")).hexdigest()
    assert want == have


def test_supervisor_live_child_guard(sup):
    """Audit r1 L4: a running ablation.py of this arm and seed is found
    (and one of another arm or seed is not)."""
    arm = sup.VARIANTS["gumbel"]
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                          "experiments/ablation.py", "--arms", arm,
                          "--seed", "0"])
    try:
        import time
        for _ in range(50):
            if p.pid in sup.live_children(arm, 0):
                break
            time.sleep(0.1)
        assert p.pid in sup.live_children(arm, 0)
        assert p.pid not in sup.live_children(arm, 1)
        assert p.pid not in sup.live_children(sup.VARIANTS["temp"], 0)
    finally:
        p.kill()
        p.wait()
    assert p.pid not in sup.live_children(arm, 0)


def test_supervisor_holds_on_either_variants_live_child(sup, monkeypatch,
                                                         tmp_path, capsys):
    """Audit r3 L1: launching gumbel while an orphaned TEMP ablation.py runs
    (no checkpoint yet) holds with an ALERT -- no second writer."""
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "wait_for_s1", lambda: _REPLICATES)
    monkeypatch.setattr(sup, "preflight", lambda arm: (True, ""))
    monkeypatch.setattr(sup, "manifest_check", lambda: (True, ""))
    monkeypatch.setattr(sup, "live_children", lambda arm, seed: (
        [4242] if arm == sup.VARIANTS["temp"] else []))
    monkeypatch.setattr(sup.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("launched"))

    def sleep(s):
        raise RuntimeError("held")

    monkeypatch.setattr(sup.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="held"):
        sup.main(["gumbel", "0"])
    assert "already running (pids [4242])" in capsys.readouterr().out


@pytest.mark.parametrize("porcelain,rc", [(" M experiments/ablation.py\n", 1),
                                           ("", 0)])
def test_supervisor_write_manifest_refuses_a_dirty_tree(sup, monkeypatch,
                                                        tmp_path, porcelain,
                                                        rc):
    """Audit r3 M1: the manifest hashes the working tree, so uncommitted
    edits to an AUDITED file refuse it (nothing written). REPO is pointed
    away from this tree: in main's copy the tree IS REPO and the main-tree
    refusal would fire first (audit r5 HIGH-1)."""
    monkeypatch.setattr(sup, "REPO", str(tmp_path / "main"))
    out = tmp_path / "m.json"
    monkeypatch.setattr(sup, "MANIFEST_OUT", str(out))
    monkeypatch.setattr(sup, "file_hashes", lambda root: {"a": "b"})
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=porcelain, stderr="")

    monkeypatch.setattr(sup.subprocess, "run", run)
    assert sup.main(["--write-manifest"]) == rc
    assert calls[0][:4] == ["git", "-C", str(ROOT), "status"]
    assert calls[0][-len(sup.AUDITED):] == sup.AUDITED
    assert out.exists() is (rc == 0)


def test_supervisor_write_manifest_refuses_the_main_tree(sup, monkeypatch):
    """Audit r4 N2: main's copy never writes the manifest."""
    monkeypatch.setattr(sup, "REPO", str(ROOT))
    monkeypatch.setattr(sup.subprocess, "run",
                        lambda *a, **k: pytest.fail("ran git"))
    assert sup.main(["--write-manifest"]) == 1


@pytest.mark.parametrize("manifest_in_main", [False, True])
def test_supervisor_preflight_failure_alerts_once_merged(sup, monkeypatch,
                                                         tmp_path, capsys,
                                                         manifest_in_main):
    # (see also test_supervisor_alerts_on_a_missing_manifest below)
    """Audit r3 M1: before the merge a failing preflight is the expected
    wait; once main holds the manifest it is an ALERT."""
    m = tmp_path / "mspacman_manifest.json"
    if manifest_in_main:
        m.write_text("{}")
    monkeypatch.setattr(sup, "MANIFEST", str(m))
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "wait_for_s1", lambda: _REPLICATES)
    monkeypatch.setattr(sup, "preflight", lambda arm: (False, "why"))

    def sleep(s):
        raise RuntimeError("enough")

    monkeypatch.setattr(sup.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="enough"):
        sup.main(["auto", "0"])
    out = capsys.readouterr().out
    assert ("ALERT: PREFLIGHT not ready" in out) is manifest_in_main
    assert "PREFLIGHT not ready (0 min): why" in out


def test_supervisor_alerts_on_a_missing_manifest(sup, monkeypatch, tmp_path,
                                                  capsys):
    """Audit r5 LOW-1: the manifest still absent from main 2 h after the
    decision (the merge watcher died or held) is an ALERT."""
    monkeypatch.setattr(sup, "MANIFEST", str(tmp_path / "absent.json"))
    monkeypatch.setattr(sup, "REPO", str(tmp_path))
    monkeypatch.setattr(sup, "wait_for_s1", lambda: _REPLICATES)
    monkeypatch.setattr(sup, "preflight", lambda arm: (False, "why"))
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) > 13:
            raise RuntimeError("enough")

    monkeypatch.setattr(sup.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="enough"):
        sup.main(["auto", "0"])
    out = capsys.readouterr().out.splitlines()
    early = [l for l in out if "PREFLIGHT not ready (0 min)" in l]
    late = [l for l in out if "PREFLIGHT not ready (120 min)" in l]
    assert early and not early[0].startswith("ALERT")
    assert late and late[0].startswith("ALERT") and "mspacman_merge" in late[0]


def test_supervisor_auto_resumes_kais_pick_after_partial(sup, monkeypatch,
                                                         capsys):
    """Audit r1 L3: PARTIAL with exactly one variant's checkpoint in the
    folder (Kai's pick, launched before a reboot) resumes that variant."""
    monkeypatch.setattr(sup, "wait_for_s1", lambda: (
        _rows([-15] * 5 + [10, 1, 1, 1, 1]), _recal([5] * 7)))
    monkeypatch.setattr(sup, "variant_ckpts", lambda seed: ["temp"])
    monkeypatch.setattr(sup, "other_variant_ckpt", lambda v, seed: None)
    picked = []

    def stop(arm):
        picked.append(arm)
        raise RuntimeError("stop before launch")

    monkeypatch.setattr(sup, "preflight", stop)
    with pytest.raises(RuntimeError, match="stop before launch"):
        sup.main(["auto", "0"])
    assert picked == [sup.VARIANTS["temp"]]
    assert "resuming temp" in capsys.readouterr().out


_REPLICATES = (_rows([-15] * 5 + [12] * 5), _recal([12] * 7))
_PARTIAL = (_rows([-15] * 5 + [10, 1, 1, 1, 1]), _recal([5] * 7))
_NO_BREAKOUT = (_rows([-15] * 10), _recal([-15] * 7))


@pytest.mark.parametrize("readout,argv,want", [
    # a definite decision picks the arm: a disagreeing explicit variant is
    # refused unless Kai overrode it; an agreeing one runs
    (_REPLICATES, ["temp", "0"], 3),
    (_NO_BREAKOUT, ["gumbel", "0"], 3),
    (_REPLICATES, ["temp", "0", "--kai-override"], "temp"),
    (_REPLICATES, ["gumbel", "0"], "gumbel"),
    (_NO_BREAKOUT, ["auto", "0"], "temp"),
    (_REPLICATES, ["auto", "0"], "gumbel"),
    # after PARTIAL either explicit pick is Kai's
    (_PARTIAL, ["temp", "0"], "temp"),
    (_PARTIAL, ["gumbel", "0"], "gumbel"),
])
def test_supervisor_variant_choice(sup, monkeypatch, readout, argv, want):
    """Audit r2 NIT: an explicit variant must agree with a definite
    decision (or carry --kai-override)."""
    monkeypatch.setattr(sup, "wait_for_s1", lambda: readout)
    monkeypatch.setattr(sup, "variant_ckpts", lambda seed: [])
    monkeypatch.setattr(sup, "other_variant_ckpt", lambda v, seed: None)
    picked = []

    def stop(arm):
        picked.append(arm)
        raise RuntimeError("stop before launch")

    monkeypatch.setattr(sup, "preflight", stop)
    if want == 3:
        assert sup.main(argv) == 3 and picked == []
    else:
        with pytest.raises(RuntimeError, match="stop before launch"):
            sup.main(argv)
        assert picked == [sup.VARIANTS[want]]


def test_supervisor_readout_clock_starts_at_the_marker(sup, monkeypatch,
                                                       tmp_path, capsys):
    """Audit r2 NIT: a relaunched supervisor dates seed 1's completion by
    its marker, so the 2-h ALERT is not postponed by the relaunch."""
    import time
    ck = tmp_path / "resume.ckpt"
    ck.write_bytes(b"marker")
    old = time.time() - 3 * 3600
    os.utime(ck, (old, old))
    monkeypatch.setattr(sup, "S1_CKPT", str(ck))
    monkeypatch.setattr(sup, "s1_readout", lambda: None)
    monkeypatch.setattr(sup, "read_rows", lambda path, raw=False, **kw: {})

    def sleep(s):
        raise RuntimeError("enough")

    monkeypatch.setattr(sup.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="enough"):
        sup.wait_for_s1()
    assert "ALERT: Gumbel seed 1 completed > 2 h ago" in capsys.readouterr().out


def test_supervisor_alerts_on_an_incomplete_readout(sup, monkeypatch,
                                                    capsys, tmp_path):
    """Audit r1 L1: seed 1 done but its readout still incomplete 2 h later
    -> an ALERT naming the missing rows (and it keeps waiting). Isolated
    from seed 1's real checkpoint (audit r4 N1)."""
    monkeypatch.setattr(sup, "S1_CKPT", str(tmp_path / "absent.ckpt"))
    clock = iter([0.0, 0.0, 3 * 3600.0, 3 * 3600.0])
    monkeypatch.setattr(sup, "seed_state", lambda ck: "done")
    monkeypatch.setattr(sup, "s1_readout", lambda: None)
    monkeypatch.setattr(sup, "read_rows", lambda path, raw=False, **kw: (
        {i: {} for i in range(45, 90, 5)} if raw else {}))
    monkeypatch.setattr(sup.time, "time", lambda: next(clock))
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 2:
            raise RuntimeError("enough")

    monkeypatch.setattr(sup.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="enough"):
        sup.wait_for_s1()
    out = capsys.readouterr().out
    assert "WAITING for Gumbel seed 1 (done; readout 9/10" in out
    assert "ALERT: Gumbel seed 1 completed > 2 h ago" in out
    assert "missing exact [90], recal [60, 65, 70, 75, 80, 85, 90]" in out


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["gumbel", "temp"])
def test_supervisor_quick_probe_passes_here(sup, variant):
    r = subprocess.run([sys.executable, "-c",
                        sup.quick_probe(sup.VARIANTS[variant])],
                       cwd=str(ROOT), capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
                       timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
