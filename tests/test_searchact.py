"""The prioract bisect's first arm (2026-09-29): effzero-repro-faithful-
searchact -- the prioract stack with SEARCH acting restored -- and the
EVAL_PRIOR_ONLY flag that keeps its in-run eval on the prior's argmax."""
import asyncio
import sys

import numpy as np
import pytest

import muzero

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parents[1] / "experiments"))
import ablation

ARM = "effzero-repro-faithful-searchact"
PRIOR = np.array([0.1, 0.2, 0.4, 0.2, 0.1])          # argmax 2


def test_arm_is_prioract_with_search_acting_restored():
    base = ablation.ARMS["effzero-repro-faithful-prioract"]
    arm = ablation.ARMS[ARM]
    delta = {k for k in set(base) | set(arm) if base.get(k) != arm.get(k)}
    assert delta == {"USE_SEARCH_ACTING", "BIRTH_SEARCH",
                     "PRIOR_ACT_STAGED_TEMP", "EVAL_PRIOR_ONLY", "_doc"}
    assert arm["USE_SEARCH_ACTING"] is True and arm["BIRTH_SEARCH"] is False
    assert arm["EVAL_PRIOR_ONLY"] is True
    # ReDo, the vectorized communal re-search and the faithful flags stay.
    for k in ("REDO_EVERY", "REDO_NETS", "REFRESH_VTREE",
              "VTREE_COMMUNAL_TRAJS", "LATENT_RENORMALIZE", "OPTIMIZER"):
        assert arm.get(k) == base.get(k), k


def test_eval_prior_only_defaults_off():
    assert muzero.EVAL_PRIOR_ONLY is False


class _Env:
    """`n` decisions, then a real terminal (or never, for a truncation)."""
    def __init__(self, n):
        self.n, self.t = n, 0

    def reset(self, seed=None):
        self.t = 0
        return np.zeros((210, 160, 3), dtype=np.uint8), {}

    def step(self, action):
        self.t += 1
        return (np.zeros((210, 160, 3), dtype=np.uint8), 0.0,
                self.t >= self.n, False, {})


def _run(monkeypatch, n_env, flag):
    calls = []

    async def act_search(stack, greedy=False, **kw):
        calls.append(greedy)
        return 4, np.eye(5)[4], 0.5          # search picks action 4

    nm = type("NM", (), {})()
    nm.training_steps = 0
    nm.initial_inference = lambda stack: (PRIOR.copy(), 0.0)
    run = object.__new__(ablation.Run)
    run.env, run.env_steps = _Env(n_env), 0
    run.searcher = type("S", (), {"network_manager": nm,
                                  "act_search": staticmethod(act_search)})()
    monkeypatch.setattr(ablation, "ACTIONS", 5)
    monkeypatch.setattr(muzero, "PAPER_TARGETS", True)
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", True)
    monkeypatch.setattr(muzero, "EVAL_PRIOR_ONLY", flag)
    return run, calls


@pytest.mark.parametrize("n_env,cap", [(3, 10), (100, 4)])   # terminal, truncated
def test_greedy_episode_acts_on_the_prior_argmax_when_flagged(monkeypatch,
                                                              n_env, cap):
    run, calls = _run(monkeypatch, n_env, True)
    traj, _, steps, _ = run.play_episode(greedy=True, cap=cap)
    assert calls == []                              # no per-step or tail search
    assert all(len(r) == 3 for r in traj)           # plain records, tail too
    assert [r[1] for r in traj[:-1]] == [2] * steps  # prior argmax


def test_greedy_episode_still_searches_without_the_flag(monkeypatch):
    run, calls = _run(monkeypatch, 3, False)
    traj, _, steps, _ = run.play_episode(greedy=True, cap=10)
    assert calls == [True] * steps                  # paper eval searches
    assert all(len(r) == 5 for r in traj)
    assert [r[1] for r in traj[:-1]] == [4] * steps


@pytest.mark.parametrize("flag", [True, False])
def test_self_play_searches_regardless_of_the_flag(monkeypatch, flag):
    """Self-play (non-greedy) always searches; a truncated tail gets its
    one greedy search, as before."""
    run, calls = _run(monkeypatch, 100, flag)
    traj, _, steps, terminated = run.play_episode(greedy=False, cap=4)
    assert not terminated
    assert calls == [False] * steps + [True]        # + the tail search
    assert all(len(r) == 5 for r in traj)
    assert [r[1] for r in traj[:-1]] == [4] * steps
    assert run.env_steps == steps


@pytest.mark.parametrize("flag,searches", [(True, False), (False, True)])
def test_cpu_eval_actor_honours_the_flag_without_ray(monkeypatch, flag,
                                                     searches):
    """Bisect audit r1 LOW-6: the --ray eval path is SelfPlayActor.
    eval_episodes on an actor that rebuilt the arm's world from its
    overrides; built in-process here (force_cpu, no Ray). It must not
    search with EVAL_PRIOR_ONLY and must without it (paper eval)."""
    import random
    import torch
    import ray_actors
    overrides = {k: v for k, v in ablation.ARMS[ARM].items()
                 if not k.startswith("_")}
    overrides["EVAL_PRIOR_ONLY"] = flag
    schedule = {"TEMPERATURE_TOTAL_STEPS": 180000,
                "OFFPOLICY_DECAY_STEPS": 54000}
    # The actor writes these into the muzero module and reseeds the global
    # RNGs in-process: register every key for restoration, save the RNGs.
    for k in list(overrides) + list(schedule):
        if hasattr(muzero, k):
            monkeypatch.setattr(muzero, k, getattr(muzero, k))
    # ...and the ablation module's env globals (ray_actors sets them).
    for k in ("ACTIONS", "EPISODE_CAP", "ENV_NAME", "FRAMESKIP"):
        monkeypatch.setattr(ablation, k, getattr(ablation, k))
    np_state, t_state = np.random.get_state(), torch.get_rng_state()
    py_state = random.getstate()      # the sequential engine's tie-breaks
    try:
        actor = ray_actors.SelfPlayActor(overrides, 6, 5000, schedule,
                                         "Pong-v4", 4, 8, server=None,
                                         server_set="eval", force_cpu=True)
        assert muzero.EVAL_PRIOR_ONLY is flag
        calls = []
        real = actor.searcher.run_full_mcts

        async def spy(*a, **kw):
            calls.append(kw.get("num_simulations"))
            return await real(*a, **kw)

        actor.searcher.run_full_mcts = spy
        out = actor.eval_episodes([500000])
        assert len(out) == 1 and out[0][1] > 0
        assert bool(calls) is searches
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(t_state)
        random.setstate(py_state)
