"""Gumbel search acting with birth-search targets (Kai, 2026-10-12):
ACTOR_GUMBEL_SEARCH and the arm effzero-repro-faithful-gumbelact8-birth.
Pins: only the actor's search ENGINE changes vs the searchact8 arm (PUCT-8
-> Gumbel-8; birth search, refresh and every other search stay PUCT), the
gates, act_search's force_gumbel dispatch, play_episode's wiring, the
run_one --ray loop end to end, the probe's search-agent mode, and the
supervisor."""
import ast
import asyncio
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import muzero
from tests.test_prioract2 import _fake_pipe, restore_rngs  # noqa: F401

ARM = "effzero-repro-faithful-gumbelact8-birth"
TWIN = "effzero-repro-faithful-searchact8-birth"
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUP = ROOT / "benchmarks" / "supervisors" / "supervise_gumbelact8.py"
PROBES = ROOT / "benchmarks" / "probes"


def _eff(spec, k):
    return spec[k] if k in spec else getattr(muzero, k, None)


# --------------------------------------------------------------------------
# the arm, defaults, gates
# --------------------------------------------------------------------------
def test_arm_changes_only_the_actor_engine():
    a, b = ablation.ARMS[TWIN], ablation.ARMS[ARM]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {
        "ACTOR_GUMBEL_SEARCH", "ACTOR_GUMBEL_STAGED_SCALE", "_doc"}
    assert b["ACTOR_GUMBEL_SEARCH"] is True
    assert b["ACTOR_GUMBEL_STAGED_SCALE"] is True     # the twin's T schedule
    assert _eff(b, "USE_GUMBEL_SEARCH") is False      # every other search: PUCT
    assert b["ACTOR_SEARCH_SIMS"] == 8 and _eff(b, "ACT_SEARCH_SIMS") == 50
    assert b["SEARCH_ACT_BIRTH_TARGETS"] is True and b["EVAL_PRIOR_ONLY"] is True
    assert b["_launch"] == {"ray": True, "cpu_actor": True}
    # m = min(GUMBEL_MAX_CANDIDATES, actions, sims): all 6 Pong actions
    assert min(_eff(b, "GUMBEL_MAX_CANDIDATES"), 6, 8) == 6


def test_defaults_off():
    assert muzero.ACTOR_GUMBEL_SEARCH is False
    assert muzero.ACTOR_GUMBEL_STAGED_SCALE is False


def test_every_startup_gate_passes_the_arm():
    with ablation.arm_config(ARM) as spec:
        ablation.ray_scope_gate(spec)
        ablation.launch_gate(spec, True, True)
        ablation.eager_collect_gate(True)
        ablation.prior_temp_gate()
        ablation.bn_recal_gate(True)
        ablation.search_act_birth_gate()
        ablation.birth_search_gate()
        ablation.sve_gate()


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_actor_gumbel_must_be_a_bool(monkeypatch, bad):
    monkeypatch.setattr(muzero, "ACTOR_GUMBEL_SEARCH", bad)
    with pytest.raises(SystemExit, match="ACTOR_GUMBEL_SEARCH must be a bool"):
        ablation.search_act_birth_gate()


def test_actor_gumbel_only_with_birth_targets(monkeypatch):
    monkeypatch.setattr(muzero, "ACTOR_GUMBEL_SEARCH", True)
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", True)
    monkeypatch.setattr(muzero, "SEARCH_ACT_BIRTH_TARGETS", False)
    with pytest.raises(SystemExit, match="only with SEARCH_ACT_BIRTH_TARGETS"):
        ablation.search_act_birth_gate()


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_staged_scale_must_be_a_bool(bad):
    with ablation.arm_config(ARM):
        muzero.ACTOR_GUMBEL_STAGED_SCALE = bad     # an arm key: restored
        with pytest.raises(SystemExit,
                           match="ACTOR_GUMBEL_STAGED_SCALE must be a bool"):
            ablation.search_act_birth_gate()


def test_staged_scale_needs_the_gumbel_actor():
    with ablation.arm_config(ARM):
        muzero.ACTOR_GUMBEL_SEARCH = False
        with pytest.raises(SystemExit, match="inert without ACTOR_GUMBEL_SEARCH"):
            ablation.search_act_birth_gate()


def test_run_one_runs_the_gate_before_anything_else(monkeypatch, tmp_path):
    """r1 L4 (A8): the gate is wired into run_one -- a bad flag stops the
    run at startup."""
    monkeypatch.setitem(ablation.ARMS, "bad-g8", {
        **ablation.ARMS[ARM], "ACTOR_GUMBEL_STAGED_SCALE": "yes"})
    with pytest.raises(SystemExit,
                       match="ACTOR_GUMBEL_STAGED_SCALE must be a bool"):
        ablation.run_one("bad-g8", 0, 2, 1, 10, 1, tmp_path / "a.jsonl",
                         use_ray=True, cpu_actor=True)
    assert not (tmp_path / "a.jsonl").exists()


def test_actor_gumbel_refuses_global_gumbel():
    # A plain setattr inside arm_config: the arm's own key, restored on exit
    # (a monkeypatch here would "restore" the ARM's value at teardown).
    assert "USE_GUMBEL_SEARCH" in ablation.ARMS[ARM]
    with ablation.arm_config(ARM):
        muzero.USE_GUMBEL_SEARCH = True
        with pytest.raises(SystemExit, match="would switch every search"):
            ablation.search_act_birth_gate()


# --------------------------------------------------------------------------
# act_search's dispatch (real searcher)
# --------------------------------------------------------------------------
@pytest.fixture
def helper(monkeypatch, request):
    import torch
    threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    torch.set_num_threads(2)
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", False)
    h = muzero.MCTSHelper(action_space=6, device=torch.device("cpu"),
                          network_ref=None, storage_ref=None,
                          enable_logging=False)
    for net in ("representation", "prediction", "dynamics", "value_prefix"):
        getattr(h.network_manager, net).eval()
    return h


def _spies(monkeypatch, scales=None):
    calls = {"gumbel": [], "puct": []}
    real_g = muzero.MCTSHelper.run_gumbel_mcts
    real_p = muzero.MCTSHelper.run_full_mcts

    async def g(self, index, num_simulations=32, discount=None, max_depth=None,
                gumbel_scale=1.0):
        calls["gumbel"].append(num_simulations)
        if scales is not None:
            scales.append(gumbel_scale)
        return await real_g(self, index, num_simulations=num_simulations,
                            discount=discount, max_depth=max_depth,
                            gumbel_scale=gumbel_scale)

    async def p(self, index, num_simulations=128, sim_batches=8, c_puct=1.0,
                discount=None, max_depth=None, root_noise=True):
        calls["puct"].append((num_simulations, root_noise))
        return await real_p(self, index, num_simulations=num_simulations,
                            sim_batches=sim_batches, c_puct=c_puct,
                            discount=discount, max_depth=max_depth,
                            root_noise=root_noise)

    monkeypatch.setattr(muzero.MCTSHelper, "run_gumbel_mcts", g)
    monkeypatch.setattr(muzero.MCTSHelper, "run_full_mcts", p)
    return calls


def _obs(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (12, *muzero.IMAGE_RESHAPE), dtype=np.uint8)


def test_force_gumbel_runs_the_gumbel_engine_and_acts_with_the_survivor(
        helper, monkeypatch):
    calls = _spies(monkeypatch)
    noise = []
    monkeypatch.setattr(muzero.MCTSHelper, "add_noise_to_node",
                        lambda self, i: noise.append(i))
    np.random.seed(3)
    action, policy, value = asyncio.run(helper.act_search(
        _obs(), num_simulations=8, force_gumbel=True))
    assert calls == {"gumbel": [8], "puct": []}
    assert action == helper.last_search_action      # the halving survivor
    assert noise == []                              # Gumbel draws, no Dirichlet
    assert np.isclose(policy.sum(), 1.0) and np.isfinite(value)


def test_force_gumbel_greedy_takes_the_improved_policy_argmax(helper,
                                                              monkeypatch):
    calls = _spies(monkeypatch)
    action, policy, _ = asyncio.run(helper.act_search(
        _obs(1), num_simulations=8, greedy=True, force_gumbel=True))
    assert calls["gumbel"] == [8] and action == int(np.argmax(policy))


def test_without_force_gumbel_the_search_is_unchanged_puct(helper,
                                                           monkeypatch):
    """Flag-off parity: the default call is the PUCT path with root noise
    (self-play) / without it (greedy), exactly as before."""
    calls = _spies(monkeypatch)
    asyncio.run(helper.act_search(_obs(), num_simulations=8))
    asyncio.run(helper.act_search(_obs(), num_simulations=8, greedy=True))
    assert calls == {"gumbel": [], "puct": [(8, True), (8, False)]}


def test_gumbel_visits_every_action_at_8_sims(helper, monkeypatch):
    """The manipulation at the arm's budget: sequential halving over all 6
    actions -- each root child gets >= 1 of the 8 simulations (PUCT-8
    leaves some unvisited)."""
    seen = []
    real_release = muzero.MCTSHelper.release_root

    def spy_release(self, node):
        seen.append({a: c.visit_count for a, c in node.children.items()})
        return real_release(self, node)

    monkeypatch.setattr(muzero.MCTSHelper, "release_root", spy_release)
    for s in range(5):
        np.random.seed(s)
        asyncio.run(helper.act_search(_obs(s), num_simulations=8,
                                      force_gumbel=True))
    assert len(seen) == 5
    for v in seen:
        assert len(v) == 6 and min(v.values()) >= 1 and sum(v.values()) == 8


# --------------------------------------------------------------------------
# the staged scale, and the engine's behaviour at the arm's budget
# --------------------------------------------------------------------------
@pytest.mark.parametrize("staged,force,greedy,want", [
    (True, True, False, 0.25),      # the actor's self-play: the staged T
    (True, True, True, 1.0),        # greedy: improved-policy argmax, scale 1
    (False, True, False, 1.0),      # the paper's scale
    (True, False, False, None)])    # not the actor: PUCT, no Gumbel at all
def test_actor_gumbel_scale_follows_the_staged_temperature(
        helper, monkeypatch, staged, force, greedy, want):
    scales = []
    calls = _spies(monkeypatch, scales)
    monkeypatch.setattr(muzero, "ACTOR_GUMBEL_STAGED_SCALE", staged)
    monkeypatch.setattr(muzero, "visit_softmax_temperature", lambda steps: 0.25)
    asyncio.run(helper.act_search(_obs(), num_simulations=8, greedy=greedy,
                                  force_gumbel=force))
    assert scales == ([] if want is None else [want])
    assert bool(calls["gumbel"]) is (want is not None)


def _tagged_root(helper):
    """A fresh act_search-style root with a uniform prior; each child is
    tagged with its action (for _fixed_q)."""
    latents, _, values = helper.network_manager.batch_inference(_obs()[None])
    root = muzero.MCTSNode(latent=latents[0:1], prior=1.0)
    root.policy = np.full(6, 1.0 / 6)
    root.expanded_value = float(values[0])
    helper.current_trajectory = [root]
    helper.batch_expand_nodes([root])
    for a, child in root.children.items():
        child.tag = a
    return root


def _fixed_q(monkeypatch, q):
    monkeypatch.setattr(
        muzero.MCTSHelper, "_normalized_q",
        lambda self, child, discount: q.get(getattr(child, "tag", None), 0.0))


def test_q_drives_the_halving_and_the_action(helper, monkeypatch):
    """r1 L4 (E2): one action with q_hat 1 (sigma ~5 logits) against flat
    others under a uniform prior is played in nearly every search at scale
    1 -- a halving that ignored Q would keep it only ~half the time."""
    _fixed_q(monkeypatch, {4: 1.0})
    wins = 0
    for seed in range(30):
        np.random.seed(seed)
        _tagged_root(helper)
        asyncio.run(helper.run_gumbel_mcts(0, num_simulations=8))
        wins += helper.last_search_action == 4
    assert wins >= 27


def test_scale_zero_is_deterministic_and_scale_one_explores(helper,
                                                            monkeypatch):
    """r1 L4 (E3) + the staged scale: with flat Q and a uniform prior the
    played action is the Gumbel draw's -- spread over the actions at scale
    1, a single action at scale 0."""
    _fixed_q(monkeypatch, {})
    picks = {1.0: set(), 0.0: set()}
    for scale in picks:
        for seed in range(20):
            np.random.seed(seed)
            _tagged_root(helper)
            asyncio.run(helper.run_gumbel_mcts(0, num_simulations=8,
                                               gumbel_scale=scale))
            picks[scale].add(helper.last_search_action)
    assert len(picks[1.0]) >= 4 and len(picks[0.0]) == 1


def test_phase1_visits_go_to_the_top_two(helper, monkeypatch):
    """r1 L4 (E5): the 6 + 2 schedule -- every action once, then the second
    visits to the two best by g + logits + sigma(q_hat) (no noise here)."""
    _fixed_q(monkeypatch, {4: 1.0, 1: 0.8})
    seen = []
    real_release = muzero.MCTSHelper.release_root

    def spy_release(self, node):
        seen.append({a: c.visit_count for a, c in node.children.items()})
        return real_release(self, node)

    monkeypatch.setattr(muzero.MCTSHelper, "release_root", spy_release)
    _tagged_root(helper)
    asyncio.run(helper.run_gumbel_mcts(0, num_simulations=8, gumbel_scale=0.0))
    assert seen == [{0: 1, 1: 2, 2: 1, 3: 1, 4: 2, 5: 1}]
    assert helper.last_search_action == 4


def test_sigma_scale_counts_this_searchs_visits(helper):
    """r1 L4 (E4): sigma = (c_visit + max visits) * c_scale, the visits
    net of the baseline."""
    kids = {a: type("C", (), {"visit_count": n})()
            for a, n in enumerate([2, 1, 1, 1, 1, 1])}
    node = type("N", (), {"children": kids})()
    assert helper._sigma_scale(node) == pytest.approx((50.0 + 2) * 0.1)
    assert helper._sigma_scale(node, {0: 1}) == pytest.approx((50.0 + 1) * 0.1)


# --------------------------------------------------------------------------
# play_episode's wiring
# --------------------------------------------------------------------------
class _Env:
    def reset(self, seed=None):
        return np.zeros((210, 160, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((210, 160, 3), dtype=np.uint8), 1.0, False, False, {}


class _Searcher:
    def __init__(self):
        self.calls = []
        self.network_manager = type("NM", (), {"training_steps": 0})()

    async def act_search(self, stack, num_simulations=None, discount=None,
                         greedy=False, force_gumbel=False):
        self.calls.append((greedy, num_simulations, force_gumbel))
        return 2, np.full(ablation.ACTIONS, 1.0 / ablation.ACTIONS), 0.5


@pytest.mark.parametrize("flag", [True, False])
def test_play_episode_passes_the_engine_flag(flag):
    assert {"ACTOR_GUMBEL_SEARCH", "EVAL_PRIOR_ONLY"} <= set(ablation.ARMS[ARM])
    with ablation.arm_config(ARM):
        muzero.ACTOR_GUMBEL_SEARCH = flag        # arm keys: restored on exit
        muzero.EVAL_PRIOR_ONLY = False           # (greedy searches too)
        run = object.__new__(ablation.Run)
        run.env, run.env_steps = _Env(), 0
        run.searcher = _Searcher()
        traj, _, steps, _ = run.play_episode(greedy=False, cap=4)
        run.play_episode(greedy=True, cap=2)
    assert run.searcher.calls == [(False, 8, flag)] * 4 + [(True, None, flag)] * 2
    assert all(len(r) == 3 for r in traj)            # birth makes the targets


# --------------------------------------------------------------------------
# run_one end to end (in-process fake pipeline, real searcher)
# --------------------------------------------------------------------------
def test_run_one_ray_loop_acts_gumbel_8_and_births_puct_50(
        monkeypatch, tmp_path, restore_rngs):  # noqa: F811
    """Through run_one --ray, every actor search runs the Gumbel engine at 8
    sims and no actor search runs PUCT, while the BIRTH engine (vectorized
    PUCT) searches every position at 50 sims; rows carry no beh_*."""
    import muzero_vtree
    import torch
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    calls = _spies(monkeypatch)
    eng = []
    real_s = muzero_vtree.VectorizedPUCT.search

    def spy_s(self, obs, num_simulations, *a, **k):
        eng.append((len(obs), num_simulations))
        return real_s(self, obs, num_simulations, *a, **k)

    monkeypatch.setattr(muzero_vtree.VectorizedPUCT, "search", spy_s)
    monkeypatch.setattr(ablation, "RayPipeline", _fake_pipe([], 0))
    monkeypatch.setattr(ablation, "ENV_NAME", "Pong-v4")
    monkeypatch.setattr(ablation, "FRAMESKIP", 4)
    monkeypatch.setattr(ablation, "EPISODE_CAP", 8)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setitem(ablation.ARMS, "e2e-g8", {
        **ablation.ARMS[ARM], "TRAINING_BATCHES": 16})
    out = tmp_path / "a.jsonl"
    ablation.run_one("e2e-g8", 0, 2, 1, 10, 1, out, checkpoint_every=1,
                     use_ray=True, cpu_actor=True)
    assert calls["gumbel"] and set(calls["gumbel"]) == {8}
    assert calls["puct"] == []
    assert eng and all(n == 50 for _, n in eng)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["iteration"] for r in rows] == [1, 2]
    assert not any(k.startswith("beh_") for r in rows for k in r)
    torch.set_num_threads(threads)


# --------------------------------------------------------------------------
# the probe's search-agent mode
# --------------------------------------------------------------------------
_PROBE_GLOBALS = {"ENV_NAME": "Pong-v4", "FRAMESKIP": 4, "EPISODE_CAP": 3000,
                  "ACTIONS": 6}
# configure_search's flags that are NOT keys of the probe's arm (arm_config
# would not restore them) -- pinned so a test cannot leak them.
_SEARCH_FLAGS = ("USE_SEARCH_ACTING", "PAPER_TARGETS", "EVAL_PRIOR_ONLY",
                 "SEARCH_ACT_BIRTH_TARGETS", "ACT_SEARCH_SIMS",
                 "USE_GUMBEL_SEARCH", "ACTOR_GUMBEL_SEARCH")


@pytest.fixture
def probe(monkeypatch, tmp_path, request):
    """The probe as a module. Its import sets ablation's Pong globals, torch
    threads and CUDA_VISIBLE_DEVICES for the whole process -- undone here
    so later tests in the same session see the values they had."""
    import torch
    before = {k: getattr(ablation, k) for k in _PROBE_GLOBALS}
    threads = torch.get_num_threads()
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    sys.path.insert(0, str(PROBES))
    try:
        import probe_exhaustive_eval as pe
    finally:
        sys.path.remove(str(PROBES))
        for k, v in before.items():
            setattr(ablation, k, v)
        torch.set_num_threads(threads)
        if cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    for k, v in _PROBE_GLOBALS.items():
        monkeypatch.setattr(ablation, k, v)
    for k in _SEARCH_FLAGS:
        monkeypatch.setattr(muzero, k, getattr(muzero, k))
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    if "PROBE_MAX_NOOP" not in os.environ:
        assert pe.MAX_NOOP == 30            # r1 L4 (P3): the readout's range
    monkeypatch.setattr(pe, "HERE", tmp_path)
    monkeypatch.setattr(pe, "MAX_NOOP", 1)
    monkeypatch.setattr(pe, "ENGINE_SET", False)
    return pe


def _snapshot(pe, tmp_path, it=5):
    """Fresh nets of the probe's own arm, saved as a snapshot."""
    import torch
    with ablation.arm_config(pe.ARM), ablation._run_length_globals(90, 2000):
        run = ablation.Run(pe.ARM, 0, ablation.ARMS[pe.ARM])
        snap = tmp_path / "run-x" / "snaps" / f"weights-iter{it:03d}.pth"
        snap.parent.mkdir(parents=True)
        torch.save({**{n: getattr(run.trainer, n).state_dict()
                       for n in ablation.NET_NAMES},
                    "training_steps": 0, "spec": {}}, snap)
    return snap


@pytest.mark.parametrize("engine", ["gumbel", "puct"])
def test_probe_evaluates_the_search_agent(probe, monkeypatch, tmp_path,
                                          engine):
    import torch
    torch.set_num_threads(2)
    snap = _snapshot(probe, tmp_path)
    calls = _spies(monkeypatch)
    monkeypatch.setattr(probe, "SEARCH_SIMS", 2)
    monkeypatch.setattr(probe, "SEARCH_ENGINE", engine)
    real_play = ablation.Run.play_episode
    monkeypatch.setattr(ablation.Run, "play_episode",
                        lambda self, greedy=False, cap=None:
                        real_play(self, greedy=greedy, cap=40))
    probe.main([str(snap)])
    rec = json.loads((tmp_path /
                      f"probe_exhaustive_eval-run-x-{engine}2-noops1.jsonl")
                     .read_text().splitlines()[0])
    assert rec["variant"] == f"search-{engine}2" and rec["max_noop"] == 1
    assert rec["search"] == {"engine": engine, "sims": 2}
    assert list(rec["games"]) == ["1"]
    if engine == "gumbel":
        assert calls["gumbel"] and set(calls["gumbel"]) == {2}
        assert calls["puct"] == []
    else:
        assert calls["gumbel"] == []
        assert calls["puct"] and set(calls["puct"]) == {(2, False)}  # greedy
    assert muzero.USE_SEARCH_ACTING is False      # arm_config restored it


def test_probe_default_mode_is_still_the_prior_argmax(probe, monkeypatch,
                                                      tmp_path):
    import torch
    torch.set_num_threads(2)
    snap = _snapshot(probe, tmp_path)
    calls = _spies(monkeypatch)
    real_play = ablation.Run.play_episode
    monkeypatch.setattr(ablation.Run, "play_episode",
                        lambda self, greedy=False, cap=None:
                        real_play(self, greedy=greedy, cap=40))
    probe.main([str(snap)])
    rec = json.loads((tmp_path / "probe_exhaustive_eval-run-x-noops1.jsonl")
                     .read_text().splitlines()[0])
    assert rec["variant"] == "raw" and "search" not in rec
    assert not (tmp_path / "probe_exhaustive_eval-run-x.jsonl").exists()
    assert calls == {"gumbel": [], "puct": []}


@pytest.mark.parametrize("engine,sims,acting,msg", [
    ("mcts", 8, False, "must be puct|gumbel"),
    ("gumbel", -1, False, "must be puct|gumbel"),
    ("gumbel", 8, True, "BN statistics are wrong inside a search")])
def test_probe_search_mode_refusals(probe, monkeypatch, engine, sims, acting,
                                    msg):
    monkeypatch.setattr(probe, "SEARCH_SIMS", sims)
    monkeypatch.setattr(probe, "SEARCH_ENGINE", engine)
    monkeypatch.setattr(probe, "ACTING", acting)
    with pytest.raises(SystemExit, match=msg):
        probe.configure_search()


@pytest.mark.parametrize("engine_set,sims,noops,msg", [
    (True, 0, 30, "without PROBE_SEARCH_SIMS"),
    (False, 0, 0, "PROBE_MAX_NOOP must be 1..30"),
    (False, 0, 31, "PROBE_MAX_NOOP must be 1..30")])
def test_probe_mode_refusals(probe, monkeypatch, engine_set, sims, noops, msg):
    """r1 L5: an engine without a budget would silently run the prior
    argmax; a noop range outside the readout's is refused."""
    monkeypatch.setattr(probe, "ENGINE_SET", engine_set)
    monkeypatch.setattr(probe, "SEARCH_SIMS", sims)
    monkeypatch.setattr(probe, "MAX_NOOP", noops)
    with pytest.raises(SystemExit, match=msg):
        probe.main([])


def test_probe_reseeds_each_search_game(probe, monkeypatch):
    """A search agent's result is reproducible: np.random is reseeded per
    noop count (Gumbel's halving draws Gumbel noise)."""
    seeds = []
    monkeypatch.setattr(probe, "SEARCH_SIMS", 8)
    monkeypatch.setattr(np.random, "seed", lambda s: seeds.append(s))
    monkeypatch.setattr(probe, "_noop_env", lambda env: type("E", (), {})())

    class R:
        env = type("Env", (), {"reset": lambda self, seed=None: None})()

        def play_episode(self, greedy=False, cap=None):
            return [], 0.0, 1, True

    for n in (1, 2, 1):
        probe.play(R(), n)
    assert seeds == [1001, 1002, 1001]


# --------------------------------------------------------------------------
# supervisor
# --------------------------------------------------------------------------
def test_supervisor_variant_suites_seed_and_queue():
    tree = ast.parse(SUP.read_text(encoding="utf-8"))

    def lit(name):
        return next(n for n in tree.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", "") == name).value

    arm, stem, flags = ast.literal_eval(lit("VARIANTS"))["ray"]
    assert arm == ARM and stem == "pong-gumbelact8"
    assert flags == ["--ray", "--cpu-actor"]
    aud = ast.literal_eval(lit("AUDITED"))
    pre = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == "preflight")
    suites = [e.value for n in ast.walk(pre) if isinstance(n, ast.List)
              for e in n.elts if isinstance(e, ast.Constant)
              and str(e.value).startswith("tests/")]
    for t in ("tests/test_gumbelact8.py", "tests/test_searchact8.py",
              "tests/test_paper_mode.py", "tests/test_mcts.py",
              "tests/test_reanalyze_refresh.py",
              "tests/test_vtree_communal.py"):
        assert t in suites and t in aud, t
    assert set(suites) <= set(aud)
    assert "benchmarks/probes/probe_exhaustive_eval.py" in aud
    src = SUP.read_text(encoding="utf-8")
    assert f"preflight ok (manifest + quick probe + {len(suites)} suites)" in src
    # each seed queued behind a completion marker, ALERTing if that run failed
    assert ast.literal_eval(lit("SEEDS")) == {0: "2026-10-12", 1: "2026-10-15"}
    loops = [ast.unparse(n) for n in tree.body if isinstance(n, ast.While)]
    assert any("_seed_state" in l and "FAILED" in l and "WAIT_NAME" in l
               for l in loops)
    # r1 L4 (S4-S7): the twin's run command and module set, this run's dir
    twin_tree = ast.parse((ROOT / "benchmarks" / "supervisors" /
                           "supervise_searchact8.py").read_text(encoding="utf-8"))

    def twin_lit(name):
        return next(n for n in twin_tree.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", "") == name).value

    assert ast.unparse(lit("COMMON")) == ast.unparse(twin_lit("COMMON"))
    assert (set(ast.literal_eval(twin_lit("AUDITED")))
            - {"benchmarks/supervisors/supervise_searchact8.py"}) <= set(aud)
    assert ast.unparse(lit("REL")) == (
        "f'benchmarks/results/{_stem}-s{SEED}-{SEEDS[SEED]}'")
    quick = ast.unparse(lit("QUICK"))
    assert "ACTOR_GUMBEL_SEARCH" in quick and "is True" in quick
    r = subprocess.run([sys.executable, str(SUP), "ray", "2"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    assert r.returncode != 0 and "pre-registered for seeds [0, 1]" in r.stderr


def _sup_ns(argv):
    """The supervisor's module-level setup (everything before the
    --write-manifest branch: constants, the per-seed paths, the helpers),
    executed with `argv` -- no preflight, no launch."""
    tree = ast.parse(SUP.read_text(encoding="utf-8"))
    stop = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.If)
                and "--write-manifest" in ast.unparse(n.test))
    code = compile(ast.Module(body=tree.body[:stop], type_ignores=[]),
                   str(SUP), "exec")
    ns = {"__file__": str(SUP), "__name__": "supervise_gumbelact8_probe"}
    old = sys.argv
    sys.argv = list(argv)
    try:
        exec(code, ns)
    finally:
        sys.argv = old
    return ns


@pytest.mark.parametrize("seed, folder, wait_name, wait_dir, wait_ckpt", [
    (0, "pong-gumbelact8-s0-2026-10-12", "SVE seed 0",
     "pong-sve2-s0-2026-10-10",
     "resume-pong-skip4-effzero-repro-faithful-prioract-noredo-temp05-"
     "bnrecal-sve-seed0.ckpt"),
    (1, "pong-gumbelact8-s1-2026-10-15", "Gumbel seed 0",
     "pong-gumbelact8-s0-2026-10-12",
     f"resume-pong-skip4-{ARM}-seed0.ckpt"),
])
def test_supervisor_seed_paths_and_queue(seed, folder, wait_name, wait_dir,
                                         wait_ckpt):
    ns = _sup_ns(["supervise_gumbelact8.py", "ray", str(seed)])
    res = os.path.join(ns["REPO"], "benchmarks", "results")
    assert ns["SEED"] == seed and ns["ARM"] == ARM
    assert ns["REL"] == f"benchmarks/results/{folder}"
    assert ns["BASE"] == os.path.join(res, folder)
    assert ns["CKPT"] == os.path.join(
        res, folder, f"resume-pong-skip4-{ARM}-seed{seed}.ckpt")
    assert ns["COMMON"][ns["COMMON"].index("--seed") + 1] == str(seed)
    assert ns["WAIT_NAME"] == wait_name
    assert ns["WAIT_CKPTS"] == [os.path.join(res, wait_dir, wait_ckpt)]
    # seed 1's wait is Gumbel seed 0 -- never itself
    assert ns["CKPT"] not in ns["WAIT_CKPTS"]


def test_supervisor_write_manifest_mode_resolves_seed_0():
    ns = _sup_ns(["supervise_gumbelact8.py", "--write-manifest"])
    assert ns["SEED"] == 0 and ns["WRITE"] is True
    assert ns["REL"] == "benchmarks/results/pong-gumbelact8-s0-2026-10-12"


def test_supervisor_seed_state_reads_the_waited_runs_marker(tmp_path):
    ns = _sup_ns(["supervise_gumbelact8.py", "ray", "1"])
    st = ns["_seed_state"]
    ck = tmp_path / "resume.ckpt"
    assert st(str(ck)) == "running"                   # nothing yet
    ck.write_bytes(b"x" * 1_000_001)                   # a live checkpoint
    assert st(str(ck)) == "running"
    (tmp_path / "supervisor.log").write_text("CRASH LOOP: 5 deaths\n")
    assert st(str(ck)) == "failed"
    (tmp_path / "supervisor.log").write_text("RUN COMPLETE s0\n")
    assert st(str(ck)) == "done"
    (tmp_path / "supervisor.log").unlink()
    ck.write_bytes(b"marker")                          # shrunk = complete
    assert st(str(ck)) == "done"
