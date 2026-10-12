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
        "ACTOR_GUMBEL_SEARCH", "_doc"}
    assert b["ACTOR_GUMBEL_SEARCH"] is True
    assert _eff(b, "USE_GUMBEL_SEARCH") is False      # every other search: PUCT
    assert b["ACTOR_SEARCH_SIMS"] == 8 and _eff(b, "ACT_SEARCH_SIMS") == 50
    assert b["SEARCH_ACT_BIRTH_TARGETS"] is True and b["EVAL_PRIOR_ONLY"] is True
    assert b["_launch"] == {"ray": True, "cpu_actor": True}
    # m = min(GUMBEL_MAX_CANDIDATES, actions, sims): all 6 Pong actions
    assert min(_eff(b, "GUMBEL_MAX_CANDIDATES"), 6, 8) == 6


def test_defaults_off():
    assert muzero.ACTOR_GUMBEL_SEARCH is False


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
def helper(monkeypatch):
    import torch
    torch.set_num_threads(2)
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", False)
    h = muzero.MCTSHelper(action_space=6, device=torch.device("cpu"),
                          network_ref=None, storage_ref=None,
                          enable_logging=False)
    for net in ("representation", "prediction", "dynamics", "value_prefix"):
        getattr(h.network_manager, net).eval()
    return h


def _spies(monkeypatch):
    calls = {"gumbel": [], "puct": []}
    real_g = muzero.MCTSHelper.run_gumbel_mcts
    real_p = muzero.MCTSHelper.run_full_mcts

    async def g(self, index, num_simulations=32, discount=None, max_depth=None):
        calls["gumbel"].append(num_simulations)
        return await real_g(self, index, num_simulations=num_simulations,
                            discount=discount, max_depth=max_depth)

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
    monkeypatch.setattr(pe, "HERE", tmp_path)
    monkeypatch.setattr(pe, "MAX_NOOP", 1)
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
    rec = json.loads((tmp_path / f"probe_exhaustive_eval-run-x-{engine}2.jsonl")
                     .read_text().splitlines()[0])
    assert rec["variant"] == f"search-{engine}2"
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
    rec = json.loads((tmp_path / "probe_exhaustive_eval-run-x.jsonl")
                     .read_text().splitlines()[0])
    assert rec["variant"] == "raw" and "search" not in rec
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
    # queued behind SVE seed 0's completion marker
    wait = ast.unparse(lit("WAIT_CKPTS"))
    assert "pong-sve2-s0-2026-10-10" in wait and "sve-seed0.ckpt" in wait
    r = subprocess.run([sys.executable, str(SUP), "ray", "1"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    assert r.returncode != 0 and "pre-registered as SEED 0" in r.stderr
