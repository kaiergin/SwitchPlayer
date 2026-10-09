"""Low-sim search acting with birth-search targets (Kai, 2026-10-09):
ACTOR_SEARCH_SIMS, SEARCH_ACT_BIRTH_TARGETS and the arm
effzero-repro-faithful-searchact8-birth. Pins: only the ACTION choice
changes vs prioract2's arm (8-sim self-play search; the 50-sim birth search
still makes every target), the gates, the self-play recording, the skipped
behaviour diagnostics, and the supervisor."""
import ast
import asyncio
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] /
                       "experiments"))
import ablation
import muzero
from tests.test_checkpoint_resume import make_run
from tests.test_prioract2 import _fake_pipe, restore_rngs  # noqa: F401

ARM = "effzero-repro-faithful-searchact8-birth"
TWIN = "effzero-repro-faithful-prioract-noredo"
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUP = ROOT / "benchmarks" / "supervisors" / "supervise_searchact8.py"


def _eff(spec, k):
    return spec[k] if k in spec else getattr(muzero, k, None)


# --------------------------------------------------------------------------
# the arm, defaults, gates
# --------------------------------------------------------------------------
def test_arm_changes_only_who_picks_the_action():
    a, b = ablation.ARMS[TWIN], ablation.ARMS[ARM]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {
        "USE_SEARCH_ACTING", "SEARCH_ACT_BIRTH_TARGETS", "ACTOR_SEARCH_SIMS",
        "PRIOR_ACT_STAGED_TEMP", "EVAL_PRIOR_ONLY", "_doc"}
    assert b["ACTOR_SEARCH_SIMS"] == 8 and _eff(b, "ACT_SEARCH_SIMS") == 50
    assert b["BIRTH_SEARCH"] is True and b["USE_SEARCH_ACTING"] is True
    assert b["_launch"] == {"ray": True, "cpu_actor": True}
    assert not _eff(b, "BN_RECAL_ACTING") and _eff(b, "PRIOR_ACT_TEMP_SCALE") == 1.0


def test_defaults_off():
    assert muzero.ACTOR_SEARCH_SIMS is None
    assert muzero.SEARCH_ACT_BIRTH_TARGETS is False


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


def test_gates_wired_into_run_one():
    import inspect
    src = inspect.getsource(ablation.run_one)
    assert "search_act_birth_gate()" in src and "birth_search_gate()" in src


@pytest.mark.parametrize("sims", [0, -1, 2.5, True])
def test_actor_sims_must_be_a_positive_int(monkeypatch, sims):
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", True)
    monkeypatch.setattr(muzero, "ACTOR_SEARCH_SIMS", sims)
    with pytest.raises(SystemExit, match="ACTOR_SEARCH_SIMS must be"):
        ablation.search_act_birth_gate()


@pytest.mark.parametrize("search_acting", [False, True])
def test_actor_sims_only_with_birth_targets(monkeypatch, search_acting):
    """r1 LOW-3: ACTOR_SEARCH_SIMS outside SEARCH_ACT_BIRTH_TARGETS -- inert
    without search acting, an untested mixed-budget mode with it."""
    monkeypatch.setattr(muzero, "ACTOR_SEARCH_SIMS", 8)
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", search_acting)
    monkeypatch.setattr(muzero, "SEARCH_ACT_BIRTH_TARGETS", False)
    with pytest.raises(SystemExit, match="only with SEARCH_ACT_BIRTH_TARGETS"):
        ablation.search_act_birth_gate()


@pytest.mark.parametrize("off", ["USE_SEARCH_ACTING", "BIRTH_SEARCH",
                                 "PAPER_TARGETS"])
def test_birth_targets_mode_needs_its_regime(monkeypatch, off):
    for k in ("USE_SEARCH_ACTING", "BIRTH_SEARCH", "PAPER_TARGETS",
              "SEARCH_ACT_BIRTH_TARGETS"):
        monkeypatch.setattr(muzero, k, True)
    monkeypatch.setattr(muzero, off, False)
    with pytest.raises(SystemExit, match="SEARCH_ACT_BIRTH_TARGETS needs"):
        ablation.search_act_birth_gate()


def test_birth_search_still_refuses_plain_search_acting(monkeypatch):
    for k in ("USE_SEARCH_ACTING", "BIRTH_SEARCH", "PAPER_TARGETS"):
        monkeypatch.setattr(muzero, k, True)
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", False)   # as the arms
    monkeypatch.setattr(muzero, "SEARCH_ACT_BIRTH_TARGETS", False)
    with pytest.raises(SystemExit, match="SEARCH_ACT_BIRTH_TARGETS"):
        ablation.birth_search_gate()
    monkeypatch.setattr(muzero, "SEARCH_ACT_BIRTH_TARGETS", True)
    ablation.birth_search_gate()


# --------------------------------------------------------------------------
# self-play recording
# --------------------------------------------------------------------------
class _Env:
    """Never terminates on its own (the cap truncates)."""
    def reset(self, seed=None):
        return np.zeros((210, 160, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((210, 160, 3), dtype=np.uint8), 1.0, False, False, {}


class _Searcher:
    def __init__(self):
        self.calls = []
        self.network_manager = type("NM", (), {"training_steps": 0})()

    async def act_search(self, stack, num_simulations=None, discount=None,
                         greedy=False):
        self.calls.append((greedy, num_simulations))
        return 2, np.full(ablation.ACTIONS, 1.0 / ablation.ACTIONS), 0.5


def _run(monkeypatch, birth_mode, eval_prior=False):
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", True)
    monkeypatch.setattr(muzero, "PAPER_TARGETS", True)
    monkeypatch.setattr(muzero, "EVAL_PRIOR_ONLY", eval_prior)
    monkeypatch.setattr(muzero, "SEARCH_ACT_BIRTH_TARGETS", birth_mode)
    monkeypatch.setattr(muzero, "ACTOR_SEARCH_SIMS", 8 if birth_mode else None)
    run = object.__new__(ablation.Run)
    run.env, run.env_steps = _Env(), 0
    run.searcher = _Searcher()
    return run


def test_birth_mode_self_play_searches_8_sims_and_records_plain_steps(monkeypatch):
    run = _run(monkeypatch, True)
    traj, _, steps, terminated = run.play_episode(greedy=False, cap=5)
    assert steps == 5 and not terminated
    assert run.searcher.calls == [(False, 8)] * 5          # no tail search
    assert all(len(r) == 3 for r in traj) and len(traj) == 6
    assert [r[1] for r in traj[:-1]] == [2] * 5            # the search's action


def test_default_search_acting_is_unchanged(monkeypatch):
    run = _run(monkeypatch, False)
    traj, _, steps, _ = run.play_episode(greedy=False, cap=5)
    assert run.searcher.calls == [(False, None)] * 5 + [(True, None)]
    assert all(len(r) == 5 for r in traj)


def test_greedy_searches_keep_the_default_budget(monkeypatch):
    run = _run(monkeypatch, True)
    run.play_episode(greedy=True, cap=3)
    assert run.searcher.calls == [(True, None)] * 3


def test_prior_eval_never_searches(monkeypatch):
    run = _run(monkeypatch, True, eval_prior=True)
    nm = run.searcher.network_manager
    nm.initial_inference = lambda stack: (
        np.full(ablation.ACTIONS, 1.0 / ablation.ACTIONS), 0.0)
    run.play_episode(greedy=True, cap=3)
    assert run.searcher.calls == []


# --------------------------------------------------------------------------
# targets: birth search runs; the behaviour diagnostics are skipped
# --------------------------------------------------------------------------
def _plain_traj(n):
    return [(np.full((12, *muzero.IMAGE_RESHAPE), i % 250, dtype=np.uint8),
             i % ablation.ACTIONS, 0.0) for i in range(n + 1)]


@pytest.mark.parametrize("search_acting,noted", [(True, False), (False, True)])
def test_birth_search_makes_the_targets_and_behaviour_is_skipped(
        monkeypatch, search_acting, noted):
    monkeypatch.setattr(muzero, "PAPER_TARGETS", True)
    monkeypatch.setattr(muzero, "BIRTH_SEARCH", True)
    monkeypatch.setattr(muzero, "USE_SEARCH_ACTING", search_acting)
    monkeypatch.setattr(muzero, "SEARCH_ACT_BIRTH_TARGETS", search_acting)
    run = make_run()
    births, notes = [], []

    def birth(traj, terminated, nets=None, prior_nets=None):
        births.append(len(traj))
        return [(o, a, r, np.full(ablation.ACTIONS, 1.0 / ablation.ACTIONS), 0.0)
                for (o, a, r) in traj]

    run._birth_search = birth
    run._note_behaviour = lambda acted, born: notes.append(len(acted))
    out = run.paper_targets(_plain_traj(4), terminated=True)
    assert births == [5] and len(out) == 5
    assert bool(notes) is noted


# --------------------------------------------------------------------------
# run_one end to end (in-process fake pipeline, real searcher)
# --------------------------------------------------------------------------
def test_run_one_ray_loop_acts_with_8_sims_and_birth_targets(
        monkeypatch, tmp_path, restore_rngs):  # noqa: F811
    import torch
    torch.set_num_threads(2)
    sims = []
    real = muzero.MCTSHelper.act_search

    async def spy(self, obs_stack, num_simulations=None, discount=None,
                  greedy=False):
        sims.append((greedy, num_simulations))
        return await real(self, obs_stack, num_simulations=num_simulations,
                          discount=discount, greedy=greedy)

    monkeypatch.setattr(muzero.MCTSHelper, "act_search", spy)
    births = []
    real_birth = ablation.Run._birth_search

    def spy_birth(self, traj, terminated, nets=None, prior_nets=None):
        births.append(len(traj))
        return real_birth(self, traj, terminated, nets=nets,
                          prior_nets=prior_nets)

    monkeypatch.setattr(ablation.Run, "_birth_search", spy_birth)
    monkeypatch.setattr(ablation, "RayPipeline", _fake_pipe([], 0))
    monkeypatch.setattr(ablation, "ENV_NAME", "Pong-v4")
    monkeypatch.setattr(ablation, "FRAMESKIP", 4)
    monkeypatch.setattr(ablation, "EPISODE_CAP", 8)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setitem(ablation.ARMS, "e2e-sa8", {
        **ablation.ARMS[ARM], "TRAINING_BATCHES": 16})
    out = tmp_path / "a.jsonl"
    ablation.run_one("e2e-sa8", 0, 2, 1, 10, 1, out, checkpoint_every=1,
                     use_ray=True, cpu_actor=True)
    selfplay = [s for g, s in sims if not g]
    assert selfplay and set(selfplay) == {8}               # the actor's budget
    assert births and all(b == 9 for b in births)          # birth made targets
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["iteration"] for r in rows] == [1, 2]
    assert not any(k.startswith("beh_") for r in rows for k in r)


# --------------------------------------------------------------------------
# supervisor
# --------------------------------------------------------------------------
def test_supervisor_variant_suites_and_seed():
    tree = ast.parse(SUP.read_text(encoding="utf-8"))
    variants = ast.literal_eval(next(
        n for n in tree.body if isinstance(n, ast.Assign)
        and getattr(n.targets[0], "id", "") == "VARIANTS").value)
    arm, stem, flags = variants["ray"]
    assert arm == ARM and stem == "pong-searchact8"
    assert flags == ["--ray", "--cpu-actor"]
    aud = ast.literal_eval(next(
        n for n in tree.body if isinstance(n, ast.Assign)
        and getattr(n.targets[0], "id", "") == "AUDITED").value)
    pre = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == "preflight")
    suites = [e.value for n in ast.walk(pre) if isinstance(n, ast.List)
              for e in n.elts if isinstance(e, ast.Constant)
              and str(e.value).startswith("tests/")]
    for t in ("tests/test_searchact8.py", "tests/test_reanalyze_refresh.py",
              "tests/test_vtree_communal.py"):
        assert t in suites and t in aud, t
    assert set(suites) <= set(aud)
    r = subprocess.run([sys.executable, str(SUP), "ray", "1"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=60)
    assert r.returncode != 0 and "pre-registered as SEED 0" in r.stderr



def test_birth_engine_runs_50_sims_and_actor_runs_8(
        monkeypatch, tmp_path, restore_rngs):  # noqa: F811
    """r1 LOW-2 (the auditor's test): through run_one --ray, the BIRTH
    engine searches every position at ACT_SEARCH_SIMS 50 (noise on the acted
    positions) while the actor's own searches run 8 sims, batch 1, noise."""
    import muzero_vtree
    import torch
    torch.set_num_threads(2)
    eng, act = [], []
    real_s = muzero_vtree.VectorizedPUCT.search

    def spy_s(self, obs, num_simulations, *a, **k):
        eng.append((len(obs), num_simulations, k.get("root_noise")))
        return real_s(self, obs, num_simulations, *a, **k)

    monkeypatch.setattr(muzero_vtree.VectorizedPUCT, "search", spy_s)
    real_a = muzero.MCTSHelper.run_full_mcts

    async def spy_a(self, index, num_simulations=128, sim_batches=8,
                    c_puct=1.0, discount=None, max_depth=None,
                    root_noise=True):
        act.append((num_simulations, sim_batches, root_noise))
        return await real_a(self, index, num_simulations=num_simulations,
                            sim_batches=sim_batches, c_puct=c_puct,
                            discount=discount, max_depth=max_depth,
                            root_noise=root_noise)

    monkeypatch.setattr(muzero.MCTSHelper, "run_full_mcts", spy_a)
    monkeypatch.setattr(ablation, "RayPipeline", _fake_pipe([], 0))
    monkeypatch.setattr(ablation, "ENV_NAME", "Pong-v4")
    monkeypatch.setattr(ablation, "FRAMESKIP", 4)
    monkeypatch.setattr(ablation, "EPISODE_CAP", 8)
    monkeypatch.setattr(ablation, "RESULTS", tmp_path)
    monkeypatch.setattr(ablation, "TAG", "")
    monkeypatch.setitem(ablation.ARMS, "e2e-sa8b", {
        **ablation.ARMS[ARM], "TRAINING_BATCHES": 16})
    ablation.run_one("e2e-sa8b", 0, 2, 1, 10, 1, tmp_path / "a.jsonl",
                     checkpoint_every=1, use_ray=True, cpu_actor=True)
    assert eng and all(n == 50 for _, n, _ in eng)
    assert any(noise for _, _, noise in eng)       # acted positions: noise on
    assert set(act) == {(8, 1, True)}
