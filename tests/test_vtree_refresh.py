"""Phase-4 integration: vectorized GPU refresh + the CPU latency-lane
actor. The load-bearing test is exact parity between refresh_targets_vtree
and the inline fresh-root refresh path (act_search per position) under
noiseless/tie-first conditions at matched batch shapes -- same
candidates, same eligibility, same _apply_refresh, so any divergence is
in the search itself and the vtree parity contract covers that.
"""
import copy
import sys

import numpy as np
import pytest
import torch

import muzero
import muzero_vtree
from tests.test_paper_mode import ACTIONS, make_trainer
from tests.test_reanalyze_refresh import obs_traj

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parent.parent / "experiments"))
import ablation

CPU = torch.device("cpu")


def make_run(replay):
    run = object.__new__(ablation.Run)
    run.trainer = make_trainer()
    run.trainer.training_steps = 500
    run.searcher = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                                     network_ref=None, storage_ref=None)
    run.replay = replay
    return run


@pytest.fixture
def quiet_search(monkeypatch):
    """Noiseless, deterministic tie-breaks, batch shape 1 on BOTH
    engines -- the conditions under which parity is exact."""
    for key, val in (("USE_GUMBEL_SEARCH", False),
                     ("USE_COMMUNAL_REANALYSIS", False),
                     ("USE_VALUE_PREFIX_IN_TREE", True),
                     ("REFRESH_VTREE", True),
                     ("REFRESH_ENTROPY_PRIORITY", False),
                     ("REFRESH_NO_SHARPEN_GUARD", True),
                     ("REFRESH_GUARD_STRIKES", 2),
                     ("MCTS_BATCHES", 1),
                     ("ACT_SEARCH_SIMS", 10),
                     ("MAX_DEPTH", 8),
                     ("PREFIX_LSTM_HORIZON", 3),
                     ("VTREE_REFRESH_BATCH", 1),
                     ("OFFPOLICY_DECAY_STEPS", None)):
        monkeypatch.setattr(muzero, key, val)
    monkeypatch.setattr(muzero, "add_dirichlet_noise",
                        lambda prior, epsilon=0.25, alpha=0.25: prior)
    monkeypatch.setattr(muzero.random, "choice", lambda seq: seq[0])
    orig = muzero_vtree.VectorizedPUCT.search

    def forced(self, obs, **kw):
        kw["root_noise"] = False
        kw["tie_break"] = "first"
        return orig(self, obs, **kw)

    monkeypatch.setattr(muzero_vtree.VectorizedPUCT, "search", forced)


def test_vtree_refresh_matches_inline_fresh_root_exactly(quiet_search,
                                                         monkeypatch):
    np.random.seed(3)
    base = [obs_traj([1.0] * 5, [5.0] * 5),
            obs_traj([0.0] * 4, [2.0] * 4, truncated=True)]
    for traj in base:
        for t in traj:
            t["born_step"] = 0

    inline_replay = copy.deepcopy(base)
    run_a = make_run(inline_replay)
    monkeypatch.setattr(muzero, "REFRESH_VTREE", False)
    run_a.refresh_targets(2)

    monkeypatch.setattr(muzero, "REFRESH_VTREE", True)
    vtree_replay = copy.deepcopy(base)
    run_b = make_run(vtree_replay)
    run_b.trainer = run_a.trainer  # identical target model
    run_b.refresh_targets_vtree(2)

    for ta, tb in zip(inline_replay, vtree_replay):
        for a, b in zip(ta, tb):
            assert np.allclose(a["target_policy"], b["target_policy"],
                               rtol=0, atol=1e-12)
            assert a["root_value"] == pytest.approx(b["root_value"],
                                                    rel=0, abs=1e-12)
            assert a["priority"] == pytest.approx(b["priority"],
                                                  rel=0, abs=1e-12)
            assert a["refreshed_step"] == b["refreshed_step"] == 500
            assert a.get("refresh_guarded") == b.get("refresh_guarded")
            assert (a.get("refresh_no_drop_strikes")
                    == b.get("refresh_no_drop_strikes"))


def test_vtree_refresh_chunks_and_applies_all(quiet_search, monkeypatch):
    monkeypatch.setattr(muzero, "VTREE_REFRESH_BATCH", 2)
    np.random.seed(5)
    traj = obs_traj([1.0] * 5, [5.0] * 5, truncated=True)
    for t in traj:
        t["born_step"] = 0
    run = make_run([traj])
    run.refresh_targets_vtree(1)
    assert all(t["refreshed_step"] == 500 for t in traj)
    for t in traj:
        assert np.isfinite(t["root_value"])
        assert t["target_policy"].sum() == pytest.approx(1.0)
        assert t["priority"] > 0


def test_vtree_refresh_rotates_no_eligible_candidates(quiet_search):
    np.random.seed(7)
    traj = obs_traj([1.0] * 3, [5.0] * 3)
    for t in traj:
        t["born_step"] = 0
        t["refresh_guarded"] = True
    old_policies = [t["target_policy"].copy() for t in traj]
    run = make_run([traj])
    run.refresh_targets_vtree(1)
    assert all(t["refreshed_step"] == 500 for t in traj), \
        "a fully-guarded candidate must still rotate (stamp advanced)"
    assert all(np.allclose(o, t["target_policy"])
               for o, t in zip(old_policies, traj)), \
        "guarded steps must not be re-searched"


def test_cpu_actor_forces_cpu_device(monkeypatch):
    for key, val in (("USE_ATARI_WRAPPERS", False),
                     ("USE_SEARCH_ACTING", False),
                     ("USE_GUMBEL_SEARCH", False)):
        monkeypatch.setattr(muzero, key, val)
    import ray_actors
    actor = ray_actors.SelfPlayActor(
        {"USE_ATARI_WRAPPERS": False, "USE_SEARCH_ACTING": False,
         "USE_GUMBEL_SEARCH": False},
        ACTIONS, seed=1,
        schedule={"TEMPERATURE_TOTAL_STEPS": 1000,
                  "OFFPOLICY_DECAY_STEPS": 300},
        env_name="Pong-v4", frameskip=30, episode_cap=4, server=None,
        force_cpu=True)
    assert actor.device.type == "cpu"
    assert actor.searcher.device.type == "cpu"
    for name in ("representation", "dynamics", "prediction",
                 "value_prefix"):
        net = getattr(actor.searcher.network_manager, name)
        assert next(net.parameters()).device.type == "cpu"


def test_cpu_actor_cli_gates():
    with pytest.raises(SystemExit, match="requires --ray"):
        ablation.run_one("effzero-repro-vtree-refresh", 0, 1, 1, 1, 1,
                         None, use_ray=False, cpu_actor=True)
    with pytest.raises(SystemExit, match="excludes"):
        ablation.run_one("effzero-repro-vtree-refresh", 0, 1, 1, 1, 1,
                         None, use_ray=True, cpu_actor=True, ray_workers=4)


def test_vtree_communal_arm_is_one_flag_from_archfix():
    """effzero-repro-vtree-communal: archfix's recipe with ONLY
    REFRESH_VTREE flipped (plus its own _doc). Any other delta would
    confound 'archfix's targets at vtree speed'."""
    base = ablation.ARMS["effzero-repro-refresh5x-communal"]
    arm = ablation.ARMS["effzero-repro-vtree-communal"]
    assert arm["REFRESH_VTREE"] is True
    assert arm["USE_COMMUNAL_REANALYSIS"] is True
    assert "REFRESH_VTREE" not in base
    for key in set(base) | set(arm):
        if key in ("REFRESH_VTREE", "_doc"):
            continue
        assert arm[key] == base[key], f"unexpected drift on {key}"


def test_vtree_arm_derivation():
    base = ablation.ARMS["effzero-repro-refresh5x-communal"]
    arm = ablation.ARMS["effzero-repro-vtree-refresh"]
    changed = {"USE_COMMUNAL_REANALYSIS", "REFRESH_VTREE",
               "REFRESH_ENTROPY_PRIORITY", "REFRESH_NO_SHARPEN_GUARD",
               "POLICY_REFRESH_COUNT", "_doc"}
    assert arm["REFRESH_VTREE"] is True
    assert arm["USE_COMMUNAL_REANALYSIS"] is False
    assert arm["POLICY_REFRESH_COUNT"] == 20
    for key in set(base) | set(arm):
        if key in changed:
            continue
        assert arm[key] == base[key], f"unexpected drift on {key}"
