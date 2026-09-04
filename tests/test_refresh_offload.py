"""Phase 3a: entropy-prioritized refresh, the no-sharpen guard, and the
--ray refresh offload's driver-side machinery.

The offload's core promise is SEMANTIC IDENTITY with the inline path:
the actor runs the same Run._search_traj and the driver applies through
the same Run._apply_refresh, so the parity test below is the load-bearing
one. RefreshActor is deliberately undecorated, so it is driven directly
in-process here; its real-ray transport (worker-process import, ObjectRef
argument deref, actor scheduling) is covered by
test_ray_harness.py::test_refresh_actor_through_a_real_cluster.
"""
import copy
import sys
import types

import numpy as np
import pytest
import torch

import muzero
from tests.test_paper_mode import ACTIONS, make_trainer
from tests.test_reanalyze_refresh import obs_traj

sys.path.insert(0, str(__import__("pathlib").Path(
    __file__).resolve().parent.parent / "experiments"))
import ablation

CPU = torch.device("cpu")


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    return float(-(p * np.log(p + 1e-12)).sum())


def flat_policy():
    return np.full(ACTIONS, 1.0 / ACTIONS)


def sharp_policy():
    p = np.full(ACTIONS, 0.02)
    p[1] = 1.0 - 0.02 * (ACTIONS - 1)
    return p


def make_run(replay):
    run = object.__new__(ablation.Run)
    run.trainer = make_trainer()
    run.trainer.training_steps = 500
    run.searcher = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                                     network_ref=None, storage_ref=None)
    run.replay = replay
    return run


# --------------------------------------------------------------------------
# Candidate selection
# --------------------------------------------------------------------------

def traj_with_entropy(policy, n=3, born=0):
    np.random.seed(hash((born, n)) % 2**31)
    traj = obs_traj([1.0] * n, [5.0] * n)
    for t in traj:
        t["born_step"] = born
        t["target_policy"] = policy.copy()
    return traj


def test_candidates_default_to_oldest_rotation(monkeypatch):
    monkeypatch.setattr(muzero, "REFRESH_ENTROPY_PRIORITY", False)
    old = traj_with_entropy(sharp_policy(), born=0)
    new = traj_with_entropy(flat_policy(), born=400)
    run = make_run([new, old])
    assert run._refresh_candidates(1) == [1], \
        "without the flag, age must decide -- not entropy"


def test_candidates_flattest_first_under_entropy_priority(monkeypatch):
    monkeypatch.setattr(muzero, "REFRESH_ENTROPY_PRIORITY", True)
    sharp_old = traj_with_entropy(sharp_policy(), born=0)
    flat_new = traj_with_entropy(flat_policy(), born=400)
    run = make_run([sharp_old, flat_new])
    assert run._refresh_candidates(1) == [1], \
        "the flat trajectory must outrank the older sharp one"
    assert run._refresh_candidates(2) == [1, 0]


def test_candidates_tie_break_by_rotation_age(monkeypatch):
    monkeypatch.setattr(muzero, "REFRESH_ENTROPY_PRIORITY", True)
    a = traj_with_entropy(flat_policy(), born=400)
    b = traj_with_entropy(flat_policy(), born=0)
    run = make_run([a, b])
    assert run._refresh_candidates(1) == [1], \
        "equal entropy must fall back to oldest-first rotation"


def test_fully_guarded_trajectory_leaves_the_rotation(monkeypatch):
    monkeypatch.setattr(muzero, "REFRESH_ENTROPY_PRIORITY", True)
    monkeypatch.setattr(muzero, "REFRESH_NO_SHARPEN_GUARD", True)
    guarded = traj_with_entropy(flat_policy(), born=0)
    for t in guarded:
        t["refresh_guarded"] = True
    live = traj_with_entropy(sharp_policy(), born=400)
    run = make_run([guarded, live])
    assert run._refresh_candidates(2) == [1], \
        "a trajectory with no eligible steps must be skipped outright"


def test_eligible_indices_respect_tail_and_guard(monkeypatch):
    traj = traj_with_entropy(flat_policy(), n=4)
    traj[-1]["has_policy_target"] = False       # terminated tail
    traj[1]["refresh_guarded"] = True
    run = make_run([traj])
    monkeypatch.setattr(muzero, "REFRESH_NO_SHARPEN_GUARD", False)
    assert run._eligible_indices(traj) == [0, 1, 2], \
        "guard flag must be inert while the feature is off"
    monkeypatch.setattr(muzero, "REFRESH_NO_SHARPEN_GUARD", True)
    assert run._eligible_indices(traj) == [0, 2]


# --------------------------------------------------------------------------
# The no-sharpen guard inside _apply_refresh
# --------------------------------------------------------------------------

def test_guard_v2_strikes_only_stable_sharp(monkeypatch):
    """Guard v2: a strike requires the NEW target to be sharp (H <=
    SHARP_H) and to REPRODUCE the stored one (KL <= KL_TOL). The fresh
    result is always applied regardless."""
    monkeypatch.setattr(muzero, "REFRESH_NO_SHARPEN_GUARD", True)
    monkeypatch.setattr(muzero, "REFRESH_GUARD_STRIKES", 1)
    traj = traj_with_entropy(sharp_policy(), n=2)
    run = make_run([traj])
    # Step 0: sharp target reproduced exactly -> stable-sharp -> guarded.
    # Step 1: FLAT result -> can never strike, stays eligible.
    run._apply_refresh(traj, {0: (sharp_policy(), -1.0),
                              1: (flat_policy(), -2.0)}, now=500)
    assert traj[0].get("refresh_guarded") is True
    assert np.allclose(traj[0]["target_policy"], sharp_policy())
    assert "refresh_guarded" not in traj[1]
    assert np.allclose(traj[1]["target_policy"], flat_policy()),         "the guard must not block application"
    assert all(t["refreshed_step"] == 500 for t in traj)


def test_guard_v2_flat_targets_never_strike(monkeypatch):
    """The test-2 failure mode: early flat targets re-searched to flat
    must STAY ELIGIBLE -- their flatness indicts the model, and freezing
    them trains the policy on weak-model targets at ~50x reuse."""
    monkeypatch.setattr(muzero, "REFRESH_NO_SHARPEN_GUARD", True)
    monkeypatch.setattr(muzero, "REFRESH_GUARD_STRIKES", 1)
    traj = traj_with_entropy(flat_policy(), n=1)
    run = make_run([traj])
    for step in range(4):
        run._apply_refresh(traj, {0: (flat_policy(), -1.0)},
                           now=100 * (step + 1))
    assert "refresh_guarded" not in traj[0]
    assert "refresh_no_drop_strikes" not in traj[0]


def test_guard_v2_redirect_resets_strikes(monkeypatch):
    """The collapsed-policy fix: a sharp target redirected onto a
    DIFFERENT action (sharp -> sharp, dH ~ 0, KL large) is PROGRESS and
    resets the count -- a correctable collapse is never frozen."""
    monkeypatch.setattr(muzero, "REFRESH_NO_SHARPEN_GUARD", True)
    monkeypatch.setattr(muzero, "REFRESH_GUARD_STRIKES", 2)
    traj = traj_with_entropy(sharp_policy(), n=1)  # sharp on action 1
    run = make_run([traj])
    run._apply_refresh(traj, {0: (sharp_policy(), -1.0)}, now=100)
    assert traj[0]["refresh_no_drop_strikes"] == 1
    redirect = np.full(ACTIONS, 0.02)
    redirect[3] = 1.0 - 0.02 * (ACTIONS - 1)  # sharp on action 3
    run._apply_refresh(traj, {0: (redirect, -1.0)}, now=200)
    assert "refresh_no_drop_strikes" not in traj[0],         "a redirect must reset, not strike"
    assert "refresh_guarded" not in traj[0]
    # Two consecutive reproductions of the NEW answer then retire it.
    run._apply_refresh(traj, {0: (redirect.copy(), -1.0)}, now=300)
    run._apply_refresh(traj, {0: (redirect.copy(), -1.0)}, now=400)
    assert traj[0].get("refresh_guarded") is True


def test_guard_off_never_marks(monkeypatch):
    monkeypatch.setattr(muzero, "REFRESH_NO_SHARPEN_GUARD", False)
    traj = traj_with_entropy(sharp_policy(), n=1)
    run = make_run([traj])
    run._apply_refresh(traj, {0: (flat_policy(), -1.0)}, now=500)
    assert "refresh_guarded" not in traj[0]


def test_offloaded_search_and_apply_match_inline(monkeypatch):
    """The load-bearing test: records shipped the way request_refresh
    ships them, searched the way RefreshActor searches them (shell +
    Run._search_traj on the target nets), applied via _apply_refresh,
    must produce EXACTLY what the inline refresh_targets produces on an
    identical trajectory with identical RNG."""
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", False)
    monkeypatch.setattr(muzero, "USE_COMMUNAL_REANALYSIS", True)
    monkeypatch.setattr(muzero, "COMMUNAL_WINDOW_ROOTS", 2)
    monkeypatch.setattr(muzero, "REFRESH_ENTROPY_PRIORITY", True)
    monkeypatch.setattr(muzero, "REFRESH_NO_SHARPEN_GUARD", True)
    # STRIKES=1 so a single refresh CAN set the guard -- at the default 2
    # the guard-parity assertion below would compare None == None
    # (verification-round F2).
    monkeypatch.setattr(muzero, "REFRESH_GUARD_STRIKES", 1)

    np.random.seed(11)
    base = obs_traj([1.0] * 5, [5.0] * 5)
    for t in base:
        t["born_step"] = 0

    # Inline side.
    inline_traj = copy.deepcopy(base)
    run = make_run([inline_traj])
    np.random.seed(77)
    __import__("random").seed(77)
    run.refresh_targets(1)

    # Offloaded side: same trainer (same target nets), actor-shaped flow.
    offload_traj = copy.deepcopy(base)
    run2 = make_run([offload_traj])
    run2.trainer = run.trainer  # identical target model
    records = [{"obs": t["obs"], "action": t["action"],
                "reward": t["reward"],
                "truncated": t.get("truncated", False),
                "has_policy_target": t.get("has_policy_target", True)}
               for t in offload_traj]
    idxs = run2._eligible_indices(offload_traj)
    shell = object.__new__(ablation.Run)
    shell.searcher = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                                       network_ref=None, storage_ref=None)
    nm = shell.searcher.network_manager
    run2.trainer._ensure_target_model()
    # Reference swap, exactly as the inline path swaps (make_trainer's
    # test-sized nets differ in geometry from the module factories, so a
    # state-dict copy cannot stand in here; the actor's LOAD path runs in
    # test_refresh_actor_runs_in_process with factory-matched geometry).
    for name in ("representation", "prediction", "dynamics", "value_prefix"):
        setattr(nm, name, getattr(run2.trainer, "target_" + name))
    np.random.seed(77)
    __import__("random").seed(77)
    results = ablation.Run._search_traj(shell, records, idxs)
    run2._apply_refresh(offload_traj, results, now=500)

    for a, b in zip(inline_traj, offload_traj):
        assert np.allclose(a["target_policy"], b["target_policy"])
        assert a["root_value"] == pytest.approx(b["root_value"])
        assert a["priority"] == pytest.approx(b["priority"])
        assert a["refreshed_step"] == b["refreshed_step"] == 500
        assert a.get("refresh_guarded") == b.get("refresh_guarded")
        assert (a.get("refresh_no_drop_strikes")
                == b.get("refresh_no_drop_strikes"))


def test_refresh_actor_runs_in_process(monkeypatch):
    """RefreshActor end-to-end without ray (it is undecorated by design):
    constructed with overrides, given weights, returns well-formed
    results for the eligible indices only."""
    for key, val in (("USE_GUMBEL_SEARCH", False),
                     ("USE_COMMUNAL_REANALYSIS", True),
                     ("COMMUNAL_WINDOW_ROOTS", 2),
                     ("MCTS_BATCHES", 1)):
        monkeypatch.setattr(muzero, key, val)  # restore after the actor's setattrs
    import ray_actors
    actor = ray_actors.RefreshActor(
        {"USE_GUMBEL_SEARCH": False, "USE_COMMUNAL_REANALYSIS": True,
         "COMMUNAL_WINDOW_ROOTS": 2, "MCTS_BATCHES": 1},
        ACTIONS, seed=3,
        schedule={"TEMPERATURE_TOTAL_STEPS": 1000,
                  "OFFPOLICY_DECAY_STEPS": None})
    # Donor states from factory-built nets (the production geometry the
    # driver's target model shares); make_trainer's test-sized nets would
    # shape-mismatch the actor's factory-built copies.
    donor = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                              network_ref=None, storage_ref=None)
    states = {n: getattr(donor.network_manager, n).state_dict()
              for n in ("representation", "dynamics", "prediction",
                        "value_prefix")}
    assert actor.set_weights(states, 500) == 500
    np.random.seed(5)
    traj = obs_traj([1.0] * 4, [5.0] * 4)
    records = [{"obs": t["obs"], "action": t["action"],
                "reward": t["reward"], "truncated": False,
                "has_policy_target": True} for t in traj]
    # A TRUNCATED tail carries a genuine search target and stays
    # searchable (a terminated tail's chain node is done and the search
    # correctly refuses it -- that case is filtered upstream by
    # _eligible_indices via has_policy_target False).
    records[-1]["truncated"] = True
    results = actor.refresh(records, [0, 2, 3])
    assert set(results) == {0, 2, 3}
    for pol, val in results.values():
        pol = np.asarray(pol)
        assert pol.shape == (ACTIONS,)
        assert pol.sum() == pytest.approx(1.0)
        assert np.isfinite(val)


# --------------------------------------------------------------------------
# poll_refresh: swap-on-ready, uid identity, eviction drop
# --------------------------------------------------------------------------

class _FakeRay:
    """Stands in for the ray module inside poll_refresh: futures are
    plain dicts {'ready': bool, 'value': results}."""

    @staticmethod
    def wait(futs, timeout=0):
        return ([f for f in futs if f["ready"]],
                [f for f in futs if not f["ready"]])

    @staticmethod
    def get(fut):
        return fut["value"]


def test_poll_refresh_applies_alive_drops_evicted_keeps_cooking(monkeypatch):
    monkeypatch.setattr(muzero, "REFRESH_NO_SHARPEN_GUARD", False)
    alive = traj_with_entropy(flat_policy(), n=2, born=0)
    evicted = traj_with_entropy(flat_policy(), n=2, born=1)
    alive[0]["traj_uid"] = "uid-alive"
    evicted[0]["traj_uid"] = "uid-evicted"
    run = make_run([alive])  # evicted is NOT in replay any more

    pipe = object.__new__(ablation.RayPipeline)
    pipe.ray = _FakeRay
    pipe.run = run
    sharp = sharp_policy()
    pipe.pending_refresh = ([
        ("uid-alive", {"ready": True, "value": {0: (sharp, -2.0)}}),
        ("uid-evicted", {"ready": True, "value": {0: (sharp, -9.0)}}),
        ("uid-alive2", {"ready": False, "value": None}),
    ], 500)
    pipe.refresh_applied = 0
    pipe.refresh_skipped = 3
    pipe.poll_refresh()
    # Applied to the alive trajectory...
    assert np.allclose(alive[0]["target_policy"], sharp)
    assert alive[0]["refreshed_step"] == 500
    # ...and the offload accounting reached the run (3a audit F2).
    stats = run._refresh_offload_stats
    assert stats["refresh_applied"] == 1
    assert stats["refresh_skipped_dispatches"] == 3
    assert stats["refresh_lag_steps"] == run.trainer.training_steps - 500
    # ...evicted result dropped without error, unfinished future kept.
    assert pipe.pending_refresh is not None
    (uid, fut), = pipe.pending_refresh[0]
    assert uid == "uid-alive2"
    assert pipe.pending_refresh[1] == 500
    # Second poll with the future now ready but its uid gone: batch closes.
    fut["ready"] = True
    fut["value"] = {0: (sharp, -1.0)}
    pipe.poll_refresh()
    assert pipe.pending_refresh is None


def test_arm_derivation_matches_control_except_the_package():
    base = ablation.ARMS["effzero-repro-refresh5x-communal"]
    arm = ablation.ARMS["effzero-repro-communal-entropyrefresh"]
    changed = {"POLICY_REFRESH_COUNT", "REFRESH_ENTROPY_PRIORITY",
               "REFRESH_NO_SHARPEN_GUARD", "REFRESH_OFFLOAD", "_doc"}
    assert arm["POLICY_REFRESH_COUNT"] == 20
    for key in set(base) | set(arm):
        if key in changed:
            continue
        assert arm[key] == base[key], f"unexpected drift on {key}"
