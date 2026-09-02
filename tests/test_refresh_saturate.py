"""Saturation refresh: fill the actor-play slack with entropy-ordered
refresh, guard-limited, floor-guaranteed. Search correctness rides on
the vtree parity gates; these tests pin the LOOP: when it stops, what
it re-selects, the floor, the accounting, and the arm's loud gates.
"""
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


@pytest.fixture
def sat_world(monkeypatch):
    for key, val in (("USE_GUMBEL_SEARCH", False),
                     ("USE_COMMUNAL_REANALYSIS", False),
                     ("USE_VALUE_PREFIX_IN_TREE", True),
                     ("REFRESH_VTREE", True),
                     ("REFRESH_SATURATE", True),
                     ("REFRESH_ENTROPY_PRIORITY", True),
                     ("REFRESH_NO_SHARPEN_GUARD", True),
                     ("REFRESH_GUARD_STRIKES", 1),
                     ("MCTS_BATCHES", 1),
                     ("ACT_SEARCH_SIMS", 6),
                     ("MAX_DEPTH", 4),
                     ("PREFIX_LSTM_HORIZON", 3),
                     ("VTREE_REFRESH_BATCH", 8),
                     ("OFFPOLICY_DECAY_STEPS", None)):
        monkeypatch.setattr(muzero, key, val)

    def run_with(n_trajs=3):
        np.random.seed(9)
        replay = []
        for k in range(n_trajs):
            traj = obs_traj([1.0] * 3, [5.0] * 3, truncated=True)
            for t in traj:
                t["born_step"] = k
            replay.append(traj)
        run = object.__new__(ablation.Run)
        run.trainer = make_trainer()
        run.trainer.training_steps = 500
        run.replay = replay
        return run

    return run_with


def test_saturate_runs_until_ready_and_reselects(sat_world):
    run = sat_world(3)
    calls = {"n": 0}

    def ready():
        calls["n"] += 1
        return calls["n"] > 2  # slack allows exactly 2 extra polls

    done = run.refresh_saturate(ready, floor=1)
    assert done >= 1
    stamped = [traj for traj in run.replay
               if traj[0].get("refreshed_step") == 500]
    assert len(stamped) == done, \
        "each unit of work is one whole trajectory, stamped on apply"
    stats = run._refresh_offload_stats
    assert stats["refresh_last_iter"] == done
    assert stats["refresh_applied"] == done


def test_saturate_floor_ignores_early_ready(sat_world):
    run = sat_world(2)
    done = run.refresh_saturate(lambda: True, floor=1)
    assert done == 1, "an early-returning actor must not starve refresh"
    done2 = run.refresh_saturate(lambda: True, floor=0)
    assert done2 == 0, "floor=0 + ready actor refreshes nothing"


def test_saturate_stops_when_everything_is_guarded(sat_world):
    run = sat_world(2)
    for traj in run.replay:
        for t in traj:
            t["refresh_guarded"] = True
    done = run.refresh_saturate(lambda: False, floor=1)
    assert done == 0, \
        "a fully-guarded buffer must terminate the loop, not spin"


def test_saturate_accumulates_accounting_across_calls(sat_world):
    run = sat_world(3)
    run.refresh_saturate(lambda: True, floor=1)
    run.refresh_saturate(lambda: True, floor=1)
    assert run._refresh_offload_stats["refresh_applied"] == 2
    assert run._refresh_offload_stats["refresh_last_iter"] == 1


def test_saturate_arm_gates():
    with pytest.raises(SystemExit, match="need --ray"):
        ablation.run_one("effzero-repro-vtree-saturate", 0, 1, 1, 1, 1,
                         None, use_ray=False)
    with pytest.raises(SystemExit, match="requires --cpu-actor"):
        ablation.run_one("effzero-repro-vtree-saturate", 0, 1, 1, 1, 1,
                         None, use_ray=True, cpu_actor=False)


def _gate_case(monkeypatch, key, value, pattern):
    spec = dict(ablation.ARMS["effzero-repro-vtree-saturate"])
    spec[key] = value
    monkeypatch.setitem(ablation.ARMS, "sat-variant", spec)
    with pytest.raises(SystemExit, match=pattern):
        ablation.run_one("sat-variant", 0, 1, 1, 1, 1, None,
                         use_ray=True, cpu_actor=True)


def test_saturate_arm_gate_demands_guard(monkeypatch):
    _gate_case(monkeypatch, "REFRESH_NO_SHARPEN_GUARD", False,
               "NO_SHARPEN_GUARD")


def test_saturate_arm_gate_demands_entropy_priority(monkeypatch):
    _gate_case(monkeypatch, "REFRESH_ENTROPY_PRIORITY", False,
               "ENTROPY_PRIORITY")


def test_saturate_arm_gate_refuses_offload(monkeypatch):
    _gate_case(monkeypatch, "REFRESH_OFFLOAD", True, "exclusive")


def test_saturate_never_repicks_within_a_call_and_is_bounded(sat_world):
    """Audit F3: consecutive strikes must land in different calls,
    and a hung actor bounds the call at one pass over the buffer."""
    run = sat_world(3)
    done = run.refresh_saturate(lambda: False, floor=0)
    assert done == 3,         "never-ready actor: exactly one pass over the buffer, no spin"


def test_saturate_arm_derivation():
    base = ablation.ARMS["effzero-repro-vtree-refresh"]
    arm = ablation.ARMS["effzero-repro-vtree-saturate"]
    assert arm["REFRESH_SATURATE"] is True
    for key in set(base) | set(arm):
        if key in ("REFRESH_SATURATE", "_doc"):
            continue
        assert arm[key] == base[key], f"unexpected drift on {key}"


def test_scheduled_quota_still_runs_without_saturation(sat_world,
                                                       monkeypatch):
    """Sequential/non-saturate configs keep the audited quota path."""
    monkeypatch.setattr(muzero, "REFRESH_SATURATE", False)
    run = sat_world(2)
    run.refresh_targets_vtree(2)
    assert all(t["refreshed_step"] == 500
               for traj in run.replay for t in traj)
