"""Phase-5 async driver units: the refresh-sweep contract's building
blocks and the arm's shape. The driver's loop-level properties
(deadlock freedom, kill/resume, cap semantics) are covered by the
2026-09-06 audit's toy A/Bs; these tests pin what a refactor could
silently break: sweep rotation order, always-replace, newcomer
refresh, the accumulate-until-row accounting, and the arm delta.
"""
import sys

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


@pytest.fixture
def async_world(monkeypatch):
    """sat-stale refresh semantics (rotation, always-replace, no guard)
    under the async arm's flags, on real tiny nets + the real vtree."""
    for key, val in (("USE_GUMBEL_SEARCH", False),
                     ("USE_COMMUNAL_REANALYSIS", False),
                     ("USE_VALUE_PREFIX_IN_TREE", True),
                     ("REFRESH_VTREE", True),
                     ("REFRESH_SATURATE", False),
                     ("REFRESH_ENTROPY_PRIORITY", False),
                     ("REFRESH_NO_SHARPEN_GUARD", False),
                     ("ASYNC_LOOP", True),
                     ("MCTS_BATCHES", 1),
                     ("ACT_SEARCH_SIMS", 6),
                     ("MAX_DEPTH", 4),
                     ("PREFIX_LSTM_HORIZON", 3),
                     ("VTREE_REFRESH_BATCH", 8),
                     ("OFFPOLICY_DECAY_STEPS", None)):
        monkeypatch.setattr(muzero, key, val)

    def run_with(n_trajs=3, stamps=None):
        np.random.seed(9)
        replay = []
        for k in range(n_trajs):
            traj = obs_traj([1.0] * 3, [5.0] * 3, truncated=True)
            for t in traj:
                t["born_step"] = k
            if stamps is not None:
                traj[0]["refreshed_step"] = stamps[k]
            replay.append(traj)
        run = object.__new__(ablation.Run)
        run.trainer = make_trainer()
        run.trainer.training_steps = 500
        run.replay = replay
        return run

    return run_with


def test_sweep_refreshes_count_oldest_in_rotation_order(async_world):
    run = async_world(3, stamps=[5, 1, 3])
    done = run.refresh_sweep(2)
    assert done == 2
    # Oldest stamps (1, then 3) were picked; the freshest (5) untouched.
    assert run.replay[1][0]["refreshed_step"] == 500
    assert run.replay[2][0]["refreshed_step"] == 500
    assert run.replay[0][0]["refreshed_step"] == 5


def test_sweep_always_replaces_targets(async_world):
    run = async_world(1, stamps=[0])
    before = [np.array(t["target_policy"]) for t in run.replay[0]
              if t.get("target_policy") is not None]
    assert run.refresh_sweep(1) == 1
    after = [np.array(t["target_policy"]) for t in run.replay[0]
             if t.get("target_policy") is not None]
    assert len(before) == len(after) and len(after) > 0
    # No guard: every position re-searched and stored (values replaced
    # too; policy equality would only hold by coincidence of the tiny
    # net, so assert the refresh stamped rather than skipped).
    assert run.replay[0][0]["refreshed_step"] == 500


def test_sweep_count_beyond_buffer_refreshes_each_once(async_world):
    run = async_world(2, stamps=[0, 0])
    done = run.refresh_sweep(10)
    assert done == 2
    stats = run._refresh_offload_stats
    assert stats["refresh_applied"] == 2
    assert stats["refresh_steps_total"] == 3 * 2


def test_refresh_trajs_hits_exactly_the_given_newcomers(async_world):
    run = async_world(3, stamps=[7, 7, 7])
    steps = run.refresh_trajs(run.replay[-1:])
    assert steps == 3  # one 3-position trajectory searched
    assert run.replay[2][0]["refreshed_step"] == 500
    assert run.replay[0][0]["refreshed_step"] == 7
    assert run.replay[1][0]["refreshed_step"] == 7


def test_accounting_accumulates_until_row_then_zeroes_window(async_world):
    run = async_world(3, stamps=[0, 1, 2])
    run.refresh_sweep(1)
    run.refresh_trajs(run.replay[-1:])
    stats = run._refresh_offload_stats
    assert stats["refresh_applied"] == 2
    assert stats["refresh_last_iter"] == 2, \
        "async accounting ACCUMULATES between rows (no iteration scope)"
    assert stats["refresh_steps_last_iter"] == 6
    assert stats["refresh_steps_total"] == 6
    run.zero_sweep_window()
    stats = run._refresh_offload_stats
    assert stats["refresh_last_iter"] == 0
    assert stats["refresh_steps_last_iter"] == 0
    assert stats["refresh_applied"] == 2, "cumulative keys survive the row"
    assert stats["refresh_steps_total"] == 6


def test_sweep_on_empty_or_zero_is_a_noop(async_world):
    run = async_world(1)
    assert run.refresh_sweep(0) == 0
    run.replay = []
    assert run.refresh_sweep(3) == 0


def test_admission_rule_bootstrap_floor_and_steady_state():
    admit = ablation.async_admit
    # Floor off: everything admits.
    assert admit(0.0, 0, 0, 1000, False)
    # Bootstrap: empty buffer admits regardless of the ratio.
    assert admit(2.0, 0, 0, 1000, True)
    # First post-bootstrap episode: needs train >= (admitted+ep)*floor.
    assert not admit(2.0, 3999, 1000, 1000, False)
    assert admit(2.0, 4000, 1000, 1000, False)
    # Steady state pins the ADMITTED ratio at >= floor.
    assert not admit(2.0, 20000, 9500, 600, False)   # would be 1.98
    assert admit(2.0, 20200, 9500, 600, False)       # exactly 2.0


def test_throttle_arm_delta_is_the_floor_only():
    spec = ablation.ARMS["effzero-repro-async-throttle"]
    base = ablation.ARMS["effzero-repro-async-fresh"]
    delta = {k: v for k, v in spec.items()
             if k != "_doc" and base.get(k) != v}
    assert delta == {"ASYNC_MIN_TRAIN_PER_ENV_STEP": 2.0}
    assert hasattr(muzero, "ASYNC_MIN_TRAIN_PER_ENV_STEP")
    assert muzero.ASYNC_MIN_TRAIN_PER_ENV_STEP == 0.0, \
        "the floor must default OFF for every other arm"


def test_async_arm_delta_is_driver_only():
    spec = ablation.ARMS["effzero-repro-async-fresh"]
    base = ablation.ARMS["effzero-repro-vtree-sat-stale"]
    delta = {k: v for k, v in spec.items()
             if k != "_doc" and base.get(k) != v}
    assert delta == {"REFRESH_SATURATE": False, "ASYNC_LOOP": True}, \
        "refresh SEMANTICS must stay sat-stale's; only the driver changes"
    for const in ("ASYNC_TRAIN_CHUNK", "ASYNC_SWEEP_TRAIN_STEPS",
                  "ASYNC_REFRESH_ON_COLLECT", "MAX_TRAIN_PER_ENV_STEP"):
        assert hasattr(muzero, const)
