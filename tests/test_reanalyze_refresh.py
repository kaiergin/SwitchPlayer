"""Part 3 of the paper reproduction: the reanalyze machinery.

Covers the Atari wrappers (flicker max-pool, no-op starts), the target-model
batch-time value refresh, and the periodic policy re-searching.
"""
import asyncio
import sys
import types

import numpy as np
import pytest
import torch

import muzero
from atari_wrappers import MaxAndSkipEnv, NoopResetEnv
from tests.test_paper_mode import ACTIONS, fake_paper_traj, make_trainer

CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# Wrappers
# --------------------------------------------------------------------------
import gymnasium as gym


class ScriptedEnv(gym.Env):
    """Minimal real gym.Env: observation = a frame filled with the counter."""

    observation_space = gym.spaces.Box(0, 255, shape=(4, 4), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self):
        self.t = 0
        self._np_random = np.random.default_rng(0)

    def reset(self, **kwargs):
        self.t = 0
        return np.full((4, 4), 0, dtype=np.uint8), {}

    def step(self, action):
        self.t += 1
        return (np.full((4, 4), self.t, dtype=np.uint8),
                1.0, False, False, {})


def test_max_and_skip_pools_the_last_two_frames_and_sums_rewards():
    env = MaxAndSkipEnv(ScriptedEnv(), skip=4)
    env.reset()
    obs, reward, *_ = env.step(0)
    # Raw frames 1..4; the max of the last two is 4; rewards sum to 4.
    assert obs.max() == 4 and obs.min() == 4
    assert reward == pytest.approx(4.0)
    obs, reward, *_ = env.step(0)
    assert obs.max() == 8
    assert reward == pytest.approx(4.0)


def test_flicker_survives_the_max_pool():
    """A sprite present in only one of the last two frames must survive."""

    class Flicker(ScriptedEnv):
        def step(self, action):
            self.t += 1
            frame = np.zeros((4, 4), dtype=np.uint8)
            if self.t % 2 == 0:      # sprite present on even frames only
                frame[2, 2] = 255
            return frame, 0.0, False, False, {}

    env = MaxAndSkipEnv(Flicker(), skip=4)
    env.reset()
    obs, *_ = env.step(0)
    assert obs[2, 2] == 255, "flickering sprite lost by the skip"


def test_noop_reset_takes_between_one_and_thirty_noops():
    counts = []

    class Counting(ScriptedEnv):
        def step(self, action):
            assert action == 0
            counts.append(1)
            return super().step(action)

    env = NoopResetEnv(Counting(), noop_max=30)
    for _ in range(5):
        counts.clear()
        env.reset()
        assert 1 <= len(counts) <= 30


# --------------------------------------------------------------------------
# Target-model value refresh
# --------------------------------------------------------------------------
def obs_traj(rewards, root_values, truncated=False):
    traj = fake_paper_traj(rewards, root_values, truncated=truncated)
    for t in traj:
        t["obs"] = np.random.randint(0, 255, (12, *muzero.IMAGE_RESHAPE),
                                     dtype=np.uint8)
    return traj


def target_value_of(trainer, obs):
    with torch.no_grad():
        latent = trainer.target_representation(
            torch.tensor(obs[None], dtype=torch.float32) / 255.0)
        _, v, _ = trainer.target_prediction(latent)
    return float(v.squeeze())


def test_bootstrap_comes_from_the_lagged_target_model(monkeypatch):
    monkeypatch.setattr(muzero, "USE_TARGET_MODEL", True)
    monkeypatch.setattr(muzero, "PER_ENABLED", False)
    trainer = make_trainer()
    trainer.training_steps = 0
    np.random.seed(7)
    traj = obs_traj([1.0] * 5, [50.0] * 5)   # stored values are 50: must be ignored
    batch = trainer.prepare_batch_paper([traj], 6, 2, ACTIONS)
    for b, ref in enumerate(batch["sample_refs"]):
        start = next(i for i, t in enumerate(traj) if t is ref)
        rsum, g, tail = muzero.paper_value_parts(traj, start, muzero.TD_STEPS,
                                                 muzero.DISCOUNT)
        if tail == len(traj) - 1 and not traj[tail].get("truncated"):
            expected = rsum
        else:
            expected = rsum + g * target_value_of(trainer, traj[tail]["obs"])
        assert batch["target_value"][b, 0] == pytest.approx(expected, rel=1e-4)


def test_target_model_lags_the_online_network(monkeypatch):
    """The whole point: mutating the online nets must NOT move the bootstrap
    until the sync interval elapses."""
    monkeypatch.setattr(muzero, "USE_TARGET_MODEL", True)
    monkeypatch.setattr(muzero, "PER_ENABLED", False)
    monkeypatch.setattr(muzero, "TARGET_MODEL_INTERVAL", 200)
    trainer = make_trainer()
    trainer.training_steps = 0
    np.random.seed(8)
    traj = obs_traj([1.0] * 6, [0.0] * 6, truncated=True)
    trainer.prepare_batch_paper([traj], 4, 2, ACTIONS)   # creates + syncs
    frozen = target_value_of(trainer, traj[3]["obs"])
    with torch.no_grad():
        # Non-uniform noise: a constant shift on zero-init value weights
        # yields equal logits per bin, which still decodes to ~0.
        torch.manual_seed(0)
        for p in trainer.prediction.parameters():
            p.add_(torch.randn_like(p))
    trainer.training_steps = 100     # inside the interval: no re-sync
    trainer.prepare_batch_paper([traj], 4, 2, ACTIONS)
    assert target_value_of(trainer, traj[3]["obs"]) == pytest.approx(frozen)
    trainer.training_steps = 250     # past the interval: re-sync happens
    trainer.prepare_batch_paper([traj], 4, 2, ACTIONS)
    assert target_value_of(trainer, traj[3]["obs"]) != pytest.approx(frozen)


# --------------------------------------------------------------------------
# Policy re-searching
# --------------------------------------------------------------------------
def test_refresh_targets_researches_the_oldest_trajectories(monkeypatch):
    sys.path.insert(0, str(__import__("pathlib").Path(
        __file__).resolve().parent.parent / "experiments"))
    import ablation
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", False)
    run = object.__new__(ablation.Run)
    run.trainer = types.SimpleNamespace(training_steps=500)
    run.searcher = muzero.MCTSHelper(action_space=ACTIONS, device=CPU,
                                     network_ref=None, storage_ref=None)
    np.random.seed(9)
    old = obs_traj([1.0] * 3, [5.0] * 3)
    for t in old:
        t["born_step"] = 0
    new = obs_traj([1.0] * 3, [5.0] * 3)
    for t in new:
        t["born_step"] = 400
    old_policies = [t["target_policy"].copy() for t in old]
    new_policies = [t["target_policy"].copy() for t in new]
    run.replay = [new, old]          # order must not matter; age does
    run.refresh_targets(1)
    assert all(t["born_step"] == 500 for t in old), "oldest not refreshed"
    assert all(t["born_step"] == 400 for t in new), "newest wrongly refreshed"
    changed = any(not np.allclose(a, t["target_policy"])
                  for a, t in zip(old_policies, old))
    assert changed, "refresh did not replace any policy target"
    assert all(np.allclose(a, t["target_policy"])
               for a, t in zip(new_policies, new))
    assert all(t["priority"] > 0 for t in old)
