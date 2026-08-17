"""The paper-reproduction data path (the effzero-repro arm's machinery).

Covers the staged visit temperature, the EfficientZero 4.3 age-shrunk
horizon, MuZero's make_target on stored root values, PER sampling with
importance weights and priority writeback, and search-based acting returning
(action, policy, value).
"""
import asyncio

import numpy as np
import pytest
import torch

import muzero
from muzero import MCTSHelper

ACTIONS = 5
CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# Staged acting temperature
# --------------------------------------------------------------------------
def test_visit_temperature_is_staged(monkeypatch):
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", 1000)
    assert muzero.visit_softmax_temperature(0) == 1.0
    assert muzero.visit_softmax_temperature(499) == 1.0
    assert muzero.visit_softmax_temperature(500) == 0.5
    assert muzero.visit_softmax_temperature(749) == 0.5
    assert muzero.visit_softmax_temperature(750) == 0.25
    monkeypatch.setattr(muzero, "TEMPERATURE_TOTAL_STEPS", None)
    assert muzero.visit_softmax_temperature(10 ** 9) == 1.0


# --------------------------------------------------------------------------
# Age-adjusted horizon (EfficientZero 4.3)
# --------------------------------------------------------------------------
def test_paper_horizon_shrinks_with_age(monkeypatch):
    monkeypatch.setattr(muzero, "TD_STEPS", 10)
    monkeypatch.setattr(muzero, "OFFPOLICY_DECAY_STEPS", 300)
    assert muzero.paper_horizon(0) == 10
    assert muzero.paper_horizon(299) == 10
    assert muzero.paper_horizon(300) == 9
    assert muzero.paper_horizon(10 ** 6) == 1     # clipped, never zero
    monkeypatch.setattr(muzero, "OFFPOLICY_DECAY_STEPS", None)
    assert muzero.paper_horizon(10 ** 6) == 10    # disabled


# --------------------------------------------------------------------------
# make_target on stored root values
# --------------------------------------------------------------------------
def fake_paper_traj(rewards, root_values, truncated=False):
    traj = [{"obs": np.zeros(1, dtype=np.uint8), "action": 0, "reward": r,
             "target_policy": np.ones(ACTIONS) / ACTIONS,
             "root_value": v, "target_value": v,
             "has_policy_target": True, "born_step": 0,
             "priority": 1.0}
            for r, v in zip(rewards, root_values)]
    traj[-1]["truncated"] = truncated
    return traj


def test_paper_value_target_matches_the_hand_sum():
    traj = fake_paper_traj([0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 0.0])
    g = 0.9
    # z_0 with horizon 2: r_1 + g*r_2 + g^2 * v_2
    assert muzero.paper_value_target(traj, 0, 2, g) == pytest.approx(
        1.0 + g * 2.0 + g * g * 30.0)


def test_paper_value_target_terminal_is_pure_monte_carlo():
    traj = fake_paper_traj([0.0, 1.0, 2.0], [10.0, 20.0, 0.0])
    # Horizon overruns; terminal stores root_value 0, so z = r_1 + g*r_2.
    assert muzero.paper_value_target(traj, 0, 10, 0.5) == pytest.approx(
        1.0 + 0.5 * 2.0)


def test_paper_value_target_truncated_tail_uses_its_stored_value():
    traj = fake_paper_traj([0.0, 1.0], [10.0, 44.0], truncated=True)
    # idx == tail: the stored (play-time) value stands, per the papers.
    assert muzero.paper_value_target(traj, 1, 5, 0.9) == pytest.approx(44.0)


# --------------------------------------------------------------------------
# Prioritized batching
# --------------------------------------------------------------------------
def make_trainer():
    from tests.test_reanalysis import build_trainer
    return build_trainer()


def test_per_samples_proportionally_and_weights_correct(monkeypatch):
    monkeypatch.setattr(muzero, "PER_ENABLED", True)
    monkeypatch.setattr(muzero, "PER_ALPHA", 1.0)
    monkeypatch.setattr(muzero, "PER_BETA", 1.0)
    trainer = make_trainer()
    traj = fake_paper_traj([1.0] * 4, [5.0] * 4)
    traj[0]["priority"] = 8.0
    traj[1]["priority"] = 1.0
    traj[2]["priority"] = 1.0
    traj[3]["priority"] = 1.0    # padding-side root; still legal (terminated)
    np.random.seed(0)
    batch = trainer.prepare_batch_paper([traj], 600, 2, ACTIONS)
    starts = [next(i for i, t in enumerate(traj) if t is ref)
              for ref in batch["sample_refs"]]
    frac0 = np.mean(np.array(starts) == 0)
    assert 0.62 < frac0 < 0.83, f"priority-8 root sampled {frac0:.0%}, expected ~8/11"
    # IS weights: rarer samples carry larger weights; max normalized to 1.
    w = batch["is_weights"]
    assert w.max() == pytest.approx(1.0)
    w0 = w[np.array(starts) == 0]
    w1 = w[np.array(starts) != 0]
    assert w0.mean() < w1.mean()


def test_paper_batch_values_and_masks(monkeypatch):
    monkeypatch.setattr(muzero, "PER_ENABLED", False)
    trainer = make_trainer()
    trainer.training_steps = 0
    traj = fake_paper_traj([0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 0.0])
    np.random.seed(0)
    batch = trainer.prepare_batch_paper([traj], 8, 2, ACTIONS)
    for b, ref in enumerate(batch["sample_refs"]):
        start = next(i for i, t in enumerate(traj) if t is ref)
        expected = muzero.paper_value_target(traj, start, muzero.TD_STEPS,
                                             muzero.DISCOUNT)
        assert batch["target_value"][b, 0] == pytest.approx(expected)
    # Padding rows carry no policy target; real rows do (except the tail).
    assert batch["has_policy_target"].dtype == bool


def test_train_step_writes_back_priorities(monkeypatch):
    monkeypatch.setattr(muzero, "PER_ENABLED", True)
    trainer = make_trainer()
    trainer.training_steps = 0
    traj = fake_paper_traj([1.0] * 6, [5.0] * 6)
    for t in traj:
        t["obs"] = np.random.randint(0, 255, (12, *muzero.IMAGE_RESHAPE),
                                     dtype=np.uint8)
        t["priority"] = 123.0
    np.random.seed(1)
    batch = trainer.prepare_batch_paper([traj], 4, muzero.UNROLL_STEPS, ACTIONS)
    loss = asyncio.run(trainer.train_step(batch))
    assert np.isfinite(loss)
    updated = [ref["priority"] for ref in batch["sample_refs"]]
    assert all(p != 123.0 for p in updated)
    assert all(p > 0 for p in updated)


# --------------------------------------------------------------------------
# Search-based acting returns (action, policy, value)
# --------------------------------------------------------------------------
def test_puct_act_search_returns_target_material(monkeypatch):
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", False)
    helper = MCTSHelper(action_space=ACTIONS, device=CPU,
                        network_ref=None, storage_ref=None)
    obs = np.random.randint(0, 255, (12, *muzero.IMAGE_RESHAPE), dtype=np.uint8)
    action, policy, value = asyncio.run(
        helper.act_search(obs, num_simulations=8, greedy=True))
    assert 0 <= action < ACTIONS
    assert policy.shape == (ACTIONS,)
    assert policy.sum() == pytest.approx(1.0)
    assert action == int(np.argmax(policy))   # greedy = visit argmax
    assert isinstance(value, float)


def test_act_search_nongreedy_samples_a_legal_action(monkeypatch):
    monkeypatch.setattr(muzero, "USE_GUMBEL_SEARCH", False)
    helper = MCTSHelper(action_space=ACTIONS, device=CPU,
                        network_ref=None, storage_ref=None)
    obs = np.random.randint(0, 255, (12, *muzero.IMAGE_RESHAPE), dtype=np.uint8)
    action, _, _ = asyncio.run(helper.act_search(obs, num_simulations=8))
    assert 0 <= action < ACTIONS


# --------------------------------------------------------------------------
# EfficientZero prefix head dims
# --------------------------------------------------------------------------
def test_prefix_head_bottleneck_shapes_and_zero_init():
    from networks import ValuePrefixNetwork
    net = ValuePrefixNetwork(latent_channels=muzero.LATENT_CHANNELS,
                             hidden_size=64,
                             support_size=muzero.VALUE_PREFIX_SUPPORT,
                             num_bins=muzero.VALUE_PREFIX_BINS,
                             head_channels=16, fc_hidden=32)
    latent = torch.rand(4, muzero.LATENT_CHANNELS,
                        muzero.LATENT_HW, muzero.LATENT_HW)
    net.eval()
    with torch.no_grad():
        prefix, logits, _ = net(latent, net.initial_state(4, CPU))
    assert logits.shape == (4, muzero.VALUE_PREFIX_BINS)
    assert torch.allclose(prefix, torch.zeros_like(prefix), atol=1e-3)
