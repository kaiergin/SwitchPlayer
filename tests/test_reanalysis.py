"""Reanalysis subsampling: index selection, the policy mask, and the loss.

Searching every timestep makes reanalysis ~2.5x slower than self-play produces
data even with two MCTS actors, so the gameplay buffer overflows and trajectories
are discarded at random (benchmarks/RESULTS.md). Searching a strided subset
closes the gap; the timesteps that were skipped must not be trained on as if
they carried a real policy target.
"""
import asyncio

import numpy as np
import pytest
import torch

import muzero
from muzero import (IMAGE_RESHAPE, LATENT_CHANNELS, MCTSHelper, TrainingHelper,
                    masked_policy_loss)

ACTIONS = 5
CPU = torch.device("cpu")


@pytest.fixture
def helper():
    return MCTSHelper(action_space=ACTIONS, device=CPU,
                      network_ref=None, storage_ref=None, enable_logging=False)


# --------------------------------------------------------------------------
# Index selection
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stride", [1, 2, 3, 4, 8])
def test_selected_fraction_matches_the_stride(helper, stride):
    length = 200
    chosen = helper.select_reanalysis_indices(length, stride=stride)
    assert len(chosen) == pytest.approx(length / stride, abs=1)


def test_stride_one_searches_everything(helper):
    assert helper.select_reanalysis_indices(50, stride=1) == set(range(50))


def test_indices_are_evenly_spaced(helper):
    """A stride rather than a random sample, so no stretch of the episode is
    left unrepresented."""
    chosen = sorted(helper.select_reanalysis_indices(100, stride=4))
    gaps = {b - a for a, b in zip(chosen, chosen[1:])}
    assert gaps == {4}


def test_offset_varies_between_passes(helper):
    """Different offsets each pass, so repeated reanalysis of the buffer
    eventually covers every timestep."""
    offsets = {min(helper.select_reanalysis_indices(100, stride=8)) for _ in range(40)}
    assert len(offsets) > 1


def test_indices_stay_in_range(helper):
    for stride in (1, 3, 7):
        chosen = helper.select_reanalysis_indices(23, stride=stride)
        assert chosen and all(0 <= i < 23 for i in chosen)


@pytest.mark.parametrize("length", [1, 2, 3, 7])
def test_short_trajectory_still_gets_searched(helper, length):
    """Regression: the offset was drawn from range(stride) regardless of the
    trajectory length, so an episode shorter than the stride could draw an
    offset past its end and be reanalysed with nothing searched -- yielding a
    trajectory with no policy targets anywhere."""
    for _ in range(50):
        chosen = helper.select_reanalysis_indices(length, stride=8)
        assert chosen, f"length {length} selected no timesteps"
        assert all(0 <= i < length for i in chosen)


def test_empty_trajectory_selects_nothing(helper):
    assert helper.select_reanalysis_indices(0, stride=4) == set()


# --------------------------------------------------------------------------
# masked_policy_loss
# --------------------------------------------------------------------------
def uniform(batch):
    return torch.full((batch, ACTIONS), 1.0 / ACTIONS)


def test_all_masked_in_matches_a_plain_mean():
    logits = torch.randn(8, ACTIONS)
    targets = uniform(8)
    plain = -torch.sum(targets * torch.log_softmax(logits, dim=1), dim=1).mean()
    assert masked_policy_loss(logits, targets, torch.ones(8)) == pytest.approx(float(plain))


def test_masked_out_positions_do_not_contribute():
    logits = torch.randn(4, ACTIONS)
    targets = uniform(4)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # Same answer as computing over only the masked-in rows.
    subset = masked_policy_loss(logits[::2], targets[::2], torch.ones(2))
    assert masked_policy_loss(logits, targets, mask) == pytest.approx(float(subset), abs=1e-6)


def test_empty_mask_gives_zero_not_nan():
    """Whole batches can miss the searched positions; dividing by a zero count
    would poison the total loss with NaN."""
    loss = masked_policy_loss(torch.randn(4, ACTIONS), uniform(4), torch.zeros(4))
    assert float(loss) == 0.0
    assert not torch.isnan(loss)


def test_placeholder_targets_cannot_drag_the_policy_uniform():
    """The point of the mask. A sharp policy scored against uniform placeholders
    should incur no loss when those positions are masked out."""
    sharp = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    placeholder = uniform(1)
    unmasked = masked_policy_loss(sharp, placeholder, torch.ones(1))
    masked = masked_policy_loss(sharp, placeholder, torch.zeros(1))
    assert float(unmasked) > 1.0, "uniform target does penalise a sharp policy"
    assert float(masked) == 0.0


def test_gradient_flows_only_from_masked_in_rows():
    logits = torch.randn(4, ACTIONS, requires_grad=True)
    mask = torch.tensor([1.0, 0.0, 0.0, 1.0])
    masked_policy_loss(logits, uniform(4), mask).backward()
    assert logits.grad[1].abs().sum() == 0.0
    assert logits.grad[2].abs().sum() == 0.0
    assert logits.grad[0].abs().sum() > 0.0


# --------------------------------------------------------------------------
# prepare_batch carries the mask
# --------------------------------------------------------------------------
def transition(has_target=True):
    return {
        "obs": np.zeros((12, *IMAGE_RESHAPE), dtype=np.uint8),
        "action": 0,
        "reward": 1.0,
        "target_policy": np.full(ACTIONS, 1.0 / ACTIONS),
        "target_value": 5.0,
        "has_policy_target": has_target,
    }


@pytest.fixture
def trainer():
    t = object.__new__(TrainingHelper)
    t.device = CPU
    t.action_space = ACTIONS
    return t


def prepare(trainer, buffer, size=4, unroll=3):
    return asyncio.run(trainer.prepare_batch(buffer, size, unroll, ACTIONS))


def test_mask_is_present_and_shaped(trainer):
    traj = [transition(i % 4 == 0) for i in range(20)]
    batch = prepare(trainer, [traj], size=4, unroll=3)
    assert batch["has_policy_target"].shape == (4, 4)
    assert batch["target_policy"].shape == (4, 4, ACTIONS)


def test_mask_reflects_which_steps_were_searched(trainer):
    """Only every fourth transition carries a target, so at most a quarter of a
    long batch should be masked in."""
    traj = [transition(i % 4 == 0) for i in range(200)]
    batch = prepare(trainer, [traj], size=64, unroll=5)
    fraction = batch["has_policy_target"].mean()
    assert 0.15 < fraction < 0.35


def test_padding_is_masked_out(trainer):
    """Regression-adjacent: trajectory padding used to contribute a uniform
    policy target and a zero value as though they were real."""
    traj = [transition(True) for _ in range(2)]
    batch = prepare(trainer, [traj], size=8, unroll=5)
    # Every sample is padded past index 1, so late positions must be masked.
    assert not batch["has_policy_target"][:, 3:].any()


def test_legacy_transitions_without_the_key_are_treated_as_searched(trainer):
    """Trajectories written before subsampling existed have no such key."""
    legacy = [{k: v for k, v in transition().items() if k != "has_policy_target"}
              for _ in range(20)]
    batch = prepare(trainer, [legacy], size=4, unroll=2)
    # Step 0 is never padding, so every sample's first position is a real
    # transition and must be treated as carrying a target.
    assert batch["has_policy_target"][:, 0].all()


# --------------------------------------------------------------------------
# End to end through train_step
# --------------------------------------------------------------------------
def build_trainer():
    from networks import (DynamicsNetwork, PredictionNetwork, RepresentationNetwork,
                          ValuePrefixNetwork)
    import torch.optim as optim
    t = object.__new__(TrainingHelper)
    t.device = CPU
    t.action_space = ACTIONS
    t.training_steps = 0
    t.representation = RepresentationNetwork(input_channels=12, latent_channels=LATENT_CHANNELS)
    t.dynamics = DynamicsNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                                 support_size=400, num_bins=51)
    t.prediction = PredictionNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                                     support_size=2000, num_bins=51)
    t.value_prefix = ValuePrefixNetwork(latent_channels=LATENT_CHANNELS,
                                        hidden_size=64,
                                        support_size=muzero.VALUE_PREFIX_SUPPORT,
                                        num_bins=51)
    t.prefix_optimizer = muzero.make_optimizer(t.value_prefix.parameters())
    t.rep_optimizer = muzero.make_optimizer(t.representation.parameters())
    t.dyn_optimizer = muzero.make_optimizer(t.dynamics.parameters())
    t.pred_optimizer = muzero.make_optimizer(t.prediction.parameters())
    return t


def fake_batch(size, unroll, mask):
    return {
        "obs": np.random.randint(0, 255, (size, unroll + 1, 12, *IMAGE_RESHAPE), dtype=np.uint8),
        "actions": np.random.randint(0, ACTIONS, (size, unroll)),
        "rewards": np.random.rand(size, unroll + 1) * 10,
        "target_policy": np.full((size, unroll + 1, ACTIONS), 1.0 / ACTIONS),
        "target_value": np.random.rand(size, unroll + 1) * 100,
        "has_policy_target": mask,
    }


def test_train_step_runs_with_a_partial_mask():
    trainer = build_trainer()
    mask = np.zeros((2, 3), dtype=bool)
    mask[:, 0] = True
    loss = asyncio.run(trainer.train_step(fake_batch(2, 2, mask)))
    assert np.isfinite(loss)


def test_train_step_survives_an_all_false_mask():
    """Possible whenever a batch happens to miss every searched position."""
    trainer = build_trainer()
    loss = asyncio.run(trainer.train_step(fake_batch(2, 2, np.zeros((2, 3), dtype=bool))))
    assert np.isfinite(loss), "an empty mask must not produce NaN"


def test_masking_changes_the_loss():
    trainer = build_trainer()
    torch.manual_seed(0)
    batch = fake_batch(2, 2, np.ones((2, 3), dtype=bool))
    full = asyncio.run(trainer.train_step(batch))

    trainer = build_trainer()
    torch.manual_seed(0)
    batch["has_policy_target"] = np.zeros((2, 3), dtype=bool)
    none = asyncio.run(trainer.train_step(batch))
    assert full != none
