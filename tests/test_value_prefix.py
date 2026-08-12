"""End-to-end value prefix (EfficientZero 4.2).

The prefix is the discounted reward SUM over an unroll, predicted in one shot by
an LSTM instead of as per-step rewards that are added up afterwards. The point
is to sidestep state aliasing: the model can be vague about *when* a reward
lands while staying accurate about *whether* it lands.
"""
import asyncio

import numpy as np
import pytest
import torch

import muzero
from muzero import (DISCOUNT, IMAGE_RESHAPE, LATENT_CHANNELS, NUM_BINS, REWARD_CLIP,
                    UNROLL_STEPS, VALUE_PREFIX_SUPPORT, discounted_prefix_targets)
from networks import ValuePrefixNetwork
from scalar_categorical import scalar_to_categorical_batch

CPU = torch.device("cpu")
LATENT_HW = (IMAGE_RESHAPE[0] // 4, IMAGE_RESHAPE[1] // 4)


def latents(batch=2):
    return torch.randn(batch, LATENT_CHANNELS, *LATENT_HW)


@pytest.fixture
def prefix_net():
    return ValuePrefixNetwork(latent_channels=LATENT_CHANNELS, hidden_size=64,
                              support_size=VALUE_PREFIX_SUPPORT, num_bins=NUM_BINS)


# --------------------------------------------------------------------------
# Prefix targets
# --------------------------------------------------------------------------
def test_prefix_at_step_zero_is_always_zero():
    """Nothing is accumulated before the unroll starts. rewards[:, 0] is the
    reward for entering the state the unroll begins from, so it belongs to no
    prefix."""
    rewards = torch.tensor([[99.0, 1.0, 2.0, 3.0]])
    assert discounted_prefix_targets(rewards)[0, 0].item() == 0.0


def test_prefix_is_the_discounted_running_sum():
    rewards = torch.tensor([[0.0, 10.0, 20.0, 30.0]])
    g = 0.5
    out = discounted_prefix_targets(rewards, discount=g)[0]
    assert out[1].item() == pytest.approx(10.0)
    assert out[2].item() == pytest.approx(10.0 + g * 20.0)
    assert out[3].item() == pytest.approx(10.0 + g * 20.0 + g * g * 30.0)


def test_prefix_shape_matches_rewards():
    rewards = torch.zeros(7, UNROLL_STEPS + 1)
    assert discounted_prefix_targets(rewards).shape == (7, UNROLL_STEPS + 1)


def test_prefix_is_monotonic_for_non_negative_rewards():
    rewards = torch.rand(4, UNROLL_STEPS + 1) * 100
    out = discounted_prefix_targets(rewards)
    assert (out[:, 1:] >= out[:, :-1] - 1e-5).all()


def test_undiscounted_prefix_is_a_plain_cumsum():
    rewards = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    out = discounted_prefix_targets(rewards, discount=1.0)[0]
    assert out.tolist() == pytest.approx([0.0, 1.0, 3.0, 6.0, 10.0])


def test_low_discount_collapses_the_prefix_onto_early_steps():
    """Documents why a heavily discounted gamma is the wrong A/B arm for this
    change: at gamma=0.5 the last three of five unroll steps contribute under a
    fifth of the prefix, so the horizon the value prefix exists to fix is gone.
    """
    rewards = torch.ones(1, UNROLL_STEPS + 1)
    late_share = {}
    for g in (0.5, 0.997):
        out = discounted_prefix_targets(rewards, discount=g)[0]
        late_share[g] = (out[-1] - out[2]).item() / out[-1].item()
    assert late_share[0.5] < 0.25
    assert late_share[0.997] > 0.55


# --------------------------------------------------------------------------
# Support sizing -- the trap flagged in ROADMAP.md
# --------------------------------------------------------------------------
def test_support_covers_a_full_unroll_of_clipped_rewards():
    """Regression guard. The prefix is a SUM, so sizing its support like a single
    reward (400) would clamp large prefixes to the top bin. Before 0ad73607 that
    produced an all-zero target and no gradient at all."""
    assert VALUE_PREFIX_SUPPORT >= REWARD_CLIP * UNROLL_STEPS


@pytest.mark.parametrize("discount", [0.5, 0.9, 0.997, 1.0])
def test_worst_case_prefix_still_projects_to_full_mass(discount):
    """Every reward at the clip ceiling for the whole unroll -- the largest
    prefix reachable -- must still land inside the support."""
    rewards = torch.full((1, UNROLL_STEPS + 1), float(REWARD_CLIP))
    target = discounted_prefix_targets(rewards, discount=discount)
    dist = scalar_to_categorical_batch(target, VALUE_PREFIX_SUPPORT, NUM_BINS, CPU)
    assert torch.allclose(dist.sum(-1), torch.ones(1, UNROLL_STEPS + 1))
    assert target.max().item() <= VALUE_PREFIX_SUPPORT


def test_undersized_support_would_be_caught():
    """Shows the failure this guards against: sized like a single reward, the
    worst-case prefix saturates the ceiling and several steps collapse onto the
    same top bin."""
    rewards = torch.full((1, UNROLL_STEPS + 1), float(REWARD_CLIP))
    target = discounted_prefix_targets(rewards, discount=1.0)
    dist = scalar_to_categorical_batch(target, REWARD_CLIP, NUM_BINS, CPU)
    saturated = (dist[0, :, -1] > 0.99).sum().item()
    assert saturated >= 3, "undersized support should crush several steps together"


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------
def test_forward_shapes(prefix_net):
    prefix_net.eval()
    state = prefix_net.initial_state(2, CPU)
    prefix, logits, state = prefix_net(latents(2), state)
    assert prefix.shape == (2, 1)
    assert logits.shape == (2, NUM_BINS)
    assert state[0].shape == (1, 2, 64) and state[1].shape == (1, 2, 64)


def test_initial_state_is_zeroed(prefix_net):
    """The hidden state *is* the accumulated reward, so a fresh unroll must start
    from nothing."""
    h, c = prefix_net.initial_state(3, CPU)
    assert torch.count_nonzero(h) == 0 and torch.count_nonzero(c) == 0


def test_hidden_state_actually_advances(prefix_net):
    prefix_net.eval()
    state = prefix_net.initial_state(2, CPU)
    _, _, state = prefix_net(latents(2), state)
    assert torch.count_nonzero(state[0]) > 0


def test_prediction_depends_on_history_not_just_the_latest_latent(prefix_net):
    """The whole point of the LSTM. Feed the same latent from two different
    histories; identical outputs would mean the recurrence is doing nothing."""
    prefix_net.eval()
    same = latents(1)
    with torch.no_grad():
        a = prefix_net(same, prefix_net.initial_state(1, CPU))[0]
        mid = prefix_net(latents(1), prefix_net.initial_state(1, CPU))[2]
        b = prefix_net(same, mid)[0]
    assert not torch.allclose(a, b, atol=1e-6)


def test_single_row_batch_works_in_eval(prefix_net):
    """Search evaluates one node at a time; BatchNorm must not blow up on a
    batch of one."""
    prefix_net.eval()
    with torch.no_grad():
        prefix, _, _ = prefix_net(latents(1), prefix_net.initial_state(1, CPU))
    assert prefix.shape == (1, 1) and torch.isfinite(prefix).all()


def test_scalar_decodes_inside_the_support(prefix_net):
    prefix_net.eval()
    with torch.no_grad():
        prefix, _, _ = prefix_net(latents(4), prefix_net.initial_state(4, CPU))
    assert (prefix >= 0).all() and (prefix <= VALUE_PREFIX_SUPPORT).all()


def test_gradients_reach_the_lstm(prefix_net):
    prefix_net.train()
    state = prefix_net.initial_state(4, CPU)
    _, logits, _ = prefix_net(latents(4), state)
    logits.sum().backward()
    assert prefix_net.lstm.weight_ih_l0.grad is not None
    assert prefix_net.lstm.weight_ih_l0.grad.abs().sum() > 0


# --------------------------------------------------------------------------
# Through train_step
# --------------------------------------------------------------------------
def test_train_step_trains_the_prefix_head():
    from tests.test_reanalysis import build_trainer, fake_batch
    trainer = build_trainer()
    before = [p.detach().clone() for p in trainer.value_prefix.parameters()]
    loss = asyncio.run(trainer.train_step(fake_batch(2, UNROLL_STEPS, np.ones((2, UNROLL_STEPS + 1), dtype=bool))))
    after = list(trainer.value_prefix.parameters())
    assert np.isfinite(loss)
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "prefix head did not train"


def test_worst_case_support_wastes_all_resolution_near_zero():
    """The finding from benchmarks/bench.py prefix.

    Sizing the support for the worst case (REWARD_CLIP * UNROLL_STEPS) is
    correct as a ceiling but leaves 40-wide bins, and a random policy's actual
    prefixes are ~0-5. Every realistic target then lands in bin 0, and the head
    has to traverse the whole support from a uniform init to get there -- it
    trains, but far too slowly to be usable. Measured: LSTM MAE 698 at support
    2000 vs 2.45 at support 32, on identical data and training.

    This is why the h(x) transform (ROADMAP priority 3) is a prerequisite for
    the value prefix rather than independent of it: it is what lets one support
    cover the worst case AND resolve small values.
    """
    typical = torch.tensor([[0.0, 0.5, 1.0, 2.0, 3.0, 5.0]])

    wide = scalar_to_categorical_batch(typical, VALUE_PREFIX_SUPPORT, NUM_BINS, CPU)
    bins_used_wide = int((wide.sum(dim=(0, 1)) > 1e-6).sum())

    narrow = scalar_to_categorical_batch(typical, 32, NUM_BINS, CPU)
    bins_used_narrow = int((narrow.sum(dim=(0, 1)) > 1e-6).sum())

    assert bins_used_wide <= 2, "worst-case support collapses realistic values into one bin"
    assert bins_used_narrow >= 5, "a data-scaled support resolves them"
