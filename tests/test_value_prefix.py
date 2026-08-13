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
from muzero import (DISCOUNT, IMAGE_RESHAPE, LATENT_CHANNELS, REWARD_CLIP, UNROLL_STEPS,
                    VALUE_PREFIX_BINS, VALUE_PREFIX_CEILING, VALUE_PREFIX_SUPPORT,
                    discounted_prefix_targets)
from networks import ValuePrefixNetwork
from scalar_categorical import scalar_to_categorical_batch

CPU = torch.device("cpu")
from muzero import LATENT_HW as _HW
LATENT_HW = (_HW, _HW)


def latents(batch=2):
    return torch.randn(batch, LATENT_CHANNELS, *LATENT_HW)


@pytest.fixture
def prefix_net():
    return ValuePrefixNetwork(latent_channels=LATENT_CHANNELS, hidden_size=64,
                              support_size=VALUE_PREFIX_SUPPORT, num_bins=VALUE_PREFIX_BINS)


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
    """Regression guard. The prefix is a SUM, so a ceiling sized like a single
    reward would clamp large prefixes. Supports are half-widths in h space now,
    so the check is on the raw ceiling they are derived from."""
    assert VALUE_PREFIX_CEILING >= REWARD_CLIP * UNROLL_STEPS
    assert VALUE_PREFIX_SUPPORT == pytest.approx(
        float(torch.sign(torch.tensor(1.0)) * (torch.sqrt(torch.tensor(VALUE_PREFIX_CEILING + 1.0)) - 1)
              + 0.001 * VALUE_PREFIX_CEILING), rel=1e-4)


@pytest.mark.parametrize("discount", [0.5, 0.9, 0.997, 1.0])
def test_worst_case_prefix_still_projects_to_full_mass(discount):
    """Every reward at the clip ceiling for the whole unroll -- the largest
    prefix reachable -- must still land inside the support."""
    rewards = torch.full((1, UNROLL_STEPS + 1), float(REWARD_CLIP))
    target = discounted_prefix_targets(rewards, discount=discount)
    dist = scalar_to_categorical_batch(target, VALUE_PREFIX_SUPPORT, VALUE_PREFIX_BINS, CPU)
    assert torch.allclose(dist.sum(-1), torch.ones(1, UNROLL_STEPS + 1))
    assert target.max().item() <= VALUE_PREFIX_CEILING


def test_undersized_support_would_be_caught():
    """Shows the failure this guards against: a support sized for one reward
    saturates once the prefix accumulates, so several unroll steps collapse onto
    the same top bin and become indistinguishable."""
    from scalar_categorical import support_for
    rewards = torch.full((1, UNROLL_STEPS + 1), float(REWARD_CLIP))
    target = discounted_prefix_targets(rewards, discount=1.0)
    dist = scalar_to_categorical_batch(target, support_for(REWARD_CLIP),
                                       VALUE_PREFIX_BINS, CPU)
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
    assert logits.shape == (2, VALUE_PREFIX_BINS)
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
    # Signed support now, so a randomly initialised head may predict negatives.
    assert (prefix.abs() <= VALUE_PREFIX_CEILING * 1.01).all()


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


def test_transform_fixes_the_resolution_that_a_linear_support_loses():
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

    transformed = scalar_to_categorical_batch(typical, VALUE_PREFIX_SUPPORT,
                                              VALUE_PREFIX_BINS, CPU)
    linear = scalar_to_categorical_batch(typical, VALUE_PREFIX_CEILING,
                                         VALUE_PREFIX_BINS, CPU, transform=False)

    assert int((linear.sum(dim=(0, 1)) > 1e-6).sum()) <= 2,         "a linear support of the same ceiling collapses these into one bin"
    assert int((transformed.sum(dim=(0, 1)) > 1e-6).sum()) >= 4,         "h() plus scaled bins must resolve them"


# --------------------------------------------------------------------------
# Optimiser state transport
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to be meaningful")
def test_adam_state_leaves_the_gpu_before_transport():
    """Regression: get_state_dict checked isinstance(tensor) over the TOP level
    of the state dict only. Optimiser tensors nest two levels down under
    state[param_id][key], so none of them were found. That was invisible with
    SGD, which keeps no state -- with Adam it ships exp_avg and exp_avg_sq
    across Ray still resident on the GPU."""
    net = torch.nn.Linear(4, 4).cuda()
    opt = muzero.make_optimizer(net.parameters())
    net(torch.randn(2, 4).cuda()).sum().backward()
    opt.step()

    assert opt.state_dict()["state"], "optimiser should have accumulated state"
    moved = muzero.get_state_dict(opt)

    def every_tensor(obj):
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from every_tensor(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from every_tensor(v)

    tensors = list(every_tensor(moved))
    assert tensors, "expected nested tensors in Adam state"
    assert all(t.device.type == "cpu" for t in tensors)


def test_nested_structures_survive_the_cpu_move():
    nested = {"a": torch.ones(2), "b": [torch.zeros(1), {"c": torch.eye(2)}], "n": 7, "s": "x"}
    out = muzero.to_cpu(nested)
    assert out["n"] == 7 and out["s"] == "x"
    assert torch.equal(out["a"], torch.ones(2))
    assert torch.equal(out["b"][1]["c"], torch.eye(2))


# --------------------------------------------------------------------------
# The tree consumes prefixes
# --------------------------------------------------------------------------
@pytest.fixture
def prefix_tree(monkeypatch):
    """The tree reads the per-step head by default (see USE_VALUE_PREFIX_IN_TREE
    and the ablation behind it), so tests of the prefix path must force it on
    rather than inherit whichever default is current."""
    monkeypatch.setattr(muzero, "USE_VALUE_PREFIX_IN_TREE", True)
    from muzero import MCTSHelper
    return MCTSHelper(action_space=5, device=CPU, network_ref=None,
                      storage_ref=None, enable_logging=False)


def chain(helper, length=6):
    h = w = muzero.LATENT_HW
    traj = [(None, i % 5, float(i), torch.randn(1, LATENT_CHANNELS, h, w),
             np.full(5, 0.2), float(i)) for i in range(length)]
    helper.create_tree_from_game(traj)
    return helper.current_trajectory


def test_edge_reward_is_the_prefix_difference(prefix_tree):
    """The identity the tree relies on. It only holds because the prefix target
    is undiscounted -- with discounting baked in, the difference would be
    gamma^depth times the reward and the backup would double-discount."""
    helper = prefix_tree
    root = chain(helper)[0]
    root.prefix, root.prefix_state = 0.0, None
    helper.batch_expand_nodes([root])

    imagined = [c for c in root.children.values() if not c.is_gameplay]
    assert imagined, "expected imagined children"
    for child in imagined:
        assert child.reward == pytest.approx(child.prefix - root.prefix, abs=1e-5)


def test_prefix_accumulates_down_a_path(prefix_tree):
    helper = prefix_tree
    root = chain(helper)[0]
    root.prefix, root.prefix_state = 0.0, None
    helper.batch_expand_nodes([root])
    child = next(c for c in root.children.values() if not c.is_gameplay)
    helper.batch_expand_nodes([child])

    grandchild = next(c for c in child.children.values() if not c.is_gameplay)
    # Two hops from the root, so the grandchild's prefix is the sum of both edges.
    assert grandchild.prefix == pytest.approx(child.reward + grandchild.reward, abs=1e-4)


def test_children_carry_their_own_lstm_state(prefix_tree):
    helper = prefix_tree
    root = chain(helper)[0]
    root.prefix, root.prefix_state = 0.0, None
    helper.batch_expand_nodes([root])
    states = [c.prefix_state for c in root.children.values() if not c.is_gameplay]
    assert all(s is not None for s in states)
    assert all(s[0].shape[1] == 1 for s in states), "one column of state per node"
    # Different actions must not share a state tensor.
    assert states[0][0].data_ptr() != states[1][0].data_ptr()


def test_search_resets_the_prefix_at_the_root(prefix_tree):
    """The prefix accumulates from the root, so it has to restart at every
    search -- otherwise the second root inherits the first one's total."""
    helper = prefix_tree
    nodes = chain(helper, length=8)
    nodes[2].prefix, nodes[2].prefix_state = 999.0, None
    asyncio.run(helper.run_full_mcts(2, num_simulations=4, sim_batches=4, c_puct=1.25))
    assert nodes[2].prefix == 0.0


def test_per_step_head_feeds_the_tree_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(muzero, "USE_VALUE_PREFIX_IN_TREE", False)
    from muzero import MCTSHelper
    helper = MCTSHelper(action_space=5, device=CPU, network_ref=None,
                        storage_ref=None, enable_logging=False)
    root = chain(helper)[0]
    helper.batch_expand_nodes([root])
    imagined = [c for c in root.children.values() if not c.is_gameplay]
    assert all(c.prefix_state is None for c in imagined)
    assert any(c.reward != 0.0 for c in imagined), "should use the reward head"
