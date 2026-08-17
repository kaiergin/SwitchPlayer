"""Pure search arithmetic: temperature, Dirichlet noise, MinMaxStats, node Q."""
import numpy as np
import pytest

from muzero import (MCTSNode, MinMaxStats, add_dirichlet_noise, apply_temperature,
                    get_temperature)


# --------------------------------------------------------------------------
# apply_temperature
# --------------------------------------------------------------------------
def test_temperature_one_is_normalised_visit_counts():
    policy = apply_temperature([1, 3, 4, 2], 1.0)
    assert policy == pytest.approx([0.1, 0.3, 0.4, 0.2])


def test_low_temperature_is_argmax():
    policy = apply_temperature([1, 9, 4], 0.0)
    assert policy.tolist() == [0.0, 1.0, 0.0]


def test_lower_temperature_sharpens():
    counts = [1, 2, 4]
    hot, cold = apply_temperature(counts, 1.0), apply_temperature(counts, 0.25)
    assert cold.max() > hot.max()
    assert cold.argmax() == hot.argmax()


@pytest.mark.parametrize("temp", [0.1, 0.5, 1.0, 2.0])
def test_always_a_distribution(temp):
    policy = apply_temperature([5, 0, 3, 1, 7], temp)
    assert policy.sum() == pytest.approx(1.0)
    assert (policy >= 0).all()


# --------------------------------------------------------------------------
# get_temperature
# --------------------------------------------------------------------------
def test_temperature_decays_monotonically_then_floors():
    schedule = [get_temperature(s) for s in (0, 25_000, 50_000, 99_999, 100_000, 500_000)]
    assert schedule[0] == pytest.approx(1.0)
    assert schedule[-1] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(schedule, schedule[1:]))


# --------------------------------------------------------------------------
# add_dirichlet_noise
# --------------------------------------------------------------------------
def test_noise_preserves_normalisation():
    prior = np.array([0.2] * 5)
    assert add_dirichlet_noise(prior).sum() == pytest.approx(1.0)


def test_noise_epsilon_zero_is_identity():
    prior = np.array([0.1, 0.6, 0.3])
    assert add_dirichlet_noise(prior, epsilon=0.0) == pytest.approx(prior)


def test_default_alpha_is_muzeros_atari_setting():
    """alpha was 0.03 -- AlphaGo's value, tuned for 361 moves -- which over 5
    actions made the Dirichlet sample effectively one-hot: ~92% of the noise
    mass landed on a single random action, so root exploration perturbed one
    arm instead of spreading. 0.25 is MuZero's Atari setting. This pins the
    defaults on BOTH noise entry points so they cannot silently diverge."""
    import inspect

    import muzero
    assert inspect.signature(add_dirichlet_noise).parameters["alpha"].default == 0.25
    # add_noise_to_node now defers to the arm-overridable module global.
    assert inspect.signature(muzero.MCTSHelper.add_noise_to_node).parameters["alpha"].default is None
    assert muzero.DIRICHLET_ALPHA == 0.25

    np.random.seed(0)
    concentration = {
        a: np.array([np.random.dirichlet([a] * 5) for _ in range(500)]).max(axis=1).mean()
        for a in (0.03, 0.25)
    }
    assert concentration[0.03] > 0.85          # what the old default did
    assert concentration[0.25] < concentration[0.03] - 0.15


# --------------------------------------------------------------------------
# MinMaxStats
# --------------------------------------------------------------------------
def test_normalize_is_identity_before_a_range_exists():
    stats = MinMaxStats()
    assert stats.normalize(500.0) == 500.0
    stats.update(7.0)
    assert stats.normalize(7.0) == 7.0  # max == min, still no range


def test_a_hair_thin_range_is_floored_not_stretched():
    """EfficientZero Eq 10's epsilon: without the floor, a tiny early-search
    Q spread is stretched across the whole [0,1], amplifying noise into large
    normalized advantages -- which feed the Gumbel sigma policy targets."""
    stats = MinMaxStats()
    stats.update(100.0)
    stats.update(100.0001)
    assert stats.normalize(100.0001) == pytest.approx(1e-4 / 0.01)
    assert stats.normalize(100.0001) < 0.02


def test_normalize_maps_observed_range_onto_unit_interval():
    stats = MinMaxStats()
    for v in (100.0, 900.0, 500.0):
        stats.update(v)
    assert stats.normalize(100.0) == pytest.approx(0.0)
    assert stats.normalize(900.0) == pytest.approx(1.0)
    assert stats.normalize(500.0) == pytest.approx(0.5)


def test_normalisation_keeps_q_comparable_to_the_puct_term():
    """The reason MinMaxStats exists: raw returns are orders of magnitude larger
    than c_puct * prior * sqrt(N) / (1 + n), so selection was always greedy."""
    stats = MinMaxStats()
    for v in (0.0, 3000.0):
        stats.update(v)
    exploration = 1.25 * 0.2 * np.sqrt(32) / (1 + 4)
    assert stats.normalize(1500.0) == pytest.approx(0.5)
    assert abs(stats.normalize(1500.0) - stats.normalize(0.0)) < 10 * exploration


# --------------------------------------------------------------------------
# MCTSNode.q_value
# --------------------------------------------------------------------------
def test_unvisited_node_has_zero_q():
    assert MCTSNode(latent=None, prior=1.0).q_value() == 0.0


def test_q_is_mean_value():
    node = MCTSNode(latent=None, prior=1.0)
    node.visit_count, node.value_sum = 4, 10.0
    assert node.q_value() == pytest.approx(2.5)


def test_virtual_loss_dilutes_q_toward_zero():
    """Virtual loss is a visit contributing no value, so an in-flight simulation
    makes a node look worse and pushes concurrent simulations elsewhere."""
    node = MCTSNode(latent=None, prior=1.0)
    node.visit_count, node.value_sum = 2, 10.0
    before = node.q_value()
    node.virtual_loss = 2
    after = node.q_value()
    assert before == pytest.approx(5.0)
    assert after == pytest.approx(2.5)
    assert after < before


def test_virtual_loss_alone_does_not_divide_by_zero():
    node = MCTSNode(latent=None, prior=1.0)
    node.virtual_loss = 3
    assert node.q_value() == 0.0
