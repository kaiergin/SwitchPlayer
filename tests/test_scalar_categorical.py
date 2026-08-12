"""Two-hot projection of scalars onto a categorical support.

Checks independent properties -- total mass, bin placement, expected-value
round-trip -- rather than comparing against a second copy of the same
arithmetic, which is how the original in-module test managed to agree with a
broken implementation.
"""
import pytest
import torch

from scalar_categorical import scalar_to_categorical_batch

DEVICE = torch.device("cpu")
REWARD_SUPPORT, VALUE_SUPPORT, BINS = 400, 2000, 51


def project(values, support=REWARD_SUPPORT, bins=BINS):
    return scalar_to_categorical_batch(torch.tensor(values, dtype=torch.float32), support, bins, DEVICE)


@pytest.mark.parametrize("value", [0.0, 1.0, 8.0, 10.0, 200.0, 396.0, 399.0, 400.0, 401.0, 5000.0])
def test_mass_always_sums_to_one(value):
    """Regression: scalars at or above support_size produced an all-zero target.

    The clamp forces upper_idx == lower_idx at the top bin, and two plain
    scatter_ calls alias there -- the second wrote upper_weight (0.0) over the
    first's 1.0. Cross-entropy against all zeros contributes no gradient, so
    the reward head could not learn from any reward >= 400, which is precisely
    where gameplay clips them.
    """
    dist = project([[value]])
    assert dist.sum().item() == pytest.approx(1.0), f"{value} projected to mass {dist.sum().item()}"


def test_top_of_support_is_one_hot_on_last_bin():
    dist = project([[REWARD_SUPPORT]])[0, 0]
    assert dist[BINS - 1].item() == pytest.approx(1.0)
    assert dist[: BINS - 1].sum().item() == pytest.approx(0.0)


def test_clipped_reward_matches_support_ceiling():
    """gameplay_process does min(reward, 400) and the reward support is 400, so
    this pairing has to survive projection or every big reward is silently lost."""
    assert project([[400.0]]).sum().item() == pytest.approx(1.0)
    assert torch.equal(project([[400.0]]), project([[1600.0]]))  # both clamp to the ceiling


def test_zero_is_one_hot_on_first_bin():
    dist = project([[0.0]])[0, 0]
    assert dist[0].item() == pytest.approx(1.0)
    assert dist[1:].sum().item() == pytest.approx(0.0)


def test_exact_bin_centre_is_one_hot():
    delta = REWARD_SUPPORT / (BINS - 1)  # 8.0
    dist = project([[3 * delta]])[0, 0]
    assert dist[3].item() == pytest.approx(1.0)


def test_midpoint_splits_between_neighbours():
    delta = REWARD_SUPPORT / (BINS - 1)
    dist = project([[2.5 * delta]])[0, 0]
    assert dist[2].item() == pytest.approx(0.5)
    assert dist[3].item() == pytest.approx(0.5)


@pytest.mark.parametrize("value", [0.0, 37.0, 123.4, 250.0, 399.9, 400.0])
def test_expected_value_round_trips(value):
    """The head decodes with sum(probs * support), so the projection must be its
    inverse to within float error."""
    dist = project([[value]])[0, 0]
    support = torch.linspace(0, REWARD_SUPPORT, BINS)
    assert float((dist * support).sum()) == pytest.approx(value, abs=1e-3)


def test_negative_values_clamp_to_zero_bin():
    """The support is [0, S], so negatives are not representable -- they must
    clamp rather than wrap to a high bin or produce empty mass."""
    dist = project([[-50.0]])[0, 0]
    assert dist[0].item() == pytest.approx(1.0)


def test_shape_is_preserved_with_bin_axis_appended():
    dist = project([[0.0, 10.0, 20.0], [30.0, 40.0, 400.0]])
    assert dist.shape == (2, 3, BINS)
    assert torch.allclose(dist.sum(-1), torch.ones(2, 3))


def test_value_support_ceiling_also_intact():
    dist = scalar_to_categorical_batch(
        torch.tensor([[VALUE_SUPPORT]], dtype=torch.float32), VALUE_SUPPORT, BINS, DEVICE)
    assert dist.sum().item() == pytest.approx(1.0)
