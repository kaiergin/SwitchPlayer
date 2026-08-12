"""Scalar <-> categorical encoding: the h transform and the two-hot projection.

Checks independent properties -- invertibility, total mass, expected-value
round-trip, bin resolution -- rather than comparing against a second copy of the
same arithmetic, which is how the original in-module test managed to agree with
a broken implementation.
"""
import pytest
import torch

from muzero import (REWARD_BINS, REWARD_CLIP, REWARD_SUPPORT, VALUE_PREFIX_BINS,
                    VALUE_PREFIX_CEILING, VALUE_PREFIX_SUPPORT)
from scalar_categorical import (create_support, h_inverse, h_transform,
                                scalar_to_categorical_batch, support_for)

DEVICE = torch.device("cpu")
# The real configuration, so these test what actually runs.
BINS = REWARD_BINS
REWARD_CEILING, PREFIX_CEILING = REWARD_CLIP, VALUE_PREFIX_CEILING
PREFIX_BINS, PREFIX_SUPPORT = VALUE_PREFIX_BINS, VALUE_PREFIX_SUPPORT


def project(values, support=REWARD_SUPPORT, bins=BINS):
    return scalar_to_categorical_batch(torch.tensor(values, dtype=torch.float32),
                                       support, bins, DEVICE)


def decode(dist, support_size=REWARD_SUPPORT, bins=None):
    """Expectation over the support, then undo h -- what the heads do."""
    support = create_support(support_size, BINS if bins is None else bins, DEVICE)
    return h_inverse((dist * support).sum(-1))


# --------------------------------------------------------------------------
# The h transform
# --------------------------------------------------------------------------
@pytest.mark.parametrize("x", [-9999.0, -2000.0, -400.0, -37.5, -1.0, 0.0,
                               1.0, 37.5, 400.0, 2000.0, 9999.0])
def test_h_is_invertible(x):
    t = torch.tensor([x])
    assert h_inverse(h_transform(t)).item() == pytest.approx(x, rel=1e-4, abs=1e-3)


def test_h_fixes_zero():
    assert h_transform(torch.zeros(1)).item() == 0.0


def test_h_is_odd():
    x = torch.tensor([3.0, 40.0, 700.0])
    assert torch.allclose(h_transform(-x), -h_transform(x), atol=1e-5)


def test_h_is_strictly_increasing():
    x = torch.tensor([-500.0, -50.0, -1.0, 0.0, 1.0, 50.0, 500.0])
    out = h_transform(x)
    assert (out[1:] > out[:-1]).all()


def test_h_compresses_large_magnitudes():
    """The reason it exists: a 100x range in raw units becomes roughly 10x in
    transformed units, so a fixed bin count stretches much further."""
    small, large = h_transform(torch.tensor([20.0])), h_transform(torch.tensor([2000.0]))
    assert (large / small).item() < 15.0


def test_h_barely_touches_small_values():
    """Compression must not destroy resolution near zero, where most rewards
    actually live."""
    assert h_transform(torch.tensor([1.0])).item() == pytest.approx(0.4152, abs=1e-3)


# --------------------------------------------------------------------------
# Total mass -- the scatter_add_ regression
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value", [0.0, 1.0, 8.0, 200.0, 399.0, 400.0, 401.0,
                                   5000.0, -1.0, -400.0, -5000.0])
def test_mass_always_sums_to_one(value):
    """Regression: scalars at or beyond the support edge produced an all-zero
    target. The clamp forces upper_idx == lower_idx there, and two plain
    scatter_ calls alias -- the second wrote upper_weight (0.0) over the first's
    1.0. Cross-entropy against all zeros contributes no gradient at all."""
    dist = project([[value]])
    assert dist.sum().item() == pytest.approx(1.0), f"{value} projected to mass {dist.sum().item()}"


def test_both_support_edges_are_one_hot():
    for ceiling, sign in [(REWARD_CEILING, 1), (REWARD_CEILING, -1)]:
        dist = project([[sign * ceiling]])[0, 0]
        edge = BINS - 1 if sign > 0 else 0
        assert dist[edge].item() == pytest.approx(1.0)


def test_clipped_reward_matches_the_support_ceiling():
    """gameplay_process does min(reward, REWARD_CLIP) and the reward support is
    derived from the same constant, so this pairing has to survive projection."""
    assert project([[float(REWARD_CEILING)]]).sum().item() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Signed support -- the point of the change
# --------------------------------------------------------------------------
def test_negative_values_are_representable():
    """Regression: the support was linspace(0, S) despite a docstring promising
    -S..S, so every negative value clamped to zero. A head could say "worth
    little" but never "bad" -- which blocks the Tetris design this repo targets."""
    assert decode(project([[-50.0]])).item() == pytest.approx(-50.0, rel=0.15)


def test_negative_and_positive_land_on_opposite_sides():
    neg = project([[-100.0]])[0, 0].argmax().item()
    pos = project([[100.0]])[0, 0].argmax().item()
    centre = (BINS - 1) / 2
    assert neg < centre < pos


def test_zero_sits_at_the_centre_bin():
    dist = project([[0.0]])[0, 0]
    assert dist[(BINS - 1) // 2].item() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------
@pytest.mark.parametrize("value", [0.0, 5.0, 37.0, -37.0, 123.4, -250.0, 400.0, -400.0])
def test_expected_value_round_trips(value):
    """Heads decode with sum(probs * support) then h_inverse, so the projection
    has to be its inverse to within bin resolution."""
    assert decode(project([[value]])).item() == pytest.approx(value, rel=0.02, abs=0.5)


def test_beyond_the_ceiling_clamps_rather_than_wrapping():
    assert decode(project([[50_000.0]])).item() == pytest.approx(REWARD_CEILING, rel=0.05)
    assert decode(project([[-50_000.0]])).item() == pytest.approx(-REWARD_CEILING, rel=0.05)


def test_shape_is_preserved_with_bin_axis_appended():
    dist = project([[0.0, 10.0, 20.0], [30.0, -40.0, 400.0]])
    assert dist.shape == (2, 3, BINS)
    assert torch.allclose(dist.sum(-1), torch.ones(2, 3))


# --------------------------------------------------------------------------
# Resolution -- the finding from benchmarks/bench.py prefix
# --------------------------------------------------------------------------
def test_wide_ceiling_still_resolves_small_values():
    """The whole point of h(). With a linear support of [0, 2000] over 51 bins
    the bins were 40 wide, so realistic prefixes of 0-5 all collapsed into bin 0
    and the value-prefix head could not train. Under h() the same ceiling
    resolves them."""
    typical = [[0.0, 0.5, 1.0, 2.0, 3.0, 5.0]]
    dist = project(typical, support=PREFIX_SUPPORT, bins=PREFIX_BINS)
    bins_used = int((dist.sum(dim=(0, 1)) > 1e-6).sum())
    assert bins_used >= 4, f"only {bins_used} bins used for values spanning 0-5"


def test_small_values_still_round_trip_under_a_wide_ceiling():
    for value in (0.5, 2.0, 5.0, 20.0):
        got = decode(project([[value]], support=PREFIX_SUPPORT, bins=PREFIX_BINS),
                     PREFIX_SUPPORT, PREFIX_BINS).item()
        assert got == pytest.approx(value, abs=1.5), f"{value} decoded as {got}"


def test_a_linear_support_would_lose_them():
    """Shows the failure h() fixes: without the transform the same ceiling
    crushes every small value into one bin."""
    typical = [[0.0, 0.5, 1.0, 2.0, 3.0, 5.0]]
    linear = scalar_to_categorical_batch(torch.tensor(typical), PREFIX_CEILING,
                                         PREFIX_BINS, DEVICE, transform=False)
    assert int((linear.sum(dim=(0, 1)) > 1e-6).sum()) <= 2
