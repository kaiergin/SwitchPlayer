"""Gradient scaling, loss weighting, and real-time pacing.

ROADMAP priorities 4 and 5. Before these, every loss term was summed raw and
every unroll step contributed at full weight, so a K-step unroll multiplied the
gradient reaching the representation by roughly K; and self-play ran at whatever
speed ALE managed, which is ~80x real time.
"""
import asyncio
import time

import numpy as np
import pytest
import torch

import muzero
from muzero import (CONSISTENCY_LOSS_WEIGHT, GAME_FPS, LATENT_GRADIENT_SCALE,
                    POLICY_LOSS_WEIGHT, RateLimiter, REWARD_LOSS_WEIGHT,
                    VALUE_LOSS_WEIGHT, real_time_step_period, scale_gradient)


# --------------------------------------------------------------------------
# scale_gradient
# --------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [0.0, 0.25, 0.5, 1.0])
def test_forward_pass_is_untouched(scale):
    x = torch.tensor([1.5, -2.0, 7.0])
    assert torch.allclose(scale_gradient(x, scale), x)


@pytest.mark.parametrize("scale", [0.0, 0.25, 0.5, 1.0, 2.0])
def test_backward_pass_is_multiplied(scale):
    x = torch.tensor([3.0], requires_grad=True)
    scale_gradient(x, scale).sum().backward()
    assert x.grad.item() == pytest.approx(scale)


def test_scaling_compounds_through_a_chain():
    """Applied once per unroll step, so a 5-step unroll attenuates the gradient
    reaching the representation by 0.5^5 rather than amplifying it by 5."""
    x = torch.tensor([1.0], requires_grad=True)
    h = x
    for _ in range(5):
        h = scale_gradient(h, 0.5)
    h.sum().backward()
    assert x.grad.item() == pytest.approx(0.5 ** 5)


def test_unroll_scaling_keeps_total_gradient_independent_of_depth():
    """1/K per step means the summed loss does not grow with unroll length."""
    grads = []
    for k in (2, 5, 10):
        x = torch.tensor([1.0], requires_grad=True)
        total = sum(scale_gradient(x * 1.0, 1.0 / k) for _ in range(k))
        total.backward()
        grads.append(x.grad.item())
    assert grads == pytest.approx([1.0, 1.0, 1.0])


# --------------------------------------------------------------------------
# Loss weights
# --------------------------------------------------------------------------
def test_value_loss_is_down_weighted():
    """Its target has by far the widest range, so at equal weight it dominates."""
    assert VALUE_LOSS_WEIGHT == 0.25
    assert VALUE_LOSS_WEIGHT < POLICY_LOSS_WEIGHT
    assert VALUE_LOSS_WEIGHT < REWARD_LOSS_WEIGHT


def test_all_weights_are_positive():
    for w in (VALUE_LOSS_WEIGHT, POLICY_LOSS_WEIGHT, REWARD_LOSS_WEIGHT,
              CONSISTENCY_LOSS_WEIGHT, muzero.PREFIX_LOSS_WEIGHT):
        assert w > 0


def test_weights_change_the_training_loss():
    from tests.test_reanalysis import build_trainer, fake_batch
    batch = fake_batch(2, muzero.UNROLL_STEPS, np.ones((2, muzero.UNROLL_STEPS + 1), dtype=bool))

    torch.manual_seed(0)
    base = asyncio.run(build_trainer().train_step(batch))

    saved = muzero.VALUE_LOSS_WEIGHT
    muzero.VALUE_LOSS_WEIGHT = 4.0
    try:
        torch.manual_seed(0)
        heavier = asyncio.run(build_trainer().train_step(batch))
    finally:
        muzero.VALUE_LOSS_WEIGHT = saved
    assert heavier > base, "raising the value weight should raise the total loss"


def test_train_step_still_finite_with_scaling():
    from tests.test_reanalysis import build_trainer, fake_batch
    mask = np.ones((2, muzero.UNROLL_STEPS + 1), dtype=bool)
    loss = asyncio.run(build_trainer().train_step(fake_batch(2, muzero.UNROLL_STEPS, mask)))
    assert np.isfinite(loss)


def test_train_step_reads_use_h_transform_at_call_time():
    """The ablation's no-transform arm overrides muzero.USE_H_TRANSFORM at
    runtime; train_step has to pick that up, not a value captured at import."""
    from tests.test_reanalysis import build_trainer, fake_batch
    saved = muzero.USE_H_TRANSFORM
    muzero.USE_H_TRANSFORM = False
    try:
        mask = np.ones((2, muzero.UNROLL_STEPS + 1), dtype=bool)
        loss = asyncio.run(build_trainer().train_step(fake_batch(2, muzero.UNROLL_STEPS, mask)))
        assert np.isfinite(loss)
    finally:
        muzero.USE_H_TRANSFORM = saved


# --------------------------------------------------------------------------
# SimSiam consistency loss
# --------------------------------------------------------------------------
def test_consistency_loss_is_negative_cosine():
    a = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    b = torch.tensor([[0.0, 1.0], [3.0, 0.0]])   # orthogonal to a, row-wise
    assert muzero.consistency_loss(a, a).item() == pytest.approx(-1.0)
    assert muzero.consistency_loss(a, b).item() == pytest.approx(0.0)


def test_consistency_loss_is_scale_free():
    """The reason for cosine over MSE: shrinking the latents must not reduce
    the loss, or the representation can collapse its way to a good score."""
    a = torch.tensor([[3.0, 1.0], [0.5, 2.0]])
    b = torch.tensor([[1.0, 2.0], [2.0, 0.5]])
    assert muzero.consistency_loss(a * 1e-3, b).item() == pytest.approx(
        muzero.consistency_loss(a, b).item(), abs=1e-5)


def test_consistency_target_is_stop_gradient():
    """The SimSiam stop-grad is load-bearing; the helper must detach the
    target even if a caller forgets to."""
    online = torch.randn(3, 8, requires_grad=True)
    target = torch.randn(3, 8, requires_grad=True)
    muzero.consistency_loss(online, target).backward()
    assert online.grad is not None
    assert target.grad is None


# --------------------------------------------------------------------------
# Reward clipping
# --------------------------------------------------------------------------
def test_clip_reward_uses_the_constant_and_is_symmetric():
    """Regression: the acting path hardcoded min(reward, 400), so changing
    REWARD_CLIP would resize the categorical supports while the data kept the
    old bound. min() also never clipped the negative side, which MsPacman
    hides but signed-reward targets (Tetris 99) will not."""
    assert muzero.clip_reward(10_000.0) == muzero.REWARD_CLIP
    assert muzero.clip_reward(-10_000.0) == -muzero.REWARD_CLIP
    assert muzero.clip_reward(7) == 7.0
    assert muzero.clip_reward(10_000.0, ceiling=5.0) == 5.0


# --------------------------------------------------------------------------
# Real-time pacing
# --------------------------------------------------------------------------
class FakeUnwrapped:
    def __init__(self, frameskip):
        self._frameskip = frameskip


class FakeEnv:
    def __init__(self, frameskip):
        self.unwrapped = FakeUnwrapped(frameskip)


def test_step_period_accounts_for_frameskip():
    """Regression on a mistake worth not repeating: MsPacman-v4 samples its
    frameskip from (2, 5), so one decision covers 3.5 raw frames and real time
    is ~17 decisions/s. Pacing at GAME_FPS would run 3.5x too fast."""
    period = real_time_step_period(FakeEnv((2, 5)))
    assert 1.0 / period == pytest.approx(60.0 / 3.5, rel=1e-3)
    assert 1.0 / period < GAME_FPS


def test_deterministic_frameskip_is_handled():
    assert 1.0 / real_time_step_period(FakeEnv(4)) == pytest.approx(15.0)


def test_no_frameskip_means_one_decision_per_frame():
    """What the Switch target looks like: real video, no skipping."""
    assert 1.0 / real_time_step_period(FakeEnv(None)) == pytest.approx(GAME_FPS)


def test_limiter_holds_the_target_rate():
    limiter = RateLimiter(0.02)   # 50/s

    async def run():
        start = time.perf_counter()
        for _ in range(10):
            await limiter.wait()
        return time.perf_counter() - start

    elapsed = asyncio.run(run())
    assert 0.14 < elapsed < 0.34, f"10 steps at 50/s took {elapsed:.3f}s"


def test_zero_period_does_not_wait():
    limiter = RateLimiter(0.0)

    async def run():
        start = time.perf_counter()
        for _ in range(200):
            await limiter.wait()
        return time.perf_counter() - start

    assert asyncio.run(run()) < 0.1


def test_falling_behind_does_not_trigger_a_catch_up_burst():
    """If a step overruns, the limiter must not then sprint through the next few
    to make up time -- that is exactly the faster-than-real-time burst it exists
    to prevent."""
    limiter = RateLimiter(0.01)

    async def run():
        await limiter.wait()
        time.sleep(0.08)          # a long, blocking step
        await limiter.wait()
        start = time.perf_counter()
        await limiter.wait()      # must still wait a full period
        return time.perf_counter() - start

    assert asyncio.run(run()) > 0.004
    assert limiter.behind >= 1
