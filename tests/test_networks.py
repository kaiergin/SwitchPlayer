"""Network output contracts -- shapes, and the two conventions that were broken."""
import numpy as np
import pytest
import torch

from muzero import LATENT_CHANNELS, IMAGE_RESHAPE, NetworkManager
from networks import DynamicsNetwork, PredictionNetwork, RepresentationNetwork

ACTIONS, BINS = 5, 51
CPU = torch.device("cpu")


@pytest.fixture
def manager():
    # network_ref is only touched by setup()/update_if_needed(), so None is fine
    # for exercising the local inference paths.
    return NetworkManager(None, action_space=ACTIONS, device=CPU)


@pytest.fixture
def latent():
    rep = RepresentationNetwork(input_channels=12, latent_channels=LATENT_CHANNELS)
    with torch.no_grad():
        return rep(torch.rand(2, 12, *IMAGE_RESHAPE))


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------
def test_representation_downsamples_by_four(latent):
    h, w = IMAGE_RESHAPE
    assert latent.shape == (2, LATENT_CHANNELS, h // 4, w // 4)


def test_dynamics_preserves_latent_shape(latent):
    dyn = DynamicsNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                          support_size=400, num_bins=BINS)
    with torch.no_grad():
        nxt, reward, reward_logits = dyn(latent, torch.zeros(2, dtype=torch.long))
    assert nxt.shape == latent.shape
    assert reward.shape == (2, 1)
    assert reward_logits.shape == (2, BINS)


def test_prediction_head_shapes(latent):
    pred = PredictionNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                             support_size=2000, num_bins=BINS)
    with torch.no_grad():
        policy_logits, value, value_logits = pred(latent)
    assert policy_logits.shape == (2, ACTIONS)
    assert value.shape == (2, 1)
    assert value_logits.shape == (2, BINS)


# --------------------------------------------------------------------------
# The policy head must emit logits, not probabilities
# --------------------------------------------------------------------------
def test_policy_head_returns_raw_logits(latent):
    """Regression: the head used to return softmax(logits) clamped to [1e-6, 1].

    Every caller named the result policy_logits and applied softmax or
    log_softmax to it, so acting used softmax(softmax(x)) -- at most a 2.7x
    spread between best and worst action -- and the training loss took
    log_softmax of a probability vector, which cannot produce a sharp policy.

    Zeroing policy_fc's weights makes the head output its bias exactly, so
    logits and probabilities are trivially distinguishable.
    """
    pred = PredictionNetwork(latent_channels=LATENT_CHANNELS, action_space=ACTIONS,
                             support_size=2000, num_bins=BINS)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    with torch.no_grad():
        pred.policy_fc.weight.zero_()
        pred.policy_fc.bias.copy_(bias)
        policy_logits, _, _ = pred(latent)

    assert policy_logits[0] == pytest.approx(bias.tolist())
    assert policy_logits.sum(dim=1)[0] != pytest.approx(1.0)  # not a distribution
    assert torch.softmax(policy_logits, dim=1).sum(dim=1)[0] == pytest.approx(1.0)


def test_softmaxing_the_head_output_gives_a_usable_distribution(manager):
    """np.random.choice rejects vectors that do not sum to 1; the old clamp
    broke normalisation, which could make acting raise outright."""
    policy, value = manager.initial_inference(np.random.randint(0, 255, (12, *IMAGE_RESHAPE), dtype=np.uint8))
    assert policy.shape == (ACTIONS,)
    assert policy.sum() == pytest.approx(1.0)
    np.random.choice(np.arange(ACTIONS), p=policy)  # must not raise
    assert isinstance(value, float)


# --------------------------------------------------------------------------
# Inference must not retain the autograd graph
# --------------------------------------------------------------------------
def test_batch_inference_returns_detached_latents(manager):
    """Regression, and the cause of the out-of-memory failures.

    Every gameplay node stores one of these latents for the whole episode. When
    they carried grad_fn, the entire forward graph -- all ResNet activations
    for the batch -- stayed resident with them: multiple GB per episode.
    """
    obs = np.random.randint(0, 255, (4, 12, *IMAGE_RESHAPE), dtype=np.uint8)
    latents, policies, values = manager.batch_inference(obs)

    assert latents.grad_fn is None, "latents still carry the forward graph"
    assert not latents.requires_grad
    assert latents.shape[0] == 4


def test_batch_inference_value_is_scalar_per_row(manager):
    """expanded_value has to be a scalar. It used to be a (1,)-shaped tensor for
    gameplay nodes and a float elsewhere, so value_sum silently became a
    tensor mid-backup."""
    obs = np.random.randint(0, 255, (3, 12, *IMAGE_RESHAPE), dtype=np.uint8)
    _, policies, values = manager.batch_inference(obs)

    assert isinstance(values, np.ndarray) and values.shape == (3,)
    assert np.isscalar(values[0]) or values[0].ndim == 0
    assert isinstance(policies, np.ndarray) and policies.shape == (3, ACTIONS)
    assert policies.sum(axis=1) == pytest.approx(np.ones(3))


def test_initial_inference_does_not_leak_graph(manager):
    """no_grad has to be inside the method: Ray actors run in their own process,
    so a `with torch.no_grad()` at the call site does nothing."""
    obs = np.random.randint(0, 255, (12, *IMAGE_RESHAPE), dtype=np.uint8)
    before = torch.is_grad_enabled()
    policy, value = manager.initial_inference(obs)
    assert torch.is_grad_enabled() == before
    assert isinstance(value, float)
